#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <time.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#define UNUSED(x) (void)(x)
#define ENABLE_WARN true
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define MIN(x, y) ((x) > (y) ? (y) : (x))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define DIVCEIL(x, y) (((x) - 1) / (y) + 1)
static constexpr unsigned SECTOR_SIZE = 512;
static constexpr unsigned PAGE_SIZE = 4096;
static constexpr unsigned BUFFER_SIZE = 4096;
static constexpr double SECOND = 1.e9;
static constexpr unsigned long GIGABYTE = 1000000000UL;
static constexpr unsigned long BWITER = 50000000UL;

#define MINORBITS 20
#define MINORMASK ((1U << MINORBITS) - 1)

#define MAJOR(dev) ((unsigned int)((dev) >> MINORBITS))
#define MINOR(dev) ((unsigned int)((dev) & MINORMASK))
#define MKDEV(ma, mi) (((ma) << MINORBITS) | (mi))

#define print_log(format, ...)                                                 \
  {                                                                            \
    fprintf(cfg().output, format, ##__VA_ARGS__);                              \
    fflush(cfg().output);                                                      \
  }

#if ENABLE_WARN
#define warn(format, ...)                                                      \
  {                                                                            \
    print_log(format "\n", ##__VA_ARGS__);                                     \
    fflush(cfg().output);                                                      \
    raise(SIGTRAP);                                                            \
  }
#define warn_if(cond, format, ...)                                             \
  {                                                                            \
    if (cond) {                                                                \
      warn(format, ##__VA_ARGS__);                                             \
    }                                                                          \
  }
#else
#define warn(format, ...)                                                      \
  {}
#define warn_if(cond, format, ...)                                             \
  {}
#endif

#define log(format, ...)                                                       \
  { print_log("[%.3f] " format "\n", GetElapsedSeconds(), ##__VA_ARGS__); }

struct Config {
  uint64_t start_time;
  FILE *output = stdout;
  uint64_t start_processing = 0;
  float host_to_device_time = 0;
  float matmul_time = 0;
  float device_to_host_time = 0;

  ~Config() { fclose(output); }
};

static Config internal_config;

Config &cfg() {
  return internal_config;
}

inline uint64_t ConvertTimespec(struct timespec &ts) {
  return (uint64_t)ts.tv_sec * 1000000000UL + ts.tv_nsec;
}

inline uint64_t ConvertTimeval(struct timeval &tv) {
  return (uint64_t)tv.tv_sec * 1000000000UL + tv.tv_usec * 1000UL;
}

inline uint64_t GetNowNanos() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ConvertTimespec(ts);
}

static inline float GetElapsedSeconds() {
  return (GetNowNanos() - cfg().start_time) / 1000000000.f;
}

#define BLOCK_SIZE 32
#define SH_ROW 1024
#define WARP 32
#define HASH_SCAL 107

struct CSRMatrix {
  int num_rows;
  int num_cols;
  int nnz;  // Number of non-zero elements
  std::vector<float> values;
  std::vector<int> col_indices;
  std::vector<int> row_offsets;
};

void convert_to_csr(const std::vector<float> &A, int a_size, CSRMatrix &csr) {
  csr.num_rows = a_size;
  csr.num_cols = a_size;
  csr.row_offsets.resize(a_size + 1, 0);

  // Calculate number of non-zero elements and row offsets
  for (int i = 0; i < a_size; ++i) {
    for (int j = 0; j < a_size; ++j) {
      if (A[i * a_size + j] != 0) {
        csr.values.push_back(A[i * a_size + j]);
        csr.col_indices.push_back(j);
        csr.row_offsets[i + 1]++;
      }
    }

    csr.row_offsets[i + 1] += csr.row_offsets[i];
  }

  csr.nnz = csr.values.size();
}

__global__ void calculate_nnz(const float *A_values, const int *A_col_indices,
                              const int *A_row_offsets, const float *B_values,
                              const int *B_col_indices,
                              const int *B_row_offsets, int wA, int wB,
                              int *C_nnz) {
  int target_row = blockIdx.x;
  int hash, old;
  __shared__ int shared_check[SH_ROW];

  for (int i = threadIdx.x; i < SH_ROW; i += blockDim.x) {
    shared_check[i] = -1;
  }

  if (target_row >= wA) {
    return;
  }

  __syncthreads();

  for (int i = A_row_offsets[target_row] + threadIdx.x;
       i < A_row_offsets[target_row + 1]; i += blockDim.x) {
    int col = A_col_indices[i];
    for (int j = B_row_offsets[col]; j < B_row_offsets[col + 1]; ++j) {
      int target_col = B_col_indices[j];
      hash = (target_col * HASH_SCAL) & (SH_ROW - 1);
      while (1) {
        if (shared_check[hash] == target_col) {
          break;
        }
        else if (shared_check[hash] == -1) {
          old = atomicCAS(&shared_check[hash], -1, target_col);
          if (old == -1) {
            break;
          }
        }
        else {
          hash = (hash + 1) & (SH_ROW - 1);
        }
      }
    }
  }

  __syncthreads();

  int count = 0;
  for (int i = threadIdx.x; i < SH_ROW; i += blockDim.x) {
    if (shared_check[i] != -1) {
      count += 1;
    }
  }
  atomicAdd(C_nnz + target_row, count);
}

#include <thrust/device_vector.h>
#include <thrust/sort.h>

struct PsumEntry {
  int col;
  float value;
};

__device__ void bitonicSort(PsumEntry *shared_psum, int length) {
  for (int k = 2; k <= length; k *= 2) {
    for (int j = k / 2; j > 0; j /= 2) {
      int idx = threadIdx.x;
      while (idx < length) {
        int ixj = idx ^ j;
        if (ixj > idx) {
          if ((idx & k) == 0) {
            if (shared_psum[idx].col > shared_psum[ixj].col) {
              // Swap entries
              PsumEntry temp = shared_psum[idx];
              shared_psum[idx] = shared_psum[ixj];
              shared_psum[ixj] = temp;
            }
          }
          else {
            if (shared_psum[idx].col < shared_psum[ixj].col) {
              // Swap entries
              PsumEntry temp = shared_psum[idx];
              shared_psum[idx] = shared_psum[ixj];
              shared_psum[ixj] = temp;
            }
          }
        }
        idx += blockDim.x;
      }
      __syncthreads();
    }
  }
}

__global__ void calculate_value_col_bin(
    const float *A_values, const int *A_col_indices, const int *A_row_offsets,
    const float *B_values, const int *B_col_indices, const int *B_row_offsets,
    float *C_values, int *C_col_indices, int *C_row_offsets, int wA, int wB) {
  int target_row = blockIdx.x;

  __shared__ PsumEntry shared_psum[SH_ROW];
  __shared__ int shared_idx;

  if (threadIdx.x == 0) {
    shared_idx = 0;
  }

  for (int i = threadIdx.x; i < SH_ROW; i += blockDim.x) {
    shared_psum[i].col = -1;
    shared_psum[i].value = 0.0f;
  }

  if (target_row >= wA) {
    return;
  }

  __syncthreads();

  for (int i = A_row_offsets[target_row] + threadIdx.x;
       i < A_row_offsets[target_row + 1]; i += blockDim.x) {
    int col = A_col_indices[i];
    for (int j = B_row_offsets[col]; j < B_row_offsets[col + 1]; ++j) {
      int target_col = B_col_indices[j];
      float partial_sum = A_values[i] * B_values[j];

      // Insert into the list
      int idx = atomicAdd(&shared_idx, 1);
      shared_psum[idx].col = target_col;
      shared_psum[idx].value = partial_sum;
    }
  }

  __syncthreads();

  // Call the sorting function
  bitonicSort(shared_psum, shared_idx);

  __syncthreads();

  // Reduce and write results to global memory
  int offset = C_row_offsets[target_row];
  if (threadIdx.x == 0) {
    for (int i = 0; i < shared_idx; ++i) {
      if (i == 0 || shared_psum[i].col != shared_psum[i - 1].col) {
        C_col_indices[offset] = shared_psum[i].col;
        C_values[offset] = shared_psum[i].value;
        ++offset;
      }
      else {
        C_values[offset - 1] += shared_psum[i].value;
      }
    }
  }
}

int main(int argc, char **argv) {
  cudaEvent_t host_to_device_time_start, host_to_device_time_end;
  cudaEvent_t matmul_time_start, matmul_time_end;
  cudaEvent_t device_to_host_time_start, device_to_host_time_end;
  cudaEventCreate(&host_to_device_time_start);
  cudaEventCreate(&host_to_device_time_end);
  cudaEventCreate(&matmul_time_start);
  cudaEventCreate(&matmul_time_end);
  cudaEventCreate(&device_to_host_time_start);
  cudaEventCreate(&device_to_host_time_end);

  cfg().start_time = GetNowNanos();
  if (argc != 3) {
    warn("Usage: %s <a matrix file> <b matrix file>\n", argv[0]);
    return 1;
  }

  std::ifstream a_file(argv[1], std::ios::binary);
  std::ifstream b_file(argv[2], std::ios::binary);
  if (!a_file.is_open() || !b_file.is_open()) {
    warn("Error: cannot open file\n");
    return 1;
  }

  std::vector<float> a_data;
  std::vector<float> b_data;

  int a_data_size = 0;
  int b_data_size = 0;

  a_file.read(reinterpret_cast<char *>(&a_data_size), sizeof(int));
  b_file.read(reinterpret_cast<char *>(&b_data_size), sizeof(int));

  a_data.resize(a_data_size * a_data_size);
  b_data.resize(b_data_size * b_data_size);

  a_file.read(reinterpret_cast<char *>(a_data.data()),
              a_data_size * a_data_size * sizeof(float));
  b_file.read(reinterpret_cast<char *>(b_data.data()),
              b_data_size * b_data_size * sizeof(float));

  a_file.close();
  b_file.close();

  // Convert matrices A and B to CSR format
  CSRMatrix csr_A, csr_B;
  convert_to_csr(a_data, a_data_size, csr_A);
  convert_to_csr(b_data, b_data_size, csr_B);

  // Allocate device memory and copy data
  float *d_A_values, *d_B_values, *d_C_values;
  int *d_A_col_indices, *d_A_row_offsets;
  int *d_B_col_indices, *d_B_row_offsets;
  int *d_C_col_indices, *d_C_row_offsets;

  cudaMalloc((void **)&d_A_values, csr_A.nnz * sizeof(float));
  cudaMalloc((void **)&d_A_col_indices, csr_A.nnz * sizeof(int));
  cudaMalloc((void **)&d_A_row_offsets, (a_data_size + 1) * sizeof(int));

  cudaMalloc((void **)&d_B_values, csr_B.nnz * sizeof(float));
  cudaMalloc((void **)&d_B_col_indices, csr_B.nnz * sizeof(int));
  cudaMalloc((void **)&d_B_row_offsets, (b_data_size + 1) * sizeof(int));

  cudaMalloc((void **)&d_C_row_offsets, (a_data_size + 1) * sizeof(int));

  cudaEventRecord(host_to_device_time_start);
  cudaMemcpy(d_A_values, csr_A.values.data(), csr_A.nnz * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_A_col_indices, csr_A.col_indices.data(), csr_A.nnz * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_A_row_offsets, csr_A.row_offsets.data(),
             (a_data_size + 1) * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(d_B_values, csr_B.values.data(), csr_B.nnz * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_B_col_indices, csr_B.col_indices.data(), csr_B.nnz * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_B_row_offsets, csr_B.row_offsets.data(),
             (b_data_size + 1) * sizeof(int), cudaMemcpyHostToDevice);

  cudaEventRecord(host_to_device_time_end);
  cudaEventSynchronize(host_to_device_time_end);
  cudaEventElapsedTime(&cfg().host_to_device_time, host_to_device_time_start,
                       host_to_device_time_end);

  // Launch kernel for hash-based SpGEMM
  calculate_nnz<<<a_data_size, BLOCK_SIZE>>>(
      d_A_values, d_A_col_indices, d_A_row_offsets, d_B_values, d_B_col_indices,
      d_B_row_offsets, a_data_size, b_data_size, d_C_row_offsets);

  // do prefix sum
  thrust::device_ptr<int> dev_ptr(d_C_row_offsets);
  thrust::exclusive_scan(dev_ptr, dev_ptr + a_data_size + 1, dev_ptr);

  // Allocate memory for C
  std::vector<int> c_row_offsets(a_data_size + 1);
  cudaMemcpy(c_row_offsets.data(), d_C_row_offsets,
             (a_data_size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
  int c_nnz = c_row_offsets[a_data_size];

  cudaMalloc((void **)&d_C_values, c_nnz * sizeof(float));
  cudaMalloc((void **)&d_C_col_indices, c_nnz * sizeof(int));

  cudaEventRecord(matmul_time_start);

  // Launch kernel for SpGEMM
  calculate_value_col_bin<<<a_data_size, BLOCK_SIZE>>>(
      d_A_values, d_A_col_indices, d_A_row_offsets, d_B_values, d_B_col_indices,
      d_B_row_offsets, d_C_values, d_C_col_indices, d_C_row_offsets,
      a_data_size, b_data_size);

  cudaEventRecord(matmul_time_end);
  cudaEventSynchronize(matmul_time_end);
  cudaEventElapsedTime(&cfg().matmul_time, matmul_time_start, matmul_time_end);

  // Copy result back to host
  cudaEventRecord(device_to_host_time_start);
  std::vector<float> c_values(c_nnz);
  std::vector<int> c_col_indices(c_nnz);
  cudaMemcpy(c_values.data(), d_C_values, c_nnz * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(c_col_indices.data(), d_C_col_indices, c_nnz * sizeof(int),
             cudaMemcpyDeviceToHost);

  // Free device memory
  cudaEventRecord(device_to_host_time_end);
  cudaEventSynchronize(device_to_host_time_end);
  cudaEventElapsedTime(&cfg().device_to_host_time, device_to_host_time_start,
                       device_to_host_time_end);
  cudaFree(d_A_values);
  cudaFree(d_A_col_indices);
  cudaFree(d_A_row_offsets);
  cudaFree(d_B_values);
  cudaFree(d_B_col_indices);
  cudaFree(d_B_row_offsets);
  cudaFree(d_C_values);
  cudaFree(d_C_col_indices);
  cudaFree(d_C_row_offsets);

  // Print statistics
  printf("%.9f %.9f %.9f %.9f\n", cfg().host_to_device_time, cfg().matmul_time,
         cfg().device_to_host_time,
         cfg().host_to_device_time + cfg().matmul_time +
             cfg().device_to_host_time);

  return 0;
}
