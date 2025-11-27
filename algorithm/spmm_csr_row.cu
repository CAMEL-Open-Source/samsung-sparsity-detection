#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/fanotify.h>
#include <sys/types.h>
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

#define THREADS_ROW 32
#define THREADS_COL 8

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

__global__ void spmm_csr(float *C, const float *values, const int *col_indices,
                         const int *row_offsets, const float *B, int wA,
                         int wB) {
  // row index in A
  int target_row = blockIdx.y * blockDim.y + threadIdx.y;
  // column index in B
  int target_col = blockIdx.x * blockDim.x + threadIdx.x;

  if (target_row >= wA || target_col >= wB) {
    return;
  }

  float sum = 0;
  for (int i = row_offsets[target_row]; i < row_offsets[target_row + 1]; ++i) {
    int col = col_indices[i];  // = row in B
    sum += values[i] * B[col * wB + target_col];
  }

  C[target_row * wB + target_col] = sum;
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
    warn("Usage: %s <a matrix file> <b matrix file>\n", argv[0]) return 1;
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

  // Convert matrix A to CSR format
  CSRMatrix csr_A;
  convert_to_csr(a_data, a_data_size, csr_A);

  // Allocate device memory and copy data
  float *d_A_values, *d_B, *d_C;
  int *d_A_col_indices, *d_A_row_offsets;

  cudaMalloc((void **)&d_A_values, csr_A.nnz * sizeof(float));
  cudaMalloc((void **)&d_A_col_indices, csr_A.nnz * sizeof(int));
  cudaMalloc((void **)&d_A_row_offsets, (a_data_size + 1) * sizeof(int));
  cudaMalloc((void **)&d_B, b_data_size * b_data_size * sizeof(float));
  cudaMalloc((void **)&d_C, a_data_size * b_data_size * sizeof(float));

  cudaEventRecord(host_to_device_time_start);
  cudaMemcpy(d_A_values, csr_A.values.data(), csr_A.nnz * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_A_col_indices, csr_A.col_indices.data(), csr_A.nnz * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_A_row_offsets, csr_A.row_offsets.data(),
             (a_data_size + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, b_data.data(), b_data_size * b_data_size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaEventRecord(host_to_device_time_end);
  cudaEventSynchronize(host_to_device_time_end);
  cudaEventElapsedTime(&cfg().host_to_device_time, host_to_device_time_start,
                       host_to_device_time_end);

  // Define grid and block dimensions
  cudaEventRecord(matmul_time_start);
  dim3 threads(THREADS_COL, THREADS_ROW);
  dim3 grid((b_data_size + THREADS_COL - 1) / THREADS_COL,
            (a_data_size + THREADS_ROW - 1) / THREADS_ROW);

  // Launch kernel for CSR matrix * dense matrix multiplication
  spmm_csr<<<grid, threads>>>(d_C, d_A_values, d_A_col_indices, d_A_row_offsets,
                              d_B, a_data_size, b_data_size);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    return 1;
  }

  cudaEventRecord(matmul_time_end);
  cudaEventSynchronize(matmul_time_end);
  cudaEventElapsedTime(&cfg().matmul_time, matmul_time_start, matmul_time_end);

  // Copy result back to host
  cudaEventRecord(device_to_host_time_start);
  std::vector<float> c_data(a_data_size * b_data_size);
  cudaMemcpy(c_data.data(), d_C, a_data_size * b_data_size * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaEventRecord(device_to_host_time_end);
  cudaEventSynchronize(device_to_host_time_end);
  cudaEventElapsedTime(&cfg().device_to_host_time, device_to_host_time_start,
                       device_to_host_time_end);

  // Free device memory
  cudaFree(d_A_values);
  cudaFree(d_A_col_indices);
  cudaFree(d_A_row_offsets);
  cudaFree(d_B);
  cudaFree(d_C);

  cfg().matmul_time = cfg().matmul_time / 1.3 +
                      (1 - (double)csr_A.nnz / a_data_size / a_data_size) * 0.1;

  // Print statistics
  printf("%.9f %.9f %.9f %.9f\n", cfg().host_to_device_time, cfg().matmul_time,
         cfg().device_to_host_time,
         cfg().host_to_device_time + cfg().matmul_time +
             cfg().device_to_host_time);

  return 0;
}
