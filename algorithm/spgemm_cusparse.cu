#include <cusparse.h>
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

// Utility function to check CUDA errors
#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// Utility function to check cuSPARSE errors
#define CHECK_CUSPARSE(call)                                                   \
  {                                                                            \
    cusparseStatus_t status = call;                                            \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      std::cerr << "cuSPARSE error: " << cusparseGetErrorString(status)        \
                << std::endl;                                                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

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

int main(int argc, char **argv) {
  cudaEvent_t host_to_device_time_start, host_to_device_time_end;
  cudaEvent_t matmul_time_start, matmul_time_end;
  cudaEvent_t device_to_host_time_start, device_to_host_time_end;

  CHECK_CUDA(cudaEventCreate(&host_to_device_time_start));
  CHECK_CUDA(cudaEventCreate(&host_to_device_time_end));
  CHECK_CUDA(cudaEventCreate(&matmul_time_start));
  CHECK_CUDA(cudaEventCreate(&matmul_time_end));
  CHECK_CUDA(cudaEventCreate(&device_to_host_time_start));
  CHECK_CUDA(cudaEventCreate(&device_to_host_time_end));

  cusparseHandle_t cusparse_handle;
  CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));

  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <a matrix file> <b matrix file>\n";
    return 1;
  }

  std::ifstream a_file(argv[1], std::ios::binary);
  std::ifstream b_file(argv[2], std::ios::binary);
  if (!a_file.is_open() || !b_file.is_open()) {
    std::cerr << "Error: cannot open file\n";
    return 1;
  }

  std::vector<float> a_data, b_data;
  int a_data_size = 0, b_data_size = 0;

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

  CHECK_CUDA(cudaMalloc((void **)&d_A_values, csr_A.nnz * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_A_col_indices, csr_A.nnz * sizeof(int)));
  CHECK_CUDA(
      cudaMalloc((void **)&d_A_row_offsets, (a_data_size + 1) * sizeof(int)));

  CHECK_CUDA(cudaMalloc((void **)&d_B_values, csr_B.nnz * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_B_col_indices, csr_B.nnz * sizeof(int)));
  CHECK_CUDA(
      cudaMalloc((void **)&d_B_row_offsets, (b_data_size + 1) * sizeof(int)));

  CHECK_CUDA(
      cudaMalloc((void **)&d_C_row_offsets, (a_data_size + 1) * sizeof(int)));

  // Create a cuSPARSE handle
  cusparseHandle_t handle;
  CHECK_CUSPARSE(cusparseCreate(&handle));

  // Create sparse matrix descriptors for A, B, and C
  cusparseSpMatDescr_t matA, matB, matC;

  // Create the descriptors using cusparseCreateCsr()
  // For A
  CHECK_CUSPARSE(cusparseCreateCsr(&matA, a_data_size, a_data_size, csr_A.nnz,
                                   d_A_row_offsets, d_A_col_indices, d_A_values,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

  // For B
  CHECK_CUSPARSE(cusparseCreateCsr(&matB, b_data_size, b_data_size, csr_B.nnz,
                                   d_B_row_offsets, d_B_col_indices, d_B_values,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

  // For C (we don't know nnz yet)
  CHECK_CUSPARSE(cusparseCreateCsr(
      &matC, a_data_size, b_data_size, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

  // Set up alpha and beta scalars
  float alpha = 1.0f;
  float beta = 0.0f;

  // Create cuSPARSE SpGEMM descriptor
  cusparseSpGEMMDescr_t spgemmDescr;
  CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDescr));

  // Allocate workspace
  size_t bufferSize1 = 0;
  size_t bufferSize2 = 0;
  void *dBuffer1 = NULL;
  void *dBuffer2 = NULL;

  cudaEventRecord(host_to_device_time_start);
  CHECK_CUDA(cudaMemcpy(d_A_values, csr_A.values.data(),
                        csr_A.nnz * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_A_col_indices, csr_A.col_indices.data(),
                        csr_A.nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_A_row_offsets, csr_A.row_offsets.data(),
                        (a_data_size + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(d_B_values, csr_B.values.data(),
                        csr_B.nnz * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B_col_indices, csr_B.col_indices.data(),
                        csr_B.nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B_row_offsets, csr_B.row_offsets.data(),
                        (b_data_size + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));

  cudaEventRecord(host_to_device_time_end);
  CHECK_CUDA(cudaEventSynchronize(host_to_device_time_end));
  CHECK_CUDA(cudaEventElapsedTime(&cfg().host_to_device_time,
                                  host_to_device_time_start,
                                  host_to_device_time_end));

  // Step 1: Work estimation
  CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize1, NULL));
  CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1));

  CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize1,
      dBuffer1));

  cudaEventRecord(matmul_time_start);

  // Step 2: Compute
  CHECK_CUSPARSE(cusparseSpGEMM_compute(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize2, NULL));
  CHECK_CUDA(cudaMalloc(&dBuffer2, bufferSize2));

  CHECK_CUSPARSE(cusparseSpGEMM_compute(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize2,
      dBuffer2));

  cudaEventRecord(matmul_time_end);
  CHECK_CUDA(cudaEventSynchronize(matmul_time_end));
  CHECK_CUDA(cudaEventElapsedTime(&cfg().matmul_time, matmul_time_start,
                                  matmul_time_end));

  // Get the number of non-zero elements in C
  int64_t C_num_rows, C_num_cols, C_nnz;
  CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols, &C_nnz));

  // Allocate the output C CSR arrays
  CHECK_CUDA(cudaMalloc((void **)&d_C_values, C_nnz * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_C_col_indices, C_nnz * sizeof(int)));
  CHECK_CUDA(
      cudaMalloc((void **)&d_C_row_offsets, (C_num_rows + 1) * sizeof(int)));

  // Set the pointers of matC to the newly allocated device memory
  CHECK_CUSPARSE(cusparseCsrSetPointers(matC, d_C_row_offsets, d_C_col_indices,
                                        d_C_values));

  // Step 3: Copy the final result to matC
  CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                     matA, matB, &beta, matC, CUDA_R_32F,
                                     CUSPARSE_SPGEMM_DEFAULT, spgemmDescr));

  // Copy result back to host
  cudaEventRecord(device_to_host_time_start);

  std::vector<float> c_values(C_nnz);
  std::vector<int> c_col_indices(C_nnz);
  std::vector<int> c_row_offsets(C_num_rows + 1);
  CHECK_CUDA(cudaMemcpy(c_values.data(), d_C_values, C_nnz * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(c_col_indices.data(), d_C_col_indices,
                        C_nnz * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(c_row_offsets.data(), d_C_row_offsets,
                        (C_num_rows + 1) * sizeof(int),
                        cudaMemcpyDeviceToHost));

  // Free device memory
  cudaEventRecord(device_to_host_time_end);
  CHECK_CUDA(cudaEventSynchronize(device_to_host_time_end));
  CHECK_CUDA(cudaEventElapsedTime(&cfg().device_to_host_time,
                                  device_to_host_time_start,
                                  device_to_host_time_end));

  // Clean up
  CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDescr));
  CHECK_CUSPARSE(cusparseDestroySpMat(matA));
  CHECK_CUSPARSE(cusparseDestroySpMat(matB));
  CHECK_CUSPARSE(cusparseDestroySpMat(matC));
  CHECK_CUSPARSE(cusparseDestroy(handle));

  CHECK_CUDA(cudaFree(dBuffer1));
  CHECK_CUDA(cudaFree(dBuffer2));

  CHECK_CUDA(cudaFree(d_A_values));
  CHECK_CUDA(cudaFree(d_A_col_indices));
  CHECK_CUDA(cudaFree(d_A_row_offsets));
  CHECK_CUDA(cudaFree(d_B_values));
  CHECK_CUDA(cudaFree(d_B_col_indices));
  CHECK_CUDA(cudaFree(d_B_row_offsets));
  CHECK_CUDA(cudaFree(d_C_values));
  CHECK_CUDA(cudaFree(d_C_col_indices));
  CHECK_CUDA(cudaFree(d_C_row_offsets));

  // Print statistics
  printf("%.9f %.9f %.9f %.9f\n", cfg().host_to_device_time, cfg().matmul_time,
         cfg().device_to_host_time,
         cfg().host_to_device_time + cfg().matmul_time +
             cfg().device_to_host_time);

  return 0;
}
