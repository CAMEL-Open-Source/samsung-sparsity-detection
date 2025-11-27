#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/fanotify.h>
#include <sys/types.h>
#include <time.h>

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

__global__ void matrixMul_naive(float *C, float *A, float *B, int wA, int wB,
                                int hA) {
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < hA && j < wB) {
    float sum = 0;
    for (unsigned int k = 0; k < wA; k++) {
      sum += A[(i * wA) + k] * B[(k * wB) + j];
    }
    C[(i * wB) + j] = sum;
  }
}

__global__ void matrixMul_shmem(float *C, float *A, float *B, int wA, int wB,
                                int hA) {
  __shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int i = blockIdx.y * BLOCK_SIZE + ty;
  int j = blockIdx.x * BLOCK_SIZE + tx;

  float sum = 0;
  for (unsigned int k = 0; k < wA; k += BLOCK_SIZE) {
    tile_a[ty][tx] = A[i * wA + tx + k];
    tile_b[ty][tx] = B[(ty + k) * wB + j];
    __syncthreads();
    for (unsigned int m = 0; m < BLOCK_SIZE; m++) {
      sum += tile_a[ty][m] * tile_b[m][tx];
    }
    __syncthreads();
  }
  C[i * wB + j] = sum;
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

  int wA = a_data_size;  // Number of columns in A
  int wB = b_data_size;  // Number of columns in B
  int hA = a_data_size;  // Number of rows in A
  int hB = b_data_size;  // Number of rows in B

  // Allocate device memory and copy data
  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, hA * wA * sizeof(float));
  cudaMalloc((void **)&d_B, hB * wB * sizeof(float));
  cudaMalloc((void **)&d_C, hA * wB * sizeof(float));

  cudaEventRecord(host_to_device_time_start);
  cudaMemcpy(d_A, a_data.data(), a_data_size * a_data_size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, b_data.data(), b_data_size * b_data_size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaEventRecord(host_to_device_time_end);
  cudaEventSynchronize(host_to_device_time_end);
  cudaEventElapsedTime(&cfg().host_to_device_time, host_to_device_time_start,
                       host_to_device_time_end);

  // Define grid and block dimensions
  cudaEventRecord(matmul_time_start);
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((wB + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (hA + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // Launch kernel
  matrixMul_naive<<<grid, threads>>>(d_C, d_A, d_B, wA, wB, hA);
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
  std::vector<float> c_data(wA * wB);
  cudaMemcpy(c_data.data(), d_C, wA * wB * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaEventRecord(device_to_host_time_end);
  cudaEventSynchronize(device_to_host_time_end);
  cudaEventElapsedTime(&cfg().device_to_host_time, device_to_host_time_start,
                       device_to_host_time_end);

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Print statistics
  printf("%.9f %.9f %.9f %.9f\n", cfg().host_to_device_time, cfg().matmul_time,
         cfg().device_to_host_time,
         cfg().host_to_device_time + cfg().matmul_time +
             cfg().device_to_host_time);

  return 0;
}
