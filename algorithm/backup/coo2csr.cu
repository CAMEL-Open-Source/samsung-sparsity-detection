#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <time.h>

#include <algorithm>
#include <random>

#define BLOCK_SIZE 32
#define USE_HASH 1

#define N_NODES 5000000
#define N_DEGREE 10

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

// Kernel to compute row_indices
__global__ void compute_row_indices(const int32_t *indices_row,
                                    int32_t *row_ptr, int64_t nnz,
                                    int64_t n_nodes) {
  int64_t edge_index = blockDim.x * blockIdx.x + threadIdx.x;
  if (edge_index >= nnz)
    return;

  // if (edge_index == 0) {
  //   int64_t row = indices_row[edge_index];
  //   row_ptr[row] = 0;
  // }

  if (edge_index > 0 &&
      indices_row[edge_index] != indices_row[edge_index - 1]) {
    int64_t row = indices_row[edge_index];
    row_ptr[row] = edge_index;
  }

  // if (edge_index == nnz - 1) {
  //   int64_t row = indices_row[edge_index] + 1;
  //   if (row <= n_nodes)
  //     row_ptr[row] = nnz;
  // }
}

__global__ void compute_row_indices_hash(const int32_t *indices_row,
                                         int32_t *row_ptr, int64_t nnz,
                                         int64_t n_nodes) {
  int64_t edge_index = blockDim.x * blockIdx.x + threadIdx.x;
  if (edge_index >= nnz)
    return;

  atomicAdd(&row_ptr[indices_row[edge_index]], 1);
}

// Function to generate sorted random array
std::vector<int> generateSortedArray(int32_t size, int32_t maxValue) {
  std::mt19937 generator(0);  // Mersenne Twister generator

  // Uniform distribution from 0 to maxValue
  std::uniform_int_distribution<int32_t> distribution(0, maxValue);

  std::vector<int32_t> arr(size);

  // Fill the array with random values
  for (int i = 0; i < size; ++i) {
    arr[i] = distribution(generator);
  }

  // Sort the array in ascending order
  std::sort(arr.begin(), arr.end());

  return arr;
}

int main() {
  int64_t nnz = N_NODES * N_DEGREE;
  int64_t n_nodes = N_NODES;

  // Generate random indices
  std::vector<int32_t> indices_row = generateSortedArray(nnz, n_nodes - 1);

  // Allocate cuda memory
  int32_t *d_indices_row, *d_row_ptr;
  cudaMalloc(&d_indices_row, nnz * sizeof(int32_t));
  cudaMalloc(&d_row_ptr, (n_nodes + 1) * sizeof(int32_t));

  // Copy data to device
  cudaMemcpy(d_indices_row, indices_row.data(), nnz * sizeof(int32_t),
             cudaMemcpyHostToDevice);

  // Launch kernel
  int n_blocks = (nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;
  uint64_t start = GetNowNanos();
#if USE_HASH
  compute_row_indices_hash<<<n_blocks, BLOCK_SIZE>>>(d_indices_row, d_row_ptr,
                                                     nnz, n_nodes);
  // prefix sum
  thrust::inclusive_scan(thrust::device_ptr<int32_t>(d_row_ptr),
                         thrust::device_ptr<int32_t>(d_row_ptr + n_nodes + 1),
                         thrust::device_ptr<int32_t>(d_row_ptr));
#else
  compute_row_indices<<<n_blocks, BLOCK_SIZE>>>(d_indices_row, d_row_ptr, nnz,
                                                n_nodes);
#endif
  cudaDeviceSynchronize();
  uint64_t end = GetNowNanos();
  std::cout << "Time: " << (end - start) / 1000000.0 << " ms" << std::endl;

  // Copy data back to host
  std::vector<int32_t> row_ptr(n_nodes + 1);
  cudaMemcpy(row_ptr.data(), d_row_ptr, (n_nodes + 1) * sizeof(int32_t),
             cudaMemcpyDeviceToHost);

  return 0;
}