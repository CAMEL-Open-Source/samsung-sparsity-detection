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
  float process_time = 0;

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

__global__ void bincount_kernel(uint32_t *list, uint32_t num_edges,
                                uint32_t *bincount) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_edges) {
    return;
  }
  atomicAdd(&bincount[list[tid]], 1);
}

int main(int argc, char **argv) {
  cfg().start_time = GetNowNanos();

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <input>\n", argv[0]);
    return 1;
  }

  std::ifstream file(argv[1], std::ios::binary);

  uint32_t num_nodes = 0;
  uint32_t num_edges = 0;

  file.read(reinterpret_cast<char *>(&num_nodes), sizeof(num_nodes));
  file.read(reinterpret_cast<char *>(&num_edges), sizeof(num_edges));

  std::vector<uint32_t> list(num_edges);
  file.read(reinterpret_cast<char *>(list.data()),
            num_edges * sizeof(uint32_t));

  uint32_t *d_bincount;
  uint32_t *d_list;
  cudaMalloc(&d_bincount, num_nodes * sizeof(uint32_t));
  cudaMalloc(&d_list, num_edges * sizeof(uint32_t));
  cudaMemcpy(d_list, list.data(), num_edges * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  for (uint32_t i = 0; i < 10; i++) {
    bincount_kernel<<<DIVCEIL(num_edges, 256), 256>>>(d_list, num_edges,
                                                      d_bincount);
  }
  cudaMemset(d_bincount, 0, num_nodes * sizeof(uint32_t));

  cudaEvent_t process_time_start, process_time_end;
  cudaEventCreate(&process_time_start);
  cudaEventCreate(&process_time_end);

  cudaEventRecord(process_time_start);
  bincount_kernel<<<DIVCEIL(num_edges, 256), 256>>>(d_list, num_edges,
                                                    d_bincount);
  cudaEventRecord(process_time_end);
  cudaEventSynchronize(process_time_end);

  std::vector<uint32_t> bincount(num_nodes);
  cudaMemcpy(bincount.data(), d_bincount, num_nodes * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  float process_time = 0;
  cudaEventElapsedTime(&process_time, process_time_start, process_time_end);

  printf("%.9f\n", process_time);
}
