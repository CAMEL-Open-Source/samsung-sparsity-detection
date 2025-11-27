#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <vector>

#define UNUSED(x) (void)(x)
#define ENABLE_WARN true
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define MIN(x, y) ((x) > (y) ? (y) : (x))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define DIVCEIL(x, y) (((x) - 1) / (y) + 1)
static constexpr unsigned PAGE_SIZE = 4096;
static constexpr unsigned BUFFER_SIZE = 4096;

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

  uint64_t prepare_done;
  uint64_t allocate_done;
  uint64_t copy_done;

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

constexpr size_t MMAP_SIZE = 1024 * 1024 * 1024UL;

int main(int argc, char **argv) {
  UNUSED(argc);
  UNUSED(argv);
  cfg().start_time = GetNowNanos();

  void *host_buffer;
  host_buffer = malloc(MMAP_SIZE);
  if (host_buffer == nullptr) {
    warn("malloc failed: %s", strerror(errno));
    return 1;
  }
  memset(host_buffer, 0, MMAP_SIZE);
  cfg().prepare_done = GetNowNanos();

  void *ptr;
  cudaMalloc(&ptr, MMAP_SIZE);
  cfg().allocate_done = GetNowNanos();

  // Copy data from host buffer to device buffer
  cudaMemcpy(ptr, host_buffer, MMAP_SIZE, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  cfg().copy_done = GetNowNanos();

  log("allocate: %.3f ms",
      (cfg().allocate_done - cfg().prepare_done) / 1000000.f);
  log("copy: %.3f ms", (cfg().copy_done - cfg().allocate_done) / 1000000.f);

  return 0;
}
