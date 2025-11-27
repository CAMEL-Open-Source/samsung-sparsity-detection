#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
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
  std::ifstream pagemap;

  uint64_t prepare_done;
  uint64_t mapping_done;
  uint64_t change_done;

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

constexpr size_t TABLE_COUNT = 8388608UL;

uint64_t virtualToPhysical(void *virtualAddress) {
  constexpr size_t pageSize = 4096;  // Typical page size
  uintptr_t virtAddr = reinterpret_cast<uintptr_t>(virtualAddress);

  // Calculate the offset in pagemap
  size_t pageIndex = virtAddr / pageSize;
  size_t offset = pageIndex * sizeof(uint64_t);

  // Seek to the offset
  cfg().pagemap.seekg(offset, std::ios::beg);

  // Read the entry
  uint64_t entry = 0;
  cfg().pagemap.read(reinterpret_cast<char *>(&entry), sizeof(entry));

  // Extract the physical frame number
  uint64_t frameNumber = entry & ((1ULL << 55) - 1);
  uint64_t physAddr = frameNumber * pageSize + (virtAddr % pageSize);

  return physAddr;
}

int main(int argc, char **argv) {
  UNUSED(argc);
  UNUSED(argv);
  cfg().start_time = GetNowNanos();

  cfg().pagemap.open("/proc/self/pagemap", std::ios::in | std::ios::binary);
  if (!cfg().pagemap.is_open()) {
    warn("Failed to open pagemap");
    return 1;
  }

  // create a hash table
  std::unordered_map<uint64_t, uint64_t> table;
  for (size_t i = 0; i < TABLE_COUNT; i++) {
    table[i] = i;
  }

  cfg().prepare_done = GetNowNanos();

  for (size_t i = 0; i < TABLE_COUNT; i++) {
    uint64_t pgx = virtualToPhysical(&table[i]);
    pgx *= 2;
    pgx = virtualToPhysical(&table[i]);
    pgx *= 2;
  }

  cfg().mapping_done = GetNowNanos();

  for (size_t i = 0; i < TABLE_COUNT; i++) {
    table[i] = i * 2;
    table[i] = table[i] * 2;
  }

  cfg().change_done = GetNowNanos();

  log("mapping: %.3f ms",
      (cfg().mapping_done - cfg().prepare_done) / 1000000.f);
  log("change: %.3f ms", (cfg().change_done - cfg().mapping_done) / 1000000.f);

  return 0;
}
