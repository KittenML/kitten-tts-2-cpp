// Shared helpers for kt2n C++ tests (also on the include path of
// tests/kernels/*_test.{cc,cu}).
#pragma once

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef KT2N_TEST_DATA_DIR_DEFAULT
#define KT2N_TEST_DATA_DIR_DEFAULT "test-data"
#endif

namespace kt2n::test {

// Directory holding tiny.kt2n & co (written by the kt2n_make_tiny fixture).
inline std::string data_dir() {
  const char* d = std::getenv("KT2N_TEST_DATA_DIR");
  return (d != nullptr && *d != '\0') ? std::string(d) : std::string(KT2N_TEST_DATA_DIR_DEFAULT);
}

inline std::string data_path(const std::string& name) { return data_dir() + "/" + name; }

inline std::vector<uint8_t> read_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

inline std::string read_text(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

inline void write_bytes(const std::string& path, const std::vector<uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write " + path);
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

// KT2N_REQUIRE_GPU=1 turns "no GPU" from a skip into a failure (GPU box, CI GPU runners).
inline bool gpu_required() {
  const char* r = std::getenv("KT2N_REQUIRE_GPU");
  return r != nullptr && std::string(r) == "1";
}

}  // namespace kt2n::test

#if defined(KT2N_HAVE_CUDA)
#include "kt2n/cuda_smoke.h"
// Use at the top of a GPU test body.
#define KT2N_GPU_OR_SKIP()                                                       \
  do {                                                                           \
    if (kt2n_cuda_device_count() == 0) {                                         \
      if (::kt2n::test::gpu_required()) {                                        \
        FAIL() << "no CUDA device visible but KT2N_REQUIRE_GPU=1";               \
      }                                                                          \
      GTEST_SKIP() << "no CUDA device visible (set KT2N_REQUIRE_GPU=1 to fail)"; \
    }                                                                            \
  } while (0)
#endif
