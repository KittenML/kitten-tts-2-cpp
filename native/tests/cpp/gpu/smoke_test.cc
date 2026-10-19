// GPU smoke: the CUDA toolchain, static runtime and arch list work end to end,
// and the device unpack is bit-identical to the CPU reference.
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kt2n/cuda_smoke.h"
#include "kt2n/engine_file.h"
#include "kt2n/ref_kernels.h"
#include "kt2n/sha256.h"
#include "test_util.h"

namespace {

TEST(GpuSmoke, DeviceAndRuntime) {
  KT2N_GPU_OR_SKIP();
  EXPECT_GE(kt2n_cuda_runtime_version(), 12080);
  EXPECT_GT(kt2n_cuda_driver_version(), 0);
  int major = 0, minor = 0;
  char name[256] = {0};
  ASSERT_EQ(kt2n_cuda_device_info(0, &major, &minor, name, sizeof(name)), 0);
  EXPECT_GE(major, 9) << "built for sm_90/100/120 only; device is sm_" << major << minor;
  std::printf("[ info ] device 0: %s (sm_%d%d), runtime %d, driver %d\n", name, major, minor,
              kt2n_cuda_runtime_version(), kt2n_cuda_driver_version());
}

TEST(GpuSmoke, UnpackMatchesCpuReferenceBitwise) {
  KT2N_GPU_OR_SKIP();
  const int64_t N = 96, K = 1024;
  std::mt19937 rng(99);
  std::uniform_int_distribution<uint32_t> any;
  std::vector<int32_t> codes(static_cast<size_t>(N * K / 16));
  for (auto& c : codes) c = static_cast<int32_t>(any(rng));  // includes code 3 and bit 31
  std::vector<uint16_t> scales(static_cast<size_t>(N * K / 128));
  for (auto& s : scales) s = static_cast<uint16_t>(any(rng) & 0x7FFFu);
  scales[0] = 0;
  std::vector<uint16_t> cpu(static_cast<size_t>(N * K)), gpu(static_cast<size_t>(N * K), 0x5555);
  kt2n::ref::unpack_ternary(codes.data(), scales.data(), N, K, cpu.data());
  const int err = kt2n_cuda_smoke_unpack(codes.data(), scales.data(), N, K, gpu.data());
  ASSERT_EQ(err, 0) << kt2n_cuda_error_string(err);
  EXPECT_EQ(std::memcmp(cpu.data(), gpu.data(), cpu.size() * 2), 0);
}

TEST(GpuSmoke, UnpackRebuildsTinyEngineFile) {
  KT2N_GPU_OR_SKIP();
  const kt2n::EngineFile f = kt2n::EngineFile::open(kt2n::test::data_path("tiny.kt2n"));
  int checked = 0;
  for (const auto& t : f.tensors()) {
    if (t.kind != kt2n::TensorKind::kTernary2) continue;
    std::vector<int32_t> codes(t.codes.len / 4);
    std::vector<uint16_t> scales(t.scales.len / 2);
    std::memcpy(codes.data(), f.bytes(t.codes).data(), t.codes.len);
    std::memcpy(scales.data(), f.bytes(t.scales).data(), t.scales.len);
    const size_t n = static_cast<size_t>(t.rows() * t.cols());
    std::vector<uint16_t> cpu(n), gpu(n, 0x5555);
    kt2n::ref::unpack_ternary(codes.data(), scales.data(), t.rows(), t.cols(), cpu.data());
    const int err = kt2n_cuda_smoke_unpack(codes.data(), scales.data(), t.rows(), t.cols(), gpu.data());
    ASSERT_EQ(err, 0) << kt2n_cuda_error_string(err);
    EXPECT_EQ(std::memcmp(cpu.data(), gpu.data(), n * 2), 0) << t.name;
    // Without -0.0 weights the unpack is the source tensor itself.
    if (t.neg_zero == 0) {
      EXPECT_EQ(kt2n::sha256_hex(gpu.data(), n * 2), t.src_sha256) << t.name;
    }
    ++checked;
  }
  EXPECT_EQ(checked, 2);
}

// Opt-in: KT2N_REAL_ENGINE_FILE=/path/to/step19309.kt2n unpacks every ternary
// matrix of a real converted checkpoint on the GPU and compares with the CPU
// reference bit for bit (both give +0.0 for every zero; -0.0 signs are only
// for rebuilding the source and are checked by kt2n_inspect --verify-all).
TEST(GpuSmoke, RealEngineFileUnpackMatchesCpu) {
  KT2N_GPU_OR_SKIP();
  const char* path = std::getenv("KT2N_REAL_ENGINE_FILE");
  if (path == nullptr || *path == '\0') GTEST_SKIP() << "set KT2N_REAL_ENGINE_FILE to run";
  const kt2n::EngineFile f = kt2n::EngineFile::open(path);
  int checked = 0;
  uint64_t weights = 0;
  for (const auto& t : f.tensors()) {
    if (t.kind != kt2n::TensorKind::kTernary2) continue;
    std::vector<int32_t> codes(t.codes.len / 4);
    std::vector<uint16_t> scales(t.scales.len / 2);
    std::memcpy(codes.data(), f.bytes(t.codes).data(), t.codes.len);
    std::memcpy(scales.data(), f.bytes(t.scales).data(), t.scales.len);
    const size_t n = static_cast<size_t>(t.rows() * t.cols());
    std::vector<uint16_t> cpu(n), gpu(n, 0x5555);
    ASSERT_EQ(kt2n::ref::unpack_ternary(codes.data(), scales.data(), t.rows(), t.cols(), cpu.data()), 0u) << t.name;
    const int err = kt2n_cuda_smoke_unpack(codes.data(), scales.data(), t.rows(), t.cols(), gpu.data());
    ASSERT_EQ(err, 0) << kt2n_cuda_error_string(err);
    ASSERT_EQ(std::memcmp(cpu.data(), gpu.data(), n * 2), 0) << t.name;
    ++checked;
    weights += n;
  }
  EXPECT_GT(checked, 0);
  std::printf("[ info ] %d ternary matrices (%llu weights) identical on GPU and CPU\n", checked,
              static_cast<unsigned long long>(weights));
}

TEST(GpuSmoke, RejectsBadShape) {
  KT2N_GPU_OR_SKIP();
  std::vector<int32_t> codes(8);
  std::vector<uint16_t> scales(1), out(128);
  EXPECT_NE(kt2n_cuda_smoke_unpack(codes.data(), scales.data(), 1, 100, out.data()), 0);
}

}  // namespace
