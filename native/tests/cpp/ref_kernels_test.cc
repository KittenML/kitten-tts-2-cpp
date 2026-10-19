#include "kt2n/ref_kernels.h"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "kt2n/bf16.h"

namespace {

using kt2n::bf16_to_f32;
using kt2n::f32_to_bf16_rne;

struct Ternary {
  int64_t N = 0, K = 0;
  std::vector<int32_t> codes;
  std::vector<uint16_t> scales;
  std::vector<uint16_t> dense;  // what pack_exact started from
};

// Random canonical ternary matrix, packed by hand (independently of the
// library's unpack).
Ternary random_ternary(int64_t N, int64_t K, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> code(0, 2);
  std::uniform_real_distribution<float> scale(0.004f, 0.09f);
  Ternary t;
  t.N = N;
  t.K = K;
  t.codes.assign(static_cast<size_t>(N * K / 16), 0);
  t.scales.resize(static_cast<size_t>(N * K / 128));
  t.dense.resize(static_cast<size_t>(N * K));
  for (auto& s : t.scales) s = f32_to_bf16_rne(scale(rng));
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t k = 0; k < K; ++k) {
      const int c = code(rng);
      const uint16_t s = t.scales[static_cast<size_t>(n * (K / 128) + k / 128)];
      t.dense[static_cast<size_t>(n * K + k)] = c == 1 ? s : c == 2 ? static_cast<uint16_t>(s | 0x8000u) : 0;
      auto& word = t.codes[static_cast<size_t>(n * (K / 16) + k / 16)];
      word = static_cast<int32_t>(static_cast<uint32_t>(word) | (static_cast<uint32_t>(c) << (2 * (k % 16))));
    }
  }
  return t;
}

std::vector<uint16_t> random_bf16(size_t n, uint32_t seed, float sigma) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, sigma);
  std::vector<uint16_t> out(n);
  for (auto& v : out) v = f32_to_bf16_rne(dist(rng));
  return out;
}

TEST(RefKernels, DecodeTernary) {
  EXPECT_EQ(kt2n::ref::decode_ternary(0, 0x3F80), 0x0000);
  EXPECT_EQ(kt2n::ref::decode_ternary(1, 0x3F80), 0x3F80);
  EXPECT_EQ(kt2n::ref::decode_ternary(2, 0x3F80), 0xBF80);
  EXPECT_EQ(kt2n::ref::decode_ternary(3, 0x3F80), 0x0000);
}

TEST(RefKernels, UnpackHandBuiltWord) {
  // Row of K=128: word 0 holds columns 0..15 with codes 1,2,0,3,1,2,0,0,...
  std::vector<int32_t> codes(8, 0);
  const uint32_t pattern[16] = {1, 2, 0, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1};
  uint32_t w = 0;
  for (int o = 0; o < 16; ++o) w |= pattern[o] << (2 * o);
  codes[0] = static_cast<int32_t>(w);         // column 15 = +1 -> bit 30
  codes[7] = static_cast<int32_t>(2u << 30);  // column 127 = -1 -> bit 31 (a negative int32)
  const std::vector<uint16_t> scales = {0x3E00};  // 0.125
  std::vector<uint16_t> out(128, 0xFFFF);
  EXPECT_EQ(kt2n::ref::unpack_ternary(codes.data(), scales.data(), 1, 128, out.data()), 1u);  // one code 3
  EXPECT_EQ(out[0], 0x3E00);
  EXPECT_EQ(out[1], 0xBE00);
  EXPECT_EQ(out[2], 0x0000);
  EXPECT_EQ(out[3], 0x0000);  // invalid code decodes to 0
  EXPECT_EQ(out[14], 0xBE00);
  EXPECT_EQ(out[15], 0x3E00);
  EXPECT_EQ(out[126], 0x0000);
  EXPECT_EQ(out[127], 0xBE00);
}

TEST(RefKernels, UnpackMatchesDense) {
  const Ternary t = random_ternary(37, 768, 7);
  std::vector<uint16_t> out(t.dense.size());
  EXPECT_EQ(kt2n::ref::unpack_ternary(t.codes.data(), t.scales.data(), t.N, t.K, out.data()), 0u);
  EXPECT_EQ(out, t.dense);
}

TEST(RefKernels, UnpackRejectsBadShapes) {
  std::vector<int32_t> codes(64);
  std::vector<uint16_t> scales(8), out(1024);
  EXPECT_THROW(kt2n::ref::unpack_ternary(codes.data(), scales.data(), 1, 100, out.data()), std::invalid_argument);
  EXPECT_THROW(kt2n::ref::unpack_ternary(codes.data(), scales.data(), -1, 128, out.data()), std::invalid_argument);
  EXPECT_THROW(kt2n::ref::unpack_ternary(codes.data(), scales.data(), 1, 0, out.data()), std::invalid_argument);
}

TEST(RefKernels, GemvFollowsDocumentedOrder) {
  const Ternary t = random_ternary(19, 512, 11);
  const int64_t M = 3;
  const auto x = random_bf16(static_cast<size_t>(M * t.K), 12, 1.0f);
  std::vector<float> y(static_cast<size_t>(M * t.N));
  kt2n::ref::ternary_gemv_ref(t.codes.data(), t.scales.data(), t.N, t.K, x.data(), M, y.data());
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < t.N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < t.K; ++k) {
        acc = acc + bf16_to_f32(t.dense[static_cast<size_t>(n * t.K + k)]) * bf16_to_f32(x[static_cast<size_t>(m * t.K + k)]);
      }
      ASSERT_EQ(std::bit_cast<uint32_t>(y[static_cast<size_t>(m * t.N + n)]), std::bit_cast<uint32_t>(acc))
          << "m " << m << " n " << n;
    }
  }
}

TEST(RefKernels, GemvCloseToF64Oracle) {
  const Ternary t = random_ternary(64, 2048, 21);
  const int64_t M = 2;
  const auto x = random_bf16(static_cast<size_t>(M * t.K), 22, 1.0f);
  std::vector<float> y(static_cast<size_t>(M * t.N));
  std::vector<double> y64(y.size());
  kt2n::ref::ternary_gemv_ref(t.codes.data(), t.scales.data(), t.N, t.K, x.data(), M, y.data());
  kt2n::ref::ternary_gemv_f64(t.codes.data(), t.scales.data(), t.N, t.K, x.data(), M, y64.data());
  for (size_t i = 0; i < y.size(); ++i) {
    const int64_t m = static_cast<int64_t>(i) / t.N;
    const int64_t n = static_cast<int64_t>(i) % t.N;
    double abs_sum = 0.0;
    for (int64_t k = 0; k < t.K; ++k) {
      abs_sum += std::fabs(static_cast<double>(bf16_to_f32(t.dense[static_cast<size_t>(n * t.K + k)])) *
                           static_cast<double>(bf16_to_f32(x[static_cast<size_t>(m * t.K + k)])));
    }
    // Sequential fp32 summation error bound: (K-1) * u * sum|terms|, u = 2^-24.
    const double bound = static_cast<double>(t.K - 1) * std::ldexp(1.0, -24) * abs_sum + 1e-30;
    ASSERT_LE(std::fabs(static_cast<double>(y[i]) - y64[i]), bound) << i;
  }
}

TEST(RefKernels, GemvIsBatchInvariant) {
  const Ternary t = random_ternary(16, 256, 31);
  const int64_t M = 7;
  const auto x = random_bf16(static_cast<size_t>(M * t.K), 32, 1.0f);
  std::vector<float> all(static_cast<size_t>(M * t.N));
  kt2n::ref::ternary_gemv_ref(t.codes.data(), t.scales.data(), t.N, t.K, x.data(), M, all.data());
  for (int64_t m = 0; m < M; ++m) {
    std::vector<float> one(static_cast<size_t>(t.N));
    kt2n::ref::ternary_gemv_ref(t.codes.data(), t.scales.data(), t.N, t.K, x.data() + m * t.K, 1, one.data());
    for (int64_t n = 0; n < t.N; ++n) {
      ASSERT_EQ(std::bit_cast<uint32_t>(one[static_cast<size_t>(n)]),
                std::bit_cast<uint32_t>(all[static_cast<size_t>(m * t.N + n)]));
    }
  }
}

TEST(RefKernels, RmsNormConstantRowReturnsWeight) {
  const int64_t D = 64;
  std::vector<uint16_t> x(static_cast<size_t>(D), f32_to_bf16_rne(2.0f));
  const auto w = random_bf16(static_cast<size_t>(D), 41, 1.0f);
  std::vector<uint16_t> y(static_cast<size_t>(D));
  kt2n::ref::rmsnorm_ref(x.data(), w.data(), 1, D, 1e-6f, y.data());
  // r = 1/sqrt(4 + 1e-6) -> 2r rounds to 1.0 in bf16 -> y == w.
  EXPECT_EQ(y, w);
}

TEST(RefKernels, RmsNormCloseToDouble) {
  const int64_t M = 4, D = 2048;
  const auto x = random_bf16(static_cast<size_t>(M * D), 51, 3.0f);
  const auto w = random_bf16(static_cast<size_t>(D), 52, 1.0f);
  std::vector<uint16_t> y(x.size());
  kt2n::ref::rmsnorm_ref(x.data(), w.data(), M, D, 1e-6f, y.data());
  for (int64_t m = 0; m < M; ++m) {
    double ss = 0.0;
    for (int64_t d = 0; d < D; ++d) {
      const double v = bf16_to_f32(x[static_cast<size_t>(m * D + d)]);
      ss += v * v;
    }
    const double r = 1.0 / std::sqrt(ss / static_cast<double>(D) + 1e-6);
    for (int64_t d = 0; d < D; ++d) {
      const double want = static_cast<double>(bf16_to_f32(w[static_cast<size_t>(d)])) *
                          static_cast<double>(bf16_to_f32(x[static_cast<size_t>(m * D + d)])) * r;
      const double got = bf16_to_f32(y[static_cast<size_t>(m * D + d)]);
      // Two bf16 roundings: within ~2^-7 relative.
      ASSERT_LE(std::fabs(got - want), std::fabs(want) * std::ldexp(1.0, -7) + 1e-30) << m << "," << d;
    }
  }
}

}  // namespace
