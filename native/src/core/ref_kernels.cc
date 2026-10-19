#include "kt2n/ref_kernels.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "kt2n/bf16.h"

namespace kt2n::ref {
namespace {

void check_shape(int64_t N, int64_t K, const char* who) {
  if (N < 0 || K <= 0 || K % kGroup != 0) {
    throw std::invalid_argument(std::string(who) + ": need N >= 0 and K > 0 with K % 128 == 0 (N=" +
                                std::to_string(N) + ", K=" + std::to_string(K) + ")");
  }
}

}  // namespace

uint64_t unpack_ternary_row(const int32_t* codes_row, const uint16_t* scales_row, int64_t K,
                            uint16_t* out_row) {
  uint64_t invalid = 0;
  const int64_t words = K / kCodesPerWord;
  for (int64_t j = 0; j < words; ++j) {
    const uint32_t word = static_cast<uint32_t>(codes_row[j]);
    // 16 columns of one word always sit inside one 128-group (128 % 16 == 0).
    const uint16_t scale = scales_row[(j * kCodesPerWord) / kGroup];
    for (int64_t o = 0; o < kCodesPerWord; ++o) {
      const uint32_t code = (word >> (2 * o)) & 3u;
      invalid += code == 3u ? 1u : 0u;
      out_row[j * kCodesPerWord + o] = decode_ternary(code, scale);
    }
  }
  return invalid;
}

uint64_t unpack_ternary(const int32_t* codes, const uint16_t* scales, int64_t N, int64_t K,
                        uint16_t* out) {
  check_shape(N, K, "unpack_ternary");
  uint64_t invalid = 0;
  for (int64_t n = 0; n < N; ++n) {
    invalid += unpack_ternary_row(codes + n * (K / kCodesPerWord), scales + n * (K / kGroup), K,
                                  out + n * K);
  }
  return invalid;
}

void ternary_gemv_ref(const int32_t* codes, const uint16_t* scales, int64_t N, int64_t K,
                      const uint16_t* x, int64_t M, float* y) {
  check_shape(N, K, "ternary_gemv_ref");
  if (M < 0) throw std::invalid_argument("ternary_gemv_ref: M < 0");
  std::vector<uint16_t> w(static_cast<size_t>(K));
  for (int64_t n = 0; n < N; ++n) {
    unpack_ternary_row(codes + n * (K / kCodesPerWord), scales + n * (K / kGroup), K, w.data());
    for (int64_t m = 0; m < M; ++m) {
      const uint16_t* xr = x + m * K;
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        const float prod = bf16_to_f32(w[static_cast<size_t>(k)]) * bf16_to_f32(xr[k]);
        acc = acc + prod;
      }
      y[m * N + n] = acc;
    }
  }
}

void ternary_gemv_f64(const int32_t* codes, const uint16_t* scales, int64_t N, int64_t K,
                      const uint16_t* x, int64_t M, double* y) {
  check_shape(N, K, "ternary_gemv_f64");
  if (M < 0) throw std::invalid_argument("ternary_gemv_f64: M < 0");
  std::vector<uint16_t> w(static_cast<size_t>(K));
  for (int64_t n = 0; n < N; ++n) {
    unpack_ternary_row(codes + n * (K / kCodesPerWord), scales + n * (K / kGroup), K, w.data());
    for (int64_t m = 0; m < M; ++m) {
      const uint16_t* xr = x + m * K;
      double acc = 0.0;
      for (int64_t k = 0; k < K; ++k) {
        acc += static_cast<double>(bf16_to_f32(w[static_cast<size_t>(k)])) *
               static_cast<double>(bf16_to_f32(xr[k]));
      }
      y[m * N + n] = acc;
    }
  }
}

void rmsnorm_ref(const uint16_t* x, const uint16_t* w, int64_t M, int64_t D, float eps, uint16_t* y) {
  if (M < 0 || D <= 0) throw std::invalid_argument("rmsnorm_ref: need M >= 0 and D > 0");
  for (int64_t m = 0; m < M; ++m) {
    const uint16_t* xr = x + m * D;
    uint16_t* yr = y + m * D;
    float ss = 0.0f;
    for (int64_t d = 0; d < D; ++d) {
      const float v = bf16_to_f32(xr[d]);
      ss = ss + v * v;
    }
    const float var = ss / static_cast<float>(D);
    const float r = 1.0f / std::sqrt(var + eps);
    for (int64_t d = 0; d < D; ++d) {
      const float h = round_through_bf16(bf16_to_f32(xr[d]) * r);
      yr[d] = f32_to_bf16_rne(bf16_to_f32(w[d]) * h);
    }
  }
}

}  // namespace kt2n::ref
