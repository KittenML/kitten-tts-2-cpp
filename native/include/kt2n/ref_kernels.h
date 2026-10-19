// CPU reference kernels: slow, exact, and with a documented order of
// operations, so GPU kernels can be checked bit-for-bit where they promise the
// same order and with a tolerance otherwise.
//
// Ternary layout (pack_exact, shared with the converter and the engine file):
//   weight W: bf16 [N, K] row-major, K % 256 == 0, every 128-group along K is
//   exactly {-s, 0, +s}; s = max|w| over the group (bf16, >= 0).
//   codes:  int32 [N, K/16]; column 16*j + o of row n sits in word (n, j) at
//           bits 2*o .. 2*o+1, LSB first. Code 0 -> 0, 1 -> +s, 2 -> -s.
//           Code 3 never appears in a valid file.
//   scales: bf16 [N, K/128]; scale (n, g) covers columns 128*g .. 128*g+127.
//
// All pointers are host pointers to contiguous row-major arrays. bf16 values
// are raw uint16_t bits (see kt2n/bf16.h).
#pragma once

#include <cstdint>

namespace kt2n::ref {

inline constexpr int64_t kGroup = 128;        // weights per scale
inline constexpr int64_t kCodesPerWord = 16;  // 2-bit codes per int32 word

// Decode one weight. Code 0 -> +0.0 (0x0000), 1 -> s, 2 -> s with the sign bit
// flipped (exactly -1.0f * s, no rounding involved). Code 3 -> 0x0000 (same as
// production's unpack()); callers that care count it via the return value of
// unpack_ternary*.
inline uint16_t decode_ternary(uint32_t code, uint16_t scale_bits) noexcept {
  return code == 1u ? scale_bits
         : code == 2u ? static_cast<uint16_t>(scale_bits ^ 0x8000u)
                      : static_cast<uint16_t>(0);
}

// Unpack one row: codes [K/16], scales [K/128] -> out bf16 [K].
// Returns the number of invalid (== 3) codes seen. K must be a multiple of 128.
uint64_t unpack_ternary_row(const int32_t* codes_row, const uint16_t* scales_row, int64_t K,
                            uint16_t* out_row);

// Unpack a whole matrix: codes [N, K/16], scales [N, K/128] -> out bf16 [N, K].
// Returns the number of invalid (== 3) codes. Throws std::invalid_argument on a
// bad shape (N < 0, K <= 0 or K % 128 != 0).
uint64_t unpack_ternary(const int32_t* codes, const uint16_t* scales, int64_t N, int64_t K,
                        uint16_t* out);

// Batched ternary GEMV: y[m, n] = sum_k W[n, k] * x[m, k], fp32 accumulation.
//   x: bf16 [M, K]; y: fp32 [M, N] (round with f32_to_bf16_rne to get a bf16 output).
// ORDER (the contract): for each (m, n), acc = 0.0f; for k = 0, 1, ..., K-1 in
// ascending order: acc = acc + float(W[n, k]) * float(x[m, k]). The scale is
// folded into the decoded bf16 weight (as tk_tc4 does), so every product of two
// bf16 values is exact in fp32 (8-bit x 8-bit significands; only a product that
// underflows below fp32's normal range would round) and only the fp32 additions
// round (one rounding per k; with or without FMA the result is identical). Rows of x are
// independent, so the result for row m does not depend on M (batch invariant).
void ternary_gemv_ref(const int32_t* codes, const uint16_t* scales, int64_t N, int64_t K,
                      const uint16_t* x, int64_t M, float* y);

// Same as ternary_gemv_ref but accumulates in double (products are exact, so
// this is within a few ulps of the exact dot product): the oracle for tolerance
// checks of kernels that sum in a different order.
void ternary_gemv_f64(const int32_t* codes, const uint16_t* scales, int64_t N, int64_t K,
                      const uint16_t* x, int64_t M, double* y);

// RMSNorm with the HF Qwen3RMSNorm semantics on bf16 input:
//   xf = float(x); ss = sum_{d ascending} xf*xf (fp32, each square exact);
//   var = ss / float(D); r = 1.0f / sqrtf(var + eps)   (correctly rounded sqrt and div);
//   h = bf16(xf * r); y = bf16(float(w) * float(h)).
// x, y: bf16 [M, D]; w: bf16 [D]. (torch's CPU rsqrt gives the same r; CUDA
// rsqrtf() can differ by up to 2 ulp, so GPU kernels either use 1.0f/sqrtf or
// compare with a tolerance. torch sums in a different order, also a tolerance.)
void rmsnorm_ref(const uint16_t* x, const uint16_t* w, int64_t M, int64_t D, float eps, uint16_t* y);

}  // namespace kt2n::ref
