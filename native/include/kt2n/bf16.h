// bfloat16 helpers on raw uint16_t bits (host side; header only).
//
// f32_to_bf16_rne() rounds to nearest, ties to even, exactly like torch's CPU
// c10::BFloat16 (round_to_nearest_even): no flush-to-zero, subnormals are
// rounded like any other value, overflow rounds to +-inf, and every NaN maps
// to the canonical quiet NaN 0x7FC0 (torch CPU behaviour; CUDA's
// __float2bfloat16_rn returns 0x7FFF instead, so compare NaNs with
// bf16_is_nan(), never bitwise).
#pragma once

#include <bit>
#include <cstdint>

namespace kt2n {

static_assert(std::endian::native == std::endian::little, "kt2n assumes a little-endian host");

inline constexpr uint16_t kBf16SignMask = 0x8000u;
inline constexpr uint16_t kBf16CanonicalNan = 0x7FC0u;

inline float bf16_to_f32(uint16_t bits) noexcept {
  return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

inline bool bf16_is_nan(uint16_t bits) noexcept {
  return (bits & 0x7F80u) == 0x7F80u && (bits & 0x007Fu) != 0;
}

inline bool bf16_is_finite(uint16_t bits) noexcept { return (bits & 0x7F80u) != 0x7F80u; }

inline uint16_t f32_to_bf16_rne(float value) noexcept {
  const uint32_t u = std::bit_cast<uint32_t>(value);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return kBf16CanonicalNan;
  const uint32_t rounding_bias = ((u >> 16) & 1u) + 0x7FFFu;
  return static_cast<uint16_t>((u + rounding_bias) >> 16);
}

// Round a float through bf16 and back (what torch does when a bf16 op stores).
inline float round_through_bf16(float value) noexcept {
  return bf16_to_f32(f32_to_bf16_rne(value));
}

inline uint16_t bf16_negate(uint16_t bits) noexcept {
  return static_cast<uint16_t>(bits ^ kBf16SignMask);
}

}  // namespace kt2n
