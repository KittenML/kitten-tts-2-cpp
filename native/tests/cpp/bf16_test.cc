#include "kt2n/bf16.h"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

namespace {

float f(uint32_t bits) { return std::bit_cast<float>(bits); }

// Independent RNE reference for finite floats: pick the nearer of the two
// bf16 neighbours (by exact double distance), ties to the even one.
uint16_t rne_reference(float x) {
  const uint32_t u = std::bit_cast<uint32_t>(x);
  const uint16_t lo = static_cast<uint16_t>(u >> 16);  // truncation toward zero (same sign)
  const uint16_t hi = static_cast<uint16_t>(lo + 1);   // next magnitude up
  const double dx = static_cast<double>(x);
  const double dlo = std::fabs(dx - static_cast<double>(kt2n::bf16_to_f32(lo)));
  // hi may be +-inf when lo is the largest finite bf16: treat the boundary as 2^128.
  double vhi = static_cast<double>(kt2n::bf16_to_f32(hi));
  if (std::isinf(vhi)) vhi = std::copysign(std::ldexp(1.0, 128), dx);
  const double dhi = std::fabs(dx - vhi);
  if (dlo < dhi) return lo;
  if (dhi < dlo) return hi;
  return (lo & 1u) == 0 ? lo : hi;
}

TEST(Bf16, ExactValues) {
  EXPECT_EQ(kt2n::bf16_to_f32(0x3F80), 1.0f);
  EXPECT_EQ(kt2n::bf16_to_f32(0xC000), -2.0f);
  EXPECT_EQ(kt2n::f32_to_bf16_rne(1.0f), 0x3F80);
  EXPECT_EQ(kt2n::f32_to_bf16_rne(-2.0f), 0xC000);
  EXPECT_EQ(kt2n::f32_to_bf16_rne(0.0f), 0x0000);
  EXPECT_EQ(kt2n::f32_to_bf16_rne(-0.0f), 0x8000);
}

TEST(Bf16, TiesToEven) {
  EXPECT_EQ(kt2n::f32_to_bf16_rne(f(0x3F808000u)), 0x3F80);  // tie, even below
  EXPECT_EQ(kt2n::f32_to_bf16_rne(f(0x3F818000u)), 0x3F82);  // tie, odd below -> up
  EXPECT_EQ(kt2n::f32_to_bf16_rne(f(0x3F808001u)), 0x3F81);  // just above half
  EXPECT_EQ(kt2n::f32_to_bf16_rne(f(0x3F807FFFu)), 0x3F80);  // just below half
  EXPECT_EQ(kt2n::f32_to_bf16_rne(f(0xBF818000u)), 0xBF82);  // negative tie
}

TEST(Bf16, SpecialsAndSubnormals) {
  const float inf = std::numeric_limits<float>::infinity();
  EXPECT_EQ(kt2n::f32_to_bf16_rne(inf), 0x7F80);
  EXPECT_EQ(kt2n::f32_to_bf16_rne(-inf), 0xFF80);
  EXPECT_EQ(kt2n::f32_to_bf16_rne(f(0x7F7FFFFFu)), 0x7F80);  // FLT_MAX rounds to inf
  EXPECT_EQ(kt2n::f32_to_bf16_rne(f(0x7F7F7FFFu)), 0x7F7F);  // stays finite
  EXPECT_EQ(kt2n::f32_to_bf16_rne(std::numeric_limits<float>::quiet_NaN()), kt2n::kBf16CanonicalNan);
  EXPECT_EQ(kt2n::f32_to_bf16_rne(f(0xFF800001u)), kt2n::kBf16CanonicalNan);  // negative signalling NaN
  EXPECT_TRUE(kt2n::bf16_is_nan(kt2n::kBf16CanonicalNan));
  EXPECT_FALSE(kt2n::bf16_is_nan(0x7F80));
  EXPECT_FALSE(kt2n::bf16_is_finite(0x7F80));
  EXPECT_TRUE(kt2n::bf16_is_finite(0x7F7F));
  // No flush to zero.
  EXPECT_EQ(kt2n::f32_to_bf16_rne(f(0x00000001u)), 0x0000);
  EXPECT_EQ(kt2n::f32_to_bf16_rne(f(0x00008000u)), 0x0000);  // tie -> even (0)
  EXPECT_EQ(kt2n::f32_to_bf16_rne(f(0x00018000u)), 0x0002);  // tie -> even (2)
  EXPECT_EQ(kt2n::f32_to_bf16_rne(f(0x00008001u)), 0x0001);
  EXPECT_EQ(kt2n::f32_to_bf16_rne(f(0x00010000u)), 0x0001);  // smallest bf16 subnormal, exact
  EXPECT_EQ(kt2n::bf16_negate(0x3F80), 0xBF80);
}

TEST(Bf16, EveryBf16RoundTrips) {
  for (uint32_t b = 0; b <= 0xFFFFu; ++b) {
    const auto bits = static_cast<uint16_t>(b);
    if (kt2n::bf16_is_nan(bits)) continue;
    ASSERT_EQ(kt2n::f32_to_bf16_rne(kt2n::bf16_to_f32(bits)), bits) << std::hex << b;
  }
}

TEST(Bf16, MatchesIndependentReference) {
  std::mt19937 rng(1234);
  std::uniform_int_distribution<uint32_t> any;
  int checked = 0;
  for (int i = 0; i < 2000000; ++i) {
    const uint32_t u = any(rng);
    const float x = f(u);
    if (!std::isfinite(x)) continue;
    ASSERT_EQ(kt2n::f32_to_bf16_rne(x), rne_reference(x)) << std::hex << u;
    ++checked;
  }
  // Every halfway case near 1.0 explicitly.
  for (uint32_t hi = 0x3F80; hi < 0x3FA0; ++hi) {
    const float x = f((hi << 16) | 0x8000u);
    ASSERT_EQ(kt2n::f32_to_bf16_rne(x), rne_reference(x));
  }
  EXPECT_GT(checked, 1900000);
}

}  // namespace
