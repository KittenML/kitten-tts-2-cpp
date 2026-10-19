// kt2n lane k4: torch-exact Philox4x32-10 exponential noise (device helpers).
//
// Derivation (torch 2.7.1+cu128):
//   t.exponential_(1, generator=g) on a contiguous fp32 row of N elements runs
//   distribution_elementwise_grid_stride_kernel<float, unroll 4> with block 256
//   and grid ceil(N/256) (6568 -> 26 x 256; capped by SMs*blocks/SM, never hit
//   here).  Thread idx = element i does
//       curand_init(seed, subsequence = i, offset = g.philox_offset, &st);
//       float4 r = curand_uniform4(&st);   // only r.x is used (ii = 0)
//   and g.philox_offset advances by ((N-1)/(256*grid*4)+1)*4 = 4 per call.
//   curand_init with offset % 4 == 0 leaves STATE = 0 and counter
//       (lo32(off/4), hi32(off/4), lo32(i), hi32(i)), key (lo32(seed), hi32(seed)),
//   so r.x = Philox4x32_10(counter, key).x * 2^-32 + 2^-33 (float), and
//       noise = u >= 1 - 2^-24 ? 2^-24 : (-1/lambda) * __logf(u),  lambda = 1,
//   where __logf is the FAST approximation: at::log<float> on the device is
//   `return __logf(x);` (ATen/NumericUtils.h, "for peak bandwidth"), not logf.
// Everything below is written with explicit _rn intrinsics so no compiler
// contraction can change a bit.
#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

namespace kt2n_samp {

__device__ __forceinline__ uint4 philox_round(uint4 ctr, uint2 key) {
  const unsigned int m0 = 0xD2511F53u;
  const unsigned int m1 = 0xCD9E8D57u;
  const unsigned int hi0 = __umulhi(m0, ctr.x);
  const unsigned int lo0 = m0 * ctr.x;
  const unsigned int hi1 = __umulhi(m1, ctr.z);
  const unsigned int lo1 = m1 * ctr.z;
  return make_uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
}

// curand_Philox4x32_10: 10 rounds, the key bumped between rounds.
__device__ __forceinline__ uint4 philox4x32_10(uint4 c, uint2 k) {
#pragma unroll
  for (int round = 0; round < 10; ++round) {
    c = philox_round(c, k);
    if (round < 9) {
      k.x += 0x9E3779B9u;
      k.y += 0xBB67AE85u;
    }
  }
  return c;
}

// The x lane of curand_uniform4 for (seed, subsequence = element, offset).
// offset must be a multiple of 4 (torch asserts it).
__device__ __forceinline__ unsigned int philox_x(uint64_t seed, uint64_t offset,
                                                 uint64_t element) {
  const uint64_t blocks = offset >> 2;  // skipahead(offset): offset/4 counter blocks
  uint4 ctr = make_uint4(static_cast<unsigned int>(blocks),
                         static_cast<unsigned int>(blocks >> 32),
                         static_cast<unsigned int>(element),
                         static_cast<unsigned int>(element >> 32));
  uint2 key = make_uint2(static_cast<unsigned int>(seed),
                         static_cast<unsigned int>(seed >> 32));
  return philox4x32_10(ctr, key).x;
}

// _curand_uniform: x * CURAND_2POW32_INV + CURAND_2POW32_INV / 2 in float.
// The product is exact (power-of-two scale), so FMA vs mul+add is moot.
__device__ __forceinline__ float uniform_from_bits(unsigned int x) {
  const float two_m32 = 2.3283064365386963e-10f;  // 2^-32 == CURAND_2POW32_INV
  const float two_m33 = 1.1641532182693481e-10f;  // 2^-33
  return __fadd_rn(__fmul_rn(__uint2float_rn(x), two_m32), two_m33);
}

// at::transformation::exponential<float>(u, 1.0f) on CUDA (at::log == __logf).
__device__ __forceinline__ float exponential_from_uniform(float u) {
  const float half_eps = 5.9604644775390625e-08f;  // FLT_EPSILON / 2 = 2^-24
  const float lg = (u >= __fsub_rn(1.0f, half_eps)) ? -half_eps : __logf(u);
  return __fmul_rn(__fdiv_rn(-1.0f, 1.0f), lg);
}

__device__ __forceinline__ float torch_exponential(uint64_t seed, uint64_t offset,
                                                   uint64_t element) {
  return exponential_from_uniform(uniform_from_bits(philox_x(seed, offset, element)));
}

}  // namespace kt2n_samp
