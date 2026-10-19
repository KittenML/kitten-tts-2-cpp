// Shared device helpers for the ops_*.cu kernels (and for any lane that
// wants to reuse the exact same numerics in a fused kernel).
//
// Every rounding point is spelled with an explicit _rn intrinsic so the bits
// do not depend on nvcc's FMA contraction (-fmad) setting.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "kt2n/cuda_caps.h"

namespace kt2n_ops_detail {

// ---------------------------------------------------------------- bf16 helpers

__device__ __forceinline__ float bf2f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 f2bf(float x) { return __float2bfloat16_rn(x); }
// fp32 -> bf16 (round to nearest even) -> fp32: torch's `.to(bfloat16)` on an
// fp32 intermediate, seen again as fp32.
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ void unpack8(const uint4& u, float (&f)[8]) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 t = __bfloat1622float2(p[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
  uint4 u;
  __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) p[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  return u;
}

__device__ __forceinline__ float2 load_pair(const __nv_bfloat16* p) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
}

__device__ __forceinline__ void store_pair(__nv_bfloat16* p, float a, float b) {
  *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a, b);
}

// ------------------------------------------------------------------ PDL hooks
// Kernels are compiled twice: PDL=true for launches with programmatic stream
// serialization (griddepcontrol.wait before touching a predecessor's output,
// launch_dependents once our loads are issued), PDL=false for ordinary
// launches. The griddepcontrol instructions cost ~0.07 us per launch when PDL
// is not in use (measured), so the non-PDL variant omits them. Same bits.

template <bool PDL>
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if constexpr (PDL) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

template <bool PDL>
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if constexpr (PDL) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// ------------------------------------------------------------ reductions
// Triton's tl.sum order (warpReduce): xor-butterfly with offsets 16, 8, 4, 2, 1,
// acc = acc + other at each level. Every lane ends with the same value.
__device__ __forceinline__ float warp_sum_bfly(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Sum of squares of one row held in production `_rms_rows` layout:
// 128 threads (4 warps), thread t owns R chunks of 8 consecutive elements at
// columns c*1024 + 8t + j (c < R, j < 8). In-thread: sequential in register
// order (chunk-major), fp32 mul then add (Triton enable_fp_fusion=False).
// Across lanes: butterfly. Across warps: ((w0 + w2) + (w1 + w3)) (Triton's
// inter-warp partial reduction over 4 lanes, xor 2 then xor 1).
// `red` is 4 floats of shared memory; one __syncthreads inside.
template <int R>
__device__ __forceinline__ float row_sumsq_128(const float (&x)[R][8], float* red) {
  float acc = __fmul_rn(x[0][0], x[0][0]);
#pragma unroll
  for (int j = 1; j < 8; ++j) acc = __fadd_rn(acc, __fmul_rn(x[0][j], x[0][j]));
#pragma unroll
  for (int c = 1; c < R; ++c) {
#pragma unroll
    for (int j = 0; j < 8; ++j) acc = __fadd_rn(acc, __fmul_rn(x[c][j], x[c][j]));
  }
  acc = warp_sum_bfly(acc);
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) red[warp] = acc;
  __syncthreads();
  return __fadd_rn(__fadd_rn(red[0], red[2]), __fadd_rn(red[1], red[3]));
}

// RMSNorm of a row already in registers (layout above); writes y (bf16 bits
// as fp32 values ready for pack8). HF / production rounding points:
//   inv = rsqrt(sum/n + eps); y = bf16( bf16(x * inv) * w ).
template <int R>
__device__ __forceinline__ void rmsnorm_row_128(const float (&x)[R][8], const float (&w)[R][8],
                                                float eps, float* red, float (&y)[R][8]) {
  const float total = row_sumsq_128<R>(x, red);
  // n = R * 1024 is a power of two: total * (1/n) is exactly total / n (the same
  // real value, one correct rounding) and needs no division routine.
  constexpr float kInvN = 1.0f / static_cast<float>(R * 1024);
  const float var = __fmul_rn(total, kInvN);
  const float inv = rsqrtf(__fadd_rn(var, eps));
#pragma unroll
  for (int c = 0; c < R; ++c) {
#pragma unroll
    for (int j = 0; j < 8; ++j) y[c][j] = __fmul_rn(rbf(__fmul_rn(x[c][j], inv)), w[c][j]);
  }
}

// ------------------------------------------------------------------ RoPE
// HF Qwen3RotaryEmbedding buffer for rope_theta = 1e6, head_dim = 128, as
// torch computes it on the CPU at model init:
//   1.0 / (1000000 ** (torch.arange(0, 128, 2, dtype=torch.int64).float() / 128))
// (63/64 entries equal the correctly rounded value; entry bits are torch's.)
// Verified against torch 2.7.1 and the loaded production model on the box.
#define KT2N_OPS_INV_FREQ_BITS                                                  \
  0x3f800000u, 0x3f4e4badu, 0x3f263de0u, 0x3f05f6efu, 0x3ed7e89bu, 0x3eadfcffu, \
  0x3e8c3504u, 0x3e61f835u, 0x3e361887u, 0x3e12bd91u, 0x3dec7fd6u, 0x3dbe94c6u, \
  0x3d99940du, 0x3d778513u, 0x3d47763fu, 0x3d20bc1du, 0x3d0186e3u, 0x3cd0c1a8u, \
  0x3ca8398bu, 0x3c879008u, 0x3c5a7bf2u, 0x3c301052u, 0x3c0de12du, 0x3be4aa46u, \
  0x3bb8449cu, 0x3b947daeu, 0x3b6f520eu, 0x3b40dac5u, 0x3b1b690du, 0x3afa78f0u, \
  0x3ac9d75cu, 0x3aa2a6f7u, 0x3a83126fu, 0x3a533f28u, 0x3a2a3b44u, 0x3a092e02u, \
  0x39dd1725u, 0x39b229fbu, 0x398f9272u, 0x39676492u, 0x393a7753u, 0x39164324u, \
  0x38f22ce2u, 0x38c327b5u, 0x389d43a4u, 0x387d75d4u, 0x384c3fbeu, 0x382497abu, \
  0x3804a2b2u, 0x37d5c441u, 0x37ac431du, 0x378ad0edu, 0x375fba4fu, 0x37344a0eu, \
  0x371148e3u, 0x36ea2732u, 0x36bcb0c1u, 0x36980e02u, 0x36751070u, 0x36457babu, \
  0x361f23e4u, 0x36003decu, 0x35ceaf79u, 0x35a68e4cu

// Per-TU copy of the table in GLOBAL memory (internal linkage; no -rdc).
// Not __constant__: measured on the RTX PRO 6000, a module that carries
// __constant__ data costs every kernel in it ~0.09 us per launch, and lanes
// reading different constant addresses serialise (+0.33 us for in-kernel
// cos/sin).
static __device__ const uint32_t g_inv_freq_bits[64] = {KT2N_OPS_INV_FREQ_BITS};

__device__ __forceinline__ float inv_freq_at(const float* inv_freq, int i) {
  return inv_freq ? inv_freq[i] : __uint_as_float(g_inv_freq_bits[i]);
}

// cos/sin of one frequency at one position, HF's rounding:
//   freqs = inv_freq @ position.float() (one fp32 product, cuBLAS K=1),
//   cos/sin in fp32 (cosf/sinf, not the fast intrinsics), then -> bf16.
__device__ __forceinline__ void rope_cs(float inv_freq, int64_t pos, float& c, float& s) {
  const float f = __fmul_rn(inv_freq, __ll2float_rn(static_cast<long long>(pos)));
  c = rbf(cosf(f));
  s = rbf(sinf(f));
}

// q*cos + rotate_half(q)*sin for the pair (a = n[i], b = n[i+64]), each
// product rounded to bf16, the sum rounded to bf16 (fp32 values returned).
__device__ __forceinline__ void rope_pair(float a, float b, float c, float s, float& lo, float& hi) {
  lo = __fsub_rn(rbf(__fmul_rn(a, c)), rbf(__fmul_rn(b, s)));
  hi = __fadd_rn(rbf(__fmul_rn(b, c)), rbf(__fmul_rn(a, s)));
}

// -------------------------------------------------------------- launching

// `kernel` must be the PDL=true instantiation when pdl != 0 (see pdl_wait).
template <typename... ExpTypes, typename... ActTypes>
static cudaError_t launch_k(void (*kernel)(ExpTypes...), dim3 grid, dim3 block, size_t smem,
                            cudaStream_t stream, int pdl, ActTypes&&... args) {
  pdl = kt2n::cuda::pdl_enabled(pdl != 0);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1u : 0u;
  // cudaLaunchKernelEx reports launch errors directly; we do not call
  // cudaGetLastError so an unrelated pending error is not consumed here.
  return cudaLaunchKernelEx(&cfg, kernel, static_cast<ActTypes&&>(args)...);
}

static inline bool aligned_to(const void* p, uintptr_t a) {
  return (reinterpret_cast<uintptr_t>(p) % a) == 0;
}

}  // namespace kt2n_ops_detail
