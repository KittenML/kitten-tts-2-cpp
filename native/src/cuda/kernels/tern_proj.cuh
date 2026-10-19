// kt2n k1-ternary: device helpers shared by the ternary projection kernels (and reusable by a fused/persistent
// step kernel).  No host code, no torch.  See include/kt2n/kernels/tern_proj.h for the layouts.
//
// Arithmetic contract (what makes a row's result independent of the batch):
//   * K is cut into 256-wide chunks; inside a chunk, lane (g, t) owns physical k = 256c + 64t + [0, 64) and MMA
//     step j (0..15) feeds k = 256c + 64t + 4j + {0,1} as the fragment's logical (2t, 2t+1) and + {2,3} as
//     (2t+8, 2t+9), identically for activations (A) and weights (B).  Every physical k is used once per chunk.
//   * A family's K range is split into WK partial sums of CPW consecutive chunks; partial w is the MMA chain
//     c = 0; for chunk in [w*CPW, (w+1)*CPW): for j in 0..15: c = mma(a, b, c);
//     the output is ((p0 + p1) + p2) + ... in fp32, rounded once to bf16 by the epilogue.
//   * mma.sync computes each output row from that row of A only, so neither M nor the other rows' contents nor
//     the row's position in the 16-row MMA tile changes a row's bits (tested, incl. adversarial rows).
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

namespace kt2n_tern {

using bf16_t = __nv_bfloat16;

constexpr int kGroup = 128;
constexpr int kChunk = 256;

__device__ __forceinline__ void mma_16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                          uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// PRMT byte tables of one (row, 128-group) from the bf16 scale bits s (s >= 0):
//   ta = bytes [0, lo(s), lo(s), 0], tb = bytes [0, hi(s), hi(s) ^ 0x80, 0]
// so code 0 -> 0x0000 (+0.0), 1 -> s, 2 -> s with the sign bit set (-s): exactly bf16(sign * s).
__device__ __forceinline__ void scale_tables(uint32_t s, uint32_t& ta, uint32_t& tb) {
    const uint32_t lo = s & 0xFFu, hi = (s >> 8) & 0xFFu;
    ta = (lo << 8) | (lo << 16);
    tb = (hi << 8) | ((hi ^ 0x80u) << 16);
}

// Step q (0..3) of a "sel" word (code of k-offset 4q+r at bit 2q+8r): b0 = bf16x2 {w(4q), w(4q+1)},
// b1 = {w(4q+2), w(4q+3)} (low half first).  6 integer instructions per 4 weights.
__device__ __forceinline__ void decode_step(uint32_t word, int q, uint32_t ta, uint32_t tb, uint32_t& b0,
                                            uint32_t& b1) {
    const uint32_t y = (word >> (2 * q)) & 0x03030303u;  // byte r = code r
    const uint32_t sel = y * 0x11u + 0x40404040u;           // nibble 2r = code, 2r+1 = 4 + code
    b0 = __byte_perm(ta, tb, sel);
    b1 = __byte_perm(ta, tb, sel >> 16);
}

__device__ __forceinline__ uint32_t word_of(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

__device__ __forceinline__ float bf16_round(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// NORM prologue (L3): 8 bf16 activations x times 8 bf16 RMSNorm weights w -> bf16(x * w) (one rounding), and
// ss += x * x in fp32 for the row's mean square.  Element 2i is the low half of word i.
__device__ __forceinline__ uint4 norm_mul8(const uint4 x, const uint4 w, float& ss) {
    const uint32_t xs[4] = {x.x, x.y, x.z, x.w};
    const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
    uint32_t o[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float x0 = __uint_as_float(xs[q] << 16), x1 = __uint_as_float(xs[q] & 0xFFFF0000u);
        const float w0 = __uint_as_float(ws[q] << 16), w1 = __uint_as_float(ws[q] & 0xFFFF0000u);
        ss = __fmaf_rn(x0, x0, ss);
        ss = __fmaf_rn(x1, x1, ss);
        const __nv_bfloat162 r = __floats2bfloat162_rn(__fmul_rn(x0, w0), __fmul_rn(x1, w1));
        o[q] = *reinterpret_cast<const uint32_t*>(&r);
    }
    return make_uint4(o[0], o[1], o[2], o[3]);
}

// HF/torch MLP: act_fn(gate) on the bf16 gate (SiLU in fp32: x / (1 + exp(-x)), rounded to bf16), times the
// bf16 up (fp32 product rounded to bf16).  Accumulators are rounded to bf16 first, as the projection would.
__device__ __forceinline__ __nv_bfloat16 silu_mul_bf16(float gate_acc, float up_acc) {
    const float gb = bf16_round(gate_acc), ub = bf16_round(up_acc);
    const float act = bf16_round(gb / (1.0f + expf(-gb)));
    return __float2bfloat16_rn(act * ub);
}

// HF residual add: bf16(proj) + bf16(residual) computed in fp32, rounded once.
__device__ __forceinline__ __nv_bfloat162 residual_bf16x2(float c0, float c1, __nv_bfloat162 r) {
    const float y0 = bf16_round(c0), y1 = bf16_round(c1);
    const float2 rf = __bfloat1622float2(r);
    return __floats2bfloat162_rn(rf.x + y0, rf.y + y1);
}

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// Weight-stream load: constant for the engine's lifetime, read once per step and never reused from L1 (each
// 16 B belongs to one lane), so: non-coherent path, no L1 allocation.
__device__ __forceinline__ uint4 ld_weights(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// Activation / residual loads: written by the previous kernel, which may still be running under PDL until
// griddepcontrol.wait returns, so they must take the coherent path (never ld.global.nc).  Plain C++ loads
// through pointers that are NOT __restrict__ (the output may alias them) keep the compiler from using the
// read-only path, and the "memory" clobber of griddepcontrol.wait keeps them after the wait.
__device__ __forceinline__ uint4 ld_act(const uint4* p) { return *p; }
__device__ __forceinline__ __nv_bfloat162 ld_act_bf16x2(const bf16_t* p) {
    return *reinterpret_cast<const __nv_bfloat162*>(p);
}

// cp.async (16 B, L2 only) with zero fill when src_bytes == 0.
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int src_bytes) {
    const unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sa), "l"(gmem), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

__device__ __forceinline__ uint4 lds128(const void* p) { return *reinterpret_cast<const uint4*>(p); }

// pack_exact word (column o at bits 2o) <-> sel word (k-offset 4q+r at bit 2q+8r): a 4x4 transpose of 2-bit fields.
__host__ __device__ __forceinline__ uint32_t nat_to_sel(uint32_t x) {
    uint32_t out = 0;
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int r = 0; r < 4; ++r) out |= ((x >> (2 * (4 * q + r))) & 3u) << (2 * q + 8 * r);
    return out;
}

__host__ __device__ __forceinline__ uint32_t sel_to_nat(uint32_t x) {
    uint32_t out = 0;
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int r = 0; r < 4; ++r) out |= ((x >> (2 * q + 8 * r)) & 3u) << (2 * (4 * q + r));
    return out;
}

}  // namespace kt2n_tern
