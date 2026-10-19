// lm-proj-wide (11 Oct): the index arithmetic of tern_wide_kernel (tern_proj.cu), host + device, so a CPU model
// can check the same expressions without a GPU.
#pragma once

#if defined(__CUDACC__)
#define KT2N_WIDE_HD __host__ __device__ __forceinline__
#else
#define KT2N_WIDE_HD inline
#endif

namespace kt2n_tern_wide {

// Shared-memory byte offset of 16-byte unit u (0..31: bf16 k = 8u .. 8u + 7 of a 256-wide chunk) of row r in activation
// buffer b, for a slice of M rows: [buffer][row][512 B], the unit XOR-swizzled with (2 * (u / 8) + (r & 1)) & 7 so that
// the 8 lanes of each 128-bit phase of a fragment read (g = 0..1, t = 0..3, unit 8t + jp) hit 8 different 16-byte bank
// groups (the GEMM kernel's xunit).
KT2N_WIDE_HD int act_off(int b, int M, int r, int u) { return ((b * M + r) * 32 + (u ^ ((2 * (u >> 3) + (r & 1)) & 7))) * 16; }

// Buffer of chunk i (0..CPW-1) of the CTA's partial p (0..PPC-1): one buffer per (partial, chunk), or with ONEBUF one per
// partial, refilled chunk by chunk.
KT2N_WIDE_HD int act_buf(int p, int i, int CPW, bool onebuf) { return onebuf ? p : p * CPW + i; }

// Global chunk (0..K/256-1) of chunk i of the CTA's partial p in cluster CTA `rank`: decode-kernel warp w = rank * PPC + p
// owns chunks [w * CPW, (w + 1) * CPW).
KT2N_WIDE_HD int g_chunk(int rank, int PPC, int p, int i, int CPW) { return (rank * PPC + p) * CPW + i; }

// float4 slot of a warp's partial for (warp column group wn, partial p of its CTA, tile item ti = nt * MT + mt, lane).
KT2N_WIDE_HD int red_idx(int PPC, int TI, int wn, int p, int ti, int lane) { return ((wn * PPC + p) * TI + ti) * 32 + lane; }

}  // namespace kt2n_tern_wide
