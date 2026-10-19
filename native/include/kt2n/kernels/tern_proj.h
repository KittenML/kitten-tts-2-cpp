/* kt2n k1-ternary: exact W2A16 ternary projections (decode M = 1..16, prefill M up to any size).
 *
 * Every projection of the model is exactly {-s, 0, +s} per 128-weight group, so a weight matrix is held as
 * 2-bit codes + one bf16 scale per (row, 128-group) and decoded in registers to the SAME bf16 values the dense
 * model has.  Activations stay bf16, accumulation is fp32 on tensor cores (mma.sync m16n8k16), output bf16.
 *
 * Default-build batch invariance: fixed family arithmetic (K order, partial accumulators and summation
 * order) gives the same per-row result for every M, launch shape, PDL setting and row position.
 * The experimental KT2N_TERN_PRODUCTION_ORDER build instead reproduces production's distinct decode
 * arithmetic classes: M=1 scalar and M=2..16 tensor-core. Those classes are not bit-identical across M;
 * the startup invariance check remains required and is not waived. Calls marked KT2N_TERN_FLAG_PREFILL
 * always retain the default fixed family arithmetic, including short alone and burst prefill groups.
 *
 * Canonical source format (what the converter / pack_exact produce):
 *   codes  int32 [rows, K/16]: column 16j+o at bits 2o..2o+1 of word j (LSB first); 0 -> 0, 1 -> +s, 2 -> -s
 *   scales bf16  [rows, K/128]
 * Kernel layout (built on the GPU by kt2n_tern_repack; opaque to callers, sizes below):
 *   codes  N*K/4 bytes, tile-major: [N/8 tiles][K/256 chunks][32 lanes][4 words]; lane (g = lane/4, t = lane%4)
 *          of tile T, chunk c holds row 8T+g, k = 256c + 64t + [0, 64), each word = 16 consecutive k with the
 *          code of k-offset 4q+r at bit 2q+8r ("sel" layout).
 *   scales N*K/128*2 bytes, [N/8 tiles][K/256 chunks][8 rows][2 groups] bf16.
 *
 * All pointers are device pointers.  Every function is asynchronous on `stream` and returns 0 on success,
 * KT2N_TERN_EINVAL for bad arguments (nothing launched), KT2N_TERN_ECUDA when the launch failed.
 * No function allocates memory or synchronizes.
 */
#ifndef KT2N_KERNELS_TERN_PROJ_H
#define KT2N_KERNELS_TERN_PROJ_H

#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#include "kt2n/kernels/tern_policy.h"

#ifdef __cplusplus
extern "C" {
#endif

/* families (fix the K split = the rounding; never change for a deployed engine) */
#define KT2N_TERN_FAM_QKV 0     /* N = 4096 (q 0..2047, k 2048..3071, v 3072..4095), K = 2048 */
#define KT2N_TERN_FAM_O 1       /* N = 2048, K = 2048 */
#define KT2N_TERN_FAM_GATE_UP 2 /* N = 12288 rows interleaved (gate_i -> row 2i, up_i -> row 2i+1), K = 2048 */
#define KT2N_TERN_FAM_DOWN 3    /* N = 2048, K = 6144 */

/* epilogues */
#define KT2N_TERN_EPI_STORE 0    /* Y[m, n] = bf16(acc) */
#define KT2N_TERN_EPI_SILU_MUL 1 /* gate_up only: Y[m, i] = bf16(bf16(silu(bf16(acc_gate_i))) * bf16(acc_up_i)), N/2 cols */
#define KT2N_TERN_EPI_RESIDUAL 2 /* Y[m, n] = bf16(float(R[m, n]) + float(bf16(acc))); Y may alias R (in place) */

/* flags */
/* KT2N_TERN_FLAG_PREFILL: always retain the fixed family arithmetic for native
 * prefill, even when the experimental decode production-order build is enabled.
 * Short alone/burst prefill calls must never select different decode classes. */
#define KT2N_TERN_FLAG_PDL 1u /* launch with programmatic dependent launch (sm_90+): weights/scales are loaded
                                 before griddepcontrol.wait; X / R are read and Y written only after it.  The
                                 kernels execute griddepcontrol.launch_dependents right after their weight loads
                                 (build flag KT2N_TERN_EARLY_TRIGGER=0: after the main loop; the large-M GEMM
                                 kernel always after its main loop), so a following kernel launched with PDL must
                                 execute griddepcontrol.wait before it touches anything this projection reads or
                                 writes (the usual PDL contract).  Never changes results. */

#define KT2N_TERN_OK 0
#define KT2N_TERN_EINVAL 1
#define KT2N_TERN_ECUDA 2

/* Bytes of the kernel-layout buffers of an [n, k] matrix (n % 8 == 0, k % 256 == 0). */
size_t kt2n_tern_codes_bytes(int n, int k);
size_t kt2n_tern_scales_bytes(int n, int k);

/* Number of K-split partial sums of a family (WK) and K chunks per split; fixed per family. */
int kt2n_tern_family_split(int family, int *warps_k, int *chunks_per_split);
/* Independent accumulators per partial sum (MMA step s of a partial -> accumulator s % ACC); fixed per family. */
int kt2n_tern_family_acc(int family);

/* Repack one canonical matrix (rows x k) into rows  row_offset + i*row_step  (i = 0..rows-1) of a kernel-layout
 * matrix with n_total rows.  qkv: q (off 0, step 1), k (off 2048), v (off 3072); gate_up: gate (off 0, step 2),
 * up (off 1, step 2).  Destination rows not written keep their old bytes.  src_scales: bf16 bits. */
int kt2n_tern_repack(const int32_t *src_codes, const void *src_scales, int rows, int k, void *dst_codes,
                     void *dst_scales, int n_total, int row_offset, int row_step, cudaStream_t stream);

/* Exact inverse of kt2n_tern_repack for the same (rows, row_offset, row_step): writes canonical codes/scales. */
int kt2n_tern_unrepack(const void *codes, const void *scales, int n_total, int k, int row_offset, int row_step,
                       int rows, int32_t *dst_codes, void *dst_scales, cudaStream_t stream);

/* Decode a kernel-layout matrix to dense bf16 [n, k] (row stride k) through the kernels' own register decode
 * (PRMT tables): proves the bf16 weights the MMA sees equal pack_exact's unpack bit for bit. */
int kt2n_tern_decode_dense(const void *codes, const void *scales, int n, int k, void *dense_out,
                           cudaStream_t stream);

/* Convenience: whole families from canonical per-matrix arrays (q/k/v and gate/up). */
int kt2n_tern_repack_qkv(const int32_t *q_codes, const void *q_scales, const int32_t *k_codes,
                         const void *k_scales, const int32_t *v_codes, const void *v_scales, int nq, int nkv,
                         int k, void *dst_codes, void *dst_scales, cudaStream_t stream);
int kt2n_tern_repack_gate_up(const int32_t *gate_codes, const void *gate_scales, const int32_t *up_codes,
                             const void *up_scales, int inter, int k, void *dst_codes, void *dst_scales,
                             cudaStream_t stream);

/* Y = X W^T with the family's fixed arithmetic.
 *   x        bf16 [m, k], row stride ldx elements (x 16-byte aligned, ldx % 8 == 0)
 *   codes/scales  kernel layout of an [n, k] matrix
 *   y        bf16 [m, n] (EPI_SILU_MUL: [m, n/2]), row stride ldy (y 4-byte aligned, ldy % 2 == 0)
 *   residual bf16 [m, n] row stride ldr (EPI_RESIDUAL only; may equal y with ldr == ldy; else NULL)
 *   m >= 1 (m <= 16: decode kernel; m > 16: prefill kernel, 16-row tiles, the same per-row arithmetic)
 * Rows >= m of x are never read and rows >= m of y never written. */
int kt2n_tern_proj(int family, const void *x, int64_t ldx, int m, const void *codes, const void *scales, int n,
                   int k, void *y, int64_t ldy, int epi, const void *residual, int64_t ldr, unsigned flags,
                   cudaStream_t stream);

/* Same, with explicit launch shape (0 = auto; for tuning and tests).  m <= 16: nt = 8-column output tiles per
 * warp, wn = tile groups per block.  m > 16: mt > 0 = register-tiled kernel (nt, wn as above, mt 16-row tiles per
 * warp); mt < 0 = shared-memory GEMM kernel with nt = column tiles per warp, wn = warps along N,
 * -mt = (warps along M) + 100 * (16-row tiles per warp - 1) (needs 16-byte aligned scales).  No shape and no kernel
 * choice ever changes a result bit. */
int kt2n_tern_proj_ex(int family, const void *x, int64_t ldx, int m, const void *codes, const void *scales,
                      int n, int k, void *y, int64_t ldy, int epi, const void *residual, int64_t ldr,
                      unsigned flags, int nt, int wn, int mt, cudaStream_t stream);

/* L3 (10 Oct): Y = (RMSNorm(X) * ln_w) W^T for decode rows (1 <= m <= 16), the norm computed inside the projection:
 * X is the residual stream (bf16 [m, k], row stride ldx), ln_w the bf16 [k] norm weight (16-byte aligned).  The MMA
 * reads bf16(x * ln_w); the per-row 1/sqrt(mean(x^2) + eps) (fp32) scales the fp32 sums before the epilogue.
 * Families QKV (EPI_STORE) and GATE_UP (EPI_SILU_MUL) only.  Statistically equal to kt2n_ops_rmsnorm +
 * kt2n_tern_proj (different rounding points), not bitwise.  Same PDL contract as kt2n_tern_proj. */
int kt2n_tern_proj_norm(int family, const void *x, int64_t ldx, int m, const void *ln_w, float eps,
                        const void *codes, const void *scales, int n, int k, void *y, int64_t ldy, int epi,
                        unsigned flags, cudaStream_t stream);

/* lm-proj-wide (11 Oct, default off; the decode step uses it only with env KT2N_LM_PROJ_WIDE=1): the same
 * projection for 1 <= m <= 32 rows (meant for 17..32) by a cluster split-K kernel that stages each CTA's K slice of
 * the activations in shared memory once.  Same per-row arithmetic as kt2n_tern_proj, bit for bit (tested memcmp).
 * Only the decode step's epilogue per family: qkv / o / down EPI_STORE, gate_up EPI_SILU_MUL.  shape: 0 = the
 * family's default, 1..3 = launch shapes (kt2n_tern_proj_wide_shape).  Returns KT2N_TERN_EINVAL, with nothing
 * launched, when this path is not available for the arguments or the device (no cluster launch, other epilogue,
 * m > 32, ...): the caller then uses kt2n_tern_proj.  Same PDL contract as kt2n_tern_proj. */
int kt2n_tern_proj_wide(int family, const void *x, int64_t ldx, int m, const void *codes, const void *scales,
                        int n, int k, void *y, int64_t ldy, int epi, const void *residual, int64_t ldr,
                        unsigned flags, int shape, cudaStream_t stream);
/* The launch shape of kt2n_tern_proj_wide's `shape`: CTAs per cluster (K split across them), column tiles per warp,
 * warps along N per CTA. */
int kt2n_tern_proj_wide_shape(int family, int shape, int *cluster, int *nt, int *wn);

/* Test hook: out[i] = the SILU_MUL epilogue's arithmetic on bf16 inputs gate[i], up[i] (count elements). */
int kt2n_tern_test_silu_mul(const void *gate, const void *up, void *out, int count, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* KT2N_KERNELS_TERN_PROJ_H */
