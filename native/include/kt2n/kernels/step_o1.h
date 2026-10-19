/* kt2n o1-opt: an optimised variant of the k5 decode step (kt2n_step_forward), same contract, same bits.
 *
 * kt2n_step_forward_o1(model, io, stream) takes exactly kt2n_step_forward's arguments (include/kt2n/kernels/step.h)
 * and enqueues the same step with launch-level changes that never touch a row's arithmetic:
 *
 *   ATTN      decode attention through this file's copy of k3's SlotKV decode kernel (same per-element
 *             operations in the same order, same split size and fold order: bitwise equal), with:
 *     ATTN_TRIGGER  griddepcontrol.launch_dependents at kernel start (k3's kernel never triggers, so the o
 *                   projection could not start loading its weights until attention had fully exited);
 *     ATTN_GRID     grid ordered (kv head, row, split) so the CTAs that have keys to read are dispatched
 *                   first and the empty splits (the grid covers max_ctx) last;
 *     ATTN_KVPRE    the first 64-key K/V tile of a split is fetched before griddepcontrol.wait when it cannot
 *                   contain this step's key (the only column the predecessor writes);
 *     L2PF          every attention CTA issues an L2 bulk prefetch of its slice of the weights the rest of
 *                   the layer needs next (o, gate_up, down, the next layer's qkv; the head at the last layer),
 *                   so the projections read them from L2 while DRAM would otherwise idle;
 *   FUSE_QK   k2's q/k RMSNorm + RoPE + K/V append done inside the attention kernel (k2's arithmetic, one
 *             launch and one dependency edge less per layer); the step's key/value row goes into the attention
 *             tile from registers.  Needs 0 <= *cursor < max_ctx (production's SlotKV) for the append.
 *   ATTN_DEEP (with FUSE_QK) all K/V tiles of a split are requested before griddepcontrol.wait (three tiles of
 *             shared memory instead of two; the step's own row comes from registers), so the attention's DRAM
 *             latency overlaps the qkv projection.
 *   HEAD      the audio head through this file's copy of k4's head kernel launched with PDL (k4's kernel has
 *             no griddepcontrol.wait, so k5 launches it without PDL): weights prefetched to L2 before the wait,
 *             same arithmetic.
 *
 *   NORM_EARLY1/2 (prof-step) the residual + RMSNorm launches trigger their dependents at kernel start
 *             (kt2n_ops_residual_rmsnorm_early), so the following projection launches and streams its weights while
 *             the projection before the norm still runs.
 *
 * Options are a process-wide bitmask (kt2n_step_o1_set_options), read when the step is enqueued, i.e. baked
 * into a CUDA graph at capture.  By default (KT2N_STEP_O1_AUTO) each bucket uses the set measured best for it:
 * row 1 0x485af, row 2 0x685af, row 3 0x4848f, rows 4-6 0x548f, rows 7-10 0x6008f, rows 11-16 0x548f.
 */
#ifndef KT2N_KERNELS_STEP_O1_H
#define KT2N_KERNELS_STEP_O1_H

#include <cuda_runtime_api.h>

#include "kt2n/kernels/step.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KT2N_STEP_O1_ATTN 0x1u
#define KT2N_STEP_O1_ATTN_TRIGGER 0x2u
#define KT2N_STEP_O1_ATTN_GRID 0x4u
#define KT2N_STEP_O1_ATTN_KVPRE 0x8u
#define KT2N_STEP_O1_L2PF 0x10u
#define KT2N_STEP_O1_HEAD 0x20u
#define KT2N_STEP_O1_L2PF_HEAD 0x40u /* with ATTN: prefetch the head to L2 during the last layer's attention */
#define KT2N_STEP_O1_FUSE_QK 0x80u   /* with ATTN: q/k norm + RoPE + K/V append inside the attention kernel */

#define KT2N_STEP_O1_HEAD_PF 0x100u  /* with HEAD: the head CTAs L2-prefetch their weight rows before the wait */
/* L2PF regions (none set = all): this layer's o, gate_up, down, the next layer's qkv */
#define KT2N_STEP_O1_PF_O 0x200u
#define KT2N_STEP_O1_PF_GU 0x400u
#define KT2N_STEP_O1_PF_DOWN 0x800u
#define KT2N_STEP_O1_PF_QKV 0x1000u
#define KT2N_STEP_O1_PF_MASK 0x1e00u
#define KT2N_STEP_O1_ATTN_DEEP 0x2000u /* with FUSE_QK: every K/V tile requested before griddepcontrol.wait */

#define KT2N_STEP_O1_L2PF_LATE 0x4000u /* like L2PF, issued by the split-0 CTAs after their own K/V reads */

#define KT2N_STEP_O1_L2PF_W 0x8000u    /* like L2PF, shared by row 0's working CTAs right after the wait */

#define KT2N_STEP_O1_ATTN_DEEP_POST 0x10000u /* with ATTN_DEEP: request the tiles after the wait, not before */

/* prof-step: residual + RMSNorm launches that trigger their dependents at kernel start
 * (kt2n_ops_residual_rmsnorm_early, include/kt2n/kernels/ops_norm_early.h), so the next projection launches and
 * streams its weights while the o / down projection before the norm is still running.
 *   NORM_EARLY1: the norm after the o projection (before gate_up);
 *   NORM_EARLY2: the norm after the down projection (before the next layer's qkv / the head). */
#define KT2N_STEP_O1_NORM_EARLY1 0x20000u
#define KT2N_STEP_O1_NORM_EARLY2 0x40000u

/* L3 (10 Oct, bit identity no longer required): rows with <= 640 keys use decode attention with one 64-key chunk
 * per split instead of KT2N_ATTN_SPLIT_CHUNKS (twice the CTAs at short contexts, no serial second K/V tile); longer
 * rows keep the default split.  Changes the softmax fold order (statistically equal, not bitwise k3).  Needs an
 * attention workspace sized for kt2n_step_attn_ws_bytes(rows, 2 * max_ctx).  Ignored with ATTN_DEEP. */
#define KT2N_STEP_O1_SPLIT64 0x100000u

/* L3 (10 Oct), EXPERIMENTAL (measured 15-28% slower at 1-4 rows): residual + RMSNorm fused
 * into the projections, rows <= 16: the o and down projections add their
 * output to the residual stream in their epilogue (EPI_RESIDUAL, in place) and the qkv (layers >= 1) and gate_up
 * projections normalise it in their prologue (kt2n_tern_proj_norm); the 56 norm launches per step become one final
 * norm before the head.  Statistically equal, not bitwise. */
#define KT2N_STEP_O1_FUSE_NORM 0x200000u

/* L3 sprint 5: with FUSE_QK (per-row K/V), each attention CTA of layer l, once its own K/V tiles have landed, issues
 * L2 bulk prefetches (cp.async.bulk.prefetch.L2) for the same row / head / key range of layer l + 1 (voice-store and
 * slot parts), so that read overlaps the projections between the two attentions, which leave DRAM mostly idle.
 * Never changes a value (L2 is the GPU's coherence point): logits stay bit-identical.  Keys beyond a per-step L2
 * budget (kt2n_step_o1_set_kv_next_budget) are not prefetched, so the prefetched K/V cannot thrash the L2. */
#define KT2N_STEP_O1_KV_NEXT 0x400000u

#define KT2N_STEP_O1_ALL 0x7ffffu
/* g_options value meaning "per-bucket table" (the default) */
#define KT2N_STEP_O1_AUTO 0x80000000u

/* Set / read the option bitmask (process-wide; affects steps enqueued afterwards, i.e. graphs captured later).
 * KT2N_STEP_O1_AUTO (default): each bucket uses its measured set (kt2n_step_o1_options_for); any other value is
 * used for every bucket (0 = exactly k5's launches). */
void kt2n_step_o1_set_options(unsigned options);
unsigned kt2n_step_o1_options(void);
/* Override one bucket's entry of the per-bucket table (rows 1..KT2N_STEP_MAX_ROWS_EXT); returns 0 or KT2N_STEP_EINVAL. */
int kt2n_step_o1_set_bucket_options(int rows, unsigned options);
/* The options a step of `rows` rows is enqueued with. */
unsigned kt2n_step_o1_options_for(int rows);
/* Decode launch shape {nt, wn} (kt2n_tern_proj_ex) of one projection family (KT2N_TERN_FAM_*) for steps of `rows`
 * rows; 0, 0 = k1's own per-M table (the default).  Shapes never change results, only speed. */
int kt2n_step_o1_set_shape(int rows, int family, int nt, int wn);
/* L3: the same with the M-tiling of k1's >16-row kernels: mt > 0 = register-tiled kernel with mt % 10 16-row tiles per
 * warp and a register cap of mt / 10 blocks per SM, mt < 0 = shared-memory GEMM kernel code (kt2n_tern_proj_ex),
 * 0 = k1's own choice.  Never changes results, except the EXPERIMENTAL mt = MT + 100 (one accumulator chain:
 * statistically equal, not bit-identical; lane log).  L3 sprint 4: mt = 1000 + 10 * MINB + acc2 = the swapped-operand
 * kernel (weights as the MMA's A operand, all rows in one CTA, nt = 16-column pairs per warp; acc2 = 1: 2 accumulator
 * chains, statistically equal), 17..32 rows; EXPERIMENT, built only with KT2N_TERN_EXPERIMENTS, measured slower. */
int kt2n_step_o1_set_shape_mt(int rows, int family, int nt, int wn, int mt);
/* L3 sprint 5: the L2 budget of KT2N_STEP_O1_KV_NEXT (bytes of one layer's prefetched K + V over all rows; default
 * 48 MiB; affects graphs captured later).  A row's keys at or beyond budget / (rows * 4 KiB) are not prefetched. */
void kt2n_step_o1_set_kv_next_budget(unsigned long long bytes);

/* The decode step (same arguments, return codes and results as kt2n_step_forward). */
int kt2n_step_forward_o1(const kt2n_step_model *model, const kt2n_step_io *io, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* KT2N_KERNELS_STEP_O1_H */
