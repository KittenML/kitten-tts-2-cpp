// kt2n attention kernels (lane k3-attn): decode and prefill attention over the K/V cache.
//
// Geometry is fixed to the Kitten TTS 2 LM: 16 query heads, 8 K/V heads (GQA 2:1),
// head_dim 128, bf16 activations and cache, fp32 math.
//
// NUMERICS (one canonical definition shared by decode and prefill; numerics version
// SPLIT_CHUNKS*1000 + 2)
//   For a query at logical position p the keys are logical positions 0..p (n = p + 1 keys).
//   Keys are cut into 64-key CHUNKS counted from the row's own key 0, and chunks into
//   SPLITS of KT2N_ATTN_SPLIT_CHUNKS chunks (KT2N_ATTN_SPLIT_KEYS keys), also from key 0.
//   All products run on tensor cores (mma.sync.m16n8k16, bf16 inputs, fp32 accumulation);
//   everything else is IEEE fp32 with explicit round-to-nearest (no fast math, no FTZ).
//   Per chunk c of a split (j = key within the chunk, valid if j < nv):
//     s_j = (MMA chain over the 8 16-dim steps of q.k_j, from 0) * (1/sqrt(128))   (fp32)
//     mc  = max_j s_j
//     first chunk of the split: m = mc; O = 0; l = 0
//     else: mn = max(m, mc); a = expf(m - mn); O = O*a; l = l*a; m = mn
//     p_j = expf(s_j - m)  (precise libdevice expf; 0 for invalid keys)
//     p_j is split exactly into three bf16 parts p = hi + mid + lo (so P is exact fp32), and
//     for each 16-key step in order, parts hi, mid, lo: O += P_part . V ; l += P_part . 1
//     (MMA chains, fp32 accumulation)
//   Splits are folded in order: F = S_0; F <- (max m; a = expf(F.m - mn), e = expf(S.m - mn);
//   F.l = F.l*a + S.l*e; F.o = F.o*a + S.o*e)   (separate fp32 mul/mul/add)
//   out[d] = bf16_rn(F.o[d] / F.l)
//   The result for a row depends only on that row's q and its own keys: bitwise independent
//   of the batch size, of the other rows, of where the row's keys sit in the cache
//   (cursor / column offset / slot), of the launch shape (split-parallel or sequential) and
//   run to run.  Prefill gives every query exactly the bits decode gives at that position (the
//   query sits in the same MMA tile rows: row g = head 0, row g + 8 = head 1).  Bits are the
//   same on every GPU of one architecture; tensor-core accumulation may differ across
//   architectures (sm_90 / sm_100 / sm_120), so do not compare bits across them.

// LAYOUTS
//   q    : bf16, row r at q + r*q_row_stride, heads contiguous: [16][128].
//   out  : bf16, row r at out + r*out_row_stride, [16][128] (= the o_proj input row).
//   k/v  : bf16, element (slot, kv_head, column, d) at
//          base + slot*kv_slot_stride + kv_head*kv_head_stride + column*128 + d.
//          Production SlotKV per layer is [slots][8][ctx][128]: kv_slot_stride = 8*ctx*128,
//          kv_head_stride = ctx*128 (the flat [2, L, slots, 8, ctx, 128] store: k = store[0, l],
//          v = store[1, l]).
//   All pointers 16-byte aligned; all strides multiples of 8 elements.
//
// Return value of every launcher: 0 on success, a cudaError_t value from the launch, or
// KT2N_ATTN_EINVAL for a bad argument (nothing launched).
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KT2N_ATTN_HEADS_Q 16
#define KT2N_ATTN_HEADS_KV 8
#define KT2N_ATTN_HEAD_DIM 128
#define KT2N_ATTN_CHUNK 64
#ifndef KT2N_ATTN_SPLIT_CHUNKS
#define KT2N_ATTN_SPLIT_CHUNKS 2
#endif
#define KT2N_ATTN_SPLIT_KEYS (KT2N_ATTN_CHUNK * KT2N_ATTN_SPLIT_CHUNKS)
#define KT2N_ATTN_MAX_ROWS 64   // decode rows per launch (engine uses <= 16)
#define KT2N_ATTN_MAX_CTX 32768 // keys per row
#define KT2N_ATTN_EINVAL 1000

// Launch flags (bitwise OR).
#define KT2N_ATTN_FLAG_PDL 0x1u                // launch with programmatic stream serialization
                                               // (the kernels wait on griddepcontrol before reading)
#define KT2N_ATTN_FLAG_PREFILL_SEQUENTIAL 0x2u // prefill: force the workspace-free shape
#define KT2N_ATTN_FLAG_PREFILL_SPLIT 0x4u      // prefill: force the split-parallel shape

// Numerics/version tag: bumps whenever the canonical arithmetic above changes.
// Returns KT2N_ATTN_SPLIT_CHUNKS * 1000 + revision.
int kt2n_attn_numerics_version(void);

// ---------------------------------------------------------------------------
// Workspace.  Arrival counters + split partials.  Zero it ONCE with kt2n_attn_workspace_init
// (memset of the whole buffer); every launch leaves its counters zero again, so it is
// CUDA-graph safe.  One workspace (sized max of the two functions below) serves any number of
// stream-ordered decode and prefill launches (every layer of a step); launches that may run
// concurrently (different streams) need their own.
size_t kt2n_attn_decode_workspace_bytes(int max_rows, int max_ctx);
size_t kt2n_attn_prefill_workspace_bytes(int max_rows, int max_q_len, int max_keys);
int kt2n_attn_workspace_init(void* workspace, size_t workspace_bytes, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Decode over production's SlotKV layout: one query token per row, rows 0..rows-1 use slots
// 0..rows-1.  Exactly production's key range (invariant_decode._attn_split):
//   pos = min(positions[r], *cursor); keys = columns [*cursor - pos, *cursor] (pos + 1 keys,
//   the current token's K/V already written at column *cursor).
// positions: device int64 [rows] (the bucket's "positions" buffer, stride 1);
// cursor: device int64 scalar.  Both are read on the device (CUDA-graph friendly).
// max_ctx: the slot buffer's column count (bounds the grid: ceil(max_ctx / SPLIT_KEYS)).
int kt2n_attn_decode_slotkv(const void* q, int64_t q_row_stride,
                            const void* k_cache, const void* v_cache,
                            int64_t kv_slot_stride, int64_t kv_head_stride,
                            const int64_t* positions, const int64_t* cursor,
                            int rows, int max_ctx,
                            void* out, int64_t out_row_stride,
                            void* workspace, size_t workspace_bytes,
                            unsigned flags, cudaStream_t stream);

// Decode over a plain per-row layout: row r's keys are columns
// [kv_start[r], kv_start[r] + kv_len[r]) of slot row_slot[r] (logical key j = column
// kv_start[r] + j).  row_slot == NULL -> slot r; kv_start == NULL -> 0.  kv_len >= 1
// (a row with kv_len <= 0 gets zeros; kv_len > max_kv_len gets NaNs: a contract violation).
// All arrays are device int32 [rows].  max_kv_len bounds the grid.
int kt2n_attn_decode_rows(const void* q, int64_t q_row_stride,
                          const void* k_cache, const void* v_cache,
                          int64_t kv_slot_stride, int64_t kv_head_stride,
                          const int32_t* row_slot, const int32_t* kv_start,
                          const int32_t* kv_len,
                          int rows, int max_kv_len,
                          void* out, int64_t out_row_stride,
                          void* workspace, size_t workspace_bytes,
                          unsigned flags, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Prefill: causal attention for each row's new queries over that row's keys.
// Row descriptor (device array, one per row; 32 bytes):
//   q_start : index of the row's first query in q / out (queries q_start .. q_start+q_len-1)
//   q_len   : number of queries (0 allowed)
//   q_pos0  : logical position of the first query (= number of keys before it, e.g. the
//             prefix length); query i attends logical keys 0 .. q_pos0 + i
//   kv_slot : slot holding the row's keys
//   kv_col0 : column of logical key 0 in that slot (keys 0 .. q_pos0+q_len-1 must be present,
//             i.e. the suffix's own K/V is written before this call, as decode does)
typedef struct kt2n_attn_prefill_row {
  int32_t q_start;
  int32_t q_len;
  int32_t q_pos0;
  int32_t kv_slot;
  int32_t kv_col0;
  int32_t reserved[3];
} kt2n_attn_prefill_row;

// max_q_len >= every row's q_len and max_keys >= every row's q_pos0 + q_len (they bound the
// grid; a row beyond max_keys gets NaNs in the split-parallel shape).
// workspace may be NULL (then the sequential shape is used).  Default (flags 0): split-parallel
// when the workspace is big enough and the sequential grid (ceil(max_q_len/32) x 8 x num_rows
// CTAs) is below 3/4 of the SM count, sequential otherwise.  Both shapes give identical bits.
int kt2n_attn_prefill(const void* q, int64_t q_row_stride,
                      const void* k_cache, const void* v_cache,
                      int64_t kv_slot_stride, int64_t kv_head_stride,
                      const kt2n_attn_prefill_row* rows_desc, int num_rows, int max_q_len,
                      int max_keys,
                      void* out, int64_t out_row_stride,
                      void* workspace, size_t workspace_bytes,
                      unsigned flags, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif
