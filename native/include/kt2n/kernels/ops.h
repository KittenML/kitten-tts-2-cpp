// kt2n small per-layer operators ("ops" lane): embedding + speaker injection,
// RMSNorm (plain and fused residual-add), q/k RMSNorm + RoPE (+ K/V append),
// K/V append, speaker projection (Linear + LayerNorm).
//
// Numerics contract:
//   * Every op computes each row with an order that depends only on that row:
//     a row's output bits do not depend on the number of rows (M) or on the
//     other rows' contents, and runs are deterministic (no atomics).
//   * RMSNorm reproduces production's Triton kernels (kittenml-api
//     fast_lm/invariant_decode.py `_rms_rows` / `_residual_rms`, num_warps=4)
//     including their fp32 reduction order, so it is bitwise equal to
//     production's La decode / fused prefill norms. HF eager Qwen3RMSNorm uses
//     torch's reduction, whose order depends on M, so against HF eager the
//     result is equal except for rare 1-ulp bf16 differences (measured).
//   * q/k norm + RoPE reproduces production's `_qk_norm_rope_kv` (one warp per
//     head) with cos/sin computed in-kernel exactly like HF
//     Qwen3RotaryEmbedding (fp32 inv_freq * fp32(pos), cosf/sinf, -> bf16).
//   * LayerNorm reproduces torch 2.7.1's vectorized CUDA layer_norm (Welford).
//
// Conventions:
//   * bf16 = __nv_bfloat16 (same bits as torch.bfloat16).
//   * Strides are in ELEMENTS, never bytes.
//   * Token ids, positions, cursors and slot/column indices are int64 device
//     arrays (torch.long), so production's buffers can be passed directly.
//   * `pdl` = 1 launches with programmatic stream serialization (PDL,
//     sm_90+): the kernel loads constant weights, then griddepcontrol.wait
//     before touching anything a predecessor wrote, then triggers its
//     dependents. pdl = 0 is an ordinary launch. Bits are identical.
//   * Every launcher returns cudaSuccess, cudaErrorInvalidValue for bad
//     arguments (nothing launched), or the launch error.
//   * rows == 0 is a no-op returning cudaSuccess.
//   * All functions are safe to call during CUDA graph capture.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KT2N_OPS_API_VERSION 1

// ---------------------------------------------------------------------------
// RMSNorm (Qwen3RMSNorm): y = bf16( bf16(x * rsqrt(mean(x^2) + eps)) * w )
//   n in {1024, 2048, 4096, 8192} (2048 for the 1.7B model).
//   x and out: 16-byte aligned rows (stride % 8 == 0). out may equal x.
//   One CTA of 128 threads per row.
// ---------------------------------------------------------------------------
cudaError_t kt2n_ops_rmsnorm(const __nv_bfloat16* x, int64_t x_stride,
                             const __nv_bfloat16* w, float eps,
                             __nv_bfloat16* out, int64_t out_stride,
                             int rows, int n, int pdl, cudaStream_t stream);

// Fused residual add + RMSNorm (production `_residual_rms`):
//   h   = bf16(residual + delta)            (fp32 add, one rounding)
//   out = RMSNorm(h) * w                    (as kt2n_ops_rmsnorm)
//   h_out may equal residual (in place, as production does) or be NULL
//   (h not stored, e.g. the last layer when only the final norm is needed).
//   out must not alias residual/delta unless it is the same row buffer used
//   in place by the same row (each element is read before it is written).
cudaError_t kt2n_ops_residual_rmsnorm(const __nv_bfloat16* residual, int64_t residual_stride,
                                      const __nv_bfloat16* delta, int64_t delta_stride,
                                      const __nv_bfloat16* w, float eps,
                                      __nv_bfloat16* h_out, int64_t h_out_stride,
                                      __nv_bfloat16* out, int64_t out_stride,
                                      int rows, int n, int pdl, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Embedding gather with optional speaker injection.
//   out[r] = table[ids[r]]    (ids out of [0, vocab) -> zero row)
// Speaker injection (runtime.py inject_speaker / prefix_kv_prefill._embed):
// at prefill the embedding of each sequence's FIRST token (logical position
// 0) is replaced by the projected speaker vector (spk_proj output, bf16,
// `hidden` wide). Rules:
//   speakers == NULL               -> no injection;
//   speaker_index != NULL (device int32[rows]) -> row r takes
//       speakers[speaker_index[r] * speaker_stride] when speaker_index[r] >= 0;
//   else speaker_row in [0, rows)  -> that row takes speakers[0..hidden).
// Decode never injects (production only injects when cache_position[0]==0).
//   hidden % 8 == 0; table/out/speakers 16-byte aligned with strides % 8 == 0.
//   Table row stride == hidden.
// ---------------------------------------------------------------------------
cudaError_t kt2n_ops_embed(const int64_t* ids, int rows,
                           const __nv_bfloat16* table, int64_t vocab, int hidden,
                           const __nv_bfloat16* speakers, int64_t speaker_stride,
                           const int32_t* speaker_index, int speaker_row,
                           __nv_bfloat16* out, int64_t out_stride,
                           int pdl, cudaStream_t stream);

// Embedding + the first layer's input RMSNorm in one launch (decode step
// start / prefill start). hidden_out gets the embedding (the residual
// stream), normed_out = RMSNorm(hidden_out) * w. Same bits as
// kt2n_ops_embed followed by kt2n_ops_rmsnorm. hidden in {1024, 2048, 4096, 8192}.
// Optionally (cos_out/sin_out non-NULL, positions required) it also writes
// this step's RoPE table for each row: cos_out[r*cs_stride + i] = bf16(cos(f_i))
// for i < 64 (and i + 64 too when cs_stride >= 128, HF's layout), same for
// sin -- exactly kt2n_ops_rope_cos_sin's values. Feeding that table to
// kt2n_ops_qk_norm_rope (cos_tab/sin_tab) gives the same bits as computing
// in-kernel and takes the transcendentals off the 28 per-layer launches.
cudaError_t kt2n_ops_embed_rmsnorm(const int64_t* ids, int rows,
                                   const __nv_bfloat16* table, int64_t vocab, int hidden,
                                   const __nv_bfloat16* speakers, int64_t speaker_stride,
                                   const int32_t* speaker_index, int speaker_row,
                                   const __nv_bfloat16* w, float eps,
                                   __nv_bfloat16* hidden_out, int64_t hidden_stride,
                                   __nv_bfloat16* normed_out, int64_t normed_stride,
                                   const int64_t* positions, const float* inv_freq,
                                   __nv_bfloat16* cos_out, __nv_bfloat16* sin_out,
                                   int64_t cs_stride, int pdl, cudaStream_t stream);

// ---------------------------------------------------------------------------
// K/V destination descriptor (host struct, copied at launch).
// Row r (0..rows-1) of a launch writes head h (0..kv_heads-1), element
// d (0..127) of its key/value to
//   k[slot * slot_stride + h * head_stride + col * col_stride + d]
// with
//   slot = row_slot ? row_slot[r] : slot_base + r * slot_step
//   col  = col_base + r * col_step + (cursor ? cursor[0] : 0)
//                                  + (row_col ? row_col[r] : 0)
// A row is skipped when slot < 0, or when num_cols > 0 and col is outside
// [0, num_cols). k or v may be NULL (that half is not written).
// Examples:
//   production SlotKV ([2, L, slots, H, ctx, D], shared device cursor,
//     rows in slots 0..B-1): see kt2n_ops_kv_target_slotkv.
//   one sequence's prefill (T tokens) into [H, T, D] (production's prefill
//     export): slot_stride 0, head_stride T*D, col_stride D, col_step 1.
//   one sequence's prefill into a SlotKV slot S right-aligned at columns
//     [start, start+T): slot_base S, slot_step 0, col_base start, col_step 1.
//   per-row native K/V [slots, ctx, H, D] at each row's own position:
//     slot_stride ctx*H*D, head_stride D, col_stride H*D, row_col = positions.
// ---------------------------------------------------------------------------
typedef struct kt2n_ops_kv_target {
  __nv_bfloat16* k;
  __nv_bfloat16* v;
  int64_t slot_stride;
  int64_t head_stride;
  int64_t col_stride;
  const int64_t* row_slot;  // device [rows] or NULL
  int64_t slot_base;
  int64_t slot_step;
  const int64_t* row_col;   // device [rows] or NULL
  const int64_t* cursor;    // device [1] or NULL
  int64_t col_base;
  int64_t col_step;
  int64_t num_cols;         // > 0: bounds check on col; 0: no check
} kt2n_ops_kv_target;

// Fill `t` for production's SlotKV store (FLAT_ADMIT layout
// [2, layers, slots, kv_heads, ctx, head_dim] bf16, contiguous) for one
// `layer`: K = store[0, layer], V = store[1, layer]; row r -> slot r; every
// row writes column cursor[0] (the shared physical cursor, a device int64).
// num_cols = ctx. Host only; returns cudaErrorInvalidValue on bad sizes.
cudaError_t kt2n_ops_kv_target_slotkv(kt2n_ops_kv_target* t, __nv_bfloat16* store,
                                      int layer, int layers, int slots, int kv_heads,
                                      int ctx, int head_dim, const int64_t* cursor);

// Plain K/V append (k already normed + rotated): k/v rows are
// [kv_heads * 128] at k + r*k_stride, v + r*v_stride. head_dim must be 128.
cudaError_t kt2n_ops_kv_append(const __nv_bfloat16* k, int64_t k_stride,
                               const __nv_bfloat16* v, int64_t v_stride,
                               const kt2n_ops_kv_target* kv,
                               int rows, int kv_heads, int head_dim,
                               int pdl, cudaStream_t stream);

// ---------------------------------------------------------------------------
// q/k RMSNorm over head_dim + RoPE (+ K/V append), production
// `_qk_norm_rope_kv` numerics:
//   n   = bf16( bf16(x * rsqrt(mean(x^2) + eps)) * w )       per head
//   f_i = fp32(inv_freq[i]) * fp32(pos)  (i < 64; dims i and i+64 share f_i)
//   c_i = bf16(cosf(f_i)), s_i = bf16(sinf(f_i))
//   out[i]    = bf16( bf16(n[i]*c_i)    - bf16(n[i+64]*s_i) )
//   out[i+64] = bf16( bf16(n[i+64]*c_i) + bf16(n[i]*s_i)    )
// Inputs: row r's q heads at q + r*q_stride (q_heads*128 elements), k heads
// at k + r*k_stride, v heads at v + r*v_stride (v only read for the K/V
// append; may be NULL when kv == NULL or kv->v == NULL).
// cos/sin, either way the same bits:
//   * cos_tab/sin_tab non-NULL: read from a per-row bf16 table
//     (cos_tab[r*cs_stride + i], i < 64; cs_stride 64 = compact, 128 = HF
//     [rows, 128] layout), e.g. written once per step by
//     kt2n_ops_embed_rmsnorm or kt2n_ops_rope_cos_sin (FASTER: recommended);
//     positions / inv_freq are then unused (may be NULL);
//   * else computed in-kernel from positions (device int64[rows], each row's
//     logical position) and inv_freq (device fp32[64], or NULL for the
//     built-in table = HF's buffer for rope_theta 1e6, head_dim 128, see
//     kt2n_ops_rope_inv_freq_default).
// Outputs: q head h of row r -> q_out + r*q_out_row_stride + h*q_out_head_stride
// (production decode: [rows, q_heads, 128] -> row stride q_heads*128, head
// stride 128; production prefill [heads, T, 128] -> row stride 128, head
// stride T*128). k_out (optional, NULL = skip) likewise for k heads. kv
// (optional, NULL = no append): rotated k and raw v go to the K/V target.
// head_dim must be 128; q/k/v/out pointers 4-byte aligned, strides even.
// Launch: grid (rows, ceil((q_heads+kv_heads)/2)), 64 threads, one warp per head.
// ---------------------------------------------------------------------------
cudaError_t kt2n_ops_qk_norm_rope(const __nv_bfloat16* q, int64_t q_stride,
                                  const __nv_bfloat16* k, int64_t k_stride,
                                  const __nv_bfloat16* v, int64_t v_stride,
                                  const __nv_bfloat16* q_w, const __nv_bfloat16* k_w, float eps,
                                  const int64_t* positions, const float* inv_freq,
                                  const __nv_bfloat16* cos_tab, const __nv_bfloat16* sin_tab,
                                  int64_t cs_stride,
                                  __nv_bfloat16* q_out, int64_t q_out_row_stride,
                                  int64_t q_out_head_stride,
                                  __nv_bfloat16* k_out, int64_t k_out_row_stride,
                                  int64_t k_out_head_stride,
                                  const kt2n_ops_kv_target* kv,
                                  int rows, int q_heads, int kv_heads, int head_dim,
                                  int pdl, cudaStream_t stream);

// HF Qwen3RotaryEmbedding cos/sin tables for the given positions, bf16
// [rows, cs_stride]: entries i < 64 always written; with cs_stride >= 128 the
// HF duplicate half (i + 64) too. Bitwise equal to HF's rotary_emb output
// (tested for every position 0..65535). Input for kt2n_ops_qk_norm_rope.
cudaError_t kt2n_ops_rope_cos_sin(const int64_t* positions, int rows, const float* inv_freq,
                                  __nv_bfloat16* cos_out, __nv_bfloat16* sin_out,
                                  int64_t cs_stride, int pdl, cudaStream_t stream);

// Host: the built-in inv_freq table (64 fp32) = HF's buffer
// 1.0 / (1e6 ** (arange(0, 128, 2).float() / 128)) as torch computes it.
void kt2n_ops_rope_inv_freq_default(float out64[64]);

// ---------------------------------------------------------------------------
// Speaker projection: spk_proj = Sequential(Linear(in_dim -> out_dim, bias),
// LayerNorm(out_dim, eps)) in bf16, as runtime.py builds it (nn.LayerNorm
// default eps 1e-5; pass ln_eps explicitly).
//   linear:    y[n] = bf16( sum_k x[k]*W[n,k] + b[n] )  fp32 accumulation in
//              a fixed order (cuBLAS's order is not reproducible: measured
//              equality against torch is reported, not guaranteed).
//   layernorm: torch 2.7.1 vectorized CUDA layer_norm, replicated
//              (Welford, block (32,4), vec 4): bitwise for the same input.
// x: [rows, in_dim], bf16 (x_is_f32 = 0) or fp32 (x_is_f32 = 1, rounded to
// bf16 first exactly like `.to(bfloat16)`). W: [out_dim, in_dim] bf16 row
// major, b/ln_w/ln_b: [out_dim]. in_dim % 256 == 0, out_dim % 8 == 0,
// out_dim % 4 == 0. out: [rows, out_dim] (row stride out_stride), 8-byte
// aligned with out_stride % 4 == 0 (the LayerNorm's vector width; checked
// before the first launch). Two launches (linear into out, then LayerNorm in
// place).
// ---------------------------------------------------------------------------
cudaError_t kt2n_ops_spk_proj(const void* x, int x_is_f32, int64_t x_stride,
                              const __nv_bfloat16* w, const __nv_bfloat16* b,
                              const __nv_bfloat16* ln_w, const __nv_bfloat16* ln_b, float ln_eps,
                              __nv_bfloat16* out, int64_t out_stride,
                              int rows, int in_dim, int out_dim, cudaStream_t stream);

// The two halves, exposed for tests and reuse.
cudaError_t kt2n_ops_linear_bias(const void* x, int x_is_f32, int64_t x_stride,
                                 const __nv_bfloat16* w, const __nv_bfloat16* b,
                                 __nv_bfloat16* out, int64_t out_stride,
                                 int rows, int in_dim, int out_dim, cudaStream_t stream);

// torch.nn.functional.layer_norm(x, (n,), w, b, eps) for bf16 x; w and b may
// be NULL. mean_out / rstd_out (fp32 [rows], optional) receive the same
// values torch.native_layer_norm returns. n % 4 == 0; out may equal x.
cudaError_t kt2n_ops_layernorm(const __nv_bfloat16* x, int64_t x_stride,
                               const __nv_bfloat16* w, const __nv_bfloat16* b, float eps,
                               __nv_bfloat16* out, int64_t out_stride,
                               float* mean_out, float* rstd_out,
                               int rows, int n, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif
