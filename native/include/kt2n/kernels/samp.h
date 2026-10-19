// kt2n lane k4: audio-only LM head, the production sampler chain, and
// torch-exact Philox exponential noise.
//
// Every entry point takes raw device pointers and a cudaStream_t. Default
// builds never synchronize or allocate and are safe to capture in a CUDA graph.
// The experimental production head warms its cuBLAS handle eagerly; an
// unprepared shape/stream refuses during capture. Scratch is stream-ordered
// per invocation and becomes graph allocation/free nodes during capture.
// All per-row state the sampler needs (settings, history, RNG seed/offset)
// lives in device buffers, so a captured step needs no host work.
//
// Numerics contract:
//   * head: fp32 accumulation in a fixed K order, bf16 output (what the
//     production sampler reads); a row's bits never depend on M or on the
//     other rows. Experimental production-head builds instead use g83's two
//     cuBLAS shape classes (M=1 and padded M=16); see the head contract below.
//   * noise: bitwise equal to torch 2.7.1+cu128
//     `torch.empty(6568, device="cuda").exponential_(1, generator=g)` for the
//     k-th call on `g = torch.Generator("cuda").manual_seed(seed)`, given
//     offset = 4 * k.
//   * sampler: production's chain (runtime._sampling_processors with
//     per_row=True, graph_safe=True on the sliced head + sampler_graph's
//     softmax and argmax(p / noise)) with torch's own reduction orders for
//     every softmax, torch/cub's stable sort order (-0.0 == +0.0) and the
//     top-p cumsum order of production's bucket 1 (KT2N_SAMP_SCAN_MULTI_ROW
//     selects buckets 2..10's).  Measured bitwise equal to production's
//     scores, probabilities and tokens on 15,005 rows.
//
// Return value of every launcher: 0 on success, a positive cudaError_t on a
// launch error, or a negative KT2N_SAMP_E* code for bad arguments.
#pragma once

#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- the audio slice of the tied embedding --------------------------------
#define KT2N_SAMP_AUDIO_LOW 151669     /* SPEECH_START: column 0 of the slice  */
#define KT2N_SAMP_AUDIO_N 6568         /* FINAL_SEG - SPEECH_START + 1          */
#define KT2N_SAMP_AUDIO_ID_BASE 151675 /* first audio code token               */
#define KT2N_SAMP_AUDIO_VOCAB 6561     /* audio code tokens                     */
#define KT2N_SAMP_WINDOW 50            /* REPETITION_WINDOW                     */
#define KT2N_SAMP_TOP_K_MAX 200        /* sampler_params.TOP_K_MAX              */
#define KT2N_SAMP_PARAM_COLS 5         /* runtime.SAMPLER_ROW_COLUMNS           */
#define KT2N_SAMP_HEAD_MAX_ROWS 16

// ---- error codes ------------------------------------------------------------
#define KT2N_SAMP_EARG -1   /* null pointer / bad size / bad stride            */
#define KT2N_SAMP_EALIGN -2 /* pointer or stride not 16-byte aligned (head)    */

// ---- sampler flags ----------------------------------------------------------
// Temperature as true division `scores / t` (params column 0) instead of the
// reciprocal multiply `scores * float32(1/t)` (column 1).  Production measures
// which one its CUDA build uses (runtime._row_temperature_uses_reciprocal);
// on torch 2.7.1+cu128 it is the reciprocal, so leave this bit clear.
#define KT2N_SAMP_TEMP_DIVISION 0x1u
// After drawing: append the token to the row's history window and advance the
// row's Philox offset by 4 (one torch exponential_ call).  This is what makes
// a captured decode step self-contained.
#define KT2N_SAMP_UPDATE_STATE 0x2u
// Top-p cumsum order.  Default (bit clear): the order torch uses for a
// ONE-row tensor (cub DeviceScan) = production's bucket 1, i.e. a request
// sampled ALONE.  Bit set: torch's >= 2-row order (scan_innermost_dim) =
// production's buckets 2..10.  The two differ only in float rounding of the
// cumulative sum (measured: 7 of 3,000 adversarial tie-heavy rows change the
// top-p mask, 1 changes the token).  kt2n applies ONE order to every row at
// every M, so a row's draw never depends on its batch; the default makes each
// row draw what production draws for it alone.
#define KT2N_SAMP_SCAN_MULTI_ROW 0x4u

// ---------------------------------------------------------------------------
// Audio head: out[m, n] = bf16( sum_k x[m, k] * w[n, k] ), fp32 accumulate.
//   x   : bf16 [M, K], row stride ldx elements (the final-norm hidden states)
//   w   : bf16 [N, K], row stride ldw elements; for the audio head pass
//         embed_tokens + KT2N_SAMP_AUDIO_LOW * hidden and N = KT2N_SAMP_AUDIO_N
//   out : bf16 [M, N], row stride ldo elements
//   1 <= M <= 16, K % 256 == 0, M * K * 2 <= 96 KiB (x is staged in shared
//   memory), x/w 16-byte aligned, ldx/ldw multiples of 8.
// Experimental KT2N_HEAD_PRODUCTION_ORDER builds use g83's cuBLAS shapes:
// M=1 directly; M>=2 padded to 16. Warm up each shape/stream before capture.
// Independently captured graphs own separate scratch and can overlap when
// their caller-owned input/output buffers are also independent.
// ---------------------------------------------------------------------------
int kt2n_samp_head_bf16(const void* x, int64_t ldx, const void* w, int64_t ldw,
                        void* out, int64_t ldo, int M, int N, int K,
                        cudaStream_t stream);

/* Native prefill's fixed row-invariant head arithmetic, including when the
 * experimental decode production-order head is enabled. Additive C ABI entry;
 * the existing decode/public head entry retains its selected arithmetic. */
int kt2n_samp_head_prefill_bf16(const void* x, int64_t ldx, const void* w, int64_t ldw,
                        void* out, int64_t ldo, int M, int N, int K,
                        cudaStream_t stream);

// ---------------------------------------------------------------------------
// Torch-exact exponential noise (the sampler draws its own noise in-kernel;
// this is the standalone form for tests and for callers that keep the noise).
//   noise[r, i] for i < N, r < M, row stride ldn floats, from
//   seeds[s], offsets[s] with s = slot_of_row ? slot_of_row[r] : r.
//   Exact for N <= 256 * (number of torch grid blocks), i.e. any N up to
//   ~250k on current GPUs; the contract only needs N = 6568.
//   advance != 0: offsets[s] += 4 after the draw (one torch call).
// ---------------------------------------------------------------------------
int kt2n_samp_exponential(const uint64_t* seeds, uint64_t* offsets,
                          const int32_t* slot_of_row, float* noise, int64_t ldn,
                          int M, int N, int advance, cudaStream_t stream);

// ---------------------------------------------------------------------------
// The full production sampler chain + exponential race, one CTA per row.
//
//   logits      bf16 [M, ldl] audio-slice logits (column c = token 151669+c)
//   For each row r, s = slot_of_row ? slot_of_row[r] : r indexes the state:
//   params      f32 [*, 5]  {temperature, float32(1/temperature), top_k,
//                            top_p threshold = float32(1 - top_p) (-1 = off),
//                            min_p}   -- runtime.sampler_row()
//   history     i32 [*, 50] the row's last generated FULL-vocabulary ids,
//                           right-aligned: valid entries are
//                           history[s, 50 - history_len[s] .. 49]
//   history_len i32 [*]     0..50 (= min(new_tokens, 50))
//   seeds       u64 [*]     the segment generator's seed (job seed)
//   offsets     u64 [*]     torch philox offset = 4 * draws so far
//   (slots must be distinct within one launch: each row writes its slot)
//   noise       f32 [M, ldn] optional: use this noise instead of drawing
//                           (the torch-noise fallback); NULL = draw in-kernel
//   out_tokens  i32 [M]     the drawn FULL-vocabulary id (151669 + column)
//   out_scores  f32 [M, 6568] optional debug: the chain's scores (sampler
//                           graph's `scores`); NULL = skip
//   out_probs   f32 [M, 6568] optional debug: the final probabilities
//                           (sampler graph's `probabilities`); NULL = skip
//   flags       KT2N_SAMP_* bits
// Rows are independent: a row's token never depends on M or on other rows.
// ---------------------------------------------------------------------------
int kt2n_samp_sample(const void* logits, int64_t ldl, const float* params,
                     int32_t* history, int32_t* history_len,
                     const uint64_t* seeds, uint64_t* offsets,
                     const int32_t* slot_of_row, const float* noise,
                     int64_t ldn, int32_t* out_tokens, float* out_scores,
                     float* out_probs, int M, unsigned flags,
                     cudaStream_t stream);

// Dynamic shared memory the sampler kernel requests (bytes); for reports.
int kt2n_samp_sample_smem_bytes(void);

#ifdef __cplusplus
}  // extern "C"
#endif
