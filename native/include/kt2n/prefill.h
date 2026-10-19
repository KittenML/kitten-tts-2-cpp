/* kt2n lane p2-prefill (M2): native LM prefill for Stage A.
 *
 * What it computes.  For each request row r of a call: the K/V of `new_tokens[r]` tokens appended after
 * `past_len[r]` keys that are already present (the voice's per-voice prefix K/V, or nothing), through all 28
 * layers, plus optionally the final-normed hidden state and the audio-slice logits at the row's LAST token.
 * Building a voice's prefix K/V is the same call with past_len = 0 (the speaker vector replaces the embedding
 * of logical position 0).  A burst is the same call with several rows of any voices and lengths.
 *
 * Numerics.  Every operator is a verified kernel-lane operator with row-invariant arithmetic (k1 ternary
 * projections, k2 embed/RMSNorm/q-k norm + RoPE + K/V append, k3 prefill attention, k4 head).  So a row's K/V
 * and logits depend only on its own tokens, speaker and past K/V: bitwise the same alone or in a burst of any
 * size/composition, and a (prefix, suffix) split gives bitwise the bits of a one-pass prefill of the whole
 * prompt.  Layer order is HF Qwen3's: h = x + o(attn(qk_norm_rope(qkv(rms(x))))); x' = h + down(silu(gate) *
 * up)(rms(h)); logits = head(rms_final(x_last)).
 *
 * K/V destination layout ("flat row", the layout production's FLAT_ADMIT admits with one strided copy):
 *   row d of kv_dest = kv_dest + d * kv_row_stride elements, holding [2][layers][kv_heads][kv_width][head_dim]
 *   bf16: K of layer l at [0][l], V at [1][l]; token at logical position t sits in column t.
 *   A row with past_len p must have columns [0, p) filled before the call (or pass past_kv[r], which this call
 *   copies in first: a contiguous [2][layers][kv_heads][p][head_dim] tensor = production's per-voice entry
 *   layout transposed to flat order).  The call writes columns [p, p + new_tokens).  So
 *   kv_dest[d][:, :, :, :n] (n = p + new_tokens) is exactly what SlotKV.admit(flat=...) consumes.
 *
 * Every pointer is a device pointer unless named host_*.  All functions return 0 on success or a
 * KT2N_PF_E* code; the last error message is available from kt2n_pf_last_error() (thread local).
 * No global state: a model and any number of contexts per process; a context is used by one host thread
 * at a time on one stream.
 */
#ifndef KT2N_PREFILL_H
#define KT2N_PREFILL_H

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KT2N_PF_API_VERSION 1
#define KT2N_PF_MAX_LAYERS 64

#define KT2N_PF_OK 0
#define KT2N_PF_EINVAL 1   /* bad argument (nothing launched) */
#define KT2N_PF_ECUDA 2    /* CUDA error (launch or allocation) */
#define KT2N_PF_EFILE 3    /* engine file missing / malformed / not this model */
#define KT2N_PF_EKERNEL 4  /* a kernel-lane launcher rejected its arguments (bug) */

/* Audio slice of the tied head (runtime.AUDIO_LOGITS_LOW / AUDIO_LOGITS_SIZE). */
#define KT2N_PF_AUDIO_LOW 151669
#define KT2N_PF_AUDIO_N 6568

/* run flags */
#define KT2N_PF_FLAG_NO_PDL 0x1u         /* launch without programmatic dependent launch (same bits) */
#define KT2N_PF_FLAG_ATTN_SEQUENTIAL 0x2u /* force k3's workspace-free prefill shape (same bits) */
/* prof-prefill variants (off by default; every one gives the same bits as the default path):            */
#define KT2N_PF_FLAG_OVERLAP_PAST 0x4u    /* calls with >= 2 rows with a past: copy past K/V on a side stream in
                                             layer groups, overlapped with compute; layer l's attention waits only
                                             for its group's copy */
#define KT2N_PF_FLAG_PROJ_TABLE 0x8u      /* prefill projections: launch shape per (family, M) from the measured
                                             table in src/prefill/pf_tern.cu -- k1's kernels with another launch
                                             shape, or the register-lean multi-stage GEMM (same bits) */
#define KT2N_PF_FLAG_ATTN_DECODE_ROWS 0x10u /* layers 0..L-2: attention through k3's decode kernel, one decode row per
                                               new token (bitwise k3's prefill: same per-element arithmetic), when
                                               the call has <= KT2N_PF_DECODE_ROWS_MAX_T new tokens */
#define KT2N_PF_FLAG_ATTN_Q8 0x20u       /* layers 0..L-2: attention over tiles of <= 8 queries with k3's decode
                                             arithmetic (src/prefill/pf_attn.cu) for calls of <= KT2N_PF_Q8_MAX_T
                                             new tokens not taken by KT2N_PF_FLAG_ATTN_DECODE_ROWS */
#define KT2N_PF_FLAG_Q8_FORCE_PAR 0x40u  /* measurement: force the split-parallel q8 shape (needs workspace) */
#define KT2N_PF_FLAG_Q8_FORCE_SEQ 0x80u  /* measurement: force the sequential q8 shape */
#define KT2N_PF_DECODE_ROWS_MAX_T 20 /* decode rows: 1.14x at 14 tokens, 1.03x at 27, 0.91x at 53 (q8 wins > 20) */
#define KT2N_PF_Q8_MAX_T 160          /* q8 tiles: faster up to ~150 tokens, slower at 236+ (bursts, one-pass) */
/* Tuned variants measured faster with bitwise-identical results: */
#define KT2N_PF_FLAGS_FAST (KT2N_PF_FLAG_OVERLAP_PAST | KT2N_PF_FLAG_PROJ_TABLE | KT2N_PF_FLAG_ATTN_DECODE_ROWS | \
                            KT2N_PF_FLAG_ATTN_Q8)
/* Test hook: the environment variable KT2N_PF_FORCE_FLAGS (integer, e.g. 0x4), read when a context is created,
 * is OR-ed into the flags of every kt2n_pf_run call on that context (lets the existing suites run a variant). */

/* ---------------------------------------------------------------------------------------------------------
 * Weights: a plain struct of device pointers.  kt2n_pf_model_load fills one from a .kt2n file; an engine that
 * already holds the same buffers (k5) can fill its own and pass it to kt2n_pf_ctx_create -- nothing here
 * owns or frees the pointers of a caller-filled struct.
 * Projections are in k1's KERNEL layout (kt2n_tern_repack*): qkv N=4096 (q|k|v), o, gate_up N=12288
 * interleaved (gate_i row 2i, up_i row 2i+1), down K=6144.
 * --------------------------------------------------------------------------------------------------------- */
typedef struct kt2n_pf_layer {
  const void* qkv_codes;
  const void* qkv_scales;
  const void* o_codes;
  const void* o_scales;
  const void* gu_codes;
  const void* gu_scales;
  const void* down_codes;
  const void* down_scales;
  const void* input_norm; /* bf16 [hidden]  input_layernorm.weight */
  const void* post_norm;  /* bf16 [hidden]  post_attention_layernorm.weight */
  const void* q_norm;     /* bf16 [head_dim] self_attn.q_norm.weight */
  const void* k_norm;     /* bf16 [head_dim] self_attn.k_norm.weight */
} kt2n_pf_layer;

typedef struct kt2n_pf_weights {
  int32_t layers;       /* 28 */
  int32_t hidden;       /* 2048 */
  int32_t inter;        /* 6144 */
  int32_t q_heads;      /* 16 */
  int32_t kv_heads;     /* 8 */
  int32_t head_dim;     /* 128 */
  int64_t vocab;        /* 158241 */
  float rms_eps;        /* 1e-6 */
  float reserved0;
  const void* embed;      /* bf16 [vocab, hidden] (tied: also the head) */
  const void* final_norm; /* bf16 [hidden] model.norm.weight */
  const float* inv_freq;  /* fp32 [head_dim/2] RoPE inverse frequencies (HF buffer); NULL = k2's built-in */
  kt2n_pf_layer layer[KT2N_PF_MAX_LAYERS];
} kt2n_pf_weights;

typedef struct kt2n_pf_model kt2n_pf_model;

/* Load a .kt2n engine file onto `device`: dense tensors copied, every projection family repacked into k1's
 * kernel layout.  One device allocation (see kt2n_pf_model_bytes).  `expect_source_sha256` (64 hex chars or
 * NULL/"") must equal the file's recorded source safetensors sha256 when given.  Synchronous. */
int kt2n_pf_model_load(const char* path, int device, const char* expect_source_sha256, kt2n_pf_model** out);
const kt2n_pf_weights* kt2n_pf_model_weights(const kt2n_pf_model* model);
size_t kt2n_pf_model_bytes(const kt2n_pf_model* model);
/* The file's recorded source safetensors sha256 (64 hex) and the engine file path. */
const char* kt2n_pf_model_source_sha256(const kt2n_pf_model* model);
void kt2n_pf_model_free(kt2n_pf_model* model);

/* ---------------------------------------------------------------------------------------------------------
 * Context: per-stream workspace for calls with up to max_rows rows, max_tokens new tokens in total and up to
 * max_keys keys per row (past + new).  Larger calls are refused (EINVAL).  Device memory: see
 * kt2n_pf_ctx_bytes (about 37 KB per token of max_tokens plus the attention workspace).
 * --------------------------------------------------------------------------------------------------------- */
typedef struct kt2n_pf_ctx kt2n_pf_ctx;

int kt2n_pf_ctx_create(const kt2n_pf_weights* weights, int device, int max_rows, int max_tokens, int max_keys,
                       kt2n_pf_ctx** out);
/* The same over a caller-owned device workspace (e.g. a torch tensor, so every device byte lives in the caller's
 * allocator / GPU memory snapshot): `workspace` 256-byte aligned, >= kt2n_pf_ctx_workspace_bytes(...) bytes,
 * zeroed on `stream` here; it must outlive the context and is never freed by it.  Only a ~100 KB pinned host
 * buffer and an event are allocated by the context itself. */
size_t kt2n_pf_ctx_workspace_bytes(int max_rows, int max_tokens, int max_keys);
int kt2n_pf_ctx_create_in(const kt2n_pf_weights* weights, int device, int max_rows, int max_tokens, int max_keys,
                          void* workspace, size_t workspace_bytes, cudaStream_t stream, kt2n_pf_ctx** out);
size_t kt2n_pf_ctx_bytes(const kt2n_pf_ctx* ctx);
void kt2n_pf_ctx_free(kt2n_pf_ctx* ctx);

/* One prefill call (asynchronous on `stream`; the host arrays are consumed before return).
 *   num_rows               1..max_rows
 *   host_new_tokens  [R]   tokens to compute per row (>= 1)
 *   host_ids         [sum] the new token ids, row after row (int64)
 *   host_past_len    [R]   keys already present before the new tokens (logical position of the first new token)
 *   host_past_kv     [R]   NULL, or per row NULL / a device pointer to that row's past K/V as a contiguous
 *                          [2][layers][kv_heads][past_len][head_dim] bf16 tensor, copied into the row first
 *   speakers               bf16 [R, hidden] (row stride `hidden`) or NULL: row r's projected speaker vector
 *                          replaces the embedding of its logical position 0 (only rows with past_len == 0)
 *   kv_dest, kv_row_stride (elements, % 8 == 0), kv_width (columns per row >= past + new)
 *   host_dest_row    [R]   NULL (row r -> dest row r) or the destination row of each request (distinct)
 *   last_hidden            bf16 [R, hidden] (ld hidden) or NULL: final-normed hidden at each row's last token
 *   logits, ld_logits      bf16 [R, ld_logits >= KT2N_PF_AUDIO_N] or NULL: audio-slice logits at the last token
 *                          (column c = token KT2N_PF_AUDIO_LOW + c), k4's head (requires R <= 16)
 * When both last_hidden and logits are NULL only the K/V is produced (the last layer skips o_proj and MLP). */
int kt2n_pf_run(kt2n_pf_ctx* ctx, int num_rows, const int32_t* host_new_tokens, const int64_t* host_ids,
                const int32_t* host_past_len, const void* const* host_past_kv, const void* speakers,
                void* kv_dest, int64_t kv_row_stride, int32_t kv_width, const int32_t* host_dest_row,
                void* last_hidden, void* logits, int64_t ld_logits, unsigned flags, cudaStream_t stream);

/* Thread-local message of the last failing call ("" if none). */
const char* kt2n_pf_last_error(void);
int kt2n_pf_api_version(void);

#ifdef __cplusplus
}
#endif

#endif /* KT2N_PREFILL_H */
