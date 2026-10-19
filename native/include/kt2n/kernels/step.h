/* kt2n k5-step: the whole Qwen3 decode step of the Kitten TTS 2 LM, assembled from the kernel lanes
 * (k1 tern_proj, k2 ops, k3 attn, k4 samp).  Host code only: every call enqueues kernels on `stream`,
 * allocates nothing, never synchronizes, and is CUDA-graph capturable (inputs are read on the device).
 *
 * One step for `rows` rows (a bucket, 1..16):
 *   embed(tokens) + RMSNorm(ln_in[0]) + RoPE table(positions)
 *   28 x [ qkv proj -> q/k norm + RoPE + K/V append at column *cursor (slot r = row r) -> decode attention
 *          -> o proj -> residual + RMSNorm(ln_post) -> gate_up proj with SiLU*up -> down proj
 *          -> residual + RMSNorm(ln_in[l+1] | final norm) ]
 *   audio head (6568 rows of the tied embedding) -> bf16 logits
 *   [optional] sampler chain + exact Philox draw -> next token ids (state updated on the device)
 *
 * Numerics: every kernel computes a row from that row's inputs only (bitwise batch invariant for rows 1..16,
 * deterministic).  Rounding points follow HF Qwen3 bf16 eager (fp32 accumulation, bf16 outputs).
 *
 * K/V layout: production's SlotKV.  Per layer, K and V are bf16 [slots, kv_heads=8, ctx, 128] with element
 * (slot, h, col, d) at base + slot*kv_slot_stride + h*kv_head_stride + col*128 + d.  Row r of the step uses
 * slot r.  The current token is written at column cursor[0]; attention covers columns
 * [cursor - min(position, cursor), cursor] (production's semantics).
 *
 * All exported symbols are extern "C" with the prefix kt2n_step_.  Return codes: 0 ok, KT2N_STEP_EINVAL
 * (bad arguments, nothing launched), otherwise the failing lane's code + KT2N_STEP_E<lane> * 100000 so the
 * caller can tell which kernel refused (see kt2n_step_error_string).
 */
#ifndef KT2N_KERNELS_STEP_H
#define KT2N_KERNELS_STEP_H

#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KT2N_STEP_ABI 1
#define KT2N_STEP_MAX_ROWS 16
/* kt2n_step_forward_o1 with KT2N_STEP_FLAG_ROW_KV accepts up to this many rows (the native LM's buckets up to 64);
 * kt2n_step_scratch_bytes / kt2n_step_scratch_bind size scratch for up to this many rows. */
#define KT2N_STEP_MAX_ROWS_EXT 64
#define KT2N_STEP_MAX_LAYERS 64

/* Geometry fixed by the kernels (Qwen3 1.7B). */
#define KT2N_STEP_HIDDEN 2048
#define KT2N_STEP_INTER 6144
#define KT2N_STEP_QKV 4096
#define KT2N_STEP_Q_HEADS 16
#define KT2N_STEP_KV_HEADS 8
#define KT2N_STEP_HEAD_DIM 128

/* io.flags */
#define KT2N_STEP_FLAG_PDL 0x1u     /* programmatic dependent launch between the step's kernels (sm_90+) */
#define KT2N_STEP_FLAG_NO_HEAD 0x2u /* stop after the final norm (normed holds the head input) */
#define KT2N_STEP_FLAG_SAMPLE 0x4u  /* run the sampler after the head (needs the samp_* fields) */
/* Per-row K/V (native LM, kt2n_step_forward_o1 only; L3, 10 Oct).  `cursor` is a device int64 [rows] array: row r
 * writes its key/value at column cursor[r] and attends columns [cursor[r] - min(positions[r], cursor[r]),
 * cursor[r]] (pass cursor = positions for the flat-row layout: logical position t at column t).  Row r's K/V slot
 * is samp_slot_of_row[r] (NULL = r), so rows can join and leave a bucket without moving K/V.  Allows rows up to
 * KT2N_STEP_MAX_ROWS_EXT.  Requires the fused attention options (KT2N_STEP_O1_ATTN | KT2N_STEP_O1_FUSE_QK), which
 * every bucket of the default table has. */
#define KT2N_STEP_FLAG_ROW_KV 0x8u

#define KT2N_STEP_OK 0
#define KT2N_STEP_EINVAL 1
/* lane error = code + lane * 100000 (lane: 1 tern, 2 ops, 3 attn, 4 samp) */
#define KT2N_STEP_ELANE 100000

typedef struct kt2n_step_layer {
  /* kernel-layout ternary weights (kt2n_tern_repack*), see kt2n_step_weight_bytes */
  const void *qkv_codes, *qkv_scales;       /* N 4096 (q|k|v), K 2048 */
  const void *o_codes, *o_scales;           /* N 2048, K 2048 */
  const void *gate_up_codes, *gate_up_scales; /* N 12288 interleaved (gate_i row 2i, up_i row 2i+1), K 2048 */
  const void *down_codes, *down_scales;     /* N 2048, K 6144 */
  /* dense bf16 */
  const void *ln_in;   /* [2048] input_layernorm.weight */
  const void *ln_post; /* [2048] post_attention_layernorm.weight */
  const void *q_norm;  /* [128] */
  const void *k_norm;  /* [128] */
  /* this layer's K/V cache, bf16 [slots, 8, ctx, 128] (strides in the model) */
  void *k_cache, *v_cache;
} kt2n_step_layer;

typedef struct kt2n_step_model {
  int32_t abi;    /* KT2N_STEP_ABI */
  int32_t layers; /* 28 */
  int64_t vocab;  /* 158241: rows of embed */
  float rms_eps;  /* 1e-6 */
  int32_t head_n; /* 6568: rows of head_w */
  const void *embed;      /* bf16 [vocab, 2048] */
  const void *final_norm; /* bf16 [2048] */
  const void *head_w;     /* bf16 [head_n, 2048], row stride 2048 (= embed + 151669*2048 for the audio slice) */
  int32_t slots;          /* K/V slots (>= rows of any step) */
  int32_t max_ctx;        /* columns per slot */
  int64_t kv_slot_stride; /* elements; 8*ctx*128 for production's layout */
  int64_t kv_head_stride; /* elements; ctx*128 */
  const kt2n_step_layer *layer; /* HOST array [layers] */
} kt2n_step_model;

typedef struct kt2n_step_io {
  int32_t rows;   /* 1..16 */
  uint32_t flags; /* KT2N_STEP_FLAG_* */
  const int64_t *tokens;    /* device [rows], full-vocabulary ids */
  const int64_t *positions; /* device [rows], logical position of the token (= row length before the step) */
  const int64_t *cursor;    /* device [1], the K/V column written this step */
  void *scratch;            /* device, kt2n_step_scratch_bytes(rows) bytes, 256-byte aligned */
  size_t scratch_bytes;
  void *attn_ws;            /* device, >= kt2n_step_attn_ws_bytes(rows, max_ctx), initialised once */
  size_t attn_ws_bytes;
  void *logits;             /* device bf16 [rows, ld_logits] (unless FLAG_NO_HEAD) */
  int64_t ld_logits;        /* >= head_n */
  /* optional sampler (FLAG_SAMPLE), see kt2n_samp_sample in kt2n/kernels/samp.h */
  const float *samp_params;   /* f32 [rows, 5] */
  int32_t *samp_history;      /* i32 [slots, 50] */
  int32_t *samp_history_len;  /* i32 [slots] */
  const uint64_t *samp_seeds; /* u64 [slots] */
  uint64_t *samp_offsets;     /* u64 [slots] */
  const int32_t *samp_slot_of_row; /* i32 [rows] or NULL (row r -> slot r) */
  int32_t *samp_tokens;       /* i32 [rows] out */
  uint32_t samp_flags;        /* KT2N_SAMP_* (UPDATE_STATE etc.) */
  /* optional shared voice prefix (ROW_KV + the fused attention only; L3, 10 Oct): bucket row r's first voice_len[r]
   * K/V columns are read from voice_kv + voice_off[r] (layout [2][layers][8][voice_len][128] bf16) instead of its
   * slot's own copy of them, so rows of one voice share those reads.  The values must equal the slot's copy (the
   * results are then unchanged).  voice_off[r] = -1: the row has no voice.  NULL: off.  (Indexed by bucket row,
   * not slot, since sprint 3: the kernel then needs no row_slot load before these.) */
  const void *voice_kv;
  const int64_t *voice_off;   /* i64 [rows] element offsets */
  const int32_t *voice_len;   /* i32 [rows] */
} kt2n_step_io;

/* Scratch layout (all bf16, row stride in elements): */
typedef struct kt2n_step_scratch_view {
  void *hidden; /* [rows, 2048] residual stream */
  void *normed; /* [rows, 2048] */
  void *qkv;    /* [rows, 4096] */
  void *query;  /* [rows, 2048] (16 heads x 128) */
  void *attn;   /* [rows, 2048] */
  void *delta;  /* [rows, 2048] o / down projection output */
  void *act;    /* [rows, 6144] */
  void *cos;    /* [rows, 64] */
  void *sin;    /* [rows, 64] */
} kt2n_step_scratch_view;

int kt2n_step_abi(void);
size_t kt2n_step_scratch_bytes(int rows);
int kt2n_step_scratch_bind(void *scratch, size_t scratch_bytes, int rows, kt2n_step_scratch_view *out);
size_t kt2n_step_attn_ws_bytes(int rows, int max_ctx);
int kt2n_step_attn_ws_init(void *ws, size_t bytes, cudaStream_t stream);

/* Kernel-layout sizes for one layer's four families. */
typedef struct kt2n_step_weight_bytes_t {
  size_t qkv_codes, qkv_scales, o_codes, o_scales, gate_up_codes, gate_up_scales, down_codes, down_scales;
} kt2n_step_weight_bytes_t;
void kt2n_step_weight_bytes(kt2n_step_weight_bytes_t *out);

/* Canonical (pack_exact) weights of one layer: int32 codes [N, K/16], bf16 scales [N, K/128]. */
typedef struct kt2n_step_canon_layer {
  const int32_t *q_codes, *k_codes, *v_codes, *o_codes, *gate_codes, *up_codes, *down_codes;
  const void *q_scales, *k_scales, *v_scales, *o_scales, *gate_scales, *up_scales, *down_scales;
} kt2n_step_canon_layer;

/* Repack one layer into the kernel layout (dst pointers of kt2n_step_layer, sized by kt2n_step_weight_bytes). */
int kt2n_step_repack_layer(const kt2n_step_canon_layer *src, const kt2n_step_layer *dst, cudaStream_t stream);

/* The decode step. */
int kt2n_step_forward(const kt2n_step_model *model, const kt2n_step_io *io, cudaStream_t stream);

/* Standalone sampler call (Stage A's NE_SAMPLER path): kt2n_samp_sample with in-kernel torch-exact noise.
 * Fields as in kt2n/kernels/samp.h; tokens are written as full-vocabulary ids (column + KT2N_SAMP_AUDIO_LOW). */
typedef struct kt2n_step_sample_io {
  int32_t rows;
  uint32_t flags; /* KT2N_SAMP_* */
  const void *logits; /* bf16 [rows, ld_logits] */
  int64_t ld_logits;
  const float *params;        /* f32 [*, 5] */
  int32_t *history;           /* i32 [*, 50] */
  int32_t *history_len;       /* i32 [*] */
  const uint64_t *seeds;      /* u64 [*] */
  uint64_t *offsets;          /* u64 [*] */
  const int32_t *slot_of_row; /* i32 [rows] or NULL */
  int32_t *tokens;            /* i32 [rows] out */
  float *scores;              /* f32 [rows, 6568] or NULL */
  float *probs;               /* f32 [rows, 6568] or NULL */
} kt2n_step_sample_io;
int kt2n_step_sample(const kt2n_step_sample_io *io, cudaStream_t stream);

/* ABI self-check for bindings: sizeof of the structs above, in this order: layer, model, io, scratch_view,
 * weight_bytes, canon_layer; then offsetof(model, layer), offsetof(io, samp_flags); then sizeof(sample_io),
 * offsetof(sample_io, probs).  Returns the count written. */
int kt2n_step_struct_layout(int64_t *out, int capacity);

/* Static description of the last error code (never NULL). */
const char *kt2n_step_error_string(int code);

#ifdef __cplusplus
}
#endif

#endif /* KT2N_KERNELS_STEP_H */
