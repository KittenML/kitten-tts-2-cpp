// kt2n k5-step: the whole decode step, assembled from the kernel lanes' launchers.
//
// Host code only (no device code of its own).  It lives under src/cuda/kernels/ so the repo's CMake glob puts
// it into kt2n_cuda (static: engine, binding) and libkt2n_kernels.so (ctypes tests), next to the launchers it
// calls.  See include/kt2n/kernels/step.h for the contract.
#include "kt2n/kernels/step.h"

#include <cuda_bf16.h>
#include <stddef.h>
#include <stdint.h>

#include "kt2n/kernels/attn.h"
#include "kt2n/kernels/ops.h"
#include "kt2n/kernels/samp.h"
#include "kt2n/kernels/tern_proj.h"

namespace {

constexpr int kH = KT2N_STEP_HIDDEN;
constexpr int kI = KT2N_STEP_INTER;
constexpr int kQKV = KT2N_STEP_QKV;
constexpr int kHQ = KT2N_STEP_Q_HEADS;
constexpr int kHK = KT2N_STEP_KV_HEADS;
constexpr int kD = KT2N_STEP_HEAD_DIM;
constexpr int kCS = 64;  // compact RoPE table width (k2: cs_stride 64)
constexpr size_t kAlign = 256;

constexpr size_t align_up(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

// Scratch arrays in order; element counts per row (bf16).
constexpr int64_t kScratchCols[] = {kH, kH, kQKV, kH, kH, kH, kI, kCS, kCS};
constexpr int kScratchArrays = sizeof(kScratchCols) / sizeof(kScratchCols[0]);

size_t scratch_bytes_rows(int rows) {
  size_t total = 0;
  for (int i = 0; i < kScratchArrays; ++i) total += align_up(static_cast<size_t>(rows) * kScratchCols[i] * 2);
  return total;
}

int lane_err(int lane, long code) {
  if (code == 0) return KT2N_STEP_OK;
  const long c = code < 0 ? 50000 - code : code;  // negative lane codes (samp) map to 50001..
  return static_cast<int>(lane * KT2N_STEP_ELANE + (c % KT2N_STEP_ELANE));
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

}  // namespace

extern "C" {

int kt2n_step_abi(void) { return KT2N_STEP_ABI; }

size_t kt2n_step_scratch_bytes(int rows) {
  if (rows < 1 || rows > KT2N_STEP_MAX_ROWS_EXT) return 0;
  return scratch_bytes_rows(rows);
}

int kt2n_step_scratch_bind(void* scratch, size_t scratch_bytes, int rows, kt2n_step_scratch_view* out) {
  if (!scratch || !out || rows < 1 || rows > KT2N_STEP_MAX_ROWS_EXT) return KT2N_STEP_EINVAL;
  if ((reinterpret_cast<uintptr_t>(scratch) % kAlign) != 0) return KT2N_STEP_EINVAL;
  if (scratch_bytes < scratch_bytes_rows(rows)) return KT2N_STEP_EINVAL;
  void* ptrs[kScratchArrays];
  char* p = static_cast<char*>(scratch);
  for (int i = 0; i < kScratchArrays; ++i) {
    ptrs[i] = p;
    p += align_up(static_cast<size_t>(rows) * kScratchCols[i] * 2);
  }
  out->hidden = ptrs[0];
  out->normed = ptrs[1];
  out->qkv = ptrs[2];
  out->query = ptrs[3];
  out->attn = ptrs[4];
  out->delta = ptrs[5];
  out->act = ptrs[6];
  out->cos = ptrs[7];
  out->sin = ptrs[8];
  return KT2N_STEP_OK;
}

size_t kt2n_step_attn_ws_bytes(int rows, int max_ctx) { return kt2n_attn_decode_workspace_bytes(rows, max_ctx); }

int kt2n_step_attn_ws_init(void* ws, size_t bytes, cudaStream_t stream) {
  return lane_err(3, kt2n_attn_workspace_init(ws, bytes, stream));
}

void kt2n_step_weight_bytes(kt2n_step_weight_bytes_t* out) {
  if (!out) return;
  out->qkv_codes = kt2n_tern_codes_bytes(kQKV, kH);
  out->qkv_scales = kt2n_tern_scales_bytes(kQKV, kH);
  out->o_codes = kt2n_tern_codes_bytes(kH, kH);
  out->o_scales = kt2n_tern_scales_bytes(kH, kH);
  out->gate_up_codes = kt2n_tern_codes_bytes(2 * kI, kH);
  out->gate_up_scales = kt2n_tern_scales_bytes(2 * kI, kH);
  out->down_codes = kt2n_tern_codes_bytes(kH, kI);
  out->down_scales = kt2n_tern_scales_bytes(kH, kI);
}

int kt2n_step_repack_layer(const kt2n_step_canon_layer* src, const kt2n_step_layer* dst, cudaStream_t stream) {
  if (!src || !dst) return KT2N_STEP_EINVAL;
  int rc = kt2n_tern_repack_qkv(src->q_codes, src->q_scales, src->k_codes, src->k_scales, src->v_codes,
                                src->v_scales, kHQ * kD, kHK * kD, kH, const_cast<void*>(dst->qkv_codes),
                                const_cast<void*>(dst->qkv_scales), stream);
  if (rc) return lane_err(1, rc);
  rc = kt2n_tern_repack(src->o_codes, src->o_scales, kH, kH, const_cast<void*>(dst->o_codes),
                        const_cast<void*>(dst->o_scales), kH, 0, 1, stream);
  if (rc) return lane_err(1, rc);
  rc = kt2n_tern_repack_gate_up(src->gate_codes, src->gate_scales, src->up_codes, src->up_scales, kI, kH,
                                const_cast<void*>(dst->gate_up_codes), const_cast<void*>(dst->gate_up_scales),
                                stream);
  if (rc) return lane_err(1, rc);
  rc = kt2n_tern_repack(src->down_codes, src->down_scales, kH, kI, const_cast<void*>(dst->down_codes),
                        const_cast<void*>(dst->down_scales), kH, 0, 1, stream);
  if (rc) return lane_err(1, rc);
  return KT2N_STEP_OK;
}

int kt2n_step_forward(const kt2n_step_model* m, const kt2n_step_io* io, cudaStream_t s) {
  if (!m || !io || m->abi != KT2N_STEP_ABI) return KT2N_STEP_EINVAL;
  if (m->layers < 1 || m->layers > KT2N_STEP_MAX_LAYERS || !m->layer || !m->embed || !m->final_norm) {
    return KT2N_STEP_EINVAL;
  }
  const int rows = io->rows;
  if (rows < 1 || rows > KT2N_STEP_MAX_ROWS || rows > m->slots || m->max_ctx < 1) return KT2N_STEP_EINVAL;
  if (io->flags & KT2N_STEP_FLAG_ROW_KV) return KT2N_STEP_EINVAL;  // per-row K/V: kt2n_step_forward_o1 only
  if (!io->tokens || !io->positions || !io->cursor) return KT2N_STEP_EINVAL;
  if (!io->scratch || io->scratch_bytes < scratch_bytes_rows(rows)) return KT2N_STEP_EINVAL;
  if (!io->attn_ws || io->attn_ws_bytes < kt2n_attn_decode_workspace_bytes(rows, m->max_ctx)) {
    return KT2N_STEP_EINVAL;
  }
  const bool head = !(io->flags & KT2N_STEP_FLAG_NO_HEAD);
  const bool sample = (io->flags & KT2N_STEP_FLAG_SAMPLE) != 0;
  if (head && (!m->head_w || m->head_n < 1 || !io->logits || io->ld_logits < m->head_n || !aligned16(m->head_w))) {
    return KT2N_STEP_EINVAL;
  }
  if (sample && (!head || m->head_n != KT2N_SAMP_AUDIO_N || !io->samp_params || !io->samp_history ||
                 !io->samp_history_len || !io->samp_seeds || !io->samp_offsets || !io->samp_tokens)) {
    return KT2N_STEP_EINVAL;
  }
  for (int l = 0; l < m->layers; ++l) {
    const kt2n_step_layer& L = m->layer[l];
    if (!L.qkv_codes || !L.qkv_scales || !L.o_codes || !L.o_scales || !L.gate_up_codes || !L.gate_up_scales ||
        !L.down_codes || !L.down_scales || !L.ln_in || !L.ln_post || !L.q_norm || !L.k_norm || !L.k_cache ||
        !L.v_cache) {
      return KT2N_STEP_EINVAL;
    }
  }

  kt2n_step_scratch_view v;
  if (kt2n_step_scratch_bind(io->scratch, io->scratch_bytes, rows, &v)) return KT2N_STEP_EINVAL;
  using bf = __nv_bfloat16;
  bf* hidden = static_cast<bf*>(v.hidden);
  bf* normed = static_cast<bf*>(v.normed);
  bf* qkv = static_cast<bf*>(v.qkv);
  bf* query = static_cast<bf*>(v.query);
  bf* attn = static_cast<bf*>(v.attn);
  bf* delta = static_cast<bf*>(v.delta);
  bf* act = static_cast<bf*>(v.act);
  bf* cos_t = static_cast<bf*>(v.cos);
  bf* sin_t = static_cast<bf*>(v.sin);

  const int pdl = (io->flags & KT2N_STEP_FLAG_PDL) ? 1 : 0;
  const unsigned tflags = pdl ? KT2N_TERN_FLAG_PDL : 0u;
  const unsigned aflags = pdl ? KT2N_ATTN_FLAG_PDL : 0u;
  const float eps = m->rms_eps;
  long rc;

  rc = kt2n_ops_embed_rmsnorm(io->tokens, rows, static_cast<const bf*>(m->embed), m->vocab, kH, nullptr, 0, nullptr,
                              -1, static_cast<const bf*>(m->layer[0].ln_in), eps, hidden, kH, normed, kH,
                              io->positions, nullptr, cos_t, sin_t, kCS, pdl, s);
  if (rc) return lane_err(2, rc);

  for (int l = 0; l < m->layers; ++l) {
    const kt2n_step_layer& L = m->layer[l];
    rc = kt2n_tern_proj(KT2N_TERN_FAM_QKV, normed, kH, rows, L.qkv_codes, L.qkv_scales, kQKV, kH, qkv, kQKV,
                        KT2N_TERN_EPI_STORE, nullptr, 0, tflags, s);
    if (rc) return lane_err(1, rc);

    kt2n_ops_kv_target kv{};
    kv.k = static_cast<bf*>(L.k_cache);
    kv.v = static_cast<bf*>(L.v_cache);
    kv.slot_stride = m->kv_slot_stride;
    kv.head_stride = m->kv_head_stride;
    kv.col_stride = kD;
    kv.row_slot = nullptr;
    kv.slot_base = 0;
    kv.slot_step = 1;
    kv.row_col = nullptr;
    kv.cursor = io->cursor;
    kv.col_base = 0;
    kv.col_step = 0;
    kv.num_cols = m->max_ctx;
    rc = kt2n_ops_qk_norm_rope(qkv, kQKV, qkv + kHQ * kD, kQKV, qkv + (kHQ + kHK) * kD, kQKV,
                               static_cast<const bf*>(L.q_norm), static_cast<const bf*>(L.k_norm), eps, nullptr,
                               nullptr, cos_t, sin_t, kCS, query, kHQ * kD, kD, nullptr, 0, 0, &kv, rows, kHQ, kHK,
                               kD, pdl, s);
    if (rc) return lane_err(2, rc);

    rc = kt2n_attn_decode_slotkv(query, kHQ * kD, L.k_cache, L.v_cache, m->kv_slot_stride, m->kv_head_stride,
                                 io->positions, io->cursor, rows, m->max_ctx, attn, kHQ * kD, io->attn_ws,
                                 io->attn_ws_bytes, aflags, s);
    if (rc) return lane_err(3, rc);

    rc = kt2n_tern_proj(KT2N_TERN_FAM_O, attn, kH, rows, L.o_codes, L.o_scales, kH, kH, delta, kH,
                        KT2N_TERN_EPI_STORE, nullptr, 0, tflags, s);
    if (rc) return lane_err(1, rc);
    rc = kt2n_ops_residual_rmsnorm(hidden, kH, delta, kH, static_cast<const bf*>(L.ln_post), eps, hidden, kH,
                                   normed, kH, rows, kH, pdl, s);
    if (rc) return lane_err(2, rc);

    rc = kt2n_tern_proj(KT2N_TERN_FAM_GATE_UP, normed, kH, rows, L.gate_up_codes, L.gate_up_scales, 2 * kI, kH,
                        act, kI, KT2N_TERN_EPI_SILU_MUL, nullptr, 0, tflags, s);
    if (rc) return lane_err(1, rc);
    rc = kt2n_tern_proj(KT2N_TERN_FAM_DOWN, act, kI, rows, L.down_codes, L.down_scales, kH, kI, delta, kH,
                        KT2N_TERN_EPI_STORE, nullptr, 0, tflags, s);
    if (rc) return lane_err(1, rc);
    const void* next_w = (l + 1 < m->layers) ? m->layer[l + 1].ln_in : m->final_norm;
    rc = kt2n_ops_residual_rmsnorm(hidden, kH, delta, kH, static_cast<const bf*>(next_w), eps, hidden, kH, normed,
                                   kH, rows, kH, pdl, s);
    if (rc) return lane_err(2, rc);
  }

  if (head) {
    rc = kt2n_samp_head_bf16(normed, kH, m->head_w, kH, io->logits, io->ld_logits, rows, m->head_n, kH, s);
    if (rc) return lane_err(4, rc);
  }
  if (sample) {
    rc = kt2n_samp_sample(io->logits, io->ld_logits, io->samp_params, io->samp_history, io->samp_history_len,
                          io->samp_seeds, io->samp_offsets, io->samp_slot_of_row, nullptr, 0, io->samp_tokens,
                          nullptr, nullptr, rows, io->samp_flags, s);
    if (rc) return lane_err(4, rc);
  }
  return KT2N_STEP_OK;
}

int kt2n_step_sample(const kt2n_step_sample_io* io, cudaStream_t stream) {
  if (!io || io->rows < 1 || io->rows > KT2N_STEP_MAX_ROWS || !io->logits || !io->tokens) return KT2N_STEP_EINVAL;
  const int rc = kt2n_samp_sample(io->logits, io->ld_logits, io->params, io->history, io->history_len, io->seeds,
                                  io->offsets, io->slot_of_row, nullptr, 0, io->tokens, io->scores, io->probs,
                                  io->rows, io->flags, stream);
  return lane_err(4, rc);
}

int kt2n_step_struct_layout(int64_t* out, int capacity) {
  const int64_t v[] = {static_cast<int64_t>(sizeof(kt2n_step_layer)),
                       static_cast<int64_t>(sizeof(kt2n_step_model)),
                       static_cast<int64_t>(sizeof(kt2n_step_io)),
                       static_cast<int64_t>(sizeof(kt2n_step_scratch_view)),
                       static_cast<int64_t>(sizeof(kt2n_step_weight_bytes_t)),
                       static_cast<int64_t>(sizeof(kt2n_step_canon_layer)),
                       static_cast<int64_t>(offsetof(kt2n_step_model, layer)),
                       static_cast<int64_t>(offsetof(kt2n_step_io, samp_flags)),
                       static_cast<int64_t>(sizeof(kt2n_step_sample_io)),
                       static_cast<int64_t>(offsetof(kt2n_step_sample_io, probs))};
  const int n = static_cast<int>(sizeof(v) / sizeof(v[0]));
  if (!out || capacity < n) return n;
  for (int i = 0; i < n; ++i) out[i] = v[i];
  return n;
}

const char* kt2n_step_error_string(int code) {
  if (code == KT2N_STEP_OK) return "ok";
  if (code == KT2N_STEP_EINVAL) return "kt2n_step: invalid argument (nothing launched)";
  switch (code / KT2N_STEP_ELANE) {
    case 1:
      return "kt2n_step: ternary projection launcher failed (k1 tern_proj)";
    case 2:
      return "kt2n_step: small-op launcher failed (k2 ops)";
    case 3:
      return "kt2n_step: attention launcher failed (k3 attn)";
    case 4:
      return "kt2n_step: head/sampler launcher failed (k4 samp)";
    default:
      return "kt2n_step: unknown error";
  }
}

}  // extern "C"
