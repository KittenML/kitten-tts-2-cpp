// kt2n::lm device-side state and the small kernels of the LM loop (lane L3).  Internal header.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace kt2n::lm::dev {

inline constexpr int kRows = 64;          // largest bucket
inline constexpr int kAdmRows = 16;       // rows per admission
inline constexpr int kAudioN = 6568;      // audio slice of the head
inline constexpr int kSilentWords = (6561 + 31) / 32;
inline constexpr int kWindow = 50;        // sampler history window
inline constexpr int kParamCols = 5;

// slot flags
inline constexpr uint32_t kDemo = 0x1u;
inline constexpr uint32_t kChunked = 0x2u;
inline constexpr uint32_t kDone = 0x4u;

// finish codes (kt2n::lm::Finish)
inline constexpr int kFinNone = 0, kFinEos = 1, kFinStop = 2, kFinBudget = 3, kFinMaxNew = 4, kFinCeiling = 5;

// One ring entry in mapped host memory: written by the post kernel, read by the host after `seq` changes.
struct RingEntry {
  uint32_t seq;  // = launch sequence + 1 once the entry is complete (written last, after a system fence)
  int32_t rows;
  int32_t token[kRows];   // full-vocabulary id, -1 = row was already done (ignore)
  int32_t index[kRows];   // new_tokens after this token
  int32_t finish[kRows];  // finish code
};

struct State {
  // per slot (slots = max_rows + 1; the last slot is the padding rows' scratch slot, always done)
  int64_t* tok;         // next input token
  int64_t* pos;         // its logical position (= the K/V column it writes)
  int32_t* new_tokens;
  int32_t* silent;
  int32_t* budget;
  int32_t* cap;         // SMP max_new_tokens, 0 = none
  int32_t* prompt_len;
  uint32_t* flags;
  int32_t* forced_len;  // tests: forced ids per slot (pool row = slot)
  int32_t* forced;      // [slots][forced_cap]
  int32_t forced_cap;
  // sampler state, per slot
  float* params;        // [slots][5]
  int32_t* history;     // [slots][50]
  int32_t* history_len; // [slots]
  uint64_t* seeds;      // [slots]
  uint64_t* offsets;    // [slots]
  // bucket rows
  int64_t* row_tok;     // [64]
  int64_t* row_pos;     // [64]
  int32_t* row_slot;    // [64]
  int32_t* samp_tok;    // [64]
  // admission rows
  int32_t* adm_slot;    // [16]
  int32_t* adm_tok;     // [16]
  // constants
  const uint32_t* silent_bits;  // [kSilentWords] sub-gate audio ids (bit i = audio id i)
  uint32_t* seq;        // device launch-sequence counter (one per post launch)
  RingEntry* ring;      // device pointer of the mapped host ring
  int32_t ring_n;
  int32_t max_ctx;      // K/V columns per slot
  int32_t ctx_limit;    // production CUDA_GRAPH_MAX_CTX (demo-chunked ceiling)
  int32_t margin;       // TAGGED_SLOT_MARGIN_TOKENS
  int32_t ceiling;      // SILENCE_FREE_BUDGET_CEILING
  int32_t silence_armed;
  // shared voice prefix (Config::share_voice_kv): per slot, the element offset of its voice's K/V in the voice
  // store (-1 = none) and the voice's prefix length
  int64_t* voff;
  int32_t* vlen;
  // ... and the same per bucket row (rowmap copies them with row_slot), so the attention kernel reads them by row
  // index, in parallel with row_slot, instead of after it (one dependent global load less per attention CTA).
  int64_t* row_voff;  // [64]
  int32_t* row_vlen;  // [64]
};

struct AdmInit {
  int32_t n;
  int32_t slot[kAdmRows];
  int32_t prompt_len[kAdmRows];
  int32_t budget[kAdmRows];
  int32_t cap[kAdmRows];
  uint32_t flags[kAdmRows];
  int32_t forced_len[kAdmRows];
  uint64_t seed[kAdmRows];
  float params[kAdmRows][kParamCols];
  int64_t voff[kAdmRows];
  int32_t vlen[kAdmRows];
};

struct RowMap {
  int32_t n;  // live rows
  int32_t b;  // bucket rows (>= n); rows n..b-1 are padding on the scratch slot
  int32_t scratch;
  int32_t slot[kRows];
};

// Launchers (stream-ordered, graph-capturable).
cudaError_t launch_post(const State& st, int rows, const int32_t* row_slot, const int32_t* samp_tok, int update_rows,
                        cudaStream_t stream);
cudaError_t launch_adm_init(const State& st, const AdmInit& a, cudaStream_t stream);
cudaError_t launch_rowmap(const State& st, const RowMap& m, cudaStream_t stream);

}  // namespace kt2n::lm::dev
