// Native MeanFlow estimator (L5): the S3Gen ConditionalDecoder (causal 1-D U-Net, 256 channels,
// 1 down + 12 mid + 1 up stage, each a causal resnet + 4 transformer blocks of 8 heads x 64), the
// IntMeanFlow (t, r) time embedding, and the 2-step mean-flow Euler solve on t = [0, 0.5, 1], no CFG.
//
// Production reference (kittenml-api ebcdd3f): the live chunk cache's estimator path
// (kitten_tts_09/chunk_cache_20260927.py `_core` / `_estimator`, G_CHUNK_FUSED=0, FUSE_QKV=1,
// POOL_ATTN=1) over the S3Gen modules (decoder.py ConditionalDecoder, matcha/transformer.py
// BasicTransformerBlock, flow_matching.py basic_euler). Weights: the `estimator` group of s3gen.kt2n
// (the container format in tools/kt2n_assets/format.py), bf16 round-to-nearest-even of production's fp32 values.
//
// What one call computes, per row (render/ops.h RowDesc):
//   * the row's suffix frames [fr_past, fr_past + fr_suffix) for every flow step, reading the K/V of
//     frames [0, fr_past) from the row's pool pages (render/ops.h PoolView.est, page table) and the
//     two-frame tail of every causal conv from StateView.est_tail (zeros when fr_past == 0);
//   * writes the K/V of every suffix frame to the row's pages (production's index_copy_ of all suffix
//     frames; the open block's K/V are overwritten by the next call), unless the page table points at
//     scratch pages (preview);
//   * saves each conv's tail at the row's new stable end (production `_conv`: shift = new_past - past)
//     unless kRowFinalize / kRowNoPoolWrite;
//   * attention: chunk-causal on absolute frames (50-frame chunks), keys < fr_past + valid where
//     valid = fr_suffix - 2 * lookahead (Q_FLOW_LOOKAHEAD=0: the look-ahead frames are not solved);
//     kRowFullAttention (stream:false whole renders) drops the chunk rule;
//   * Euler: x <- x + (r - t) * v in bf16, as production; the output is x after the last step, fp32.
//
// Layouts (all channels-last, rows packed back to back at RowDesc::fr_off):
//   x0, mu, cond: fp32 [total_frames][80]; spks: fp32 [n_rows][80]; mel_out: fp32 [total_frames][80].
//
// Precision (MathMode::kBf16F32Acc): bf16 weights and activations, fp32 accumulation
// on tensor cores (mma.sync m16n8k16), LayerNorm statistics, softmax and Mish/GELU in fp32, bf16
// rounding at production's module boundaries. Not bit-identical to production (statistical gates).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kt2n/render/ops.h"

struct CUstream_st;

namespace kt2n::render::est {

// Step-time variants: the 2-step solve uses kStep0 (t=0, r=0.5) and kStep1 (t=0.5, r=1); the 1-step
// stream preview uses kPreview (t=0, r=1).
enum class StepTime : int32_t { kStep0 = 0, kStep1 = 1, kPreview = 2 };
inline constexpr int32_t kStepTimes = 3;

struct EstCall {
  int32_t n_rows = 0;           // real rows
  int32_t rows_cap = 0;         // rows bucket (RowDesc entries; padding rows have slot == -1)
  int32_t frames_cap = 0;       // packed suffix frames bucket (>= sum of fr_suffix); multiple of 64
  int32_t max_keys = 0;         // bound on fr_keys over the rows (attention split count; graph key)
  int32_t flow_steps = 2;       // 2, or 1 (preview)
  const RowDesc* rows = nullptr;      // device [rows_cap]
  const int32_t* page_table = nullptr;  // device
  PoolView pool{};
  StateView state{};
  const float* x0 = nullptr;    // device fp32 [frames_cap][80] (noise of the suffix frames)
  const float* mu = nullptr;    // device fp32 [frames_cap][80]
  const float* cond = nullptr;  // device fp32 [frames_cap][80]
  const float* spks = nullptr;  // device fp32 [rows_cap][80]
  float* mel_out = nullptr;     // device fp32 [frames_cap][80]
  float* deriv_out[2] = {nullptr, nullptr};  // optional: per-step derivative, fp32 [frames_cap][80] (checks)
  void* workspace = nullptr;    // device, Estimator::workspace_bytes(rows_cap, frames_cap, max_keys)
};

class Estimator {
 public:
  ~Estimator();
  Estimator(const Estimator&) = delete;
  Estimator& operator=(const Estimator&) = delete;

  // Reads the `estimator` group of s3gen.kt2n, checks names/shapes against the frozen geometry,
  // uploads the weights in the kernels' layouts and precomputes the three step-time embeddings and
  // every resnet's time projection (functions of the weights and the fixed step times only).
  static std::unique_ptr<Estimator> load(const std::string& s3gen_path);

  size_t workspace_bytes(int32_t rows_cap, int32_t frames_cap, int32_t max_keys) const;
  // Bytes of conv tail per (step, slot): StateView::est_tail_width.
  static int32_t tail_width();

  // Enqueue the whole call on `stream` (no host sync, no allocation; graph-capturable). With
  // use_pdl every kernel is launched with programmatic dependent launch.
  void run(const EstCall& call, CUstream_st* stream, bool use_pdl = true) const;

  // Kernels launched per call (for the nsys comparison).
  int32_t kernels_per_call(int32_t flow_steps) const;

  size_t weight_bytes() const;

  // Debug phase trace (only in builds with -DKT2N_EST_TRACE; otherwise both return false / 0).
  // buf: device memory for `cap` records of 48 bytes (kid u32, cta u32, 5 x u64 %globaltimer stamps);
  // nullptr turns tracing off. Resets the record count.
  static bool debug_trace(void* buf, uint32_t cap);
  static uint32_t debug_trace_count();

 private:
  Estimator();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kt2n::render::est
