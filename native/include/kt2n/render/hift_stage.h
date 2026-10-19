// L7: the L6 HiFT library (include/kt2n/render/hift.h, C ABI) as the render call's post-graph stage.
//
// Per call, after the flow graph and on the same stream:
//   * rows are grouped by vocoder width T = mel cache carried in + new frames (L6 renders equal-width
//     groups; production groups the same way);
//   * a pack kernel builds each group's input [B][80][T] = the slot's mel cache ([80][mc]) followed by the
//     row's new mel frames from CallDesc::mel_out;
//   * per row a Kt2nHiftRow from L6's planner (kt2n_hift_plan, the same rules as is3 launch) with the seed
//     3,000,000 + seed x 10,000 + committed tokens (RowDesc::philox_seed), the host mt19937 phases, the
//     slot's caches (ping-pong per HOLD call, owned here) and `out` = the call's packed PCM at pcm_off;
//   * kt2n_hift_run_rows per group (one HiFT handle per stream role: the library keeps per-handle scratch).
// Weights: the `hift` group of s3gen.kt2n (weight norm folded by the converter), in L6's 319-tensor order
// (spikes/m3-render/prod_render.py hift_weight_list; stride-1 convs transposed to [cin][K][cout], Snake's
// 1 / (alpha + 1e-9) precomputed, the periodic Hann(16) STFT window).
// CUDA builds only (src/render/hift_stage.cu).
#pragma once

#include <memory>
#include <string>

#include "kt2n/render/call.h"

namespace kt2n::render {

// One HiFT handle's capacity: max_t = widest row (frames), max_frame_rows = B x T bound of one launch (L6 workspace
// ~750 KB per frame-row: f0 / noise / sines / 13 upsampled chains / 15 source chains, hift.cu create2).
struct HiftRoleSize {
  int32_t max_t = 1024;
  int32_t max_frame_rows = 2048;
  // L6's per-handle work policy (kt2n_hift_set_policy): convs with >= tc_min_k x 1024 outputs per call use the tensor
  // cores; -1 = L6's default (128, tuned for first-commit latency).
  int32_t tc_min_k = -1;
};

class HiftStage {
 public:
  // Per-role sizes: S_urgent only renders commits 0-1 (T <= 8 cached + 2 x 13 frames); S_bulk renders finals and
  // whole renders (stream:false), whose T = 2 x (tokens + 3) must fit in ONE launch (HiFT's source phase is a
  // running sum over the whole row, so a row cannot be split in time without changing it).
  // bulk_ctx: S_bulk's green context when RenderCallerConfig::bulk_sms > 0 (the bulk handle's streams and events
  // must belong to it); nullptr = primary context.
  // aux: the S_aux handle (RenderCallerConfig::aux_max_rows > 0); max_t 0 = no aux role.
  static std::unique_ptr<HiftStage> load(const std::string& s3gen_path, int32_t slots, int32_t max_b, HiftRoleSize urgent,
                                         HiftRoleSize bulk, std::string* err, CUctx_st* bulk_ctx = nullptr,
                                         HiftRoleSize aux = HiftRoleSize{0, 0});
  // Widest row (frames) the role's handle renders.
  int32_t max_width(int32_t role) const;
  // max_frame_rows bounds B x T of one HiFT launch (L6 workspace, ~600 KB per frame-row); groups are split to fit. max_t must
  // cover the longest row (a whole render of a 4096-token row: 2 x 4099 frames).
  static std::unique_ptr<HiftStage> load(const std::string& s3gen_path, int32_t slots, int32_t max_b, int32_t max_t,
                                         int32_t max_frame_rows, std::string* err);
  ~HiftStage();
  HiftStage(const HiftStage&) = delete;
  HiftStage& operator=(const HiftStage&) = delete;
  RenderPostStage post_stage();
  // Rows whose planner output disagreed with the L7 geometry (pcm count): should stay 0.
  int64_t plan_mismatches() const;

  // Graph mode (default OFF since 12:10; env KT2N_HIFT_GRAPH=1 turns it on). Measured in one process with the
  // arms interleaved in the flow benchmark: the
  // same PCM bitwise, host launch 0.2-0.5 ms lower per call, but GPU time never lower and ~1.4 ms HIGHER at B10
  // (commit 0 and 50-token commits), so eager is the default (GPU time is what limits B10/B20 throughput).
  // In graph mode each group's pack kernel + L6 run is one
  // captured CUDA graph per (stream role, B, T), launched on the role's stream. Two effects:
  //   * one launch instead of ~90 (L6: 88 kernels per call), and
  //   * every HiFT kernel runs at the PRIORITY of the role's stream. Eagerly, L6's three side streams
  //     (f0/source chain, resblock branches, noise) are created at the lowest priority, so an urgent
  //     first commit's HiFT competed with bulk renders; a graph launched into S_urgent runs all its nodes
  //     at S_urgent's priority (cudaGraphInstantiate without UseNodePriority).
  // A shape is captured on its SECOND use (finals have one-off widths; capturing those would cost more than
  // it saves), or at prewarm(). Row / pack descriptors live at fixed device addresses per role; they are
  // refreshed by H2D copies on the same stream just before each launch, so stream order keeps them safe.
  void set_graph_mode(bool on);
  // Capture the shapes production's alone schedule hits first (urgent: commit 0 = 1 x 8, commit 1 = 1 x 18,
  // shared commit 0 = b x 10; bulk: 1 x {18, 28, 38, 58, 108}). Capture only records; nothing runs.
  void prewarm(const RenderCaller& caller, int32_t max_shared_rows);
  // Run L6 once on scratch rows per shape class (both roles, small and large widths) so CUDA's lazy module
  // loading happens at startup instead of inside the first live call (5+ ms of the first commit 0 otherwise).
  void warm(const RenderCaller& caller);
  struct GraphStats {
    int64_t graph_launches = 0, eager_launches = 0, captures = 0;
    int32_t cached = 0;
  };
  GraphStats graph_stats() const;

  struct Impl;

 private:
  HiftStage();
  std::unique_ptr<Impl> impl_;
};

}  // namespace kt2n::render
