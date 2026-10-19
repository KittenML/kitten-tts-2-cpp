// L7 render state on the device: one arena allocated at startup (pool pages, per-slot state, voice
// tables), admission uploads (noise tables, voice prefixes) and the L7 kernels:
//   kStateNoiseGather  the flow's initial noise for each row's suffix frames (prompt region from the
//                      voice's numpy prompt noise, target frames from the row's table, zeros past it);
//   kStatePrefixCopy   copy-on-admit of a voice prefix's partial last page into the row's first private
//                      page (all three K/V domains).
// CUDA builds only (src/render/render_call.cu).
#pragma once

#include <cstdint>
#include <vector>

#include "kt2n/render/call.h"
#include "kt2n/render/ops.h"
#include "kt2n/render/state.h"

namespace kt2n::render {

struct ArenaConfig {
  int32_t device = 0;
  int32_t pool_pages = 256;           // 256 x 16.5 MB = 4.2 GB
  int32_t slots = 64;                 // render-state slots (rows admitted at once)
  int32_t noise_cap = 2 * kMaxRowTokens;
  int32_t est_tail_width = 0;         // from the estimator lane; 0 until it lands
  int32_t voices = 64;
  int32_t max_prompt_frames = 600;
  int32_t max_prefix_pages = 16;
};

class RenderArena {
 public:
  explicit RenderArena(const ArenaConfig& cfg);
  ~RenderArena();
  RenderArena(const RenderArena&) = delete;
  RenderArena& operator=(const RenderArena&) = delete;

  const PoolView& pool() const { return pool_; }
  const StateView& state() const { return state_; }
  const VoiceView& voices() const { return voices_; }
  size_t device_bytes() const { return bytes_; }

  // Admission: zero the slot's noise table and upload frames [0, w) (host [80][w] row-major) on `stream`.
  // `host` must stay valid until the copy completes (pinned memory for an async copy).
  void upload_row_noise(int32_t slot, const float* host, int32_t w, CUstream_st* stream);
  // Write frames [w0, w1) of a slot's table (the 5,000,000 + seed extension).
  void upload_row_noise_range(int32_t slot, const float* host, int32_t w0, int32_t w1, CUstream_st* stream);
  // Voice tables (synchronous; startup / voice build): prompt noise [80][frames], prompt tokens,
  // prompt feat [frames][80], speaker conditioning [80].
  // Copies on `stream` (+ a sync of that stream only) when given; nullptr = synchronous copies on the legacy
  // default stream, which also wait for every blocking stream's queued work (e.g. an LM stream).
  void set_voice(int32_t voice, int32_t prompt_frames, const float* prompt_noise, const int32_t* prompt_tokens,
                 const float* prompt_feat, const float* spk_cond, CUstream_st* stream = nullptr);

  // kStatePrefixCopy for one row: page src -> dst in every domain (+ the voice's conv tails into the
  // slot when the tables exist).
  void copy_prefix_page(int32_t src_page, int32_t dst_page, CUstream_st* stream);
  // A voice-prefix build leaves its conv tails in the build slot: save them as the voice's prefix tails,
  // and copy them into a newly admitted row's slot (chunk_cache _load_prefix). Zero a slot's tails and
  // HiFT caches (a row starting at position 0).
  void save_prefix_tails(int32_t slot, int32_t voice, CUstream_st* stream);
  void load_prefix_tails(int32_t voice, int32_t slot, CUstream_st* stream);
  void clear_slot(int32_t slot, CUstream_st* stream);
  const ArenaConfig& config() const { return cfg_; }

 private:
  ArenaConfig cfg_;
  void* base_ = nullptr;
  size_t bytes_ = 0;
  PoolView pool_{};
  StateView state_{};
  VoiceView voices_{};
};

// The L7 noise-gather stage: writes the initial flow noise of every real row's suffix frames to
// CallDesc::mel_out, packed [total_frames][kMel] (frame fr_off + i of row r = absolute frame
// fr_past + i). The estimator lane consumes the same values through kEstInputAssemble; this stage is
// the oracle check and the reference for that op.
RenderStage noise_gather_stage();

// A stand-in for the not-yet-landed encoder / estimator / HiFT stages: writes each real row's
// pcm_count samples (value = float((slot << 16) | (sample & 0xFFFF)), exact) so packing, copies and
// timing can be tested.
RenderStage stub_pcm_stage();

}  // namespace kt2n::render
