// Render geometry and publication rules, independent of launch timing.
// Shared and single-request rows use token widths 4,5,5,10,15,25,50, repeating the last width.
// Each commit renders through its boundary plus three look-ahead tokens, then crops those tokens.
// The first publication trims/fades its head and holds a 480-sample splice; later commits hold
// 3840 samples and skip any tail already published. Final flushes append three silence tokens
// with ID 4299 and publish the remaining samples.
// Coalescing waits for partners within 12 tokens when at least 0.3 seconds are buffered.
// A preview consumes eight tokens and publishes five. Opening-silence removal is capped at
// 48 tokens, retaining one context token and waiting for two speech tokens before closing.
#pragma once

#include <cstdint>
#include <vector>

#include "kt2n/render/ops.h"

namespace kt2n::render {

struct CommitPlanConfig {
  std::vector<int32_t> shared_schedule{4, 5, 5, 10, 15, 25, 50};
  std::vector<int32_t> alone_schedule{4, 5, 5, 10, 15, 25, 50};
  int32_t decide_at_tokens = 4;
  int32_t lookahead = kLookahead;
  int32_t tail_fade = kTailFadeSamples;
  bool first_commit_tail = true;
  bool short_first_commit = true;
  int32_t coalesce_tokens = 12;
  double coalesce_min_reservoir_s = 0.3;
  bool preview_enabled = true;
  int32_t preview_tokens = kPreviewTokens;
};

// What one render of one row covers and publishes. All counts are host-known before the launch,
// so descriptors and the PCM packing are built without reading anything back from the GPU.
struct RenderGeometry {
  Phase phase = Phase::kSteadyCommit;
  bool finalize = false;
  int32_t commit_index = 0;      // the row's commit index at launch (0 = first publication of the segment)
  int32_t render_tokens = 0;     // S3Gen tokens the flow sees (ids[:render_tokens], + 3 x 4299 on finals)
  int32_t silence_tokens = 0;    // 3 on finals / whole renders (appended 4299s, inside render_tokens)
  int32_t lookahead = 0;         // trailing tokens cropped (0 when finalize)
  int32_t committed_before = 0;  // tokens already committed (published audio covers them)
  int32_t committed_after = 0;   // == render_tokens - lookahead
  int32_t new_frames = 0;        // mel frames this call publishes (before HiFT caches): 2 x (after - before)
  int32_t mel_cache_in = 0;      // HiFT mel-cache frames carried in (0 on a first publication)
  int32_t src_cache_in = 0;      // HiFT source-cache samples carried in (== held tail length)
  int32_t waveform_samples = 0;  // HiFT output: (mel_cache_in + new_frames) x 480
  int32_t publish_skip = 0;      // head samples already played by the previous first-commit tail
  int32_t tail_fade = 0;         // splice length (linear) used with publish_skip
  int32_t pcm_samples = 0;       // what this call publishes
  bool trim_fade_head = false;   // first publication of the segment: s3gen.trim_fade on the head
  bool crossfade_head = false;   // later publication: Hamming cross-fade with the held tail
  bool short_crossfade = false;  // held tail is < 3840 samples (whole Hamming of 2 x overlap)
  bool tail_published = false;   // first commit publishes its held tail too (all but tail_fade)
  int32_t held_tail_after = 0;   // samples held back after this call (0 on finals)
};

// One segment row's commit bookkeeping (production's IncrementalState fields that decide geometry).
class RowCommitState {
 public:
  explicit RowCommitState(const CommitPlanConfig* cfg);

  // Called on the first turn the row is passed to the acoustic step holding >= decide_at_tokens
  // S3Gen tokens; `sharing` = startup_sharing() read on that turn. Idempotent after the first call.
  void decide_alone(int32_t tokens_held, int32_t sharing);
  bool decided() const { return decided_; }
  bool alone() const { return alone_; }

  // ready(): no launch in flight and tokens >= next + look-ahead.
  bool ready(int32_t tokens_held) const;
  // Tokens still needed before ready (coalescing partner test); may be <= 0.
  int32_t tokens_needed(int32_t tokens_held) const { return next_commit_tokens_ + cfg_->lookahead - tokens_held; }

  // Geometry of the next commit (requires ready()). Does not change state.
  RenderGeometry plan_commit() const { return plan_commit_at(next_commit_tokens_); }
  // Geometry of a commit to an explicit boundary `end` (the conductor's next_commit_tokens).
  RenderGeometry plan_commit_at(int32_t end) const;
  // Geometry of the final flush of a row holding `tokens_held` S3Gen tokens (+3 x 4299, finalize).
  RenderGeometry plan_final(int32_t tokens_held) const;
  // Advance committed / next / tail when the render is queued, before the GPU runs.
  void commit_launched(const RenderGeometry& g);

  int32_t committed_tokens() const { return committed_tokens_; }
  int32_t next_commit_tokens() const { return next_commit_tokens_; }
  int32_t commit_index() const { return commit_index_; }
  int32_t held_tail() const { return held_tail_; }
  int32_t mel_cache() const { return mel_cache_; }
  bool has_published() const { return has_published_; }
  bool inflight() const { return inflight_; }
  void set_inflight(bool v) { inflight_ = v; }

 private:
  int32_t width_after(int32_t commit_index_after) const;

  const CommitPlanConfig* cfg_;
  bool decided_ = false;
  bool alone_ = false;
  bool inflight_ = false;
  bool has_published_ = false;   // speech_tail is not None (a publication exists)
  int32_t committed_tokens_ = 0;
  int32_t next_commit_tokens_ = 0;
  int32_t commit_index_ = 0;
  int32_t held_tail_ = 0;              // len(speech_tail) (== source cache samples)
  int32_t mel_cache_ = 0;              // mel cache frames
  int32_t published_tail_samples_ = 0; // already played part of the held tail (first-commit tail)
};

// Whole render (stream:false DEMO "s2" row): ids + 3 x 4299, finalize, full attention, from position 0,
// trim_fade head, everything published.
RenderGeometry plan_whole_render(int32_t tokens);

// 1-step stream preview (Phase::kPreview): the first `preview_tokens` tokens, finalize=False (3 cropped),
// publishes (preview_tokens - 3) x 960 samples with trim_fade head and a 480-sample linear fade-out.
RenderGeometry plan_preview(const CommitPlanConfig& cfg);

// may this row render a preview on this turn?
struct PreviewInputs {
  bool preview_ready = false;      // the row already published (preview or commit)
  int32_t tokens_held = 0;         // S3Gen tokens
  int32_t committed_tokens = 0;
  int32_t first_commit_tokens = 0; // first scheduled commit width
  bool held_short_or_demo = false; // short-mode / DEMO held row
};
bool preview_eligible(const CommitPlanConfig& cfg, const PreviewInputs& in);

// coalescing, for ONE group of ready rows. `partner_needed` = the smallest tokens_needed
// in (0, coalesce_tokens] over NOT-ready rows of the same group key (no launch in flight, no pending
// preview, not cancelled), or <= 0 when there is none. reservoir_s < 0 means "no estimate" (None).
struct CoalesceRow {
  int32_t commit_index = 0;
  int32_t tokens_held = 0;
  int32_t next_commit_tokens = 0;
  double reservoir_s = -1.0;
};
bool coalesce_group_waits(const CommitPlanConfig& cfg, const CoalesceRow* rows, int32_t n_rows,
                          int32_t partner_needed);

// Opening-silence skip: the LM's audio ids -> the ids S3Gen sees (segment 0, not DEMO).
struct OpeningSkipConfig {
  bool enabled = true;
  int32_t max_tokens = 48;
  int32_t keep_tokens = 1;
  int32_t run_tokens = 2;
  int32_t strict_after = 0;
};
// Opening silence uses the 93-id sub-gate set minus {1466, 3653}.
bool opening_silence_id(int32_t audio_id);

class OpeningSkip {
 public:
  OpeningSkip(const OpeningSkipConfig& cfg, bool armed);
  // Admit one sampled audio id (0..6560); appends what S3Gen may see to `out`.
  void admit(int32_t audio_id, std::vector<int32_t>& out);
  // the whole generation was opening -> hand everything back (skipped then pending).
  void restore(std::vector<int32_t>& out);
  bool closed() const { return closed_; }
  int32_t skipped() const { return int32_t(skipped_.size()); }
  int32_t kept() const { return kept_; }

 private:
  OpeningSkipConfig cfg_;
  bool closed_;
  int32_t kept_ = 0;
  std::vector<int32_t> skipped_;
  std::vector<int32_t> pending_;
};

}  // namespace kt2n::render
