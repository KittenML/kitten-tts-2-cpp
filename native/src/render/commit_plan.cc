// Production commit geometry (see include/kt2n/render/commit_plan.h for the references).
#include "kt2n/render/commit_plan.h"

#include <algorithm>
#include <stdexcept>

namespace kt2n::render {
namespace {

int32_t scheduled(const std::vector<int32_t>& schedule, int32_t index) {
  // Repeat the final schedule width.
  if (schedule.empty()) throw std::logic_error("empty commit schedule");
  return schedule[size_t(std::min<int32_t>(index, int32_t(schedule.size()) - 1))];
}

}  // namespace

RowCommitState::RowCommitState(const CommitPlanConfig* cfg) : cfg_(cfg) {
  // is3 new_state: schedule set -> first commit = schedule head, taken verbatim (no chunk alignment).
  next_commit_tokens_ = scheduled(cfg_->shared_schedule, 0);
}

void RowCommitState::decide_alone(int32_t tokens_held, int32_t sharing) {
  if (decided_) return;
  if (cfg_->decide_at_tokens > 0 && tokens_held < cfg_->decide_at_tokens) return;  // _d_row: no decision yet
  decided_ = true;
  alone_ = (sharing == 1);
  // _d_prepare_rows: commit 0 of an alone row = D_SCHEDULE_ALONE[0] (not while a launch is in flight).
  if (alone_ && commit_index_ == 0 && !inflight_) next_commit_tokens_ = scheduled(cfg_->alone_schedule, 0);
}

bool RowCommitState::ready(int32_t tokens_held) const {
  return !inflight_ && tokens_held >= next_commit_tokens_ + cfg_->lookahead;
}

int32_t RowCommitState::width_after(int32_t commit_index_after) const {
  // Select the shared or single-request commit schedule.
  return alone_ ? scheduled(cfg_->alone_schedule, commit_index_after)
                : scheduled(cfg_->shared_schedule, commit_index_after);
}

RenderGeometry RowCommitState::plan_commit_at(int32_t end) const {
  if (end <= committed_tokens_) throw std::invalid_argument("commit boundary not past the committed tokens");
  RenderGeometry g;
  g.phase = has_published_ ? Phase::kSteadyCommit : Phase::kFirstCommit;
  g.finalize = false;
  g.commit_index = commit_index_;
  g.lookahead = cfg_->lookahead;
  g.render_tokens = end + cfg_->lookahead;  // ids[:next + 3] exactly
  g.committed_before = committed_tokens_;
  g.committed_after = end;
  g.new_frames = kMelPerToken * (g.committed_after - g.committed_before);
  g.mel_cache_in = mel_cache_;
  g.src_cache_in = held_tail_;
  g.waveform_samples = (g.mel_cache_in + g.new_frames) * kSamplesPerFrame;
  const int32_t w = g.waveform_samples;
  g.trim_fade_head = !has_published_;
  g.crossfade_head = has_published_;
  g.short_crossfade = has_published_ && cfg_->short_first_commit && held_tail_ < kSourceCacheSamples;
  const bool tail_now = cfg_->first_commit_tail && commit_index_ == 0;
  const bool short_ok = cfg_->short_first_commit && commit_index_ == 0 && cfg_->first_commit_tail &&
                        w > 2 * cfg_->tail_fade;
  if (w <= kSourceCacheSamples && !short_ok)
    throw std::runtime_error("incremental waveform shorter than HiFT cache");
  int32_t emitted = tail_now ? w - cfg_->tail_fade : w - kSourceCacheSamples;
  g.tail_published = tail_now;
  g.held_tail_after = std::min(kSourceCacheSamples, w);
  if (published_tail_samples_ > 0 && has_published_) {
    g.publish_skip = published_tail_samples_;
    g.tail_fade = cfg_->tail_fade;
    emitted = emitted <= published_tail_samples_ ? 0 : emitted - published_tail_samples_;
  }
  g.pcm_samples = std::max(0, emitted);
  return g;
}

RenderGeometry RowCommitState::plan_final(int32_t tokens_held) const {
  RenderGeometry g;
  g.phase = Phase::kFinalFlush;
  g.finalize = true;
  g.commit_index = commit_index_;
  g.silence_tokens = 3;
  g.render_tokens = tokens_held + 3;
  g.lookahead = 0;
  g.committed_before = committed_tokens_;
  g.committed_after = g.render_tokens;
  g.new_frames = kMelPerToken * (g.committed_after - g.committed_before);
  g.mel_cache_in = mel_cache_;
  g.src_cache_in = held_tail_;
  g.waveform_samples = (g.mel_cache_in + g.new_frames) * kSamplesPerFrame;
  g.trim_fade_head = !has_published_;
  g.crossfade_head = has_published_;
  g.short_crossfade = has_published_ && cfg_->short_first_commit && held_tail_ < kSourceCacheSamples;
  int32_t emitted = g.waveform_samples;
  if (published_tail_samples_ > 0 && has_published_) {
    g.publish_skip = published_tail_samples_;
    g.tail_fade = cfg_->tail_fade;
    emitted = emitted <= published_tail_samples_ ? 0 : emitted - published_tail_samples_;
  }
  g.pcm_samples = emitted;
  g.held_tail_after = 0;
  return g;
}

void RowCommitState::commit_launched(const RenderGeometry& g) {
  if (g.finalize) {
    committed_tokens_ = g.committed_after;
    held_tail_ = 0;
    published_tail_samples_ = 0;
    has_published_ = true;
    return;
  }
  // is3 launch (non-final): caches, tail, committed, index, next.
  mel_cache_ = std::min(kMelCacheFrames, g.mel_cache_in + g.new_frames);
  held_tail_ = g.held_tail_after;
  published_tail_samples_ = g.tail_published ? held_tail_ - cfg_->tail_fade : 0;
  has_published_ = true;
  committed_tokens_ = g.committed_after;
  commit_index_ += 1;
  next_commit_tokens_ = committed_tokens_ + width_after(commit_index_);
}

RenderGeometry plan_whole_render(int32_t tokens) {
  RenderGeometry g;
  g.phase = Phase::kWholeRender;
  g.finalize = true;
  g.silence_tokens = 3;
  g.render_tokens = tokens + 3;
  g.committed_after = g.render_tokens;
  g.new_frames = kMelPerToken * g.render_tokens;
  g.waveform_samples = g.new_frames * kSamplesPerFrame;
  g.trim_fade_head = true;
  g.pcm_samples = g.waveform_samples;
  return g;
}

RenderGeometry plan_preview(const CommitPlanConfig& cfg) {
  RenderGeometry g;
  g.phase = Phase::kPreview;
  g.finalize = false;
  g.render_tokens = cfg.preview_tokens;
  g.lookahead = kLookahead;
  g.committed_after = cfg.preview_tokens - kLookahead;
  g.new_frames = kMelPerToken * g.committed_after;
  g.waveform_samples = g.new_frames * kSamplesPerFrame;
  g.trim_fade_head = true;
  g.pcm_samples = g.waveform_samples;  // 5 tokens = 4800 samples, 480-sample fade-out at the end
  return g;
}

bool preview_eligible(const CommitPlanConfig& cfg, const PreviewInputs& in) {
  if (!cfg.preview_enabled || in.preview_ready || in.held_short_or_demo) return false;
  if (in.tokens_held < cfg.preview_tokens || in.committed_tokens != 0) return false;
  // A preview is redundant when the first commit already needs at least preview_tokens tokens.
  if (in.first_commit_tokens + cfg.lookahead >= cfg.preview_tokens) return false;
  return true;
}

bool coalesce_group_waits(const CommitPlanConfig& cfg, const CoalesceRow* rows, int32_t n_rows,
                          int32_t partner_needed) {
  if (cfg.coalesce_tokens <= 0 || n_rows <= 0) return false;
  const bool has_partner = partner_needed > 0 && partner_needed <= cfg.coalesce_tokens;
  for (int32_t i = 0; i < n_rows; ++i) {
    const CoalesceRow& r = rows[i];
    if (r.commit_index < 1) return false;  // a first commit never waits
    const int32_t overdue = r.tokens_held - (r.next_commit_tokens + cfg.lookahead);
    if (!has_partner || overdue + partner_needed > cfg.coalesce_tokens) return false;
    if (!(r.reservoir_s >= 0.0) || r.reservoir_s < cfg.coalesce_min_reservoir_s) return false;
  }
  return true;
}

// ---- opening-silence skip -----------------------------------------------------------------------------
namespace {
// runtime.py:871 SUB_GATE_AUDIO_IDS (93 ids, measured on S3Gen sha d65cb687...).
constexpr int32_t kSubGateIds[] = {
    331,  729,  735,  736,  770,  979,  1458, 1463, 1465, 1466, 1490, 1701, 1704, 1788, 1789, 1809,
    1947, 1950, 2005, 2028, 2031, 2109, 2112, 2139, 2933, 2948, 2957, 2960, 3003, 3246, 3247, 3408,
    3653, 3662, 3676, 3677, 3680, 3685, 3704, 3713, 3729, 3732, 3888, 3891, 3894, 3945, 3969, 3972,
    3975, 3976, 4053, 4056, 4057, 4134, 4137, 4138, 4215, 4216, 4218, 4219, 4227, 4296, 4299, 4300,
    4308, 5109, 5190, 5352, 5891, 5900, 5919, 6078, 6081, 6082, 6159, 6162, 6163, 6168, 6243, 6252,
    6321, 6324, 6325, 6333, 6399, 6402, 6405, 6406, 6414, 6483, 6486, 6487, 6495,
};
static_assert(sizeof(kSubGateIds) / sizeof(kSubGateIds[0]) == 93, "SUB_GATE_AUDIO_IDS has 93 ids");
}  // namespace

bool opening_silence_id(int32_t audio_id) {
  if (audio_id == 1466 || audio_id == 3653) return false;  // SUB_GATE_SPEECH_LEVEL_AUDIO_IDS
  return std::binary_search(std::begin(kSubGateIds), std::end(kSubGateIds), audio_id);
}

OpeningSkip::OpeningSkip(const OpeningSkipConfig& cfg, bool armed) : cfg_(cfg), closed_(!(armed && cfg.enabled)) {}

void OpeningSkip::admit(int32_t value, std::vector<int32_t>& out) {
  if (closed_ || !out.empty()) {
    closed_ = true;
    out.push_back(value);
    return;
  }
  const int32_t held = int32_t(skipped_.size() + pending_.size());
  const int32_t cap = cfg_.max_tokens;
  if (opening_silence_id(value) && held < cap) {
    skipped_.insert(skipped_.end(), pending_.begin(), pending_.end());
    pending_.clear();
    skipped_.push_back(value);
    return;
  }
  const bool strict = cfg_.strict_after > 0 && held >= cfg_.strict_after;
  if (held < cap && int32_t(pending_.size()) + 1 < cfg_.run_tokens && !strict) {
    pending_.push_back(value);
    return;
  }
  closed_ = true;
  if (cfg_.keep_tokens > 0 && !skipped_.empty()) {
    const size_t keep = std::min(size_t(cfg_.keep_tokens), skipped_.size());
    out.insert(out.end(), skipped_.end() - ptrdiff_t(keep), skipped_.end());
    kept_ = int32_t(keep);
  }
  out.insert(out.end(), pending_.begin(), pending_.end());
  pending_.clear();
  out.push_back(value);
}

void OpeningSkip::restore(std::vector<int32_t>& out) {
  if (!out.empty() || (skipped_.empty() && pending_.empty())) return;
  out.insert(out.end(), skipped_.begin(), skipped_.end());
  out.insert(out.end(), pending_.begin(), pending_.end());
  skipped_.clear();
  pending_.clear();
}

}  // namespace kt2n::render
