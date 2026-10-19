// Production order of one streamed response's host-audio path; see
// include/kt2n/post/response.h. "py:" quotes kittenml-api@ebcdd3f runtime.py.
#include "kt2n/post/response.h"

#include <cmath>

namespace kt2n::post {

ResponsePost::ResponsePost(const ResponsePostConfig& cfg) : cfg_(cfg) {
  if (cfg_.onset_trim) {
    LeadingSilenceTrimmer t(cfg_.onset);
    // py: n_withhold_wrap: only at speed 1.0 (|speed - 1| <= 1e-6)
    if (cfg_.withhold && std::fabs(cfg_.speed - 1.0) <= 1e-6) {
      withholder_ = std::make_unique<KeptTokenWithholder>(std::move(t), cfg_.withhold_floor_ms, cfg_.sample_rate);
    } else {
      trimmer_ = std::make_unique<LeadingSilenceTrimmer>(std::move(t));
    }
  }
  // py: new_speed_stretcher: atempo when |speed - 1| >= 1e-6
  if (speed_stretch_continuous(cfg_.speed)) stretcher_ = std::make_unique<SpeedStretcher>(cfg_.speed, cfg_.sample_rate);
}

void ResponsePost::set_kept_opening_tokens(int kept) {
  if (withholder_) withholder_->set_kept_tokens(kept);
}

std::optional<double> ResponsePost::trimmed_ms() const {
  if (withholder_) return withholder_->trimmed_ms();
  if (trimmer_) return trimmer_->trimmed_ms();
  return std::nullopt;
}

int64_t ResponsePost::dropped_samples() const {
  if (withholder_) return withholder_->dropped_samples();
  if (trimmer_) return trimmer_->dropped_samples();
  return 0;
}

void ResponsePost::begin_segment(int index, const SeamContext& seam) {
  index_ = index;
  seam_.reset();
  // py: seam = SeamEdgeTrimmer(trim_head=...) if q_dc_seam_trim(control, index, lane) else None
  if (seam_trim_applies(seam, index)) seam_ = std::make_unique<SeamEdgeTrimmer>(seam_trim_head(seam, index));
}

Pcm ResponsePost::stretch(Pcm chunk) {
  // py: audio = _time_stretch(ready_chunk, speed) (identity at 1.0) or speed_stretcher.feed(ready_chunk)
  if (!stretcher_) return chunk;
  return stretcher_->feed(chunk);
}

Pcm ResponsePost::trim_and_stretch(Pcm chunk) {
  if (withholder_) {
    chunk = withholder_->feed(chunk);
  } else if (trimmer_) {
    chunk = trimmer_->feed(chunk);
  }
  if (chunk.empty()) return {};
  return stretch(std::move(chunk));
}

Pcm ResponsePost::push(Samples chunk) {
  Pcm c(chunk.begin(), chunk.end());
  if (seam_) {
    c = seam_->feed(c);
    if (c.empty()) return {};
  }
  return trim_and_stretch(std::move(c));
}

std::vector<Pcm> ResponsePost::end_segment(bool has_more) {
  std::vector<Pcm> out;
  auto add = [&out](Pcm p) {
    if (!p.empty()) out.push_back(std::move(p));
  };
  if (seam_ && !seam_->finished()) {
    // py: chunk = seam.finish(trim_tail=index > 0 or has_more()); then the trimmer and the stretch
    Pcm tail = seam_->finish(seam_trim_tail(index_, has_more));
    if (!tail.empty()) add(trim_and_stretch(std::move(tail)));
  }
  if ((withholder_ || trimmer_) && !has_more) {
    // py: held = trimmer.flush() on the last segment only, then stretched and published
    Pcm held = withholder_ ? withholder_->flush() : trimmer_->flush();
    if (!held.empty()) add(stretch(std::move(held)));
  }
  if (has_more) {
    // py: gap = np.zeros(int(SAMPLE_RATE * q_chunk_gap_seconds(control)) - q_credit); trimmer.feed(gap); stretch
    Pcm gap(static_cast<size_t>(chunk_gap_samples(cfg_.gap_seconds, 0, cfg_.sample_rate)), 0.0f);
    if (withholder_) {
      gap = withholder_->feed(gap);
    } else if (trimmer_) {
      gap = trimmer_->feed(gap);
    }
    if (stretcher_ && !gap.empty()) gap = stretcher_->feed(gap);
    add(gap);
  }
  if (stretcher_ && !has_more) add(stretcher_->flush());
  return out;
}

Pcm ResponsePost::finish() {
  if (stretcher_ && !stretcher_->flushed()) return stretcher_->flush();
  return {};
}

}  // namespace kt2n::post
