// One streamed response's host-audio path, in production's exact order
// (kittenml-api@ebcdd3f runtime.py ``generate_stream`` + the incremental
// branch of ``_generate_stream_with_ref``, default settings):
//
//   per engine chunk of segment i:
//     seam trimmer (if q_dc_seam_trim) -> kept-token withhold + onset trim
//     -> (initial reservoir: 0 live) -> atempo stretch (speed != 1) -> publish
//   end of segment i:
//     seam.finish(trim_tail = i > 0 or has_more) -> trim -> stretch -> publish
//     last segment: onset trimmer flush -> stretch -> publish
//     has_more: 0.16 s of zeros -> onset trimmer -> stretch -> publish
//     last segment: stretcher flush -> publish
//
// Then (not in this class, as in production): hold_startup (StartupHold) ->
// normalize_audio -> PeakNorm.process -> PCM16 / WAV / encoder. The outer
// ``_publish_trimmed`` gate is a pass-through in this branch (the trimmer has
// already disarmed when anything reaches it) and the internal-silence limiter
// and the Q shaper are off live, so neither appears here.
#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "kt2n/post/atempo.h"
#include "kt2n/post/post.h"

namespace kt2n::post {

struct ResponsePostConfig {
  double speed = 1.0;
  bool onset_trim = true;   // KITTENTTS_09_ONSET_TRIM=1 and not a DEMO request
  bool withhold = true;     // N_WITHHOLD_KEPT_TOKEN_AUDIO=1 (applied at speed 1 only)
  OnsetPolicy onset{};
  double withhold_floor_ms = 20.0;
  double gap_seconds = 0.16;  // CHUNK_GAP_SECONDS
  int sample_rate = kSampleRate;
};

class ResponsePost {
 public:
  explicit ResponsePost(const ResponsePostConfig& cfg = ResponsePostConfig{});
  // control.n_kept_opening_tokens (the opening-silence skip's kept count), known
  // before any audio exists. Ignored when the withhold is not active.
  void set_kept_opening_tokens(int kept);
  // Segment `index` starts; `seam` describes the request/lane now.
  void begin_segment(int index, const SeamContext& seam);
  // One engine chunk of the current segment -> audio to publish now (maybe empty).
  Pcm push(Samples chunk);
  // The current segment's audio is complete; `has_more`: another segment follows.
  // Returns the publications in production's order (each one is a separate
  // yield there, and the startup hold evaluates its target once per publication):
  // seam tail, onset-trimmer flush (last segment), gap, stretcher flush (last).
  // Empty publications are omitted.
  std::vector<Pcm> end_segment(bool has_more);
  // After the last segment (production's belt-and-braces stretcher flush).
  Pcm finish();

  // _trimmed_ms_so_far for the startup hold (nullopt: no onset trimmer).
  std::optional<double> trimmed_ms() const;
  int64_t dropped_samples() const;
  bool stretching() const noexcept { return stretcher_ != nullptr; }
  // The current segment's seam trimmer (nullptr: none for this segment). Read after end_segment() for
  // production's api.seam_trimmed note (runtime.py _dc_note_seam: seam.stats() after seam.finish). Added by L8.
  const SeamEdgeTrimmer* seam() const noexcept { return seam_.get(); }

 private:
  Pcm trim_and_stretch(Pcm chunk);
  Pcm stretch(Pcm chunk);

  ResponsePostConfig cfg_;
  std::unique_ptr<LeadingSilenceTrimmer> trimmer_;
  std::unique_ptr<KeptTokenWithholder> withholder_;
  std::unique_ptr<SpeedStretcher> stretcher_;
  std::unique_ptr<SeamEdgeTrimmer> seam_;
  int index_ = -1;
};

}  // namespace kt2n::post
