// Adapted from FFmpeg libavfilter/af_atempo.c through a prior Python port.
// Upstream copyright (c) 2012 Pavel Koshevoy; LGPL-2.1-or-later.
// See native/NOTICE and native/licenses/FFmpeg-LGPL-2.1.txt.
// KittenML adaptation, 2026: mono C++ streaming and staged speed changes.
// ffmpeg's atempo (WSOLA: Hann-feathered fragments, cross-correlation
// alignment within +-window/2, ffmpeg's drift term and weighting) for mono
// float32, fed in any chunking (output does not depend on the split), plus
// SpeedStretcher (equal stages within 0.5..2.0; exact_end = feed silence, cut
// to round(samples_in / speed)).
//
// Numerics: blend in float64, rounded to float32 once per feed, like the Python.
// The alignment correlation is a plain float64 dot product per lag; the Python
// uses np.correlate, which sums through BLAS ddot in a CPU-dependent order. The
// only possible difference is a near-tie of the weighted argmax (the Python
// itself can flip such a tie between CPUs).
#pragma once

#include <cstdint>
#include <vector>

#include "kt2n/post/post.h"

namespace kt2n::post {

class ATempo {
 public:
  explicit ATempo(double tempo, int sample_rate = kSampleRate);
  Pcm feed(Samples chunk);
  Pcm flush();
  int window() const noexcept { return window_; }
  bool flushed() const noexcept { return flushed_; }
  const std::vector<double>& hann() const noexcept { return hann_; }

 private:
  struct Fragment {
    int64_t pos0 = 0;
    int64_t pos1 = 0;
    int64_t nsamples = 0;
    std::vector<double> data;   // window long, zero past nsamples
    std::vector<double> wdata;  // data * hann (valid when has_wdata)
    bool has_wdata = false;
  };
  enum class State { kLoad, kAdjust, kReload, kOverlapAdd, kFlush };

  Fragment& curr() { return frags_[nfrag_ % 2]; }
  Fragment& prev() { return frags_[(nfrag_ + 1) % 2]; }
  bool load_frag(bool flushing);
  int64_t align(const Fragment& frag, const Fragment& prev, int64_t drift) const;
  int64_t adjust_position();
  void overlap_add(std::vector<double>& out);
  void advance();
  void apply(std::vector<double>& out);
  void trim_history();
  Pcm emit(const std::vector<double>& out);

  int window_;
  int half_;
  int ring_;
  double tempo_;
  std::vector<double> hann_;
  std::vector<double> buf_;  // input history: samples [base_, base_ + buf_.size())
  int64_t base_ = 0;
  int64_t avail_ = 0;
  int64_t loaded_ = 0;
  int64_t out_pos_ = 0;
  int64_t nfrag_ = 0;
  State state_ = State::kLoad;
  Fragment frags_[2];
  bool flushed_ = false;
};

// The fewest equal ffmpeg stages whose tempo lies in 0.5..2.0.
std::vector<double> stage_tempos(double speed);

class SpeedStretcher {
 public:
  explicit SpeedStretcher(double speed, int sample_rate = kSampleRate, bool exact_end = true);
  Pcm feed(Samples chunk);
  Pcm flush();
  bool flushed() const noexcept { return flushed_; }
  int64_t samples_in() const noexcept { return samples_in_; }
  int64_t samples_out() const noexcept { return samples_out_; }
  size_t stages() const noexcept { return stages_.size(); }

 private:
  double speed_;
  std::vector<double> tempos_;
  std::vector<ATempo> stages_;
  bool exact_end_;
  int64_t samples_in_ = 0;
  int64_t samples_out_ = 0;
  bool flushed_ = false;
};

// Production builds a stretcher only when |speed - 1| >= 1e-6 and the method is
// atempo (live KITTENTTS_09_SPEED_STRETCH=atempo).
inline bool speed_stretch_continuous(double speed) {
  return (speed - 1.0 >= 0 ? speed - 1.0 : 1.0 - speed) >= 1e-6;
}

}  // namespace kt2n::post
