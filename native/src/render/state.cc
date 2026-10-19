// Paged K/V pool, slots and row pages (include/kt2n/render/state.h).
#include "kt2n/render/state.h"

#include <algorithm>
#include <stdexcept>

namespace kt2n::render {

HostPagePool::HostPagePool(PoolView view) : view_(view) {
  if (view_.n_pages < 0) throw std::invalid_argument("negative page count");
  refs_.assign(size_t(view_.n_pages), 0);
  free_.reserve(size_t(view_.n_pages));
  for (int32_t p = view_.n_pages - 1; p >= 0; --p) free_.push_back(p);  // pop_back hands out 0, 1, 2, ...
}

bool HostPagePool::alloc(int32_t n, int32_t* out_pages) {
  if (n < 0) throw std::invalid_argument("alloc: negative count");
  if (int32_t(free_.size()) < n) return false;
  for (int32_t i = 0; i < n; ++i) {
    const int32_t p = free_.back();
    free_.pop_back();
    refs_[size_t(p)] = 1;
    out_pages[i] = p;
  }
  return true;
}

void HostPagePool::retain(const int32_t* pages, int32_t n) {
  for (int32_t i = 0; i < n; ++i) {
    const int32_t p = pages[i];
    if (p < 0 || p >= view_.n_pages || refs_[size_t(p)] <= 0) throw std::logic_error("retain of a free page");
    ++refs_[size_t(p)];
  }
}

void HostPagePool::release(const int32_t* pages, int32_t n) {
  for (int32_t i = 0; i < n; ++i) {
    const int32_t p = pages[i];
    if (p < 0 || p >= view_.n_pages || refs_[size_t(p)] <= 0) throw std::logic_error("release of a free page");
    if (--refs_[size_t(p)] == 0) free_.push_back(p);
  }
}

SlotPool::SlotPool(int32_t slots) {
  used_.assign(size_t(slots), 0);
  for (int32_t s = slots - 1; s >= 0; --s) free_.push_back(s);
}

int32_t SlotPool::acquire() {
  if (free_.empty()) return -1;
  const int32_t s = free_.back();
  free_.pop_back();
  used_[size_t(s)] = 1;
  return s;
}

void SlotPool::release(int32_t slot) {
  if (slot < 0 || size_t(slot) >= used_.size() || !used_[size_t(slot)]) throw std::logic_error("release of a free slot");
  used_[size_t(slot)] = 0;
  free_.push_back(slot);
}

void RowPages::admit(HostPagePool& pool, const VoicePrefix* prefix) {
  if (!pages_.empty()) throw std::logic_error("RowPages::admit on a row that holds pages");
  n_shared_ = 0;
  tok_past_ = 0;
  copy_src_ = -1;
  copied_ = false;
  if (prefix != nullptr && prefix->built && prefix->stable_tokens > 0) {
    n_shared_ = int32_t(prefix->shared_pages.size());
    if (n_shared_ > 0) {
      pool.retain(prefix->shared_pages.data(), n_shared_);
      pages_.assign(prefix->shared_pages.begin(), prefix->shared_pages.end());
    }
    copy_src_ = prefix->partial_page;
    tok_past_ = prefix->stable_tokens;
  }
}

bool RowPages::ensure_frames(HostPagePool& pool, int32_t frames) {
  const int32_t need = pages_for_frames(frames) - int32_t(pages_.size());
  if (need <= 0) return true;
  const size_t old = pages_.size();
  pages_.resize(old + size_t(need));
  if (!pool.alloc(need, pages_.data() + old)) {
    pages_.resize(old);
    return false;
  }
  return true;
}

void RowPages::release(HostPagePool& pool) {
  if (!pages_.empty()) pool.release(pages_.data(), int32_t(pages_.size()));
  pages_.clear();
  n_shared_ = 0;
  tok_past_ = 0;
  copy_src_ = -1;
  copied_ = false;
}

int32_t row_pages_needed(const VoicePrefix* prefix, int32_t prompt_tokens, int32_t tokens, bool demo) {
  const int32_t span = pages_for_frames(kMelPerToken * (prompt_tokens + tokens + kLookahead));
  if (demo || prefix == nullptr || !prefix->built) return span;
  return span - int32_t(prefix->shared_pages.size());
}

size_t StateSizes::bytes() const {
  const size_t s = size_t(slots);
  size_t b = target_noise_floats() * sizeof(float);
  b += s * 2 * kEncDim * sizeof(float) * 2;                       // enc_pre_tail + enc_up_tail
  b += size_t(kFlowSteps) * s * size_t(est_tail_width) * 2;       // est_tail (bf16)
  b += s * kMel * kMelCacheFrames * sizeof(float);                // hift_mel_cache
  b += s * kSourceCacheSamples * sizeof(float) * 2;               // hift_src_cache + hift_held_tail
  b += s * kHarmonics * sizeof(float);                            // hift_phase
  return b;
}

}  // namespace kt2n::render
