// L7 render state: paged K/V pool, render-state slots, voice prefixes and per-row page tables.
// Host side only (the device arena is allocated once at startup by src/render/render_call.cu and
// handed in as a PoolView / StateView). Thread-compatible: only the render dispatcher thread calls it.
//
// Production reference: kitten_tts_09/chunk_cache_20260927.py (ChunkCache :144, _acquire/_load_row,
// _build_prefix / _load_prefix, run :607). Production keeps a fixed slot per row capped at
// G_CHUNK_CAP_TOKENS=1088 and renders longer rows eagerly; here a row owns pages (64 frames = 32 tokens,
// render/ops.h kPageBytes = 16.5 MB, ~0.52 MB per token), so any row up to kMaxRowTokens renders on the
// same path and memory is spent per page actually used.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "kt2n/render/ops.h"

namespace kt2n::render {

// Pages needed to hold `frames` frames.
constexpr int32_t pages_for_frames(int32_t frames) { return (frames + kPageFrames - 1) / kPageFrames; }
// stable_prefix_tokens (acoustic_prefix.py:47): whole 25-token blocks whose look-ahead stays in the prompt.
constexpr int32_t stable_prefix_tokens(int32_t prompt_tokens) {
  return prompt_tokens < kLookahead ? 0 : (prompt_tokens - kLookahead) / kTokenChunk * kTokenChunk;
}

// Free-list page allocator with reference counts (shared voice-prefix pages). LIFO reuse.
class HostPagePool final : public PagePool {
 public:
  explicit HostPagePool(PoolView view);
  int32_t pages_total() const override { return view_.n_pages; }
  int32_t pages_free() const override { return int32_t(free_.size()); }
  bool alloc(int32_t n, int32_t* out_pages) override;
  void retain(const int32_t* pages, int32_t n) override;
  void release(const int32_t* pages, int32_t n) override;
  PoolView view() const override { return view_; }
  int32_t refcount(int32_t page) const { return refs_[size_t(page)]; }

 private:
  PoolView view_;
  std::vector<int32_t> free_;
  std::vector<int32_t> refs_;
};

// Render-state slot allocator (StateView rows).
class SlotPool {
 public:
  explicit SlotPool(int32_t slots);
  int32_t acquire();  // -1 when none free (admission back-pressure)
  void release(int32_t slot);
  int32_t free_count() const { return int32_t(free_.size()); }

 private:
  std::vector<int32_t> free_;
  std::vector<uint8_t> used_;
};

// A voice's prompt geometry and its prefix in the pool (VoiceView row).
struct VoicePrefix {
  int32_t voice = -1;
  int32_t prompt_tokens = 0;
  int32_t prompt_frames = 0;
  int32_t stable_tokens = 0;          // pool K/V final for tokens [0, stable)
  std::vector<int32_t> prompt_ids;    // the decoder prompt's speech tokens (suffixes start inside it)
  std::vector<int32_t> shared_pages;  // full pages covering frames [0, 64 * n) (read-only, retained)
  int32_t partial_page = -1;          // page holding frames [64 * n, 2 * stable) (copied into the row's
                                      // first private page at admission), -1 if 2 * stable is page-aligned
  bool built = false;
};

// One segment row's pool pages and render position.
class RowPages {
 public:
  RowPages() = default;
  // Start a row on `voice` (prefix pages retained; a partial page is NOT shared: the row gets its own
  // copy target, see needs_prefix_copy()). Without a built prefix the row starts at position 0.
  void admit(HostPagePool& pool, const VoicePrefix* prefix);
  // Make sure pages cover `frames` frames (all-or-nothing). False = back-pressure, nothing changed.
  bool ensure_frames(HostPagePool& pool, int32_t frames);
  void release(HostPagePool& pool);

  int32_t tok_past() const { return tok_past_; }  // chunk_past: tokens whose K/V are final in the pool
  void set_tok_past(int32_t t) { tok_past_ = t; }
  const std::vector<int32_t>& pages() const { return pages_; }
  int32_t n_shared() const { return n_shared_; }
  bool needs_prefix_copy() const { return copy_src_ >= 0 && !copied_; }
  int32_t prefix_copy_src() const { return copy_src_; }
  void mark_prefix_copied() { copied_ = true; }

 private:
  std::vector<int32_t> pages_;
  int32_t n_shared_ = 0;
  int32_t tok_past_ = 0;
  int32_t copy_src_ = -1;
  bool copied_ = false;
};

// Exact upper bound of the pool pages a row of `tokens` LM ids allocates over its life (admission reservation):
//   * streaming row: its private pages grow to cover 2 x (prompt + tokens + 3) frames (the final renders
//     ids + 3 silence tokens; a commit renders at most ids[:n] + 3 look-ahead), minus the voice prefix's shared
//     full pages it retains instead of allocating (the prefix's partial page is COPIED into a private page,
//     so it counts);
//   * demo row (stream:false whole render, or a preview): scratch pages for the same span from position 0,
//     nothing shared.
// `prefix` may be null or unbuilt (the row then starts at position 0 and shares nothing).
int32_t row_pages_needed(const VoicePrefix* prefix, int32_t prompt_tokens, int32_t tokens, bool demo);

// Bytes for a pool of `pages` pages (device arena sizing) and pages that fit in `bytes`.
constexpr size_t pool_bytes(int32_t pages) { return size_t(pages) * kPageBytes; }
constexpr int32_t pages_in_bytes(size_t bytes) { return int32_t(bytes / kPageBytes); }

// StateView layout helpers (sizes in elements per slot).
struct StateSizes {
  int32_t slots = 0;
  int32_t noise_cap = 2 * kMaxRowTokens;  // frames
  int32_t est_tail_width = 0;             // from the estimator lane (sum over 31 convs of 2 x in_ch)
  size_t target_noise_floats() const { return size_t(slots) * kMel * size_t(noise_cap); }
  size_t bytes() const;
};

}  // namespace kt2n::render
