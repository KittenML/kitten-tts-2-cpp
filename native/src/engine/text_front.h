// L8: the text front end in production's order:
//   decide_request -> L9a prepare -> L9b plan_segments -> per planned segment L9a
//   normalize_planned_segment -> L9b resplit_segment -> per LM chunk target_suffix + budget.
// L9a (NeMo glue on OpenFst) is compiled in only with KT2N_BUILD_TEXT; without it the front end
// runs a PASSTHROUGH normaliser (text unchanged) and says so (real_normalizer() == false): that
// mode exists for plumbing tests and is never the RC.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "kt2n/textplan.h"

namespace kt2n::engine {

struct PlannedRequest {
  textplan::RequestText decision;
  std::string language;
  std::shared_ptr<void> prepared;          // L9a PreparedText (opaque here)
  std::vector<std::string> raw_segments;   // planned segments (prepared text), UTF-8
};

class TextFront {
 public:
  // text_data_dir: NeMo grammar dir (tagger.fst, verbalizer.fst, post_processor.fst, *_words.txt) plus
  // nemo_whitelist.txt (L9b abbreviations). NULL = passthrough. tokenizer_json NULL = code points as ids
  // (stub LM only).
  static std::unique_ptr<TextFront> create(const char* text_data_dir, const char* tokenizer_json, int instances,
                                           std::string* err);
  virtual ~TextFront() = default;
  // emotion_preset: a WS phrase's turn stamp (EmotionTurn::stamp), which wins over the phrase text.
  virtual PlannedRequest plan(const std::string& raw, const std::string& language,
                              std::optional<bool> emotion_preset = std::nullopt, bool normalize = true) = 0;
  // Planned segment `index` -> its LM chunks (normalized, re-split).
  virtual std::vector<std::u32string> normalize_segment(const PlannedRequest& p, size_t index) = 0;
  virtual std::vector<int32_t> suffix(const std::u32string& chunk, bool emotion) const = 0;
  // tokenizer.encode(text, add_special_tokens=False) (clone transcripts).
  virtual std::vector<int32_t> encode(const std::string& utf8) const = 0;
  virtual bool real_normalizer() const = 0;
  virtual bool real_tokenizer() const = 0;
};

}  // namespace kt2n::engine
