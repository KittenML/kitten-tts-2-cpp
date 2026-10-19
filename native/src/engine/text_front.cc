// L8: text front end wiring (see text_front.h).
#include "build_flags.h"
#include "text_front.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <mutex>

#if KT2N_ENGINE_WITH_TEXT
#include "incremental.h"
#include "normalizer.h"
#endif

namespace kt2n::engine {
namespace tp = kt2n::textplan;

namespace {
class Front final : public TextFront {
 public:
  PlannedRequest plan(const std::string& raw, const std::string& language,
                      std::optional<bool> emotion_preset, [[maybe_unused]] bool normalize) override {
    PlannedRequest p;
    p.language = language;
    p.decision = tp::decide_request(tp::from_utf8(raw), emotion_preset);
    std::string prepared_text = raw;
    bool wfst = false;
    tp::Emphasis emphasis = p.decision.emphasis;
#if KT2N_ENGINE_WITH_TEXT
    if (normalizer_) {
      auto prepared = std::make_shared<kt2n::text::PreparedText>(normalizer_->prepare(raw, language, normalize));
      prepared_text = prepared->text;
      wfst = prepared->wfst;
      emphasis.clear();
      for (const auto& e : prepared->emphasis) emphasis.push_back(tp::from_utf8(e));
      p.prepared = prepared;
    }
#endif
    const auto planned = tp::plan_segments(tp::from_utf8(prepared_text), wfst, emphasis, abbreviations_);
    for (const auto& s : planned) {
      p.raw_segments.push_back(tp::to_utf8(s));
    }
    return p;
  }

  std::vector<std::u32string> normalize_segment(const PlannedRequest& p, size_t index) override {
    std::string text = p.raw_segments[index];
#if KT2N_ENGINE_WITH_TEXT
    if (normalizer_ && p.prepared) {
      const auto& prepared = *static_cast<const kt2n::text::PreparedText*>(p.prepared.get());
      kt2n::text::IncrementalPolicy policy;
      if (index == 0) {
        kt2n::text::SpeechTextNormalizer::SegmentZeroGuard guard(*normalizer_);
        text = kt2n::text::normalize_planned_segment(*normalizer_, prepared, p.raw_segments, index, policy).text;
      } else {
        text = kt2n::text::normalize_planned_segment(*normalizer_, prepared, p.raw_segments, index, policy).text;
      }
    }
#endif
    tp::Emphasis emphasis = p.decision.emphasis;
#if KT2N_ENGINE_WITH_TEXT
    if (p.prepared) {
      emphasis.clear();
      for (const auto& e : static_cast<const kt2n::text::PreparedText*>(p.prepared.get())->emphasis)
        emphasis.push_back(tp::from_utf8(e));
    }
#endif
    return tp::resplit_segment(tp::from_utf8(text), emphasis);
  }

  std::vector<int32_t> suffix(const std::u32string& chunk, bool emotion) const override {
    if (tokenizer_) return tp::target_suffix(*tokenizer_, chunk, emotion);
    // Stub LM only: code points stand in for token ids.
    std::vector<int32_t> ids{tp::kTextStart};
    for (char32_t c : chunk) ids.push_back(int32_t(c % 150000));
    ids.push_back(tp::kFinalSeg);
    ids.push_back(tp::kSpeechStart);
    return ids;
  }
  std::vector<int32_t> encode(const std::string& utf8) const override {
    if (tokenizer_) return tokenizer_->encode(std::string_view(utf8));
    std::vector<int32_t> ids;
    for (char32_t c : tp::from_utf8(utf8)) ids.push_back(int32_t(c % 150000));
    return ids;
  }
  bool real_normalizer() const override {
#if KT2N_ENGINE_WITH_TEXT
    return normalizer_ != nullptr;
#else
    return false;
#endif
  }
  bool real_tokenizer() const override { return tokenizer_.has_value(); }

#if KT2N_ENGINE_WITH_TEXT
  std::unique_ptr<kt2n::text::SpeechTextNormalizer> normalizer_;
#endif
  std::optional<tp::Tokenizer> tokenizer_;
  tp::Abbreviations abbreviations_ = tp::Abbreviations::from_lines("");
};
}  // namespace

std::unique_ptr<TextFront> TextFront::create(const char* text_data_dir, const char* tokenizer_json, int instances,
                                             std::string* err) {
  auto f = std::make_unique<Front>();
  try {
    if (tokenizer_json) f->tokenizer_ = tp::Tokenizer::load(tokenizer_json);
    if (text_data_dir) {
      const std::string dir = text_data_dir;
      const std::string wl = dir + "/nemo_whitelist.txt";
      if (!std::filesystem::exists(wl)) {
        if (err) *err = "text_data_dir has no nemo_whitelist.txt (L9b abbreviations)";
        return nullptr;
      }
      f->abbreviations_ = tp::Abbreviations::load(wl);
#if KT2N_ENGINE_WITH_TEXT
      kt2n::text::NormalizerConfig cfg;
      cfg.multilingual = std::getenv("KITTENTTS_09_MULTILINGUAL_NORMALIZATION") &&
                         std::string(std::getenv("KITTENTTS_09_MULTILINGUAL_NORMALIZATION")) == "1";
      cfg.grammar_dir = dir;
      cfg.grammar_instances = std::max(1, instances);
      f->normalizer_ = kt2n::text::SpeechTextNormalizer::create(cfg, err);
      if (!f->normalizer_) return nullptr;
#else
      (void)instances;
      if (err) *err = "built without KT2N_BUILD_TEXT: the native normaliser (L9a) is not compiled in";
      return nullptr;
#endif
    }
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return nullptr;
  }
  return f;
}

}  // namespace kt2n::engine
