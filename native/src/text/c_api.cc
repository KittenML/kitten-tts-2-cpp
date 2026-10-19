#include "kt2n/text.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "normalizer.h"
#include "unicode.h"

#define KT2N_TEXT_API extern "C" __attribute__((visibility("default")))

struct kt2n_text {
  std::unique_ptr<kt2n::text::SpeechTextNormalizer> normalizer;
};

namespace {
char* dup(const std::string& s, int64_t* len) {
  char* p = static_cast<char*>(std::malloc(s.size() + 1));
  if (p == nullptr) return nullptr;
  std::memcpy(p, s.data(), s.size());
  p[s.size()] = '\0';
  if (len) *len = static_cast<int64_t>(s.size());
  return p;
}
}  // namespace

KT2N_TEXT_API void kt2n_text_config_default(kt2n_text_config* cfg) {
  const kt2n::text::NormalizerConfig d;
  cfg->grammar_dir = nullptr;
  cfg->grammar_instances = d.grammar_instances;
  cfg->cache_entries = d.cache_entries;
  cfg->fast_path = d.fast_path;
  cfg->demo_formatting = d.demo_formatting;
  cfg->emo = d.emo_tags || d.emo_request_level;
  cfg->unit_words = d.unit_words;
  cfg->version_words = d.version_words;
  cfg->language_mode = 1;
}

KT2N_TEXT_API kt2n_text* kt2n_text_create(const kt2n_text_config* cfg, char** error) {
  try {
    kt2n::text::NormalizerConfig c;
    c.multilingual = std::getenv("KITTENTTS_09_MULTILINGUAL_NORMALIZATION") &&
                    std::string(std::getenv("KITTENTTS_09_MULTILINGUAL_NORMALIZATION")) == "1";
    c.grammar_dir = cfg->grammar_dir ? cfg->grammar_dir : "";
    c.grammar_instances = cfg->grammar_instances;
    c.cache_entries = cfg->cache_entries;
    c.fast_path = cfg->fast_path != 0;
    c.demo_formatting = cfg->demo_formatting != 0;
    c.emo_tags = c.emo_request_level = cfg->emo != 0;
    c.unit_words = cfg->unit_words != 0;
    c.version_words = cfg->version_words != 0;
    c.language_mode = cfg->language_mode == 0 ? "voice" : (cfg->language_mode == 2 ? "all" : "text");
    std::string err;
    auto n = kt2n::text::SpeechTextNormalizer::create(c, &err);
    if (!n || n->status() != "nemo_ready") {
      if (error) *error = dup(err.empty() ? (n ? n->status() : "create failed") : err, nullptr);
      return nullptr;
    }
    auto* t = new kt2n_text;
    t->normalizer = std::move(n);
    return t;
  } catch (const std::exception& e) {
    if (error) *error = dup(e.what(), nullptr);
    return nullptr;
  }
}

KT2N_TEXT_API void kt2n_text_destroy(kt2n_text* t) { delete t; }

KT2N_TEXT_API int kt2n_text_normalize(kt2n_text* t, const char* text, int64_t len, const char* language, char** out,
                                      int64_t* out_len) {
  return kt2n_text_normalize_ex(t, text, len, language, 1, out, out_len);
}

KT2N_TEXT_API int kt2n_text_normalize_ex(kt2n_text* t, const char* text, int64_t len, const char* language,
                                       int normalize, char** out, int64_t* out_len) {
  if (!out || len < 0 || (normalize != 0 && normalize != 1)) return 2;
  *out = nullptr;
  if (out_len) *out_len = 0;
  if (t == nullptr || (text == nullptr && len)) return 2;
  const std::string_view in(text ? text : "", static_cast<size_t>(len));
  if (!kt2n::text::valid_utf8(in)) return 2;
  try {
    const auto prepared = t->normalizer->prepare(in, language ? language : "en", normalize != 0);
    kt2n::text::NormalizeOptions options;
    options.wfst = prepared.wfst;
    options.voice_language = prepared.voice_language;
    options.voice_masked_values = prepared.voice_masked_values;
    options.normalize = prepared.normalize;
    options.fallback = &prepared.raw;
    options.emphasis = &prepared.emphasis;
    options.priority = 0;
    const std::string r = t->normalizer->normalize_prepared(prepared.text, options);
    *out = dup(r, out_len);
    return *out ? 0 : 2;
  } catch (...) {
    return 2;
  }
}

KT2N_TEXT_API int kt2n_text_nemo(kt2n_text* t, const char* text, int64_t len, char** out, int64_t* out_len) {
  *out = nullptr;
  if (out_len) *out_len = 0;
  if (t == nullptr || (text == nullptr && len)) return 2;
  const std::string_view in(text ? text : "", static_cast<size_t>(len));
  if (!kt2n::text::valid_utf8(in)) return 2;
  try {
    kt2n::text::NemoResult r = t->normalizer->grammar_uncached(in);
    if (!r.ok) return 1;
    *out = dup(r.text, out_len);
    return *out ? 0 : 2;
  } catch (...) {
    return 2;
  }
}

KT2N_TEXT_API void kt2n_text_free(char* p) { std::free(p); }
