#include "normalizer.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <map>
#include <stdexcept>

#include "pyre.h"
#include "unicode.h"
#if KT2N_TEXT_WITH_VOICE
#include "voice/voice_normalizer.h"
#endif

namespace kt2n::text {
namespace {

// ---------------------------------------------------------------------------
// Patterns (normalization.py @ ebcdd3f; translation rules in pyre.h).
// ---------------------------------------------------------------------------
const char* const kExpressionTags[] = {"mundane", "nervous", "tender",        "angry",     "excited",
                                       "stern",   "sad",     "contemplative", "surprised", "joyful"};
const char* const kVocalTags[] = {"pause", "sigh", "gasp", "laugh", "giggle", "sob", "scoff", "growl", "um", "gulp"};

// re.escape(tag) under re.IGNORECASE, letter by letter.
std::string ignorecase_word(std::string_view word) {
  std::string out;
  for (char c : word) {
    const std::u32string set = py_ignorecase_set(c);
    out += "[";
    for (char32_t cp : set) {
      char buf[16];
      snprintf(buf, sizeof(buf), "\\x{%X}", static_cast<unsigned>(cp));
      out += buf;
    }
    out += "]";
  }
  return out;
}

std::string tag_alternation(const char* const* tags, size_t n) {
  std::string out;
  for (size_t i = 0; i < n; ++i) {
    if (i) out += "|";
    out += ignorecase_word(tags[i]);
  }
  return out;
}

struct Patterns {
  Regex dash{py_pattern(R"(\s*(?:[\x{2014}\x{2013}]|--)\s*)")};
  Regex dash_fix1{py_pattern(R"(,\s*,+)")};
  Regex dash_fix2{py_pattern(R"(\s+([,.!?;:]))")};
  Regex dash_fix3{py_pattern(R"(,\s*([.!?]))")};
  Regex dash_fix4{py_pattern(R"(\s{2,})")};
  Regex dash_fix5{py_pattern(R"(^\s*,\s*)")};
  Regex tag_span{"\\[(?:" + tag_alternation(kExpressionTags, 10) + ")\\]|<(?:" + tag_alternation(kVocalTags, 10) +
                 ")>"};
  Regex non_tag_bracket{R"(\[([^\[\]\n]{1,60})\]|<([^<>\n]{1,60})>|\(([^()\n]{1,60})\))"};
  Regex demo_non_tag_bracket{R"(\[([^\[\]\n]{1,400})\]|<([^<>\n]{1,60})>|\(([^()\n]{1,400})\))"};
  Regex demo_comma_collision{py_pattern(R"(,\s*([,.!?;:]))")};
  Regex ui_glyphs{R"([\x{2022}\x{2023}\x{2043}\x{25e6}\x{25b8}\x{21b3}\x{2192}\x{2190}\x{2191}\x{2193}]+)"};
  Regex straight_quote_pair{R"re("([^"\n]*)")re"};
  Regex trailing_nonterminal{py_pattern(R"([,;:\-\x{2013}\x{2014}]+\s*$)")};
  Regex emphasis_span{py_pattern(R"(\(\(\((?=[^()\n]*[^\s()])([^()\n]{1,80})\)\)\))")};
  Regex control_placeholder{R"(\x{E000}[0-9]+\x{E001})"};
  Regex caret_power{py_pattern(R"(\^(?=\s*[-+(]?\d|\())")};
  Regex middle_dot{py_pattern(R"((?<=\S)\x{00b7}(?=\S))")};
  Regex ellipsis{py_pattern(R"((?:\.\s+){2,}\.|\x{2026}|\.\.\.)")};
  Regex line_wrap{py_pattern(R"(\s*\n+\s*)")};
  Regex glued_terminal{R"((?<=[a-z]{2})([.!?;:])(?=[A-Z][a-z]))"};
  Regex version{R"(\b([vV])(\d+(?:\.\d+)+))"};
  Regex digit_terminal{py_pattern(R"((?<=\d)\s*([?!])(?=\s|$))")};
  Regex address{py_pattern(R"(\S*(?:@|://)\S*|\bwww\.\S+|\S*[A-Za-z0-9]\.[A-Za-z]{2,}/\S*)"
                           R"(|\S*[A-Za-z0-9]:\d+/\S*|\S*/\S*\?\S*=\S*)")};
  Regex spaces_tabs{R"([ \t]{2,})"};
  Regex safe_chars{py_pattern(R"(^[A-Za-z \t\n\r,;:!?'"\x{2018}\x{2019}\x{201c}\x{201d}.\-()]*$)")};
  Regex word_token{R"([A-Za-z][A-Za-z'\x{2019}\-]*\.?)"};
  Regex word{R"([^\W\d_]+)"};
  Regex final_token{py_pattern(R"([^\s]+\z)")};
  Regex internal_period{R"([A-Za-z]\.[A-Za-z])"};
  Regex split_plain_word{R"([A-Za-z]{2,})"};
  Regex split_straight_quote{R"((?<![A-Za-z])'|'(?![A-Za-z]))"};
  Regex n_per_unit{py_pattern(R"((?:\bper\s+|/)(kWh|MWh|GWh|TWh|Wh)\b)")};
  Regex n_num_energy{py_pattern(R"((?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(kWh|MWh|GWh|TWh|Wh)\b)")};
  Regex n_num_bytes{py_pattern(R"((?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(KB|kB|MB|GB|TB)\b)")};
  Regex n_num_ms{py_pattern(R"((?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?ms\b)")};
  Regex n_version_after_word{py_pattern(
      R"((?<![\w.])((?:[Vv]ersion|VERSION|[Vv]er\.|[Bb]uild|[Ff]irmware|[Rr]elease)\s+))"
      R"((\d{1,4}(?:\.\d{1,4}){2,})(?![\w]|\.\d))")};
  Regex sentence_end{py_pattern(R"((?<=[.!?\x{3002}\x{ff01}\x{ff1f}\x{061f}\x{0964}])["\x{2019}\x{201d}')\]]*\s+)")};
  Regex mask_range{R"([\x{F0000}-\x{FFFFF}])"};
  // _MATH_GLYPH_RE: every glyph is one code point, so the alternation is a class.
  std::map<char32_t, std::string> glyph_words;
  std::unique_ptr<Regex> math_glyph;

  Patterns() {
    const std::pair<char32_t, const char*> words[] = {
        {0x03b1, "alpha"}, {0x03b2, "beta"}, {0x03b3, "gamma"}, {0x03b4, "delta"}, {0x03b5, "epsilon"},
        {0x03b6, "zeta"}, {0x03b7, "eta"}, {0x03b8, "theta"}, {0x03b9, "iota"}, {0x03ba, "kappa"},
        {0x03bb, "lambda"}, {0x03bc, "mu"}, {0x03bd, "nu"}, {0x03be, "xi"}, {0x03c0, "pi"}, {0x03c1, "rho"},
        {0x03c3, "sigma"}, {0x03c4, "tau"}, {0x03c5, "upsilon"}, {0x03c6, "phi"}, {0x03d5, "phi"},
        {0x03c7, "chi"}, {0x03c8, "psi"}, {0x03c9, "omega"}, {0x0393, "gamma"}, {0x0394, "delta"},
        {0x0398, "theta"}, {0x039b, "lambda"}, {0x039e, "xi"}, {0x03a0, "pi"}, {0x03a3, "sigma"},
        {0x03a6, "phi"}, {0x03a8, "psi"}, {0x03a9, "omega"},
        {'=', "equals"}, {0x2260, "is not equal to"}, {0x2248, "approximately equals"},
        {0x2264, "is less than or equal to"}, {0x2265, "is greater than or equal to"},
        {0x00b1, "plus or minus"}, {0x00d7, "times"}, {0x00f7, "divided by"}, {0x00b7, "dot"},
        {0x2212, "minus"}, {0x2217, "star"}, {0x2218, "composed with"}, {0x22a4, "transpose"},
        {0x223c, "distributed as"}, {0x221d, "proportional to"}, {0x2261, "is equivalent to"},
        {0x2208, "in"}, {0x2209, "not in"}, {0x2282, "subset of"}, {0x2286, "subset of"},
        {0x222a, "union"}, {0x2229, "intersect"}, {0x2295, "plus"}, {0x2297, "times"}, {'^', "to the"},
        {0x2211, "sum of"}, {0x220f, "product of"}, {0x222b, "integral of"}, {0x2202, "partial"},
        {0x2207, "gradient of"}, {0x221a, "square root of"}, {0x221e, "infinity"}, {0x2206, "delta"},
        {0x2200, "for all"}, {0x2203, "there exists"}, {0x21d2, "implies"}, {0x21d4, "if and only if"},
        {0x2225, "norm"}, {0x2016, "norm"}, {0x230a, "floor of"}, {0x230b, ""}, {0x2308, "ceiling of"},
        {0x2309, ""}, {0x211d, "R"}, {0x2115, "N"}, {0x2124, "Z"}, {0x211a, "Q"}, {0x2102, "C"},
        {0x1d53c, "the expectation of"},
    };
    for (const auto& [cp, w] : words) glyph_words[cp] = w;
    const char32_t sup[] = {0x2070, 0x00b9, 0x00b2, 0x00b3, 0x2074, 0x2075, 0x2076, 0x2077, 0x2078, 0x2079};
    const char32_t sub[] = {0x2080, 0x2081, 0x2082, 0x2083, 0x2084, 0x2085, 0x2086, 0x2087, 0x2088, 0x2089};
    for (int d = 0; d < 10; ++d) {
      glyph_words[sup[d]] = std::string(1, static_cast<char>('0' + d));
      glyph_words[sub[d]] = std::string(1, static_cast<char>('0' + d));
    }
    std::string cls = "[";
    for (const auto& [cp, w] : glyph_words) {
      if (cp == '^' || cp == 0x00b7) continue;
      char buf[16];
      snprintf(buf, sizeof(buf), "\\x{%X}", static_cast<unsigned>(cp));
      cls += buf;
    }
    cls += "]";
    math_glyph = std::make_unique<Regex>(cls);
  }
};

const Patterns& P() {
  static const Patterns p;
  return p;
}

const std::u32string kAlreadyTerminated = U".!?。！？؟।\"’”')]";

const std::unordered_set<std::string>& english_markers() {
  static const std::unordered_set<std::string> s = [] {
    std::unordered_set<std::string> out;
    for (const auto& w : py_split_ws(
             "the and of to is are were with for that this these those they their there "
             "have has had been being which would could should from what when where who "
             "whom whose it its he she we you your our his her him them not but at by or "
             "if into than then about after before over just only some any each every "
             "because while can did does how why very more most my be up out nobody yes one"))
      out.insert(w);
    return out;
  }();
  return s;
}

const std::unordered_set<std::string>& foreign_markers() {
  static const std::unordered_set<std::string> s = [] {
    std::unordered_set<std::string> out;
    for (const auto& w : py_split_ws(
             "der die das und ist nicht ein eine einen einem dem den des auf sich auch wir "
             "ich sie es zu für wie aber oder dass noch nach bei sind werden wird hat "
             "haben um im "
             "le la les et est une un du dans pour pas que qui sur avec ce cette sont "
             "mais au aux nous vous ils elle "
             "el los las y del en por con una uno para se su sus lo como más pero está "
             "son al "
             "il gli di che e della delle dei degli nel nella sono è anche ma alla "
             "o os um uma não com em do da dos das na mas são pelo pela"))
      out.insert(w);
    return out;
  }();
  return s;
}

const std::unordered_set<std::string>& abbreviations() {
  static const std::unordered_set<std::string> s = {
      "mr.",    "mrs.",  "ms.",   "dr.",   "prof.",  "rev.",  "fr.",   "sr.",   "jr.",    "st.",   "sgt.",
      "lt.",    "capt.", "col.",  "gen.",  "gov.",   "pres.", "supt.", "hon.",  "messrs.", "mme.", "mlle.",
      "jan.",   "feb.",  "mar.",  "apr.",  "jun.",   "jul.",  "aug.",  "sep.",  "sept.",  "oct.",  "nov.",
      "dec.",   "mon.",  "tue.",  "tues.", "wed.",   "thu.",  "thur.", "thurs.", "fri.",  "sat.",  "sun.",
      "ave.",   "blvd.", "rd.",   "ln.",   "ct.",    "dr",    "sq.",   "mt.",   "ft.",    "pt.",   "apt.",
      "bldg.",  "dept.", "univ.", "inc.",  "ltd.",   "co.",   "corp.", "llc.",  "plc.",   "est.",  "no.",
      "nos.",   "vol.",  "vols.", "ed.",   "eds.",   "fig.",  "figs.", "ch.",   "chap.",  "sec.",  "pp.",
      "approx.", "min.", "max.",  "avg.",  "qty.",   "etc.",  "vs.",   "viz.",  "cf.",    "al.",   "ibid.",
      "op.",    "seq."};
  return s;
}

// Code-point helpers on UTF-8 strings.
std::u32string U(std::string_view s) { return to_u32(s); }
std::string S(std::u32string_view s) { return to_utf8(s); }

std::string placeholder(size_t i) {
  std::string out;
  append_utf8(out, 0xE000);
  out += std::to_string(i);
  append_utf8(out, 0xE001);
  return out;
}

// _emphasis_matches: `emphasis` nullptr = the request text (all count).
bool emphasis_counts(const Match& m, const std::vector<std::string>* emphasis) {
  if (emphasis == nullptr) return true;
  const std::string_view g = m.group(0);
  for (const auto& e : *emphasis) {
    if (e == g) return true;
  }
  return false;
}

struct Protected {
  std::string text;
  std::vector<std::string> tags;
};

// _protect_tags(text, emphasis=...)
Protected protect_tags(std::string_view text, const std::vector<std::string>* emphasis, bool emo) {
  Protected p;
  std::string protected_text(text);
  if (emo && !(emphasis != nullptr && emphasis->empty())) {
    bool any = false;
    for (const Match& m : P().emphasis_span.finditer(text)) any |= emphasis_counts(m, emphasis);
    if (any) {
      protected_text = P().emphasis_span.sub(text, [&](const Match& m) {
        if (!emphasis_counts(m, emphasis)) return std::string(m.group(0));
        p.tags.emplace_back(m.group(0));
        return placeholder(p.tags.size() - 1);
      });
    }
  }
  p.text = P().tag_span.sub(protected_text, [&](const Match& m) {
    p.tags.emplace_back(m.group(0));
    return placeholder(p.tags.size() - 1);
  });
  return p;
}

std::string restore_tags(std::string text, const std::vector<std::string>& tags) {
  for (size_t i = tags.size(); i-- > 0;) text = py_replace(text, placeholder(i), tags[i]);
  return text;
}

std::string reformat_non_tag_brackets(std::string_view text, bool demo) {
  auto replace = [](const Match& m) {
    for (size_t g = 1; g <= m.groups(); ++g) {
      if (m.has(static_cast<int>(g))) return ", " + py_strip(m.group(static_cast<int>(g))) + ",";
    }
    return std::string();
  };
  if (!demo) return P().non_tag_bracket.sub(text, replace);
  std::string t = P().demo_non_tag_bracket.sub(text, replace);
  while (true) {
    std::string cleaned = P().demo_comma_collision.sub(t, "\\1");
    if (cleaned == t) return cleaned;
    t = std::move(cleaned);
  }
}

std::string speak_math_glyphs(std::string_view text_in) {
  const Patterns& p = P();
  std::string text = p.ellipsis.sub(text_in, [&](const Match& m) {
    // match.string is the text this sub runs on (one piece between addresses)
    const std::string_view s = m.subject;
    char32_t before = 0, after = 0;
    if (m.start() > 0) {
      size_t i = m.start() - 1;
      while (i > 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80) --i;
      size_t j = i;
      before = decode(s, j);
    }
    if (m.end() < s.size()) {
      size_t j = m.end();
      after = decode(s, j);
    }
    if (before && after && py_isdigit(before) && py_isdigit(after)) return std::string(m.group(0));
    return std::string(", ");
  });
  text = p.math_glyph->sub(text, [&](const Match& m) {
    size_t i = 0;
    const char32_t cp = decode(m.group(0), i);
    return " " + p.glyph_words.at(cp) + " ";
  });
  text = p.caret_power.sub(text, " to the ");
  text = p.middle_dot.sub(text, " dot ");
  return p.spaces_tabs.sub(text, " ");
}

std::string demo_steps(std::string_view value_in) {
  const Patterns& p = P();
  std::string value = p.glued_terminal.sub(value_in, "\\1 ");
  value = speak_math_glyphs(value);
  value = p.version.sub(value, [](const Match& m) {
    return std::string(m.group(1)) + " " + join(py_split(m.group(2), "."), " point ");
  });
  return p.digit_terminal.sub(value, " \\1");
}

std::string outside_addresses(std::string_view text) {
  std::string out;
  size_t pos = 0;
  for (const Match& m : P().address.finditer(text)) {
    out += demo_steps(text.substr(pos, m.start() - pos));
    out += m.group(0);
    pos = m.end();
  }
  out += demo_steps(text.substr(pos));
  return out;
}

std::string demo_preformat(std::string_view protected_text) {
  const std::string value = P().line_wrap.sub(protected_text, " ");
  return P().spaces_tabs.sub(outside_addresses(value), " ");
}

std::string capitalize_and_terminate(std::string text, bool capitalize, bool terminate) {
  if (text.empty()) return text;
  if (capitalize) {
    for (size_t i = 0; i < text.size();) {
      const size_t a = i;
      const char32_t c = decode(text, i);
      if (py_isalpha(c)) {
        const std::u32string up = py_upper(std::u32string(1, c));
        text = text.substr(0, a) + S(up) + text.substr(i);
        break;
      }
    }
  }
  std::string stripped = py_rstrip(text);
  if (!terminate) return stripped;
  if (stripped.empty()) return stripped;
  {
    const std::u32string u = U(stripped);
    if (kAlreadyTerminated.find(u.back()) != std::u32string::npos) return stripped;
  }
  stripped = py_rstrip(P().trailing_nonterminal.sub(stripped, ""));
  return stripped.empty() ? stripped : stripped + ".";
}

std::string n_expand_units(std::string text) {
  if (text.empty() || (text.find("Wh") == std::string::npos && text.find('B') == std::string::npos &&
                       text.find("ms") == std::string::npos))
    return text;
  static const std::map<std::string, std::string> energy = {
      {"kWh", "kilowatt"}, {"MWh", "megawatt"}, {"GWh", "gigawatt"}, {"TWh", "terawatt"}, {"Wh", "watt"}};
  static const std::map<std::string, std::string> bytes = {
      {"KB", "kilobyte"}, {"kB", "kilobyte"}, {"MB", "megabyte"}, {"GB", "gigabyte"}, {"TB", "terabyte"}};
  auto plural = [](std::string_view number, const std::string& word) {
    const std::string n = py_replace(number, ",", "");
    return (n == "1" || n == "1.0") ? word : word + "s";
  };
  const Patterns& p = P();
  text = p.n_per_unit.sub(text, [&](const Match& m) {
    const std::string& w = energy.at(std::string(m.group(1)));
    return m.group(0).substr(0, 1) == "/" ? " per " + w + " hour" : "per " + w + " hour";
  });
  text = p.n_num_energy.sub(text, [&](const Match& m) {
    return std::string(m.group(1)) + " " + energy.at(std::string(m.group(2))) + " " + plural(m.group(1), "hour");
  });
  text = p.n_num_bytes.sub(text, [&](const Match& m) {
    return std::string(m.group(1)) + " " + plural(m.group(1), bytes.at(std::string(m.group(2))));
  });
  text = p.n_num_ms.sub(text, [&](const Match& m) {
    return std::string(m.group(1)) + " " + plural(m.group(1), "millisecond");
  });
  return text;
}

std::string n_expand_versions(std::string text) {
  if (text.empty() || text.find('.') == std::string::npos) return text;
  return P().n_version_after_word.sub(text, [](const Match& m) {
    return std::string(m.group(1)) + join(py_split(m.group(2), "."), " point ");
  });
}

std::unordered_set<std::string> read_words(const std::string& path) {
  std::unordered_set<std::string> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) out.insert(line);
  }
  return out;
}

// control_spans(text, emphasis) -> [start, end) byte offsets, disjoint, ordered.
std::vector<std::pair<size_t, size_t>> control_spans(std::string_view value, const std::vector<std::string>* emphasis,
                                                     bool emo) {
  std::vector<std::pair<size_t, size_t>> found;
  if (emo && !(emphasis != nullptr && emphasis->empty())) {
    for (const Match& m : P().emphasis_span.finditer(value)) {
      if (emphasis_counts(m, emphasis)) found.emplace_back(m.start(), m.end());
    }
  }
  for (const Match& m : P().tag_span.finditer(value)) found.emplace_back(m.start(), m.end());
  std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) {
    return a.first != b.first ? a.first < b.first : a.second > b.second;
  });
  std::vector<std::pair<size_t, size_t>> spans;
  for (const auto& s : found) {
    if (!spans.empty() && s.first < spans.back().second) continue;
    spans.push_back(s);
  }
  return spans;
}

}  // namespace

// ---------------------------------------------------------------------------
std::string normalize_dashes(std::string_view text) {
  const Patterns& p = P();
  std::string v = p.dash.sub(text, ", ");
  v = p.dash_fix1.sub(v, ", ");
  v = p.dash_fix2.sub(v, "\\1");
  v = p.dash_fix3.sub(v, "\\1");
  v = p.dash_fix4.sub(v, " ");
  v = p.dash_fix5.sub(v, "");
  return py_strip(v);
}

bool text_has_expression_tag(std::string_view text) { return P().tag_span.search(text); }

bool text_is_english(std::string_view text) {
  int english = 0, foreign = 0, words = 0;
  for (const Match& m : P().word.finditer(text)) {
    std::u32string word;
    for (char32_t c : U(m.group(0))) {
      if (py_isalpha(c)) word.push_back(c);
    }
    if (word.empty()) continue;
    ++words;
    bool ascii = true;
    for (char32_t c : word) {
      if (!py_isascii(c)) {
        ascii = false;
        if (!py_latin_name(c)) return false;
      }
    }
    const std::string lowered = S(py_casefold(word));
    if (english_markers().count(lowered)) {
      ++english;
    } else if (foreign_markers().count(lowered) || !ascii) {
      ++foreign;
    }
  }
  if (!words) return false;
  return foreign == 0 || english > foreign;
}

std::unique_ptr<SpeechTextNormalizer> SpeechTextNormalizer::create(const NormalizerConfig& cfg, std::string* error) {
  std::unique_ptr<SpeechTextNormalizer> n(new SpeechTextNormalizer(cfg));
  P();  // compile every pattern now, not on the first request
  n->cache_.entries = static_cast<size_t>(std::max(0, cfg.cache_entries));
  if (!cfg.enabled) {
    n->status_ = "disabled";
    return n;
  }
  if (cfg.multilingual) {
#if KT2N_TEXT_WITH_VOICE
    kitten_voice::warm();
#else
    if (error) *error = "Built without multilingual voice normalization";
    return nullptr;
#endif
  }
  std::string err;
  n->pool_ = std::make_unique<NemoPool>(cfg.grammar_dir, std::max(1, cfg.grammar_instances), &err);
  if (!n->pool_->ok()) {
    const std::string why = "text grammar unavailable in grammar_dir '" + cfg.grammar_dir + "': " + err;
    if (error) *error = why;
    if (!cfg.allow_unavailable_grammar) return nullptr;  // loud: the caller refuses to start
    // Opt-in, production's behaviour: the grammar never runs (fail open).
    n->status_ = "unavailable:" + err;
    n->pool_.reset();
    return n;
  }
  n->whitelist_ = read_words(cfg.grammar_dir + "/whitelist_words.txt");
  n->data_words_ = read_words(cfg.grammar_dir + "/data_words.txt");
  if (n->whitelist_.empty() || n->data_words_.empty()) {
    // Production derives both sets from NeMo's install; here they are shipped
    // files, so an empty set means a broken text_data_dir, not a valid state.
    const std::string why = "text grammar word tables missing or empty in grammar_dir '" + cfg.grammar_dir +
                            "': whitelist_words.txt " + std::to_string(n->whitelist_.size()) +
                            " words, data_words.txt " + std::to_string(n->data_words_.size()) + " words";
    if (error) *error = why;
    if (!cfg.allow_unavailable_grammar) return nullptr;
  }
  if (n->whitelist_.empty()) n->cfg_.fast_path = false;  // as warm(): no table, no fast path
  // warm-up call (production warms with this exact text)
  n->pool_->normalize("Warm up 1 2 3.");
  n->status_ = "nemo_ready";
  n->test_failures_left_ = cfg.test_fail_grammar_calls;
  // N_SPLIT never fans out wider than the grammar copies (budget = idle copies).
  n->helpers_ = std::make_unique<HelperPool>(std::clamp(cfg.grammar_instances - 1, 0, 7));
  return n;
}

HelperPool::HelperPool(int threads) {
  for (int i = 0; i < threads; ++i) threads_.emplace_back([this] { loop(); });
}

HelperPool::~HelperPool() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : threads_) t.join();
}

void HelperPool::loop() {
  for (;;) {
    std::function<void()>* task = nullptr;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
      if (queue_.empty()) return;  // stop_ and drained
      task = queue_.front();
      queue_.pop_front();
    }
    (*task)();
  }
}

void HelperPool::run(std::vector<std::function<void()>>& tasks) {
  if (tasks.empty()) return;
  if (tasks.size() == 1 || threads_.empty()) {
    for (auto& t : tasks) t();
    return;
  }
  std::mutex done_mu;
  std::condition_variable done_cv;
  size_t left = tasks.size() - 1;
  std::exception_ptr error;
  std::vector<std::function<void()>> wrapped;
  wrapped.reserve(tasks.size() - 1);
  for (size_t k = 1; k < tasks.size(); ++k) {
    wrapped.emplace_back([&, k] {
      std::exception_ptr e;
      try {
        tasks[k]();
      } catch (...) {
        e = std::current_exception();
      }
      std::lock_guard<std::mutex> lock(done_mu);
      if (e && !error) error = e;
      if (--left == 0) done_cv.notify_one();
    });
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& w : wrapped) queue_.push_back(&w);
  }
  if (wrapped.size() == 1) {
    cv_.notify_one();
  } else {
    cv_.notify_all();
  }
  std::exception_ptr head_error;
  try {
    tasks[0]();
  } catch (...) {
    head_error = std::current_exception();
  }
  std::unique_lock<std::mutex> lock(done_mu);
  done_cv.wait(lock, [&] { return left == 0; });
  if (head_error) std::rethrow_exception(head_error);
  if (error) std::rethrow_exception(error);
}

void SpeechTextNormalizer::run_parallel(std::vector<std::function<void()>>& tasks) {
  if (helpers_) {
    helpers_->run(tasks);
  } else {
    for (auto& t : tasks) t();
  }
}

std::string SpeechTextNormalizer::status() const { return status_; }

std::vector<std::pair<size_t, size_t>> SpeechTextNormalizer::control_spans(
    std::string_view text, const std::vector<std::string>* emphasis) const {
  return kt2n::text::control_spans(text, emphasis, emo_enabled());
}

bool SpeechTextNormalizer::Cache::get(const std::string& key, std::string* value) {
  if (entries == 0 || cp_len(key) > 2048) return false;
  std::lock_guard<std::mutex> lock(mu);
  auto it = index.find(key);
  if (it == index.end()) return false;
  order.splice(order.begin(), order, it->second);
  *value = it->second->second;
  return true;
}

void SpeechTextNormalizer::Cache::put(const std::string& key, const std::string& value) {
  if (entries == 0 || cp_len(key) > 2048) return;
  std::lock_guard<std::mutex> lock(mu);
  auto it = index.find(key);
  if (it != index.end()) {
    it->second->second = value;
    order.splice(order.begin(), order, it->second);
    return;
  }
  order.emplace_front(key, value);
  index[key] = order.begin();
  while (order.size() > entries) {
    index.erase(order.back().first);
    order.pop_back();
  }
}

NemoResult SpeechTextNormalizer::grammar(std::string_view protected_text, int priority) {
  const std::string key(protected_text);
  std::string cached;
  if (cache_.entries) {
    if (cache_.get(key, &cached)) {
      counters_.cache_hits++;
      return NemoResult{true, cached, ""};
    }
    counters_.cache_misses++;
  }
  counters_.grammar_calls++;
  if (test_failures_left_.load() > 0 && test_failures_left_.fetch_sub(1) > 0)
    return NemoResult{false, "", "RuntimeError"};
  NemoResult r = pool_->normalize(protected_text, priority);
  if (r.ok && !py_strip(r.text).empty()) cache_.put(key, r.text);
  return r;
}

PreparedText SpeechTextNormalizer::prepare(std::string_view text, std::string_view language, bool normalize) const {
  const bool emo = emo_enabled();
  PreparedText out;
  if (emo) {
    for (const Match& m : P().emphasis_span.finditer(text)) out.emphasis.emplace_back(m.group(0));
  }
  if (!normalize || (cfg_.multilingual && cfg_.enabled)) {
    out.text = std::string(text);
    out.raw = out.text;
    out.normalize = normalize;
    if (normalize) {
      out.voice_language = py_split(py_casefold(language), "-")[0];
#if KT2N_TEXT_WITH_VOICE
      out.text = kitten_voice::prepare_for_segmentation(text, out.voice_language);
      out.voice_masked_values = out.text != out.raw;
#endif
    }
    return out;
  }
  const Protected prot = protect_tags(text, nullptr, emo);
  std::string code = py_casefold(language.empty() ? std::string_view("en") : language);
  code = py_split(code, "-")[0];
  auto cleanup = [&](std::string value, bool demo) {
    if (demo) value = demo_preformat(value);
    return normalize_dashes(P().ui_glyphs.sub(reformat_non_tag_brackets(value, demo), " "));
  };
  const bool everywhere = cfg_.language_mode == "all";
  const bool demo = cfg_.demo_formatting && (code == "en" || everywhere);
  std::string raw = cleanup(prot.text, demo);
  const bool english = code == "en" || everywhere || (cfg_.language_mode == "text" && text_is_english(raw));
  if (english && cfg_.demo_formatting && !demo) raw = cleanup(prot.text, true);
  if (!english || !cfg_.enabled || raw.empty() || !pool_) {
    out.text = restore_tags(raw, prot.tags);
    out.raw = out.text;
    out.wfst = false;
    return out;
  }
  const std::string prepared = P().straight_quote_pair.sub(raw, [](const Match& m) {
    return "“" + std::string(m.group(1)) + "”";
  });
  out.text = restore_tags(prepared, prot.tags);
  out.raw = restore_tags(raw, prot.tags);
  out.wfst = true;
  return out;
}

bool SpeechTextNormalizer::nemo_can_change(std::string_view text) const {
  if (text.empty() || !P().safe_chars.match(text)) return true;
  if (whitelist_.empty()) return true;
  for (const Match& m : P().word_token.finditer(text)) {
    const std::string token(m.group(0));
    std::string word = token;
    while (!word.empty() && word.back() == '.') word.pop_back();
    if (whitelist_.count(py_casefold(token)) || whitelist_.count(py_casefold(word))) return true;
    const std::u32string w = U(word);
    if (w.size() >= 2 && py_str_isupper(w)) return true;
    if (w.size() == 1) {
      const std::string f = py_casefold(word);
      if (f != "a" && f != "i") return true;
    }
  }
  return false;
}

bool SpeechTextNormalizer::ends_on_abbreviation(std::string_view text) const {
  const std::string stripped = py_rstrip_chars(py_rstrip(text), U"\"'’”)]");
  if (stripped.empty() || stripped.back() != '.') return false;
  Match m;
  if (!P().final_token.search(stripped, 0, &m)) return false;
  const std::string token = py_lstrip_chars(m.group(0), U"\"'“‘([");
  if (token.empty()) return false;
  if (P().internal_period.search(token)) return true;
  const std::string word = py_rstrip_chars(token, U".");
  const std::u32string w = U(word);
  if (w.size() == 1 && py_isalpha(w[0])) return true;
  if (!py_str_isalpha(w)) return false;
  if (whitelist_.count(py_casefold(token)) || whitelist_.count(py_casefold(word))) return true;
  return abbreviations().count(py_casefold(token)) > 0;
}

std::vector<size_t> SpeechTextNormalizer::grammar_cut_points(std::string_view text) const {
  std::vector<size_t> cuts;
  if (data_words_.empty() || text.find('"') != std::string_view::npos || P().split_straight_quote.search(text))
    return cuts;
  static const std::u32string openers = U"([{“‘";
  static const std::u32string closers = U")]}”’";
  const std::vector<std::string> words = py_split(text, " ");
  int depth = 0;
  size_t start = 0;
  for (size_t position = 0; position + 1 < words.size(); ++position) {
    const std::string& word = words[position];
    const std::u32string uw = U(word);
    for (char32_t c : uw) {
      if (openers.find(c) != std::u32string::npos) {
        depth += 1;
      } else if (closers.find(c) != std::u32string::npos) {
        if (c == 0x2019 && P().split_plain_word.search(word)) continue;
        depth = std::max(0, depth - 1);
      }
    }
    start += uw.size();
    const size_t index = start;
    start += 1;
    const std::string& right = words[position + 1];
    const std::string left = py_rstrip_chars(word, U",;");
    if (depth || left.empty() || right.empty()) continue;
    if (!(P().split_plain_word.fullmatch(left) && P().split_plain_word.fullmatch(right))) continue;
    if (py_str_isupper(U(left)) || py_str_isupper(U(right))) continue;
    if (data_words_.count(py_casefold(left)) || data_words_.count(py_casefold(right))) continue;
    cuts.push_back(index);
  }
  return cuts;
}

std::vector<std::string> SpeechTextNormalizer::split_for_grammar(std::string_view text, int target, int max_pieces,
                                                                 int minimum) const {
  const std::u32string value = U(text);
  const long len = static_cast<long>(value.size());
  if (target <= 0 || max_pieces <= 1 || len <= target) return {std::string(text)};
  const long count = std::min<long>(max_pieces, (len + target - 1) / target);
  const std::vector<size_t> cuts = grammar_cut_points(text);
  if (count <= 1 || cuts.empty()) return {std::string(text)};
  std::vector<long> chosen;
  long previous = 0;
  for (long number = 1; number < count; ++number) {
    const double ideal = static_cast<double>(len) * static_cast<double>(number) / static_cast<double>(count);
    bool have = false;
    long best = 0;
    double best_d = 0;
    for (size_t c : cuts) {
      const long cut = static_cast<long>(c);
      if (cut - previous >= minimum && len - cut >= minimum) {
        const double d = std::fabs(static_cast<double>(cut) - ideal);
        if (!have || d < best_d) {
          have = true;
          best = cut;
          best_d = d;
        }
      }
    }
    if (!have) break;
    if (best <= previous) continue;
    chosen.push_back(best);
    previous = best;
  }
  if (chosen.empty()) return {std::string(text)};
  std::vector<std::string> pieces;
  long s = 0;
  for (long cut : chosen) {
    pieces.push_back(S(std::u32string_view(value).substr(s, cut - s)));
    s = cut + 1;
  }
  pieces.push_back(S(std::u32string_view(value).substr(s)));
  return pieces;
}

std::vector<std::string> SpeechTextNormalizer::grammar_pieces(std::string_view text, int limit,
                                                              const std::vector<std::string>& emphasis) const {
  const std::string value(text);
  if (limit <= 0 || static_cast<long>(cp_len(value)) <= limit) return {value};
  // EMO: mask control spans (length-preserving) so no piece splits one.
  if (emo_enabled() && !P().mask_range.search(value)) {
    const auto spans = kt2n::text::control_spans(value, &emphasis, true);
    if (!spans.empty()) {
      std::string masked;
      std::vector<std::pair<std::string, std::string>> restore;
      size_t pos = 0;
      for (size_t i = 0; i < spans.size(); ++i) {
        const std::u32string span = U(std::string_view(value).substr(spans[i].first, spans[i].second - spans[i].first));
        std::u32string mark(span.size() - 1, static_cast<char32_t>(0xF0000 + i));
        mark.push_back(span.back());
        masked += value.substr(pos, spans[i].first - pos);
        masked += S(mark);
        restore.emplace_back(S(mark), S(span));
        pos = spans[i].second;
      }
      masked += value.substr(pos);
      std::vector<std::string> out = grammar_pieces(masked, limit, {});
      for (auto& piece : out) {
        for (const auto& [m, s] : restore) piece = py_replace(piece, m, s);
      }
      return out;
    }
  }
  std::vector<std::string> sentences;
  size_t start = 0;
  for (const Match& m : P().sentence_end.finditer(value)) {
    const std::string head = py_rstrip(std::string_view(value).substr(0, m.end()));
    const size_t end = head.size();
    const std::string sentence = py_strip(std::string_view(value).substr(start, end > start ? end - start : 0));
    if (!sentence.empty()) {
      if (!sentences.empty() && ends_on_abbreviation(sentences.back())) {
        sentences.back() += " " + sentence;
      } else {
        sentences.push_back(sentence);
      }
    }
    start = m.end();
  }
  const std::string tail = py_strip(std::string_view(value).substr(start));
  if (!tail.empty()) {
    if (!sentences.empty() && ends_on_abbreviation(sentences.back())) {
      sentences.back() += " " + tail;
    } else {
      sentences.push_back(tail);
    }
  }
  std::vector<std::string> pieces;
  for (const auto& sentence : sentences) {
    if (!pieces.empty() && static_cast<long>(cp_len(pieces.back()) + 1 + cp_len(sentence)) <= limit) {
      pieces.back() += " " + sentence;
    } else {
      pieces.push_back(sentence);
    }
  }
  if (pieces.empty()) return {value};
  return pieces;
}

bool SpeechTextNormalizer::needs_grammar(std::string_view prepared, const NormalizeOptions& opts) const {
  if (!opts.wfst) return false;
  static const std::vector<std::string> kNone;
  const std::vector<std::string>* emphasis = opts.emphasis ? opts.emphasis : &kNone;
  std::string protected_text = protect_tags(prepared, emphasis, emo_enabled()).text;
  if (cfg_.unit_words) protected_text = n_expand_units(std::move(protected_text));
  if (cfg_.version_words) protected_text = n_expand_versions(std::move(protected_text));
  const bool use_fast_path =
      opts.fast_path < 0 ? cfg_.fast_path : (opts.fast_path > 0 && !whitelist_.empty());
  return !(use_fast_path && !nemo_can_change(protected_text));
}

std::string SpeechTextNormalizer::normalize_prepared(std::string_view prepared, const NormalizeOptions& opts) {
  if (!opts.normalize) return std::string(prepared);
  static const std::vector<std::string> kNone;
  const std::vector<std::string>* emphasis = opts.emphasis ? opts.emphasis : &kNone;
  const bool emo = emo_enabled();
  std::string restored(prepared);
  if (opts.voice_masked_values) restored = py_replace(restored, "\xef\xbf\xbf", " ");
  Protected prot = protect_tags(restored, emphasis, emo);
  std::string protected_text = std::move(prot.text);
  if (cfg_.unit_words && opts.wfst) protected_text = n_expand_units(std::move(protected_text));
  if (cfg_.version_words && opts.wfst) protected_text = n_expand_versions(std::move(protected_text));
  auto finish = [&](std::string value) {
    return restore_tags(capitalize_and_terminate(std::move(value), opts.capitalize, opts.terminate), prot.tags);
  };
  if (!opts.voice_language.empty()) {
#if KT2N_TEXT_WITH_VOICE
    if (opts.voice_language == "en" && cfg_.unit_words) protected_text = n_expand_units(std::move(protected_text));
    std::string result = kitten_voice::normalize(protected_text, opts.voice_language);
    static const Regex digit("\\p{Nd}");
    const std::string numeric_text = prot.tags.empty() ? result : P().control_placeholder.sub(result, "");
    if (opts.voice_language == "en" && pool_ && !digit.search(numeric_text) && nemo_can_change(numeric_text)) {
      const NemoResult legacy = grammar(result, opts.priority);
      if (legacy.ok && !legacy.text.empty()) result = legacy.text;
    }
    return finish(std::move(result));
#else
    throw std::runtime_error("Built without multilingual voice normalization");
#endif
  }
  if (!opts.wfst) return finish(protected_text);
  const bool use_fast_path =
      opts.fast_path < 0 ? cfg_.fast_path : (opts.fast_path > 0 && !whitelist_.empty());
  if (use_fast_path && !nemo_can_change(protected_text)) {
    counters_.fast_path_skips++;
    return finish(normalize_dashes(protected_text));
  }
  if (use_fast_path) counters_.fast_path_runs++;
  if (pool_) {
    for (int attempt = 0; attempt < 2; ++attempt) {
      if (attempt) counters_.retries++;
      NemoResult r = grammar(protected_text, opts.priority);
      if (r.ok && !r.text.empty() && !py_strip(r.text).empty()) return finish(normalize_dashes(r.text));
    }
  }
  counters_.fail_open++;
  if (opts.fallback == nullptr) return finish(protected_text);
  Protected fb = protect_tags(*opts.fallback, emphasis, emo);
  return restore_tags(capitalize_and_terminate(fb.text, opts.capitalize, opts.terminate), fb.tags);
}

std::string SpeechTextNormalizer::normalize(std::string_view text, std::string_view language) {
  const PreparedText p = prepare(text, language);
  NormalizeOptions o;
  o.wfst = p.wfst;
  o.voice_language = p.voice_language;
  o.normalize = p.normalize;
  o.voice_masked_values = p.voice_masked_values;
  o.fallback = &p.raw;
  o.emphasis = &p.emphasis;
  o.priority = 0;  // the whole-text call is what the first byte waits on
  return normalize_prepared(p.text, o);
}

}  // namespace kt2n::text
