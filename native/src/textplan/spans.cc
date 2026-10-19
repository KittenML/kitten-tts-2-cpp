// Emotion controls: [tag], <event> and (((emphasis))) spans
// (kitten_tts_09/normalization.py @ ebcdd3f, EMO on).
#include <algorithm>

#include "textplan_internal.h"

namespace kt2n::textplan {
namespace {

constexpr std::u32string_view kExpressionTags[] = {
    U"mundane", U"nervous", U"tender", U"angry", U"excited",
    U"stern", U"sad", U"contemplative", U"surprised", U"joyful"};
constexpr std::u32string_view kVocalTags[] = {U"pause", U"sigh", U"gasp", U"laugh", U"giggle",
                                              U"sob", U"scoff", U"growl", U"um", U"gulp"};
constexpr char32_t kMaskBase = 0xF0000;
constexpr char32_t kMaskLimit = 0xFFFF0;

bool icase_equal_at(std::u32string_view s, std::size_t i, std::u32string_view name) {
  if (i + name.size() > s.size()) return false;
  for (std::size_t k = 0; k < name.size(); ++k)
    if (!re_icase_letter(s[i + k], char(name[k]))) return false;
  return true;
}

// The end of a _TAG_SPAN match at ``i`` (or npos).  ``expression_only``: the
// [expression] alternative alone (_LEADING_EXPRESSION_TAG).
std::size_t tag_at(std::u32string_view s, std::size_t i, bool expression_only = false) {
  if (i >= s.size()) return U32::npos;
  if (s[i] == U'[') {
    for (auto name : kExpressionTags) {
      const std::size_t close = i + 1 + name.size();
      if (close < s.size() && s[close] == U']' && icase_equal_at(s, i + 1, name)) return close + 1;
    }
  } else if (s[i] == U'<' && !expression_only) {
    for (auto name : kVocalTags) {
      const std::size_t close = i + 1 + name.size();
      if (close < s.size() && s[close] == U'>' && icase_equal_at(s, i + 1, name)) return close + 1;
    }
  }
  return U32::npos;
}

std::vector<Match> tag_matches(std::u32string_view s) {
  std::vector<Match> out;
  std::size_t i = 0;
  while (i < s.size()) {
    const std::size_t e = tag_at(s, i);
    if (e != U32::npos) {
      out.push_back({i, e});
      i = e;
    } else {
      ++i;
    }
  }
  return out;
}

bool emphasis_body(char32_t c) { return c != U'(' && c != U')' && c != U'\n'; }

// _EMPHASIS_SPAN = \(\(\((?=[^()\n]*[^\s()])([^()\n]{1,80})\)\)\)
std::size_t emphasis_at(std::u32string_view s, std::size_t i) {
  if (i + 3 > s.size() || s[i] != U'(' || s[i + 1] != U'(' || s[i + 2] != U'(') return U32::npos;
  std::size_t j = i + 3;
  std::size_t k = j;
  bool visible = false;
  while (k < s.size() && emphasis_body(s[k])) {
    if (!py_isspace(s[k])) visible = true;
    ++k;
  }
  const std::size_t run = k - j;
  if (!visible || run < 1 || run > 80) return U32::npos;
  if (k + 3 <= s.size() && s[k] == U')' && s[k + 1] == U')' && s[k + 2] == U')') return k + 3;
  return U32::npos;
}

std::vector<Match> emphasis_matches_all(std::u32string_view s) {
  std::vector<Match> out;
  std::size_t i = 0;
  while (i < s.size()) {
    const std::size_t e = emphasis_at(s, i);
    if (e != U32::npos) {
      out.push_back({i, e});
      i = e;
    } else {
      ++i;
    }
  }
  return out;
}

std::vector<Match> emphasis_matches(std::u32string_view s, const Emphasis* emphasis) {
  std::vector<Match> all = emphasis_matches_all(s);
  if (emphasis == nullptr) return all;
  std::vector<Match> out;
  if (emphasis->empty()) return out;
  for (const Match& m : all) {
    const std::u32string_view text = s.substr(m.start, m.end - m.start);
    if (std::find(emphasis->begin(), emphasis->end(), text) != emphasis->end()) out.push_back(m);
  }
  return out;
}

}  // namespace

Emphasis emphasis_spans(std::u32string_view text) {
  Emphasis out;
  for (const Match& m : emphasis_matches_all(text)) out.emplace_back(text.substr(m.start, m.end - m.start));
  return out;
}

std::vector<Span> control_spans(std::u32string_view text, const Emphasis* emphasis) {
  std::vector<Span> found;
  for (const Match& m : emphasis_matches(text, emphasis)) found.push_back({m.start, m.end});
  for (const Match& m : tag_matches(text)) found.push_back({m.start, m.end});
  std::stable_sort(found.begin(), found.end(), [](const Span& a, const Span& b) {
    if (a.start != b.start) return a.start < b.start;
    return a.end > b.end;
  });
  std::vector<Span> spans;
  for (const Span& span : found) {
    if (!spans.empty() && span.start < spans.back().end) continue;
    spans.push_back(span);
  }
  return spans;
}

bool text_has_expression_tag(std::u32string_view text) { return !tag_matches(text).empty(); }

bool text_has_tag(std::u32string_view text, const Emphasis* emphasis) {
  return !tag_matches(text).empty() || !emphasis_matches(text, emphasis).empty();
}

std::optional<U32> leading_expression_tag(std::u32string_view text) {
  std::size_t i = 0;
  while (i < text.size() && py_isspace(text[i])) ++i;
  const std::size_t e = tag_at(text, i, /*expression_only=*/true);
  if (e == U32::npos) return std::nullopt;
  return U32(text.substr(i, e - i));
}

bool speakable_outside_spans(std::u32string_view text, const Emphasis* emphasis) {
  const std::vector<Span> spans = control_spans(text, emphasis);
  if (spans.empty()) return true;
  std::size_t position = 0;
  auto any_alnum = [&](std::size_t a, std::size_t b) {
    for (std::size_t k = a; k < b; ++k)
      if (py_isalnum(text[k])) return true;
    return false;
  };
  for (const Span& span : spans) {
    if (any_alnum(position, span.start)) return true;
    position = span.end;
  }
  return any_alnum(position, text.size());
}

std::optional<Span> cut_inside_control_span(std::u32string_view text, std::size_t cut,
                                            const Emphasis* emphasis) {
  for (const Span& span : control_spans(text, emphasis))
    if (span.start < cut && cut < span.end) return span;
  return std::nullopt;
}

Masked mask_control_spans(std::u32string_view text, const Emphasis& emphasis, std::size_t first) {
  Masked out;
  for (char32_t c : text) {
    if (c >= kMaskBase && c <= 0xFFFFF) {
      out.text = U32(text);
      return out;
    }
  }
  const std::vector<Span> spans = control_spans(text, &emphasis);
  if (spans.empty() || first + spans.size() > std::size_t(kMaskLimit - kMaskBase)) {
    out.text = U32(text);
    return out;
  }
  std::size_t position = 0;
  std::size_t index = first;
  for (const Span& span : spans) {
    const std::u32string_view original = text.substr(span.start, span.end - span.start);
    U32 masked(original.size() - 1, char32_t(kMaskBase + index));
    masked.push_back(original.back());
    out.text.append(text.substr(position, span.start - position));
    out.text += masked;
    out.restore.emplace_back(std::move(masked), U32(original));
    position = span.end;
    ++index;
  }
  out.text.append(text.substr(position));
  return out;
}

U32 unmask_control_spans(U32 text, const std::vector<std::pair<U32, U32>>& restore) {
  for (const auto& [masked, span] : restore) {
    // str.replace(masked, span): every non-overlapping occurrence, left to right.
    if (masked.empty()) continue;
    U32 out;
    std::size_t position = 0;
    while (true) {
      const std::size_t hit = text.find(masked, position);
      if (hit == U32::npos) break;
      out.append(text, position, hit - position);
      out += span;
      position = hit + masked.size();
    }
    out.append(text, position, U32::npos);
    text = std::move(out);
  }
  return text;
}

bool in_mask_range(char32_t c) { return c >= kMaskBase && c <= 0xFFFFF; }

}  // namespace kt2n::textplan
