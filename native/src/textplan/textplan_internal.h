// Private helpers shared by the src/textplan translation units.
#pragma once

#include "kt2n/textplan.h"

namespace kt2n::textplan {

void append_utf8(std::string& out, char32_t c);
bool uni_letter(char32_t c);  // \p{L}
bool uni_number(char32_t c);  // \p{N}
bool re_icase_letter(char32_t c, char letter);
U32 py_lstrip_chars(std::u32string_view s, std::u32string_view chars);

// Python str.rfind(sub, 0, end): last start index of ``sub`` fully inside
// [0, end), or -1.
inline long py_rfind(std::u32string_view s, std::u32string_view sub, std::size_t end) {
  end = std::min(end, s.size());
  if (sub.size() > end) return -1;
  for (std::size_t i = end - sub.size() + 1; i-- > 0;) {
    if (s.compare(i, sub.size(), sub) == 0) return long(i);
  }
  return -1;
}

inline bool is_terminal_sentence_char(char32_t c) {
  // [.!?。！？؟।]
  return c == U'.' || c == U'!' || c == U'?' || c == U'。' || c == U'！' ||
         c == U'？' || c == U'؟' || c == U'।';
}
inline bool is_sentence_closer(char32_t c) {
  // ["’”\')\]]
  return c == U'"' || c == U'’' || c == U'”' || c == U'\'' || c == U')' || c == U']';
}

// _SENTENCE_END = (?<=[.!?。！？؟।])["’”')\]]*\s+ : match at ``i`` within
// [0, endpos) (lookbehind may read before i).  Returns the match end or npos.
inline std::size_t sentence_end_at(std::u32string_view s, std::size_t i, std::size_t endpos) {
  if (i == 0 || i > endpos || !is_terminal_sentence_char(s[i - 1])) return U32::npos;
  std::size_t j = i;
  while (j < endpos && is_sentence_closer(s[j])) ++j;
  std::size_t k = j;
  while (k < endpos && py_isspace(s[k])) ++k;
  if (k == j) return U32::npos;
  return k;
}

struct Match {
  std::size_t start, end;
};
// re.finditer(_SENTENCE_END, s, pos, endpos)
inline std::vector<Match> sentence_ends(std::u32string_view s, std::size_t pos = 0,
                                        std::size_t endpos = U32::npos) {
  endpos = std::min(endpos, s.size());
  std::vector<Match> out;
  std::size_t i = pos;
  while (i <= endpos) {
    const std::size_t e = sentence_end_at(s, i, endpos);
    if (e != U32::npos) {
      out.push_back({i, e});
      i = e;  // non-empty match: continue at its end
    } else {
      ++i;
    }
  }
  return out;
}
// re.split(_SENTENCE_END, s)
inline std::vector<U32> sentence_split(std::u32string_view s) {
  std::vector<U32> parts;
  std::size_t last = 0;
  for (const Match& m : sentence_ends(s)) {
    parts.emplace_back(s.substr(last, m.start - last));
    last = m.end;
  }
  parts.emplace_back(s.substr(last));
  return parts;
}
// [part.strip() for part in _SENTENCE_END.split(s) if part.strip()]
inline std::vector<U32> sentences_of(std::u32string_view s) {
  std::vector<U32> out;
  for (const U32& part : sentence_split(s)) {
    U32 stripped = py_strip(part);
    if (!stripped.empty()) out.push_back(std::move(stripped));
  }
  return out;
}

// _CLAUSE_BREAK / _REALTIME_CLAUSE_BREAK = [,;:](?=\s)|\s[—–-](?=\s)
inline std::vector<Match> clause_breaks(std::u32string_view s) {
  std::vector<Match> out;
  std::size_t i = 0;
  const std::size_t n = s.size();
  while (i < n) {
    const char32_t c = s[i];
    if ((c == U',' || c == U';' || c == U':') && i + 1 < n && py_isspace(s[i + 1])) {
      out.push_back({i, i + 1});
      i += 1;
      continue;
    }
    if (py_isspace(c) && i + 2 < n &&
        (s[i + 1] == U'—' || s[i + 1] == U'–' || s[i + 1] == U'-') &&
        py_isspace(s[i + 2])) {
      out.push_back({i, i + 2});
      i += 2;
      continue;
    }
    ++i;
  }
  return out;
}

inline U32 join(const std::vector<U32>& parts, std::u32string_view sep) {
  U32 out;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (i) out += sep;
    out += parts[i];
  }
  return out;
}

}  // namespace kt2n::textplan
