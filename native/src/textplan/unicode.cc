// Code points and CPython str predicates for the native text tail.
#include "textplan_internal.h"

#include <algorithm>
#include <array>
#include <unordered_map>

namespace kt2n::textplan {
namespace {

struct Range {
  char32_t lo, hi;
};
struct Fold {
  char32_t cp;
  char32_t out[3];
};
struct IgnoreCase {
  char letter;
  char32_t cp;
};

#include "unicode_tables.inc"

template <std::size_t N>
bool in_ranges(const Range (&table)[N], char32_t c) {
  std::size_t lo = 0, hi = N;
  while (lo < hi) {
    std::size_t mid = (lo + hi) / 2;
    if (table[mid].hi < c) {
      lo = mid + 1;
    } else if (table[mid].lo > c) {
      hi = mid;
    } else {
      return true;
    }
  }
  return false;
}

}  // namespace

// ---- UTF-8 -----------------------------------------------------------------
U32 from_utf8(std::string_view s) {
  U32 out;
  out.reserve(s.size());
  std::size_t i = 0;
  const std::size_t n = s.size();
  while (i < n) {
    const unsigned char b0 = static_cast<unsigned char>(s[i]);
    if (b0 < 0x80) {
      out.push_back(b0);
      ++i;
      continue;
    }
    int len = 0;
    char32_t cp = 0;
    char32_t min = 0;
    if ((b0 & 0xE0) == 0xC0) {
      len = 2, cp = b0 & 0x1F, min = 0x80;
    } else if ((b0 & 0xF0) == 0xE0) {
      len = 3, cp = b0 & 0x0F, min = 0x800;
    } else if ((b0 & 0xF8) == 0xF0) {
      len = 4, cp = b0 & 0x07, min = 0x10000;
    }
    bool ok = len > 0 && i + len <= n;
    for (int k = 1; ok && k < len; ++k) {
      const unsigned char b = static_cast<unsigned char>(s[i + k]);
      if ((b & 0xC0) != 0x80) {
        ok = false;
      } else {
        cp = (cp << 6) | (b & 0x3F);
      }
    }
    if (ok && (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))) ok = false;
    if (!ok) {
      out.push_back(0xFFFD);
      ++i;
      continue;
    }
    out.push_back(cp);
    i += len;
  }
  return out;
}

void append_utf8(std::string& out, char32_t c) {
  if (c < 0x80) {
    out.push_back(char(c));
  } else if (c < 0x800) {
    out.push_back(char(0xC0 | (c >> 6)));
    out.push_back(char(0x80 | (c & 0x3F)));
  } else if (c < 0x10000) {
    out.push_back(char(0xE0 | (c >> 12)));
    out.push_back(char(0x80 | ((c >> 6) & 0x3F)));
    out.push_back(char(0x80 | (c & 0x3F)));
  } else {
    out.push_back(char(0xF0 | (c >> 18)));
    out.push_back(char(0x80 | ((c >> 12) & 0x3F)));
    out.push_back(char(0x80 | ((c >> 6) & 0x3F)));
    out.push_back(char(0x80 | (c & 0x3F)));
  }
}

std::string to_utf8(std::u32string_view text) {
  std::string out;
  out.reserve(text.size());
  for (char32_t c : text) append_utf8(out, c);
  return out;
}

// ---- CPython predicates ----------------------------------------------------
bool py_isspace(char32_t c) {
  if (c < 0x80) return c == ' ' || (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x1F);
  return in_ranges(kPySpace, c);
}
bool py_isalnum(char32_t c) {
  if (c < 0x80) return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  return in_ranges(kPyAlnum, c);
}
bool py_isalpha(char32_t c) {
  if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  return in_ranges(kPyAlpha, c);
}
bool uni_letter(char32_t c) { return in_ranges(kUniLetter, c); }
bool uni_number(char32_t c) { return in_ranges(kUniNumber, c); }

bool py_isalpha_str(std::u32string_view s) {
  if (s.empty()) return false;
  for (char32_t c : s)
    if (!py_isalpha(c)) return false;
  return true;
}

U32 py_strip(std::u32string_view s) {
  std::size_t a = 0, b = s.size();
  while (a < b && py_isspace(s[a])) ++a;
  while (b > a && py_isspace(s[b - 1])) --b;
  return U32(s.substr(a, b - a));
}
U32 py_lstrip(std::u32string_view s) {
  std::size_t a = 0;
  while (a < s.size() && py_isspace(s[a])) ++a;
  return U32(s.substr(a));
}
U32 py_rstrip(std::u32string_view s) {
  std::size_t b = s.size();
  while (b > 0 && py_isspace(s[b - 1])) --b;
  return U32(s.substr(0, b));
}
U32 py_rstrip_chars(std::u32string_view s, std::u32string_view chars) {
  std::size_t b = s.size();
  while (b > 0 && chars.find(s[b - 1]) != std::u32string_view::npos) --b;
  return U32(s.substr(0, b));
}
U32 py_lstrip_chars(std::u32string_view s, std::u32string_view chars) {
  std::size_t a = 0;
  while (a < s.size() && chars.find(s[a]) != std::u32string_view::npos) ++a;
  return U32(s.substr(a));
}

U32 py_casefold(std::u32string_view s) {
  static const std::unordered_map<char32_t, std::array<char32_t, 3>> table = [] {
    std::unordered_map<char32_t, std::array<char32_t, 3>> m;
    m.reserve(std::size(kCasefold));
    for (const auto& f : kCasefold) m.emplace(f.cp, std::array<char32_t, 3>{f.out[0], f.out[1], f.out[2]});
    return m;
  }();
  U32 out;
  out.reserve(s.size());
  for (char32_t c : s) {
    if (c < 0x80) {
      out.push_back((c >= 'A' && c <= 'Z') ? c + 32 : c);
      continue;
    }
    auto it = table.find(c);
    if (it == table.end()) {
      out.push_back(c);
      continue;
    }
    for (char32_t o : it->second)
      if (o) out.push_back(o);
  }
  return out;
}

// Python re IGNORECASE equality of a text code point with an ASCII
// lowercase pattern letter.
bool re_icase_letter(char32_t c, char letter) {
  if (c == char32_t(letter) || c == char32_t(letter - 32)) return true;
  if (c < 0x80) return false;
  for (const auto& e : kIgnoreCase)
    if (e.letter == letter && e.cp == c) return true;
  return false;
}

}  // namespace kt2n::textplan
