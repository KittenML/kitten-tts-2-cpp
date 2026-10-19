#include "unicode.h"

#include <algorithm>

namespace kt2n::text {
namespace {

struct CpRange {
  char32_t lo, hi;
};
struct CpMap {
  char32_t cp;
  uint8_t n;
  char32_t out[3];
};

#include "unicode_tables.inc"

template <size_t N>
bool in_ranges(const CpRange (&table)[N], char32_t c) {
  size_t lo = 0, hi = N;
  while (lo < hi) {
    const size_t mid = (lo + hi) / 2;
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

template <size_t N>
const CpMap* find_map(const CpMap (&table)[N], char32_t c) {
  const CpMap* end = table + N;
  const CpMap* it = std::lower_bound(table, end, c, [](const CpMap& m, char32_t v) { return m.cp < v; });
  return (it != end && it->cp == c) ? it : nullptr;
}

}  // namespace

char32_t decode(std::string_view s, size_t& i) {
  const auto b0 = static_cast<unsigned char>(s[i]);
  if (b0 < 0x80) {
    ++i;
    return b0;
  }
  if ((b0 & 0xE0) == 0xC0) {
    const char32_t c = ((b0 & 0x1F) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3F);
    i += 2;
    return c;
  }
  if ((b0 & 0xF0) == 0xE0) {
    const char32_t c = ((b0 & 0x0F) << 12) | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6) |
                       (static_cast<unsigned char>(s[i + 2]) & 0x3F);
    i += 3;
    return c;
  }
  const char32_t c = ((b0 & 0x07) << 18) | ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12) |
                     ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6) |
                     (static_cast<unsigned char>(s[i + 3]) & 0x3F);
  i += 4;
  return c;
}

void append_utf8(std::string& out, char32_t c) {
  if (c < 0x80) {
    out.push_back(static_cast<char>(c));
  } else if (c < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (c >> 6)));
    out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
  } else if (c < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (c >> 12)));
    out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (c >> 18)));
    out.push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (c & 0x3F)));
  }
}

std::u32string to_u32(std::string_view s) {
  std::u32string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size();) out.push_back(decode(s, i));
  return out;
}

std::string to_utf8(std::u32string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char32_t c : s) append_utf8(out, c);
  return out;
}

bool valid_utf8(std::string_view s) {
  size_t i = 0;
  const size_t n = s.size();
  while (i < n) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    size_t len;
    char32_t c;
    if (b0 < 0x80) {
      ++i;
      continue;
    } else if ((b0 & 0xE0) == 0xC0) {
      len = 2;
      c = b0 & 0x1F;
    } else if ((b0 & 0xF0) == 0xE0) {
      len = 3;
      c = b0 & 0x0F;
    } else if ((b0 & 0xF8) == 0xF0) {
      len = 4;
      c = b0 & 0x07;
    } else {
      return false;
    }
    if (i + len > n) return false;
    for (size_t k = 1; k < len; ++k) {
      const auto b = static_cast<unsigned char>(s[i + k]);
      if ((b & 0xC0) != 0x80) return false;
      c = (c << 6) | (b & 0x3F);
    }
    if ((len == 2 && c < 0x80) || (len == 3 && c < 0x800) || (len == 4 && c < 0x10000)) return false;
    if (c > 0x10FFFF || (c >= 0xD800 && c <= 0xDFFF)) return false;
    i += len;
  }
  return true;
}

size_t cp_len(std::string_view s) {
  size_t n = 0;
  for (char ch : s) n += (static_cast<unsigned char>(ch) & 0xC0) != 0x80;
  return n;
}

bool py_isalpha(char32_t c) { return in_ranges(kPyAlpha, c); }
bool py_isdigit(char32_t c) { return in_ranges(kPyDigit, c); }
bool py_isalnum(char32_t c) { return in_ranges(kPyAlnum, c); }
bool py_isspace(char32_t c) { return in_ranges(kPySpace, c); }
bool py_latin_name(char32_t c) { return in_ranges(kPyLatinName, c); }
bool py_is_upper_char(char32_t c) { return in_ranges(kPyUpperCased, c); }
bool py_is_lower_char(char32_t c) { return in_ranges(kPyLowerCased, c); }
bool py_is_title_char(char32_t c) { return in_ranges(kPyTitle, c); }

std::u32string py_upper(std::u32string_view s) {
  std::u32string out;
  out.reserve(s.size());
  for (char32_t c : s) {
    if (const CpMap* m = find_map(kPyUpper, c)) {
      out.append(m->out, m->n);
    } else {
      out.push_back(c);
    }
  }
  return out;
}

std::u32string py_casefold(std::u32string_view s) {
  std::u32string out;
  out.reserve(s.size());
  for (char32_t c : s) {
    if (const CpMap* m = find_map(kPyCasefold, c)) {
      out.append(m->out, m->n);
    } else {
      out.push_back(c);
    }
  }
  return out;
}

std::string py_casefold(std::string_view s) {
  bool ascii = true;
  for (char ch : s) ascii &= static_cast<unsigned char>(ch) < 0x80;
  if (ascii) {
    std::string out(s);
    for (char& ch : out) {
      if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return out;
  }
  return to_utf8(py_casefold(to_u32(s)));
}

// str.isupper(): every cased character is uppercase and there is at least one.
// Python: cased = isupper || islower || istitle (per char); returns False on any
// lowercase or titlecase char.
bool py_str_isupper(std::u32string_view s) {
  bool cased = false;
  for (char32_t c : s) {
    if (py_is_lower_char(c) || py_is_title_char(c)) return false;
    if (py_is_upper_char(c)) cased = true;
  }
  return cased;
}

bool py_str_isalpha(std::u32string_view s) {
  if (s.empty()) return false;
  for (char32_t c : s) {
    if (!py_isalpha(c)) return false;
  }
  return true;
}

bool py_str_isascii(std::string_view s) {
  for (char ch : s) {
    if (static_cast<unsigned char>(ch) >= 0x80) return false;
  }
  return true;
}

namespace {
template <class Pred>
std::string strip_impl(std::string_view s, Pred pred, bool left, bool right) {
  const std::u32string u = to_u32(s);
  size_t a = 0, b = u.size();
  if (left) {
    while (a < b && pred(u[a])) ++a;
  }
  if (right) {
    while (b > a && pred(u[b - 1])) --b;
  }
  return to_utf8(std::u32string_view(u).substr(a, b - a));
}
}  // namespace

std::string py_strip(std::string_view s) { return strip_impl(s, py_isspace, true, true); }
std::string py_rstrip(std::string_view s) { return strip_impl(s, py_isspace, false, true); }
std::string py_lstrip(std::string_view s) { return strip_impl(s, py_isspace, true, false); }
std::string py_strip_chars(std::string_view s, std::u32string_view chars) {
  return strip_impl(s, [&](char32_t c) { return chars.find(c) != std::u32string_view::npos; }, true, true);
}
std::string py_rstrip_chars(std::string_view s, std::u32string_view chars) {
  return strip_impl(s, [&](char32_t c) { return chars.find(c) != std::u32string_view::npos; }, false, true);
}
std::string py_lstrip_chars(std::string_view s, std::u32string_view chars) {
  return strip_impl(s, [&](char32_t c) { return chars.find(c) != std::u32string_view::npos; }, true, false);
}

std::vector<std::string> py_split_ws(std::string_view s) {
  std::vector<std::string> out;
  std::string cur;
  bool in_word = false;
  for (size_t i = 0; i < s.size();) {
    const size_t start = i;
    const char32_t c = decode(s, i);
    if (py_isspace(c)) {
      if (in_word) {
        out.push_back(std::move(cur));
        cur.clear();
        in_word = false;
      }
    } else {
      cur.append(s.substr(start, i - start));
      in_word = true;
    }
  }
  if (in_word) out.push_back(std::move(cur));
  return out;
}

std::vector<std::string> py_split(std::string_view s, std::string_view sep) {
  std::vector<std::string> out;
  size_t pos = 0;
  while (true) {
    const size_t at = s.find(sep, pos);
    if (at == std::string_view::npos) {
      out.emplace_back(s.substr(pos));
      return out;
    }
    out.emplace_back(s.substr(pos, at - pos));
    pos = at + sep.size();
  }
}

std::string py_replace(std::string_view s, std::string_view from, std::string_view to) {
  if (from.empty()) return std::string(s);
  std::string out;
  size_t pos = 0;
  while (true) {
    const size_t at = s.find(from, pos);
    if (at == std::string_view::npos) {
      out.append(s.substr(pos));
      return out;
    }
    out.append(s.substr(pos, at - pos));
    out.append(to);
    pos = at + from.size();
  }
}

std::string join(const std::vector<std::string>& parts, std::string_view sep) {
  std::string out;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i) out.append(sep);
    out.append(parts[i]);
  }
  return out;
}

std::u32string py_ignorecase_set(char letter) {
  if (letter >= 'A' && letter <= 'Z') letter = static_cast<char>(letter - 'A' + 'a');
  if (letter < 'a' || letter > 'z') return std::u32string(1, static_cast<char32_t>(letter));
  return std::u32string(kPyIgnoreCase[letter - 'a']);
}

bool moses_is_alpha(char32_t c) { return in_ranges(kMosesIsAlpha, c); }
bool moses_is_sc(char32_t c) { return in_ranges(kMosesIsSc, c); }

bool moses_is_cjk(char32_t c) {
  static constexpr CpRange kCjk[] = {{4352, 4607},     {11904, 42191},   {43072, 43135},
                                     {44032, 55215},   {63744, 64255},   {65072, 65103},
                                     {65381, 65500},   {94208, 101119},  {110592, 110895},
                                     {110960, 111359}, {131072, 196607}};
  for (const auto& r : kCjk) {
    if (c <= r.hi) return c >= r.lo;
  }
  return false;
}

}  // namespace kt2n::text
