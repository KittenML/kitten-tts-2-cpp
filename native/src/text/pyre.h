// L9a text: PCRE2 (UTF + UCP, JIT) with Python `re` call semantics.
//
// Patterns are written in PCRE2 syntax by hand from production's Python
// patterns; the translation rules (checked by gen_unicode_tables.py on the
// reference interpreter, Unicode 14.0.0 on both sides with PCRE2 10.42):
//   Python \w, \d, \b      -> PCRE2 \w, \d, \b under UCP (identical sets)
//   Python \s / \S         -> PCRE2 \s / \S under UCP EXCEPT U+1C..U+1F and
//                             U+180E; patterns use kPyS / kPyNotS instead
//   Python \Z              -> PCRE2 \z
//   re.IGNORECASE (ASCII)  -> explicit classes from py_ignorecase_set()
// sub()/finditer() follow Python 3.7+: an empty match right after a previous
// match is allowed, and the next search after an empty match may not return
// an empty match at the same position.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

struct pcre2_real_code_8;

namespace kt2n::text {

// Python's whitespace set as a PCRE2 class body (without brackets).
inline constexpr const char* kPySClass =
    "\\t\\n\\x{0B}\\x{0C}\\r\\x{1C}-\\x{1F} \\x{85}\\x{A0}\\x{1680}\\x{2000}-\\x{200A}"
    "\\x{2028}\\x{2029}\\x{202F}\\x{205F}\\x{3000}";
// Expand "\s" and "\S" (outside character classes) and "[...\s...]" inside a
// class into the Python set. Only these two escapes are rewritten.
std::string py_pattern(std::string_view pattern);

struct Match {
  std::string_view subject;
  // Byte offsets per group; -1 when the group did not participate.
  std::vector<ptrdiff_t> starts, ends;
  size_t start(int g = 0) const { return static_cast<size_t>(starts[g]); }
  size_t end(int g = 0) const { return static_cast<size_t>(ends[g]); }
  bool has(int g) const { return g < static_cast<int>(starts.size()) && starts[g] >= 0; }
  std::string_view group(int g = 0) const {
    return has(g) ? subject.substr(start(g), end(g) - start(g)) : std::string_view();
  }
  size_t groups() const { return starts.size() - 1; }
};

class Regex {
 public:
  // `pattern` is PCRE2 syntax; run it through py_pattern() first when it uses
  // \s or \S. Throws std::runtime_error on a compile error.
  explicit Regex(std::string_view pattern, uint32_t options = 0);
  ~Regex();
  Regex(const Regex&) = delete;
  Regex& operator=(const Regex&) = delete;

  // re.search(subject, pos): first match at or after byte offset `pos`.
  bool search(std::string_view subject, size_t pos, Match* m, uint32_t match_options = 0) const;
  bool search(std::string_view subject, Match* m = nullptr) const { return search(subject, 0, m); }
  // re.match: anchored at `pos`.
  bool match(std::string_view subject, size_t pos = 0, Match* m = nullptr) const;
  // re.fullmatch.
  bool fullmatch(std::string_view subject, Match* m = nullptr) const;
  std::vector<Match> finditer(std::string_view subject) const;
  using Repl = std::function<std::string(const Match&)>;
  std::string sub(std::string_view subject, const Repl& repl) const;
  // Replacement template with \1..\9 group references (unset group -> "").
  std::string sub(std::string_view subject, std::string_view repl) const;
  // re.split(pattern, subject) with Python semantics (groups are included).
  std::vector<std::string> split(std::string_view subject) const;

 private:
  pcre2_real_code_8* code_ = nullptr;
  pcre2_real_code_8* anchored_ = nullptr;  // same pattern, ANCHORED | ENDANCHORED
  bool jit_ = false;
};

}  // namespace kt2n::text
