// L11: consent-recording text check (see consent_phrase.h).
#include "consent_phrase.h"

#include <array>
#include <unordered_map>

namespace kt2n::voice {

const char* const kConsentPhrase =
    "I am the owner of this voice and I consent to its use to create a synthetic voice.";

namespace {

// Python str.casefold() restricted to what [a-z0-9] can see: ASCII letters/digits of the fold, 0 = separator.
// The 19 non-ASCII code points whose fold contains an ASCII letter or digit (Python 3, all of Unicode).
struct Fold {
  uint32_t cp;
  const char* ascii;  // the fold with every non-ASCII code point replaced by ' '
};
constexpr Fold kFolds[] = {
    {0x00DF, "ss"},  {0x0130, "i "},  {0x0149, " n"},  {0x017F, "s"},   {0x01F0, "j "},
    {0x1E96, "h "},  {0x1E97, "t "},  {0x1E98, "w "},  {0x1E99, "y "},  {0x1E9A, "a "},
    {0x1E9E, "ss"},  {0x212A, "k"},   {0xFB00, "ff"},  {0xFB01, "fi"},  {0xFB02, "fl"},
    {0xFB03, "ffi"}, {0xFB04, "ffl"}, {0xFB05, "st"},  {0xFB06, "st"},
};

// UTF-8 decode, one code point at a time; malformed bytes give U+FFFD (a separator either way).
uint32_t next_cp(const std::string& s, size_t& i) {
  const unsigned char c = (unsigned char)s[i];
  int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
  if (n == 0 || i + n > s.size()) {
    ++i;
    return 0xFFFD;
  }
  uint32_t cp = n == 1 ? c : n == 2 ? (c & 0x1f) : n == 3 ? (c & 0x0f) : (c & 0x07);
  for (int k = 1; k < n; ++k) {
    const unsigned char cc = (unsigned char)s[i + k];
    if ((cc >> 6) != 2) {
      ++i;
      return 0xFFFD;
    }
    cp = (cp << 6) | (cc & 0x3f);
  }
  i += n;
  return cp;
}

void fold_into(uint32_t cp, std::string& out) {
  if (cp < 0x80) {
    char c = (char)cp;
    if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    out.push_back(((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) ? c : ' ');
    return;
  }
  for (const Fold& f : kFolds)
    if (f.cp == cp) {
      out += f.ascii;
      return;
    }
  out.push_back(' ');
}

}  // namespace

std::vector<std::string> phrase_words(const std::string& utf8) {
  std::string folded;
  folded.reserve(utf8.size());
  for (size_t i = 0; i < utf8.size();) fold_into(next_cp(utf8, i), folded);
  std::vector<std::string> words;
  std::string cur;
  for (char c : folded) {
    if (c == ' ') {
      if (!cur.empty()) words.push_back(std::move(cur)), cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) words.push_back(std::move(cur));
  return words;
}

double sequence_matcher_ratio(const std::string& a, const std::string& b) {
  const int la = (int)a.size(), lb = (int)b.size();
  if (la + lb == 0) return 1.0;
  // __chain_b: element -> ascending indices in b; isjunk None; autojunk drops popular elements (n >= 200)
  std::array<std::vector<int>, 256> b2j;
  for (int j = 0; j < lb; ++j) b2j[(unsigned char)b[j]].push_back(j);
  if (lb >= 200) {
    const size_t ntest = (size_t)(lb / 100 + 1);
    for (auto& v : b2j)
      if (v.size() > ntest) v.clear();
  }
  // get_matching_blocks: only the total matched size is needed for ratio()
  struct Range {
    int alo, ahi, blo, bhi;
  };
  std::vector<Range> queue{{0, la, 0, lb}};
  long matches = 0;
  std::unordered_map<int, int> j2len, newj2len;
  while (!queue.empty()) {
    const Range r = queue.back();
    queue.pop_back();
    // find_longest_match(alo, ahi, blo, bhi)
    int besti = r.alo, bestj = r.blo, bestsize = 0;
    j2len.clear();
    for (int i = r.alo; i < r.ahi; ++i) {
      newj2len.clear();
      for (int j : b2j[(unsigned char)a[i]]) {
        if (j < r.blo) continue;
        if (j >= r.bhi) break;
        auto it = j2len.find(j - 1);
        const int k = (it == j2len.end() ? 0 : it->second) + 1;
        newj2len[j] = k;
        if (k > bestsize) besti = i - k + 1, bestj = j - k + 1, bestsize = k;
      }
      std::swap(j2len, newj2len);
    }
    // extend with equal non-junk neighbours (bjunk is empty: popular elements count as non-junk here); the two
    // junk-extension loops never run
    while (besti > r.alo && bestj > r.blo && a[besti - 1] == b[bestj - 1]) --besti, --bestj, ++bestsize;
    while (besti + bestsize < r.ahi && bestj + bestsize < r.bhi && a[besti + bestsize] == b[bestj + bestsize])
      ++bestsize;
    if (bestsize) {
      matches += bestsize;
      if (r.alo < besti && r.blo < bestj) queue.push_back({r.alo, besti, r.blo, bestj});
      if (besti + bestsize < r.ahi && bestj + bestsize < r.bhi)
        queue.push_back({besti + bestsize, r.ahi, bestj + bestsize, r.bhi});
    }
  }
  return 2.0 * (double)matches / (double)(la + lb);
}

ConsentTextVerdict check_consent_text(const std::string& transcript, const std::string& language,
                                      const std::string& phrase) {
  ConsentTextVerdict v;
  // language.casefold().split("-", 1)[0] != "en": no non-ASCII code point folds to exactly "e" or "n", so only an
  // ASCII "en" (any case) before the first '-' passes.
  std::string head;
  for (size_t i = 0; i < language.size() && language[i] != '-'; ++i) {
    char c = language[i];
    if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    head.push_back(c);
  }
  if (head != "en") {
    v.status = 208;
    v.code = "unsupported_consent_language";
    v.param = "language";
    v.message = "KittenTTS 0.9 custom voices currently support English consent recordings.";
    return v;
  }
  auto join = [](const std::vector<std::string>& w) {
    std::string s;
    for (size_t i = 0; i < w.size(); ++i) s += (i ? " " : "") + w[i];
    return s;
  };
  const std::vector<std::string> obs_words = phrase_words(transcript);
  const std::string observed = join(obs_words), expected = join(phrase_words(phrase));
  bool required = true;
  for (const char* r : {"owner", "voice", "consent", "synthetic"}) {
    bool found = false;
    for (const auto& w : obs_words) found = found || w == r;
    required = required && found;
  }
  v.similarity = sequence_matcher_ratio(expected, observed);
  if (!required || v.similarity < kConsentPhraseMinRatio) {
    v.status = 207;
    v.code = "consent_phrase_mismatch";
    v.param = "recording";
    v.message = "The recording does not match the required KittenML consent phrase.";
  }
  return v;
}

}  // namespace kt2n::voice
