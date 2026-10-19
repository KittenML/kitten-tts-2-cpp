// L11: production's consent-recording text check (src/voice/consent_phrase.{h,cc}) against Python's own answers
// (re + difflib, generated into consent_vectors.inc): the status and the SequenceMatcher ratio, bit for bit.
#include <cstdio>
#include <string>

#include "consent_phrase.h"

namespace {
struct Case {
  const char* transcript;
  const char* language;
  int status;
  double ratio;
};
const Case kCases[] = {
#include "consent_vectors.inc"
};
}  // namespace

int main() {
  using namespace kt2n::voice;
  int fails = 0, n = 0;
  for (const Case& c : kCases) {
    ++n;
    const ConsentTextVerdict v = check_consent_text(c.transcript, c.language);
    const bool ok = v.status == c.status && (c.status == 208 || v.similarity == c.ratio);
    if (!ok) {
      ++fails;
      std::fprintf(stderr, "FAIL case %d: status %d (want %d) ratio %.17g (want %.17g) lang '%s'\n", n, v.status,
                   c.status, v.similarity, c.ratio, c.language);
    }
    if (v.status == 207 && (v.code != "consent_phrase_mismatch" || v.param != "recording")) ++fails;
    if (v.status == 208 && (v.code != "unsupported_consent_language" || v.param != "language")) ++fails;
  }
  // direct difflib cases (Python: SequenceMatcher(None, a, b).ratio())
  struct R {
    const char *a, *b;
    double r;
  } rs[] = {{"", "", 1.0}, {"abc", "", 0.0}, {"abcd", "bcde", 0.75}, {"qabxcd", "abycdf", 0.6666666666666666}};
  for (const R& r : rs)
    if (sequence_matcher_ratio(r.a, r.b) != r.r) {
      ++fails;
      std::fprintf(stderr, "FAIL ratio('%s','%s') = %.17g want %.17g\n", r.a, r.b, sequence_matcher_ratio(r.a, r.b), r.r);
    }
  if (fails) {
    std::fprintf(stderr, "test_consent_phrase: %d failure(s) of %d\n", fails, n);
    return 1;
  }
  std::printf("test_consent_phrase: PASS (%d cases)\n", n);
  return 0;
}
