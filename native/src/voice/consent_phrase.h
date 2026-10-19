// L11: production's consent-recording text check, ported (kittenml-api ebcdd3f kitten_tts_09/runtime.py
// _validate_consent_transcript :11054, _phrase_words :11050, CUSTOM_VOICE_CONSENT_PHRASE :2437):
//
//   language.casefold().split("-", 1)[0] != "en"            -> 208 unsupported_consent_language (param language)
//   observed = " ".join(re.findall(r"[a-z0-9]+", transcript.casefold()))
//   expected = the same over the consent phrase
//   {"owner","voice","consent","synthetic"} not all in observed.split()
//   or difflib.SequenceMatcher(None, expected, observed).ratio() < 0.86
//                                                           -> 207 consent_phrase_mismatch (param recording)
//
// casefold: only the code points whose Python casefold contains an ASCII letter or digit matter for [a-z0-9]
// (A-Z plus 19 others, e.g. U+00DF -> "ss", U+212A -> "k", U+FB01 -> "fi"; generated with Python 3 over all of
// Unicode); every other non-ASCII code point folds to something the regex skips. SequenceMatcher is a line-for-
// line port (autojunk on: elements of b more frequent than len(b)//100 + 1 are dropped when len(b) >= 200).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kt2n::voice {

extern const char* const kConsentPhrase;  // production default (the env override is unset in all three deploys)
constexpr double kConsentPhraseMinRatio = 0.86;

struct ConsentTextVerdict {
  int32_t status = 0;  // 0, 207 or 208
  std::string code;    // "", "consent_phrase_mismatch", "unsupported_consent_language"
  std::string param;   // "", "recording", "language"
  std::string message;
  double similarity = 0.0;
};

// transcript: the stripped consent transcript (VoiceService::validate_transcript first). language: as sent.
ConsentTextVerdict check_consent_text(const std::string& transcript, const std::string& language,
                                      const std::string& phrase = kConsentPhrase);
// re.findall(r"[a-z0-9]+", s.casefold()) on UTF-8 input (malformed bytes are skipped like separators).
std::vector<std::string> phrase_words(const std::string& utf8);
// difflib.SequenceMatcher(None, a, b).ratio() for byte strings (the joined words are ASCII).
double sequence_matcher_ratio(const std::string& a, const std::string& b);

}  // namespace kt2n::voice
