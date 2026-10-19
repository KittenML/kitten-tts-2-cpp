// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string>
#include <string_view>
namespace kitten_voice {
inline constexpr const char* rule_version = "voice-20261015-3";
// Structured values only. Ordinary prose and protected identifiers are
// retained. Thread safe. All number calculations use decimal strings / exact
// integers.
std::string normalize(std::string_view text, std::string_view language);
void warm();
// Protect whitespace inside structured values while the caller plans chunks.
// U+FFFF is internal; input already containing it is returned unchanged.
std::string prepare_for_segmentation(std::string_view text,
                                     std::string_view language);
}  // namespace kitten_voice
extern "C" {
// Returned strings are allocated by the library. NULL = invalid input/error.
char* kitten_voice_normalize(const char* text, long long length,
                             const char* language, long long* out_length);
void kitten_voice_free(void* output);
int kitten_voice_warm();
char* kitten_voice_prepare(const char* text, long long length,
                           const char* language, long long* out_length);
}
