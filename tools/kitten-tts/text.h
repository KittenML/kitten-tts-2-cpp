#pragma once
#include <string>
#include <vector>
#include <kitten_text_processing/normalizer.hpp>
namespace kitten {
std::string normalize(const std::string &, const kitten_text_processing::Normalizer &, bool enabled = true);
bool has_expression(const std::string &);
size_t characters(const std::string &);
std::vector<std::string> split(const std::string &, int max_chars = 380, int min_chars = 130);
std::vector<float> join(const std::vector<std::vector<float>> &, float gap = 0.16f);
}
