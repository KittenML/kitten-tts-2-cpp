#pragma once
#include "json.hpp"
#include <map>
#include <string>

namespace kitten {
using json = nlohmann::ordered_json;
struct assets {
    json config;
    std::string model, decoder, voices;
};
assets resolve_assets(const std::map<std::string, std::string> & args);
}
