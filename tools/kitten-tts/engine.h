#pragma once
#include "assets.h"
#include <functional>
#include <memory>
#include <vector>

namespace kitten {
struct generation_result {
    std::vector<float> audio;
    json report;
};
class engine {
    struct impl;
    std::unique_ptr<impl> p;
public:
    explicit engine(const std::map<std::string, std::string> & options);
    ~engine();
    json metadata() const;
    std::string normalize(const std::string & text);
    generation_result generate(const std::map<std::string, std::string> & options,
        const std::function<bool(const std::vector<float> &)> & on_chunk = {});
};
}
