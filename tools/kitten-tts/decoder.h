#pragma once
#include <memory>
#include <string>
#include <vector>
namespace kitten {
class decoder {
    struct impl;
    std::unique_ptr<impl> p;
public:
    decoder(const std::string & path, int threads, int seed);
    ~decoder();
    void seed(int value);
    std::vector<float> decode(const std::vector<int> & ids, const std::string & conditioning_json);
};
}
