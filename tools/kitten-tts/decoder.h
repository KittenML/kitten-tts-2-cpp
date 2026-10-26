#pragma once
#include "json.hpp"
#include <memory>
#include <string>
#include <vector>
namespace kitten {
class decoder {
    struct impl;
    std::unique_ptr<impl> p;
public:
    // One voice's S3 conditioning on the decoder device. The decoder keeps no voice, so one decoder can serve several voices.
    struct voice;
    // device is "cpu" (default) or "cuda". A CUDA decoder needs a TorchScript file
    // exported with `export.py --device cuda` and a CUDA build of LibTorch.
    decoder(const std::string & path, int threads, int seed, const std::string & device = "cpu");
    ~decoder();
    void seed(int value);
    // Parse one voice's S3 conditioning (voices.json entry) and copy it to the decoder device.
    std::shared_ptr<const voice> prepare_voice(const nlohmann::ordered_json & conditioning) const;
    std::vector<float> decode(const std::vector<int> & ids, const voice & v);
    // Decoder work that runs on the CPU, for reports.
    std::vector<std::string> cpu_stages() const;
};
}
