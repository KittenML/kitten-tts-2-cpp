#include "decoder.h"
#include "json.hpp"
#include <torch/script.h>
#include <ATen/Parallel.h>
namespace kitten {
struct decoder::impl { int threads; torch::jit::Module module; explicit impl(const std::string & path, int n_threads): threads(n_threads), module(torch::jit::load(path,torch::kCPU)) { module.eval(); } };
decoder::decoder(const std::string & path,int threads,int seed) {
    at::set_num_threads(threads); at::manual_seed(seed); p=std::make_unique<impl>(path,threads);
}
decoder::~decoder() = default;
void decoder::seed(int value) { at::set_num_threads(p->threads); at::manual_seed(value); }
std::vector<float> decoder::decode(const std::vector<int> & ids,const std::string & conditioning_json) {
    if(ids.empty()) return {};
    c10::InferenceMode guard;
    auto j=nlohmann::json::parse(conditioning_json);
    std::vector<int64_t> tokens(ids.begin(),ids.end()); tokens.insert(tokens.end(),3,4299);
    auto ref=j.at("prompt_token").at(0).get<std::vector<int64_t>>();
    std::vector<float> feat;
    for(auto & row:j.at("prompt_feat").at(0)) {auto v=row.get<std::vector<float>>();feat.insert(feat.end(),v.begin(),v.end());}
    auto emb=j.at("embedding").at(0).get<std::vector<float>>();
    auto opts=torch::TensorOptions().dtype(torch::kFloat32);
    auto output=p->module.forward({torch::from_blob(tokens.data(),{1,int64_t(tokens.size())},torch::kInt64),
        torch::from_blob(ref.data(),{1,int64_t(ref.size())},torch::kInt64),
        torch::from_blob(feat.data(),{1,int64_t(feat.size()/80),80},opts),
        torch::from_blob(emb.data(),{1,int64_t(emb.size())},opts)}).toTensor().contiguous();
    return std::vector<float>(output.data_ptr<float>(),output.data_ptr<float>()+output.numel());
}
}
