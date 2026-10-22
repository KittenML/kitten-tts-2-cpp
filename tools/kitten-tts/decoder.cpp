#include "decoder.h"
#include "json.hpp"
#include <torch/script.h>
#include <torch/cuda.h>
#include <torch/version.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 9)
#include <c10/core/CachingDeviceAllocator.h>
#define KITTEN_DEVICE_ALLOCATOR_STATS 1
#endif
namespace kitten {
namespace {
// export.py stores this next to the graph. Files without it were traced on the CPU.
constexpr const char * metadata_file = "kitten_decoder.json";

bool env_flag(const char * name, bool fallback) {
    const char * v = std::getenv(name);
    if (!v || !*v) return fallback;
    return std::string(v) != "0";
}

void report_cuda_memory(const torch::Device & device, const char * when) {
#ifdef KITTEN_DEVICE_ALLOCATOR_STATS
    using c10::CachingAllocator::StatType;
    auto stats = c10::getDeviceAllocator(c10::DeviceType::CUDA)->getDeviceStats(device.has_index() ? device.index() : 0);
    const auto & a = stats.allocated_bytes[size_t(StatType::AGGREGATE)];
    const auto & r = stats.reserved_bytes[size_t(StatType::AGGREGATE)];
    const double mib = 1024.0 * 1024.0;
    std::fprintf(stderr, "kitten-tts: decoder %s %s: allocated %.1f MiB (peak %.1f), reserved %.1f MiB (peak %.1f)\n",
        device.str().c_str(), when, a.current / mib, a.peak / mib, r.current / mib, r.peak / mib);
#else
    (void) device; (void) when;
#endif
}
}

struct decoder::voice { torch::Tensor prompt_token, prompt_feat, embedding; };

struct decoder::impl {
    int threads;
    torch::Device device;
    torch::jit::Module module;
    bool empty_cache = false;
    std::string rng = "device";
    bool sinegen_on_device = false;
    impl(const std::string & path, torch::Device dev, int n_threads): threads(n_threads), device(dev) {
        torch::jit::ExtraFilesMap extra{{metadata_file, ""}};
        // A CUDA trace keeps some small constants on the CPU (s3gen draws its phase noise there), so load tensors where they were saved.
        module = device.is_cuda() ? torch::jit::load(path, {}, extra) : torch::jit::load(path, device, extra);
        module.eval();
        std::string traced = "cpu";
        if (!extra[metadata_file].empty()) {
            auto m = nlohmann::json::parse(extra[metadata_file]);
            traced = m.value("device", "cpu");
            rng = m.value("rng", "device");
            sinegen_on_device = m.value("sinegen", "cpu") == "device";
        }
        if (traced != c10::DeviceTypeName(device.type(), true))
            throw std::runtime_error("decoder " + path + " was exported for " + traced + ", not " + device.str() +
                "; export a matching decoder with export.py --device " + c10::DeviceTypeName(device.type(), true) +
                " or change --decoder-device");
    }
};

decoder::decoder(const std::string & path, int threads, int seed, const std::string & device) {
    if (device != "cpu" && device != "cuda") throw std::runtime_error("--decoder-device must be cpu or cuda");
    at::set_num_threads(threads);
    if (device == "cuda") {
        if (!torch::cuda::is_available())
            throw std::runtime_error("--decoder-device cuda needs a CUDA build of LibTorch and a visible GPU");
        // Keep GEMMs and convolutions in full FP32, as on the CPU. cuDNN allows TF32 convolutions on SM80+ by default.
        at::globalContext().setAllowTF32CuBLAS(false);
        at::globalContext().setAllowTF32CuDNN(false);
        // No profiling runs: the profiling executor made the first two CUDA calls take 10 s and 18 s on an RTX 4090.
        torch::jit::getProfilingMode() = false;
    }
    at::manual_seed(seed);
    // Pin cuda:0, where the trace keeps its weights. A bare "cuda" means the thread's current GPU, which GGML moves to --main-gpu.
    p = std::make_unique<impl>(path, device == "cuda" ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU), threads);
    if (p->device.is_cuda()) {
        p->empty_cache = env_flag("KITTEN_DECODER_EMPTY_CACHE", false);
        std::fprintf(stderr, "kitten-tts: decoder on %s (LibTorch %s, TF32 off, TorchScript graph passes off)\n", p->device.str().c_str(), TORCH_VERSION);
        report_cuda_memory(p->device, "loaded");
    }
}
decoder::~decoder() = default;
// LibTorch's thread count is process-wide, so set this decoder's count again. at::manual_seed also seeds every CUDA generator when CUDA is available.
void decoder::seed(int value) { at::set_num_threads(p->threads); at::manual_seed(value); }
std::shared_ptr<const decoder::voice> decoder::prepare_voice(const nlohmann::ordered_json & j) const {
    c10::InferenceMode guard;
    auto ref=j.at("prompt_token").at(0).get<std::vector<int64_t>>();
    std::vector<float> feat;
    for(auto & row:j.at("prompt_feat").at(0)) {auto v=row.get<std::vector<float>>();feat.insert(feat.end(),v.begin(),v.end());}
    auto emb=j.at("embedding").at(0).get<std::vector<float>>();
    auto opts=torch::TensorOptions().dtype(torch::kFloat32);
    // from_blob does not own the vectors: copy to the device, or clone on the CPU
    auto keep=[&](const torch::Tensor & t){return p->device.is_cuda()?t.to(p->device):t.clone();};
    return std::make_shared<voice>(voice{keep(torch::from_blob(ref.data(),{1,int64_t(ref.size())},torch::kInt64)),
        keep(torch::from_blob(feat.data(),{1,int64_t(feat.size()/80),80},opts)),
        keep(torch::from_blob(emb.data(),{1,int64_t(emb.size())},opts))});
}
std::vector<std::string> decoder::cpu_stages() const {
    if(!p->device.is_cuda()) return {"S3 decoder (CPU LibTorch)"};
    std::vector<std::string> stages={"S3 decoder: SineGen phase draw (9 values per chunk, CPU generator, copied to the GPU)"};
    if(p->rng=="cpu") stages.push_back("S3 decoder: noise draws (CPU generator, copied to the GPU; --rng cpu export)");
    if(!p->sinegen_on_device) stages.push_back("S3 decoder: SineGen source buffer zero-fill (9 floats per sample, copied to the GPU; decoder.pt exported without the device-side SineGen)");
    stages.push_back("S3 decoder: token upload and waveform copy to host");
    return stages;
}
std::vector<float> decoder::decode(const std::vector<int> & ids, const voice & v) {
    if(ids.empty()) return {};
    c10::InferenceMode guard;
    // On CUDA, skip TorchScript graph passes. Constant pooling and CSE compare every pair of similar weight constants with a GPU equal() and a sync, which made the first call take 4 s instead of 0.8 s. Thread-local, so it is set here.
    torch::jit::GraphOptimizerEnabledGuard optimize(!p->device.is_cuda() && torch::jit::getGraphExecutorOptimize());
    std::vector<int64_t> tokens(ids.begin(),ids.end()); tokens.insert(tokens.end(),3,4299);
    const auto & dev=p->device;  // .to() is a no-op for CPU tensors on a CPU decoder
    auto output=p->module.forward({torch::from_blob(tokens.data(),{1,int64_t(tokens.size())},torch::kInt64).to(dev),
        v.prompt_token,v.prompt_feat,v.embedding}).toTensor().to(torch::kCPU).contiguous();
    if(dev.is_cuda()) {
        report_cuda_memory(dev, "after decode");
#ifdef KITTEN_DEVICE_ALLOCATOR_STATS
        if(p->empty_cache) c10::getDeviceAllocator(c10::DeviceType::CUDA)->emptyCache();
#endif
    }
    return std::vector<float>(output.data_ptr<float>(),output.data_ptr<float>()+output.numel());
}
}
