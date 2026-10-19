#include "assets.h"
#include "download.h"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
#if KITTEN_TTS_NATIVE
#include "kt2n/sha256.h"
#include <array>
#endif

namespace kitten {
namespace fs = std::filesystem;
static std::string env(const char * name, const std::string & fallback = {}) {
    const char * value = std::getenv(name);
    return value && *value ? value : fallback;
}
static void check_path(const std::string & value) {
    if (value.empty() || value.front() == '/' || value.back() == '/') throw std::runtime_error("invalid repository path: " + value);
    for (unsigned char c : value) {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '/')) throw std::runtime_error("invalid repository path: " + value);
    }
    for (const auto & part : fs::path(value)) {
        if (part == "." || part == "..") throw std::runtime_error("invalid repository path: " + value);
    }
    if (value.find("//") != std::string::npos) throw std::runtime_error("invalid repository path: " + value);
}
assets resolve_assets(const std::map<std::string, std::string> & args) {
    auto get = [&](const std::string & key, const std::string & fallback) { auto it = args.find(key); return it == args.end() ? fallback : it->second; };
    bool local = args.count("--assets");
    if (local && (args.count("--repo") || args.count("--revision"))) throw std::runtime_error("--assets cannot be combined with --repo or --revision");
    std::string repo = get("--repo", "KittenML/kitten-tts-2"), revision = get("--revision", "main");
    check_path(repo); check_path(revision);
    // Encode slashes in revision names so cache paths cannot collide.
    std::string cache_revision = revision;
    for (char & c : cache_revision) if (c == '/') c = '@';
    fs::path root = local ? fs::path(args.at("--assets")) : fs::path(get("--cache-dir", env("XDG_CACHE_HOME", env("HOME", ".") + "/.cache") + "/kitten-tts")) / repo / cache_revision;
    common_download_opts opts;
    opts.offline = args.count("--offline");
    opts.bearer_token = env("HF_TOKEN");
    std::string base = env("HF_ENDPOINT", "https://huggingface.co") + "/" + repo + "/resolve/" + revision + "/";
    auto fetch = [&](const std::string & file) {
        check_path(file);
        auto path = (root / file).string();
        if (!local) {
            int status = common_download_file_single(base + file, path, opts);
            if (status != 304 && (status < 200 || status >= 300)) throw std::runtime_error("cannot download " + file + " (HTTP " + std::to_string(status) + ") from " + repo);
        }
        if (!fs::is_regular_file(path)) throw std::runtime_error("missing asset: " + path);
        return path;
    };
    assets result;
    std::ifstream config_file(fetch("config.json"));
    config_file >> result.config;
    if (result.config.at("type") != "KITTEN2") throw std::runtime_error("only KITTEN2 is supported");
#if KITTEN_TTS_NATIVE
    if (get("--backend", "cpu") == "cuda") {
        if (get("--decoder", "default") != "default") throw std::runtime_error("CUDA supports only the default decoder; student decoders require --backend cpu");
        if (!result.config.contains("native")) throw std::runtime_error("model config has no native CUDA assets; prepare a native bundle or select a repository with a native manifest");
        const auto & manifest = result.config.at("native");
        if (manifest.at("version") != 1 || manifest.value("decoder", "default") != "default") throw std::runtime_error("unsupported native asset manifest");
        auto asset = [&](const json & entry) {
            const auto path = fetch(entry.at("file").get<std::string>());
            if (fs::file_size(path) != entry.at("size").get<uint64_t>()) throw std::runtime_error("native asset size mismatch: " + path);
            const auto expected = entry.at("sha256").get<std::string>();
            if (!kt2n::is_sha256_hex(expected)) throw std::runtime_error("invalid native asset SHA256: " + path);
            kt2n::Sha256 hash;
            std::ifstream input(path, std::ios::binary);
            std::array<char, 65536> buffer;
            while (input) {
                input.read(buffer.data(), buffer.size());
                hash.update(buffer.data(), static_cast<size_t>(input.gcount()));
            }
            if (!input.eof() || hash.hexdigest() != expected) throw std::runtime_error("native asset SHA256 mismatch: " + path);
            return path;
        };
        result.model = asset(manifest.at("lm"));
        result.decoder = asset(manifest.at("acoustic"));
        result.voices = asset(manifest.at("voices"));
        result.tokenizer = asset(manifest.at("tokenizer"));
        auto directory = manifest.at("text").at("directory").get<std::string>();
        check_path(directory);
        if (manifest.at("text").at("files").empty()) throw std::runtime_error("native text asset list is empty");
        for (const auto & entry : manifest.at("text").at("files")) {
            const auto file = entry.at("file").get<std::string>();
            if (file.rfind(directory + "/", 0) != 0) throw std::runtime_error("native text asset is outside its directory");
            asset(entry);
        }
        result.text_data = (root / directory).string();
        if (manifest.contains("levers")) result.levers = asset(manifest.at("levers"));
        if (args.count("--reference-audio")) {
            if (!manifest.contains("cloning")) throw std::runtime_error("model config has no native cloning assets; prepare the bundle with --with-cloning");
            const auto & cloning = manifest.at("cloning");
            if (cloning.at("version") != 1) throw std::runtime_error("unsupported native cloning manifest");
            const auto clone_dir = cloning.at("directory").get<std::string>();
            check_path(clone_dir);
            std::vector<std::string> files;
            for (const auto & entry : cloning.at("files")) {
                const auto file = entry.at("file").get<std::string>();
                if (file.rfind(clone_dir + "/", 0) != 0) throw std::runtime_error("native cloning asset is outside its directory");
                asset(entry);
                files.push_back(file);
            }
            for (const auto * name : {"s3gen.safetensors", "voicebuild_consts.safetensors", "voicebuild_pyannote.safetensors"}) {
                if (std::find(files.begin(), files.end(), clone_dir + "/" + name) == files.end()) throw std::runtime_error(std::string("missing native cloning asset: ") + name);
            }
            result.voice_build_dir = (root / clone_dir).string();
        }
        return result;
    }
#endif
    bool need_model = !args.count("--decode-tokens"), need_decoder = !args.count("--tokens-only") || args.count("--decode-tokens");
    if (!result.config.contains("cpp")) {
        if (!local) throw std::runtime_error("model config has no cpp assets; the model publisher must upload the prepared native assets and cpp manifest");
        if (args.count("--decoder") && args.at("--decoder") != result.config.value("cpp_decoder", "default")) throw std::runtime_error("this legacy asset directory contains a different decoder");
        result.voices = fetch("voices.json");
        if (need_decoder) result.decoder = fetch("decoder.pt");
        if (need_model) result.model = args.count("--model") ? args.at("--model") : fetch("model-tq2_1.gguf");
        return result;
    }
    const auto & cpp = result.config.at("cpp");
    if (cpp.at("version") != 1) throw std::runtime_error("unsupported cpp asset manifest version");
    std::string name = get("--decoder", result.config.value("default_decoder", "default"));
    if (!cpp.at("decoders").contains(name)) throw std::runtime_error("unknown decoder: " + name);
    const auto & decoder = cpp.at("decoders").at(name);
    auto asset = [&](const json & entry) {
        auto path = fetch(entry.at("file").get<std::string>());
        if (fs::file_size(path) != entry.at("size").get<uint64_t>()) throw std::runtime_error("asset size mismatch; remove and download again: " + path);
        return path;
    };
    result.voices = asset(decoder.at("voices"));
    if (need_model) result.model = args.count("--model") ? args.at("--model") : asset(cpp.at("gguf"));
    if (need_decoder) result.decoder = asset(decoder.at("torchscript"));
    return result;
}
}
