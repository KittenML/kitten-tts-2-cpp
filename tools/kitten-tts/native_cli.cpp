#include "native_cli.h"
#include "assets.h"
#include "kt2n/engine.h"
#include "native_stats.h"

#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace kitten {
namespace {
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }

void check(int status, const kt2n_error & error = {}) {
    if (status != KT2N_OK) throw std::runtime_error(std::string(error.message[0] ? error.message : "native engine error") + " (status " + std::to_string(status) + ")");
}

bool can_clone() {
    std::istringstream info(kt2n_build_info());
    std::string token;
    while (info >> token) if (token == KT2N_CAP_VOICE_CLONING) return true;
    return false;
}

int number(const std::string & text, const char * option, int minimum) {
    size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size() || value < minimum) throw std::runtime_error(std::string("invalid ") + option);
    return value;
}

void wav_header(std::ostream & out, uint32_t bytes) {
    auto u16 = [&](uint16_t n) { out.put(n & 255); out.put(n >> 8); };
    auto u32 = [&](uint32_t n) { for (int i = 0; i < 4; ++i) out.put((n >> (8 * i)) & 255); };
    out.write("RIFF", 4); u32(bytes + 36); out.write("WAVEfmt ", 8); u32(16);
    u16(1); u16(1); u32(24000); u32(48000); u16(2); u16(16);
    out.write("data", 4); u32(bytes);
}

struct signals {
    using handler = void (*)(int);
    handler old_int, old_term, old_pipe;
    signals() : old_int(std::signal(SIGINT, on_signal)), old_term(std::signal(SIGTERM, on_signal)), old_pipe(std::signal(SIGPIPE, SIG_IGN)) { interrupted = 0; }
    ~signals() { std::signal(SIGINT, old_int); std::signal(SIGTERM, old_term); std::signal(SIGPIPE, old_pipe); }
};

kt2n_voice_id clone_voice(kt2n_engine * engine, const std::vector<char> & audio, const std::string & transcript) {
    kt2n_voice_build_params params{};
    params.struct_size = sizeof(params); params.flags = KT2N_VB_SAVED;
    params.audio = audio.data(); params.audio_len = audio.size();
    params.reference_transcript = transcript.c_str(); params.language = "en";
    kt2n_job_id job = 0; kt2n_error error{};
    check(kt2n_voice_build(engine, &params, &job, &error), error);
    bool cancelled = false;
    for (;;) {
        if (interrupted && !cancelled) { check(kt2n_cancel(engine, job, KT2N_CANCEL_REQUEST)); cancelled = true; }
        kt2n_event events[16]{}; int32_t count = 0;
        check(kt2n_poll(engine, events, 16, 100, &count));
        bool done = false; int status = KT2N_OK; kt2n_voice_id voice = 0;
        for (int32_t i = 0; i < count; ++i) {
            const auto & event = events[i];
            if (event.id == job && (event.kind == KT2N_EVENT_VOICE_BUILT || event.kind == KT2N_EVENT_DONE)) {
                done = true; status = event.status; voice = event.voice;
                if (event.error) error = *event.error;
            }
        }
        kt2n_events_release(engine, events, count);
        if (!done) continue;
        check(kt2n_release(engine, job));
        check(status, error);
        if (cancelled || interrupted) throw std::runtime_error("reference voice build cancelled");
        if (!voice) throw std::runtime_error("reference voice build returned no voice");
        return voice;
    }
}
}

int run_native_cli(const std::map<std::string, std::string> & args) {
    auto get = [&](const std::string & key, const std::string & fallback) { auto it = args.find(key); return it == args.end() ? fallback : it->second; };
    for (const auto & option : {"--model", "--decoder-threads", "--no-reference", "--no-repack", "--greedy", "--tokens-only", "--frontend", "--logits", "--force-tokens", "--decode-tokens", "--repetition-penalty", "--repetition-window", "--run-penalty", "--run-grace", "--chunk-chars", "--chunk-min-chars", "--chunk-gap"}) {
        if (args.count(option)) throw std::runtime_error(std::string(option) + " is supported only by --backend cpu");
    }
    if (get("--decoder", "default") != "default") throw std::runtime_error("CUDA supports only the default decoder; student decoders require --backend cpu");
    const bool cloning = args.count("--reference-audio");
    if (cloning && args.count("--voice")) throw std::runtime_error("--reference-audio cannot be combined with --voice");
    if (cloning && get("--reference-transcript", "").empty()) throw std::runtime_error("--reference-transcript is required with --reference-audio");
    if (!cloning && args.count("--reference-transcript")) throw std::runtime_error("reference options require --reference-audio");
    if (cloning && !args.count("--download-only") && !can_clone()) throw std::runtime_error("reference cloning requires a library built with KT2N_BUILD_CLONING=ON and bounded custom voice capacity");
    if (!args.count("--text") && !args.count("--download-only")) throw std::runtime_error("--text is required");
    const int device = number(get("--device", "0"), "--device", 0);
    const int streams = number(get("--max-streams", "1"), "--max-streams", 1);
    const int repeats = number(get("--repeat", "1"), "--repeat", 1);
    if (repeats > 100) throw std::runtime_error("--repeat must be 1..100");
    const auto format = get("--format", "wav");
    if (format != "wav" && format != "pcm") throw std::runtime_error("--format must be wav or pcm");
    const auto stream_option = get("--stream", "true");
    if (stream_option != "true" && stream_option != "false") throw std::runtime_error("--stream must be true or false");
    const bool streaming = stream_option == "true";
    const auto output_path = get("--output", format == "wav" ? "output.wav" : "output.pcm");
    if (output_path == "-" && repeats != 1) throw std::runtime_error("--repeat cannot be combined with --output -");
    auto assets = resolve_assets(args);
    if (args.count("--download-only")) { std::cerr << "Assets ready\n"; return 0; }
    std::vector<char> reference;
    if (cloning) {
        std::ifstream file(args.at("--reference-audio"), std::ios::binary);
        if (!file) throw std::runtime_error("cannot read reference audio: " + args.at("--reference-audio"));
        constexpr size_t limit = 10 * 1024 * 1024;
        reference.resize(limit + 1);
        file.read(reference.data(), static_cast<std::streamsize>(reference.size()));
        reference.resize(static_cast<size_t>(file.gcount()));
        if (file.bad() || reference.empty()) throw std::runtime_error("reference audio is empty or unreadable");
        if (reference.size() > limit) throw std::runtime_error("reference audio must be at most 10 MiB");
    }
    const auto text_data = get("--data", assets.text_data);
    std::string levers;
    if (!assets.levers.empty()) { std::ifstream file(assets.levers); levers.assign(std::istreambuf_iterator<char>(file), {}); }
    kt2n_engine_config config{};
    config.struct_size = sizeof(config); config.abi_version = KT2N_ABI_VERSION;
    config.flags = cloning ? 0 : KT2N_CFG_INFERENCE_ONLY;
    if (cloning) {
        config.voice_build_dir = assets.voice_build_dir.c_str();
        config.inline_voice_lru = 1; config.clone_slots = 1;
        config.max_custom_voices = 1;
    }
    config.device = device; config.max_streams = streams; config.app = "us";
    config.lm_file = assets.model.c_str(); config.s3gen_file = assets.decoder.c_str();
    config.voices_file = assets.voices.c_str(); config.tokenizer_json = assets.tokenizer.c_str();
    config.text_data_dir = text_data.c_str(); config.levers_json = levers.empty() ? nullptr : levers.c_str();
    if (args.count("--threads")) config.threads_text = number(args.at("--threads"), "--threads", 1);
    kt2n_engine * raw = nullptr;
    kt2n_error error{};
    check(kt2n_engine_create(&config, &raw, &error), error);
    std::unique_ptr<kt2n_engine, decltype(&kt2n_engine_destroy)> engine(raw, kt2n_engine_destroy);
    const auto stats = json::parse(detail::read_native_runtime_stats(engine.get()));
    if (stats.value("lm", "") != "l3" || stats.value("render", "") != "l7" || stats.value("normalizer", "") != "native" || stats.value("tokenizer", "") != "native") throw std::runtime_error("CUDA requires the complete native engine, tokenizer and normalizer");
    kt2n_voice_id voice = 0;
    signals handlers;
    auto voice_name = cloning ? "reference" : get("--voice", assets.config.value("default_voice", "Bruno"));
    if (cloning) voice = clone_voice(engine.get(), reference, args.at("--reference-transcript"));
    else check(kt2n_voice_lookup(engine.get(), voice_name.c_str(), &voice, &error), error);
    for (int repeat = 0; repeat < repeats; ++repeat) {
        std::ofstream file;
        if (output_path != "-") { file.open(output_path, std::ios::binary); if (!file) throw std::runtime_error("cannot write " + output_path); }
        std::ostream & out = output_path == "-" ? std::cout : file;
        const bool file_wav = streaming && output_path != "-" && format == "wav";
        if (file_wav) wav_header(out, 0);
        kt2n_request_params request{};
        request.struct_size = sizeof(request); request.voice = voice; request.speed = 1.0;
        request.text = args.at("--text").c_str(); request.text_len = args.at("--text").size();
        request.format = format == "wav" && (!streaming || output_path == "-") ? KT2N_FORMAT_WAV : KT2N_FORMAT_PCM;
        request.output = streaming ? KT2N_OUTPUT_STREAM : KT2N_OUTPUT_COMPLETE;
        if (args.count("--no-normalize")) request.flags |= KT2N_REQ_NORMALIZE_OFF;
        if (args.count("--seed")) {
            const auto & value = args.at("--seed"); size_t consumed = 0;
            if (value.empty() || value[0] == '-') throw std::runtime_error("invalid --seed");
            request.seed = std::stoull(value, &consumed);
            if (consumed != value.size()) throw std::runtime_error("invalid --seed");
            request.flags |= KT2N_REQ_SEED;
        }
        if (args.count("--preset")) {
            const auto & preset = args.at("--preset");
            if (preset != "stable" && preset != "expressive") throw std::runtime_error("--preset must be stable or expressive");
            request.sampler.present |= KT2N_SAMPLER_HAS_MODE;
            request.sampler.mode = preset == "stable" ? KT2N_SAMPLER_STABLE : KT2N_SAMPLER_EXPRESSIVE;
        }
        auto real = [&](const char * option, uint32_t bit, double & value) { if (args.count(option)) { size_t n = 0; value = std::stod(args.at(option), &n); if (n != args.at(option).size()) throw std::runtime_error(std::string("invalid ") + option); request.sampler.present |= bit; } };
        real("--temperature", KT2N_SAMPLER_HAS_TEMPERATURE, request.sampler.temperature);
        real("--top-p", KT2N_SAMPLER_HAS_TOP_P, request.sampler.top_p);
        real("--min-p", KT2N_SAMPLER_HAS_MIN_P, request.sampler.min_p);
        if (args.count("--top-k")) { request.sampler.top_k = number(args.at("--top-k"), "--top-k", 0); request.sampler.present |= KT2N_SAMPLER_HAS_TOP_K; }
        if (args.count("--max-tokens")) { request.sampler.max_new_tokens = number(args.at("--max-tokens"), "--max-tokens", 1); request.sampler.present |= KT2N_SAMPLER_HAS_MAX_NEW_TOKENS; }
        const auto start = std::chrono::steady_clock::now();
        double first_audio_ms = -1; uint64_t bytes = 0; bool done = false, cancelled = false;
        kt2n_request_id id = 0;
        check(kt2n_submit(engine.get(), &request, &id, &error), error);
        while (!done) {
            if (interrupted && !cancelled) { check(kt2n_cancel(engine.get(), id, KT2N_CANCEL_REQUEST)); cancelled = true; }
            kt2n_event events[16]{}; int32_t count = 0;
            check(kt2n_poll(engine.get(), events, 16, 100, &count));
            int terminal_status = KT2N_OK; kt2n_error terminal_error{};
            for (int32_t i = 0; i < count; ++i) {
                const auto & event = events[i];
                if (event.kind == KT2N_EVENT_AUDIO && !cancelled) {
                    if (first_audio_ms < 0 && event.data_len) first_audio_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                    out.write(reinterpret_cast<const char *>(event.data), event.data_len); out.flush(); bytes += event.data_len;
                }
                if (event.kind == KT2N_EVENT_DONE) { done = true; terminal_status = event.status; if (event.error) terminal_error = *event.error; }
            }
            kt2n_events_release(engine.get(), events, count);
            if (done) check(kt2n_release(engine.get(), id));
            if (!out) { if (!done) kt2n_cancel(engine.get(), id, KT2N_CANCEL_REQUEST); throw std::runtime_error("audio output failed"); }
            check(terminal_status, terminal_error);
        }
        if (file_wav) {
            if (bytes > std::numeric_limits<uint32_t>::max() - 36) throw std::runtime_error("audio exceeds WAV size limit");
            file.seekp(0); wav_header(file, static_cast<uint32_t>(bytes)); file.flush();
            if (!file) throw std::runtime_error("cannot finalize WAV header");
        }
        json report = {{"backend", "cuda"}, {"device", device}, {"voice", voice_name}, {"sample_rate", 24000}, {"format", format}, {"stream", streaming}, {"bytes", bytes}, {"first_audio_ms", first_audio_ms}, {"elapsed_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()}, {"iteration", repeat + 1}};
        if (args.count("--report")) { std::ofstream report_file(args.at("--report")); report_file << report.dump(2) << '\n'; if (!report_file) throw std::runtime_error("cannot write report"); }
        std::cerr << report.dump() << '\n';
    }
    return 0;
}
}
