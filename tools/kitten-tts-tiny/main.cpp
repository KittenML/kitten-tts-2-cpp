// kitten-tts-tiny: command line for Kitten TTS 2 mini / nano / pico GGUF models.
//
//   kitten-tts-tiny -m kitten_tts_2_nano_en-q8_0.gguf --phonemes "həlˈoʊ wˈɜːld." -o out.wav
//   kitten-tts-tiny -m model.gguf --info
//   kitten-tts-tiny -m model.gguf --tokens 0,50,83,...,10,0 --style style.f32 --noise none --raw out.f32
//
// Text to phonemes is done by the Python front end (kitten_tts_cpp.KittenTTS2Tiny); this tool takes phonemes.
#include "kitten_tts_tiny.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static void usage() {
    fprintf(stderr,
        "usage: kitten-tts-tiny -m MODEL.gguf [options]\n"
        "  --device D         auto | cpu | cuda | metal | vulkan | ggml device name (default auto)\n"
        "  -t, --threads N    CPU threads (default: the physical cores this process may use, at most 8)\n"
        "  --no-dequantize    compute in the file's weight type (f16 / q8_0) instead of converting to f32 at load\n"
        "  --allow-tf32       GPUs: allow TF32 (CUDA) / fp16 (Vulkan, Metal) matrix products: faster on some GPUs, lossy\n"
        "  --info             print model information (JSON) and exit\n"
        "  --phonemes P       phonemes to speak (one chunk, as kittenml's front end produces them)\n"
        "  --phonemes-file F  read the phonemes from a UTF-8 text file (on Windows, the command line may not pass\n"
        "                     IPA characters through to this program)\n"
        "  --lang L           language code (default en)\n"
        "  --voice V          voice key or name (default: the language's first voice)\n"
        "  --speed S          speaking rate, 0.25 to 4 (default 1.0, higher is faster)\n"
        "  --seed N           noise seed (default 0)\n"
        "  -o FILE.wav        write 32-bit float WAV\n"
        "  raw model call (parity tests):\n"
        "  --tokens a,b,c     token ids instead of phonemes\n"
        "  --style FILE.f32   256 float32 style values (default: --voice row for the token count)\n"
        "  --noise MODE       random | none | FILE.f32 (n_samples x 9 standard-normal values)\n"
        "  --durations FILE   int32 frames per token (overrides the predicted durations)\n"
        "  --raw FILE.f32     write the raw waveform (no trimming or fades)\n"
        "  --dur-out FILE     write int32 durations\n"
        "  --repeat N         run N times and print timings of each run\n");
}

static std::vector<char> read_file(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(1); }
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static void write_wav(const std::string & path, const float * x, size_t n, int sr) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(1); }
    auto u32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto u16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
    const uint32_t data = (uint32_t) (n * 4);
    fwrite("RIFF", 1, 4, f); u32(36 + data); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); u32(16); u16(3); u16(1); u32(sr); u32(sr * 4); u16(4); u16(32);
    fwrite("data", 1, 4, f); u32(data);
    fwrite(x, 4, n, f);
    fclose(f);
}

int main(int argc, char ** argv) {
    std::string model, device = "auto", phonemes, lang = "en", voice, out, tokens_s, style_f, noise = "random", dur_f, raw, dur_out;
    int threads = 0, repeat = 1;
    float speed = 1.0f;
    unsigned long long seed = 0;
    bool info = false, no_dequant = false, allow_tf32 = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { usage(); exit(1); } return argv[++i]; };
        if (a == "-m" || a == "--model") model = next();
        else if (a == "--device") device = next();
        else if (a == "-t" || a == "--threads") threads = atoi(next().c_str());
        else if (a == "--info") info = true;
        else if (a == "--no-dequantize") no_dequant = true;
        else if (a == "--allow-tf32") allow_tf32 = true;
        else if (a == "--phonemes") phonemes = next();
        else if (a == "--phonemes-file") {
            const std::vector<char> b = read_file(next());
            phonemes.assign(b.begin(), b.end());
            if (phonemes.rfind("\xEF\xBB\xBF", 0) == 0) phonemes.erase(0, 3);   // UTF-8 byte order mark (Notepad)
            while (!phonemes.empty() && (phonemes.back() == '\n' || phonemes.back() == '\r')) phonemes.pop_back();
        }
        else if (a == "--lang") lang = next();
        else if (a == "--voice") voice = next();
        else if (a == "--speed") speed = (float) atof(next().c_str());
        else if (a == "--seed") seed = strtoull(next().c_str(), nullptr, 10);
        else if (a == "-o") out = next();
        else if (a == "--tokens") tokens_s = next();
        else if (a == "--style") style_f = next();
        else if (a == "--noise") noise = next();
        else if (a == "--durations") dur_f = next();
        else if (a == "--raw") raw = next();
        else if (a == "--dur-out") dur_out = next();
        else if (a == "--repeat") repeat = atoi(next().c_str());
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); usage(); return 1; }
    }
    if (model.empty()) { usage(); return 1; }
    kt_tiny_params p = kt_tiny_default_params();
    p.device = device.c_str();
    p.n_threads = threads;
    p.dequantize = no_dequant ? 0 : 1;
    p.allow_tf32 = allow_tf32 ? 1 : 0;
    auto t0 = std::chrono::steady_clock::now();
    kt_tiny_model * m = kt_tiny_load(model.c_str(), &p);
    if (!m) { fprintf(stderr, "error: %s\n", kt_tiny_last_error()); return 1; }
    double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (info) { printf("%s\n", kt_tiny_info_json(m)); kt_tiny_free(m); return 0; }
    fprintf(stderr, "loaded in %.1f ms\n", load_ms);

    for (int r = 0; r < repeat; r++) {
        float * audio = nullptr;
        size_t n = 0;
        int rc;
        if (!tokens_s.empty()) {
            std::vector<int32_t> tok;
            std::stringstream ss(tokens_s);
            std::string item;
            while (std::getline(ss, item, ',')) tok.push_back(atoi(item.c_str()));
            std::vector<float> style(256);
            if (!style_f.empty()) {
                auto b = read_file(style_f);
                if (b.size() != 256 * 4) { fprintf(stderr, "style file must hold 256 float32\n"); return 1; }
                memcpy(style.data(), b.data(), b.size());
            } else if (kt_tiny_voice_style(m, lang.c_str(), voice.c_str(), (int) tok.size(), style.data()) != 0) {
                fprintf(stderr, "error: %s\n", kt_tiny_last_error()); return 1;
            }
            kt_tiny_synth_options o = kt_tiny_default_options();
            o.speed = speed;
            o.seed = seed;
            std::vector<char> nbuf, dbuf;
            if (noise == "none") o.noise_mode = KT_TINY_NOISE_NONE;
            else if (noise != "random") {
                nbuf = read_file(noise);
                o.noise_mode = KT_TINY_NOISE_EXTERNAL;
                o.noise = (const float *) nbuf.data();
                o.noise_len = nbuf.size() / 4;
            }
            if (!dur_f.empty()) {
                dbuf = read_file(dur_f);
                if (dbuf.size() != 4 * tok.size()) {
                    fprintf(stderr, "durations file must hold one int32 per token (%zu tokens, %zu bytes)\n", tok.size(), dbuf.size());
                    kt_tiny_free(m); return 1;
                }
                o.durations = (const int32_t *) dbuf.data();
            }
            int32_t * dur = nullptr;
            rc = kt_tiny_synthesize(m, tok.data(), (int) tok.size(), style.data(), &o, &audio, &n, &dur);
            if (rc == 0 && !dur_out.empty()) {
                FILE * f = fopen(dur_out.c_str(), "wb");
                if (!f) { fprintf(stderr, "cannot write %s\n", dur_out.c_str()); kt_tiny_free_buffer(dur); kt_tiny_free_buffer(audio); kt_tiny_free(m); return 1; }
                fwrite(dur, 4, tok.size(), f);
                fclose(f);
            }
            kt_tiny_free_buffer(dur);
        } else {
            rc = kt_tiny_generate_chunk(m, phonemes.c_str(), lang.c_str(), voice.c_str(), speed, 1, seed, &audio, &n);
        }
        if (rc != 0) { fprintf(stderr, "error: %s\n", kt_tiny_last_error()); kt_tiny_free(m); return 1; }
        fprintf(stderr, "%s\n", kt_tiny_last_report(m));
        if (r == repeat - 1) {
            if (!out.empty()) write_wav(out, audio, n, 24000);
            if (!raw.empty()) {
                FILE * f = fopen(raw.c_str(), "wb");
                if (!f) { fprintf(stderr, "cannot write %s\n", raw.c_str()); kt_tiny_free_buffer(audio); kt_tiny_free(m); return 1; }
                fwrite(audio, 4, n, f);
                fclose(f);
            }
        }
        kt_tiny_free_buffer(audio);
    }
    kt_tiny_free(m);
    return 0;
}
