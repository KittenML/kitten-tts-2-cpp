// Kitten TTS 2 mini / nano / pico (KITTEN2_TINY) on ggml.
//
// The network is the StyleTTS2 / Kokoro family as shipped in kittenml 0.9.8 (kittentts2_tiny/networks):
// PL-BERT (ALBERT) + text encoder (embedding, CNN, BiLSTM) + prosody predictor (BiLSTMs, AdaLayerNorm,
// AdaIN residual blocks) + iSTFTNet decoder. Three graphs run per chunk:
//   A  tokens -> ALBERT, duration encoder, durations, text encoder           (token rate)
//   B  frame-aligned features -> shared BiLSTM, F0 and N curves               (frame rate)
//   C  decoder + generator + inverse STFT -> waveform                        (sample rate)
// Between them the host turns durations into a frame -> token map and builds the harmonic source
// (SineGen + forward STFT) from the F0 curve, both cheap.
//
// Activations are [C, T] (ne0 = channels, ne1 = time). Convolutions are matrix products over a
// sliding-window view of the activation (weights converted to k-major layout by the converter).
// On the CPU backend, AdaIN + activation and the LSTM recurrence run as fused custom operations;
// On CUDA, the LSTM recurrence and the harmonic source run as CUDA kernels of this library (kitten_tts_tiny_cuda.cu);
// other GPU backends use plain ggml ops. f16 / q8_0 weights are converted to f32 at load unless asked not to.
// KT_TINY_PROFILE=1 prints time per op. Builds with KITTEN_TTS_TINY_DEBUG_ENV=ON also read debugging variables
// (KT_TINY_DUMP=dir writes intermediate tensors, KT_TINY_LOAD_F0N=dir replaces F0 / N, path switches for A/B tests).
#include "kitten_tts_tiny.h"
#ifdef KT_TINY_CUDA
#include "kitten_tts_tiny_cuda.h"
#endif

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <fstream>
#include <sched.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

using json = nlohmann::json;

namespace {

thread_local std::string g_last_error;

// The speed range of both C entry points (and of the Python class). It bounds one chunk's audio and memory: at speed
// 0.02 a 300-phoneme chunk would make 767 s of audio and use 4 GB.
constexpr float KT_SPEED_MIN = 0.25f, KT_SPEED_MAX = 4.0f;

void check_speed(float speed) {
    if (!(speed >= KT_SPEED_MIN && speed <= KT_SPEED_MAX)) throw std::invalid_argument("speed must be from 0.25 to 4 (1.0 is normal)");
}

#if defined(__linux__)
// CPUs the cgroup CPU quota allows (docker --cpus, Kubernetes CPU limits), rounded up; 0 = no quota found.
int cgroup_cpu_quota() {
    std::vector<std::string> dirs = {"/sys/fs/cgroup"};
    std::ifstream cg("/proc/self/cgroup");
    for (std::string line; std::getline(cg, line);) {
        if (line.rfind("0::", 0) == 0 && line.size() > 3 && line.substr(3) != "/") dirs.insert(dirs.begin(), "/sys/fs/cgroup" + line.substr(3));
    }
    for (const auto & d : dirs) {   // cgroup v2: "<quota> <period>" or "max <period>"
        std::ifstream f(d + "/cpu.max");
        std::string q;
        long long period = 0;
        if (f >> q >> period && q != "max" && period > 0) {
            const long long quota = atoll(q.c_str());
            if (quota > 0) return (int) ((quota + period - 1) / period);
        }
    }
    std::ifstream fq("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), fp("/sys/fs/cgroup/cpu/cpu.cfs_period_us");   // cgroup v1
    long long quota = 0, period = 0;
    if (fq >> quota && fp >> period && quota > 0 && period > 0) return (int) ((quota + period - 1) / period);
    return 0;
}
#endif

// Default CPU threads: the physical cores this process may run on, at most 8. Counting logical CPUs instead
// oversubscribes the cores when hardware threads (SMT) share a core, when the process is limited to some CPUs
// (taskset, docker --cpuset-cpus) or by a CPU quota (docker --cpus), and ggml's threads then wait for each other at
// every graph node: on a 4-core / 8-thread Windows x64 computer, 8 threads were 2.5 to 12 times slower than 4, and under
// a 2- or 4-CPU affinity mask on Linux, 8 threads were 30 to 80 times slower. Apple: performance cores only.
int default_threads() {
    int n = 0;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        std::set<std::string> cores;   // one entry per physical core: the core's list of hardware threads
        for (int c = 0; c < CPU_SETSIZE; c++) {
            if (!CPU_ISSET(c, &set)) continue;
            std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
            std::string line;
            cores.insert(f && std::getline(f, line) && !line.empty() ? line : "cpu" + std::to_string(c));
        }
        n = (int) cores.size();
    }
    const int quota = cgroup_cpu_quota();
    if (quota > 0 && (n <= 0 || quota < n)) n = quota;
#elif defined(_WIN32)
    DWORD_PTR pmask = 0, smask = 0;   // the process's CPUs in its group (0 when it spans several groups)
    const bool have_mask = GetProcessAffinityMask(GetCurrentProcess(), &pmask, &smask) && pmask != 0;
    GROUP_AFFINITY tg = {};
    const WORD group = GetThreadGroupAffinity(GetCurrentThread(), &tg) ? tg.Group : 0;
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<char> buf(len);
    if (len > 0 && GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len)) {
        for (DWORD off = 0; off + offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor) <= len;) {
            const auto * info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data() + off);
            if (info->Size == 0) break;
            if (info->Relationship == RelationProcessorCore) {   // one entry per physical core
                for (WORD g = 0; g < info->Processor.GroupCount; g++) {
                    const GROUP_AFFINITY & ga = info->Processor.GroupMask[g];
                    if (!have_mask || (ga.Group == group && (ga.Mask & pmask) != 0)) { n++; break; }
                }
            }
            off += info->Size;
        }
    }
    if (n <= 0 && have_mask) {
        for (DWORD_PTR b = pmask; b; b &= b - 1) n++;
    }
#elif defined(__APPLE__) && defined(__MACH__)
    // every performance level except the efficiency cores: on an M4 (4 performance + 6 efficiency) that is perflevel0
    // alone, as before; on an M6 (2 "Super" + 4 "Performance" + 6 "Efficiency") it is 6 cores, where 2 threads (perflevel0
    // only) made the CPU 2.1 times slower than 6
    int32_t v = 0;
    size_t len = sizeof(v);
    for (int lvl = 0; lvl < 8; lvl++) {
        char name[64] = {0}, key[64];
        size_t nlen = sizeof(name) - 1;
        snprintf(key, sizeof(key), "hw.perflevel%d.name", lvl);
        if (sysctlbyname(key, name, &nlen, nullptr, 0) != 0) break;
        snprintf(key, sizeof(key), "hw.perflevel%d.physicalcpu", lvl);
        len = sizeof(v);
        if (sysctlbyname(key, &v, &len, nullptr, 0) != 0 || v <= 0) break;
        if (lvl > 0 && strcasestr(name, "efficiency") != nullptr) break;
        n += v;
    }
    if (n <= 0) {
        len = sizeof(v);
        if (sysctlbyname("hw.physicalcpu", &v, &len, nullptr, 0) == 0 && v > 0) n = v;
    }
#endif
    if (n <= 0) n = (int) std::thread::hardware_concurrency();
    return std::max(1, std::min(8, n));
}

// KT_TINY_PROFILE is read once, at the first model load: a graph run never calls getenv, which is not thread-safe
// against the setenv of a Vulkan model loading on another thread.
bool profile_enabled() {
    static const bool on = getenv("KT_TINY_PROFILE") != nullptr;
    return on;
}

// Debugging / A-B switches from the environment: only in builds made with KITTEN_TTS_TINY_DEBUG_ENV=ON, so an
// inherited variable cannot change a released library's output or make it write files.
const char * debug_env(const char * name) {
#ifdef KT_TINY_DEBUG_ENV
    return getenv(name);
#else
    (void) name;
    return nullptr;
#endif
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string lower(std::string s) {
    for (auto & c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

// UTF-8 -> code points
std::vector<uint32_t> utf8_decode(const std::string & s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char) s[i];
        uint32_t cp; int n;
        if (c < 0x80) { cp = c; n = 1; }
        else if ((c >> 5) == 0x6) { cp = c & 0x1f; n = 2; }
        else if ((c >> 4) == 0xe) { cp = c & 0x0f; n = 3; }
        else if ((c >> 3) == 0x1e) { cp = c & 0x07; n = 4; }
        else { i++; continue; }
        if (i + n > s.size()) break;
        for (int k = 1; k < n; k++) cp = (cp << 6) | ((unsigned char) s[i + k] & 0x3f);
        out.push_back(cp);
        i += n;
    }
    return out;
}

} // namespace

struct kt_tiny_model {
    // hyper-parameters
    int hid = 0, n_layer = 0, style_dim = 128, te_kernel = 5;
    std::string dec_type;
    bool mag_softplus = false;
    std::vector<int> rb_k;
    std::vector<std::vector<int>> rb_d;
    std::vector<int> up_r, up_k;
    int up_ch = 0, n_fft = 20, hop = 5, sample_rate = 24000;
    int a_hidden = 768, a_heads = 12, a_layers = 12;
    float a_eps = 1e-12f;

    std::string model_name, weight_type, device_name;
    json config, arch;
    std::map<uint32_t, int> vocab;
    std::map<std::string, std::vector<float>> voices;   // "lang/key" -> rows * 256
    std::map<std::string, std::vector<std::string>> lang_voices;
    std::vector<std::string> languages;

    // weights
    ggml_context * ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor *> W;
    std::vector<float> src_lin_w; float src_lin_b = 0.0f;   // m_source.l_linear on the host

    ggml_backend_t backend = nullptr, cpu = nullptr;
    std::vector<ggml_backend_t> backends;
    ggml_backend_sched_t sched = nullptr;
    size_t sched_size = 0;
    // Metal: the text and prosody graphs (A, B: ALBERT, LSTMs, small convolutions) run on the CPU backend with the CPU
    // fast path, and only the decoder graph (C) on the GPU. Their weights live in a CPU buffer, the decoder's on the GPU.
    bool front_cpu = false;
    ggml_context * ctx_w_cpu = nullptr;
    ggml_backend_buffer_t buf_w_cpu = nullptr;
    ggml_backend_sched_t sched_cpu = nullptr;
    size_t sched_cpu_size = 0;
    int n_threads = 4;
    bool window_cont = false;   // materialise convolution windows (GPU backends)
    bool tf32_disabled = false;
    bool cpu_fused = false;     // CPU backend: fused custom ops for AdaIN + activation
    bool dequantize = true;     // f16 / q8_0 weights converted to f32 at load
    bool strict_f32 = true;     // f32 matrix products without TF32 (CUDA)
    bool gpu_lstm = false;      // CUDA: LSTM recurrence as one kernel per BiLSTM
    bool gpu_source = false;    // CUDA: harmonic source + STFT on the GPU
    bool gpu_adain = false;     // CUDA: AdaIN + activation + convolution (implicit GEMM) as library kernels
    int igemm_min_t = 0;        // CUDA: convolutions with at least this many output steps use the implicit-GEMM kernel (< 0: never)
    int64_t n_vocab = 0;        // rows of the token embeddings (token ids must be below this)
#ifdef KT_TINY_CUDA
    kt_cuda_ctx * cuda = nullptr;
#endif

    std::string info, report;
    json rep;

    ggml_tensor * w(const std::string & name) const {
        auto it = W.find(name);
        if (it == W.end()) throw std::runtime_error("missing tensor " + name);
        return it->second;
    }
    ggml_tensor * wo(const std::string & name) const {
        auto it = W.find(name);
        return it == W.end() ? nullptr : it->second;
    }
};

namespace {

// ---------------------------------------------------------------------------------------------------
// graph helpers. x is [C, T] unless stated.

struct adain_act_params;

// An operation ggml has no GPU kernel for: the graph holds a placeholder node of the right shape, the scheduler stops
// after it (eval callback), and the library's own CUDA kernel fills the placeholder's data before the graph goes on.
struct gpu_op {
    int kind = 0;                 // 1 = BiLSTM recurrence, 2 = harmonic source
    std::string key;              // LSTM weights
    ggml_tensor * gx = nullptr;   // LSTM input projections [4H, T, 2]
    int H = 0, T = 0;
    ggml_tensor * f0 = nullptr, * p300 = nullptr;   // harmonic source inputs
    int L0 = 0;
    int64_t frames = 0;
};

// Parameters of one fused AdaIN + activation + convolution window node (a GGML_OP_CUSTOM node with a NULL function that
// the CUDA backend hands to adain_cuda_handler). Sources: x, gamma / beta, optional norm weight / bias, optional snake alpha / 1/alpha.
// every NULL-function custom node's user data starts with its kind, so one CUDA handler serves them all
enum { KT_NODE_ADAIN = 1, KT_NODE_LSTM = 2 };

struct adain_gpu_params {
    int kind = KT_NODE_ADAIN;
#ifdef KT_TINY_CUDA
    kt_cuda_ctx * cuda = nullptr;
#endif
    int act = 0, Cp = 0, K = 0, d = 1, pl = 0, Tout = 0;
    float slope = 0.2f;
    int Cout = 0;               // > 0: the node is the whole convolution (implicit GEMM, bias included), else only the windows
};

struct graph_ctx {
    kt_tiny_model * m;
    ggml_context * ctx;
    ggml_cgraph * gf;
    std::map<ggml_tensor *, gpu_op> gpu_ops;     // placeholder node -> operation
    const kt_tiny_synth_options * opts = nullptr;
    std::vector<std::unique_ptr<adain_act_params>> op_params;   // user data of the CPU custom ops, alive per graph
    std::vector<std::pair<ggml_tensor *, std::vector<char>>> inputs;
    std::vector<std::unique_ptr<adain_gpu_params>> adain_params;
    std::vector<std::unique_ptr<struct lstm_gpu_params>> lstm_params;
    int adain_max_C = 0; long long adain_max_pad = 0;   // scratch the fused kernels need
    double conv_flops = 0, win_bytes = 0;         // fused convolutions: 2 Cp K Tout Cout, bytes of their windows

    ggml_tensor * input_f32(const char * name, std::vector<int64_t> ne, const float * data) {
        ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, (int) ne.size(), ne.data());
        ggml_set_name(t, name);
        ggml_set_input(t);
        std::vector<char> buf(ggml_nbytes(t));
        memcpy(buf.data(), data, buf.size());
        inputs.emplace_back(t, std::move(buf));
        return t;
    }
    ggml_tensor * input_i32(const char * name, const std::vector<int32_t> & v) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) v.size());
        ggml_set_name(t, name);
        ggml_set_input(t);
        std::vector<char> buf(ggml_nbytes(t));
        memcpy(buf.data(), v.data(), buf.size());
        inputs.emplace_back(t, std::move(buf));
        return t;
    }
};

// matrix product with f32 accumulation requested (cuBLAS would otherwise accumulate f16 weights in f16)
// (and on CUDA no TF32 unless the model was loaded with allow_tf32)
thread_local ggml_prec t_mm_prec = GGML_PREC_F32_STRICT;
ggml_tensor * mm(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_mul_mat_set_prec(y, t_mm_prec);
    return y;
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    ggml_tensor * y = mm(ctx, w, x);
    if (b) y = ggml_add(ctx, y, b);
    return y;
}

// Copy the sliding window to a contiguous matrix before the matrix product (an explicit im2col).
// The CPU backend reads the overlapping rows directly; GPU BLAS libraries need a row stride >= row length.
thread_local bool t_window_cont = false;
thread_local bool t_cpu_fused = false;     // CPU backend: fused AdaIN + activation custom ops
thread_local bool t_gpu_lstm = false;      // CUDA: LSTM recurrence as a library kernel
thread_local bool t_gpu_adain = false;     // CUDA: fused AdaIN + activation + convolution
thread_local bool t_gpu_nodes = false;     // CUDA: library kernels as graph nodes (custom-node hook registered)

// a BiLSTM recurrence as a graph node (NULL-function custom node run by the CUDA backend's hook)
struct lstm_gpu_params {
    int kind = KT_NODE_LSTM;
#ifdef KT_TINY_CUDA
    kt_cuda_ctx * cuda = nullptr;
#endif
    std::string key;
    int H = 0, T = 0;
};

// A view whose rows overlap (row stride nb1 smaller than a row): the sliding window of a convolution.
// ggml_view_* checks the size as if the view were contiguous, so the view is made small and then widened;
// its true extent (ggml_nbytes) stays inside x.
ggml_tensor * window_view(ggml_context * ctx, ggml_tensor * x, int64_t ne0, int64_t ne1, size_t nb1, int64_t ne2, size_t nb2) {
    ggml_tensor * v = ggml_view_1d(ctx, x, 1, 0);
    v->ne[0] = ne0; v->ne[1] = ne1; v->ne[2] = ne2; v->ne[3] = 1;
    v->nb[1] = nb1; v->nb[2] = nb2; v->nb[3] = nb2 * ne2;
    GGML_ASSERT(ggml_nbytes(v) <= ggml_nbytes(x));
    return v;
}

// Zero padding in time ([C, T] -> [C, pl + T + pr]) as a 1-D pad of the flat buffer: the same bytes, and the CUDA
// pad kernel puts ne1 in the grid's y dimension (limit 65535), which long generator inputs exceed.
ggml_tensor * pad_time(ggml_context * ctx, ggml_tensor * x, int pl, int pr) {
    if (!pl && !pr) return x;
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    const int64_t C = x->ne[0], T = x->ne[1];
    ggml_tensor * f = ggml_reshape_1d(ctx, x, C * T);
    f = ggml_pad_ext(ctx, f, (int) (C * pl), (int) (C * pr), 0, 0, 0, 0, 0, 0);
    return ggml_reshape_2d(ctx, f, C, pl + T + pr);
}

// Zero channels appended ([C, T] -> [Cp, T]) by concatenating a zero tensor (contiguous concat has no grid limit).
ggml_tensor * pad_channels(ggml_context * ctx, ggml_tensor * x, int64_t Cp) {
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    const int64_t C = x->ne[0], T = x->ne[1];
    if (C >= Cp) return x;
    ggml_tensor * z = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Cp - C, T), 0.0f);
    return ggml_concat(ctx, x, z, 0);
}

// stride-1 Conv1d. w is ggml [Cin_p, K, Cout] (k-major), dilation d, zero padding pl / pr in time.
ggml_tensor * conv1d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int d, int pl, int pr) {
    const int64_t Cp = w->ne[0], K = w->ne[1], Cout = w->ne[2];
    if (d > 1 && x->ne[2] == d && pl == 0 && pr == 0) {
        // already in the phase-major layout [Cp, V, d] (written so by adain_act_cpu): Tout = d*V - d*(K-1)
        GGML_ASSERT(x->ne[0] == Cp);
        const int64_t V = x->ne[1], U = V - K + 1;
        ggml_tensor * w2 = ggml_reshape_2d(ctx, w, Cp * K, Cout);
        ggml_tensor * xv = window_view(ctx, x, Cp * K, U, x->nb[1], d, x->nb[2]);
        ggml_tensor * y = mm(ctx, w2, xv);                                       // [Cout, U, d]
        y = ggml_cont(ctx, ggml_permute(ctx, y, 0, 2, 1, 3));                   // [Cout, d, U]
        y = ggml_reshape_2d(ctx, y, Cout, d * U);
        if (b) y = ggml_add(ctx, y, b);
        return y;                                                               // caller crops to its length
    }
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    if (x->ne[0] < Cp) x = pad_channels(ctx, x, Cp);
    GGML_ASSERT(x->ne[0] == Cp);
    x = pad_time(ctx, x, pl, pr);
    const int64_t Tp = x->ne[1];
    const int64_t Tout = Tp - (int64_t) d * (K - 1);
    GGML_ASSERT(Tout > 0);
    ggml_tensor * w2 = ggml_reshape_2d(ctx, w, Cp * K, Cout);
    ggml_tensor * y;
    if (K == 1) {
        y = mm(ctx, w2, x);
    } else if (d == 1) {
        // column t of the view = x[:, t .. t+K-1] flattened: overlapping rows, no copy on the CPU
        ggml_tensor * xv = window_view(ctx, x, Cp * K, Tout, x->nb[1], 1, x->nb[1] * Tout);
        if (t_window_cont) xv = ggml_cont(ctx, xv);
        y = mm(ctx, w2, xv);
    } else {
        // dilation d: split time into d phases (t = d*v + r); each phase is a dilation-1 convolution
        const int64_t V = (Tp + d - 1) / d;
        if (V * d > Tp) x = pad_time(ctx, x, 0, (int) (V * d - Tp));
        ggml_tensor * x3 = ggml_reshape_3d(ctx, x, Cp, d, V);
        x3 = ggml_cont(ctx, ggml_permute(ctx, x3, 0, 2, 1, 3));                 // [Cp, V, d]
        const int64_t U = V - K + 1;
        ggml_tensor * xv = window_view(ctx, x3, Cp * K, U, x3->nb[1], d, x3->nb[2]);
        if (t_window_cont) xv = ggml_cont(ctx, xv);
        y = mm(ctx, w2, xv);                                           // [Cout, U, d]
        y = ggml_cont(ctx, ggml_permute(ctx, y, 0, 2, 1, 3));                   // [Cout, d, U]
        y = ggml_reshape_2d(ctx, y, Cout, d * U);
        y = ggml_view_2d(ctx, y, Cout, Tout, y->nb[1], 0);
    }
    if (b) y = ggml_add(ctx, y, b);
    return y;
}

// strided Conv1d (stride s, padding p, original kernel K) via space-to-depth: s time steps -> channels.
ggml_tensor * conv1d_strided(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int s, int p, int K) {
    const int64_t C = x->ne[0], T = x->ne[1];
    const int64_t Tout = (T + 2 * p - K) / s + 1;
    const int64_t Kp = w->ne[1];
    const int64_t total = (int64_t) s * (Tout + Kp - 1);
    const int64_t rp = total - T - p;
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    if (rp >= 0) {
        x = pad_time(ctx, x, p, (int) rp);
    } else {
        x = pad_time(ctx, x, p, 0);
        x = ggml_cont(ctx, ggml_view_2d(ctx, x, C, total, x->nb[1], 0));
    }
    x = ggml_reshape_2d(ctx, x, C * s, total / s);
    return conv1d(ctx, x, w, b, 1, 0, 0);
}

// ConvTranspose1d: w ggml [Cin, K, Cout] -> mul_mat to [Cout*K, T] columns, then col2im_1d overlap-add.
ggml_tensor * conv_transpose1d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int s, int p) {
    const int64_t Cin = w->ne[0], K = w->ne[1], Cout = w->ne[2];
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    ggml_tensor * cols = mm(ctx, ggml_reshape_2d(ctx, w, Cin, K * Cout), x);   // [K*Cout, T]
    ggml_tensor * y = ggml_col2im_1d(ctx, cols, s, (int) Cout, p);                      // [Tout, Cout]
    y = ggml_cont(ctx, ggml_transpose(ctx, y));                                          // [Cout, Tout]
    if (b) y = ggml_add(ctx, y, b);
    return y;
}

ggml_tensor * leaky(ggml_context * ctx, ggml_tensor * x, float slope) {
    return ggml_leaky_relu(ctx, x, slope, false);
}

ggml_tensor * vec_as_row(ggml_context * ctx, ggml_tensor * v, int64_t off, int64_t n) {
    // n values of a contiguous vector as a [1, n] tensor (broadcast over ne0)
    return ggml_reshape_2d(ctx, ggml_view_1d(ctx, v, n, off * ggml_element_size(v)), 1, n);
}

// AdaIN1d: InstanceNorm1d over time (optionally affine) then (1 + gamma) * x + beta, gamma / beta = fc(s).
ggml_tensor * adain(graph_ctx & g, ggml_tensor * x, ggml_tensor * s, const std::string & pfx) {
    ggml_context * ctx = g.ctx;
    const int64_t C = x->ne[0];
    ggml_tensor * h = linear(ctx, s, g.m->w(pfx + ".fc.weight"), g.m->w(pfx + ".fc.bias"));   // [2C, 1]
    ggml_tensor * gam = ggml_scale_bias(ctx, vec_as_row(ctx, h, 0, C), 1.0f, 1.0f);             // 1 + gamma
    ggml_tensor * bet = vec_as_row(ctx, h, C, C);
    ggml_tensor * xt = ggml_cont(ctx, ggml_transpose(ctx, x));                                  // [T, C]
    xt = ggml_norm(ctx, xt, 1e-5f);
    ggml_tensor * nw = g.m->wo(pfx + ".norm.weight");
    if (nw) {
        xt = ggml_mul(ctx, xt, ggml_reshape_2d(ctx, nw, 1, C));
        xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, g.m->w(pfx + ".norm.bias"), 1, C));
    }
    xt = ggml_add(ctx, ggml_mul(ctx, xt, gam), bet);
    return ggml_cont(ctx, ggml_transpose(ctx, xt));
}

// ---- CPU fast path: AdaIN + activation (+ the next convolution's zero padding) as two custom ops ----
// The generic graph spends most decoder time on transposes for the instance norm and on separate passes for
// the affine, the snake and the padding. On the CPU backend two custom ops do the same in two passes:
//   stats: each thread sums its block of time steps per channel (sum, sum of squares in double)
//   apply: each thread reduces the partial sums, then y = act((x - mean) * rstd * (1 + gamma) [*w + b] + beta)
//          and writes it into a buffer that already holds the zero padding the next convolution needs.
// Used only when the graph runs on the CPU backend (custom ops have no GPU kernels).
constexpr int KT_STAT_SLOTS = 128;

struct adain_act_params {
    int act = 0;          // 0 = none, 1 = leaky relu, 2 = snake
    float slope = 0.2f;
    float eps = 1e-5f;
    int pl = 0, pr = 0;   // zero rows (time steps) before / after
    int64_t C = 0;        // real channels (dst may have more: zero channel padding)
    int phase = 1;        // > 1: write time step tp at (tp % phase, tp / phase), the layout of a dilated convolution
};

void kt_stats_op(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * x = dst->src[0];
    const int64_t C = x->ne[0], T = x->ne[1];
    double * part = (double *) dst->data;            // [KT_STAT_SLOTS][2][C]
    for (int sl = ith; sl < KT_STAT_SLOTS; sl += nth) {
        if (sl >= nth) memset(part + (size_t) sl * 2 * C, 0, 2 * C * sizeof(double));
    }
    const int64_t per = (T + nth - 1) / nth, t0 = per * ith, t1 = std::min(T, t0 + per);
    double * sum = part + (size_t) ith * 2 * C, * sq = sum + C;
    for (int64_t c = 0; c < C; c++) { sum[c] = 0; sq[c] = 0; }
    for (int64_t t = t0; t < t1; t++) {
        const float * row = (const float *) ((const char *) x->data + t * x->nb[1]);
        for (int64_t c = 0; c < C; c++) { const double v = row[c]; sum[c] += v; sq[c] += v * v; }
    }
}

void kt_adain_apply_op(ggml_tensor * dst, int ith, int nth, void * ud) {
    const auto * p = (const adain_act_params *) ud;
    const ggml_tensor * x = dst->src[0], * part_t = dst->src[1], * h = dst->src[2];
    const ggml_tensor * nw = dst->src[3], * nb = dst->src[4], * al = dst->src[5], * ial = dst->src[6];
    const int64_t C = p->C, T = x->ne[1], Cd = dst->ne[0];
    const double * part = (const double *) part_t->data;
    std::vector<float> mean(C), a(C), b(C);
    const float * hv = (const float *) h->data;
    for (int64_t c = 0; c < C; c++) {
        double su = 0, sq = 0;
        for (int sl = 0; sl < KT_STAT_SLOTS; sl++) { su += part[(size_t) sl * 2 * C + c]; sq += part[(size_t) sl * 2 * C + C + c]; }
        const double mu = su / T, var = std::max(0.0, sq / T - mu * mu);
        const float rstd = (float) (1.0 / std::sqrt(var + p->eps));
        float sc = rstd * (1.0f + hv[c]);
        float sh = hv[C + c];
        if (nw) {
            const float w = ((const float *) nw->data)[c], bb = ((const float *) nb->data)[c];
            sh = (1.0f + hv[c]) * bb + sh;
            sc = sc * w;
        }
        mean[c] = (float) mu; a[c] = sc; b[c] = sh;
    }
    const float * alpha = al ? (const float *) al->data : nullptr, * ialpha = ial ? (const float *) ial->data : nullptr;
    const int64_t To = dst->ne[1] * dst->ne[2];          // rows of dst (padded time, possibly phase-major)
    const int64_t V = dst->ne[1];
    const int64_t per = (To + nth - 1) / nth, r0 = per * ith, r1 = std::min(To, r0 + per);
    for (int64_t r = r0; r < r1; r++) {
        float * out = (float *) ((char *) dst->data + r * dst->nb[1]);
        const int64_t tp = p->phase > 1 ? (r % V) * p->phase + r / V : r;
        const int64_t t = tp - p->pl;
        if (t < 0 || t >= T) { memset(out, 0, Cd * sizeof(float)); continue; }
        const float * row = (const float *) ((const char *) x->data + t * x->nb[1]);
        for (int64_t c = 0; c < C; c++) {
            float y = (row[c] - mean[c]) * a[c] + b[c];
            if (p->act == 1) y = y > 0.0f ? y : y * p->slope;
            else if (p->act == 2) { const float sn = std::sin(alpha[c] * y); y = y + ialpha[c] * (sn * sn); }
            out[c] = y;
        }
        for (int64_t c = C; c < Cd; c++) out[c] = 0.0f;
    }
}

// AdaIN (+ activation) on the CPU: x [C, T] -> [Cpad, pl + T + pr]
ggml_tensor * adain_act_cpu(graph_ctx & g, ggml_tensor * x, ggml_tensor * s, const std::string & pfx, int act,
                            const std::string & alpha_name, int64_t c_pad, int pl, int pr, int phase = 1) {
    ggml_context * ctx = g.ctx;
    kt_tiny_model * m = g.m;
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    const int64_t C = x->ne[0], T = x->ne[1];
    ggml_tensor * h = linear(ctx, s, m->w(pfx + ".fc.weight"), m->w(pfx + ".fc.bias"));      // [2C, 1]
    ggml_tensor * st_args[] = {x};
    ggml_tensor * stats = ggml_custom_4d(ctx, GGML_TYPE_F32, 4 * C, KT_STAT_SLOTS, 1, 1, st_args, 1, kt_stats_op,
                                         std::min(m->n_threads, KT_STAT_SLOTS), nullptr);
    auto prm = std::make_unique<adain_act_params>();
    prm->act = act; prm->pl = pl; prm->pr = pr; prm->C = C; prm->phase = phase;
    ggml_tensor * nw = m->wo(pfx + ".norm.weight");
    // sources by position: x, partial sums, fc output, optional affine norm weight / bias, optional snake alpha / 1/alpha
    ggml_tensor * args[7] = {x, stats, h, nw, nw ? m->w(pfx + ".norm.bias") : nullptr,
                             act == 2 ? m->w(alpha_name) : nullptr, act == 2 ? m->w(alpha_name + ".inv") : nullptr};
    adain_act_params * raw = prm.get();
    g.op_params.push_back(std::move(prm));
    const int64_t Tp = pl + T + pr, V = (Tp + phase - 1) / phase;
    return ggml_custom_4d(ctx, GGML_TYPE_F32, std::max(C, c_pad), phase > 1 ? V : Tp, phase > 1 ? phase : 1, 1,
                          args, 7, kt_adain_apply_op, std::min(m->n_threads, KT_STAT_SLOTS), raw);
}

// AdaLayerNorm: LayerNorm over channels (no affine) then (1 + gamma) * x + beta.
ggml_tensor * ada_layer_norm(graph_ctx & g, ggml_tensor * x, ggml_tensor * s, const std::string & pfx) {
    ggml_context * ctx = g.ctx;
    const int64_t C = x->ne[0];
    ggml_tensor * h = linear(ctx, s, g.m->w(pfx + ".fc.weight"), g.m->w(pfx + ".fc.bias"));
    ggml_tensor * gam = ggml_scale_bias(ctx, ggml_view_1d(ctx, h, C, 0), 1.0f, 1.0f);
    ggml_tensor * bet = ggml_view_1d(ctx, h, C, C * sizeof(float));
    x = ggml_norm(ctx, x, 1e-5f);
    return ggml_add(ctx, ggml_mul(ctx, x, gam), bet);
}

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    x = ggml_norm(ctx, x, eps);
    return ggml_add(ctx, ggml_mul(ctx, x, w), b);
}

ggml_tensor * upsample2(ggml_context * ctx, ggml_tensor * x) {
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    const int64_t C = x->ne[0], T = x->ne[1];
    ggml_tensor * x3 = ggml_reshape_3d(ctx, x, C, 1, T);
    return ggml_reshape_2d(ctx, ggml_concat(ctx, x3, x3, 1), C, 2 * T);
}

// depthwise ConvTranspose1d(C, C, 3, stride 2, padding 1, output_padding 1): out[2m] = w1 x[m],
// out[2m+1] = w2 x[m] + w0 x[m+1].
ggml_tensor * dw_pool(graph_ctx & g, ggml_tensor * x, const std::string & pfx) {
    ggml_context * ctx = g.ctx;
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    const int64_t C = x->ne[0], T = x->ne[1];
    ggml_tensor * w = g.m->w(pfx + ".weight");                 // [C, 3]
    ggml_tensor * w0 = ggml_view_1d(ctx, w, C, 0);
    ggml_tensor * w1 = ggml_view_1d(ctx, w, C, w->nb[1]);
    ggml_tensor * w2 = ggml_view_1d(ctx, w, C, 2 * w->nb[1]);
    ggml_tensor * even = ggml_mul(ctx, x, w1);
    ggml_tensor * odd = ggml_mul(ctx, x, w2);
    if (T > 1) {
        ggml_tensor * nxt = ggml_view_2d(ctx, x, C, T - 1, x->nb[1], x->nb[1]);
        nxt = pad_time(ctx, ggml_cont(ctx, nxt), 0, 1);
        odd = ggml_add(ctx, odd, ggml_mul(ctx, nxt, w0));
    }
    ggml_tensor * y = ggml_concat(ctx, ggml_reshape_3d(ctx, even, C, 1, T), ggml_reshape_3d(ctx, odd, C, 1, T), 1);
    y = ggml_reshape_2d(ctx, y, C, 2 * T);
    return ggml_add(ctx, y, g.m->w(pfx + ".bias"));
}

// CUDA: conv1d(act(AdaIN(x)), w, b, dilation d, zero padding pl / pr) as one custom node (the library's kernels, run on the graph's
// stream). Without the implicit-GEMM kernel the node only writes the windows [Cp * K, Tout] and a ggml product and bias follow.
ggml_tensor * adain_conv_gpu(graph_ctx & g, ggml_tensor * x, ggml_tensor * s, const std::string & pfx, int act, float slope,
                             const std::string & alpha_name, ggml_tensor * w, ggml_tensor * b, int d, int pl, int pr,
                             ggml_tensor * resid = nullptr, bool * resid_done = nullptr) {
    ggml_context * ctx = g.ctx;
    kt_tiny_model * m = g.m;
    if (x->nb[0] != sizeof(float)) x = ggml_cont(ctx, x);
    const int64_t C = x->ne[0], T = x->ne[1];
    const int64_t Cp = w->ne[0], K = w->ne[1], Cout = w->ne[2];
    const int64_t Tout = T + pl + pr - (int64_t) d * (K - 1);
    GGML_ASSERT(Tout > 0 && Cp >= C);
    ggml_tensor * h = linear(ctx, s, m->w(pfx + ".fc.weight"), m->w(pfx + ".fc.bias"));      // [2C, 1]
    auto prm = std::make_unique<adain_gpu_params>();
#ifdef KT_TINY_CUDA
    prm->cuda = m->cuda;
#endif
    prm->act = act; prm->slope = slope; prm->Cp = (int) Cp; prm->K = (int) K; prm->d = d; prm->pl = pl; prm->Tout = (int) Tout;
    ggml_tensor * nw = m->wo(pfx + ".norm.weight");
    ggml_tensor * args[9] = {x, h, nw, nw ? m->w(pfx + ".norm.bias") : nullptr,
                             act == 2 ? m->w(alpha_name) : nullptr, act == 2 ? m->w(alpha_name + ".inv") : nullptr, nullptr, nullptr, nullptr};
    adain_gpu_params * raw = prm.get();
    g.adain_params.push_back(std::move(prm));
    g.adain_max_C = std::max<int>(g.adain_max_C, (int) C);
    g.adain_max_pad = std::max<long long>(g.adain_max_pad, (long long) (Tout + (int64_t) d * (K - 1)) * Cp);
    g.conv_flops += 2.0 * Cp * K * Tout * Cout;
    if (m->igemm_min_t >= 0 && Tout >= m->igemm_min_t && Cp % 16 == 0 && Cout % 4 == 0 && w->type == GGML_TYPE_F32 &&
        (!b || b->type == GGML_TYPE_F32)) {
        // the node is the whole convolution: no window matrix, and the bias is added in the kernel
        raw->Cout = (int) Cout;
        args[6] = w; args[7] = b;
        int nargs = 8;
        if (resid && resid_done && resid->type == GGML_TYPE_F32 && resid->ne[0] == Cout && resid->ne[1] == Tout && resid->nb[0] == sizeof(float) &&
            resid->nb[1] == Cout * sizeof(float) && b) {
            args[8] = resid;   // x + conv(...) inside the kernel
            nargs = 9;
            *resid_done = true;
        }
        return ggml_custom_4d(ctx, GGML_TYPE_F32, Cout, Tout, 1, 1, args, nargs, nullptr, 1, raw);
    }
    g.win_bytes += (double) Cp * K * Tout * sizeof(float);
    ggml_tensor * win = ggml_custom_4d(ctx, GGML_TYPE_F32, Cp * K, Tout, 1, 1, args, 6, nullptr, 1, raw);
    ggml_tensor * y = mm(ctx, ggml_reshape_2d(ctx, w, Cp * K, Cout), win);
    return b ? ggml_add(ctx, y, b) : y;
}

// AdainResBlk1d (predictor F0 / N blocks, decoder encode / decode blocks)
ggml_tensor * adain_resblk(graph_ctx & g, ggml_tensor * x, ggml_tensor * s, const std::string & pfx, bool upsample) {
    ggml_context * ctx = g.ctx;
    kt_tiny_model * m = g.m;
    ggml_tensor * w1 = m->w(pfx + ".conv1.weight"), * w2 = m->w(pfx + ".conv2.weight");
    ggml_tensor * r;
    if (t_gpu_adain) {
        if (upsample) {
            r = adain(g, x, s, pfx + ".norm1");
            r = leaky(ctx, r, 0.2f);
            r = dw_pool(g, r, pfx + ".pool");
            r = conv1d(ctx, r, w1, m->w(pfx + ".conv1.bias"), 1, 1, 1);
        } else {
            r = adain_conv_gpu(g, x, s, pfx + ".norm1", 1, 0.2f, "", w1, m->w(pfx + ".conv1.bias"), 1, 1, 1);
        }
        r = adain_conv_gpu(g, r, s, pfx + ".norm2", 1, 0.2f, "", w2, m->w(pfx + ".conv2.bias"), 1, 1, 1);
    } else if (t_cpu_fused) {
        if (upsample) {
            r = adain_act_cpu(g, x, s, pfx + ".norm1", 1, "", 0, 0, 0);
            r = dw_pool(g, r, pfx + ".pool");
            r = conv1d(ctx, r, w1, m->w(pfx + ".conv1.bias"), 1, 1, 1);
        } else {
            r = adain_act_cpu(g, x, s, pfx + ".norm1", 1, "", w1->ne[0], 1, 1);
            r = conv1d(ctx, r, w1, m->w(pfx + ".conv1.bias"), 1, 0, 0);
        }
        r = adain_act_cpu(g, r, s, pfx + ".norm2", 1, "", w2->ne[0], 1, 1);
        r = conv1d(ctx, r, w2, m->w(pfx + ".conv2.bias"), 1, 0, 0);
    } else {
        r = adain(g, x, s, pfx + ".norm1");
        r = leaky(ctx, r, 0.2f);
        if (upsample) r = dw_pool(g, r, pfx + ".pool");
        r = conv1d(ctx, r, w1, m->w(pfx + ".conv1.bias"), 1, 1, 1);
        r = adain(g, r, s, pfx + ".norm2");
        r = leaky(ctx, r, 0.2f);
        r = conv1d(ctx, r, w2, m->w(pfx + ".conv2.bias"), 1, 1, 1);
    }
    ggml_tensor * sc = x;
    if (upsample) sc = upsample2(ctx, sc);
    if (ggml_tensor * wsc = m->wo(pfx + ".conv1x1.weight")) sc = conv1d(ctx, sc, wsc, nullptr, 1, 0, 0);
    return ggml_scale(ctx, ggml_add(ctx, r, sc), 0.70710678118654752f);
}

ggml_tensor * snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * a, ggml_tensor * inv_a) {
    ggml_tensor * t = ggml_sqr(ctx, ggml_sin(ctx, ggml_mul(ctx, x, a)));
    return ggml_add(ctx, x, ggml_mul(ctx, t, inv_a));
}

// AdaINResBlock1 (generator): three (AdaIN, snake, dilated conv, AdaIN, snake, conv) residual units
ggml_tensor * adain_resblock1(graph_ctx & g, ggml_tensor * x, ggml_tensor * s, const std::string & pfx, int k,
                              const std::vector<int> & dil) {
    ggml_context * ctx = g.ctx;
    kt_tiny_model * m = g.m;
    for (size_t j = 0; j < dil.size(); j++) {
        const std::string J = std::to_string(j);
        ggml_tensor * c1 = m->w(pfx + ".convs1." + J + ".weight"), * c2 = m->w(pfx + ".convs2." + J + ".weight");
        const int p1 = (k * dil[j] - dil[j]) / 2, p2 = (k - 1) / 2;
        ggml_tensor * xt;
        if (t_gpu_adain) {
            xt = adain_conv_gpu(g, x, s, pfx + ".adain1." + J, 2, 0.0f, pfx + ".alpha1." + J, c1, m->w(pfx + ".convs1." + J + ".bias"), dil[j], p1, p1);
            bool fused = false;   // the residual add is done inside the convolution kernel when it can
            xt = adain_conv_gpu(g, xt, s, pfx + ".adain2." + J, 2, 0.0f, pfx + ".alpha2." + J, c2, m->w(pfx + ".convs2." + J + ".bias"), 1, p2, p2, x, &fused);
            if (fused) { x = xt; continue; }
        } else if (t_cpu_fused) {
            const int64_t T = x->ne[1];
            xt = adain_act_cpu(g, x, s, pfx + ".adain1." + J, 2, pfx + ".alpha1." + J, c1->ne[0], p1, p1, dil[j]);
            xt = conv1d(ctx, xt, c1, m->w(pfx + ".convs1." + J + ".bias"), dil[j], 0, 0);
            if (xt->ne[1] != T) xt = ggml_view_2d(ctx, xt, xt->ne[0], T, xt->nb[1], 0);
            xt = adain_act_cpu(g, xt, s, pfx + ".adain2." + J, 2, pfx + ".alpha2." + J, c2->ne[0], p2, p2);
            xt = conv1d(ctx, xt, c2, m->w(pfx + ".convs2." + J + ".bias"), 1, 0, 0);
        } else {
            xt = adain(g, x, s, pfx + ".adain1." + J);
            xt = snake(ctx, xt, m->w(pfx + ".alpha1." + J), m->w(pfx + ".alpha1." + J + ".inv"));
            xt = conv1d(ctx, xt, c1, m->w(pfx + ".convs1." + J + ".bias"), dil[j], p1, p1);
            xt = adain(g, xt, s, pfx + ".adain2." + J);
            xt = snake(ctx, xt, m->w(pfx + ".alpha2." + J), m->w(pfx + ".alpha2." + J + ".inv"));
            xt = conv1d(ctx, xt, c2, m->w(pfx + ".convs2." + J + ".bias"), 1, p2, p2);
        }
        x = ggml_add(ctx, xt, x);
    }
    return x;
}

ggml_tensor * concat_tree(ggml_context * ctx, const std::vector<ggml_tensor *> & v, size_t lo, size_t hi, int dim) {
    if (hi - lo == 1) return v[lo];
    size_t mid = (lo + hi) / 2;
    return ggml_concat(ctx, concat_tree(ctx, v, lo, mid, dim), concat_tree(ctx, v, mid, hi, dim), dim);
}

// CPU fast path for the LSTM recurrence: one custom op runs all time steps of both directions
// (one thread per direction) instead of ~15 graph nodes per step. src0 = input projections incl. biases
// [4H, T, 2] (direction 1 already time-reversed), src1 = W_hh [H, 4H, 2] (any type, converted to f32 here).
void kt_lstm_op(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * gx = dst->src[0], * whh = dst->src[1];
    const int64_t H = whh->ne[0], G = 4 * H, T = gx->ne[1];
    const auto * tt = ggml_get_type_traits(whh->type);
    for (int d = ith; d < 2; d += nth) {
        // W_hh transposed to [H][4H]: the step is a sum of axpy's over h, which vectorises without reordering sums
        std::vector<float> Wt((size_t) G * H), row(H);
        for (int64_t r = 0; r < G; r++) {
            const char * src = (const char *) whh->data + r * whh->nb[1] + d * whh->nb[2];
            if (whh->type == GGML_TYPE_F32) memcpy(row.data(), src, H * sizeof(float));
            else tt->to_float(src, row.data(), H);
            for (int64_t k = 0; k < H; k++) Wt[k * G + r] = row[k];
        }
        std::vector<float> h(H, 0.0f), c(H, 0.0f), gt(G);
        for (int64_t st = 0; st < T; st++) {
            const float * g0 = (const float *) ((const char *) gx->data + st * gx->nb[1] + d * gx->nb[2]);
            for (int64_t r = 0; r < G; r++) gt[r] = 0.0f;
            for (int64_t k = 0; k < H; k++) {
                const float hk = h[k];
                const float * wk = &Wt[k * G];
                float * gp = gt.data();
                for (int64_t r = 0; r < G; r++) gp[r] += wk[r] * hk;
            }
            for (int64_t r = 0; r < G; r++) gt[r] += g0[r];
            for (int64_t k = 0; k < H; k++) {
                const float ig = 1.0f / (1.0f + std::exp(-gt[k]));
                const float fg = 1.0f / (1.0f + std::exp(-gt[H + k]));
                const float gg = std::tanh(gt[2 * H + k]);
                const float og = 1.0f / (1.0f + std::exp(-gt[3 * H + k]));
                c[k] = fg * c[k] + ig * gg;
                h[k] = og * std::tanh(c[k]);
            }
            const int64_t t = d == 0 ? st : T - 1 - st;
            memcpy((char *) dst->data + t * dst->nb[1] + d * H * sizeof(float), h.data(), H * sizeof(float));
        }
    }
}

// bidirectional single-layer LSTM (PyTorch gate order i, f, g, o). x [In, T] -> [2H, T].
// Both directions run in one batched step: direction 1 reads the time-reversed input.
ggml_tensor * bilstm(graph_ctx & g, ggml_tensor * x, const std::string & pfx, ggml_tensor * rev) {
    ggml_context * ctx = g.ctx;
    kt_tiny_model * m = g.m;
    ggml_tensor * wih = m->w(pfx + ".w_ih");   // [In, 4H, 2]
    ggml_tensor * whh = m->w(pfx + ".w_hh");   // [H, 4H, 2]
    ggml_tensor * b = m->w(pfx + ".b");        // [4H, 2]
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    const int64_t In = x->ne[0], T = x->ne[1], H = whh->ne[0], G = 4 * H;
    ggml_tensor * xr = ggml_get_rows(ctx, x, rev);
    ggml_tensor * x2 = ggml_concat(ctx, ggml_reshape_3d(ctx, x, In, T, 1), ggml_reshape_3d(ctx, xr, In, T, 1), 2);
    ggml_tensor * gx = mm(ctx, wih, x2);                         // [4H, T, 2]
    gx = ggml_add(ctx, gx, ggml_reshape_3d(ctx, b, G, 1, 2));
    if (t_cpu_fused) {
        ggml_tensor * args[] = {gx, whh};
        return ggml_custom_4d(ctx, GGML_TYPE_F32, 2 * H, T, 1, 1, args, 2, kt_lstm_op, 2, nullptr);
    }
    if (t_gpu_lstm && t_gpu_nodes) {
        // graph node run on the backend's stream (no host wait, inside CUDA graphs); source 0 = gx [4H, T, 2]
        auto prm = std::make_unique<lstm_gpu_params>();
#ifdef KT_TINY_CUDA
        prm->cuda = m->cuda;
#endif
        prm->key = pfx; prm->H = (int) H; prm->T = (int) T;
        ggml_tensor * args[] = {gx};
        lstm_gpu_params * raw = prm.get();
        g.lstm_params.push_back(std::move(prm));
        return ggml_custom_4d(ctx, GGML_TYPE_F32, 2 * H, T, 1, 1, args, 1, nullptr, 1, raw);
    }
    if (t_gpu_lstm) {
        // placeholder [2H, T] that depends on gx (so gx is computed and still alive); the CUDA kernel overwrites it
        ggml_tensor * ph = ggml_cont(ctx, ggml_view_2d(ctx, gx, 2 * H, T, gx->nb[1], 0));
        ggml_set_name(ph, "kt_gpu_lstm");
        gpu_op op;
        op.kind = 1; op.key = pfx; op.gx = gx; op.H = (int) H; op.T = (int) T;
        g.gpu_ops[ph] = op;
        return ph;
    }
    ggml_tensor * h = nullptr, * c = nullptr;
    std::vector<ggml_tensor *> outs;
    outs.reserve(T);
    for (int64_t t = 0; t < T; t++) {
        ggml_tensor * gt = ggml_view_3d(ctx, gx, G, 1, 2, gx->nb[1], gx->nb[2], t * gx->nb[1]);
        gt = h ? ggml_add(ctx, mm(ctx, whh, h), gt) : ggml_cont(ctx, gt);
        ggml_tensor * sg = ggml_sigmoid(ctx, gt);
        ggml_tensor * tg = ggml_tanh(ctx, gt);
        const size_t es = sizeof(float);
        ggml_tensor * gi = ggml_view_3d(ctx, sg, H, 1, 2, sg->nb[1], sg->nb[2], 0);
        ggml_tensor * gf = ggml_view_3d(ctx, sg, H, 1, 2, sg->nb[1], sg->nb[2], H * es);
        ggml_tensor * gg = ggml_view_3d(ctx, tg, H, 1, 2, tg->nb[1], tg->nb[2], 2 * H * es);
        ggml_tensor * go = ggml_view_3d(ctx, sg, H, 1, 2, sg->nb[1], sg->nb[2], 3 * H * es);
        ggml_tensor * ig = ggml_mul(ctx, gi, gg);
        c = c ? ggml_add(ctx, ggml_mul(ctx, gf, c), ig) : ig;
        h = ggml_mul(ctx, go, ggml_tanh(ctx, c));
        outs.push_back(h);
    }
    ggml_tensor * hs = concat_tree(ctx, outs, 0, outs.size(), 1);         // [H, T, 2], step order
    ggml_tensor * fwd = ggml_view_2d(ctx, hs, H, T, hs->nb[1], 0);
    ggml_tensor * bwd = ggml_view_2d(ctx, hs, H, T, hs->nb[1], hs->nb[2]);
    bwd = ggml_get_rows(ctx, bwd, rev);                                     // back to time order
    return ggml_concat(ctx, fwd, bwd, 0);
}

ggml_tensor * gelu_new(ggml_context * ctx, ggml_tensor * x) {
    // 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), written out: the CPU ggml_gelu uses an fp16 table
    ggml_tensor * x3 = ggml_mul(ctx, ggml_sqr(ctx, x), x);
    ggml_tensor * inner = ggml_add(ctx, x, ggml_scale(ctx, x3, 0.044715f));
    ggml_tensor * t = ggml_tanh(ctx, ggml_scale(ctx, inner, 0.7978845608028654f));
    return ggml_mul(ctx, ggml_scale(ctx, x, 0.5f), ggml_scale_bias(ctx, t, 1.0f, 1.0f));
}

ggml_tensor * albert(graph_ctx & g, ggml_tensor * tok, ggml_tensor * pos) {
    ggml_context * ctx = g.ctx;
    kt_tiny_model * m = g.m;
    const int64_t T = tok->ne[0];
    ggml_tensor * tte = m->w("bert.embeddings.token_type_embeddings.weight");
    ggml_tensor * e = ggml_get_rows(ctx, m->w("bert.embeddings.word_embeddings.weight"), tok);
    e = ggml_add(ctx, e, ggml_get_rows(ctx, m->w("bert.embeddings.position_embeddings.weight"), pos));
    e = ggml_add(ctx, e, ggml_view_1d(ctx, tte, tte->ne[0], 0));
    e = layer_norm(ctx, e, m->w("bert.embeddings.LayerNorm.weight"), m->w("bert.embeddings.LayerNorm.bias"), m->a_eps);
    ggml_tensor * h = linear(ctx, e, m->w("bert.encoder.embedding_hidden_mapping_in.weight"),
                             m->w("bert.encoder.embedding_hidden_mapping_in.bias"));
    const std::string L = "bert.layer.";   // the shared ALBERT layer (renamed by the converter)
    const int nh = m->a_heads, dh = m->a_hidden / m->a_heads;
    for (int l = 0; l < m->a_layers; l++) {
        ggml_tensor * q = linear(ctx, h, m->w(L + "attention.query.weight"), m->w(L + "attention.query.bias"));
        ggml_tensor * k = linear(ctx, h, m->w(L + "attention.key.weight"), m->w(L + "attention.key.bias"));
        ggml_tensor * v = linear(ctx, h, m->w(L + "attention.value.weight"), m->w(L + "attention.value.bias"));
        q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, dh, nh, T), 0, 2, 1, 3));   // [dh, T, nh]
        k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, dh, nh, T), 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, dh, nh, T), 1, 2, 0, 3));   // [T, dh, nh]
        ggml_tensor * kq = mm(ctx, k, q);                                             // [Tk, Tq, nh]
        kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f / std::sqrt((float) dh), 0.0f);
        ggml_tensor * o = mm(ctx, v, kq);                                             // [dh, Tq, nh]
        o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));                                   // [dh, nh, Tq]
        o = ggml_reshape_2d(ctx, o, m->a_hidden, T);
        o = linear(ctx, o, m->w(L + "attention.dense.weight"), m->w(L + "attention.dense.bias"));
        ggml_tensor * a = layer_norm(ctx, ggml_add(ctx, h, o), m->w(L + "attention.LayerNorm.weight"),
                                     m->w(L + "attention.LayerNorm.bias"), m->a_eps);
        ggml_tensor * f = linear(ctx, a, m->w(L + "ffn.weight"), m->w(L + "ffn.bias"));
        f = gelu_new(ctx, f);
        f = linear(ctx, f, m->w(L + "ffn_output.weight"), m->w(L + "ffn_output.bias"));
        h = layer_norm(ctx, ggml_add(ctx, f, a), m->w(L + "full_layer_layer_norm.weight"),
                       m->w(L + "full_layer_layer_norm.bias"), m->a_eps);
    }
    return h;
}

// ---------------------------------------------------------------------------------------------------
// running graphs

ggml_backend_sched_t ensure_sched(kt_tiny_model * m, size_t n_nodes, bool cpu_only = false) {
    ggml_backend_sched_t & sc = cpu_only ? m->sched_cpu : m->sched;
    size_t & sz = cpu_only ? m->sched_cpu_size : m->sched_size;
    if (sc && sz >= n_nodes) return sc;
    if (sc) ggml_backend_sched_free(sc);
    sz = std::max<size_t>(n_nodes, 1 << 16);
    ggml_backend_t * bk = cpu_only ? &m->cpu : m->backends.data();
    sc = ggml_backend_sched_new(bk, nullptr, cpu_only ? 1 : (int) m->backends.size(), sz, false, true);
    if (!sc) throw std::runtime_error("ggml_backend_sched_new failed");
    return sc;
}

// KT_TINY_PROFILE=1: time every graph node (by op, and by op + unary/view kind) via the scheduler's eval callback
struct op_profile {
    double last = 0;
    std::map<std::string, std::pair<double, int>> by_op;
};

void run_gpu_op(graph_ctx & g, ggml_tensor * t, const gpu_op & op);

// scheduler eval callback: stops after each placeholder of a library GPU kernel (and after every node when profiling)
struct eval_state {
    graph_ctx * g = nullptr;
    op_profile * prof = nullptr;
    std::string error;
    // KT_TINY_NODE_DUMP=DIR (debug builds): every f32 node's output is written to DIR/<graph>_<node index>_<op>.f32,
    // so that two devices' runs of the same graph can be compared node by node
    const char * node_dump = nullptr;
    int graph_id = 0;
    std::map<const ggml_tensor *, int> node_index;
};

bool eval_cb(ggml_tensor * t, bool ask, void * ud) {
    auto * st = (eval_state *) ud;
    auto it = st->g->gpu_ops.find(t);
    if (ask) return st->prof != nullptr || st->node_dump != nullptr || it != st->g->gpu_ops.end();
    if (st->node_dump && t->type == GGML_TYPE_F32) {
        auto ni = st->node_index.find(t);
        if (ni != st->node_index.end()) {
            std::vector<float> v(ggml_nelements(t));
            if (ggml_is_contiguous(t)) {
                ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
            } else {
                v.clear();   // views with gaps or overlaps: not dumped
            }
            char name[512];
            snprintf(name, sizeof(name), "%s/%c_%05d_%s.f32", st->node_dump, 'A' + st->graph_id, ni->second, ggml_op_desc(t));
            if (FILE * f = fopen(name, "wb")) {
                fwrite(v.data(), sizeof(float), v.size(), f);
                fclose(f);
            }
        }
    }
    if (it != st->g->gpu_ops.end()) {
        try {
            run_gpu_op(*st->g, t, it->second);
        } catch (const std::exception & e) {
            st->error = e.what();
            return false;   // stop the graph
        }
    }
    if (st->prof) {
        const double now = now_ms();
        std::string k = it != st->g->gpu_ops.end() ? std::string(it->second.kind == 1 ? "LSTM_CUDA" : "SOURCE_CUDA") : ggml_op_desc(t);
        auto & e = st->prof->by_op[k];
        e.first += now - st->prof->last;
        e.second += 1;
        st->prof->last = now;
    }
    return true;
}

#ifdef KT_TINY_CUDA
// Error of a fused kernel (the handler cannot throw through the backend). ggml's CUDA backend calls the handler on the
// thread that runs ggml_backend_sched_graph_compute, so one per thread keeps models on separate threads independent.
thread_local std::string g_adain_err;

// GGML_OP_CUSTOM handler of the CUDA backend: runs the fused AdaIN kernels on the graph's stream
void adain_cuda_handler(ggml_tensor * dst, void * stream, void * ud) {
    if (*(const int *) ud == KT_NODE_LSTM) {
        const auto * q = (const lstm_gpu_params *) ud;
        const ggml_tensor * gx = dst->src[0];
        const long long es = (long long) sizeof(float);
        if (kt_cuda_lstm_async(q->cuda, stream, q->key.c_str(), (const float *) gx->data, (long long) gx->nb[1] / es,
                               (long long) gx->nb[2] / es, (float *) dst->data, (long long) dst->nb[1] / es, q->H, q->T) != 0) {
            g_adain_err = std::string("CUDA LSTM: ") + kt_cuda_last_error();
        }
        return;
    }
    const auto * p = (const adain_gpu_params *) ud;
    const ggml_tensor * x = dst->src[0], * h = dst->src[1], * nw = dst->src[2], * nb = dst->src[3], * al = dst->src[4], * ial = dst->src[5];
    if (p->Cout > 0) {
        const ggml_tensor * w = dst->src[6], * bi = dst->src[7], * rs = dst->src[8];
        if (kt_cuda_adain_conv(p->cuda, stream, (const float *) x->data, (long long) (x->nb[1] / sizeof(float)), (int) x->ne[0], (int) x->ne[1],
                               (const float *) h->data, nw ? (const float *) nw->data : nullptr, nb ? (const float *) nb->data : nullptr,
                               al ? (const float *) al->data : nullptr, ial ? (const float *) ial->data : nullptr, p->act, p->slope, 1e-5f,
                               p->Cp, p->K, p->d, p->pl, p->Tout, (const float *) w->data, bi ? (const float *) bi->data : nullptr, rs ? (const float *) rs->data : nullptr, p->Cout,
                               (float *) dst->data) != 0) {
            g_adain_err = std::string("CUDA AdaIN conv: ") + kt_cuda_last_error();
        }
        return;
    }
    if (kt_cuda_adain_window(p->cuda, stream, (const float *) x->data, (long long) (x->nb[1] / sizeof(float)), (int) x->ne[0], (int) x->ne[1],
                             (const float *) h->data, nw ? (const float *) nw->data : nullptr, nb ? (const float *) nb->data : nullptr,
                             al ? (const float *) al->data : nullptr, ial ? (const float *) ial->data : nullptr, p->act, p->slope, 1e-5f,
                             p->Cp, p->K, p->d, p->pl, p->Tout, (float *) dst->data) != 0) {
        g_adain_err = std::string("CUDA AdaIN: ") + kt_cuda_last_error();
    }
}
#endif

struct graph_run {
    graph_ctx g;
    std::vector<char> mem;
    size_t n_nodes;
    bool on_cpu;   // run on the CPU backend alone (Metal models: graphs A and B)
    graph_run(kt_tiny_model * m, size_t max_nodes, bool cpu_only = false) : n_nodes(max_nodes), on_cpu(cpu_only) {
        const size_t n_tensors = max_nodes * 2 + 256;
        mem.resize(ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(max_nodes, false) + 1024 * 1024);
        ggml_init_params ip = { mem.size(), mem.data(), true };
        g.m = m;
        g.ctx = ggml_init(ip);
        g.gf = ggml_new_graph_custom(g.ctx, n_nodes, false);
    }
    ~graph_run() { ggml_free(g.ctx); }
    void output(ggml_tensor * t) {
        ggml_set_output(t);
        ggml_build_forward_expand(g.gf, t);
    }
    void compute() {
        kt_tiny_model * m = g.m;
        ggml_backend_sched_t sched = ensure_sched(m, 2 * (size_t) ggml_graph_n_nodes(g.gf) + 4096, on_cpu);
#ifdef KT_TINY_CUDA
        if (g.adain_max_C > 0 && kt_cuda_adain_reserve(m->cuda, g.adain_max_C, g.adain_max_pad) != 0) {
            throw std::runtime_error(std::string("CUDA AdaIN scratch: ") + kt_cuda_last_error());
        }
        g_adain_err.clear();
#endif
        // node indices for KT_TINY_NODE_DUMP, taken before the scheduler (and a backend's graph optimisation) reorders nodes
        std::map<const ggml_tensor *, int> node_index;
        const char * node_dump = debug_env("KT_TINY_NODE_DUMP");
        if (node_dump) for (int i = 0; i < ggml_graph_n_nodes(g.gf); i++) node_index[ggml_graph_node(g.gf, i)] = i;
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, g.gf)) throw std::runtime_error("graph allocation failed");
        for (auto & in : g.inputs) ggml_backend_tensor_set(in.first, in.second.data(), 0, in.second.size());
        op_profile prof;
        const bool profile = profile_enabled();
        eval_state es;
        es.g = &g;
        if (profile) {
            es.prof = &prof;
            prof.last = now_ms();
        }
        if (node_dump) {
            static thread_local int graph_counter = 0;
            es.node_dump = node_dump;
            es.graph_id = graph_counter++ % 3;   // graphs A (tokens), B (frames), C (decoder) of one synthesis
            es.node_index = std::move(node_index);
        }
        const bool cb = profile || es.node_dump || !g.gpu_ops.empty();
        if (cb) ggml_backend_sched_set_eval_callback(sched, eval_cb, &es);
        const ggml_status st = ggml_backend_sched_graph_compute(sched, g.gf);
        if (cb) ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
        if (!es.error.empty()) throw std::runtime_error(es.error);
#ifdef KT_TINY_CUDA
        if (!g_adain_err.empty()) { const std::string e = g_adain_err; g_adain_err.clear(); throw std::runtime_error(e); }
#endif
        if (profile) {
            std::vector<std::pair<double, std::string>> v;
            for (auto & e : prof.by_op) v.push_back({e.second.first, e.first + " x" + std::to_string(e.second.second)});
            std::sort(v.rbegin(), v.rend());
            fprintf(stderr, "profile (graph of %d nodes, fused conv %.1f GFLOP, windows %.0f MB:", ggml_graph_n_nodes(g.gf), g.conv_flops / 1e9, g.win_bytes / 1e6);
            for (size_t i = 0; i < std::min<size_t>(v.size(), 12); i++) fprintf(stderr, " %s %.1f ms;", v[i].second.c_str(), v[i].first);
            fprintf(stderr, "\n");
        }
        if (st != GGML_STATUS_SUCCESS) throw std::runtime_error("graph compute failed");
        size_t total = 0;
        if (on_cpu) total = ggml_backend_sched_get_buffer_size(sched, m->cpu);
        else for (auto * b : m->backends) total += ggml_backend_sched_get_buffer_size(sched, b);
        m->rep["splits"].push_back(ggml_backend_sched_get_n_splits(sched));
        m->rep["compute_buffer_mb_max"] = std::max(m->rep.value("compute_buffer_mb_max", 0.0), total / 1e6);
        m->rep["graph_nodes"].push_back(ggml_graph_n_nodes(g.gf));
    }
};

std::vector<float> get_f32(ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

void dump(const char * name, const std::vector<float> & v) {
    const char * dir = debug_env("KT_TINY_DUMP");
    if (!dir || !*dir) return;
    std::string p = std::string(dir) + "/" + name + ".f32";
    FILE * f = fopen(p.c_str(), "wb");
    if (!f) return;
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
}

// run fn(begin, end) over [0, n) on up to n_threads threads
void parallel_for(int n_threads, int64_t n, const std::function<void(int64_t, int64_t)> & fn) {
    const int nt = (int) std::max<int64_t>(1, std::min<int64_t>(n_threads, n));
    if (nt == 1) { fn(0, n); return; }
    std::vector<std::thread> th;
    const int64_t per = (n + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        const int64_t b = t * per, e = std::min(n, b + per);
        if (b < e) th.emplace_back(fn, b, e);
    }
    for (auto & x : th) x.join();
}

// harmonic source (SineGen + SourceModuleHnNSF + forward STFT), float operations in PyTorch's order.
// f0: [L0] curve at 2x frame rate; returns har [frames][n_fft + 2] (ne0 = n_fft + 2).
int samples_per_f0(const kt_tiny_model * m) {
    int upp = m->hop;
    for (int r : m->up_r) upp *= r;
    return upp;
}

// per-block radians increment and its cumulative phase [L0][9] (cumsum accumulates in double, as PyTorch on the CPU)
std::vector<float> harmonic_phase(const kt_tiny_model * m, const std::vector<float> & f0) {
    const int nh = 9, upp = samples_per_f0(m);
    const int64_t L0 = (int64_t) f0.size();
    std::vector<float> p300((size_t) L0 * nh);
    std::vector<double> acc(nh, 0.0);
    const float pi_f = (float) M_PI;
    for (int64_t d = 0; d < L0; d++) {
        for (int h = 0; h < nh; h++) {
            float fn = f0[d] * (float) (h + 1);
            float rad = fn / (float) m->sample_rate;
            float md = std::fmod(rad, 1.0f);
            if (md != 0.0f && md < 0.0f) md += 1.0f;
            acc[h] += (double) md;
            float ph = (float) acc[h];
            ph = ph * 2.0f;
            ph = ph * pi_f;
            p300[d * nh + h] = ph * (float) upp;
        }
    }
    return p300;
}

std::vector<float> harmonic_source(const kt_tiny_model * m, const std::vector<float> & f0, const kt_tiny_synth_options & o,
                                   int64_t & n_frames_out) {
    const int nh = 9, upp = samples_per_f0(m);
    const int64_t L0 = (int64_t) f0.size(), L = L0 * upp;
    if (o.noise_mode == KT_TINY_NOISE_EXTERNAL && (!o.noise || o.noise_len != (size_t) (L * nh))) {
        throw std::runtime_error("external noise must have " + std::to_string(L * nh) + " values");
    }
    const std::vector<float> p300 = harmonic_phase(m, f0);
    std::vector<float> src(L);
    const bool dbg = debug_env("KT_TINY_DUMP") != nullptr;
    std::vector<float> phase_dbg(dbg ? (size_t) L * nh : 0);
    const float scale = (float) (1.0 / (double) upp);
    // samples in fixed blocks, each with its own generator seeded from (seed, block): the noise does not
    // depend on the thread count
    const int64_t BLK = 8192, nblk = (L + BLK - 1) / BLK;
    auto run_blocks = [&](int64_t b0, int64_t b1) {
        for (int64_t b = b0; b < b1; b++) {
            std::mt19937_64 rng(o.seed * 0x9E3779B97F4A7C15ull + (uint64_t) b);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            for (int64_t n = b * BLK; n < std::min(L, (b + 1) * BLK); n++) {
#if defined(__aarch64__) || defined(_M_ARM64)
                float s = std::fma(scale, (float) n + 0.5f, -0.5f);   // PyTorch's arm64 build contracts this (checked)
#else
                float s = scale * ((float) n + 0.5f) - 0.5f;          // and its x86 build does not (checked)
#endif
                if (s < 0.0f) s = 0.0f;
                int64_t i0 = std::min<int64_t>((int64_t) std::floor(s), L0 - 1);
                float l1 = std::min(std::max(s - (float) i0, 0.0f), 1.0f);
                int64_t i1 = i0 + (i0 < L0 - 1 ? 1 : 0);
                float l0 = 1.0f - l1;
                const float fv = f0[n / upp];
                const float uv = fv > 10.0f ? 1.0f : 0.0f;
                const float namp = uv * 0.003f + ((1.0f - uv) * 0.1f) / 3.0f;
                float acc1 = 0.0f;
                for (int h = 0; h < nh; h++) {
                    // PyTorch's CPU linear interpolation computes fma(p0, w0, p1 * w1) (checked bit-exact on x86)
                    float ph = std::fma(p300[i0 * nh + h], l0, p300[i1 * nh + h] * l1);
                    if (dbg) phase_dbg[n * nh + h] = ph;
                    float sw = std::sin(ph) * 0.1f;
                    float z = 0.0f;
                    if (o.noise_mode == KT_TINY_NOISE_RANDOM) z = nd(rng);
                    else if (o.noise_mode == KT_TINY_NOISE_EXTERNAL) z = o.noise[n * nh + h];
                    sw = sw * uv + namp * z;
                    acc1 += m->src_lin_w[h] * sw;
                }
                src[n] = std::tanh(acc1 + m->src_lin_b);
            }
        }
    };
    parallel_for(m->n_threads, nblk, run_blocks);
    dump("src", src);
    if (dbg) dump("phase", phase_dbg);
    // STFT: n_fft, hop, periodic Hann, center=True with reflect padding; magnitude and angle
    const int N = m->n_fft, hop = m->hop, nb = N / 2 + 1, pad = N / 2;
    const int64_t frames = L / hop + 1;
    std::vector<double> win(N);
    for (int i = 0; i < N; i++) win[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / N);
    auto at = [&](int64_t i) -> float {
        i -= pad;
        if (i < 0) i = -i;
        if (i >= L) i = 2 * (L - 1) - i;
        return src[std::max<int64_t>(0, std::min<int64_t>(L - 1, i))];
    };
    std::vector<float> har((size_t) frames * (N + 2));
    std::vector<double> cs(N * nb), sn(N * nb);
    for (int k = 0; k < nb; k++) for (int i = 0; i < N; i++) {
        cs[k * N + i] = std::cos(2.0 * M_PI * k * i / N);
        sn[k * N + i] = std::sin(2.0 * M_PI * k * i / N);
    }
    parallel_for(m->n_threads, frames, [&](int64_t f0_, int64_t f1_) {
        std::vector<double> seg(N);
        for (int64_t f = f0_; f < f1_; f++) {
            for (int i = 0; i < N; i++) seg[i] = (double) at(f * hop + i) * win[i];
            for (int k = 0; k < nb; k++) {
                double re = 0, im = 0;
                for (int i = 0; i < N; i++) { re += seg[i] * cs[k * N + i]; im -= seg[i] * sn[k * N + i]; }
                if (k == 0 || 2 * k == N) im = 0.0;   // a real FFT returns exact zeros here: angle 0 or +pi
                float fr = (float) re, fi = (float) im;
                har[f * (N + 2) + k] = std::hypot(fr, fi);
                har[f * (N + 2) + nb + k] = std::atan2(fi, fr);
            }
        }
    });
    n_frames_out = frames;
    return har;
}

// inverse STFT pieces: basis [22 -> n_fft] (irfft x window / N) and the inverse window envelope
std::vector<float> istft_basis(int N) {
    const int nb = N / 2 + 1;
    std::vector<float> B((size_t) N * 2 * nb);
    for (int n = 0; n < N; n++) {
        double w = 0.5 - 0.5 * std::cos(2.0 * M_PI * n / N);
        for (int k = 0; k < nb; k++) {
            double c = (k == 0 || k == N / 2) ? 1.0 : 2.0;
            double ang = 2.0 * M_PI * k * n / N;
            B[n * 2 * nb + k] = (float) (c * std::cos(ang) * w / N);
            B[n * 2 * nb + nb + k] = (k == 0 || k == N / 2) ? 0.0f : (float) (-c * std::sin(ang) * w / N);
        }
    }
    return B;
}

std::vector<float> istft_inv_envelope(int N, int hop, int64_t frames) {
    const int64_t full = (frames - 1) * hop + N, out = (frames - 1) * hop;
    std::vector<double> env(full, 0.0), w2(N);
    for (int n = 0; n < N; n++) {   // the window once (it was recomputed per frame: ~25 ms per 13 s chunk)
        const float w = (float) (0.5 - 0.5 * std::cos(2.0 * M_PI * n / N));
        w2[n] = (double) w * w;
    }
    for (int64_t f = 0; f < frames; f++) for (int n = 0; n < N; n++) env[f * hop + n] += w2[n];
    std::vector<float> inv(out);
    for (int64_t i = 0; i < out; i++) {
        double e = env[i + N / 2];
        inv[i] = e > 1e-11 ? (float) (1.0 / e) : 1.0f;
    }
    return inv;
}

void run_gpu_op(graph_ctx & g, ggml_tensor * t, const gpu_op & op) {
#ifdef KT_TINY_CUDA
    kt_tiny_model * m = g.m;
    // read the placeholder's sources from the node the scheduler ran: when a source lives on another backend (graph
    // inputs start on the CPU), the scheduler has replaced it with its copy on the GPU
    for (int i = 0; i < 3; i++) {
        const ggml_tensor * sv = t->src[i];
        if (sv && sv->buffer && ggml_backend_buffer_is_host(sv->buffer)) throw std::runtime_error("CUDA kernel input is not on the GPU");
    }
    if (op.kind == 1) {
        const long long es = (long long) sizeof(float);
        if (kt_cuda_lstm(m->cuda, op.key.c_str(), (const float *) t->src[0]->data, (long long) op.gx->nb[1] / es,
                         (long long) op.gx->nb[2] / es, (float *) t->data, (long long) t->nb[1] / es, op.H, op.T) != 0) {
            throw std::runtime_error(std::string("CUDA LSTM: ") + kt_cuda_last_error());
        }
    } else {
        const kt_tiny_synth_options & o = *g.opts;
        const int mode = o.noise_mode == KT_TINY_NOISE_NONE ? KT_CUDA_NOISE_NONE
                       : o.noise_mode == KT_TINY_NOISE_EXTERNAL ? KT_CUDA_NOISE_EXTERNAL : KT_CUDA_NOISE_RANDOM;
        if (kt_cuda_harmonic(m->cuda, (const float *) t->src[1]->data, (const float *) t->src[2]->data, op.L0, samples_per_f0(m),
                             m->src_lin_w.data(), m->src_lin_b, mode, (unsigned long long) o.seed, o.noise, o.noise_len,
                             m->n_fft, m->hop, (float *) t->data, (long long) op.frames) != 0) {
            throw std::runtime_error(std::string("CUDA harmonic source: ") + kt_cuda_last_error());
        }
    }
#else
    (void) g; (void) t; (void) op;
    throw std::runtime_error("this build has no CUDA kernels");
#endif
}

// ---------------------------------------------------------------------------------------------------

void synthesize(kt_tiny_model * m, const std::vector<int32_t> & tokens, const float * style, const kt_tiny_synth_options & o,
                std::vector<float> & audio, std::vector<int32_t> & dur_out) {
    m->rep = json::object();
    m->rep["graph_nodes"] = json::array();
    m->rep["splits"] = json::array();
    // graphs A and B: on the CPU backend's fast path when the model runs them there (Metal)
    t_window_cont = m->front_cpu ? false : m->window_cont;
    t_cpu_fused = m->front_cpu ? true : m->cpu_fused;
    t_gpu_lstm = m->gpu_lstm;
    t_gpu_adain = false;
    t_gpu_nodes = m->gpu_adain;   // the custom-node hook is registered
    t_mm_prec = m->strict_f32 ? GGML_PREC_F32_STRICT : GGML_PREC_F32;
    const double t0 = now_ms();
    const int64_t T = (int64_t) tokens.size();
    if (T < 2) throw std::runtime_error("need at least 2 tokens");
    if (T > 512) throw std::runtime_error("at most 512 tokens per chunk");
    for (int32_t id : tokens) {
        if (id < 0 || id >= m->n_vocab) {
            throw std::runtime_error("token id " + std::to_string(id) + " out of range [0, " + std::to_string(m->n_vocab) + ")");
        }
    }
    const float * sa = style;          // decoder (timbre)
    const float * sp = style + 128;    // predictor (prosody)
    std::vector<int32_t> pos(T), rev(T);
    for (int64_t i = 0; i < T; i++) { pos[i] = (int32_t) i; rev[i] = (int32_t) (T - 1 - i); }

    // ---- graph A: tokens ----
    std::vector<float> dur_raw, d_host, ten_host;
    const int hid = m->hid, dC = m->hid + m->style_dim;
    {
        // graph size: per-step LSTM nodes only exist on GPU backends without the library's LSTM kernel
        graph_run r(m, t_cpu_fused || t_gpu_lstm ? 8192 : 16384 + (size_t) 40 * T * (m->n_layer + 2), m->front_cpu);
        graph_ctx & g = r.g;
        ggml_context * ctx = g.ctx;
        ggml_tensor * tok = g.input_i32("tokens", tokens);
        ggml_tensor * tpos = g.input_i32("positions", pos);
        ggml_tensor * trev = g.input_i32("reverse", rev);
        ggml_tensor * s = g.input_f32("style_p", {m->style_dim, 1}, sp);
        ggml_tensor * bert = albert(g, tok, tpos);
        ggml_tensor * d_en = linear(ctx, bert, m->w("bert_encoder.weight"), m->w("bert_encoder.bias"));
        ggml_tensor * srep = ggml_repeat_4d(ctx, s, m->style_dim, T, 1, 1);
        ggml_tensor * x = ggml_concat(ctx, d_en, srep, 0);
        for (int l = 0; l < m->n_layer; l++) {
            const std::string P = "predictor.text_encoder.lstms.";
            x = bilstm(g, x, P + std::to_string(2 * l), trev);
            x = ada_layer_norm(g, x, s, P + std::to_string(2 * l + 1));
            x = ggml_concat(ctx, x, srep, 0);
        }
        ggml_tensor * d = x;                                                   // [hid + 128, T]
        ggml_tensor * dl = bilstm(g, d, "predictor.lstm", trev);
        ggml_tensor * dur = linear(ctx, dl, m->w("predictor.duration_proj.linear_layer.weight"),
                                   m->w("predictor.duration_proj.linear_layer.bias"));
        dur = ggml_sum_rows(ctx, ggml_sigmoid(ctx, dur));                     // [1, T]
        ggml_tensor * t = ggml_get_rows(ctx, m->w("text_encoder.embedding.weight"), tok);
        const int pk = (m->te_kernel - 1) / 2;
        for (int l = 0; l < m->n_layer; l++) {
            const std::string P = "text_encoder.cnn." + std::to_string(l);
            t = conv1d(ctx, t, m->w(P + ".0.weight"), m->w(P + ".0.bias"), 1, pk, pk);
            t = layer_norm(ctx, t, m->w(P + ".1.gamma"), m->w(P + ".1.beta"), 1e-5f);
            t = leaky(ctx, t, 0.2f);
        }
        ggml_tensor * ten = bilstm(g, t, "text_encoder.lstm", trev);         // [hid, T]
        r.output(dur);
        r.output(d);
        r.output(ten);
        r.compute();
        dur_raw = get_f32(dur);
        d_host = get_f32(d);
        ten_host = get_f32(ten);
    }
    const double t1 = now_ms();
    dump("dur_raw", dur_raw);
    dump("d", d_host);
    dump("t_en", ten_host);

    // ---- durations -> frame map ----
    dur_out.assign(T, 1);
    for (int64_t i = 0; i < T; i++) {
        if (o.durations) {
            if (o.durations[i] < 0 || o.durations[i] > 1000) throw std::runtime_error("durations must be in [0, 1000] frames");
            dur_out[i] = o.durations[i];
        } else {
            float v = dur_raw[i] / o.speed;
            float rv = std::nearbyint(v);
            // a very low speed would stretch a phoneme without bound (and overflow int32): same limit as given durations
            if (!(rv <= 1000.0f)) throw std::runtime_error("speed is too low: a phoneme would last more than 1000 frames");
            dur_out[i] = std::max(1, (int32_t) rv);
        }
    }
    std::vector<int32_t> idx;
    for (int64_t i = 0; i < T; i++) for (int k = 0; k < dur_out[i]; k++) idx.push_back((int32_t) i);
    const int64_t F = (int64_t) idx.size();
    if (F < 1) throw std::runtime_error("no frames");
    std::vector<int32_t> revF(F);
    for (int64_t i = 0; i < F; i++) revF[i] = (int32_t) (F - 1 - i);

    // ---- graph B: frames -> F0, N ----
    std::vector<float> f0, nn;
    {
        graph_run r(m, t_cpu_fused || t_gpu_lstm ? 8192 : 16384 + (size_t) 40 * F, m->front_cpu);
        graph_ctx & g = r.g;
        ggml_context * ctx = g.ctx;
        ggml_tensor * dd = g.input_f32("d", {dC, T}, d_host.data());
        ggml_tensor * fi = g.input_i32("frame_tokens", idx);
        ggml_tensor * fr = g.input_i32("reverse_frames", revF);
        ggml_tensor * s = g.input_f32("style_p", {m->style_dim, 1}, sp);
        ggml_tensor * en = ggml_get_rows(ctx, dd, fi);                         // [hid + 128, F]
        ggml_tensor * x = bilstm(g, en, "predictor.shared", fr);
        ggml_tensor * F0 = x, * N = x;
        for (int j = 0; j < 3; j++) {
            F0 = adain_resblk(g, F0, s, "predictor.F0." + std::to_string(j), j == 1);
            N = adain_resblk(g, N, s, "predictor.N." + std::to_string(j), j == 1);
        }
        F0 = conv1d(ctx, F0, m->w("predictor.F0_proj.weight"), m->w("predictor.F0_proj.bias"), 1, 0, 0);
        N = conv1d(ctx, N, m->w("predictor.N_proj.weight"), m->w("predictor.N_proj.bias"), 1, 0, 0);
        r.output(F0);
        r.output(N);
        r.compute();
        f0 = get_f32(F0);
        nn = get_f32(N);
    }
    if (const char * dir = debug_env("KT_TINY_LOAD_F0N")) {
        // debugging: replace the predicted F0 / N curves with files (same length) to test the decoder alone
        const std::pair<const char *, std::vector<float> *> files[] = {{"F0", &f0}, {"N", &nn}};
        for (const auto & pr : files) {
            FILE * f = fopen((std::string(dir) + "/" + pr.first + ".f32").c_str(), "rb");
            if (!f) throw std::runtime_error("KT_TINY_LOAD_F0N: missing file");
            size_t got = fread(pr.second->data(), sizeof(float), pr.second->size(), f);
            fclose(f);
            if (got != pr.second->size()) throw std::runtime_error("KT_TINY_LOAD_F0N: wrong length");
        }
    }
    const double t2 = now_ms();
    dump("F0", f0);
    dump("N", nn);

    // ---- host: harmonic source (CUDA: only its cumulative phase), text features per frame ----
    int64_t n_har = 0;
    std::vector<float> har, p300;
    if (m->gpu_source) {
        const int64_t L = (int64_t) f0.size() * samples_per_f0(m);
        if (o.noise_mode == KT_TINY_NOISE_EXTERNAL && (!o.noise || o.noise_len != (size_t) (L * 9))) {
            throw std::runtime_error("external noise must have " + std::to_string(L * 9) + " values");
        }
        p300 = harmonic_phase(m, f0);
        n_har = L / m->hop + 1;
    } else {
        har = harmonic_source(m, f0, o, n_har);
        dump("har", har);
    }
    std::vector<float> asr((size_t) hid * F);
    for (int64_t f = 0; f < F; f++) memcpy(&asr[f * hid], &ten_host[(size_t) idx[f] * hid], hid * sizeof(float));
    std::vector<float> inv_env = istft_inv_envelope(m->n_fft, m->hop, n_har);
    const double t3 = now_ms();

    // ---- graph C: decoder + generator + iSTFT ----
    {
        t_gpu_adain = m->gpu_adain;
        t_window_cont = m->window_cont;
        t_cpu_fused = m->cpu_fused;
        graph_run r(m, 16384);
        graph_ctx & g = r.g;
        g.opts = &o;
        ggml_context * ctx = g.ctx;
        const int64_t L0 = (int64_t) f0.size();
        ggml_tensor * tasr = g.input_f32("asr", {hid, F}, asr.data());
        ggml_tensor * tf0 = g.input_f32("F0", {1, L0}, f0.data());
        ggml_tensor * tn = g.input_f32("N", {1, L0}, nn.data());
        ggml_tensor * thar;
        if (m->gpu_source) {
            // placeholder [n_fft + 2, frames] filled by the CUDA harmonic source; F0 and the phases are its extra sources
            // so that they stay allocated until it runs. A fill (not a copy of an unwritten tensor) so that nothing reads
            // uninitialised memory before the kernel overwrites it.
            ggml_tensor * tp = g.input_f32("p300", {9, L0}, p300.data());
            thar = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m->n_fft + 2, n_har), 0.0f);
            thar->src[1] = tf0;
            thar->src[2] = tp;
            ggml_set_name(thar, "kt_gpu_source");
            gpu_op op;
            op.kind = 2; op.f0 = tf0; op.p300 = tp; op.L0 = (int) L0; op.frames = n_har;
            g.gpu_ops[thar] = op;
        } else {
            thar = g.input_f32("har", {m->n_fft + 2, n_har}, har.data());
        }
        ggml_tensor * tenv = g.input_f32("inv_env", {(int64_t) inv_env.size()}, inv_env.data());
        ggml_tensor * s = g.input_f32("style_a", {m->style_dim, 1}, sa);
        ggml_tensor * F0c = conv1d_strided(ctx, tf0, m->w("decoder.F0_conv.weight"), m->w("decoder.F0_conv.bias"), 2, 1, 3);
        ggml_tensor * Nc = conv1d_strided(ctx, tn, m->w("decoder.N_conv.weight"), m->w("decoder.N_conv.bias"), 2, 1, 3);
        ggml_tensor * x = ggml_concat(ctx, ggml_concat(ctx, tasr, F0c, 0), Nc, 0);
        x = adain_resblk(g, x, s, "decoder.encode", false);
        ggml_tensor * ares = conv1d(ctx, tasr, m->w("decoder.asr_res.0.weight"), m->w("decoder.asr_res.0.bias"), 1, 0, 0);
        bool res = true;
        for (int j = 0; j < 4; j++) {
            if (res) x = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, x, ares, 0), F0c, 0), Nc, 0);
            x = adain_resblk(g, x, s, "decoder.decode." + std::to_string(j), j == 3);
            if (j == 3) res = false;
        }
        // generator
        const int nk = (int) m->rb_k.size(), nu = (int) m->up_r.size();
        for (int i = 0; i < nu; i++) {
            const std::string I = std::to_string(i);
            x = leaky(ctx, x, 0.1f);
            ggml_tensor * xs;
            if (i + 1 < nu) {
                int sf = 1;
                for (int k = i + 1; k < nu; k++) sf *= m->up_r[k];
                xs = conv1d_strided(ctx, thar, m->w("decoder.generator.noise_convs." + I + ".weight"),
                                    m->w("decoder.generator.noise_convs." + I + ".bias"), sf, (sf + 1) / 2, sf * 2);
                xs = adain_resblock1(g, xs, s, "decoder.generator.noise_res." + I, 7, {1, 3, 5});
            } else {
                xs = conv1d(ctx, thar, m->w("decoder.generator.noise_convs." + I + ".weight"),
                            m->w("decoder.generator.noise_convs." + I + ".bias"), 1, 0, 0);
                xs = adain_resblock1(g, xs, s, "decoder.generator.noise_res." + I, 11, {1, 3, 5});
            }
            const int u = m->up_r[i], k = m->up_k[i];
            x = conv_transpose1d(ctx, x, m->w("decoder.generator.ups." + I + ".weight"),
                                 m->w("decoder.generator.ups." + I + ".bias"), u, (k - u) / 2);
            if (i == nu - 1) {
                ggml_tensor * c1 = ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], x->nb[1]);
                x = ggml_concat(ctx, c1, x, 1);                                 // ReflectionPad1d((1, 0))
            }
            x = ggml_add(ctx, x, xs);
            ggml_tensor * acc = nullptr;
            for (int j = 0; j < nk; j++) {
                ggml_tensor * rb = adain_resblock1(g, x, s, "decoder.generator.resblocks." + std::to_string(i * nk + j),
                                                   m->rb_k[j], m->rb_d[j]);
                acc = acc ? ggml_add(ctx, acc, rb) : rb;
            }
            x = ggml_scale(ctx, acc, 1.0f / (float) nk);
        }
        x = leaky(ctx, x, 0.01f);
        x = conv1d(ctx, x, m->w("decoder.generator.conv_post.weight"), m->w("decoder.generator.conv_post.bias"), 1, 3, 3);
        // inverse STFT
        const int nb = m->n_fft / 2 + 1;
        ggml_tensor * mag = ggml_cont(ctx, ggml_view_2d(ctx, x, nb, x->ne[1], x->nb[1], 0));
        ggml_tensor * ph = ggml_cont(ctx, ggml_view_2d(ctx, x, nb, x->ne[1], x->nb[1], nb * sizeof(float)));
        ggml_tensor * spec = m->mag_softplus ? ggml_softplus(ctx, mag) : ggml_exp(ctx, mag);
        ph = ggml_sin(ctx, ph);
        ggml_tensor * re = ggml_mul(ctx, spec, ggml_cos(ctx, ph));
        ggml_tensor * im = ggml_mul(ctx, spec, ggml_sin(ctx, ph));
        ggml_tensor * ri = ggml_concat(ctx, re, im, 0);                          // [2 nb, frames]
        ggml_tensor * fr = mm(ctx, m->w("istft.basis"), ri);           // [n_fft, frames]
        ggml_tensor * y = ggml_col2im_1d(ctx, fr, m->hop, 1, m->n_fft / 2);      // [hop (frames - 1), 1]
        y = ggml_mul(ctx, ggml_reshape_1d(ctx, y, y->ne[0]), tenv);
        r.output(y);
        r.compute();
        audio = get_f32(y);
    }
    const double t4 = now_ms();
    m->rep["tokens"] = T;
    m->rep["frames"] = F;
    m->rep["samples"] = audio.size();
    m->rep["text_ms"] = t1 - t0;
    m->rep["prosody_ms"] = t2 - t1;
    m->rep["source_ms"] = t3 - t2;
    m->rep["decoder_ms"] = t4 - t3;
    m->rep["total_ms"] = t4 - t0;
    m->rep["audio_s"] = (double) audio.size() / m->sample_rate;
    m->rep["rtf"] = (t4 - t0) / 1000.0 / std::max(1e-9, (double) audio.size() / m->sample_rate);
    m->rep["weights_mb"] = (ggml_backend_buffer_get_size(m->buf_w) + (m->buf_w_cpu ? ggml_backend_buffer_get_size(m->buf_w_cpu) : 0)) / 1e6;
    m->report = m->rep.dump();
    dump("audio", audio);
}

const std::vector<float> & voice_pack(kt_tiny_model * m, const std::string & lang, const std::string & voice, std::string & key_out) {
    auto lv = m->lang_voices.find(lang);
    if (lv == m->lang_voices.end()) throw std::runtime_error("this model does not speak '" + lang + "'");
    std::string want = lower(voice);
    std::string key;
    if (want.empty()) {
        if (lv->second.empty()) throw std::runtime_error("no voices for " + lang);
        key = lv->second[0];
    } else {
        for (auto & k : lv->second) if (lower(k) == want) key = k;
        if (key.empty()) {
            for (auto & v : m->config["voices"][lang]) {
                if (lower(v.value("name", "")) == want || lower(v.value("description", "")) == want) key = v.value("key", "");
            }
        }
    }
    auto it = m->voices.find(lang + "/" + key);
    if (key.empty() || it == m->voices.end()) throw std::runtime_error("no voice '" + voice + "' for " + lang);
    key_out = key;
    return it->second;
}

void load(kt_tiny_model * m, const char * path, const kt_tiny_params * p) {
    profile_enabled();   // read the environment before a Vulkan load below may change it
    ggml_context * meta = nullptr;
    gguf_init_params gp = { true, &meta };
    gguf_context * gg = gguf_init_from_file(path, gp);
    if (!gg) throw std::runtime_error(std::string("cannot read GGUF ") + path);
    // released on every exit, including errors
    std::unique_ptr<gguf_context, void (*)(gguf_context *)> gg_guard(gg, gguf_free);
    std::unique_ptr<ggml_context, void (*)(ggml_context *)> meta_guard(meta, ggml_free);
    auto get_str = [&](const char * k) -> std::string {
        int64_t id = gguf_find_key(gg, k);
        return id < 0 ? std::string() : std::string(gguf_get_val_str(gg, id));
    };
    m->model_name = get_str("kitten.model");
    m->weight_type = get_str("kitten.weight_type");
    m->arch = json::parse(get_str("kitten.architecture"));
    m->config = json::parse(get_str("kitten.config"));
    json vocab = json::parse(get_str("kitten.vocab"));
    json vjson = json::parse(get_str("kitten.voices"));
    m->mag_softplus = get_str("kitten.magnitude_activation") == "softplus";
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        auto cps = utf8_decode(it.key());
        if (cps.size() == 1) m->vocab[cps[0]] = it.value().get<int>();
    }
    const json & a = m->arch;
    m->hid = a.at("hidden_dim");
    m->n_layer = a.at("n_layer");
    m->style_dim = a.value("style_dim", 128);
    m->te_kernel = a.value("text_encoder_kernel_size", 5);
    const json & dec = a.at("decoder");
    m->dec_type = dec.at("type");
    m->rb_k = dec.at("resblock_kernel_sizes").get<std::vector<int>>();
    m->rb_d = dec.at("resblock_dilation_sizes").get<std::vector<std::vector<int>>>();
    m->up_r = dec.at("upsample_rates").get<std::vector<int>>();
    m->up_k = dec.at("upsample_kernel_sizes").get<std::vector<int>>();
    m->up_ch = dec.at("upsample_initial_channel");
    m->n_fft = dec.at("gen_istft_n_fft");
    m->hop = dec.at("gen_istft_hop_size");
    const json & al = a.at("albert");
    m->a_hidden = al.at("hidden_size");
    m->a_heads = al.at("num_attention_heads");
    m->a_layers = al.at("num_hidden_layers");
    m->a_eps = al.value("layer_norm_eps", 1e-12f);
    m->sample_rate = m->config.value("sample_rate", 24000);
    if (m->style_dim != 128) throw std::runtime_error("style_dim must be 128");

    // backend
    std::string want = p && p->device ? lower(p->device) : "auto";
    auto is_metal = [](const std::string & n) { return n.rfind("mtl", 0) == 0 || n.rfind("metal", 0) == 0; };
    // ggml offers f32 matrix products and norms on Metal only when the GPU reports the Apple7 or Metal3 family. macOS
    // virtual machines (their "Apple Paravirtual device" reports Apple5 at most) do not, so the decoder's products would
    // move to the CPU and its overlapping window views abort there. "auto" then uses the CPU; "metal" is an error.
    auto metal_runs_decoder = [](ggml_backend_dev_t d) {
        ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * c = ggml_init(ip);
        if (!c) return false;
        ggml_tensor * w = ggml_new_tensor_2d(c, GGML_TYPE_F32, 64, 64);
        ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, 64, 8);
        ggml_tensor * y = ggml_mul_mat(c, w, x);
        ggml_mul_mat_set_prec(y, GGML_PREC_F32_STRICT);
        const bool ok = ggml_backend_dev_supports_op(d, y) && ggml_backend_dev_supports_op(d, ggml_norm(c, x, 1e-5f));
        ggml_free(c);
        return ok;
    };
    if (ggml_backend_dev_count() == 0) ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        std::string name = lower(ggml_backend_dev_name(d));
        auto type = ggml_backend_dev_type(d);
        bool gpu = type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
        if (want == "auto" ? gpu && (!is_metal(name) || metal_runs_decoder(d)) : want == "cpu" ? type == GGML_BACKEND_DEVICE_TYPE_CPU
            : want == "cuda" ? name.rfind("cuda", 0) == 0
            : want == "metal" ? is_metal(name)
            : want == "vulkan" ? name.rfind("vulkan", 0) == 0 : name == want) {
            dev = d;
            break;
        }
    }
    if (!dev && want == "auto") dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) {
        std::string msg = "no ggml device '" + want + "' in this build";
        if (want == "cuda") msg += " (an NVIDIA GPU needs the GPU build of this package, kitten-tts-cpp-cu12 on Linux or kitten-tts-cpp-cu13 on Windows; it is not on PyPI yet: build it as in packaging/README.md, section Building)";
        else if (want == "vulkan") msg += " (the pip packages do not include vulkan; build from source with -DGGML_VULKAN=ON)";
        else if (want == "metal") msg += " (this build has no Metal backend: build with -DGGML_METAL=ON on a Mac)";
        throw std::runtime_error(msg);
    }
    m->device_name = ggml_backend_dev_name(dev);
    const std::string dname = lower(m->device_name);
    if (is_metal(dname) && !metal_runs_decoder(dev)) {
        throw std::runtime_error("the Metal device '" + m->device_name + "' cannot run this model's matrix products in ggml "
                                 "(its GPU reports neither the Apple7 nor the Metal3 family, as in macOS virtual machines); "
                                 "use device=\"cpu\"");
    }
    // CUDA: every matrix product of this model asks for strict f32 (GGML_PREC_F32_STRICT: no TF32 tensor cores), set
    // per product, nothing process-wide. TF32 costs ~25 dB of agreement with PyTorch and ~2.5 dB of log-mel on pico.
    m->strict_f32 = !(p && p->allow_tf32);
    m->tf32_disabled = m->strict_f32 && dname.rfind("cuda", 0) == 0;
    bool vk_f16_off = false;
    if (dname.rfind("vulkan", 0) == 0 && !(p && p->allow_tf32)) {
        // ggml's Vulkan shaders compute f32 matrix products in fp16, or on cooperative-matrix (tensor core) paths with
        // reduced precision, when the device supports them: ~2.9 dB of log-mel on pico on an NVIDIA GPU. ggml-vulkan reads
        // these variables once, when it creates the device, so they are set only when Vulkan was chosen, just before
        // that (unless already set); they then apply to every user of that Vulkan device in the process.
        for (const char * var : {"GGML_VK_DISABLE_F16", "GGML_VK_DISABLE_COOPMAT", "GGML_VK_DISABLE_COOPMAT2"}) {
            if (getenv(var)) continue;
#ifdef _WIN32
            _putenv_s(var, "1");
#else
            setenv(var, "1", 0);
#endif
        }
        vk_f16_off = true;
    }
    m->backend = ggml_backend_dev_init(dev, nullptr);
    if (!m->backend) throw std::runtime_error("cannot initialise " + m->device_name);
    const int nt = p && p->n_threads > 0 ? p->n_threads : default_threads();
    m->n_threads = nt;
    const char * wc = debug_env("KT_TINY_WINDOW_COPY");
    // Metal's matrix kernels read src1 rows through their stride, so the overlapping window view needs no copy there
    // (decoder 1.9 times faster on an Apple M6, output identical); cuBLAS and Vulkan need the copy
    m->window_cont = wc ? atoi(wc) != 0 : ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU && !is_metal(dname);
    const char * fz = debug_env("KT_TINY_CPU_FUSED");
    m->cpu_fused = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && (!fz || atoi(fz) != 0);
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        m->cpu = m->backend;
    } else {
        m->cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    {
        ggml_backend_dev_t cd = ggml_backend_get_device(m->cpu);
        ggml_backend_reg_t reg = cd ? ggml_backend_dev_backend_reg(cd) : nullptr;
        auto fn = reg ? (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads") : nullptr;
        if (fn) fn(m->cpu, nt);
    }
    m->backends.push_back(m->backend);
    if (m->cpu != m->backend) m->backends.push_back(m->cpu);
    // Metal: text and prosody graphs on the CPU (their LSTM recurrences and small products are faster there and exact;
    // the CPU fast path's custom ops have no Metal kernels), the decoder on the GPU
    m->front_cpu = is_metal(dname);
    if (const char * fc = debug_env("KT_TINY_FRONT_CPU")) m->front_cpu = m->front_cpu && atoi(fc) != 0;

    // weights: everything except the voices goes to the device; voices and the source linear stay on the host
    m->dequantize = p ? p->dequantize != 0 : true;
    if (const char * dq = debug_env("KT_TINY_DEQUANTIZE")) m->dequantize = atoi(dq) != 0;
#ifdef KT_TINY_CUDA
    if (dname.rfind("cuda", 0) == 0) {
        const char * gk = debug_env("KT_TINY_GPU_KERNELS");
        if (!gk || atoi(gk) != 0) {
            m->cuda = kt_cuda_init(atoi(m->device_name.c_str() + 4));
            if (!m->cuda) throw std::runtime_error(std::string("CUDA kernels: ") + kt_cuda_last_error());
            m->gpu_lstm = true;
            m->gpu_source = true;
            const char * ga = debug_env("KT_TINY_GPU_ADAIN");
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            auto set_handler = reg ? (void (*)(void (*)(ggml_tensor *, void *, void *))) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_set_custom_handler") : nullptr;
            if (set_handler && (!ga || atoi(ga) != 0)) {
                set_handler(adain_cuda_handler);
                m->gpu_adain = true;
                if (const char * ig = debug_env("KT_TINY_IGEMM")) m->igemm_min_t = atoi(ig);
            }
        }
    }
#endif
    const int64_t n_t = gguf_get_n_tensors(gg);
    ggml_init_params ip = { ggml_tensor_overhead() * (size_t) (n_t + 8), nullptr, true };
    m->ctx_w = ggml_init(ip);
    if (m->front_cpu) m->ctx_w_cpu = ggml_init(ip);
    for (int64_t i = 0; i < n_t; i++) {
        std::string name = gguf_get_tensor_name(gg, i);
        if (name.rfind("voice.", 0) == 0) continue;
        ggml_tensor * src = ggml_get_tensor(meta, name.c_str());
        // front_cpu: only the decoder's weights go to the device; the text and prosody weights to the CPU buffer
        ggml_context * cw = m->front_cpu && name.rfind("decoder.", 0) != 0 ? m->ctx_w_cpu : m->ctx_w;
        ggml_tensor * t = m->dequantize && src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_I32
            ? ggml_new_tensor(cw, GGML_TYPE_F32, ggml_n_dims(src), src->ne) : ggml_dup_tensor(cw, src);
        ggml_set_name(t, name.c_str());
        m->W[name] = t;
    }
    const int nb = m->n_fft / 2 + 1;
    ggml_tensor * basis = ggml_new_tensor_2d(m->ctx_w, GGML_TYPE_F32, 2 * nb, m->n_fft);
    ggml_set_name(basis, "istft.basis");
    m->W["istft.basis"] = basis;
    m->buf_w = ggml_backend_alloc_ctx_tensors(m->ctx_w, m->backend);
    if (!m->buf_w) throw std::runtime_error("cannot allocate weights on " + m->device_name);
    ggml_backend_buffer_set_usage(m->buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (m->front_cpu) {
        m->buf_w_cpu = ggml_backend_alloc_ctx_tensors(m->ctx_w_cpu, m->cpu);
        if (!m->buf_w_cpu) throw std::runtime_error("cannot allocate weights on the CPU");
        ggml_backend_buffer_set_usage(m->buf_w_cpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    FILE * f = fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("cannot open ") + path);
    std::unique_ptr<FILE, int (*)(FILE *)> f_guard(f, fclose);
    const size_t data_off = gguf_get_data_offset(gg);
    std::vector<char> buf;
    for (int64_t i = 0; i < n_t; i++) {
        std::string name = gguf_get_tensor_name(gg, i);
        ggml_tensor * src = ggml_get_tensor(meta, name.c_str());
        const size_t sz = ggml_nbytes(src);
        buf.resize(sz);
#ifdef _WIN32
        _fseeki64(f, (long long) (data_off + gguf_get_tensor_offset(gg, i)), SEEK_SET);
#else
        fseeko(f, (off_t) (data_off + gguf_get_tensor_offset(gg, i)), SEEK_SET);
#endif
        if (fread(buf.data(), 1, sz, f) != sz) throw std::runtime_error("short read: " + name);
        if (name.rfind("voice.", 0) == 0) {
            // voice.<lang>.<key>
            std::string rest = name.substr(6);
            size_t dot = rest.find('.');
            std::string lang = rest.substr(0, dot), key = rest.substr(dot + 1);
            // style rows are read as float32 by index, so a voice must be float32 with at least one row of 256 values
            if (src->type != GGML_TYPE_F32 || src->ne[0] != 256 || ggml_nrows(src) < 1) {
                throw std::runtime_error("voice " + name + ": expected float32 rows of 256 values");
            }
            std::vector<float> v(sz / sizeof(float));
            memcpy(v.data(), buf.data(), sz);
            m->voices[lang + "/" + key] = std::move(v);
            continue;
        }
        ggml_tensor * dst = m->W[name];
        std::vector<float> f32;
        if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_I32) {
            f32.resize(ggml_nelements(src));
            ggml_get_type_traits(src->type)->to_float(buf.data(), f32.data(), ggml_nelements(src));
        }
        if (dst->type != src->type) {
            ggml_backend_tensor_set(dst, f32.data(), 0, f32.size() * sizeof(float));
        } else {
            ggml_backend_tensor_set(dst, buf.data(), 0, sz);
        }
#ifdef KT_TINY_CUDA
        if (m->gpu_lstm && name.size() > 5 && name.compare(name.size() - 5, 5, ".w_hh") == 0) {
            // the CUDA LSTM kernel keeps its own f32 copy of W_hh, transposed for coalesced reads
            const float * hw = src->type == GGML_TYPE_F32 ? (const float *) buf.data() : f32.data();
            if (kt_cuda_add_lstm(m->cuda, name.substr(0, name.size() - 5).c_str(), hw, (int) src->ne[0]) != 0) {
                throw std::runtime_error(std::string("CUDA LSTM weights: ") + kt_cuda_last_error());
            }
        }
#endif
        if (name == "decoder.generator.m_source.l_linear.weight") {
            m->src_lin_w.assign((const float *) buf.data(), (const float *) buf.data() + 9);
        } else if (name == "decoder.generator.m_source.l_linear.bias") {
            m->src_lin_b = *(const float *) buf.data();
        }
    }
    std::vector<float> B = istft_basis(m->n_fft);
    ggml_backend_tensor_set(basis, B.data(), 0, B.size() * sizeof(float));
    if (m->src_lin_w.size() != 9) throw std::runtime_error("missing m_source.l_linear");

    for (auto it = vjson.begin(); it != vjson.end(); ++it) {
        m->languages.push_back(it.key());
        std::vector<std::string> keys;
        for (auto & k : it.value()) if (m->voices.count(it.key() + "/" + k.get<std::string>())) keys.push_back(k.get<std::string>());
        m->lang_voices[it.key()] = keys;
    }
    m->n_vocab = std::min(m->w("bert.embeddings.word_embeddings.weight")->ne[1], m->w("text_encoder.embedding.weight")->ne[1]);

    json info;
    info["model"] = m->model_name;
    info["weight_type"] = m->weight_type;
    info["dequantized_to_f32"] = m->dequantize && m->weight_type != "f32";
    info["device"] = m->device_name;
    info["threads"] = m->n_threads;
    info["tf32_disabled"] = m->tf32_disabled;
    info["vulkan_f32_strict"] = vk_f16_off;
    info["gpu_kernels"] = m->gpu_lstm;
    info["sample_rate"] = m->sample_rate;
    info["languages"] = m->languages;
    info["voices"] = m->lang_voices;
    info["decoder"] = m->dec_type;
    info["weights_mb"] = (ggml_backend_buffer_get_size(m->buf_w) + (m->buf_w_cpu ? ggml_backend_buffer_get_size(m->buf_w_cpu) : 0)) / 1e6;
    info["front_on_cpu"] = m->front_cpu;
    info["config"] = m->config;
    info["vocab"] = vocab;
    m->info = info.dump();
}

} // namespace

extern "C" {

kt_tiny_params kt_tiny_default_params(void) {
    kt_tiny_params p;
    p.device = "auto";
    p.n_threads = 0;
    p.dequantize = 1;
    p.allow_tf32 = 0;
    return p;
}

kt_tiny_synth_options kt_tiny_default_options(void) {
    kt_tiny_synth_options o;
    o.speed = 1.0f;
    o.seed = 0;
    o.noise_mode = KT_TINY_NOISE_RANDOM;
    o.noise = nullptr;
    o.noise_len = 0;
    o.durations = nullptr;
    return o;
}

const char * kt_tiny_last_error(void) { return g_last_error.c_str(); }

kt_tiny_model * kt_tiny_load(const char * gguf_path, const kt_tiny_params * params) {
    auto * m = new kt_tiny_model();
    try {
        load(m, gguf_path, params);
        return m;
    } catch (const std::exception & e) {
        g_last_error = e.what();
        kt_tiny_free(m);
        return nullptr;
    }
}

void kt_tiny_free(kt_tiny_model * m) {
    if (!m) return;
    if (m->sched) ggml_backend_sched_free(m->sched);
    if (m->sched_cpu) ggml_backend_sched_free(m->sched_cpu);
#ifdef KT_TINY_CUDA
    kt_cuda_free(m->cuda);
#endif
    if (m->buf_w) ggml_backend_buffer_free(m->buf_w);
    if (m->ctx_w) ggml_free(m->ctx_w);
    if (m->buf_w_cpu) ggml_backend_buffer_free(m->buf_w_cpu);
    if (m->ctx_w_cpu) ggml_free(m->ctx_w_cpu);
    if (m->cpu && m->cpu != m->backend) ggml_backend_free(m->cpu);
    if (m->backend) ggml_backend_free(m->backend);
    delete m;
}

const char * kt_tiny_info_json(kt_tiny_model * m) { return m ? m->info.c_str() : ""; }
const char * kt_tiny_last_report(kt_tiny_model * m) { return m ? m->report.c_str() : ""; }
void kt_tiny_free_buffer(void * p) { free(p); }

int kt_tiny_voice_style(kt_tiny_model * m, const char * language, const char * voice, int n_tokens, float * style256) {
    try {
        if (!m || !style256) throw std::invalid_argument("model and style256 must not be NULL");
        std::string key;
        const std::vector<float> * pack = &voice_pack(m, language ? language : "en", voice ? voice : "", key);
        int rows = (int) (pack->size() / 256);
        int row = std::min(std::max(n_tokens - 1, 0), rows - 1);
        memcpy(style256, pack->data() + (size_t) row * 256, 256 * sizeof(float));
        return 0;
    } catch (const std::exception & e) {
        g_last_error = e.what();
        return -1;
    }
}

int kt_tiny_synthesize(kt_tiny_model * m, const int32_t * tokens, int n_tokens, const float * style256,
                       const kt_tiny_synth_options * options, float ** audio, size_t * n_audio, int32_t ** durations) {
    try {
        if (!m || !tokens || !style256 || !audio || !n_audio) {
            throw std::invalid_argument("model, tokens, style256, audio and n_audio must not be NULL");
        }
        if (n_tokens < 2 || n_tokens > 512) throw std::invalid_argument("n_tokens must be in [2, 512]");
        kt_tiny_synth_options o = options ? *options : kt_tiny_default_options();
        check_speed(o.speed);
        if (o.noise_mode < KT_TINY_NOISE_RANDOM || o.noise_mode > KT_TINY_NOISE_EXTERNAL) throw std::invalid_argument("unknown noise_mode");
        std::vector<int32_t> tok(tokens, tokens + n_tokens);
        std::vector<float> wav;
        std::vector<int32_t> dur;
        synthesize(m, tok, style256, o, wav, dur);
        float * a = (float *) malloc(std::max<size_t>(1, wav.size()) * sizeof(float));
        if (!a) throw std::bad_alloc();
        int32_t * dd = nullptr;
        if (durations) {
            dd = (int32_t *) malloc(std::max<size_t>(1, dur.size()) * sizeof(int32_t));
            if (!dd) { free(a); throw std::bad_alloc(); }
            if (!dur.empty()) memcpy(dd, dur.data(), dur.size() * sizeof(int32_t));
            *durations = dd;
        }
        if (!wav.empty()) memcpy(a, wav.data(), wav.size() * sizeof(float));
        *audio = a;
        *n_audio = wav.size();
        return 0;
    } catch (const std::exception & e) {
        g_last_error = e.what();
        return -1;
    }
}

int kt_tiny_generate_chunk(kt_tiny_model * m, const char * phonemes_utf8, const char * language, const char * voice,
                           float speed, int keep_start, uint64_t seed, float ** audio, size_t * n_audio) {
    try {
        if (!m || !audio || !n_audio) throw std::invalid_argument("model, audio and n_audio must not be NULL");
        check_speed(speed);
        const double t0 = now_ms();
        std::vector<int32_t> tok = {0};
        const int comma = m->vocab.count(',') ? m->vocab[','] : -1;
        for (uint32_t cp : utf8_decode(phonemes_utf8 ? phonemes_utf8 : "")) {
            auto it = m->vocab.find(cp);
            if (it != m->vocab.end()) tok.push_back(it->second);
            else if ((cp == '(' || cp == ')') && comma >= 0) tok.push_back(comma);
        }
        tok.push_back(10);
        tok.push_back(0);
        std::string key;
        const std::vector<float> * pack = &voice_pack(m, language ? language : "en", voice ? voice : "", key);
        int rows = (int) (pack->size() / 256);
        int row = std::min((int) tok.size() - 1, rows - 1);
        kt_tiny_synth_options o = kt_tiny_default_options();
        o.speed = speed;
        o.seed = seed;
        std::vector<float> w;
        std::vector<int32_t> dur;
        synthesize(m, tok, pack->data() + (size_t) row * 256, o, w, dur);
        for (auto & v : w) v = std::min(1.0f, std::max(-1.0f, v));
        int64_t total = 0;
        for (int32_t d : dur) total += d;
        const int64_t n = (int64_t) w.size();
        const int64_t spf = std::max<int64_t>(1, (int64_t) std::nearbyint((double) n / (double) std::max<int64_t>(total, 1)));
        const int64_t lead = keep_start ? 0 : std::max<int64_t>(0, dur.front() - 1) * spf;
        const int64_t trail = std::max<int64_t>(0, dur.back() - 1) * spf;
        int64_t b = 0, e = n;
        if (lead + trail + spf < n) { b = lead; e = n - trail; }
        std::vector<float> out(w.begin() + b, w.begin() + e);
        const int64_t fl = std::min<int64_t>(120, (int64_t) out.size() / 8);
        for (int64_t i = 0; i < fl; i++) {
            const float r = fl > 1 ? (float) i / (float) (fl - 1) : 0.0f;
            out[i] *= r;
            out[out.size() - 1 - i] *= r;
        }
        float * a = (float *) malloc(std::max<size_t>(1, out.size()) * sizeof(float));
        if (!a) throw std::bad_alloc();
        if (!out.empty()) memcpy(a, out.data(), out.size() * sizeof(float));
        *audio = a;
        *n_audio = out.size();
        m->rep["voice"] = key;
        m->rep["chunk_ms"] = now_ms() - t0;
        m->report = m->rep.dump();
        return 0;
    } catch (const std::exception & e) {
        g_last_error = e.what();
        return -1;
    }
}

} // extern "C"
