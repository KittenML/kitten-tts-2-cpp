// CUDA kernels for Kitten TTS 2 mini / nano / pico: the parts of the model that ggml has no GPU op for.
//   - the LSTM recurrence (both directions of a BiLSTM in one launch; one thread block per direction)
//   - the harmonic source (SineGen + SourceModuleHnNSF) and its forward STFT
// Without these, a GPU backend runs the LSTM as ~15 graph nodes per time step and the harmonic source on the CPU.
// Float operations follow the CPU code in kitten_tts_tiny.cpp one by one (no contraction into fused multiply-adds
// where the CPU code has none), so CPU and CUDA agree to rounding of the math library.
#include "kitten_tts_tiny_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct kt_cuda_ctx {
    int device = 0;
    cudaStream_t stream = nullptr;
    std::map<std::string, float *> lstm_wt;   // W_hh transposed to [2][H][4H], per LSTM (one-block kernel)
    std::map<std::string, float *> lstm_w;    // W_hh as ggml stores it, [2][4H][H] (multi-block kernel)
    unsigned * bar = nullptr;                 // step counters of the multi-block kernel, one per direction
    float * hbuf = nullptr; size_t hbuf_cap = 0;
    int coop = -1;                            // device supports cooperative launches (-1 = not asked yet)
    double * stft_tab = nullptr;              // [2][nb][n_fft] cos / sin table
    int stft_n = 0;
    float * src = nullptr;  size_t src_cap = 0;
    float * noise = nullptr; size_t noise_cap = 0;
    float * ast = nullptr;  size_t ast_cap = 0;   // AdaIN partial sums (doubles) and per-channel mean / scale / shift
    float * amab = nullptr; size_t amab_cap = 0;
    float * apad = nullptr; size_t apad_cap = 0;   // activated, zero padded signal [Tp][Cp] before the window copy
    std::string err;
};

namespace {

thread_local std::string g_cuda_err;

bool ok(kt_cuda_ctx * c, cudaError_t e, const char * what) {
    if (e == cudaSuccess) return true;
    const std::string msg = std::string(what) + ": " + cudaGetErrorString(e);
    if (c) c->err = msg;
    g_cuda_err = msg;
    return false;
}

bool grow(kt_cuda_ctx * c, float ** p, size_t * cap, size_t n) {
    if (*cap >= n) return true;
    if (*p) cudaFree(*p);
    *p = nullptr; *cap = 0;
    if (!ok(c, cudaMalloc(p, n * sizeof(float)), "cudaMalloc")) return false;
    *cap = n;
    return true;
}

// One block per direction. Thread r computes gate row r: sum_k W[r][k] h[k] (k in order, as the CPU op), then
// threads k < H update the cell. W is stored [H][4H] so that a warp reads 32 consecutive rows for each k.
__global__ void lstm_kernel(const float * __restrict__ gx, long long gx_s1, long long gx_s2,
                            const float * __restrict__ wt, float * __restrict__ out, long long out_s1, int H, int T) {
    extern __shared__ float sm[];
    const int G = 4 * H;
    float * h = sm;            // [H]
    float * c = sm + H;        // [H]
    float * gt = sm + 2 * H;   // [G]
    const int d = blockIdx.x;
    const float * W = wt + (size_t) d * H * G;
    for (int k = threadIdx.x; k < H; k += blockDim.x) { h[k] = 0.0f; c[k] = 0.0f; }
    __syncthreads();
    for (int st = 0; st < T; st++) {
        const float * g0 = gx + st * gx_s1 + d * gx_s2;
        for (int r = threadIdx.x; r < G; r += blockDim.x) {
            float acc = 0.0f;
            const float * w = W + r;
#pragma unroll 8
            for (int k = 0; k < H; k++) acc = __fadd_rn(acc, __fmul_rn(w[(size_t) k * G], h[k]));
            gt[r] = __fadd_rn(acc, g0[r]);
        }
        __syncthreads();
        const int t = d == 0 ? st : T - 1 - st;
        for (int k = threadIdx.x; k < H; k += blockDim.x) {
            const float ig = 1.0f / (1.0f + expf(-gt[k]));
            const float fg = 1.0f / (1.0f + expf(-gt[H + k]));
            const float gg = tanhf(gt[2 * H + k]);
            const float og = 1.0f / (1.0f + expf(-gt[3 * H + k]));
            const float cn = __fadd_rn(__fmul_rn(fg, c[k]), __fmul_rn(ig, gg));
            c[k] = cn;
            const float hn = __fmul_rn(og, tanhf(cn));
            h[k] = hn;
            out[t * out_s1 + (long long) d * H + k] = hn;
        }
        __syncthreads();
    }
}

// Multi-block version: each direction is split over NB blocks, block b owning hidden units [b*U, (b+1)*U) and their four
// gate rows, whose W_hh slice stays in shared memory for the whole sequence. After each step a block publishes its part
// of h in global memory and waits on a per-direction step counter, so all 2 * NB blocks must be resident at once. On the
// callback path that is guaranteed by a cooperative launch; as a graph node (an ordinary launch) it rests on the
// occupancy check in lstm_launch only, which assumes this process has the GPU to itself (with other processes on the
// same GPU, e.g. under MPS, the blocks could wait forever). Each gate's dot product is split over TPR threads and
// reduced with warp shuffles.
__global__ void lstm_mb_kernel(const float * __restrict__ gx, long long gx_s1, long long gx_s2, const float * __restrict__ w,
                               float * __restrict__ out, long long out_s1, int H, int T, int NB, int TPR,
                               unsigned * bar, float * hbuf) {
    extern __shared__ float sm[];
    const int U = H / NB, R = 4 * U;
    const int d = blockIdx.x / NB, b = blockIdx.x % NB;
    float * ws = sm;                 // [R][H]: row g*U + j = gate g of unit b*U + j
    float * hs = ws + (size_t) R * H; // [H]
    float * gs = hs + H;             // [R]
    const float * W = w + (size_t) d * 4 * H * H;
    for (int i = threadIdx.x; i < R * H; i += blockDim.x) {
        const int row = i / H, k = i % H, g = row / U, j = row % U;
        ws[i] = W[(size_t) (g * H + b * U + j) * H + k];
    }
    float c = 0.0f;                  // cell state of unit b*U + threadIdx.x (threads < U)
    const int row = threadIdx.x / TPR, lane = threadIdx.x % TPR;
    volatile unsigned * vb = bar + d;
    for (int st = 0; st < T; st++) {
        const float * hp = hbuf + (size_t) ((st & 1) * 2 + d) * H;
        for (int k = threadIdx.x; k < H; k += blockDim.x) hs[k] = st == 0 ? 0.0f : __ldcg(hp + k);
        __syncthreads();
        {
            float acc = 0.0f;
            if (row < R) {
                const float * wr = ws + (size_t) row * H;
                for (int k = lane; k < H; k += TPR) acc = fmaf(wr[k], hs[k], acc);
            }
            for (int o = TPR / 2; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o, TPR);   // every lane takes part
            if (row < R && lane == 0) {
                const int g = row / U, j = row % U;
                gs[row] = acc + gx[st * gx_s1 + d * gx_s2 + g * H + b * U + j];
            }
        }
        __syncthreads();
        const int t = d == 0 ? st : T - 1 - st;
        float * hn_out = hbuf + (size_t) (((st + 1) & 1) * 2 + d) * H;
        if (threadIdx.x < U) {
            const int j = threadIdx.x;
            const float ig = 1.0f / (1.0f + expf(-gs[j]));
            const float fg = 1.0f / (1.0f + expf(-gs[U + j]));
            const float gg = tanhf(gs[2 * U + j]);
            const float og = 1.0f / (1.0f + expf(-gs[3 * U + j]));
            c = fg * c + ig * gg;
            const float hn = og * tanhf(c);
            out[t * out_s1 + (long long) d * H + b * U + j] = hn;
            __stcg(hn_out + b * U + j, hn);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();
            atomicAdd(bar + d, 1u);
            const unsigned want = (unsigned) (st + 1) * (unsigned) NB;
            while (*vb < want) { }
            __threadfence();
        }
        __syncthreads();
    }
}

__device__ inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// standard normal for (seed, index): counter-based, so the result does not depend on the launch shape
__device__ inline float normal_at(uint64_t seed, uint64_t i) {
    const uint64_t z = splitmix64(seed * 0xD1B54A32D192ED03ull ^ splitmix64(i));
    const float u1 = ((float) (z >> 40) + 1.0f) * (1.0f / 16777216.0f);          // (0, 1]
    const float u2 = (float) ((z >> 16) & 0xFFFFFFull) * (1.0f / 16777216.0f);    // [0, 1)
    return sqrtf(-2.0f * logf(u1)) * cospif(2.0f * u2);
}

struct source_params {
    float lin_w[9];
    float lin_b;
    float scale;       // 1 / upp
    int upp;
    int L0;
    long long L;
    int mode;          // KT_CUDA_NOISE_*
    unsigned long long seed;
};

// SineGen + SourceModuleHnNSF per output sample (same float order as harmonic_source() on x86)
__global__ void source_kernel(const float * __restrict__ f0, const float * __restrict__ p300, const float * __restrict__ noise,
                              source_params p, float * __restrict__ src) {
    const long long n = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= p.L) return;
    float s = __fadd_rn(__fmul_rn(p.scale, __fadd_rn((float) n, 0.5f)), -0.5f);
    if (s < 0.0f) s = 0.0f;
    long long i0 = (long long) floorf(s);
    if (i0 > p.L0 - 1) i0 = p.L0 - 1;
    const float l1 = fminf(fmaxf(__fadd_rn(s, -(float) i0), 0.0f), 1.0f);
    const long long i1 = i0 + (i0 < p.L0 - 1 ? 1 : 0);
    const float l0 = __fadd_rn(1.0f, -l1);
    const float fv = f0[n / p.upp];
    const float uv = fv > 10.0f ? 1.0f : 0.0f;
    const float namp = __fadd_rn(__fmul_rn(uv, 0.003f), __fdiv_rn(__fmul_rn(__fadd_rn(1.0f, -uv), 0.1f), 3.0f));
    float acc = 0.0f;
#pragma unroll
    for (int h = 0; h < 9; h++) {
        const float ph = fmaf(p300[i0 * 9 + h], l0, __fmul_rn(p300[i1 * 9 + h], l1));
        float sw = __fmul_rn(sinf(ph), 0.1f);
        float z = 0.0f;
        if (p.mode == KT_CUDA_NOISE_RANDOM) z = normal_at(p.seed, (uint64_t) n * 9 + h);
        else if (p.mode == KT_CUDA_NOISE_EXTERNAL) z = noise[n * 9 + h];
        sw = __fadd_rn(__fmul_rn(sw, uv), __fmul_rn(namp, z));
        acc = __fadd_rn(acc, __fmul_rn(p.lin_w[h], sw));
    }
    src[n] = tanhf(__fadd_rn(acc, p.lin_b));
}

// forward STFT (periodic Hann, center, reflect padding), magnitude and angle per bin, in double like the CPU code
__global__ void stft_kernel(const float * __restrict__ src, long long L, const double * __restrict__ tab, int N, int hop,
                            long long frames, float * __restrict__ har) {
    const long long f = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= frames) return;
    const int nb = N / 2 + 1, pad = N / 2;
    double seg[64];
    for (int i = 0; i < N; i++) {
        long long j = f * hop + i - pad;
        if (j < 0) j = -j;
        if (j >= L) j = 2 * (L - 1) - j;
        if (j < 0) j = 0;
        if (j > L - 1) j = L - 1;
        const double w = 0.5 - 0.5 * cos(2.0 * M_PI * i / N);
        seg[i] = (double) src[j] * w;
    }
    const double * cs = tab, * sn = tab + nb * N;
    for (int k = 0; k < nb; k++) {
        double re = 0, im = 0;
        for (int i = 0; i < N; i++) { re += seg[i] * cs[k * N + i]; im -= seg[i] * sn[k * N + i]; }
        if (k == 0 || 2 * k == N) im = 0.0;
        const float fr = (float) re, fi = (float) im;
        har[f * (N + 2) + k] = hypotf(fr, fi);
        har[f * (N + 2) + nb + k] = atan2f(fi, fr);
    }
}

} // namespace

namespace { bool lstm_mb_shape(kt_cuda_ctx * c, int H, int & NB, int & TPR, int & threads, size_t & shm); }

extern "C" {

const char * kt_cuda_last_error(void) { return g_cuda_err.c_str(); }

kt_cuda_ctx * kt_cuda_init(int device) {
    auto * c = new kt_cuda_ctx();
    c->device = device;
    if (!ok(c, cudaSetDevice(device), "cudaSetDevice") ||
        !ok(c, cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking), "cudaStreamCreate")) {
        kt_cuda_free(c);
        return nullptr;
    }
    return c;
}

void kt_cuda_free(kt_cuda_ctx * c) {
    if (!c) return;
    cudaSetDevice(c->device);
    for (auto & e : c->lstm_wt) cudaFree(e.second);
    for (auto & e : c->lstm_w) cudaFree(e.second);
    if (c->bar) cudaFree(c->bar);
    if (c->hbuf) cudaFree(c->hbuf);
    if (c->stft_tab) cudaFree(c->stft_tab);
    if (c->src) cudaFree(c->src);
    if (c->noise) cudaFree(c->noise);
    if (c->ast) cudaFree(c->ast);
    if (c->amab) cudaFree(c->amab);
    if (c->apad) cudaFree(c->apad);
    if (c->stream) cudaStreamDestroy(c->stream);
    delete c;
}

int kt_cuda_add_lstm(kt_cuda_ctx * c, const char * key, const float * whh, int H) {
    // whh as ggml stores it: [2][4H][H] (ne0 = H) -> [2][H][4H]
    const int G = 4 * H;
    std::vector<float> wt((size_t) 2 * H * G);
    for (int d = 0; d < 2; d++)
        for (int r = 0; r < G; r++)
            for (int k = 0; k < H; k++) wt[(size_t) d * H * G + (size_t) k * G + r] = whh[(size_t) d * G * H + (size_t) r * H + k];
    float * p = nullptr;
    if (!ok(c, cudaSetDevice(c->device), "cudaSetDevice") || !ok(c, cudaMalloc(&p, wt.size() * sizeof(float)), "cudaMalloc") ||
        !ok(c, cudaMemcpy(p, wt.data(), wt.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy")) {
        if (p) cudaFree(p);
        return -1;
    }
    c->lstm_wt[key] = p;
    float * q = nullptr;
    if (!ok(c, cudaMalloc(&q, wt.size() * sizeof(float)), "cudaMalloc") ||
        !ok(c, cudaMemcpy(q, whh, wt.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy")) {
        if (q) cudaFree(q);
        return -1;
    }
    c->lstm_w[key] = q;
    int NB = 0, TPR = 1, threads = 0;
    size_t shm = 0;
    if (lstm_mb_shape(c, H, NB, TPR, threads, shm)) {
        if (!grow(c, &c->hbuf, &c->hbuf_cap, (size_t) 4 * H)) return -1;
        if (!c->bar && !ok(c, cudaMalloc(&c->bar, 2 * sizeof(unsigned)), "cudaMalloc")) return -1;
        int cur = 0;
        cudaFuncAttributes fa;
        if (cudaFuncGetAttributes(&fa, lstm_mb_kernel) == cudaSuccess) cur = fa.maxDynamicSharedSizeBytes;
        if ((int) shm > cur && !ok(c, cudaFuncSetAttribute(lstm_mb_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shm), "smem attr")) return -1;
    }
    return 0;
}

int kt_cuda_has_lstm(kt_cuda_ctx * c, const char * key) { return c && c->lstm_wt.count(key) ? 1 : 0; }

} // extern "C"

namespace {

// multi-block layout for hidden size H: NB blocks per direction such that the W_hh slice fits in shared memory
bool lstm_mb_shape(kt_cuda_ctx * c, int H, int & NB, int & TPR, int & threads, size_t & shm) {
    int max_shm = 0;
    cudaDeviceGetAttribute(&max_shm, cudaDevAttrMaxSharedMemoryPerBlockOptin, c->device);
    NB = 0;
    for (int nb : {8, 16, 32}) {
        if (H % nb) continue;
        const int U = H / nb;
        const size_t need = ((size_t) 4 * U * H + H + 4 * U) * sizeof(float);
        if (need <= (size_t) max_shm && 4 * U <= 256) { NB = nb; shm = need; break; }
    }
    if (!NB) return false;
    const int R = 4 * (H / NB);
    TPR = 1;
    while (TPR < 32 && R * TPR * 2 <= 256) TPR *= 2;
    threads = std::max(32, ((R * TPR + 31) / 32) * 32);
    return true;
}

// launch on `stream`; sync = wait for it (callback path). Without sync (a graph node), the multi-block kernel is used only
// when all its blocks fit on the GPU at once (occupancy check), since a CUDA graph cannot hold a cooperative launch here.
int lstm_launch(kt_cuda_ctx * c, cudaStream_t stream, bool sync, const char * key, const float * gx, long long gx_s1,
                long long gx_s2, float * out, long long out_s1, int H, int T) {
    auto it = c->lstm_wt.find(key);
    if (it == c->lstm_wt.end()) { g_cuda_err = std::string("no LSTM weights ") + key; return -1; }
    if (!ok(c, cudaSetDevice(c->device), "cudaSetDevice")) return -1;
    if (c->coop < 0) {
        int v = 0;
        cudaDeviceGetAttribute(&v, cudaDevAttrCooperativeLaunch, c->device);
        c->coop = v;
    }
    int NB = 0, TPR = 1, threads = 0;
    size_t shm_mb = 0;
    bool one_block = false;
#ifdef KT_TINY_DEBUG_ENV
    one_block = getenv("KT_TINY_LSTM_ONE_BLOCK") != nullptr;   // A/B switch, debugging builds only
#endif
    if (!one_block && lstm_mb_shape(c, H, NB, TPR, threads, shm_mb) && c->hbuf && c->hbuf_cap >= (size_t) 4 * H && c->bar) {
        const float * w = c->lstm_w[key];
        float * hb = c->hbuf;
        unsigned * bar = c->bar;
        if (!ok(c, cudaMemsetAsync(c->bar, 0, 2 * sizeof(unsigned), stream), "memset")) return -1;
        if (sync && c->coop > 0) {
            void * args[] = {(void *) &gx, &gx_s1, &gx_s2, (void *) &w, (void *) &out, &out_s1, &H, &T, &NB, &TPR, (void *) &bar, (void *) &hb};
            const cudaError_t e = cudaLaunchCooperativeKernel((const void *) lstm_mb_kernel, dim3(2 * NB), dim3(threads), args, shm_mb, stream);
            if (e == cudaSuccess) return ok(c, cudaStreamSynchronize(stream), "lstm_mb_kernel sync") ? 0 : -1;
            cudaGetLastError();   // e.g. too many blocks to be resident at once: one block per direction below
        } else if (!sync) {
            int per_sm = 0, sms = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, lstm_mb_kernel, threads, shm_mb);
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, c->device);
            if ((long long) per_sm * sms >= 2LL * NB) {
                lstm_mb_kernel<<<2 * NB, threads, shm_mb, stream>>>(gx, gx_s1, gx_s2, w, out, out_s1, H, T, NB, TPR, bar, hb);
                return ok(c, cudaGetLastError(), "lstm_mb_kernel") ? 0 : -1;
            }
        }
    }
    const int G = 4 * H;
    const int nt = G <= 1024 ? ((G + 31) / 32) * 32 : 1024;
    const size_t shm = (size_t) (2 * H + G) * sizeof(float);
    lstm_kernel<<<2, nt, shm, stream>>>(gx, gx_s1, gx_s2, it->second, out, out_s1, H, T);
    if (!ok(c, cudaGetLastError(), "lstm_kernel")) return -1;
    return sync ? (ok(c, cudaStreamSynchronize(stream), "lstm_kernel sync") ? 0 : -1) : 0;
}

} // namespace

extern "C" {

int kt_cuda_lstm(kt_cuda_ctx * c, const char * key, const float * gx, long long gx_s1, long long gx_s2,
                 float * out, long long out_s1, int H, int T) {
    return lstm_launch(c, c->stream, true, key, gx, gx_s1, gx_s2, out, out_s1, H, T);
}

int kt_cuda_lstm_async(kt_cuda_ctx * c, void * stream, const char * key, const float * gx, long long gx_s1, long long gx_s2,
                       float * out, long long out_s1, int H, int T) {
    return lstm_launch(c, (cudaStream_t) stream, false, key, gx, gx_s1, gx_s2, out, out_s1, H, T);
}

int kt_cuda_harmonic(kt_cuda_ctx * c, const float * f0, const float * p300, int L0, int upp, const float * lin_w, float lin_b,
                     int noise_mode, unsigned long long seed, const float * noise_host, size_t noise_len,
                     int n_fft, int hop, float * har, long long frames) {
    if (!ok(c, cudaSetDevice(c->device), "cudaSetDevice")) return -1;
    if (n_fft > 64) { g_cuda_err = "n_fft > 64"; return -1; }
    const long long L = (long long) L0 * upp;
    if (!grow(c, &c->src, &c->src_cap, (size_t) L)) return -1;
    const float * nz = nullptr;
    if (noise_mode == KT_CUDA_NOISE_EXTERNAL) {
        if (!noise_host || noise_len != (size_t) L * 9) { g_cuda_err = "external noise has the wrong length"; return -1; }
        if (!grow(c, &c->noise, &c->noise_cap, noise_len)) return -1;
        if (!ok(c, cudaMemcpyAsync(c->noise, noise_host, noise_len * sizeof(float), cudaMemcpyHostToDevice, c->stream), "noise copy")) return -1;
        nz = c->noise;
    }
    if (c->stft_n != n_fft) {
        const int nb = n_fft / 2 + 1;
        std::vector<double> tab((size_t) 2 * nb * n_fft);
        for (int k = 0; k < nb; k++) for (int i = 0; i < n_fft; i++) {
            tab[(size_t) k * n_fft + i] = std::cos(2.0 * M_PI * k * i / n_fft);
            tab[(size_t) nb * n_fft + (size_t) k * n_fft + i] = std::sin(2.0 * M_PI * k * i / n_fft);
        }
        if (c->stft_tab) cudaFree(c->stft_tab);
        c->stft_tab = nullptr;
        if (!ok(c, cudaMalloc(&c->stft_tab, tab.size() * sizeof(double)), "cudaMalloc") ||
            !ok(c, cudaMemcpy(c->stft_tab, tab.data(), tab.size() * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy")) return -1;
        c->stft_n = n_fft;
    }
    source_params p;
    memcpy(p.lin_w, lin_w, sizeof(p.lin_w));
    p.lin_b = lin_b;
    p.scale = (float) (1.0 / (double) upp);
    p.upp = upp; p.L0 = L0; p.L = L; p.mode = noise_mode; p.seed = seed;
    source_kernel<<<(unsigned) ((L + 255) / 256), 256, 0, c->stream>>>(f0, p300, nz, p, c->src);
    if (!ok(c, cudaGetLastError(), "source_kernel")) return -1;
    stft_kernel<<<(unsigned) ((frames + 127) / 128), 128, 0, c->stream>>>(c->src, L, c->stft_tab, n_fft, hop, frames, har);
    if (!ok(c, cudaGetLastError(), "stft_kernel")) return -1;
    return ok(c, cudaStreamSynchronize(c->stream), "harmonic sync") ? 0 : -1;
}

} // extern "C"

// ---- fused AdaIN + activation + zero padding + convolution window (im2col) ----
// x is [T][C] (channel fastest, row stride s1 floats). The convolution input is act(norm(x)) zero padded by pl rows; its
// window for output step t is the K rows t, t + d, ... t + (K - 1) d of the padded signal, each Cp wide (channels >= C are 0).
// Same float operations as kt_adain_apply_op on the CPU (instance norm sums in double, no fused multiply-add).
constexpr int KT_ASLICE = 256;

__global__ void adain_stats_kernel(const float * __restrict__ x, long long s1, int C, int T, double * __restrict__ part) {
    const int c = blockIdx.x * 32 + threadIdx.x;
    const int per = (T + KT_ASLICE - 1) / KT_ASLICE;
    const int t0 = blockIdx.y * per, t1 = min(T, t0 + per);
    double s = 0, q = 0;
    if (c < C) {
        for (int t = t0 + threadIdx.y; t < t1; t += 8) { const double v = x[(long long) t * s1 + c]; s += v; q += v * v; }
    }
    __shared__ double ss[8][32], sq[8][32];
    ss[threadIdx.y][threadIdx.x] = s; sq[threadIdx.y][threadIdx.x] = q;
    __syncthreads();
    if (threadIdx.y == 0 && c < C) {
        for (int i = 1; i < 8; i++) { s += ss[i][threadIdx.x]; q += sq[i][threadIdx.x]; }
        part[((size_t) blockIdx.y * 2) * C + c] = s;
        part[((size_t) blockIdx.y * 2 + 1) * C + c] = q;
    }
}

__global__ void adain_final_kernel(const double * __restrict__ part, const float * __restrict__ h, const float * __restrict__ nw,
                                   const float * __restrict__ nb, int C, int T, float eps, float * __restrict__ mab) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    double su = 0, sq = 0;
    for (int sl = 0; sl < KT_ASLICE; sl++) { su += part[((size_t) sl * 2) * C + c]; sq += part[((size_t) sl * 2 + 1) * C + c]; }
    const double mu = su / T, var = fmax(0.0, sq / T - mu * mu);
    const float rstd = (float) (1.0 / sqrt(var + (double) eps));
    float sc = __fmul_rn(rstd, __fadd_rn(1.0f, h[c]));
    float sh = h[C + c];
    if (nw) {
        sh = __fadd_rn(__fmul_rn(__fadd_rn(1.0f, h[c]), nb[c]), sh);
        sc = __fmul_rn(sc, nw[c]);
    }
    mab[c] = (float) mu; mab[C + c] = sc; mab[2 * C + c] = sh;
}

__global__ void adain_window_kernel(const float * __restrict__ x, long long s1, int C, int T, const float * __restrict__ mab,
                                    const float * __restrict__ alpha, const float * __restrict__ ialpha, int act, float slope,
                                    int Cp, int K, int d, int pl, float * __restrict__ out) {
    const long long t = blockIdx.x;
    const int KC = K * Cp;
    float * o = out + t * KC;
    for (int kc = threadIdx.x; kc < KC; kc += blockDim.x) {
        const int k = kc / Cp, c = kc - k * Cp;
        const long long ts = t + (long long) k * d - pl;
        float y = 0.0f;
        if (c < C && ts >= 0 && ts < T) {
            y = __fadd_rn(__fmul_rn(__fsub_rn(x[ts * s1 + c], mab[c]), mab[C + c]), mab[2 * C + c]);
            if (act == 1) y = y > 0.0f ? y : __fmul_rn(y, slope);
            else if (act == 2) { const float sn = sinf(__fmul_rn(alpha[c], y)); y = __fadd_rn(y, __fmul_rn(ialpha[c], __fmul_rn(sn, sn))); }
        }
        o[kc] = y;
    }
}

// activated, zero padded signal [Tp][Cp] (row i is time step i - pl)
__global__ void adain_act_kernel(const float * __restrict__ x, long long s1, int C, int T, const float * __restrict__ mab,
                                 const float * __restrict__ alpha, const float * __restrict__ ialpha, int act, float slope,
                                 int Cp, int pl, long long n, float * __restrict__ pad) {
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x) {
        const long long r = i / Cp;
        const int c = (int) (i - r * Cp);
        const long long ts = r - pl;
        float y = 0.0f;
        if (c < C && ts >= 0 && ts < T) {
            y = __fadd_rn(__fmul_rn(__fsub_rn(x[ts * s1 + c], mab[c]), mab[C + c]), mab[2 * C + c]);
            if (act == 1) y = y > 0.0f ? y : __fmul_rn(y, slope);
            else if (act == 2) { const float sn = sinf(__fmul_rn(alpha[c], y)); y = __fadd_rn(y, __fmul_rn(ialpha[c], __fmul_rn(sn, sn))); }
        }
        pad[i] = y;
    }
}

// window t = rows t, t + d, ..., t + (K - 1) d of the padded signal, 16 bytes per thread (Cp % 4 == 0)
__global__ void window_copy_kernel(const float4 * __restrict__ pad, int Cp4, int K, int d, float4 * __restrict__ out) {
    const long long t = blockIdx.x;
    const int n4 = K * Cp4;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const int k = i / Cp4, c4 = i - k * Cp4;
        out[t * n4 + i] = pad[(t + (long long) k * d) * Cp4 + c4];
    }
}

extern "C" {

// Scratch for the AdaIN kernels of one graph (allocated before the graph runs: no allocation while it is captured).
int kt_cuda_adain_reserve(kt_cuda_ctx * c, int C, long long pad_floats) {
    if (!ok(c, cudaSetDevice(c->device), "cudaSetDevice")) return -1;
    if (!grow(c, &c->ast, &c->ast_cap, (size_t) KT_ASLICE * 4 * C) || !grow(c, &c->amab, &c->amab_cap, (size_t) 3 * C) ||
        !grow(c, &c->apad, &c->apad_cap, (size_t) pad_floats)) return -1;
    return 0;
}

// Launches on the given stream (the graph's) and does not wait.
int kt_cuda_adain_window(kt_cuda_ctx * c, void * stream, const float * x, long long xs1, int C, int T, const float * h, const float * nw,
                         const float * nb, const float * alpha, const float * ialpha, int act, float slope, float eps,
                         int Cp, int K, int d, int pl, int Tout, float * out) {
    cudaStream_t st = (cudaStream_t) stream;
    const long long Tp = (long long) Tout + (long long) d * (K - 1);   // T + pl + pr
    const long long n = Tp * Cp;
    if ((size_t) KT_ASLICE * 4 * C > c->ast_cap || (size_t) 3 * C > c->amab_cap || (size_t) n > c->apad_cap) {
        g_cuda_err = "AdaIN scratch not reserved";
        return -1;
    }
    adain_stats_kernel<<<dim3((C + 31) / 32, KT_ASLICE), dim3(32, 8), 0, st>>>(x, xs1, C, T, (double *) c->ast);
    adain_final_kernel<<<(C + 127) / 128, 128, 0, st>>>((const double *) c->ast, h, nw, nb, C, T, eps, c->amab);
    const int KC = K * Cp;
    if (Cp % 4 == 0) {
        const unsigned nb2 = (unsigned) std::min<long long>((n + 255) / 256, 65535LL * 4);
        adain_act_kernel<<<nb2, 256, 0, st>>>(x, xs1, C, T, c->amab, alpha, ialpha, act, slope, Cp, pl, n, c->apad);
        window_copy_kernel<<<(unsigned) Tout, std::min(256, ((KC / 4 + 31) / 32) * 32), 0, st>>>((const float4 *) c->apad, Cp / 4, K, d, (float4 *) out);
    } else {
        const int threads = std::min(256, ((KC + 31) / 32) * 32);
        adain_window_kernel<<<(unsigned) Tout, threads, 0, st>>>(x, xs1, C, T, c->amab, alpha, ialpha, act, slope, Cp, K, d, pl, out);
    }
    if (!ok(c, cudaGetLastError(), "adain kernels")) return -1;
    return 0;
}

} // extern "C"

// ---- implicit-GEMM 1-D convolution on the activated, padded signal (no window copy) ----
// out[t][co] = bias[co] + sum_{k < K, c < Cp} w[co][k * Cp + c] * pad[(t + k * d) * Cp + c] (+ resid). pad is [Tp][Cp].
// f32 FMA, no TF32. Tile 128 x 128 per block (256 threads), 8 x 8 per thread, k-slices of BK (Cp % BK == 0).
// The next slice is loaded into registers while the current one is multiplied from shared memory. PAD avoids bank conflicts.
constexpr int KT_IBM = 128, KT_IBN = 128;

template <int BK, int PAD>
__global__ void __launch_bounds__(256, 2) igemm_conv_kernel(const float * __restrict__ pad, const float * __restrict__ w,
                                                             const float * __restrict__ bias, const float * __restrict__ resid, int Cp, int K,
                                                             int d, int Cout, int Tout, float * __restrict__ out) {
    constexpr int NL = BK / 8;                               // float4 loads per thread and tile (256 threads x 2 per row)
    __shared__ __align__(16) float As[2][BK][KT_IBM + PAD];   // [k][co]
    __shared__ __align__(16) float Bs[2][BK][KT_IBN + PAD];   // [k][t]
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;                 // tx: group of 4 output channels, ty: group of 4 time steps
    const int co0 = blockIdx.y * KT_IBM, t0 = blockIdx.x * KT_IBN;
    const int KC = Cp * K, nk = KC / BK;
    const int lr = tid >> 1, lh = (tid & 1) * 4;            // load row (0..127) and float offset (0 or 4) inside each 8 floats
    const bool a_ok = co0 + lr < Cout, b_ok = t0 + lr < Tout;
    const float * wrow = w + (size_t) (co0 + lr) * KC + lh;
    const float * prow = pad + (size_t) (t0 + lr) * Cp + lh;

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;

    float4 ra[NL], rb[NL];
#pragma unroll
    for (int q = 0; q < NL; q++) {
        ra[q] = make_float4(0.f, 0.f, 0.f, 0.f);
        rb[q] = ra[q];
        if (a_ok) ra[q] = __ldg((const float4 *) (wrow + q * 8));
        if (b_ok) rb[q] = __ldg((const float4 *) (prow + q * 8));
    }
    int tap = 0, c0 = 0;                                     // slice 0 is tap 0, channels 0..BK-1
#pragma unroll
    for (int q = 0; q < NL; q++) {
        As[0][q * 8 + lh + 0][lr] = ra[q].x; As[0][q * 8 + lh + 1][lr] = ra[q].y; As[0][q * 8 + lh + 2][lr] = ra[q].z; As[0][q * 8 + lh + 3][lr] = ra[q].w;
        Bs[0][q * 8 + lh + 0][lr] = rb[q].x; Bs[0][q * 8 + lh + 1][lr] = rb[q].y; Bs[0][q * 8 + lh + 2][lr] = rb[q].z; Bs[0][q * 8 + lh + 3][lr] = rb[q].w;
    }
    __syncthreads();

    for (int it = 0; it < nk; it++) {
        const int buf = it & 1;
        const bool more = it + 1 < nk;
        if (more) {
            c0 += BK;
            if (c0 == Cp) { c0 = 0; tap++; }
            const float * wn = wrow + (size_t) (it + 1) * BK;
            const float * pn = prow + (size_t) tap * d * Cp + c0;
#pragma unroll
            for (int q = 0; q < NL; q++) {
                if (a_ok) ra[q] = __ldg((const float4 *) (wn + q * 8));
                if (b_ok) rb[q] = __ldg((const float4 *) (pn + q * 8));
            }
        }
#pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            const float4 a0 = *(const float4 *) &As[buf][kk][tx * 4], a1 = *(const float4 *) &As[buf][kk][64 + tx * 4];
            const float4 b0 = *(const float4 *) &Bs[buf][kk][ty * 4], b1 = *(const float4 *) &Bs[buf][kk][64 + ty * 4];
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] = fmaf(a[j], b[i], acc[i][j]);
        }
        if (more) {
            const int nb = buf ^ 1;
#pragma unroll
            for (int q = 0; q < NL; q++) {
                As[nb][q * 8 + lh + 0][lr] = ra[q].x; As[nb][q * 8 + lh + 1][lr] = ra[q].y; As[nb][q * 8 + lh + 2][lr] = ra[q].z; As[nb][q * 8 + lh + 3][lr] = ra[q].w;
                Bs[nb][q * 8 + lh + 0][lr] = rb[q].x; Bs[nb][q * 8 + lh + 1][lr] = rb[q].y; Bs[nb][q * 8 + lh + 2][lr] = rb[q].z; Bs[nb][q * 8 + lh + 3][lr] = rb[q].w;
            }
        }
        __syncthreads();
    }

    // acc[i][j]: time step t0 + (i < 4 ? ty * 4 + i : 60 + ty * 4 + i), channel co0 + (j < 4 ? tx * 4 + j : 60 + tx * 4 + j)
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int t = t0 + (i < 4 ? ty * 4 + i : 60 + ty * 4 + i);
        if (t >= Tout) continue;
#pragma unroll
        for (int g = 0; g < 2; g++) {
            const int co = co0 + g * 64 + tx * 4;
            if (co >= Cout) continue;
            float4 v = make_float4(acc[i][g * 4], acc[i][g * 4 + 1], acc[i][g * 4 + 2], acc[i][g * 4 + 3]);
            if (bias) {
                const float4 bb = __ldg((const float4 *) (bias + co));
                v.x = __fadd_rn(v.x, bb.x); v.y = __fadd_rn(v.y, bb.y); v.z = __fadd_rn(v.z, bb.z); v.w = __fadd_rn(v.w, bb.w);
            }
            if (resid) {   // residual connection: (conv + bias) + resid, the same two roundings as two ggml_add nodes
                const float4 rr = __ldg((const float4 *) (resid + (size_t) t * Cout + co));
                v.x = __fadd_rn(v.x, rr.x); v.y = __fadd_rn(v.y, rr.y); v.z = __fadd_rn(v.z, rr.z); v.w = __fadd_rn(v.w, rr.w);
            }
            *(float4 *) (out + (size_t) t * Cout + co) = v;
        }
    }
}

extern "C" {

// AdaIN + activation + zero padding (same kernels as kt_cuda_adain_window) then the implicit-GEMM convolution; no window copy.
// w: [Cout][Cp * K] f32 (ggml [Cp * K, Cout]); bias: [Cout] or NULL; resid: [Tout][Cout] added to the result or NULL;
// out: [Tout][Cout]. Needs Cp % 16 == 0 and Cout % 4 == 0.
int kt_cuda_adain_conv(kt_cuda_ctx * c, void * stream, const float * x, long long xs1, int C, int T, const float * h, const float * nw,
                       const float * nb, const float * alpha, const float * ialpha, int act, float slope, float eps,
                       int Cp, int K, int d, int pl, int Tout, const float * w, const float * bias, const float * resid, int Cout, float * out) {
    cudaStream_t st = (cudaStream_t) stream;
    if (Cp % 16 != 0 || Cout % 4 != 0) { g_cuda_err = "implicit-GEMM convolution needs Cp % 16 == 0 and Cout % 4 == 0"; return -1; }
    const long long Tp = (long long) Tout + (long long) d * (K - 1);
    const long long n = Tp * Cp;
    if ((size_t) KT_ASLICE * 4 * C > c->ast_cap || (size_t) 3 * C > c->amab_cap || (size_t) n > c->apad_cap) {
        g_cuda_err = "AdaIN scratch not reserved";
        return -1;
    }
    adain_stats_kernel<<<dim3((C + 31) / 32, KT_ASLICE), dim3(32, 8), 0, st>>>(x, xs1, C, T, (double *) c->ast);
    adain_final_kernel<<<(C + 127) / 128, 128, 0, st>>>((const double *) c->ast, h, nw, nb, C, T, eps, c->amab);
    const unsigned nb2 = (unsigned) std::min<long long>((n + 255) / 256, 65535LL * 4);
    adain_act_kernel<<<nb2, 256, 0, st>>>(x, xs1, C, T, c->amab, alpha, ialpha, act, slope, Cp, pl, n, c->apad);
    const dim3 grid((unsigned) ((Tout + KT_IBN - 1) / KT_IBN), (unsigned) ((Cout + KT_IBM - 1) / KT_IBM));
    igemm_conv_kernel<16, 4><<<grid, 256, 0, st>>>(c->apad, w, bias, resid, Cp, K, d, Cout, Tout, out);
    if (!ok(c, cudaGetLastError(), "adain conv kernels")) return -1;
    return 0;
}

} // extern "C"
