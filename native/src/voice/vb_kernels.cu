// L11 native voice build: op library (see vb_kernels.cuh for the contract of each op).
#include "vb_kernels.cuh"

#include <mutex>
#include <stdexcept>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace kt2n::voice {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

namespace {

inline unsigned blocks_for(long long n, int threads) { return (unsigned)((n + threads - 1) / threads); }
unsigned grid_cap(long long n);

__device__ __forceinline__ float act_apply(float v, Act a, float leaky) {
  switch (a) {
    case Act::kRelu: return v > 0.f ? v : 0.f;
    case Act::kLeaky: return v >= 0.f ? v : v * leaky;
    case Act::kGelu: return 0.5f * v * (1.f + erff(v * 0.70710678118654752440f));
    case Act::kSigmoid: return 1.f / (1.f + expf(-v));
    default: return v;
  }
}

struct EpiDev {
  const float* bias;
  const float* scale;
  const float* shift;
  const float* residual;
  int per_row;
  int act_first;
  Act act;
  float leaky;
};

__device__ __forceinline__ float epi_apply(const EpiDev& e, float v, int m, int n, long long idx) {
  int ch = e.per_row ? m : n;
  if (e.bias) v += e.bias[ch];
  if (e.act_first) {
    v = act_apply(v, e.act, e.leaky);
    if (e.scale) v = v * e.scale[ch] + e.shift[ch];
  } else {
    if (e.scale) v = v * e.scale[ch] + e.shift[ch];
    v = act_apply(v, e.act, e.leaky);
  }
  if (e.residual) v += e.residual[idx];
  return v;
}

// 64x64 tile, BK=16, 256 threads, 4x4 outputs per thread (rows ty+16i, cols tx+16j).
constexpr int TM = 64, TN = 64, TK = 16;

template <bool TRANS_B>
__global__ void __launch_bounds__(256) sgemm_kernel(int M, int N, int K, const float* __restrict__ A, int lda,
                                                    long long sa, const float* __restrict__ B, int ldb,
                                                    long long sb, float* __restrict__ C, int ldc, long long sc,
                                                    EpiDev ep) {
  A += blockIdx.z * sa;
  B += blockIdx.z * sb;
  C += blockIdx.z * sc;
  __shared__ float As[TK][TM + 4];
  __shared__ float Bs[TK][TN + 4];
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  const int m0 = blockIdx.y * TM, n0 = blockIdx.x * TN;
  float acc[4][4] = {};
  for (int k0 = 0; k0 < K; k0 += TK) {
    // A tile: 64 rows x 16 k; thread t loads 4 elements: row (t>>2) + 0..? use linear index
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      int li = threadIdx.x + r * 256;  // 0..1023
      int am = li >> 4, ak = li & 15;  // k fastest -> coalesced along K
      int gm = m0 + am, gk = k0 + ak;
      As[ak][am] = (gm < M && gk < K) ? A[(long long)gm * lda + gk] : 0.f;
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      int li = threadIdx.x + r * 256;
      if (TRANS_B) {
        int bn = li >> 4, bk = li & 15;
        int gn = n0 + bn, gk = k0 + bk;
        Bs[bk][bn] = (gn < N && gk < K) ? B[(long long)gn * ldb + gk] : 0.f;
      } else {
        int bk = li >> 6, bn = li & 63;
        int gn = n0 + bn, gk = k0 + bk;
        Bs[bk][bn] = (gn < N && gk < K) ? B[(long long)gk * ldb + gn] : 0.f;
      }
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < TK; ++kk) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = As[kk][ty + 16 * i];
#pragma unroll
      for (int j = 0; j < 4; ++j) b[j] = Bs[kk][tx + 16 * j];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int m = m0 + ty + 16 * i;
    if (m >= M) continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int n = n0 + tx + 16 * j;
      if (n >= N) continue;
      long long idx = (long long)m * ldc + n;
      C[idx] = epi_apply(ep, acc[i][j], m, n, idx);
    }
  }
}

EpiDev to_dev(const Epilogue& e) {
  return EpiDev{e.bias, e.scale, e.shift, e.residual, e.per_row ? 1 : 0, e.act_first ? 1 : 0, e.act, e.leaky_slope};
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Large-tile SGEMM: 128x128 block tile, BK=8, 256 threads, 8x8 register tile per thread, fp32 FFMA.
// Optional deterministic split-K: partial sums go to a workspace [S][M][N] and are added in split order
// by sgemm_reduce_k, which then applies the epilogue (no atomics, so results do not depend on timing).
namespace {
constexpr int BM2 = 128, BN2 = 128, BK2 = 8;

template <bool TRANS_B>
__global__ void __launch_bounds__(256) sgemm128_kernel(int M, int N, int K, const float* __restrict__ A, int lda,
                                                       const float* __restrict__ B, int ldb, float* __restrict__ C,
                                                       int ldc, int k_per_split, float* __restrict__ partial,
                                                       EpiDev ep) {
  __shared__ __align__(16) float As[2][BK2][BM2];
  __shared__ __align__(16) float Bs[2][BK2][BN2];
  const int tid = threadIdx.x;
  const int tx = tid & 15, ty = tid >> 4;  // 16 x 16 threads
  const int m0 = blockIdx.y * BM2, n0 = blockIdx.x * BN2;
  const int kz = blockIdx.z;
  const int kbeg = kz * k_per_split;
  const int kend = min(K, kbeg + k_per_split);
  // loader mapping: 256 threads x 4 elements = 1024 = 128 x 8
  const int la_m = tid >> 1, la_k = (tid & 1) * 4;  // A: row la_m, k la_k..la_k+3
  float ra[4], rb[4];
  auto load_a = [&](int k0) {
    int gm = m0 + la_m;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int gk = k0 + la_k + i;
      ra[i] = (gm < M && gk < kend) ? A[(long long)gm * lda + gk] : 0.f;
    }
  };
  auto load_b = [&](int k0) {
    if (TRANS_B) {
      int gn = n0 + la_m;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int gk = k0 + la_k + i;
        rb[i] = (gn < N && gk < kend) ? B[(long long)gn * ldb + gk] : 0.f;
      }
    } else {
      // B[k][n]: thread loads k = tid>>5 (0..7), n = (tid&31)*4 .. +3
      int gk = k0 + (tid >> 5);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int gn = n0 + (tid & 31) * 4 + i;
        rb[i] = (gn < N && gk < kend) ? B[(long long)gk * ldb + gn] : 0.f;
      }
    }
  };
  auto store = [&](int buf) {
#pragma unroll
    for (int i = 0; i < 4; ++i) As[buf][la_k + i][la_m] = ra[i];
    if (TRANS_B) {
#pragma unroll
      for (int i = 0; i < 4; ++i) Bs[buf][la_k + i][la_m] = rb[i];
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) Bs[buf][tid >> 5][(tid & 31) * 4 + i] = rb[i];
    }
  };
  float acc[8][8] = {};
  int buf = 0;
  if (kbeg < kend) {
    load_a(kbeg);
    load_b(kbeg);
    store(0);
  }
  __syncthreads();
  for (int k0 = kbeg; k0 < kend; k0 += BK2) {
    bool more = k0 + BK2 < kend;
    if (more) {
      load_a(k0 + BK2);
      load_b(k0 + BK2);
    }
#pragma unroll
    for (int kk = 0; kk < BK2; ++kk) {
      float a[8], b[8];
      // rows ty*4 + {0..3} and 64 + ty*4 + {0..3}; cols tx*4 + {0..3} and 64 + tx*4 + {0..3}
      float4 a0 = *reinterpret_cast<const float4*>(&As[buf][kk][ty * 4]);
      float4 a1 = *reinterpret_cast<const float4*>(&As[buf][kk][64 + ty * 4]);
      float4 b0 = *reinterpret_cast<const float4*>(&Bs[buf][kk][tx * 4]);
      float4 b1 = *reinterpret_cast<const float4*>(&Bs[buf][kk][64 + tx * 4]);
      a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
      a[4] = a1.x; a[5] = a1.y; a[6] = a1.z; a[7] = a1.w;
      b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w;
      b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    if (more) {
      store(buf ^ 1);
      __syncthreads();
      buf ^= 1;
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int m = m0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + (i - 4));
    if (m >= M) continue;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int n = n0 + (j < 4 ? tx * 4 + j : 64 + tx * 4 + (j - 4));
      if (n >= N) continue;
      if (partial) {
        partial[((long long)kz * M + m) * N + n] = acc[i][j];
      } else {
        long long idx = (long long)m * ldc + n;
        C[idx] = epi_apply(ep, acc[i][j], m, n, idx);
      }
    }
  }
}

// Variant: BK=16, float4 global loads (K-contiguous operands, aligned), 2 blocks per SM.
template <bool TRANS_B>
__global__ void __launch_bounds__(256, 2) sgemm128v_kernel(int M, int N, int K, const float* __restrict__ A, int lda,
                                                          const float* __restrict__ B, int ldb, float* __restrict__ C,
                                                          int ldc, int k_per_split, float* __restrict__ partial,
                                                          EpiDev ep) {
  constexpr int BK = 16;
  __shared__ __align__(16) float As[2][BK][BM2];
  __shared__ __align__(16) float Bs[2][BK][BN2];
  const int tid = threadIdx.x;
  const int tx = tid & 15, ty = tid >> 4;
  const int m0 = blockIdx.y * BM2, n0 = blockIdx.x * BN2;
  const int kz = blockIdx.z;
  const int kbeg = kz * k_per_split;
  const int kend = min(K, kbeg + k_per_split);
  const int lr = tid >> 1, lk = (tid & 1) * 8;  // row lr, k lk..lk+7
  float ra[8], rb[8];
  auto load_rows = [&](const float* __restrict__ P, int ld, int r0, int R, int k0, float* dst) {
    int gr = r0 + lr;
    int gk = k0 + lk;
    if (gr < R && gk + 7 < kend) {
      const float4* q = reinterpret_cast<const float4*>(P + (long long)gr * ld + gk);
      float4 u = __ldg(q), v = __ldg(q + 1);
      dst[0] = u.x; dst[1] = u.y; dst[2] = u.z; dst[3] = u.w;
      dst[4] = v.x; dst[5] = v.y; dst[6] = v.z; dst[7] = v.w;
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) dst[i] = (gr < R && gk + i < kend) ? P[(long long)gr * ld + gk + i] : 0.f;
    }
  };
  auto load_b_kn = [&](int k0) {  // B[k][n]: k = tid>>4, n = (tid&15)*8..+7
    int gk = k0 + (tid >> 4);
    int gn = n0 + (tid & 15) * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) rb[i] = (gk < kend && gn + i < N) ? B[(long long)gk * ldb + gn + i] : 0.f;
  };
  auto store = [&](int buf) {
#pragma unroll
    for (int i = 0; i < 8; ++i) As[buf][lk + i][lr] = ra[i];
    if (TRANS_B) {
#pragma unroll
      for (int i = 0; i < 8; ++i) Bs[buf][lk + i][lr] = rb[i];
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) Bs[buf][tid >> 4][(tid & 15) * 8 + i] = rb[i];
    }
  };
  auto load = [&](int k0) {
    load_rows(A, lda, m0, M, k0, ra);
    if (TRANS_B) load_rows(B, ldb, n0, N, k0, rb);
    else load_b_kn(k0);
  };
  float acc[8][8] = {};
  int buf = 0;
  if (kbeg < kend) {
    load(kbeg);
    store(0);
  }
  __syncthreads();
  for (int k0 = kbeg; k0 < kend; k0 += BK) {
    bool more = k0 + BK < kend;
    if (more) load(k0 + BK);
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a[8], b[8];
      float4 a0 = *reinterpret_cast<const float4*>(&As[buf][kk][ty * 4]);
      float4 a1 = *reinterpret_cast<const float4*>(&As[buf][kk][64 + ty * 4]);
      float4 b0 = *reinterpret_cast<const float4*>(&Bs[buf][kk][tx * 4]);
      float4 b1 = *reinterpret_cast<const float4*>(&Bs[buf][kk][64 + tx * 4]);
      a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
      a[4] = a1.x; a[5] = a1.y; a[6] = a1.z; a[7] = a1.w;
      b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w;
      b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    if (more) {
      store(buf ^ 1);
      __syncthreads();
      buf ^= 1;
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int m = m0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + (i - 4));
    if (m >= M) continue;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int n = n0 + (j < 4 ? tx * 4 + j : 64 + tx * 4 + (j - 4));
      if (n >= N) continue;
      if (partial) {
        partial[((long long)kz * M + m) * N + n] = acc[i][j];
      } else {
        long long idx = (long long)m * ldc + n;
        C[idx] = epi_apply(ep, acc[i][j], m, n, idx);
      }
    }
  }
}

__global__ void sgemm_reduce_k(int M, int N, int S, const float* __restrict__ partial, float* C, int ldc, EpiDev ep) {
  long long total = (long long)M * N;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    int m = (int)(i / N), n = (int)(i % N);
    float v = 0.f;
    for (int s = 0; s < S; ++s) v += partial[(long long)s * total + i];
    long long idx = (long long)m * ldc + n;
    C[idx] = epi_apply(ep, v, m, n, idx);
  }
}

int g_sm_count = 0;
struct StreamWs {
  cudaStream_t stream;
  float* ws;
  size_t cap;
};
// Every stream that runs split-K GEMMs registers its own partials buffer here. A VoiceBuilder registers 3 (main,
// decoder-ref side branch, LM-prefix tokens). Sprint 3 found that 4 slots silently overflowed when a second
// builder lived in the same process (e2e_check's layout A/B): its side streams then fell back to the shared
// default buffer, which its main stream also used, and concurrent split-K partials raced. 64 slots, and a full
// table is now an error instead of a silent fallback.
constexpr int kMaxStreamWs = 64;
StreamWs g_stream_ws[kMaxStreamWs];
int g_n_stream_ws = 0;
std::mutex g_stream_ws_mu;
int g_gemm_variant = 1;
int g_split_policy = 0;  // 0: fill the SMs (default); 1: never split
float* g_splitk_ws = nullptr;
size_t g_splitk_cap = 0;

}  // namespace

void sgemm_set_stream_workspace(cudaStream_t stream, float* ws, size_t floats) {
  std::lock_guard<std::mutex> lk(g_stream_ws_mu);
  for (int i = 0; i < g_n_stream_ws; ++i)
    if (g_stream_ws[i].stream == stream) {
      g_stream_ws[i] = {stream, ws, floats};
      return;
    }
  if (g_n_stream_ws >= kMaxStreamWs) throw std::runtime_error("sgemm: stream workspace table full");
  g_stream_ws[g_n_stream_ws++] = {stream, ws, floats};
}

void sgemm_clear_stream_workspace(cudaStream_t stream) {
  std::lock_guard<std::mutex> lk(g_stream_ws_mu);
  for (int i = 0; i < g_n_stream_ws; ++i)
    if (g_stream_ws[i].stream == stream) {
      g_stream_ws[i] = g_stream_ws[--g_n_stream_ws];
      return;
    }
}

void sgemm_set_variant(int variant, int split_policy) {
  g_gemm_variant = variant;
  g_split_policy = split_policy;
}

void sgemm_set_workspace(float* ws, size_t floats, int sm_count) {
  g_splitk_ws = ws;
  g_splitk_cap = floats;
  g_sm_count = sm_count;
}

namespace {
// The split-K workspace for this stream (concurrent streams must not share partial buffers).
inline void ws_for(cudaStream_t s, float** ws, size_t* cap) {
  std::lock_guard<std::mutex> lk(g_stream_ws_mu);
  for (int i = 0; i < g_n_stream_ws; ++i)
    if (g_stream_ws[i].stream == s) {
      *ws = g_stream_ws[i].ws;
      *cap = g_stream_ws[i].cap;
      return;
    }
  *ws = g_splitk_ws;
  *cap = g_splitk_cap;
}

bool sgemm_big(cudaStream_t s, int M, int N, int K, const float* A, int lda, const float* B, int ldb, bool trans_b,
               float* C, int ldc, const Epilogue& ep) {
  if ((long long)M * N < 128LL * 128 * 2 || K < 64) return false;
  int tiles = ((M + BM2 - 1) / BM2) * ((N + BN2 - 1) / BN2);
  int sms = g_sm_count > 0 ? g_sm_count : 128;
  int splits = 1;
  while (g_split_policy == 0 && tiles * splits < sms && splits < 16 && K / (splits * 2) >= 128) splits *= 2;
  float* wsp = nullptr;
  size_t wcap = 0;
  ws_for(s, &wsp, &wcap);
  if (splits > 1 && (!wsp || (size_t)splits * M * N > wcap)) splits = 1;
  int kps = ((K + splits - 1) / splits + BK2 - 1) / BK2 * BK2;
  splits = (K + kps - 1) / kps;
  dim3 grid((N + BN2 - 1) / BN2, (M + BM2 - 1) / BM2, splits);
  EpiDev e = to_dev(ep);
  float* part = splits > 1 ? wsp : nullptr;
  const bool aligned = (lda % 4 == 0) && (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
                       (!trans_b || ((ldb % 4 == 0) && (reinterpret_cast<uintptr_t>(B) % 16 == 0)));
  if (g_gemm_variant == 1 && aligned) {
    if (trans_b)
      sgemm128v_kernel<true><<<grid, 256, 0, s>>>(M, N, K, A, lda, B, ldb, C, ldc, kps, part, e);
    else
      sgemm128v_kernel<false><<<grid, 256, 0, s>>>(M, N, K, A, lda, B, ldb, C, ldc, kps, part, e);
  } else if (trans_b)
    sgemm128_kernel<true><<<grid, 256, 0, s>>>(M, N, K, A, lda, B, ldb, C, ldc, kps, part, e);
  else
    sgemm128_kernel<false><<<grid, 256, 0, s>>>(M, N, K, A, lda, B, ldb, C, ldc, kps, part, e);
  if (part) sgemm_reduce_k<<<grid_cap((long long)M * N), 256, 0, s>>>(M, N, splits, part, C, ldc, e);
  check(cudaGetLastError(), "sgemm128");
  return true;
}
}  // namespace

namespace {
// y[m] = A[m,:] . x  (N == 1): one warp per output row, fp32 FFMA in lane-strided order, shuffle tree.
__global__ void gemv_k(int M, int K, const float* __restrict__ A, int lda, const float* __restrict__ x, int incx,
                       float* C, int ldc, EpiDev ep) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= M) return;
  const float* a = A + (long long)warp * lda;
  float acc = 0.f;
  for (int k = lane; k < K; k += 32) acc = fmaf(a[k], x[(long long)k * incx], acc);
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) {
    long long idx = (long long)warp * ldc;
    C[idx] = epi_apply(ep, acc, warp, 0, idx);
  }
}

// 64x64 kernel with a K range per blockIdx.z (split-K partials, reduced by sgemm_reduce_k).
template <bool TRANS_B>
__global__ void __launch_bounds__(256) sgemm64s_kernel(int M, int N, int K, const float* __restrict__ A, int lda,
                                                       const float* __restrict__ B, int ldb, int kps,
                                                       float* __restrict__ partial) {
  __shared__ float As[TK][TM + 4];
  __shared__ float Bs[TK][TN + 4];
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  const int m0 = blockIdx.y * TM, n0 = blockIdx.x * TN;
  const int kbeg = blockIdx.z * kps, kend = min(K, kbeg + kps);
  float acc[4][4] = {};
  for (int k0 = kbeg; k0 < kend; k0 += TK) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      int li = threadIdx.x + r * 256;
      int am = li >> 4, ak = li & 15;
      int gm = m0 + am, gk = k0 + ak;
      As[ak][am] = (gm < M && gk < kend) ? A[(long long)gm * lda + gk] : 0.f;
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      int li = threadIdx.x + r * 256;
      if (TRANS_B) {
        int bn = li >> 4, bk = li & 15;
        int gn = n0 + bn, gk = k0 + bk;
        Bs[bk][bn] = (gn < N && gk < kend) ? B[(long long)gn * ldb + gk] : 0.f;
      } else {
        int bk = li >> 6, bn = li & 63;
        int gn = n0 + bn, gk = k0 + bk;
        Bs[bk][bn] = (gn < N && gk < kend) ? B[(long long)gk * ldb + gn] : 0.f;
      }
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < TK; ++kk) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = As[kk][ty + 16 * i];
#pragma unroll
      for (int j = 0; j < 4; ++j) b[j] = Bs[kk][tx + 16 * j];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int m = m0 + ty + 16 * i;
    if (m >= M) continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int n = n0 + tx + 16 * j;
      if (n < N) partial[((long long)blockIdx.z * M + m) * N + n] = acc[i][j];
    }
  }
}
}  // namespace

void sgemm_batched(cudaStream_t s, int batch, int M, int N, int K, const float* A, int lda, long long sa,
                   const float* B, int ldb, long long sb, bool trans_b, float* C, int ldc, long long sc,
                   const Epilogue& ep) {
  if (batch == 1 && N == 1 && !trans_b) {
    gemv_k<<<(M * 32 + 255) / 256, 256, 0, s>>>(M, K, A, lda, B, ldb, C, ldc, to_dev(ep));
    check(cudaGetLastError(), "gemv");
    return;
  }
  if (batch == 1 && sgemm_big(s, M, N, K, A, lda, B, ldb, trans_b, C, ldc, ep)) return;
  float* wsp = nullptr;
  size_t wcap = 0;
  ws_for(s, &wsp, &wcap);
  if (batch == 1 && wsp && K >= 128) {
    int tiles = ((M + TM - 1) / TM) * ((N + TN - 1) / TN);
    int sms = g_sm_count > 0 ? g_sm_count : 128;
    int splits = 1;
    while (tiles * splits < sms && splits < 16 && K / (splits * 2) >= 32) splits *= 2;
    if (splits > 1 && (size_t)splits * M * N <= wcap) {
      int kps = ((K + splits - 1) / splits + TK - 1) / TK * TK;
      splits = (K + kps - 1) / kps;
      dim3 grid((N + TN - 1) / TN, (M + TM - 1) / TM, splits);
      if (trans_b)
        sgemm64s_kernel<true><<<grid, 256, 0, s>>>(M, N, K, A, lda, B, ldb, kps, wsp);
      else
        sgemm64s_kernel<false><<<grid, 256, 0, s>>>(M, N, K, A, lda, B, ldb, kps, wsp);
      sgemm_reduce_k<<<grid_cap((long long)M * N), 256, 0, s>>>(M, N, splits, wsp, C, ldc, to_dev(ep));
      check(cudaGetLastError(), "sgemm64 split");
      return;
    }
  }
  dim3 grid((N + TN - 1) / TN, (M + TM - 1) / TM, batch);
  if (trans_b)
    sgemm_kernel<true><<<grid, 256, 0, s>>>(M, N, K, A, lda, sa, B, ldb, sb, C, ldc, sc, to_dev(ep));
  else
    sgemm_kernel<false><<<grid, 256, 0, s>>>(M, N, K, A, lda, sa, B, ldb, sb, C, ldc, sc, to_dev(ep));
  check(cudaGetLastError(), "sgemm");
}

void sgemm(cudaStream_t s, int M, int N, int K, const float* A, int lda, const float* B, int ldb, bool trans_b,
           float* C, int ldc, const Epilogue& ep) {
  sgemm_batched(s, 1, M, N, K, A, lda, 0, B, ldb, 0, trans_b, C, ldc, 0, ep);
}

// ---------------------------------------------------------------------------------------------
namespace {

__global__ void im2col1d_k(const float* __restrict__ x, int cin, int T, int K, int stride, int dil, int pad,
                           int t_out, float* __restrict__ col) {
  long long total = (long long)cin * K * t_out;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    int t = (int)(i % t_out);
    long long r = i / t_out;
    int k = (int)(r % K), ci = (int)(r / K);
    int src = t * stride + k * dil - pad;
    col[i] = (src >= 0 && src < T) ? x[(long long)ci * T + src] : 0.f;
  }
}

__global__ void im2col2d_k(const float* __restrict__ x, int cin, int H, int W, int kh, int kw, int sh, int sw,
                           int ph, int pw, int Ho, int Wo, float* __restrict__ col) {
  long long cols = (long long)Ho * Wo;
  long long total = (long long)cin * kh * kw * cols;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    long long c = i % cols;
    long long r = i / cols;
    int wo = (int)(c % Wo), ho = (int)(c / Wo);
    int j = (int)(r % kw);
    long long r2 = r / kw;
    int ii = (int)(r2 % kh), ci = (int)(r2 / kh);
    int hi = ho * sh + ii - ph, wi = wo * sw + j - pw;
    col[i] = (hi >= 0 && hi < H && wi >= 0 && wi < W) ? x[((long long)ci * H + hi) * W + wi] : 0.f;
  }
}

}  // namespace
namespace {
unsigned grid_cap(long long n) {
  long long b = (n + 255) / 256;
  return (unsigned)(b > 65535LL * 16 ? 65535LL * 16 : b);
}

}  // namespace

void im2col1d(cudaStream_t s, const float* x, int cin, int T, int K, int stride, int dil, int pad, int t_out,
              float* col) {
  long long n = (long long)cin * K * t_out;
  im2col1d_k<<<grid_cap(n), 256, 0, s>>>(x, cin, T, K, stride, dil, pad, t_out, col);
  check(cudaGetLastError(), "im2col1d");
}

void im2col2d(cudaStream_t s, const float* x, int cin, int H, int W, int kh, int kw, int sh, int sw, int ph,
              int pw, int Ho, int Wo, float* col) {
  long long n = (long long)cin * kh * kw * Ho * Wo;
  im2col2d_k<<<grid_cap(n), 256, 0, s>>>(x, cin, H, W, kh, kw, sh, sw, ph, pw, Ho, Wo, col);
  check(cudaGetLastError(), "im2col2d");
}

// ---------------------------------------------------------------------------------------------
namespace {

__global__ void affine_act_k(const float* __restrict__ x, float* __restrict__ y, int C, int T, int ldx, int ldy,
                             const float* __restrict__ scale, const float* __restrict__ shift, Act act,
                             int act_first, float leaky) {
  long long total = (long long)C * T;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    int c = (int)(i / T), t = (int)(i % T);
    float v = x[(long long)c * ldx + t];
    if (act_first) {
      v = act_apply(v, act, leaky);
      if (scale) v = v * scale[c] + shift[c];
    } else {
      if (scale) v = v * scale[c] + shift[c];
      v = act_apply(v, act, leaky);
    }
    y[(long long)c * ldy + t] = v;
  }
}

__global__ void add_k(const float* a, const float* b, float* y, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    y[i] = a[i] + b[i];
}
__global__ void mul_k(const float* a, const float* b, float* y, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    y[i] = a[i] * b[i];
}
__global__ void act_k(const float* x, float* y, long long n, Act a, float leaky) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    y[i] = act_apply(x[i], a, leaky);
}
__global__ void abs_k(float* x, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    x[i] = fabsf(x[i]);
}
__global__ void scale_k(float* x, long long n, float a) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    x[i] = x[i] * a;
}

__global__ void maxpool_k(const float* x, float* y, int C, int T, int k, int t_out) {
  long long total = (long long)C * t_out;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    int c = (int)(i / t_out), t = (int)(i % t_out);
    const float* p = x + (long long)c * T + (long long)t * k;
    float m = p[0];
    for (int j = 1; j < k; ++j) m = fmaxf(m, p[j]);
    y[i] = m;
  }
}

// Block reduction of doubles (blockDim.x == 256).
__device__ double block_sum(double v, double* sh) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  double r = 0.0;
  if (threadIdx.x < 32) {
    r = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.0;
    for (int o = 16; o > 0; o >>= 1) r += __shfl_down_sync(0xffffffffu, r, o);
    if (threadIdx.x == 0) sh[0] = r;
  }
  __syncthreads();
  r = sh[0];
  __syncthreads();
  return r;
}

__device__ float block_sumf(float v, float* sh) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) sh[w] = v;
  __syncthreads();
  float r = 0.f;
  if (threadIdx.x < 32) {
    r = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) r += __shfl_down_sync(0xffffffffu, r, o);
    if (threadIdx.x == 0) sh[0] = r;
  }
  __syncthreads();
  r = sh[0];
  __syncthreads();
  return r;
}

__global__ void instance_norm_k(const float* x, float* y, int C, int T, const float* w, const float* b, float eps,
                                Act act, float leaky) {
  __shared__ double sh[32];
  int c = blockIdx.x;
  const float* p = x + (long long)c * T;
  double s = 0.0;
  for (int t = threadIdx.x; t < T; t += blockDim.x) s += p[t];
  double mean = block_sum(s, sh) / T;
  double v = 0.0;
  for (int t = threadIdx.x; t < T; t += blockDim.x) {
    double d = p[t] - mean;
    v += d * d;
  }
  double var = block_sum(v, sh) / T;
  float inv = (float)(1.0 / sqrt(var + (double)eps));
  float fm = (float)mean;
  float ww = w ? w[c] : 1.f, bb = b ? b[c] : 0.f;
  for (int t = threadIdx.x; t < T; t += blockDim.x) {
    float o = (p[t] - fm) * inv * ww + bb;
    y[(long long)c * T + t] = act_apply(o, act, leaky);
  }
}

__global__ void stats_pool_k(const float* x, int C, int T, int ldx, float* out) {
  __shared__ double sh[32];
  int c = blockIdx.x;
  const float* p = x + (long long)c * ldx;
  double s = 0.0;
  for (int t = threadIdx.x; t < T; t += blockDim.x) s += p[t];
  double mean = block_sum(s, sh) / T;
  double v = 0.0;
  for (int t = threadIdx.x; t < T; t += blockDim.x) {
    double d = p[t] - mean;
    v += d * d;
  }
  double var = block_sum(v, sh) / (T > 1 ? T - 1 : 1);
  if (threadIdx.x == 0) {
    out[c] = (float)mean;
    out[C + c] = (float)sqrt(var);
  }
}

__global__ void cam_context_k(const float* x, int C, int T, int seg, float* ctx) {
  __shared__ double sh[32];
  int c = blockIdx.x;
  const float* p = x + (long long)c * T;
  double s = 0.0;
  for (int t = threadIdx.x; t < T; t += blockDim.x) s += p[t];
  float mean = (float)(block_sum(s, sh) / T);
  int nseg = (T + seg - 1) / seg;
  for (int g = 0; g < nseg; ++g) {
    int t0 = g * seg, t1 = min(T, t0 + seg);
    float acc = 0.f;
    for (int t = t0 + threadIdx.x; t < t1; t += blockDim.x) acc += p[t];
    __shared__ float shf[32];
    float tot = block_sumf(acc, shf);
    float avg = tot / (float)(t1 - t0);
    for (int t = t0 + threadIdx.x; t < t1; t += blockDim.x) ctx[(long long)c * T + t] = mean + avg;
  }
}

__global__ void sub_col_mean_k(float* x, int T, int F) {
  __shared__ double sh[32];
  int f = blockIdx.x;
  double s = 0.0;
  for (int t = threadIdx.x; t < T; t += blockDim.x) s += x[(long long)t * F + f];
  float mean = (float)(block_sum(s, sh) / T);
  for (int t = threadIdx.x; t < T; t += blockDim.x) x[(long long)t * F + f] -= mean;
}

__global__ void layer_norm_k(const float* x, float* y, int D, const float* w, const float* b, float eps) {
  __shared__ float sh[32];
  long long r = blockIdx.x;
  const float* p = x + r * D;
  float s = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) s += p[i];
  float mean = block_sumf(s, sh) / D;
  float v = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float d = p[i] - mean;
    v += d * d;
  }
  float var = block_sumf(v, sh) / D;
  float inv = rsqrtf(var + eps);
  for (int i = threadIdx.x; i < D; i += blockDim.x) y[r * D + i] = (p[i] - mean) * inv * w[i] + b[i];
}

}  // namespace

void affine_act(cudaStream_t s, const float* x, float* y, int C, int T, int ldx, int ldy, const float* scale,
                const float* shift, Act act, bool act_first, float leaky) {
  long long n = (long long)C * T;
  affine_act_k<<<grid_cap(n), 256, 0, s>>>(x, y, C, T, ldx, ldy, scale, shift, act, act_first ? 1 : 0, leaky);
  check(cudaGetLastError(), "affine_act");
}
void eltwise_add(cudaStream_t s, const float* a, const float* b, float* y, long long n) {
  add_k<<<grid_cap(n), 256, 0, s>>>(a, b, y, n);
  check(cudaGetLastError(), "add");
}
void eltwise_mul(cudaStream_t s, const float* a, const float* b, float* y, long long n) {
  mul_k<<<grid_cap(n), 256, 0, s>>>(a, b, y, n);
  check(cudaGetLastError(), "mul");
}
void eltwise_act(cudaStream_t s, const float* x, float* y, long long n, Act act, float leaky) {
  act_k<<<grid_cap(n), 256, 0, s>>>(x, y, n, act, leaky);
  check(cudaGetLastError(), "act");
}
void abs_inplace(cudaStream_t s, float* x, long long n) {
  abs_k<<<grid_cap(n), 256, 0, s>>>(x, n);
  check(cudaGetLastError(), "abs");
}
void scale_inplace(cudaStream_t s, float* x, long long n, float a) {
  scale_k<<<grid_cap(n), 256, 0, s>>>(x, n, a);
  check(cudaGetLastError(), "scale");
}
void maxpool1d(cudaStream_t s, const float* x, float* y, int C, int T, int k, int t_out) {
  long long n = (long long)C * t_out;
  maxpool_k<<<grid_cap(n), 256, 0, s>>>(x, y, C, T, k, t_out);
  check(cudaGetLastError(), "maxpool");
}
namespace {
// Long-sequence path: per-(channel, chunk) partial sums in fp64, then a normalise pass.
__global__ void in_partial_k(const float* x, int T, int chunks, double* part) {
  __shared__ double sh[32];
  int c = blockIdx.y, b = blockIdx.x;
  long long t0 = (long long)T * b / chunks, t1 = (long long)T * (b + 1) / chunks;
  const float* p = x + (long long)c * T;
  double s1 = 0.0, s2 = 0.0;
  for (long long t = t0 + threadIdx.x; t < t1; t += blockDim.x) {
    double v = p[t];
    s1 += v;
    s2 += v * v;
  }
  s1 = block_sum(s1, sh);
  s2 = block_sum(s2, sh);
  if (threadIdx.x == 0) {
    part[((long long)c * chunks + b) * 2] = s1;
    part[((long long)c * chunks + b) * 2 + 1] = s2;
  }
}
__global__ void in_apply_k(const float* x, float* y, int C, int T, int chunks, const double* part, const float* w,
                           const float* b, float eps, Act act, float leaky) {
  long long total = (long long)C * T;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    int c = (int)(i / T);
    double s1 = 0.0, s2 = 0.0;
    for (int k = 0; k < chunks; ++k) {
      s1 += part[((long long)c * chunks + k) * 2];
      s2 += part[((long long)c * chunks + k) * 2 + 1];
    }
    double mean = s1 / T;
    double var = s2 / T - mean * mean;
    if (var < 0) var = 0;
    float inv = (float)(1.0 / sqrt(var + (double)eps));
    float o = (x[i] - (float)mean) * inv * (w ? w[c] : 1.f) + (b ? b[c] : 0.f);
    y[i] = act_apply(o, act, leaky);
  }
}
}  // namespace

void instance_norm(cudaStream_t s, const float* x, float* y, int C, int T, const float* w, const float* b, float eps,
                   Act act, float leaky, double* scratch) {
  if (C >= 32 || T < 8192 || scratch == nullptr) {
    instance_norm_k<<<C, 256, 0, s>>>(x, y, C, T, w, b, eps, act, leaky);
  } else {
    int chunks = 64;
    in_partial_k<<<dim3(chunks, C), 256, 0, s>>>(x, T, chunks, scratch);
    in_apply_k<<<grid_cap((long long)C * T), 256, 0, s>>>(x, y, C, T, chunks, scratch, w, b, eps, act, leaky);
  }
  check(cudaGetLastError(), "instance_norm");
}
void stats_pool(cudaStream_t s, const float* x, int C, int T, int ldx, float* out) {
  stats_pool_k<<<C, 256, 0, s>>>(x, C, T, ldx, out);
  check(cudaGetLastError(), "stats_pool");
}
namespace {
__global__ void cam_context_seg_k(const float* x, int C, int T, int seg, int nseg, float* ctx) {
  __shared__ double sh[32];
  __shared__ float shf[32];
  int c = blockIdx.x;
  const float* p = x + (long long)c * T;
  double s = 0.0;
  for (int t = threadIdx.x; t < T; t += blockDim.x) s += p[t];
  float mean = (float)(block_sum(s, sh) / T);
  for (int g = 0; g < nseg; ++g) {
    int t0 = g * seg, t1 = min(T, t0 + seg);
    float acc = 0.f;
    for (int t = t0 + threadIdx.x; t < t1; t += blockDim.x) acc += p[t];
    float tot = block_sumf(acc, shf);
    if (threadIdx.x == 0) ctx[(long long)c * nseg + g] = mean + tot / (float)(t1 - t0);
  }
}
// y[c][t] *= m[c][t / seg]
__global__ void mul_seg_k(const float* y, const float* m, float* out, int C, int T, int seg, int nseg) {
  long long total = (long long)C * T;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    int c = (int)(i / T), t = (int)(i % T);
    out[i] = y[i] * m[(long long)c * nseg + t / seg];
  }
}
}  // namespace

void cam_context_seg(cudaStream_t s, const float* x, int C, int T, int seg, float* ctx_seg) {
  int nseg = (T + seg - 1) / seg;
  cam_context_seg_k<<<C, 256, 0, s>>>(x, C, T, seg, nseg, ctx_seg);
  check(cudaGetLastError(), "cam_context_seg");
}
void mul_seg(cudaStream_t s, const float* y, const float* m, float* out, int C, int T, int seg) {
  int nseg = (T + seg - 1) / seg;
  mul_seg_k<<<grid_cap((long long)C * T), 256, 0, s>>>(y, m, out, C, T, seg, nseg);
  check(cudaGetLastError(), "mul_seg");
}

namespace {
// One block per segment g: h = relu(W1[64][128] ctx[:,g] + b1); m = sigmoid(W2[32][64] h + b2).
__global__ void cam_mlp_k(const float* ctx, int nseg, const float* W1, const float* b1, const float* W2,
                          const float* b2, float* m) {
  __shared__ float c[128];
  __shared__ float h[64];
  int g = blockIdx.x;
  for (int i = threadIdx.x; i < 128; i += blockDim.x) c[i] = ctx[(long long)i * nseg + g];
  __syncthreads();
  if (threadIdx.x < 64) {
    float acc = 0.f;
    for (int k = 0; k < 128; ++k) acc = fmaf(W1[threadIdx.x * 128 + k], c[k], acc);
    acc += b1[threadIdx.x];
    h[threadIdx.x] = acc > 0.f ? acc : 0.f;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    float acc = 0.f;
    for (int k = 0; k < 64; ++k) acc = fmaf(W2[threadIdx.x * 64 + k], h[k], acc);
    acc += b2[threadIdx.x];
    m[(long long)threadIdx.x * nseg + g] = 1.f / (1.f + expf(-acc));
  }
}
}  // namespace

void cam_mlp(cudaStream_t s, const float* ctx_seg, int nseg, const float* W1, const float* b1, const float* W2,
             const float* b2, float* m_seg) {
  cam_mlp_k<<<nseg, 64, 0, s>>>(ctx_seg, nseg, W1, b1, W2, b2, m_seg);
  check(cudaGetLastError(), "cam_mlp");
}

void cam_context(cudaStream_t s, const float* x, int C, int T, int seg, float* ctx) {
  cam_context_k<<<C, 256, 0, s>>>(x, C, T, seg, ctx);
  check(cudaGetLastError(), "cam_context");
}
void sub_col_mean(cudaStream_t s, float* x, int T, int F) {
  sub_col_mean_k<<<F, 256, 0, s>>>(x, T, F);
  check(cudaGetLastError(), "sub_col_mean");
}
void layer_norm(cudaStream_t s, const float* x, float* y, int rows, int D, const float* w, const float* b, float eps) {
  layer_norm_k<<<rows, 256, 0, s>>>(x, y, D, w, b, eps);
  check(cudaGetLastError(), "layer_norm");
}

// ---------------------------------------------------------------------------------------------
namespace {

// scipy upfirdn order: for each output, products x[s]*h in ascending s, float32 add (no FMA).
__global__ void resample_poly_k(const float* __restrict__ x, int n, const float* __restrict__ h, float* y,
                                int n_out) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= n_out) return;
  // out[o] = sum_s h[3o + 30 - 2s] x[s],  0 <= 3o+30-2s <= 60
  int s_lo = (3 * o - 30 + 1) / 2;  // ceil((3o-30)/2) for 3o-30 >= 0
  if (3 * o - 30 < 0) s_lo = 0;
  int s_hi = (3 * o + 30) / 2;
  if (s_hi > n - 1) s_hi = n - 1;
  float acc = 0.f;
  for (int s = s_lo; s <= s_hi; ++s) acc = __fadd_rn(acc, __fmul_rn(x[s], h[3 * o + 30 - 2 * s]));
  y[o] = acc;
}

__global__ void resample_sinc_k(const float* __restrict__ x, int n, const float* __restrict__ ker, float* y,
                                int n_out) {
  int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= n_out) return;
  int j = o >> 1, c = o & 1;
  const float* kk = ker + c * 23;
  float acc = 0.f;
  for (int k = 0; k < 23; ++k) {
    int src = 3 * j + k - 10;
    float v = (src >= 0 && src < n) ? x[src] : 0.f;
    acc = fmaf(kk[k], v, acc);
  }
  y[o] = acc;
}

__global__ void make_frames_k(const float* __restrict__ x, int n, int hop, int pad_left, int reflect, int win_len,
                              const float* __restrict__ win, int n_fft, int kaldi, float* __restrict__ out) {
  extern __shared__ float fr[];
  __shared__ float sh[32];
  int f = blockIdx.x;
  for (int i = threadIdx.x; i < win_len; i += blockDim.x) {
    long long idx = (long long)f * hop + i - pad_left;
    float v;
    if (reflect) {
      if (idx < 0) idx = -idx;
      if (idx >= n) idx = 2LL * (n - 1) - idx;
      v = x[idx];
    } else {
      v = (idx >= 0 && idx < n) ? x[idx] : 0.f;
    }
    fr[i] = v;
  }
  __syncthreads();
  if (kaldi) {
    float s = 0.f;
    for (int i = threadIdx.x; i < win_len; i += blockDim.x) s += fr[i];
    float mean = block_sumf(s, sh) / win_len;
    for (int i = threadIdx.x; i < win_len; i += blockDim.x) fr[i] -= mean;
    __syncthreads();
    // pre-emphasis 0.97 with replicate padding: y[i] = x[i] - 0.97 x[max(i-1,0)]
    float vals[8];
    int cnt = 0;
    for (int i = threadIdx.x; i < win_len; i += blockDim.x) vals[cnt++] = fr[i] - 0.97f * fr[i > 0 ? i - 1 : 0];
    __syncthreads();
    cnt = 0;
    for (int i = threadIdx.x; i < win_len; i += blockDim.x) fr[i] = vals[cnt++];
    __syncthreads();
  }
  float* o = out + (long long)f * n_fft;
  for (int i = threadIdx.x; i < n_fft; i += blockDim.x) o[i] = i < win_len ? fr[i] * win[i] : 0.f;
}

__global__ void dft_basis_k(int n_fft, int bins, float* basis) {
  long long total = (long long)n_fft * 2 * bins;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    int col = (int)(i % (2 * bins));
    int t = (int)(i / (2 * bins));
    int k = col < bins ? col : col - bins;
    long long ph = ((long long)k * t) % n_fft;
    double ang = 6.283185307179586476925 * (double)ph / (double)n_fft;
    basis[i] = col < bins ? (float)cos(ang) : (float)(-sin(ang));
  }
}

__global__ void spec_mag_k(const float* spec, int frames, int bins, int mode, float* mag) {
  long long total = (long long)frames * bins;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    int f = (int)(i / bins), k = (int)(i % bins);
    float re = spec[(long long)f * 2 * bins + k], im = spec[(long long)f * 2 * bins + bins + k];
    float p = re * re + im * im;
    if (mode == 0) mag[i] = sqrtf(p + 1e-9f);
    else if (mode == 1) mag[i] = p;
    else {
      float a = hypotf(re, im);
      mag[i] = a * a;
    }
  }
}

__global__ void log_floor_k(float* x, long long n, int mode) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    float v = x[i];
    if (mode == 0) x[i] = logf(fmaxf(v, 1e-5f));
    else if (mode == 1) x[i] = logf(fmaxf(v, FLT_EPSILON));
    else x[i] = log10f(fmaxf(v, 1e-10f));
  }
}

__global__ void max_reduce_k(const float* x, long long n, float* out) {
  __shared__ float sh[256];
  float m = -FLT_MAX;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    m = fmaxf(m, x[i]);
  sh[threadIdx.x] = m;
  __syncthreads();
  for (int o = 128; o > 0; o >>= 1) {
    if (threadIdx.x < o) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + o]);
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = sh[0];
}

__global__ void s3_finish_k(float* x, long long n, const float* partial, int np) {
  __shared__ float mx;
  if (threadIdx.x == 0) {
    float m = -FLT_MAX;
    for (int i = 0; i < np; ++i) m = fmaxf(m, partial[i]);
    mx = m - 8.0f;
  }
  __syncthreads();
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    x[i] = (fmaxf(x[i], mx) + 4.0f) / 4.0f;
}

}  // namespace

void resample_poly_2_3(cudaStream_t s, const float* x, int n, const float* taps61, float* y, int n_out) {
  resample_poly_k<<<blocks_for(n_out, 256), 256, 0, s>>>(x, n, taps61, y, n_out);
  check(cudaGetLastError(), "resample_poly");
}
void resample_sinc_3_2(cudaStream_t s, const float* x, int n, const float* kernel2x23, float* y, int n_out) {
  resample_sinc_k<<<blocks_for(n_out, 256), 256, 0, s>>>(x, n, kernel2x23, y, n_out);
  check(cudaGetLastError(), "resample_sinc");
}
void make_frames(cudaStream_t s, const float* x, int n, int n_frames, int hop, int pad_left, bool reflect,
                 int win_len, const float* win, int n_fft, bool kaldi, float* out) {
  if (n_frames <= 0) return;
  make_frames_k<<<n_frames, 256, win_len * sizeof(float), s>>>(x, n, hop, pad_left, reflect ? 1 : 0, win_len, win,
                                                              n_fft, kaldi ? 1 : 0, out);
  check(cudaGetLastError(), "make_frames");
}
void make_dft_basis(cudaStream_t s, int n_fft, int bins, float* basis) {
  long long n = (long long)n_fft * 2 * bins;
  dft_basis_k<<<grid_cap(n), 256, 0, s>>>(n_fft, bins, basis);
  check(cudaGetLastError(), "dft_basis");
}
void spec_mag(cudaStream_t s, const float* spec, int frames, int bins, int mode, float* mag) {
  long long n = (long long)frames * bins;
  spec_mag_k<<<grid_cap(n), 256, 0, s>>>(spec, frames, bins, mode, mag);
  check(cudaGetLastError(), "spec_mag");
}
void log_floor(cudaStream_t s, float* x, long long n, int mode) {
  log_floor_k<<<grid_cap(n), 256, 0, s>>>(x, n, mode);
  check(cudaGetLastError(), "log_floor");
}
void s3_logmel_finish(cudaStream_t s, float* x, long long n, float* scratch) {
  int np = 120;
  max_reduce_k<<<np, 256, 0, s>>>(x, n, scratch);
  s3_finish_k<<<grid_cap(n), 256, 0, s>>>(x, n, scratch, np);
  check(cudaGetLastError(), "s3_finish");
}

// ---------------------------------------------------------------------------------------------
namespace {

__global__ void rotary_k(float* x, int T, int H, const float* freqs) {
  long long total = (long long)T * H * 32;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    int d = (int)(i % 32);
    long long r = i / 32;
    int h = (int)(r % H), t = (int)(r / H);
    float ang = (float)t * freqs[d];
    float c = cosf(ang), sn = sinf(ang);
    float* p = x + (long long)t * H * 64 + h * 64;
    float a = p[d], b = p[d + 32];
    p[d] = a * c + (-b) * sn;
    p[d + 32] = b * c + a * sn;
  }
}

__global__ void fsmn_k(const float* v, const float* w, float* out, int T, int C, int K) {
  long long total = (long long)T * C;
  int pad = (K - 1) / 2;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    int c = (int)(i % C), t = (int)(i / C);
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
      int src = t + k - pad;
      if (src >= 0 && src < T) acc = fmaf(w[c * K + k], v[(long long)src * C + c], acc);
    }
    out[i] = acc + v[i];
  }
}

// One block per (query row tile of 4 rows, head). Scores for 4 rows in shared memory.
constexpr int ATT_Q = 4;
__global__ void attention_k(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                            float* __restrict__ out, int T, int H) {
  extern __shared__ float sc[];  // ATT_Q * T
  __shared__ float qs[ATT_Q][64];
  __shared__ float red[32];
  int h = blockIdx.y;
  int t0 = blockIdx.x * ATT_Q;
  int ld = H * 64;
  for (int i = threadIdx.x; i < ATT_Q * 64; i += blockDim.x) {
    int r = i / 64, d = i % 64;
    qs[r][d] = (t0 + r < T) ? q[(long long)(t0 + r) * ld + h * 64 + d] : 0.f;
  }
  __syncthreads();
  for (int j = threadIdx.x; j < T; j += blockDim.x) {
    const float* kp = k + (long long)j * ld + h * 64;
    float acc[ATT_Q] = {};
    for (int d = 0; d < 64; ++d) {
      float kv = kp[d];
#pragma unroll
      for (int r = 0; r < ATT_Q; ++r) acc[r] = fmaf(qs[r][d], kv, acc[r]);
    }
#pragma unroll
    for (int r = 0; r < ATT_Q; ++r) sc[r * T + j] = acc[r];
  }
  __syncthreads();
  for (int r = 0; r < ATT_Q; ++r) {
    float m = -FLT_MAX;
    for (int j = threadIdx.x; j < T; j += blockDim.x) m = fmaxf(m, sc[r * T + j]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
      float mm = red[0];
      for (int i = 1; i < (int)(blockDim.x >> 5); ++i) mm = fmaxf(mm, red[i]);
      red[0] = mm;
    }
    __syncthreads();
    m = red[0];
    __syncthreads();
    float s = 0.f;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
      float e = expf(sc[r * T + j] - m);
      sc[r * T + j] = e;
      s += e;
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
      float ss = 0.f;
      for (int i = 0; i < (int)(blockDim.x >> 5); ++i) ss += red[i];
      red[0] = ss;
    }
    __syncthreads();
    float inv = 1.f / red[0];
    __syncthreads();
    for (int j = threadIdx.x; j < T; j += blockDim.x) sc[r * T + j] *= inv;
  }
  __syncthreads();
  // out[t0+r][h*64+d] = sum_j p[r][j] v[j][d]; 256 threads = 4 rows x 64 dims
  int r = threadIdx.x / 64, d = threadIdx.x % 64;
  if (r < ATT_Q && t0 + r < T) {
    float acc = 0.f;
    for (int j = 0; j < T; ++j) acc = fmaf(sc[r * T + j], v[(long long)j * ld + h * 64 + d], acc);
    out[(long long)(t0 + r) * ld + h * 64 + d] = acc;
  }
}

__global__ void fsq_k(const float* h, int T, int32_t* codes) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  float mu = 0.f;
  float p = 1.f;
  for (int i = 0; i < 8; ++i) {
    float v = tanhf(h[t * 8 + i]) * 0.9990000128746033f;
    v = rintf(v) + 1.f;
    mu += v * p;
    p *= 3.f;
  }
  codes[t] = (int32_t)mu;
}

__global__ void transpose_k(const float* x, float* y, int R, int C) {
  __shared__ float tile[32][33];
  int bx = blockIdx.x * 32, by = blockIdx.y * 32;
  for (int i = threadIdx.y; i < 32; i += 8) {
    int r = by + i, c = bx + threadIdx.x;
    if (r < R && c < C) tile[i][threadIdx.x] = x[(long long)r * C + c];
  }
  __syncthreads();
  for (int i = threadIdx.y; i < 32; i += 8) {
    int c = bx + i, r = by + threadIdx.x;
    if (r < R && c < C) y[(long long)c * R + r] = tile[threadIdx.x][i];
  }
}

__global__ void copy2d_k(const float* x, int ldx, float* y, int ldy, int rows, int cols) {
  long long total = (long long)rows * cols;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
       i += (long long)gridDim.x * blockDim.x) {
    int r = (int)(i / cols), c = (int)(i % cols);
    y[(long long)r * ldy + c] = x[(long long)r * ldx + c];
  }
}

__global__ void tile_k(const float* x, int n, float* y, int n_out) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n_out; i += (long long)gridDim.x * blockDim.x)
    y[i] = x[i % n];
}

}  // namespace

void rotary(cudaStream_t s, float* x, int T, int H, const float* freqs32) {
  long long n = (long long)T * H * 32;
  rotary_k<<<grid_cap(n), 256, 0, s>>>(x, T, H, freqs32);
  check(cudaGetLastError(), "rotary");
}
void fsmn(cudaStream_t s, const float* v, const float* w, float* out, int T, int C, int K) {
  long long n = (long long)T * C;
  fsmn_k<<<grid_cap(n), 256, 0, s>>>(v, w, out, T, C, K);
  check(cudaGetLastError(), "fsmn");
}
namespace {
// Row softmax in place, fp32 (max-subtracted), one block per row.
__global__ void softmax_rows_k(float* x, int cols) {
  __shared__ float red[32];
  float* p = x + (long long)blockIdx.x * cols;
  float m = -FLT_MAX;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) m = fmaxf(m, p[j]);
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    float mm = red[0];
    for (int i = 1; i < (int)(blockDim.x >> 5); ++i) mm = fmaxf(mm, red[i]);
    red[0] = mm;
  }
  __syncthreads();
  m = red[0];
  __syncthreads();
  float sum = 0.f;
  for (int j = threadIdx.x; j < cols; j += blockDim.x) {
    float e = expf(p[j] - m);
    p[j] = e;
    sum += e;
  }
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float ss = 0.f;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i) ss += red[i];
    red[0] = ss;
  }
  __syncthreads();
  float inv = 1.f / red[0];
  for (int j = threadIdx.x; j < cols; j += blockDim.x) p[j] *= inv;
}
}  // namespace

void attention_gemm(cudaStream_t s, const float* q, const float* k, const float* v, float* out, int T, int H,
                    float* scores) {
  const int ld = H * 64;
  // S[h] = Q_h K_h^T  (q, k already scaled by 64^-0.25 each)
  sgemm_batched(s, H, T, T, 64, q, ld, 64, k, ld, 64, true, scores, T, (long long)T * T);
  softmax_rows_k<<<H * T, 256, 0, s>>>(scores, T);
  check(cudaGetLastError(), "softmax");
  // O[h] = P[h] V_h
  sgemm_batched(s, H, T, 64, T, scores, T, (long long)T * T, v, ld, 64, false, out, ld, 64);
}

void attention(cudaStream_t s, const float* q, const float* k, const float* v, float* out, int T, int H) {
  dim3 grid((T + ATT_Q - 1) / ATT_Q, H);
  size_t shm = (size_t)ATT_Q * T * sizeof(float);
  if (shm > 48 * 1024) check(cudaFuncSetAttribute(attention_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm),
                             "attention smem");
  attention_k<<<grid, 256, shm, s>>>(q, k, v, out, T, H);
  check(cudaGetLastError(), "attention");
}
void fsq_codes(cudaStream_t s, const float* h, int T, int32_t* codes) {
  fsq_k<<<blocks_for(T, 128), 128, 0, s>>>(h, T, codes);
  check(cudaGetLastError(), "fsq");
}
void transpose(cudaStream_t s, const float* x, float* y, int R, int C) {
  dim3 grid((C + 31) / 32, (R + 31) / 32);
  transpose_k<<<grid, dim3(32, 8), 0, s>>>(x, y, R, C);
  check(cudaGetLastError(), "transpose");
}
void copy2d(cudaStream_t s, const float* x, int ldx, float* y, int ldy, int rows, int cols) {
  long long n = (long long)rows * cols;
  copy2d_k<<<grid_cap(n), 256, 0, s>>>(x, ldx, y, ldy, rows, cols);
  check(cudaGetLastError(), "copy2d");
}
void tile(cudaStream_t s, const float* x, int n, float* y, int n_out) {
  tile_k<<<grid_cap(n_out), 256, 0, s>>>(x, n, y, n_out);
  check(cudaGetLastError(), "tile");
}

}  // namespace kt2n::voice
