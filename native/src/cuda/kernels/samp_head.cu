// kt2n lane k4: the audio-only LM head (6,568 rows of the tied embedding).
//
// out[m, n] = bf16_rn( sum_k x[m, k] * w[n, k] ) with fp32 accumulation.
//
// Row invariance by construction: every output element is computed by one
// warp with the SAME instruction sequence whatever M is and whatever the
// other rows hold --
//   lane l owns k = kb + 8 l .. kb + 8 l + 7 for kb = 0, 256, 512, ...;
//   per lane: acc = fma(x[k], w[k], acc) in ascending k (explicit __fmaf_rn);
//   then an xor-butterfly over the 32 lanes (16, 8, 4, 2, 1);
// M, the number of columns a warp handles and the grid only decide WHICH warp
// computes an element, never HOW.  The kernel is weight-bandwidth bound
// (27 MB of bf16 per call), so all M rows share one pass over the weights.
#include "kt2n/kernels/samp.h"

#include <cuda_bf16.h>
#include <stdint.h>
#if defined(KT2N_HEAD_PRODUCTION_ORDER) && KT2N_HEAD_PRODUCTION_ORDER
#include <cublas_v2.h>
#include <limits.h>
#include <map>
#include <tuple>
#include <new>
#include <mutex>
#endif

namespace kt2n_samp {
namespace {

// CTA size is chosen at launch so that the column groups cover every SM in a
// single round (RTX PRO 6000: 188 SMs); 576 threads keeps <= 113 registers.
constexpr int kHeadMaxThreads = 576;

__device__ __forceinline__ void unpack8(const uint4 v, float (&f)[8]) {
  const unsigned int w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    f[2 * q] = __uint_as_float(w[q] << 16);              // element 2q (low half)
    f[2 * q + 1] = __uint_as_float(w[q] & 0xFFFF0000u);  // element 2q+1
  }
}

// x (M rows x K) is staged once per CTA in shared memory (M * K * 2 bytes,
// 64 KB at M = 16, K = 2048); weights stream from DRAM once, one k-block
// (256 columns of K) prefetched ahead.
template <int M, int CPW>
__global__ void __launch_bounds__(kHeadMaxThreads, 1)
samp_head_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx,
                 const __nv_bfloat16* __restrict__ w, int64_t ldw,
                 __nv_bfloat16* __restrict__ out, int64_t ldo, int N, int K) {
  extern __shared__ uint4 xs[];  // [M][K / 8]
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const int kv = K >> 3;
  for (int i = threadIdx.x; i < M * kv; i += blockDim.x) {
    const int m = i / kv, c = i - m * kv;
    xs[i] = __ldg(reinterpret_cast<const uint4*>(x + static_cast<int64_t>(m) * ldx) + c);
  }
  __syncthreads();
  const int groups = (N + CPW - 1) / CPW;
  for (int g = blockIdx.x * warps + warp; g < groups; g += gridDim.x * warps) {
    const int n0 = g * CPW;
    float acc[M][CPW];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int c = 0; c < CPW; ++c) acc[m][c] = 0.0f;

    const uint4* wcol[CPW];
#pragma unroll
    for (int c = 0; c < CPW; ++c) {
      const int n = (n0 + c < N) ? (n0 + c) : (N - 1);  // clamp: tail columns are computed, not stored
      wcol[c] = reinterpret_cast<const uint4*>(w + static_cast<int64_t>(n) * ldw) + lane;
    }
    uint4 wcur[CPW];
#pragma unroll
    for (int c = 0; c < CPW; ++c) wcur[c] = __ldcs(wcol[c]);

    for (int kb = 0; kb < K; kb += 256) {
      uint4 wnext[CPW];
      if (kb + 256 < K) {
#pragma unroll
        for (int c = 0; c < CPW; ++c) wnext[c] = __ldcs(wcol[c] + ((kb + 256) >> 3));
      }
      float wf[CPW][8];
#pragma unroll
      for (int c = 0; c < CPW; ++c) unpack8(wcur[c], wf[c]);
      const int xi = (kb >> 3) + lane;
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float xf[8];
        unpack8(xs[m * kv + xi], xf);
        // lane owns k = kb + 8 lane + j: acc = fma(x, w, acc), j ascending
#pragma unroll
        for (int c = 0; c < CPW; ++c)
#pragma unroll
          for (int j = 0; j < 8; ++j) acc[m][c] = __fmaf_rn(xf[j], wf[c][j], acc[m][c]);
      }
      if (kb + 256 < K) {
#pragma unroll
        for (int c = 0; c < CPW; ++c) wcur[c] = wnext[c];
      }
    }

#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int c = 0; c < CPW; ++c) {
        float v = acc[m][c];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
        acc[m][c] = v;  // identical in every lane (a + b == b + a)
      }

    // lane e % 32 stores element e = m * CPW + c
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int c = 0; c < CPW; ++c) {
        const int e = m * CPW + c;
        if ((e & 31) == lane && n0 + c < N)
          out[static_cast<int64_t>(m) * ldo + n0 + c] = __float2bfloat16_rn(acc[m][c]);
      }
  }
}

#ifndef KT2N_SAMP_HEAD_CPW_SMALL
#define KT2N_SAMP_HEAD_CPW_SMALL 2
#endif

template <int M>
int launch_head(const __nv_bfloat16* x, int64_t ldx, const __nv_bfloat16* w, int64_t ldw,
                __nv_bfloat16* out, int64_t ldo, int N, int K, cudaStream_t stream) {
  constexpr int CPW = (M <= 8) ? KT2N_SAMP_HEAD_CPW_SMALL : 2;
  const size_t smem = static_cast<size_t>(M) * K * sizeof(__nv_bfloat16);
  if (smem > 96 * 1024) return KT2N_SAMP_EARG;
  cudaError_t err = cudaFuncSetAttribute(samp_head_kernel<M, CPW>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem));
  if (err != cudaSuccess) return static_cast<int>(err);
  const int groups = (N + CPW - 1) / CPW;
  int device = 0, sms = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || sms < 1)
    sms = 1;
  // one CTA per SM (x is staged per CTA), as many warps as one round needs
  int warps = (groups + sms - 1) / sms;
  warps = warps < 4 ? 4 : (warps > kHeadMaxThreads / 32 ? kHeadMaxThreads / 32 : warps);
  int blocks = (groups + warps - 1) / warps;
  if (blocks > sms) blocks = sms;
  samp_head_kernel<M, CPW><<<blocks, warps * 32, smem, stream>>>(x, ldx, w, ldw, out, ldo, N, K);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace
}  // namespace kt2n_samp

#if defined(KT2N_HEAD_PRODUCTION_ORDER) && KT2N_HEAD_PRODUCTION_ORDER
namespace {
// Eager warm-up creates process-owned state before graph capture. Captured
// nodes retain its device addresses even after the capturing thread exits.
struct HeadHandles {
  struct Context {
    cublasHandle_t handle;
  };
  // Handles are warm-up state only. Every invocation owns separate scratch,
  // including invocations captured into independently replayable graphs.
  std::map<std::tuple<int, uintptr_t, int, int>, Context> devices;
  std::mutex mutex;
};

int production_head(const void* x, int64_t ldx, const void* w, int64_t ldw,
                    void* out, int64_t ldo, int M, int N, int K, cudaStream_t stream) {
  if (ldw > INT_MAX) return KT2N_SAMP_EARG;
  int device = 0;
  cudaError_t cuda_rc = cudaGetDevice(&device);
  if (cuda_rc != cudaSuccess) return static_cast<int>(cuda_rc);
  // Captured cuBLAS work must outlive the capturing host thread.
  // Bound the number of distinct stream/shape contexts; refuse new ones beyond
  // that bound instead of growing indefinitely in a long-lived worker.
  static HeadHandles* const handles = new (std::nothrow) HeadHandles;
  if (!handles) return static_cast<int>(cudaErrorMemoryAllocation);
  const std::lock_guard<std::mutex> lock(handles->mutex);
  const auto key = std::make_tuple(device, reinterpret_cast<uintptr_t>(stream), N, K);
  auto found = handles->devices.find(key);
  if (found == handles->devices.end()) {
    cudaStreamCaptureStatus capture;
    cuda_rc = cudaStreamIsCapturing(stream, &capture);
    if (cuda_rc != cudaSuccess) return static_cast<int>(cuda_rc);
    if (capture != cudaStreamCaptureStatusNone) return KT2N_SAMP_EARG;
    if (handles->devices.size() >= 64) return KT2N_SAMP_EARG;
    cublasHandle_t handle = nullptr;
    cublasStatus_t rc = cublasCreate(&handle);
    if (rc != CUBLAS_STATUS_SUCCESS) return static_cast<int>(rc);
    rc = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    if (rc != CUBLAS_STATUS_SUCCESS) {
      (void)cublasDestroy(handle);
      return static_cast<int>(rc);
    }
    HeadHandles::Context context{handle};
    try {
      found = handles->devices.emplace(key, context).first;
    } catch (const std::bad_alloc&) {
      (void)cublasDestroy(handle);
      return static_cast<int>(cudaErrorMemoryAllocation);
    }
  }
  cublasHandle_t handle = found->second.handle;
  cublasStatus_t rc = cublasSetStream(handle, stream);
  if (rc != CUBLAS_STATUS_SUCCESS) return static_cast<int>(rc);
  // cudaMallocAsync/cudaFreeAsync become allocation/free graph nodes during
  // capture. Scratch belongs to this invocation, not to a stream/shape key;
  // two distinct graphs replaying on different streams cannot overwrite it.
  // Match Torch 2.7.1's default workspace on sm120 (:4096:2:16:8).
  constexpr size_t blas_bytes = 4096 * 1024 * 2 + 16 * 1024 * 8;
  const size_t input_bytes = static_cast<size_t>(16) * K * sizeof(__nv_bfloat16);
  const size_t output_bytes = static_cast<size_t>(16) * N * sizeof(__nv_bfloat16);
  void* scratch = nullptr;
  cuda_rc = cudaMallocAsync(&scratch, blas_bytes + input_bytes + output_bytes, stream);
  if (cuda_rc != cudaSuccess) return static_cast<int>(cuda_rc);
  auto finish = [&](int result) {
    const cudaError_t free_rc = cudaFreeAsync(scratch, stream);
    return result != 0 ? result : static_cast<int>(free_rc);
  };
  rc = cublasSetWorkspace(handle, scratch, blas_bytes);
  if (rc != CUBLAS_STATUS_SUCCESS) return finish(static_cast<int>(rc));
  void* input_scratch = static_cast<char*>(scratch) + blas_bytes;
  void* output_scratch = static_cast<char*>(input_scratch) + input_bytes;
  const float alpha = 1.0f, beta = 0.0f;
  if (M >= 2) {
    // g83 enables TC_HEAD with TC_MIN_ROWS=2: all these buckets use an M=16
    // GEMM with unused rows zero. M=1 has a different cuBLAS reduction order.
    const size_t row_bytes = static_cast<size_t>(K) * sizeof(__nv_bfloat16);
    cuda_rc = cudaMemsetAsync(input_scratch, 0, 16 * row_bytes, stream);
    if (cuda_rc == cudaSuccess)
      cuda_rc = cudaMemcpy2DAsync(input_scratch, row_bytes, x,
                                  static_cast<size_t>(ldx) * sizeof(__nv_bfloat16),
                                  row_bytes, M, cudaMemcpyDeviceToDevice, stream);
    if (cuda_rc != cudaSuccess) return finish(static_cast<int>(cuda_rc));
    rc = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, 16, K, &alpha,
                      w, CUDA_R_16BF, static_cast<int>(ldw), input_scratch, CUDA_R_16BF, K,
                      &beta, output_scratch, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (rc != CUBLAS_STATUS_SUCCESS) return finish(static_cast<int>(rc));
    const size_t out_bytes = static_cast<size_t>(N) * sizeof(__nv_bfloat16);
    cuda_rc = cudaMemcpy2DAsync(out, static_cast<size_t>(ldo) * sizeof(__nv_bfloat16),
                                output_scratch, out_bytes, out_bytes, M,
                                cudaMemcpyDeviceToDevice, stream);
    return finish(static_cast<int>(cuda_rc));
  }
  // Production's la_head computes each row as an M=1 BF16 linear call.
  // W's row-major [N,K] bytes are column-major [K,N], hence op(W)=T.
  for (int row = 0; row < M; ++row) {
    const auto* input = static_cast<const __nv_bfloat16*>(x) + static_cast<int64_t>(row) * ldx;
    auto* output = static_cast<__nv_bfloat16*>(out) + static_cast<int64_t>(row) * ldo;
    rc = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, 1, K, &alpha,
                      w, CUDA_R_16BF, static_cast<int>(ldw), input, CUDA_R_16BF, K,
                      &beta, output, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (rc != CUBLAS_STATUS_SUCCESS) return finish(static_cast<int>(rc));
  }
  return finish(0);
}
}  // namespace
#endif

static int head_bf16_dispatch(const void* x, int64_t ldx, const void* w, int64_t ldw,
                              void* out, int64_t ldo, int M, int N, int K,
                              cudaStream_t stream, bool legacy_prefill) {
  using namespace kt2n_samp;
  if (!x || !w || !out || M < 1 || M > KT2N_SAMP_HEAD_MAX_ROWS || N < 1 || K < 256 ||
      (K % 256) != 0 || ldx < K || ldw < K || ldo < N)
    return KT2N_SAMP_EARG;
#if defined(KT2N_HEAD_PRODUCTION_ORDER) && KT2N_HEAD_PRODUCTION_ORDER
  if (static_cast<size_t>(M) * K * sizeof(__nv_bfloat16) > 96 * 1024) return KT2N_SAMP_EARG;
#endif
  if ((reinterpret_cast<uintptr_t>(x) & 15u) || (reinterpret_cast<uintptr_t>(w) & 15u) ||
      (ldx % 8) || (ldw % 8))
    return KT2N_SAMP_EALIGN;
  const auto* xb = static_cast<const __nv_bfloat16*>(x);
  const auto* wb = static_cast<const __nv_bfloat16*>(w);
  auto* ob = static_cast<__nv_bfloat16*>(out);
#if defined(KT2N_HEAD_PRODUCTION_ORDER) && KT2N_HEAD_PRODUCTION_ORDER
  if (!legacy_prefill) return production_head(xb, ldx, wb, ldw, ob, ldo, M, N, K, stream);
#else
  (void)legacy_prefill;
#endif
  switch (M) {
#define KT2N_SAMP_HEAD_CASE(R) \
  case R:                      \
    return launch_head<R>(xb, ldx, wb, ldw, ob, ldo, N, K, stream);
    KT2N_SAMP_HEAD_CASE(1)
    KT2N_SAMP_HEAD_CASE(2)
    KT2N_SAMP_HEAD_CASE(3)
    KT2N_SAMP_HEAD_CASE(4)
    KT2N_SAMP_HEAD_CASE(5)
    KT2N_SAMP_HEAD_CASE(6)
    KT2N_SAMP_HEAD_CASE(7)
    KT2N_SAMP_HEAD_CASE(8)
    KT2N_SAMP_HEAD_CASE(9)
    KT2N_SAMP_HEAD_CASE(10)
    KT2N_SAMP_HEAD_CASE(11)
    KT2N_SAMP_HEAD_CASE(12)
    KT2N_SAMP_HEAD_CASE(13)
    KT2N_SAMP_HEAD_CASE(14)
    KT2N_SAMP_HEAD_CASE(15)
    KT2N_SAMP_HEAD_CASE(16)
#undef KT2N_SAMP_HEAD_CASE
    default:
      return KT2N_SAMP_EARG;
  }
}

extern "C" int kt2n_samp_head_bf16(const void* x, int64_t ldx, const void* w, int64_t ldw,
                                   void* out, int64_t ldo, int M, int N, int K,
                                   cudaStream_t stream) {
  return head_bf16_dispatch(x, ldx, w, ldw, out, ldo, M, N, K, stream, false);
}

extern "C" int kt2n_samp_head_prefill_bf16(const void* x, int64_t ldx, const void* w, int64_t ldw,
                                           void* out, int64_t ldo, int M, int N, int K,
                                           cudaStream_t stream) {
  return head_bf16_dispatch(x, ldx, w, ldw, out, ldo, M, N, K, stream, true);
}
