// GPU smoke kernel: unpack pack_exact codes + scales to bf16, bit-identical to
// kt2n::ref::unpack_ternary. One thread per 32-bit code word (16 weights).
#include <cuda_runtime.h>
#include <stdint.h>

#include <cstring>

#include "kt2n/cuda_smoke.h"

namespace {

constexpr int64_t kGroup = 128;
constexpr int64_t kCodesPerWord = 16;

__global__ void unpack_ternary_kernel(const uint32_t* __restrict__ codes, const uint16_t* __restrict__ scales,
                                      int64_t n, int64_t k, uint16_t* __restrict__ out) {
  const int64_t words_per_row = k / kCodesPerWord;
  const int64_t total = n * words_per_row;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = idx / words_per_row;
    const int64_t j = idx - row * words_per_row;
    const uint32_t word = codes[idx];
    const uint16_t s = scales[row * (k / kGroup) + (j * kCodesPerWord) / kGroup];
    uint16_t* dst = out + row * k + j * kCodesPerWord;
#pragma unroll
    for (int o = 0; o < 16; ++o) {
      const uint32_t c = (word >> (2 * o)) & 3u;
      dst[o] = c == 1u ? s : (c == 2u ? static_cast<uint16_t>(s ^ 0x8000u) : static_cast<uint16_t>(0));
    }
  }
}

}  // namespace

extern "C" int kt2n_cuda_device_count(void) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    (void)cudaGetLastError();  // clear the sticky-free error state
    return 0;
  }
  return count;
}

extern "C" int kt2n_cuda_runtime_version(void) {
  int v = 0;
  return cudaRuntimeGetVersion(&v) == cudaSuccess ? v : 0;
}

extern "C" int kt2n_cuda_driver_version(void) {
  int v = 0;
  return cudaDriverGetVersion(&v) == cudaSuccess ? v : 0;
}

extern "C" int kt2n_cuda_device_info(int dev, int* major, int* minor, char* name, int name_len) {
  cudaDeviceProp prop{};
  const cudaError_t err = cudaGetDeviceProperties(&prop, dev);
  if (err != cudaSuccess) return static_cast<int>(err);
  if (major != nullptr) *major = prop.major;
  if (minor != nullptr) *minor = prop.minor;
  if (name != nullptr && name_len > 0) {
    std::strncpy(name, prop.name, static_cast<size_t>(name_len) - 1);
    name[name_len - 1] = '\0';
  }
  return 0;
}

extern "C" int kt2n_cuda_unpack_ternary_launch(const int32_t* d_codes, const uint16_t* d_scales, int64_t n,
                                               int64_t k, uint16_t* d_out, void* stream) {
  if (n < 0 || k <= 0 || k % kGroup != 0) return static_cast<int>(cudaErrorInvalidValue);
  const int64_t total = n * (k / kCodesPerWord);
  if (total == 0) return 0;
  constexpr int kThreads = 256;
  const int64_t want = (total + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(want < 65535 ? want : 65535);
  unpack_ternary_kernel<<<blocks, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const uint32_t*>(d_codes), d_scales, n, k, d_out);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int kt2n_cuda_smoke_unpack(const int32_t* codes, const uint16_t* scales, int64_t n, int64_t k,
                                      uint16_t* out) {
  if (n < 0 || k <= 0 || k % kGroup != 0) return static_cast<int>(cudaErrorInvalidValue);
  const size_t codes_bytes = static_cast<size_t>(n) * static_cast<size_t>(k / kCodesPerWord) * sizeof(int32_t);
  const size_t scales_bytes = static_cast<size_t>(n) * static_cast<size_t>(k / kGroup) * sizeof(uint16_t);
  const size_t out_bytes = static_cast<size_t>(n) * static_cast<size_t>(k) * sizeof(uint16_t);
  int32_t* d_codes = nullptr;
  uint16_t* d_scales = nullptr;
  uint16_t* d_out = nullptr;
  cudaError_t err = cudaSuccess;
  auto step = [&err](cudaError_t e) {
    if (err == cudaSuccess) err = e;
    return err == cudaSuccess;
  };
  if (step(cudaMalloc(&d_codes, codes_bytes > 0 ? codes_bytes : 1)) &&
      step(cudaMalloc(&d_scales, scales_bytes > 0 ? scales_bytes : 1)) &&
      step(cudaMalloc(&d_out, out_bytes > 0 ? out_bytes : 1)) &&
      step(cudaMemcpy(d_codes, codes, codes_bytes, cudaMemcpyHostToDevice)) &&
      step(cudaMemcpy(d_scales, scales, scales_bytes, cudaMemcpyHostToDevice)) &&
      step(cudaMemset(d_out, 0xAB, out_bytes)) &&
      step(static_cast<cudaError_t>(kt2n_cuda_unpack_ternary_launch(d_codes, d_scales, n, k, d_out, nullptr))) &&
      step(cudaDeviceSynchronize())) {
    step(cudaMemcpy(out, d_out, out_bytes, cudaMemcpyDeviceToHost));
  }
  if (d_codes != nullptr) cudaFree(d_codes);
  if (d_scales != nullptr) cudaFree(d_scales);
  if (d_out != nullptr) cudaFree(d_out);
  return static_cast<int>(err);
}

extern "C" const char* kt2n_cuda_error_string(int err) {
  return cudaGetErrorString(static_cast<cudaError_t>(err));
}
