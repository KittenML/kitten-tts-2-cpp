// No model assets required. Covers the Ampere/Ada fallback and compares it to the cluster path where available.
#include "kt2n/cuda_caps.h"
#include "kt2n/bf16.h"
#include "../../src/cuda/kernels/ops_common.cuh"
#include "../../src/render/estimator/est_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace est = kt2n::render::est;
using bf16 = est::bf16;

#define CHECK(call) do { const auto err = (call); if (err != cudaSuccess) \
  throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(err)); } while (0)

template <typename T> struct Buffer {
  T* p = nullptr;
  size_t n;
  explicit Buffer(size_t count) : n(count) { CHECK(cudaMallocManaged(&p, n * sizeof(T))); }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  T& operator[](size_t i) { return p[i]; }
};

float rb(float v) { return kt2n::round_through_bf16(v); }
float fp(bf16 v) { return __bfloat162float(v); }
bf16 bf(float v) { return __float2bfloat16_rn(v); }
bool same(bf16 a, bf16 b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }

__global__ void pdl_value(int* values, int index) {
  kt2n_ops_detail::pdl_wait<true>();
  kt2n_ops_detail::pdl_trigger<true>();
  if (threadIdx.x == 0) values[index] = index ? values[index - 1] + 7 : 3;
}

void test_pdl(cudaStream_t stream) {
  Buffer<int> values(2);
  for (bool graph : {false, true}) {
    values[0] = values[1] = -1;
    if (graph) CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < 2; ++i)
      CHECK(kt2n_ops_detail::launch_k(pdl_value, dim3(1), dim3(32), 0, stream, 1, values.p, i));
    if (graph) {
      cudaGraph_t g;
      cudaGraphExec_t exec;
      CHECK(cudaStreamEndCapture(stream, &g));
      CHECK(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));
      CHECK(cudaGraphLaunch(exec, stream));
      CHECK(cudaStreamSynchronize(stream));
      CHECK(cudaGraphExecDestroy(exec));
      CHECK(cudaGraphDestroy(g));
    } else CHECK(cudaStreamSynchronize(stream));
    if (values[0] != 3 || values[1] != 10) throw std::runtime_error("PDL dependency failed");
  }
}

template <int MODE>
void launch_conv(est::ConvArgs a, int frames, bool cluster, cudaStream_t stream) {
  const size_t smem = est::conv_smem_bytes(a.c0 + a.c1, MODE == 1 ? a.res_c0 + a.res_c1 : 0, cluster ? 4 : 2);
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(frames / 16, 4);
  cfg.blockDim = dim3(128);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  if (cluster) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {1, 4, 1};
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    CHECK(cudaLaunchKernelEx(&cfg, est::est_conv_kernel<MODE>, a));
  } else {
    CHECK(cudaLaunchKernelEx(&cfg, est::est_conv_kernel<MODE, 2, true>, a));
    if constexpr (MODE != 2) {
      est::est_conv_portable_epilogue<MODE><<<frames / 16, 512, 0, stream>>>(a);
      CHECK(cudaPeekAtLastError());
    }
  }
}

template <int MODE>
void test_conv(int cin, int cres, bool clusters, cudaStream_t stream) {
  constexpr int frames = 48, channels = 256;
  Buffer<bf16> input(frames * cin), residual(frames * std::max(cres, 1));
  Buffer<bf16> weights(channels * 3 * cin), res_weights(channels * std::max(cres, 1));
  Buffer<bf16> bias(channels), gain(channels), offset(channels), temb(channels), res_bias(channels);
  Buffer<bf16> tail(2 * cin * 2), stage(2 * cin * 2), output(frames * channels), fast(frames * channels);
  Buffer<bf16> temp(frames * channels * 2);
  Buffer<est::EstRow> rows(2);
  Buffer<int32_t> row_map(frames);
  for (size_t i = 0; i < input.n; ++i) input[i] = bf(float(int(i * 17 % 31) - 15) / 16);
  for (size_t i = 0; i < residual.n; ++i) residual[i] = bf(float(int(i * 7 % 23) - 11) / 16);
  for (size_t i = 0; i < tail.n; ++i) tail[i] = bf(float(int(i * 11 % 17) - 8) / 16);
  for (size_t i = 0; i < weights.n; ++i) weights[i] = bf(0);
  for (size_t i = 0; i < res_weights.n; ++i) res_weights[i] = bf(0);
  for (int n = 0; n < channels; ++n) {
    for (int j = 0; j < 3; ++j)
      weights[(size_t(n) * 3 + j) * cin + (n * 7 + j * 31) % cin] = bf(float(j - 1) / 4);
    if (cres) res_weights[size_t(n) * cres + (n * 13) % cres] = bf(0.5f);
    bias[n] = bf(float(n % 7 - 3) / 32);
    gain[n] = bf(1.f + float(n % 5) / 8);
    offset[n] = bf(float(n % 3 - 1) / 16);
    temb[n] = bf(float(n % 11 - 5) / 8);
    res_bias[n] = bf(float(n % 9 - 4) / 32);
  }
  rows[0] = {0, 0, 16, 13, 0, 0, 7, 0};
  rows[1] = {16, 50, 16, 16, 0, 1, 16, 0};
  for (int f = 0; f < frames; ++f) row_map[f] = f < 32 ? f / 16 : -1;
  est::ConvArgs a{};
  a.in0 = input.p; a.c0 = cin == 512 ? 256 : cin; a.ld0 = cin;
  a.in1 = cin == 512 ? input.p + 256 : nullptr; a.c1 = cin - a.c0; a.ld1 = cin;
  a.w = weights.p; a.bias = bias.p; a.ln_g = gain.p; a.ln_b = offset.p; a.temb = temb.p;
  a.res_in0 = residual.p; a.res_c0 = cres == 512 ? 256 : cres; a.res_ld0 = cres;
  a.res_in1 = cres == 512 ? residual.p + 256 : nullptr; a.res_c1 = cres - a.res_c0; a.res_ld1 = cres;
  a.res_w = res_weights.p; a.res_b = res_bias.p;
  a.out = output.p; a.ld_out = channels; a.erows = rows.p; a.frame_row = row_map.p;
  a.tail_state = tail.p; a.tail_slot_stride = cin * 2; a.tail_stage = stage.p; a.stage_row_stride = cin * 2;
  a.portable_values = temp.p; a.portable_residual = temp.p + frames * channels;
  CHECK(cudaFuncSetAttribute(est::est_conv_kernel<MODE, 2, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            int(est::conv_smem_bytes(cin, MODE == 1 ? cres : 0, 2))));
  if (clusters) CHECK(cudaFuncSetAttribute(est::est_conv_kernel<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         int(est::conv_smem_bytes(cin, MODE == 1 ? cres : 0))));
  launch_conv<MODE>(a, frames, false, stream);
  CHECK(cudaStreamSynchronize(stream));
  auto read_input = [&](int row, int t, int c) {
    if (row < 0 || t >= rows[row].valid) return 0.f;
    if (t >= 0) return fp(input[size_t(rows[row].fr_off + t) * cin + c]);
    return rows[row].past ? fp(tail[size_t(rows[row].slot) * cin * 2 + c * 2 + t + 2]) : 0.f;
  };
  for (int f = 0; f < frames; ++f) {
    const int row = row_map[f], t = row < 0 ? 0 : f - rows[row].fr_off;
    std::vector<float> v(channels), rv(channels);
    float sum = 0;
    for (int n = 0; n < channels; ++n) {
      float acc = 0;
      for (int j = 0; j < 3; ++j)
        acc += read_input(row, t - 2 + j, (n * 7 + j * 31) % cin) * float(j - 1) / 4;
      v[n] = rb(acc + fp(bias[n]));
      sum += v[n];
      if constexpr (MODE == 1) {
        const float r = row >= 0 && t < rows[row].valid ? fp(residual[size_t(f) * cres + (n * 13) % cres]) : 0.f;
        rv[n] = rb(r * 0.5f + fp(res_bias[n]));
      }
      const bf16 raw = MODE == 2 ? output[f * channels + n] : temp[f * channels + n];
      if (!same(raw, bf(v[n]))) throw std::runtime_error("raw convolution differs from exact CPU oracle");
      if constexpr (MODE == 1)
        if (!same(temp[frames * channels + f * channels + n], bf(rv[n])))
          throw std::runtime_error("residual convolution differs from exact CPU oracle");
    }
    const float mean = sum / channels;
    float variance = 0;
    for (float x : v) variance += (x - mean) * (x - mean);
    const float inv_std = 1.f / std::sqrt(variance / channels + 1e-5f);
    for (int n = 0; n < channels; ++n) {
      float expected = v[n];
      if constexpr (MODE != 2) {
        expected = rb((expected - mean) * inv_std * fp(gain[n]) + fp(offset[n]));
        expected = rb(expected * std::tanh(std::log1p(std::exp(expected))));
        expected *= row >= 0 && t < rows[row].valid ? 1.f : 0.f;
        if constexpr (MODE == 0) expected = rb(expected + fp(temb[n]));
        if constexpr (MODE == 1) expected = rb(expected + rv[n]);
      }
      const float actual = fp(output[f * channels + n]);
      if (!std::isfinite(actual) || std::abs(actual - expected) > 0.016f * std::max(1.f, std::abs(expected)))
        throw std::runtime_error("normalized output differs from CPU oracle");
    }
  }
  for (int row = 0; row < 2; ++row)
    for (int c = 0; c < cin; ++c)
      for (int j = 0; j < 2; ++j)
        if (!same(stage[(row * cin + c) * 2 + j], bf(read_input(row, rows[row].shift - 2 + j, c))))
          throw std::runtime_error("convolution tail state mismatch");

  std::vector<bf16> reference(output.p, output.p + output.n);
  launch_conv<MODE>(a, 16, false, stream);
  CHECK(cudaStreamSynchronize(stream));
  if (std::memcmp(reference.data(), output.p, 16 * channels * sizeof(bf16)))
    throw std::runtime_error("batch size changed convolution output");
  for (size_t i = 0; i < output.n; ++i) output[i] = bf(NAN);
  CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  launch_conv<MODE>(a, frames, false, stream);
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CHECK(cudaStreamEndCapture(stream, &graph));
  CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  CHECK(cudaGraphLaunch(exec, stream));
  CHECK(cudaStreamSynchronize(stream));
  if (std::memcmp(reference.data(), output.p, output.n * sizeof(bf16)))
    throw std::runtime_error("graph replay changed convolution output");
  CHECK(cudaGraphExecDestroy(exec));
  CHECK(cudaGraphDestroy(graph));
  if (clusters) {
    a.out = fast.p;
    launch_conv<MODE>(a, frames, true, stream);
    CHECK(cudaStreamSynchronize(stream));
    if (std::memcmp(reference.data(), fast.p, fast.n * sizeof(bf16)))
      throw std::runtime_error("portable and clustered output differ");
  }
  std::printf("  conv mode=%d cin=%d cres=%d: CPU oracle, tails, graph%s passed\n",
              MODE, cin, cres, clusters ? ", cluster parity" : "");
}

int main() {
  try {
    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    if (!count) throw std::runtime_error("no CUDA devices");
    for (int device = 0; device < count; ++device) {
      CHECK(cudaSetDevice(device));
      cudaDeviceProp prop{};
      CHECK(cudaGetDeviceProperties(&prop, device));
      const auto caps = kt2n::cuda::current_caps();
      if (caps.major != prop.major || caps.minor != prop.minor || caps.pdl() != (prop.major >= 9))
        throw std::runtime_error("device capability cache mismatch");
      if (prop.major < 8) throw std::runtime_error("SM 80 or newer is required");
      std::printf("%s sm_%d%d: PDL=%d clusters=%d\n", prop.name, prop.major, prop.minor, caps.pdl(), caps.clusters);
      cudaStream_t stream;
      CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      test_pdl(stream);
      test_conv<0>(320, 0, caps.clusters, stream);
      test_conv<0>(512, 0, caps.clusters, stream);
      test_conv<1>(256, 256, caps.clusters, stream);
      test_conv<1>(256, 512, caps.clusters, stream);
      test_conv<2>(256, 0, caps.clusters, stream);
      test_conv<3>(256, 0, caps.clusters, stream);
      CHECK(cudaStreamDestroy(stream));
    }
    CHECK(cudaSetDevice(0));
    if (kt2n::cuda::current_caps().major != kt2n::cuda::device_caps(0).major)
      throw std::runtime_error("device switch invalidated capability cache");
    std::puts("GPU portability tests passed");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
