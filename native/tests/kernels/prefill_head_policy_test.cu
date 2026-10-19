// Native prefill's head alone/burst regression. Synthetic BF16 operands only;
// this does not establish complete prefill or production audio identity.
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
#include "kt2n/kernels/samp.h"
#include "test_util.h"

namespace {
struct Buffer {
  void* p = nullptr;
  explicit Buffer(size_t bytes) {
    if (cudaMalloc(&p, bytes) != cudaSuccess) throw std::runtime_error("policy allocation failed");
  }
  ~Buffer() { if (p) cudaFree(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

TEST(PrefillPolicy, LegacyHeadIsBitwiseAloneAndBurst) {
  KT2N_GPU_OR_SKIP();
  constexpr int rows = 16, k = 2048, n = 6568;
  std::mt19937 rng(20261008);
  std::vector<uint16_t> x(rows*k), w(n*k), alone(rows*n), burst(rows*n);
  for (auto& v : x) { const auto sign=rng()&0x8000u; const auto mag=0x3e80u+rng()%0x200u; v=static_cast<uint16_t>(sign|mag); }
  for (auto& v : w) { const auto sign=rng()&0x8000u; const auto mag=0x3b00u+rng()%0x180u; v=static_cast<uint16_t>(sign|mag); }
  Buffer dx(x.size()*2), dw(w.size()*2), dy(alone.size()*2), db(burst.size()*2);
  ASSERT_EQ(cudaMemcpy(dx.p,x.data(),x.size()*2,cudaMemcpyHostToDevice),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dw.p,w.data(),w.size()*2,cudaMemcpyHostToDevice),cudaSuccess);
  for (int r=0;r<rows;r++) {
    ASSERT_EQ(kt2n_samp_head_prefill_bf16(static_cast<uint16_t*>(dx.p)+r*k,k,dw.p,k,
                                       static_cast<uint16_t*>(dy.p)+r*n,n,1,n,k,nullptr),0);
  }
  ASSERT_EQ(cudaMemcpy(alone.data(),dy.p,alone.size()*2,cudaMemcpyDeviceToHost),cudaSuccess);
  for (int count : {1,2,4,10,16}) {
    ASSERT_EQ(kt2n_samp_head_prefill_bf16(dx.p,k,dw.p,k,db.p,n,count,n,k,nullptr),0);
    ASSERT_EQ(cudaMemcpy(burst.data(),db.p,size_t(count)*n*2,cudaMemcpyDeviceToHost),cudaSuccess);
    EXPECT_EQ(std::memcmp(alone.data(),burst.data(),size_t(count)*n*2),0) << "rows=" << count;
  }
}
}  // namespace
