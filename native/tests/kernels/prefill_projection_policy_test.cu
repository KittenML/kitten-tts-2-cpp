// Fixed native-prefill projection arithmetic, independently of decode classes.
// Synthetic operands; this is not complete prefill/production audio evidence.
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
#include "kt2n/kernels/tern_proj.h"
#include "test_util.h"

namespace {
struct Buffer {
  void* p = nullptr;
  explicit Buffer(size_t bytes) {
    if (cudaMalloc(&p,bytes)!=cudaSuccess) throw std::runtime_error("projection policy allocation failed");
  }
  ~Buffer(){if(p)cudaFree(p);}
  Buffer(const Buffer&)=delete;
  Buffer& operator=(const Buffer&)=delete;
};
TEST(PrefillPolicy, ProjectionIsBitwiseAloneAndBurst) {
  KT2N_GPU_OR_SKIP();
  std::mt19937 rng(20261008);
  constexpr int rows=16;
  for(int family=0;family<4;family++) {
    const int k=family==KT2N_TERN_FAM_DOWN?6144:2048;
    const int n=family==KT2N_TERN_FAM_QKV?4096:family==KT2N_TERN_FAM_GATE_UP?12288:2048;
    std::vector<int32_t> codes(size_t(n)*k/16);
    std::vector<uint16_t> scales(size_t(n)*k/128),x(rows*k),res(rows*n);
    for(auto& value:codes){uint32_t bits=0;for(int j=0;j<16;j++)bits|=(rng()%3u)<<(2*j);std::memcpy(&value,&bits,4);}
    for(auto& value:scales)value=static_cast<uint16_t>(0x3b00u+rng()%0x180u);
    for(auto& value:x){const auto sign=rng()&0x8000u;const auto mag=0x3e80u+rng()%0x200u;value=static_cast<uint16_t>(sign|mag);}
    for(auto& value:res){const auto sign=rng()&0x8000u;const auto mag=0x3e00u+rng()%0x100u;value=static_cast<uint16_t>(sign|mag);}
    Buffer dc(codes.size()*4),ds(scales.size()*2),packed(kt2n_tern_codes_bytes(n,k)),ps(kt2n_tern_scales_bytes(n,k));
    Buffer dx(x.size()*2),dr(res.size()*2),alone(size_t(rows)*n*2),burst(size_t(rows)*n*2);
    ASSERT_EQ(cudaMemcpy(dc.p,codes.data(),codes.size()*4,cudaMemcpyHostToDevice),cudaSuccess);
    ASSERT_EQ(cudaMemcpy(ds.p,scales.data(),scales.size()*2,cudaMemcpyHostToDevice),cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dx.p,x.data(),x.size()*2,cudaMemcpyHostToDevice),cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dr.p,res.data(),res.size()*2,cudaMemcpyHostToDevice),cudaSuccess);
    ASSERT_EQ(kt2n_tern_repack(static_cast<int32_t*>(dc.p),ds.p,n,k,packed.p,ps.p,n,0,1,nullptr),0);
    for(int epi=0;epi<3;epi++) {
      if(epi==KT2N_TERN_EPI_SILU_MUL&&family!=KT2N_TERN_FAM_GATE_UP)continue;
      const int cols=epi==KT2N_TERN_EPI_SILU_MUL?n/2:n;
      std::vector<uint16_t> expected(rows*cols),actual(rows*cols);
      for(unsigned pdl: {0u,KT2N_TERN_FLAG_PDL}) {
        const unsigned flags=KT2N_TERN_FLAG_PREFILL|pdl;
        for(int r=0;r<rows;r++) {
          ASSERT_EQ(kt2n_tern_proj(family,static_cast<uint16_t*>(dx.p)+r*k,k,1,packed.p,ps.p,n,k,
                                  static_cast<uint16_t*>(alone.p)+r*cols,cols,epi,
                                  epi==KT2N_TERN_EPI_RESIDUAL?static_cast<uint16_t*>(dr.p)+r*n:nullptr,
                                  epi==KT2N_TERN_EPI_RESIDUAL?n:0,flags,nullptr),0);
        }
        ASSERT_EQ(cudaMemcpy(expected.data(),alone.p,expected.size()*2,cudaMemcpyDeviceToHost),cudaSuccess);
        for(int count: {1,2,4,10,16}) {
          ASSERT_EQ(kt2n_tern_proj(family,dx.p,k,count,packed.p,ps.p,n,k,burst.p,cols,epi,
                                  epi==KT2N_TERN_EPI_RESIDUAL?dr.p:nullptr,
                                  epi==KT2N_TERN_EPI_RESIDUAL?n:0,flags,nullptr),0);
          ASSERT_EQ(cudaMemcpy(actual.data(),burst.p,size_t(count)*cols*2,cudaMemcpyDeviceToHost),cudaSuccess);
          EXPECT_EQ(std::memcmp(expected.data(),actual.data(),size_t(count)*cols*2),0)
            <<"family="<<family<<" epilogue="<<epi<<" rows="<<count<<" PDL="<<pdl;
        }
      }
    }
  }
}
}  // namespace
