// Experimental production arithmetic; enabled only by an explicit qualification build.
// Observed Triton sm_120 layout: [1,8], [2,16], [4,1]. Separate RN operations,
// exp2.approx with production LOG2E bits and div.full preserve its lowering.
#pragma once
#ifndef KT2N_ATTN_PRODUCTION_ORDER
#define KT2N_ATTN_PRODUCTION_ORDER 0
#endif
#if KT2N_ATTN_PRODUCTION_ORDER
namespace kt2n_attn { namespace production {
__device__ __forceinline__ float production_exp(float x) {
  float y;
  x = __fmul_rn(x, __int_as_float(0x3fb8aa3b));
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float production_div(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ float fold_eight(const float* p) {
  const float a = __fadd_rn(p[0], p[1]);
  const float b = __fadd_rn(p[2], p[3]);
  const float c = __fadd_rn(p[4], p[5]);
  const float d = __fadd_rn(p[6], p[7]);
  return __fadd_rn(__fadd_rn(a,c), __fadd_rn(b,d));
}
struct Tile {float m,l,o;};
__device__ __forceinline__ Tile chunk_tile(const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, int64_t start, int64_t n, int chunk) {
  __shared__ float scores[64], probabilities[64], max_parts[8], sums[8];
  __shared__ float values[8][128];
  const int tid = threadIdx.x, r = tid / 16, d0 = (tid % 16) * 8;
  for (int i = 0; i < 8; ++i) {
    const int j = r + 8*i;
    const int64_t key = chunk * 64 + j;
    float products[8];
    for (int d = 0; d < 8; ++d) {
      const float kv = key < n ? __bfloat162float(k[(start + key)*128 + d0 + d]) : 0.f;
      const float qv = __bfloat162float(q[d0+d]);
      products[d] = __fmul_rn(kv,qv);
    }
    float dot = __fadd_rn(products[0], products[1]);
    for (int d = 2; d < 8; ++d) dot = __fadd_rn(products[d],dot);
    for (int offset = 8; offset; offset /= 2)
      dot = __fadd_rn(dot,__shfl_xor_sync(0xffffffffu,dot,offset));
    if ((tid % 16) == 0) scores[j] = key < n ? __fmul_rn(dot,0.08838834764831845f) : -INFINITY;
  }
  __syncthreads();
  if ((tid % 16) == 0) {
    float m = fmaxf(scores[r],scores[r+8]);
    for (int i = 2; i < 8; ++i) m = fmaxf(scores[r+8*i],m);
    max_parts[r] = m;
  }
  __syncthreads();
  const float m = fmaxf(fmaxf(fmaxf(max_parts[0],max_parts[1]),fmaxf(max_parts[4],max_parts[5])),
                       fmaxf(fmaxf(max_parts[2],max_parts[3]),fmaxf(max_parts[6],max_parts[7])));
  if (tid < 64) probabilities[tid] = chunk*64+tid < n ? production_exp(__fsub_rn(scores[tid],m)) : 0.f;
  __syncthreads();
  if ((tid % 16) == 0) {
    float l = __fadd_rn(probabilities[r],probabilities[r+8]);
    for (int i = 2; i < 8; ++i) l = __fadd_rn(probabilities[r+8*i],l);
    sums[r] = l;
  }
  for (int d = 0; d < 8; ++d) {
    float products[8];
    for (int i = 0; i < 8; ++i) {
      const int j = r+8*i;
      const int64_t key = chunk*64+j;
      const float vv = key < n ? __bfloat162float(v[(start+key)*128+d0+d]) : 0.f;
      products[i] = __fmul_rn(probabilities[j],vv);
    }
    float o = __fadd_rn(products[0],products[1]);
    for (int i = 2; i < 8; ++i) o = __fadd_rn(products[i],o);
    values[r][d0+d] = o;
  }
  __syncthreads();
  float vals[8];
  for (int i = 0; i < 8; ++i) vals[i] = values[i][tid];
  const Tile result{m,fold_eight(sums),fold_eight(vals)};
  __syncthreads();
  return result;
}

struct Range {int64_t n,start,slot; bool invalid;};
template<bool SLOTKV>
__device__ __forceinline__ Range range_for(int row,int max_ctx,const int64_t* positions,const int64_t* cursor,
    const int32_t* row_slot,const int32_t* kv_start,const int32_t* kv_len,
    int64_t head_stride) {
  Range r{};
  if constexpr(SLOTKV) {
    const int64_t cur=*cursor, pos=imin64(positions[row],cur);
    r.n=pos+1;r.start=cur-pos;r.slot=row;
    r.invalid=cur<0 || cur>=head_stride/D;
  } else {
    r.n=kv_len[row];r.start=kv_start?kv_start[row]:0;r.slot=row_slot?row_slot[row]:row;
  }
  if (r.n<=0) {r.invalid=false;return r;}
  r.invalid=r.invalid || r.slot<0 || r.start<0 || r.n>max_ctx ||
      (r.n>0 && (r.start>head_stride/D || r.n>head_stride/D-r.start));
  return r;
}
template<bool SLOTKV>
static __global__ void decode_partials(const __nv_bfloat16* q,int64_t q_stride,
    const __nv_bfloat16* k,const __nv_bfloat16* v,int64_t slot_stride,int64_t head_stride,
    const int64_t* positions,const int64_t* cursor,const int32_t* row_slot,
    const int32_t* kv_start,const int32_t* kv_len,int max_ctx,float* parts) {
  pdl_wait();
  const int row=blockIdx.x/HQ,head=blockIdx.x%HQ,chunk=blockIdx.y;
  const Range r=range_for<SLOTKV>(row,max_ctx,positions,cursor,row_slot,kv_start,kv_len,head_stride);
  if(r.invalid || r.n<=0 || chunk*CHUNK>=r.n) return;
  const int64_t base=r.slot*slot_stride+(head/GQA)*head_stride;
  const Tile t=chunk_tile(q+row*q_stride+head*D,k+base,v+base,r.start,r.n,chunk);
  float* dst=parts+(static_cast<int64_t>(blockIdx.x)*gridDim.y+chunk)*130;
  dst[2+threadIdx.x]=t.o;
  if(threadIdx.x==0) {dst[0]=t.m;dst[1]=t.l;}
}
template<bool SLOTKV>
static __global__ void decode_combine(const float* parts,const int64_t* positions,
    const int64_t* cursor,const int32_t* row_slot,const int32_t* kv_start,
    const int32_t* kv_len,int max_ctx,int64_t head_stride,int nch,
    __nv_bfloat16* out,int64_t out_stride) {
  pdl_wait();
  const int row=blockIdx.x/HQ,head=blockIdx.x%HQ,d=threadIdx.x;
  const Range r=range_for<SLOTKV>(row,max_ctx,positions,cursor,row_slot,kv_start,kv_len,head_stride);
  __nv_bfloat16* dest=out+row*out_stride+head*D+d;
  if(r.invalid || r.n<=0) {*dest=__float2bfloat16_rn(r.invalid?NAN:0.f);return;}
  const float* src=parts+static_cast<int64_t>(blockIdx.x)*nch*130;
  float m=src[0],l=src[1],o=src[2+d];
  for(int c=1;c<(r.n+CHUNK-1)/CHUNK;++c) {
    const float* s=src+c*130;const float mn=fmaxf(m,s[0]);
    const float a=production_exp(__fsub_rn(m,mn)),e=production_exp(__fsub_rn(s[0],mn));
    l=__fadd_rn(__fmul_rn(l,a),__fmul_rn(s[1],e));
    o=__fadd_rn(__fmul_rn(o,a),__fmul_rn(s[2+d],e));m=mn;
  }
  *dest=__float2bfloat16_rn(production_div(o,l));
}
static __global__ void prefill_sequential(const __nv_bfloat16* q,int64_t q_stride,
    const __nv_bfloat16* k,const __nv_bfloat16* v,int64_t slot_stride,int64_t head_stride,
    const kt2n_attn_prefill_row* desc,int max_keys,__nv_bfloat16* out,int64_t out_stride) {
  pdl_wait();
  const auto r=desc[blockIdx.z];const int query=blockIdx.x,head=blockIdx.y,d=threadIdx.x;
  if(query>=r.q_len) return;
  if(r.q_start<0) return; // No valid output address exists for a malformed descriptor.
  __nv_bfloat16* dest=out+(static_cast<int64_t>(r.q_start)+query)*out_stride+head*D+d;
  const int64_t n=static_cast<int64_t>(r.q_pos0)+query+1;
  if(r.q_pos0<0 || r.kv_slot<0 || r.kv_col0<0 || n>max_keys ||
      n>head_stride/D-r.kv_col0) {*dest=__float2bfloat16_rn(NAN);return;}
  const int64_t base=static_cast<int64_t>(r.kv_slot)*slot_stride+(head/GQA)*head_stride;
  const auto qp=q+(static_cast<int64_t>(r.q_start)+query)*q_stride+head*D;
  Tile t=chunk_tile(qp,k+base,v+base,r.kv_col0,n,0);
  for(int c=1;c<(n+CHUNK-1)/CHUNK;++c) {
    const Tile s=chunk_tile(qp,k+base,v+base,r.kv_col0,n,c);const float mn=fmaxf(t.m,s.m);
    const float a=production_exp(__fsub_rn(t.m,mn)),e=production_exp(__fsub_rn(s.m,mn));
    t.l=__fadd_rn(__fmul_rn(t.l,a),__fmul_rn(s.l,e));
    t.o=__fadd_rn(__fmul_rn(t.o,a),__fmul_rn(s.o,e));t.m=mn;
  }
  *dest=__float2bfloat16_rn(production_div(t.o,t.l));
}

// Forced split prefill parallelizes individual 64-key chunks. Reuse one
// 32-query tile's partial buffer between launches to bound scratch even for
// long prefixes. Combine walks chunks in the same order as the sequential
// path; tiling changes scheduling, never the arithmetic for a query.
constexpr int PREFILL_TILE = 32;
static __global__ void prefill_partials(const __nv_bfloat16* q,int64_t q_stride,
    const __nv_bfloat16* k,const __nv_bfloat16* v,int64_t slot_stride,int64_t head_stride,
    const kt2n_attn_prefill_row* desc,int max_keys,int query0,float* parts) {
  pdl_wait();
  const auto r=desc[blockIdx.z];
  const int query=query0+blockIdx.x,head=blockIdx.y%HQ,chunk=blockIdx.y/HQ;
  if(query>=r.q_len || r.q_start<0) return;
  const int64_t n=static_cast<int64_t>(r.q_pos0)+query+1;
  if(r.q_pos0<0 || r.kv_slot<0 || r.kv_col0<0 || n>max_keys ||
      n>head_stride/D-r.kv_col0 || chunk*CHUNK>=n) return;
  const int64_t base=static_cast<int64_t>(r.kv_slot)*slot_stride+(head/GQA)*head_stride;
  const auto qp=q+(static_cast<int64_t>(r.q_start)+query)*q_stride+head*D;
  const Tile t=chunk_tile(qp,k+base,v+base,r.kv_col0,n,chunk);
  const int nch=gridDim.y/HQ;
  float* dst=parts+((static_cast<int64_t>(blockIdx.z)*gridDim.x+blockIdx.x)*HQ+head)*nch*130+chunk*130;
  dst[2+threadIdx.x]=t.o;
  if(threadIdx.x==0) {dst[0]=t.m;dst[1]=t.l;}
}
static __global__ void prefill_combine(const float* parts,const kt2n_attn_prefill_row* desc,
    int max_keys,int64_t head_stride,int query0,int nch,__nv_bfloat16* out,int64_t out_stride) {
  pdl_wait();
  const auto r=desc[blockIdx.z];const int query=query0+blockIdx.x,head=blockIdx.y,d=threadIdx.x;
  if(query>=r.q_len || r.q_start<0) return;
  __nv_bfloat16* dest=out+(static_cast<int64_t>(r.q_start)+query)*out_stride+head*D+d;
  const int64_t n=static_cast<int64_t>(r.q_pos0)+query+1;
  if(r.q_pos0<0 || r.kv_slot<0 || r.kv_col0<0 || n>max_keys ||
      n>head_stride/D-r.kv_col0) {*dest=__float2bfloat16_rn(NAN);return;}
  const float* src=parts+((static_cast<int64_t>(blockIdx.z)*gridDim.x+blockIdx.x)*HQ+head)*nch*130;
  float m=src[0],l=src[1],o=src[2+d];
  for(int c=1;c<(n+CHUNK-1)/CHUNK;++c) {
    const float* s=src+c*130;const float mn=fmaxf(m,s[0]);
    const float a=production_exp(__fsub_rn(m,mn)),e=production_exp(__fsub_rn(s[0],mn));
    l=__fadd_rn(__fmul_rn(l,a),__fmul_rn(s[1],e));
    o=__fadd_rn(__fmul_rn(o,a),__fmul_rn(s[2+d],e));m=mn;
  }
  *dest=__float2bfloat16_rn(production_div(o,l));
}
}} // namespace kt2n_attn::production
#endif
