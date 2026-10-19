// kt2n decode attention (lane k3-attn): one query token per row over the K/V cache.
//
// Grid (split, kv_head, row); 128 threads = 4 warps.  A CTA covers BOTH query heads of its
// K/V head, so every K/V byte is read once.  It walks the SPLIT_CHUNKS 64-key chunks of its
// split (K and V tiles in shared memory; next K prefetched during P.V, next V during the next
// Q.K) with tensor-core MMAs: warp w computes the scores of keys 16w .. 16w+15 and the output
// dims 32w .. 32w+31.  The query occupies tile rows 0 (head 0) and 8 (head 1) of a 16-row MMA
// tile, exactly the rows a prefill query occupies, so the per-element arithmetic is the
// prefill's.  A row whose keys fit one split writes its output directly; otherwise each split
// writes (m, l, o) to the workspace and the last split to arrive (one atomic counter per
// (row, kv_head), reset by that CTA) folds the split partials in split order.  Who arrives
// last never changes the arithmetic.  See attn.h for the numerics contract.
#include "attn_common.cuh"
#include "attn_production.cuh"

namespace kt2n_attn {

constexpr int DEC_THREADS = 128;
constexpr int PSTR = 132;  // floats per (split, head) partial: m, l, 0, 0, o[128]
constexpr size_t COUNTER_BYTES = 4096;
static_assert(KT2N_ATTN_MAX_ROWS * HKV * 4 <= (int)COUNTER_BYTES, "counter area");

__host__ __device__ inline int splits_for(int64_t keys) {
  return static_cast<int>((keys + SPLIT_KEYS - 1) / SPLIT_KEYS);
}

__device__ __forceinline__ uint32_t ld32(const __nv_bfloat16* p) { return *reinterpret_cast<const uint32_t*>(p); }

template <bool SLOTKV>
__global__ void __launch_bounds__(DEC_THREADS)
    decode_kernel(const __nv_bfloat16* __restrict__ q, int64_t q_row_stride,
                  const __nv_bfloat16* __restrict__ kc, const __nv_bfloat16* __restrict__ vc,
                  int64_t slot_stride, int64_t head_stride, const int64_t* __restrict__ positions,
                  const int64_t* __restrict__ cursor, const int32_t* __restrict__ row_slot,
                  const int32_t* __restrict__ kv_start, const int32_t* __restrict__ kv_len,
                  int grid_splits, __nv_bfloat16* __restrict__ out, int64_t out_row_stride,
                  int* __restrict__ counters, float* __restrict__ partials) {
  __shared__ __align__(16) uint4 sKV[2 * CHUNK * SEGS];  // K | V (swizzled); partials in the merge
  __shared__ __align__(16) float sP[GQA][CHUNK];         // probabilities of head 0 / head 1
  __shared__ float sMax[DEC_THREADS / 32][GQA];
  __shared__ int sLast;

  uint4* sK = sKV;
  uint4* sV = sKV + CHUNK * SEGS;
  pdl_wait();
  const int split = blockIdx.x, kvh = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;

  int64_t n, col0, slot;
  if constexpr (SLOTKV) {
    const int64_t cur = *cursor;
    const int64_t pos = imin64(positions[b], cur);
    n = pos + 1;
    col0 = cur - pos;
    slot = b;
  } else {
    n = kv_len[b];
    col0 = kv_start ? kv_start[b] : 0;
    slot = row_slot ? row_slot[b] : b;
  }
  __nv_bfloat16* orow = out + b * out_row_stride + kvh * GQA * D;
  if (n <= 0 || splits_for(n) > grid_splits) {
    // n <= 0: no keys -> zeros.  More keys than the grid covers: contract violation -> NaN.
    if (split == 0) {
      const float fill = n <= 0 ? 0.f : __int_as_float(0x7fc00000);
      for (int i = tid; i < GQA * D; i += DEC_THREADS) orow[i] = __float2bfloat16_rn(fill);
    }
    return;
  }
  const int nsplit = splits_for(n);
  if (split >= nsplit) return;

  const int64_t key0 = static_cast<int64_t>(split) * SPLIT_KEYS;
  const int64_t left = n - key0;
  const int nch = static_cast<int>(imin64(SPLIT_CHUNKS, (left + CHUNK - 1) / CHUNK));
  const int64_t base = slot * slot_stride + kvh * head_stride + (col0 + key0) * D;
  const __nv_bfloat16* kb = kc + base;
  const __nv_bfloat16* vb = vc + base;
  auto nvalid = [&](int c) { return static_cast<int>(imin64(CHUNK, left - static_cast<int64_t>(c) * CHUNK)); };
  auto koff = [&](int c) { return static_cast<int64_t>(c) * CHUNK * D; };

  load_tile<DEC_THREADS>(sK, kb, nvalid(0), tid);
  cp_async_commit();
  load_tile<DEC_THREADS>(sV, vb, nvalid(0), tid);
  cp_async_commit();

  // Query A fragments: tile row 0 = head 0, row 8 = head 1 (lanes g == 0); other rows zero.
  FragA qa[8];
  {
    const __nv_bfloat16* q0 = q + b * q_row_stride + kvh * GQA * D;
    const __nv_bfloat16* q1 = q0 + D;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const bool on = g == 0;
      qa[ks].x[0] = on ? ld32(q0 + 16 * ks + 2 * t) : 0u;
      qa[ks].x[1] = on ? ld32(q1 + 16 * ks + 2 * t) : 0u;
      qa[ks].x[2] = on ? ld32(q0 + 16 * ks + 8 + 2 * t) : 0u;
      qa[ks].x[3] = on ? ld32(q1 + 16 * ks + 8 + 2 * t) : 0u;
    }
  }

  // Running split state: m per head; O = this warp's dims 32w .. 32w+31 (4 tiles); l.
  float m0 = 0.f, m1 = 0.f;
  FragC o[4], lf;
  for (int c = 0; c < nch; ++c) {
    const int nv = nvalid(c);
    cp_async_wait<1>();
    __syncthreads();  // K(c) visible
    FragC s0, s1;
    qk_tile(s0, qa, sK, 2 * warp, lane);
    qk_tile(s1, qa, sK, 2 * warp + 1, lane);
    const int j0 = 16 * warp + 2 * t, j1 = j0 + 8;  // keys of s0.{c0,c1} / s1.{c0,c1} (+1 for c1)
    float sc[8] = {s0.x[0], s0.x[1], s0.x[2], s0.x[3], s1.x[0], s1.x[1], s1.x[2], s1.x[3]};
    const bool v[8] = {j0 < nv, j0 + 1 < nv, j0 < nv, j0 + 1 < nv, j1 < nv, j1 + 1 < nv, j1 < nv, j1 + 1 < nv};
#pragma unroll
    for (int i = 0; i < 8; ++i) sc[i] = v[i] ? score_of(sc[i]) : -INFINITY;
    const float w0 = quad_max(fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[4], sc[5])));  // head 0 (row 0)
    const float w1 = quad_max(fmaxf(fmaxf(sc[2], sc[3]), fmaxf(sc[6], sc[7])));  // head 1 (row 8)
    if (lane == 0) {
      sMax[warp][0] = w0;
      sMax[warp][1] = w1;
    }
    __syncthreads();  // every Q.K read of sK done; sMax visible
    if (c + 1 < nch) load_tile<DEC_THREADS>(sK, kb + koff(c + 1), nvalid(c + 1), tid);
    cp_async_commit();
    const float mc0 = fmaxf(fmaxf(sMax[0][0], sMax[1][0]), fmaxf(sMax[2][0], sMax[3][0]));
    const float mc1 = fmaxf(fmaxf(sMax[0][1], sMax[1][1]), fmaxf(sMax[2][1], sMax[3][1]));
    float a0 = 1.f, a1 = 1.f;
    if (c == 0) {
      m0 = mc0;
      m1 = mc1;
    } else {
      const float n0 = fmaxf(m0, mc0), n1 = fmaxf(m1, mc1);
      a0 = expf(__fsub_rn(m0, n0));
      a1 = expf(__fsub_rn(m1, n1));
      m0 = n0;
      m1 = n1;
    }
    if (g == 0) {
      sP[0][j0] = v[0] ? prob_of(sc[0], m0) : 0.f;
      sP[0][j0 + 1] = v[1] ? prob_of(sc[1], m0) : 0.f;
      sP[1][j0] = v[2] ? prob_of(sc[2], m1) : 0.f;
      sP[1][j0 + 1] = v[3] ? prob_of(sc[3], m1) : 0.f;
      sP[0][j1] = v[4] ? prob_of(sc[4], m0) : 0.f;
      sP[0][j1 + 1] = v[5] ? prob_of(sc[5], m0) : 0.f;
      sP[1][j1] = v[6] ? prob_of(sc[6], m1) : 0.f;
      sP[1][j1 + 1] = v[7] ? prob_of(sc[7], m1) : 0.f;
    }
    cp_async_wait<1>();
    __syncthreads();  // V(c) and sP visible
    if (c == 0) {
#pragma unroll
      for (int i = 0; i < 4; ++i) zero(o[i]);
      zero(lf);
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) scale_frag(o[i], a0, a1);
      scale_frag(lf, a0, a1);
    }
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      float p[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      if (g == 0) {
        const int k0 = 16 * ks + 2 * t, k1 = k0 + 8;
        p[0] = sP[0][k0];
        p[1] = sP[0][k0 + 1];
        p[2] = sP[1][k0];
        p[3] = sP[1][k0 + 1];
        p[4] = sP[0][k1];
        p[5] = sP[0][k1 + 1];
        p[6] = sP[1][k1];
        p[7] = sP[1][k1 + 1];
      }
      const PFrag pf = make_pfrag(p);
      pv_step<4>(o, pf, sV, ks, 4 * warp, lane);
      l_step(lf, pf);
    }
    __syncthreads();  // every P.V read of sV / sP done
    if (c + 1 < nch) load_tile<DEC_THREADS>(sV, vb + koff(c + 1), nvalid(c + 1), tid);
    cp_async_commit();
  }
  cp_async_wait<0>();
  const float l0 = lf.x[0], l1 = lf.x[2];  // valid in lanes g == 0

  if (nsplit == 1) {
    if (g == 0) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int d = 8 * (4 * warp + i) + 2 * t;
        *reinterpret_cast<uint32_t*>(orow + d) = finish2(o[i].x[0], o[i].x[1], l0);
        *reinterpret_cast<uint32_t*>(orow + D + d) = finish2(o[i].x[2], o[i].x[3], l1);
      }
    }
    return;
  }

  // Multi-split row: publish this split's partial, the last arriver folds them in order.
  const size_t row_parts = (static_cast<size_t>(b) * HKV + kvh) * grid_splits;
  {
    float* p0 = partials + ((row_parts + split) * GQA + 0) * PSTR;
    float* p1 = p0 + PSTR;
    if (tid == 0) {
      *reinterpret_cast<float4*>(p0) = make_float4(m0, l0, 0.f, 0.f);
      *reinterpret_cast<float4*>(p1) = make_float4(m1, l1, 0.f, 0.f);
    }
    if (g == 0) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int d = 8 * (4 * warp + i) + 2 * t;
        *reinterpret_cast<float2*>(p0 + 4 + d) = make_float2(o[i].x[0], o[i].x[1]);
        *reinterpret_cast<float2*>(p1 + 4 + d) = make_float2(o[i].x[2], o[i].x[3]);
      }
    }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const int prev = atomicAdd(&counters[b * HKV + kvh], 1);
    sLast = (prev == nsplit - 1);
  }
  __syncthreads();
  if (!sLast) return;
  __threadfence();

  // Merge: thread -> head hh = tid / 64, dims 2*(tid % 64), +1.
  const int hh = tid >> 6, dp = tid & 63;
  constexpr int CAP = static_cast<int>(sizeof(sKV) / (GQA * PSTR * sizeof(float)));
  float* sPart = reinterpret_cast<float*>(sKV);
  const float* rp = partials + row_parts * GQA * PSTR;
  float Fm = 0.f, Fl = 0.f, Fo0 = 0.f, Fo1 = 0.f;
  for (int s0 = 0; s0 < nsplit; s0 += CAP) {
    const int cnt = min(CAP, nsplit - s0);
    const int n4 = cnt * GQA * PSTR / 4;
    const float4* src = reinterpret_cast<const float4*>(rp + static_cast<size_t>(s0) * GQA * PSTR);
    float4* dst = reinterpret_cast<float4*>(sPart);
    for (int i = tid; i < n4; i += DEC_THREADS) dst[i] = __ldcg(src + i);
    __syncthreads();
    for (int k = 0; k < cnt; ++k) {
      const float* P = sPart + (k * GQA + hh) * PSTR;
      const float pm = P[0], pl = P[1];
      const float2 po = *reinterpret_cast<const float2*>(P + 4 + 2 * dp);
      if (s0 + k == 0) {
        Fm = pm;
        Fl = pl;
        Fo0 = po.x;
        Fo1 = po.y;
      } else {
        const FoldCoef f = fold_coef(Fm, pm);
        Fl = fold_val(Fl, f.a, pl, f.e);
        Fo0 = fold_val(Fo0, f.a, po.x, f.e);
        Fo1 = fold_val(Fo1, f.a, po.y, f.e);
        Fm = f.m;
      }
    }
    __syncthreads();
  }
  *reinterpret_cast<uint32_t*>(orow + hh * D + 2 * dp) = finish2(Fo0, Fo1, Fl);
  if (tid == 0) counters[b * HKV + kvh] = 0;
}

inline bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

inline void prefer_shared() {
  static const bool done = [] {
    cudaFuncSetAttribute(decode_kernel<true>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(decode_kernel<false>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaGetLastError();
    return true;
  }();
  (void)done;
}

template <bool SLOTKV>
int launch_decode(const void* q, int64_t q_row_stride, const void* k_cache, const void* v_cache,
                  int64_t kv_slot_stride, int64_t kv_head_stride, const int64_t* positions,
                  const int64_t* cursor, const int32_t* row_slot, const int32_t* kv_start,
                  const int32_t* kv_len, int rows, int max_ctx, void* out, int64_t out_row_stride,
                  void* workspace, size_t workspace_bytes, unsigned flags, cudaStream_t stream) {
  if (rows == 0) return 0;
  if (rows < 0 || rows > KT2N_ATTN_MAX_ROWS || max_ctx <= 0 || max_ctx > KT2N_ATTN_MAX_CTX)
    return KT2N_ATTN_EINVAL;
  if (!q || !k_cache || !v_cache || !out || !workspace) return KT2N_ATTN_EINVAL;
  if (SLOTKV ? (!positions || !cursor) : !kv_len) return KT2N_ATTN_EINVAL;
  if (!aligned16(q) || !aligned16(k_cache) || !aligned16(v_cache) || !aligned16(out) || !aligned16(workspace))
    return KT2N_ATTN_EINVAL;
  if ((q_row_stride % 8) || (out_row_stride % 8) || (kv_slot_stride % 8) || (kv_head_stride % 8))
    return KT2N_ATTN_EINVAL;
  if (workspace_bytes < kt2n_attn_decode_workspace_bytes(rows, max_ctx)) return KT2N_ATTN_EINVAL;
#if KT2N_ATTN_PRODUCTION_ORDER
  const int nch = (max_ctx + CHUNK - 1) / CHUNK;
  auto* parts = reinterpret_cast<float*>(static_cast<char*>(workspace) + COUNTER_BYTES);
  cudaError_t pe = launch_kernel(production::decode_partials<SLOTKV>, dim3(rows * HQ, nch), dim3(D),
      0, stream, (flags & KT2N_ATTN_FLAG_PDL) != 0,
      static_cast<const __nv_bfloat16*>(q), q_row_stride,
      static_cast<const __nv_bfloat16*>(k_cache), static_cast<const __nv_bfloat16*>(v_cache),
      kv_slot_stride, kv_head_stride, positions, cursor, row_slot, kv_start, kv_len, max_ctx, parts);
  if (pe != cudaSuccess) return static_cast<int>(pe);
  pe = launch_kernel(production::decode_combine<SLOTKV>, dim3(rows * HQ), dim3(D),
      0, stream, (flags & KT2N_ATTN_FLAG_PDL) != 0, parts, positions, cursor, row_slot, kv_start,
      kv_len, max_ctx, kv_head_stride, nch, static_cast<__nv_bfloat16*>(out), out_row_stride);
  return static_cast<int>(pe);
#else
  prefer_shared();
  const int gs = splits_for(max_ctx);
  int* counters = static_cast<int*>(workspace);
  float* partials = reinterpret_cast<float*>(static_cast<char*>(workspace) + COUNTER_BYTES);
  const cudaError_t e = launch_kernel(
      decode_kernel<SLOTKV>, dim3(gs, HKV, rows), dim3(DEC_THREADS), 0, stream, (flags & KT2N_ATTN_FLAG_PDL) != 0,
      static_cast<const __nv_bfloat16*>(q), q_row_stride, static_cast<const __nv_bfloat16*>(k_cache),
      static_cast<const __nv_bfloat16*>(v_cache), kv_slot_stride, kv_head_stride, positions, cursor, row_slot,
      kv_start, kv_len, gs, static_cast<__nv_bfloat16*>(out), out_row_stride, counters, partials);
  return static_cast<int>(e);
#endif
}

}  // namespace kt2n_attn

extern "C" {

int kt2n_attn_numerics_version(void) {
#if KT2N_ATTN_PRODUCTION_ORDER
  return 1003; // One 64-key partial per chunk, production scalar reduction and exp/div lowering.
#else
  return KT2N_ATTN_SPLIT_CHUNKS * 1000 + kt2n_attn::NUMERICS_REVISION;
#endif
}

size_t kt2n_attn_decode_workspace_bytes(int max_rows, int max_ctx) {
  using namespace kt2n_attn;
#if KT2N_ATTN_PRODUCTION_ORDER
  if (max_rows < 1 || max_rows > KT2N_ATTN_MAX_ROWS || max_ctx < 1 || max_ctx > KT2N_ATTN_MAX_CTX) return 0;
  return COUNTER_BYTES + static_cast<size_t>(max_rows) * HQ * ((max_ctx+CHUNK-1)/CHUNK) * 130 * sizeof(float);
#else
  if (max_rows < 1) max_rows = 1;
  if (max_ctx < 1) max_ctx = 1;
  const size_t parts = static_cast<size_t>(max_rows) * HKV * splits_for(max_ctx) * GQA * PSTR * sizeof(float);
  return COUNTER_BYTES + parts;
#endif
}

int kt2n_attn_workspace_init(void* workspace, size_t workspace_bytes, cudaStream_t stream) {
  if (!workspace || workspace_bytes < kt2n_attn::COUNTER_BYTES) return KT2N_ATTN_EINVAL;
  return static_cast<int>(cudaMemsetAsync(workspace, 0, workspace_bytes, stream));
}

int kt2n_attn_decode_slotkv(const void* q, int64_t q_row_stride, const void* k_cache, const void* v_cache,
                            int64_t kv_slot_stride, int64_t kv_head_stride, const int64_t* positions,
                            const int64_t* cursor, int rows, int max_ctx, void* out, int64_t out_row_stride,
                            void* workspace, size_t workspace_bytes, unsigned flags, cudaStream_t stream) {
  return kt2n_attn::launch_decode<true>(q, q_row_stride, k_cache, v_cache, kv_slot_stride, kv_head_stride,
                                        positions, cursor, nullptr, nullptr, nullptr, rows, max_ctx, out,
                                        out_row_stride, workspace, workspace_bytes, flags, stream);
}

int kt2n_attn_decode_rows(const void* q, int64_t q_row_stride, const void* k_cache, const void* v_cache,
                          int64_t kv_slot_stride, int64_t kv_head_stride, const int32_t* row_slot,
                          const int32_t* kv_start, const int32_t* kv_len, int rows, int max_kv_len, void* out,
                          int64_t out_row_stride, void* workspace, size_t workspace_bytes, unsigned flags,
                          cudaStream_t stream) {
  return kt2n_attn::launch_decode<false>(q, q_row_stride, k_cache, v_cache, kv_slot_stride, kv_head_stride,
                                         nullptr, nullptr, row_slot, kv_start, kv_len, rows, max_kv_len, out,
                                         out_row_stride, workspace, workspace_bytes, flags, stream);
}

}  // extern "C"
