// prof-prefill variant (KT2N_PF_FLAG_ATTN_Q8): prefill attention as k3's DECODE kernel generalised to a tile of up
// to 8 consecutive queries of one row (attn_decode.cu decode_kernel, same helpers from attn_common.cuh).
//
// Why: for a single request (one row, 14..150 new tokens) k3's prefill kernel runs ceil(q/32) x 8 x splits CTAs
// (24..160 on 188 SMs) whose 4 warps each walk 8 queries x 64 keys x 128 dims serially (Nsight Compute: 0.06..0.4
// waves, latency-bound, the largest kernel of a layer).  The decode kernel instead spreads one query over 4 warps
// (warp w: keys 16w..16w+15 of a chunk for Q.K, dims 32w..32w+31 for P.V); here the same CTA serves 8 queries at
// once (every MMA row is used: row g = query g head 0, row g + 8 = query g head 1 -- the rows prefill uses), so a
// K/V chunk is loaded once per 8 queries and the grid is ceil(q/8) x 8 x splits CTAs.
//
// Numerics: per query row, operand for operand the decode kernel's arithmetic (k3 proves decode == prefill bitwise
// at every position): Q.K chain per score tile (qk_tile), max over the chunk (exact), m update / rescale /
// probabilities (prob_of), the hi/mid/lo P.V and l chains (pv_step, l_step) in the same order, split partials
// folded in split order (fold_coef / fold_val), finish2.  A query whose keys end inside the split keeps going
// through later chunks with P = 0 and scale 1 (like k3's prefill kernel) but its state is captured at its own
// last chunk, so nothing after it can touch its result.
#include "attn_common.cuh"
#include "pf_attn.h"

namespace kt2n_pf_attn {

using namespace kt2n_attn;

constexpr int THREADS = 128;
constexpr int ROWS = 16;   // MMA rows: query g head 0 -> row g, head 1 -> row g + 8
constexpr int PSTR = 132;  // floats per (split, row) partial: m, l, 0, 0, o[128]
constexpr size_t COUNTER_BYTES = 4096;

__host__ __device__ inline int splits_for(int64_t keys) {
  return static_cast<int>((keys + SPLIT_KEYS - 1) / SPLIT_KEYS);
}

__device__ __forceinline__ uint32_t ld32(const __nv_bfloat16* p) { return *reinterpret_cast<const uint32_t*>(p); }

// SEQ = false: grid (split, kv head, tile), one CTA per K split; queries spanning several splits publish (m, l, o)
// partials and the last split CTA of the tile folds them in split order.  SEQ = true: grid (1, kv head, tile), the
// CTA walks every split in order and folds in registers (no workspace) -- k3's two prefill shapes, same bits.
template <bool SEQ>
__global__ void __launch_bounds__(THREADS)
    q8_kernel(const __nv_bfloat16* __restrict__ q, int64_t q_row_stride, const __nv_bfloat16* __restrict__ kc,
              const __nv_bfloat16* __restrict__ vc, int64_t slot_stride, int64_t head_stride,
              const kt2n_pf_q8_tile* __restrict__ tiles, int grid_splits, __nv_bfloat16* __restrict__ out,
              int64_t out_row_stride, int* __restrict__ counters, float* __restrict__ partials) {
  __shared__ __align__(16) uint4 sKV[2 * CHUNK * SEGS];  // K | V (swizzled)
  __shared__ __align__(16) float sP[ROWS][CHUNK];        // probabilities per MMA row
  __shared__ float sMax[THREADS / 32][ROWS];
  __shared__ int sLast;
  uint4* sK = sKV;
  uint4* sV = sKV + CHUNK * SEGS;
  pdl_wait();
  const int split = SEQ ? 0 : static_cast<int>(blockIdx.x), kvh = blockIdx.y, tix = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const kt2n_pf_q8_tile td = tiles[tix];
  const int64_t n_max = static_cast<int64_t>(td.pos0) + td.nq;  // keys of the tile's last query
  const int nsplit_tile = splits_for(n_max);
  if constexpr (!SEQ) {
    if (nsplit_tile > grid_splits) {  // contract violation: NaN, loudly
      if (split == 0)
        for (int i = tid; i < td.nq * GQA * D; i += THREADS) {
          const int qi = i / (GQA * D), d = i % (GQA * D);
          out[static_cast<int64_t>(td.q_start + qi) * out_row_stride + kvh * GQA * D + d] =
              __float2bfloat16_rn(__int_as_float(0x7fc00000));
        }
      return;
    }
    if (split >= nsplit_tile) return;
  }
  const int nch_tile = static_cast<int>((n_max + CHUNK - 1) / CHUNK);
  const int c_begin = SEQ ? 0 : split * SPLIT_CHUNKS;
  const int c_end = SEQ ? nch_tile : min(nch_tile, c_begin + SPLIT_CHUNKS);
  const int64_t base = static_cast<int64_t>(td.slot) * slot_stride + kvh * head_stride;
  const __nv_bfloat16* kb = kc + base;
  const __nv_bfloat16* vb = vc + base;
  auto nvalid = [&](int c) { return static_cast<int>(imin64(CHUNK, n_max - static_cast<int64_t>(c) * CHUNK)); };
  auto koff = [&](int c) { return static_cast<int64_t>(c) * CHUNK * D; };

  load_tile<THREADS>(sK, kb + koff(c_begin), nvalid(c_begin), tid);
  cp_async_commit();
  load_tile<THREADS>(sV, vb + koff(c_begin), nvalid(c_begin), tid);
  cp_async_commit();

  // This lane's query (tile rows g / g + 8).
  const bool qvalid = g < td.nq;
  const int64_t n_g = qvalid ? static_cast<int64_t>(td.pos0) + g + 1 : 0;  // keys query g attends
  const int qnch = static_cast<int>((n_g + CHUNK - 1) / CHUNK);              // chunks it attends
  const int nsplit_g = splits_for(n_g);

  FragA qa[8];
  {
    const __nv_bfloat16* q0 = q + static_cast<int64_t>(td.q_start + (qvalid ? g : 0)) * q_row_stride + kvh * GQA * D;
    const __nv_bfloat16* q1 = q0 + D;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      qa[ks].x[0] = qvalid ? ld32(q0 + 16 * ks + 2 * t) : 0u;
      qa[ks].x[1] = qvalid ? ld32(q1 + 16 * ks + 2 * t) : 0u;
      qa[ks].x[2] = qvalid ? ld32(q0 + 16 * ks + 8 + 2 * t) : 0u;
      qa[ks].x[3] = qvalid ? ld32(q1 + 16 * ks + 8 + 2 * t) : 0u;
    }
  }

  float m0 = 0.f, m1 = 0.f;
  FragC o[4], lf;
  FragC Fo[4];
  float Fm0 = 0.f, Fm1 = 0.f, Fl0 = 0.f, Fl1 = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) zero(Fo[i]);
  for (int c = c_begin; c < c_end; ++c) {
    const int nv = static_cast<int>(imax64(0, imin64(CHUNK, n_g - static_cast<int64_t>(c) * CHUNK)));
    const bool first = (c % SPLIT_CHUNKS) == 0;  // a split starts: decode's chunk 0
    cp_async_wait<1>();
    __syncthreads();  // K(c) visible
    FragC s0, s1;
    qk_tile(s0, qa, sK, 2 * warp, lane);
    qk_tile(s1, qa, sK, 2 * warp + 1, lane);
    const int j0 = 16 * warp + 2 * t, j1 = j0 + 8;
    float sc[8] = {s0.x[0], s0.x[1], s0.x[2], s0.x[3], s1.x[0], s1.x[1], s1.x[2], s1.x[3]};
    const bool v[8] = {j0 < nv, j0 + 1 < nv, j0 < nv, j0 + 1 < nv, j1 < nv, j1 + 1 < nv, j1 < nv, j1 + 1 < nv};
#pragma unroll
    for (int i = 0; i < 8; ++i) sc[i] = v[i] ? score_of(sc[i]) : -INFINITY;
    const float w0 = quad_max(fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[4], sc[5])));  // row g (head 0)
    const float w1 = quad_max(fmaxf(fmaxf(sc[2], sc[3]), fmaxf(sc[6], sc[7])));  // row g + 8 (head 1)
    if (t == 0) {
      sMax[warp][g] = w0;
      sMax[warp][g + 8] = w1;
    }
    __syncthreads();  // every Q.K read of sK done; sMax visible
    if (c + 1 < c_end) load_tile<THREADS>(sK, kb + koff(c + 1), nvalid(c + 1), tid);
    cp_async_commit();
    const float mc0 = fmaxf(fmaxf(sMax[0][g], sMax[1][g]), fmaxf(sMax[2][g], sMax[3][g]));
    const float mc1 = fmaxf(fmaxf(sMax[0][g + 8], sMax[1][g + 8]), fmaxf(sMax[2][g + 8], sMax[3][g + 8]));
    const bool part = nv > 0;  // the query has keys in this chunk
    float a0 = 1.f, a1 = 1.f;
    if (part) {
      if (first) {
        m0 = mc0;
        m1 = mc1;
      } else {
        const float n0 = fmaxf(m0, mc0), n1 = fmaxf(m1, mc1);
        a0 = expf(__fsub_rn(m0, n0));
        a1 = expf(__fsub_rn(m1, n1));
        m0 = n0;
        m1 = n1;
      }
    }
    sP[g][j0] = v[0] ? prob_of(sc[0], m0) : 0.f;
    sP[g][j0 + 1] = v[1] ? prob_of(sc[1], m0) : 0.f;
    sP[g + 8][j0] = v[2] ? prob_of(sc[2], m1) : 0.f;
    sP[g + 8][j0 + 1] = v[3] ? prob_of(sc[3], m1) : 0.f;
    sP[g][j1] = v[4] ? prob_of(sc[4], m0) : 0.f;
    sP[g][j1 + 1] = v[5] ? prob_of(sc[5], m0) : 0.f;
    sP[g + 8][j1] = v[6] ? prob_of(sc[6], m1) : 0.f;
    sP[g + 8][j1 + 1] = v[7] ? prob_of(sc[7], m1) : 0.f;
    cp_async_wait<1>();
    __syncthreads();  // V(c) and sP visible
    if (first) {
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
      const int k0 = 16 * ks + 2 * t, k1 = k0 + 8;
      const float p[8] = {sP[g][k0], sP[g][k0 + 1], sP[g + 8][k0], sP[g + 8][k0 + 1],
                          sP[g][k1], sP[g][k1 + 1], sP[g + 8][k1], sP[g + 8][k1 + 1]};
      const PFrag pf = make_pfrag(p);
      pv_step<4>(o, pf, sV, ks, 4 * warp, lane);
      l_step(lf, pf);
    }
    // This query's split ends here: capture it (SEQ: fold it into the splits before, in order).
    if (part && ((c % SPLIT_CHUNKS) == SPLIT_CHUNKS - 1 || c == qnch - 1)) {
      if (SEQ && c >= SPLIT_CHUNKS) {
        const FoldCoef f0 = fold_coef(Fm0, m0), f1 = fold_coef(Fm1, m1);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          Fo[i].x[0] = fold_val(Fo[i].x[0], f0.a, o[i].x[0], f0.e);
          Fo[i].x[1] = fold_val(Fo[i].x[1], f0.a, o[i].x[1], f0.e);
          Fo[i].x[2] = fold_val(Fo[i].x[2], f1.a, o[i].x[2], f1.e);
          Fo[i].x[3] = fold_val(Fo[i].x[3], f1.a, o[i].x[3], f1.e);
        }
        Fl0 = fold_val(Fl0, f0.a, lf.x[0], f0.e);
        Fl1 = fold_val(Fl1, f1.a, lf.x[2], f1.e);
        Fm0 = f0.m;
        Fm1 = f1.m;
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) Fo[i] = o[i];
        Fl0 = lf.x[0];
        Fl1 = lf.x[2];
        Fm0 = m0;
        Fm1 = m1;
      }
    }
    __syncthreads();  // every P.V read of sV / sP done
    if (c + 1 < c_end) load_tile<THREADS>(sV, vb + koff(c + 1), nvalid(c + 1), tid);
    cp_async_commit();
  }
  cp_async_wait<0>();

  const bool in_split = qvalid && qnch > c_begin;  // the query has keys in the chunks this CTA walked
  if (SEQ || nsplit_g == 1) {  // final result (SEQ: every split folded; else the query fits split 0)
    if (in_split) {
      __nv_bfloat16* orow = out + static_cast<int64_t>(td.q_start + g) * out_row_stride + kvh * GQA * D;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int d = 8 * (4 * warp + i) + 2 * t;
        *reinterpret_cast<uint32_t*>(orow + d) = finish2(Fo[i].x[0], Fo[i].x[1], Fl0);
        *reinterpret_cast<uint32_t*>(orow + D + d) = finish2(Fo[i].x[2], Fo[i].x[3], Fl1);
      }
    }
  }
  if constexpr (SEQ) {
    return;
  } else {
    if (nsplit_tile == 1) return;
    // Publish this split's partials of the multi-split queries; the last split CTA to arrive folds them in order.
    const size_t tile_parts = (static_cast<size_t>(tix) * HKV + kvh) * grid_splits;
    if (in_split && nsplit_g > 1) {
      float* p0 = partials + ((tile_parts + split) * ROWS + g) * PSTR;
      float* p1 = partials + ((tile_parts + split) * ROWS + g + 8) * PSTR;
      if (warp == 0 && t == 0) {
        *reinterpret_cast<float4*>(p0) = make_float4(Fm0, Fl0, 0.f, 0.f);
        *reinterpret_cast<float4*>(p1) = make_float4(Fm1, Fl1, 0.f, 0.f);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int d = 8 * (4 * warp + i) + 2 * t;
        *reinterpret_cast<float2*>(p0 + 4 + d) = make_float2(Fo[i].x[0], Fo[i].x[1]);
        *reinterpret_cast<float2*>(p1 + 4 + d) = make_float2(Fo[i].x[2], Fo[i].x[3]);
      }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      const int prev = atomicAdd(&counters[tix * HKV + kvh], 1);
      sLast = (prev == nsplit_tile - 1);
    }
    __syncthreads();
    if (!sLast) return;
    __threadfence();
    // Merge: item = (MMA row, dim pair); rows of queries that span > 1 split, every split in order.
    for (int item = tid; item < ROWS * (D / 2); item += THREADS) {
      const int row = item / (D / 2), dp = item % (D / 2);
      const int qi = row & 7, hh = row >> 3;
      if (qi >= td.nq) continue;
      const int ns = splits_for(static_cast<int64_t>(td.pos0) + qi + 1);
      if (ns <= 1) continue;
      float Fm = 0.f, Fl = 0.f, Fa = 0.f, Fb = 0.f;
      for (int s2 = 0; s2 < ns; ++s2) {
        const float* P = partials + ((tile_parts + s2) * ROWS + row) * PSTR;
        const float2 ml = __ldcg(reinterpret_cast<const float2*>(P));
        const float2 po = __ldcg(reinterpret_cast<const float2*>(P + 4 + 2 * dp));
        if (s2 == 0) {
          Fm = ml.x;
          Fl = ml.y;
          Fa = po.x;
          Fb = po.y;
        } else {
          const FoldCoef f = fold_coef(Fm, ml.x);
          Fl = fold_val(Fl, f.a, ml.y, f.e);
          Fa = fold_val(Fa, f.a, po.x, f.e);
          Fb = fold_val(Fb, f.a, po.y, f.e);
          Fm = f.m;
        }
      }
      __nv_bfloat16* orow = out + static_cast<int64_t>(td.q_start + qi) * out_row_stride + kvh * GQA * D + hh * D;
      *reinterpret_cast<uint32_t*>(orow + 2 * dp) = finish2(Fa, Fb, Fl);
    }
    if (tid == 0) counters[tix * HKV + kvh] = 0;
  }
}

}  // namespace kt2n_pf_attn

extern "C" size_t kt2n_pf_q8_workspace_bytes(int tiles, int max_keys) {
  using namespace kt2n_pf_attn;
  if (tiles < 1) tiles = 1;
  if (max_keys < 1) max_keys = 1;
  return COUNTER_BYTES + static_cast<size_t>(tiles) * HKV * splits_for(max_keys) * ROWS * PSTR * sizeof(float);
}

extern "C" int kt2n_pf_q8_attention(const void* q, int64_t q_row_stride, const void* k_cache, const void* v_cache,
                                    int64_t kv_slot_stride, int64_t kv_head_stride, const kt2n_pf_q8_tile* tiles,
                                    int num_tiles, int max_keys, void* out, int64_t out_row_stride, void* workspace,
                                    size_t workspace_bytes, int pdl, int mode, cudaStream_t stream) {
  using namespace kt2n_pf_attn;
  if (num_tiles == 0) return 0;
  if (num_tiles < 0 || num_tiles > 65535 || max_keys < 1 || max_keys > KT2N_ATTN_MAX_CTX || !q || !k_cache ||
      !v_cache || !tiles || !out || mode < 0 || mode > 2)
    return KT2N_ATTN_EINVAL;
  if (((reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(k_cache) | reinterpret_cast<uintptr_t>(v_cache) |
        reinterpret_cast<uintptr_t>(out) | reinterpret_cast<uintptr_t>(workspace)) & 15u) ||
      (q_row_stride % 8) || (out_row_stride % 8) || (kv_slot_stride % 8) || (kv_head_stride % 8))
    return KT2N_ATTN_EINVAL;
  const bool ws_ok = workspace && num_tiles * HKV * 4 <= static_cast<int>(COUNTER_BYTES) &&
                     workspace_bytes >= kt2n_pf_q8_workspace_bytes(num_tiles, max_keys);
  bool seq;
  if (mode == 1) {
    if (!ws_ok) return KT2N_ATTN_EINVAL;
    seq = false;
  } else if (mode == 2) {
    seq = true;
  } else {  // auto: split-parallel while the per-split grid is small (single requests), sequential for bursts
    seq = !ws_ok || num_tiles * HKV >= KT2N_PF_Q8_SEQ_MIN_CTAS;
  }
  static const bool carve = [] {
    cudaFuncSetAttribute(q8_kernel<false>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(q8_kernel<true>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaGetLastError();
    return true;
  }();
  (void)carve;
  const int gs = splits_for(max_keys);
  int* counters = static_cast<int*>(workspace);
  float* partials = ws_ok ? reinterpret_cast<float*>(static_cast<char*>(workspace) + COUNTER_BYTES) : nullptr;
  auto qp = static_cast<const __nv_bfloat16*>(q);
  auto kp = static_cast<const __nv_bfloat16*>(k_cache);
  auto vp = static_cast<const __nv_bfloat16*>(v_cache);
  auto op = static_cast<__nv_bfloat16*>(out);
  const cudaError_t e =
      seq ? launch_kernel(q8_kernel<true>, dim3(1, HKV, num_tiles), dim3(THREADS), 0, stream, pdl != 0, qp,
                          q_row_stride, kp, vp, kv_slot_stride, kv_head_stride, tiles, gs, op, out_row_stride, counters,
                          partials)
          : launch_kernel(q8_kernel<false>, dim3(gs, HKV, num_tiles), dim3(THREADS), 0, stream, pdl != 0, qp,
                          q_row_stride, kp, vp, kv_slot_stride, kv_head_stride, tiles, gs, op, out_row_stride, counters,
                          partials);
  return static_cast<int>(e);
}
