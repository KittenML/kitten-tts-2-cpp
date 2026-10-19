// kt2n prefill attention (lane k3-attn): causal attention of a row's new queries over that
// row's keys (prefix + the suffix itself), batched over rows with a device descriptor array.
//
// A CTA holds a tile of 32 queries x 2 query heads of one K/V head: 4 warps, warp w owns
// queries 8w .. 8w+7 as the 16 rows of its MMA tiles (row g = query g head 0, row g + 8 =
// query g head 1 -- the rows decode uses for its single query).  Every query row runs exactly
// decode's per-chunk arithmetic (the shared helpers in attn_common.cuh, same operands, same
// order) and folds splits at the same boundaries, so a query's output is bitwise what
// kt2n_attn_decode_* returns at that position.
//
// Two launch shapes, bitwise identical (the fold order never changes, only who computes it):
//  * sequential: grid (tile, kv_head, row); the CTA walks every split of its tile in order.
//    No workspace.  Best when there are many tiles (long suffixes, big bursts).
//  * split-parallel: grid (tile, kv_head x split, row); one CTA per split writes its split
//    partial (m, l, o) to the workspace and the last of the tile's splits to arrive folds them
//    in split order.  Best for few tiles over long prefixes (one short suffix).
#include "attn_common.cuh"
#include "attn_production.cuh"

namespace kt2n_attn {

constexpr int PF_WARPS = 4;
constexpr int PF_THREADS = 32 * PF_WARPS;
constexpr int PF_BQ = 8 * PF_WARPS;    // queries per tile
constexpr int PF_ROWS = PF_BQ * GQA;   // query rows per tile; partial row = 2 * query + head
constexpr int PF_PART = 132;           // floats per (split, query row) partial: m, l, 0, 0, o[128]

__host__ __device__ inline int pf_splits_for(int64_t keys) {
  return static_cast<int>((keys + SPLIT_KEYS - 1) / SPLIT_KEYS);
}
__host__ __device__ inline size_t pf_counter_bytes(int rows, int tiles) {
  const size_t raw = static_cast<size_t>(rows) * tiles * HKV * sizeof(int);
  return (raw + 4095) / 4096 * 4096;
}

__device__ __forceinline__ uint32_t pf_ld32(const __nv_bfloat16* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}

template <bool PAR>
__global__ void __launch_bounds__(PF_THREADS)
    prefill_kernel(const __nv_bfloat16* __restrict__ q, int64_t q_row_stride,
                   const __nv_bfloat16* __restrict__ kc, const __nv_bfloat16* __restrict__ vc,
                   int64_t slot_stride, int64_t head_stride, const kt2n_attn_prefill_row* __restrict__ desc,
                   __nv_bfloat16* __restrict__ out, int64_t out_row_stride, int grid_splits, int grid_tiles,
                   int* __restrict__ counters, float* __restrict__ partials) {
  __shared__ __align__(16) uint4 sK[CHUNK * SEGS];
  __shared__ __align__(16) uint4 sV[CHUNK * SEGS];
  __shared__ __align__(16) uint4 sQ[PF_ROWS * SEGS];  // row = 2 * query + head, swizzled by query
  int& sLast = *reinterpret_cast<int*>(sQ);            // reused after the chunk loop (merge flag)
  pdl_wait();

  const int tile = blockIdx.x, r = blockIdx.z;
  const int kvh = PAR ? static_cast<int>(blockIdx.y) % HKV : static_cast<int>(blockIdx.y);
  const int split = PAR ? static_cast<int>(blockIdx.y) / HKV : 0;
  const kt2n_attn_prefill_row rd = desc[r];
  const int q0 = tile * PF_BQ;
  if (q0 >= rd.q_len) return;
  const int nq = min(PF_BQ, rd.q_len - q0);
  const int64_t pos_first = static_cast<int64_t>(rd.q_pos0) + q0;
  const int64_t nkeys = pos_first + nq;  // keys the tile's last query attends
  const int nchunks = static_cast<int>((nkeys + CHUNK - 1) / CHUNK);
  const int tile_splits = pf_splits_for(nkeys);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int qi = 8 * warp + g;  // this lane's query (tile rows g / g+8 of its warp)
  const bool qvalid = qi < nq;
  const int64_t pq = pos_first + qi;
  const int qnch = qvalid ? static_cast<int>(pq / CHUNK) + 1 : 0;  // chunks it attends
  __nv_bfloat16* orow = out + static_cast<int64_t>(rd.q_start + q0 + qi) * out_row_stride + kvh * GQA * D;

  if constexpr (PAR) {
    if (tile_splits > grid_splits) {  // contract violation (max_keys too small): NaN, loudly
      if (split == 0 && qvalid)
        for (int i = t; i < GQA * D; i += 4) orow[i] = __float2bfloat16_rn(__int_as_float(0x7fc00000));
      return;
    }
    if (split >= tile_splits) return;
  }
  const int c_begin = PAR ? split * SPLIT_CHUNKS : 0;
  const int c_end = PAR ? min(nchunks, (split + 1) * SPLIT_CHUNKS) : nchunks;

  const int64_t base = static_cast<int64_t>(rd.kv_slot) * slot_stride + kvh * head_stride +
                       static_cast<int64_t>(rd.kv_col0) * D;
  const __nv_bfloat16* kb = kc + base;
  const __nv_bfloat16* vb = vc + base;
  auto nvalid = [&](int c) { return static_cast<int>(imin64(CHUNK, nkeys - static_cast<int64_t>(c) * CHUNK)); };
  auto koff = [&](int c) { return static_cast<int64_t>(c) * CHUNK * D; };

  // Q tile -> shared memory (queries beyond nq zero-filled); A fragments come from ldmatrix.
  for (int i = tid; i < PF_ROWS * SEGS; i += PF_THREADS) {
    const int row = i / SEGS, seg = i % SEGS, qq = row >> 1, hh = row & 1;
    const bool ok = qq < nq;
    const __nv_bfloat16* src =
        q + static_cast<int64_t>(rd.q_start + q0 + (ok ? qq : 0)) * q_row_stride + (kvh * GQA + hh) * D + seg * 8;
    cp_async16(&sQ[row * SEGS + (seg ^ (qq & 7))], src, ok);
  }
  cp_async_commit();
  // ldmatrix row address of this lane for A fragments: matrix mi = lane >> 3 -> head mi & 1,
  // dims +8 * (mi >> 1); row = query 8w + (lane & 7).
  const int aq = 8 * warp + (lane & 7), ah = (lane >> 3) & 1, aseg = lane >> 4;
  load_tile<PF_THREADS>(sK, kb + koff(c_begin), nvalid(c_begin), tid);
  cp_async_commit();
  load_tile<PF_THREADS>(sV, vb + koff(c_begin), nvalid(c_begin), tid);
  cp_async_commit();


  // Running split state (rows head 0 / head 1 of this lane's query) and the final state F.
  float Sm0 = 0.f, Sm1 = 0.f, Fm0 = 0.f, Fm1 = 0.f, Fl0 = 0.f, Fl1 = 0.f;
  FragC O[16], L, F[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) zero(F[i]);

  for (int c = c_begin; c < c_end; ++c) {
    // Valid keys of this chunk for this lane's query (0: the query does not reach it).
    const int nvq =
        qvalid ? static_cast<int>(imax64(0, imin64(CHUNK, pq - static_cast<int64_t>(c) * CHUNK + 1))) : 0;
    cp_async_wait<1>();
    __syncthreads();  // K(c) visible

    FragC st[8];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) zero(st[nt]);
#pragma unroll
    for (int ks = 0; ks < 8; ks += 2) {
      FragA a0, a1;
      ldsm_x4(a0.x, &sQ[(2 * aq + ah) * SEGS + ((2 * ks + aseg) ^ (aq & 7))]);
      ldsm_x4(a1.x, &sQ[(2 * aq + ah) * SEGS + ((2 * ks + 2 + aseg) ^ (aq & 7))]);
      qk_step2<8>(st, a0, a1, sK, 2 * ks, lane);
    }
    float mc0 = -INFINITY, mc1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int j = 8 * nt + 2 * t;
      const bool v0 = j < nvq, v1 = j + 1 < nvq;
      st[nt].x[0] = v0 ? score_of(st[nt].x[0]) : -INFINITY;
      st[nt].x[1] = v1 ? score_of(st[nt].x[1]) : -INFINITY;
      st[nt].x[2] = v0 ? score_of(st[nt].x[2]) : -INFINITY;
      st[nt].x[3] = v1 ? score_of(st[nt].x[3]) : -INFINITY;
      mc0 = fmaxf(mc0, fmaxf(st[nt].x[0], st[nt].x[1]));
      mc1 = fmaxf(mc1, fmaxf(st[nt].x[2], st[nt].x[3]));
    }
    mc0 = quad_max(mc0);
    mc1 = quad_max(mc1);
    __syncthreads();  // every Q.K read of sK done
    if (c + 1 < c_end) load_tile<PF_THREADS>(sK, kb + koff(c + 1), nvalid(c + 1), tid);
    cp_async_commit();

    const bool part = nvq > 0;
    const bool first = (c % SPLIT_CHUNKS) == 0;
    float a0 = 1.f, a1 = 1.f;
    if (part) {
      if (first) {
        Sm0 = mc0;
        Sm1 = mc1;
      } else {
        const float n0 = fmaxf(Sm0, mc0), n1 = fmaxf(Sm1, mc1);
        a0 = expf(__fsub_rn(Sm0, n0));
        a1 = expf(__fsub_rn(Sm1, n1));
        Sm0 = n0;
        Sm1 = n1;
      }
    }
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int j = 8 * nt + 2 * t;
      const bool v0 = j < nvq, v1 = j + 1 < nvq;
      st[nt].x[0] = v0 ? prob_of(st[nt].x[0], Sm0) : 0.f;
      st[nt].x[1] = v1 ? prob_of(st[nt].x[1], Sm0) : 0.f;
      st[nt].x[2] = v0 ? prob_of(st[nt].x[2], Sm1) : 0.f;
      st[nt].x[3] = v1 ? prob_of(st[nt].x[3], Sm1) : 0.f;
    }
    cp_async_wait<1>();
    __syncthreads();  // V(c) visible

    if (first) {
#pragma unroll
      for (int i = 0; i < 16; ++i) zero(O[i]);
      zero(L);
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) scale_frag(O[i], a0, a1);
      scale_frag(L, a0, a1);
    }
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const float p[8] = {st[2 * ks].x[0],     st[2 * ks].x[1],     st[2 * ks].x[2],     st[2 * ks].x[3],
                          st[2 * ks + 1].x[0], st[2 * ks + 1].x[1], st[2 * ks + 1].x[2], st[2 * ks + 1].x[3]};
      const PFrag pf = make_pfrag(p);
      pv_step<16>(O, pf, sV, ks, 0, lane);
      l_step(L, pf);
    }
    // This query's split ends here: capture it (its running state is dead afterwards).
    if (part && ((c % SPLIT_CHUNKS) == SPLIT_CHUNKS - 1 || c == qnch - 1)) {
      if (!PAR && c >= SPLIT_CHUNKS) {
        const FoldCoef f0 = fold_coef(Fm0, Sm0), f1 = fold_coef(Fm1, Sm1);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          F[i].x[0] = fold_val(F[i].x[0], f0.a, O[i].x[0], f0.e);
          F[i].x[1] = fold_val(F[i].x[1], f0.a, O[i].x[1], f0.e);
          F[i].x[2] = fold_val(F[i].x[2], f1.a, O[i].x[2], f1.e);
          F[i].x[3] = fold_val(F[i].x[3], f1.a, O[i].x[3], f1.e);
        }
        Fl0 = fold_val(Fl0, f0.a, L.x[0], f0.e);
        Fl1 = fold_val(Fl1, f1.a, L.x[2], f1.e);
        Fm0 = f0.m;
        Fm1 = f1.m;
      } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) F[i] = O[i];
        Fl0 = L.x[0];
        Fl1 = L.x[2];
        Fm0 = Sm0;
        Fm1 = Sm1;
      }
    }
    __syncthreads();  // every P.V read of sV done
    if (c + 1 < c_end) load_tile<PF_THREADS>(sV, vb + koff(c + 1), nvalid(c + 1), tid);
    cp_async_commit();
  }
  cp_async_wait<0>();

  auto store_out = [&]() {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int d = 8 * i + 2 * t;
      *reinterpret_cast<uint32_t*>(orow + d) = finish2(F[i].x[0], F[i].x[1], Fl0);
      *reinterpret_cast<uint32_t*>(orow + D + d) = finish2(F[i].x[2], F[i].x[3], Fl1);
    }
  };

  if constexpr (!PAR) {
    if (qvalid) store_out();
    return;
  } else {
    if (tile_splits == 1) {  // the split result is the final one
      if (qvalid) store_out();
      return;
    }
    // Publish this split's partial for the queries that reach it; the last arriver merges.
    const size_t tile_parts = (static_cast<size_t>(r) * grid_tiles + tile) * HKV + kvh;
    const size_t row0 = ((tile_parts * grid_splits + split) * PF_ROWS + 2 * qi) * PF_PART;
    if (qvalid && qnch > c_begin) {
      float* p0 = partials + row0;
      float* p1 = p0 + PF_PART;
      if (t == 0) {
        *reinterpret_cast<float4*>(p0) = make_float4(Fm0, Fl0, 0.f, 0.f);
        *reinterpret_cast<float4*>(p1) = make_float4(Fm1, Fl1, 0.f, 0.f);
      }
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int d = 8 * i + 2 * t;
        *reinterpret_cast<float2*>(p0 + 4 + d) = make_float2(F[i].x[0], F[i].x[1]);
        *reinterpret_cast<float2*>(p1 + 4 + d) = make_float2(F[i].x[2], F[i].x[3]);
      }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      const int prev = atomicAdd(&counters[tile_parts], 1);
      sLast = (prev == tile_splits - 1);
    }
    __syncthreads();
    if (!sLast) return;
    __threadfence();
    if (qvalid) {
      const int nsq = static_cast<int>(pq / SPLIT_KEYS) + 1;  // splits this query attends
      const size_t sstride = static_cast<size_t>(PF_ROWS) * PF_PART;
      const float* b0 = partials + ((tile_parts * grid_splits) * PF_ROWS + 2 * qi) * PF_PART;
      for (int s = 0; s < nsq; ++s) {
        const float* p0 = b0 + s * sstride;
        const float* p1 = p0 + PF_PART;
        const float2 ml0 = __ldcg(reinterpret_cast<const float2*>(p0));
        const float2 ml1 = __ldcg(reinterpret_cast<const float2*>(p1));
        if (s == 0) {
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            const int d = 8 * i + 2 * t;
            const float2 x0 = __ldcg(reinterpret_cast<const float2*>(p0 + 4 + d));
            const float2 x1 = __ldcg(reinterpret_cast<const float2*>(p1 + 4 + d));
            F[i].x[0] = x0.x;
            F[i].x[1] = x0.y;
            F[i].x[2] = x1.x;
            F[i].x[3] = x1.y;
          }
          Fm0 = ml0.x;
          Fl0 = ml0.y;
          Fm1 = ml1.x;
          Fl1 = ml1.y;
        } else {
          const FoldCoef f0 = fold_coef(Fm0, ml0.x), f1 = fold_coef(Fm1, ml1.x);
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            const int d = 8 * i + 2 * t;
            const float2 x0 = __ldcg(reinterpret_cast<const float2*>(p0 + 4 + d));
            const float2 x1 = __ldcg(reinterpret_cast<const float2*>(p1 + 4 + d));
            F[i].x[0] = fold_val(F[i].x[0], f0.a, x0.x, f0.e);
            F[i].x[1] = fold_val(F[i].x[1], f0.a, x0.y, f0.e);
            F[i].x[2] = fold_val(F[i].x[2], f1.a, x1.x, f1.e);
            F[i].x[3] = fold_val(F[i].x[3], f1.a, x1.y, f1.e);
          }
          Fl0 = fold_val(Fl0, f0.a, ml0.y, f0.e);
          Fl1 = fold_val(Fl1, f1.a, ml1.y, f1.e);
          Fm0 = f0.m;
          Fm1 = f1.m;
        }
      }
      store_out();
    }
    if (tid == 0) counters[tile_parts] = 0;
  }
}

inline bool pf_aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

inline void prefill_setup() {
  static const bool done = [] {
    cudaFuncSetAttribute(prefill_kernel<false>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(prefill_kernel<true>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaGetLastError();
    return true;
  }();
  (void)done;
}

inline int sm_count() {
  int dev = 0, n = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
    cudaGetLastError();
    return 128;
  }
  return n;
}

}  // namespace kt2n_attn

extern "C" size_t kt2n_attn_prefill_workspace_bytes(int max_rows, int max_q_len, int max_keys) {
  using namespace kt2n_attn;
  if (max_rows < 1) max_rows = 1;
  if (max_q_len < 1) max_q_len = 1;
  if (max_keys < 1) max_keys = 1;
#if KT2N_ATTN_PRODUCTION_ORDER
  const int tiles = (max_q_len + PF_BQ - 1) / PF_BQ;
  return pf_counter_bytes(max_rows, tiles) +
      static_cast<size_t>(max_rows) * (max_q_len < production::PREFILL_TILE ? max_q_len : production::PREFILL_TILE) *
      HQ * ((static_cast<size_t>(max_keys) + CHUNK - 1) / CHUNK) * 130 * sizeof(float);
#else
  const int tiles = (max_q_len + PF_BQ - 1) / PF_BQ;
  const size_t parts = static_cast<size_t>(max_rows) * tiles * HKV * pf_splits_for(max_keys) * PF_ROWS * PF_PART *
                       sizeof(float);
  return pf_counter_bytes(max_rows, tiles) + parts;
#endif
}

extern "C" int kt2n_attn_prefill(const void* q, int64_t q_row_stride, const void* k_cache, const void* v_cache,
                                 int64_t kv_slot_stride, int64_t kv_head_stride,
                                 const kt2n_attn_prefill_row* rows_desc, int num_rows, int max_q_len, int max_keys,
                                 void* out, int64_t out_row_stride, void* workspace, size_t workspace_bytes,
                                 unsigned flags, cudaStream_t stream) {
  using namespace kt2n_attn;
  if (num_rows == 0 || max_q_len == 0) return 0;
  if (num_rows < 0 || num_rows > 65535 || max_q_len < 0 || max_keys < max_q_len || max_keys > KT2N_ATTN_MAX_CTX)
    return KT2N_ATTN_EINVAL;
  if (!q || !k_cache || !v_cache || !rows_desc || !out) return KT2N_ATTN_EINVAL;
  if (!pf_aligned16(q) || !pf_aligned16(k_cache) || !pf_aligned16(v_cache) || !pf_aligned16(out))
    return KT2N_ATTN_EINVAL;
  if ((q_row_stride % 8) || (out_row_stride % 8) || (kv_slot_stride % 8) || (kv_head_stride % 8))
    return KT2N_ATTN_EINVAL;
  if ((flags & KT2N_ATTN_FLAG_PREFILL_SEQUENTIAL) && (flags & KT2N_ATTN_FLAG_PREFILL_SPLIT)) return KT2N_ATTN_EINVAL;
#if KT2N_ATTN_PRODUCTION_ORDER
  const int tiles = (max_q_len + PF_BQ - 1) / PF_BQ;
  const bool ws_ok = workspace && pf_aligned16(workspace) &&
      workspace_bytes >= kt2n_attn_prefill_workspace_bytes(num_rows, max_q_len, max_keys);
  const bool forced_split = (flags & KT2N_ATTN_FLAG_PREFILL_SPLIT) != 0;
  if (forced_split && !ws_ok) return KT2N_ATTN_EINVAL;
  const bool par = forced_split || (!(flags & KT2N_ATTN_FLAG_PREFILL_SEQUENTIAL) &&
      ws_ok && pf_splits_for(max_keys) > 1 &&
      4 * static_cast<int64_t>(tiles) * HKV * num_rows < 3 * sm_count());
  if (par) {
    const int nch = (max_keys + CHUNK - 1) / CHUNK;
    const bool pdl = (flags & KT2N_ATTN_FLAG_PDL) != 0;
    // Keep the public reusable-workspace counter prefix untouched and zero.
    // This implementation folds in a separate kernel and needs no counters.
    auto* parts = reinterpret_cast<float*>(static_cast<char*>(workspace) + pf_counter_bytes(num_rows, tiles));
    for (int query0 = 0; query0 < max_q_len; query0 += production::PREFILL_TILE) {
      const int queries = min(production::PREFILL_TILE, max_q_len - query0);
      cudaError_t rc = launch_kernel(production::prefill_partials,
          dim3(queries, HQ*nch, num_rows), dim3(D), 0, stream, pdl,
          static_cast<const __nv_bfloat16*>(q), q_row_stride,
          static_cast<const __nv_bfloat16*>(k_cache), static_cast<const __nv_bfloat16*>(v_cache),
          kv_slot_stride, kv_head_stride, rows_desc, max_keys, query0, parts);
      if (rc != cudaSuccess) return static_cast<int>(rc);
      rc = launch_kernel(production::prefill_combine,
          dim3(queries, HQ, num_rows), dim3(D), 0, stream, pdl,
          parts, rows_desc, max_keys, kv_head_stride, query0, nch,
          static_cast<__nv_bfloat16*>(out), out_row_stride);
      if (rc != cudaSuccess) return static_cast<int>(rc);
    }
    return 0;
  }
  return static_cast<int>(launch_kernel(production::prefill_sequential,
      dim3(max_q_len, HQ, num_rows), dim3(D), 0, stream, (flags & KT2N_ATTN_FLAG_PDL) != 0,
      static_cast<const __nv_bfloat16*>(q), q_row_stride,
      static_cast<const __nv_bfloat16*>(k_cache), static_cast<const __nv_bfloat16*>(v_cache),
      kv_slot_stride, kv_head_stride, rows_desc, max_keys, static_cast<__nv_bfloat16*>(out), out_row_stride));
#else
  prefill_setup();
  const int tiles = (max_q_len + PF_BQ - 1) / PF_BQ;
  const int splits = pf_splits_for(max_keys);
  const bool ws_ok = workspace && pf_aligned16(workspace) &&
                     workspace_bytes >= kt2n_attn_prefill_workspace_bytes(num_rows, max_q_len, max_keys);
  bool par;
  if (flags & KT2N_ATTN_FLAG_PREFILL_SPLIT) {
    if (!ws_ok) return KT2N_ATTN_EINVAL;
    par = true;
  } else if (flags & KT2N_ATTN_FLAG_PREFILL_SEQUENTIAL) {
    par = false;
  } else {
    // auto: split-parallel only when the sequential grid leaves most SMs idle.  MEASURED on
    // the RTX PRO 6000 (188 SMs): one-pass 512 (128 CTAs) and one 100-token suffix (32 CTAs)
    // are faster split; bursts of 10 suffixes (160 / 320 CTAs) are faster sequential.
    par = ws_ok && splits > 1 && 4 * static_cast<int64_t>(tiles) * HKV * num_rows < 3 * sm_count();
  }
  const bool pdl = (flags & KT2N_ATTN_FLAG_PDL) != 0;
  int* counters = static_cast<int*>(workspace);
  float* partials =
      ws_ok ? reinterpret_cast<float*>(static_cast<char*>(workspace) + pf_counter_bytes(num_rows, tiles)) : nullptr;
  const dim3 grid(tiles, par ? HKV * splits : HKV, num_rows);
  auto qp = static_cast<const __nv_bfloat16*>(q);
  auto kp = static_cast<const __nv_bfloat16*>(k_cache);
  auto vp = static_cast<const __nv_bfloat16*>(v_cache);
  auto op = static_cast<__nv_bfloat16*>(out);
  const cudaError_t e =
      par ? launch_kernel(prefill_kernel<true>, grid, dim3(PF_THREADS), 0, stream, pdl, qp, q_row_stride, kp, vp,
                          kv_slot_stride, kv_head_stride, rows_desc, op, out_row_stride, splits, tiles, counters,
                          partials)
          : launch_kernel(prefill_kernel<false>, grid, dim3(PF_THREADS), 0, stream, pdl, qp, q_row_stride, kp, vp,
                          kv_slot_stride, kv_head_stride, rows_desc, op, out_row_stride, splits, tiles, counters,
                          partials);
  return static_cast<int>(e);
#endif
}
