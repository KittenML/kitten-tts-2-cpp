// kt2n lane k4: the production sampler chain + exponential race, natively.
//
// One CTA of exactly 1024 threads per row.  1024 is not a tuning choice: it is
// the block torch 2.7.1 uses for softmax over a 6,568-wide fp32 row
// (cunn_SoftMaxForwardReg<..., reg_cnt = 7>), and every softmax below repeats
// that kernel's reduction tree bit for bit:
//   thread t sums exp(x[t + 1024 j] - max) for j = 0..6 in order, then a
//   shfl_down(16,8,4,2,1) warp tree, then warp 0 reduces the 32 warp sums
//   with the same tree; output = exp(x - max) / sum.
// The top-p cumsum repeats either torch's one-row cumsum (cub DeviceScan,
// the default: production's bucket 1) or torch's >= 2-row innermost-dim scan
// (KT2N_SAMP_SCAN_MULTI_ROW: production's buckets 2..10) -- the two round
// differently, which is why production itself is not batch-invariant here.
//
// Chain (runtime._sampling_processors(per_row=True, graph_safe=True) on the
// sliced head, then fast_lm.sampler_graph's softmax and argmax(p / noise)):
//   1. FastTokenRunPenalty      s[last] -= 1.3 * (run - 9) when run >= 10
//   2. FastRepetitionPenaltyExempt  history columns: s<0 ? s*1.1 : s*(1/1.1),
//      exempt SPEECH_END / STOP / silence keep the post-run value
//   3. temperature              s * float32(1/t)   (or s / t, flag)
//   4. top-k                    s < kth largest -> -inf (k=0 off, k<=200)
//   5. top-p                    stable ascending sort, softmax, cumsum,
//                               cum <= float32(1-top_p) -> -inf (last kept)
//   6. min-p                    softmax p < min_p * max(p) -> -inf
//   7. draw                     argmax(softmax(s) / noise), lowest index on ties
#include "kt2n/kernels/samp.h"

#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdint.h>

#include "samp_philox.cuh"

namespace kt2n_samp {
namespace {

constexpr int kThreads = 1024;
constexpr int kWarps = kThreads / 32;
constexpr int kN = KT2N_SAMP_AUDIO_N;               // 6568
constexpr int kPer = (kN + kThreads - 1) / kThreads;  // 7 = torch reg_cnt
constexpr int kWindow = KT2N_SAMP_WINDOW;
constexpr int kSortCap = 8192;                       // next pow2 >= kN
constexpr int kScanX = 512;                          // torch num_threads_x
constexpr int kScanChunk = 2 * kScanX;               // 1024
constexpr int kLow = KT2N_SAMP_AUDIO_LOW;
constexpr int kAudioBase = KT2N_SAMP_AUDIO_ID_BASE;
constexpr int kAudioEnd = KT2N_SAMP_AUDIO_ID_BASE + KT2N_SAMP_AUDIO_VOCAB;
// Exempt columns of the repetition penalty: SPEECH_END, STOP, AUDIO_ID_BASE+4299.
constexpr int kExempt0 = 151670 - kLow;
constexpr int kExempt1 = 151674 - kLow;
constexpr int kExempt2 = kAudioBase + 4299 - kLow;

static_assert(kPer == 7, "torch's softmax reg_cnt for 6568 columns is 7");
static_assert(kSortCap >= kN, "sort buffer must hold the whole row");

struct Smem {
  float s[kN];                 // the row's scores (kN * 4 is a multiple of 16)
  unsigned long long keys[kSortCap];  // top-p sort: (ordered float << 32) | column
  float scan[kScanChunk];      // torch's row_buf
  float wagg[16];              // one-row scan: warp aggregates per tile
  int hist[256];
  float redf[kWarps];
  int redi[kWarps];
  float bcastf[4];
  int bcasti[8];
};

__device__ __forceinline__ unsigned int ordered_key(float f) {
  // cub radix-sort twiddle: ascending unsigned order == ascending float order
  // with -0.0 < +0.0 (torch's stable radix sort sees the same order).
  const unsigned int u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// The top-p sort key: torch.sort on CUDA is cub's radix sort, which maps
// -0.0 to +0.0 before extracting digits (radix_rank_sort_operations.cuh,
// ProcessFloatMinusZero), so the two zeros are EQUAL keys ordered by index.
__device__ __forceinline__ unsigned int sort_key(float f) {
  return ordered_key(f == 0.0f ? 0.0f : f);
}

__device__ __forceinline__ float key_to_float(unsigned int k) {
  const unsigned int u = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
  return __uint_as_float(u);
}

// torch cuda_utils::BlockReduce<float, Add> as used by blockReduceWarp:
// warp shfl_down tree, warp sums to shared, warp 0 reduces them; broadcast.
__device__ __forceinline__ float block_sum_torch(float v, Smem& sm) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, off));
  __syncthreads();
  if (lane == 0) sm.redf[wid] = v;
  __syncthreads();
  if (wid == 0) {
    float w = sm.redf[lane];  // kWarps == 32: every lane has a warp sum
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) w = __fadd_rn(w, __shfl_down_sync(0xffffffffu, w, off));
    if (lane == 0) sm.bcastf[0] = w;
  }
  __syncthreads();
  return sm.bcastf[0];
}

__device__ __forceinline__ float block_max(float v, Smem& sm) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  __syncthreads();
  if (lane == 0) sm.redf[wid] = v;
  __syncthreads();
  if (wid == 0) {
    float w = sm.redf[lane];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) w = fmaxf(w, __shfl_down_sync(0xffffffffu, w, off));
    if (lane == 0) sm.bcastf[1] = w;
  }
  __syncthreads();
  return sm.bcastf[1];
}

// exp(x - max) as torch's SumExpFloat / SoftMaxForwardEpilogue compute it.
__device__ __forceinline__ float exp_shifted(float x, float mx) { return expf(__fsub_rn(x, mx)); }

// Sum of exp(s[i] - mx) over the row in torch's order (-inf terms add +0).
__device__ __forceinline__ float softmax_denominator(const float* s, float mx, Smem& sm) {
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int i = threadIdx.x + j * kThreads;
    if (i < kN) {
      const float v = s[i];
      if (v != -CUDART_INF_F) acc = __fadd_rn(acc, exp_shifted(v, mx));
    }
  }
  return block_sum_torch(acc, sm);
}

// Radix select: the kk-th largest ordered key (kk >= 1) of the row.
__device__ unsigned int kth_largest_key(const unsigned int (&u)[kPer], int kk, Smem& sm) {
  unsigned int prefix = 0u, mask = 0u;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int b = threadIdx.x; b < 256; b += kThreads) sm.hist[b] = 0;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const int i = threadIdx.x + j * kThreads;
      if (i < kN && (u[j] & mask) == prefix) atomicAdd(&sm.hist[(u[j] >> shift) & 255u], 1);
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      const int lane = threadIdx.x;
      // lane covers bins 255-8*lane .. 248-8*lane (descending)
      int local = 0;
#pragma unroll
      for (int q = 0; q < 8; ++q) local += sm.hist[255 - (lane * 8 + q)];
      int incl = local;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= off) incl += other;
      }
      const int excl = incl - local;
      const bool mine = excl < kk && kk <= incl;
      if (mine) {
        int running = excl;
        for (int q = 0; q < 8; ++q) {
          const int b = 255 - (lane * 8 + q);
          const int c = sm.hist[b];
          if (running + c >= kk) {
            sm.bcasti[1] = b;
            sm.bcasti[2] = running;
            break;
          }
          running += c;
        }
      }
    }
    __syncthreads();
    const unsigned int digit = static_cast<unsigned int>(sm.bcasti[1]);
    kk -= sm.bcasti[2];
    prefix |= digit << shift;
    mask |= 255u << shift;
    __syncthreads();
  }
  return prefix;
}

__device__ void bitonic_sort_u64(unsigned long long* keys, int P) {
  for (int k = 2; k <= P; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < P; i += kThreads) {
        const int ixj = i ^ j;
        if (ixj > i) {
          const unsigned long long a = keys[i];
          const unsigned long long b = keys[ixj];
          const bool ascending = (i & k) == 0;
          if ((a > b) == ascending) {
            keys[i] = b;
            keys[ixj] = a;
          }
        }
      }
      __syncthreads();
    }
  }
}

// The sorted row's softmax value at sorted position pos (0 below base).
__device__ __forceinline__ float sorted_prob(const Smem& sm, int pos, int base, float mx, float z1) {
  if (pos < base || pos >= kN) return 0.0f;
  const float v = key_to_float(static_cast<unsigned int>(sm.keys[pos - base] >> 32));
  return __fdiv_rn(exp_shifted(v, mx), z1);
}

// sorted_remove = cum <= thr, sorted_remove[..., -1] = 0, scattered back.
__device__ __forceinline__ void top_p_remove(Smem& sm, float* s, int pos, int base, float cum, float thr) {
  if (pos >= base && pos < kN - 1 && cum <= thr) {
    const int idx = static_cast<int>(sm.keys[pos - base] & 0xFFFFFFFFull);
    s[idx] = -CUDART_INF_F;
  }
}

// torch's cumsum for >= 2 rows (scan_innermost_dim): 512 x-threads, chunks of
// 1024, Sklansky in shared memory, the running total added to element 0 of
// each chunk.  Chunks wholly below base are all +0 and leave the running
// total at exactly +0, so the scan starts at base's chunk.
__device__ void top_p_scan_multi_row(Smem& sm, float* s, int base, float mx, float z1, float thr) {
  const int t = threadIdx.x;
  float block_total = 0.0f;
  for (int chunk = base / kScanChunk; chunk * kScanChunk < kN; ++chunk) {
    const int col0 = chunk * kScanChunk;
    if (t < kScanX) {
      sm.scan[t] = sorted_prob(sm, col0 + t, base, mx, z1);
      sm.scan[t + kScanX] = sorted_prob(sm, col0 + t + kScanX, base, mx, z1);
      if (t == 0) sm.scan[0] = __fadd_rn(sm.scan[0], block_total);
    }
    __syncthreads();
    for (int m = 0; m <= 9; ++m) {  // log_num_threads_x = 9 for 2..16 rows of 6568
      if (t < kScanX) {
        const int sz = 1 << m;
        const int a = ((t >> m) << (m + 1)) | sz;
        const int ti = a + (t % sz);
        const int si = a - 1;
        sm.scan[ti] = __fadd_rn(sm.scan[ti], sm.scan[si]);
      }
      __syncthreads();
    }
    if (t < kScanX) {
      top_p_remove(sm, s, col0 + t, base, sm.scan[t], thr);
      top_p_remove(sm, s, col0 + t + kScanX, base, sm.scan[t + kScanX], thr);
    }
    block_total = sm.scan[kScanChunk - 1];
    __syncthreads();
  }
}

// torch's cumsum for ONE row: cub::DeviceScan::InclusiveScan (CCCL 2.7,
// CUDA 12.8) with std::plus<float> -- not a "primitive" op, so the sm90
// default tuning: 128 threads x 15 items = 1920-item tiles (6568 -> 4 tiles),
// BLOCK_SCAN_WARP_SCANS: per-thread sequential reduce, warp Kogge-Stone
// (shfl_up 1..16), warp prefixes summed in warp order, the tile prefix
// added, then the per-thread sequential inclusive scan.  Tile prefixes are
// taken as the in-order chain incl[t] = incl[t-1] + agg[t]; cub's decoupled
// look-back can also produce (agg2 + agg1) + agg0 for the last tile, which
// only matters when non-zero probabilities reach back past the last tile
// (more than 808 survivors) -- torch's own result is timing-dependent there.
__device__ void top_p_scan_one_row(Smem& sm, float* s, int base, float mx, float z1, float thr) {
  constexpr int kTileThreads = 128, kItems = 15, kTile = kTileThreads * kItems;
  constexpr int kTiles = (kN + kTile - 1) / kTile;  // 4
  const int t = threadIdx.x;
  const bool active = t < kTiles * kTileThreads;   // 512 virtual threads
  const int tile = t / kTileThreads, vt = t % kTileThreads, lane = t & 31, vw = vt >> 5;
  const int first = tile * kTile + vt * kItems;
  float agg = 0.0f;
  if (active) {
    agg = sorted_prob(sm, first, base, mx, z1);
#pragma unroll
    for (int i = 1; i < kItems; ++i) agg = __fadd_rn(agg, sorted_prob(sm, first + i, base, mx, z1));
  }
  float inc = agg;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const float lower = __shfl_up_sync(0xffffffffu, inc, off);
    if (lane >= off) inc = __fadd_rn(lower, inc);
  }
  float exc = __shfl_up_sync(0xffffffffu, inc, 1);  // lane 0: invalid
  if (active && lane == 31) sm.wagg[tile * 4 + vw] = inc;
  __syncthreads();
  if (active) {
    float tile_agg[kTiles];
    float warp_prefix = 0.0f;
#pragma unroll
    for (int tt = 0; tt < kTiles; ++tt) {
      float b = sm.wagg[tt * 4];
#pragma unroll
      for (int w = 1; w < 4; ++w) {
        if (tt == tile && vw == w) warp_prefix = b;
        b = __fadd_rn(b, sm.wagg[tt * 4 + w]);
      }
      tile_agg[tt] = b;
    }
    if (vw != 0) {
      exc = __fadd_rn(warp_prefix, exc);
      if (lane == 0) exc = warp_prefix;
    }
    if (tile > 0) {
      float tile_prefix = tile_agg[0];
#pragma unroll
      for (int tt = 1; tt < kTiles; ++tt)
        if (tt < tile) tile_prefix = __fadd_rn(tile_prefix, tile_agg[tt]);
      exc = (vt == 0) ? tile_prefix : __fadd_rn(tile_prefix, exc);
    }
    const bool apply = !(tile == 0 && vt == 0);
    float run = 0.0f;
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      const float x = sorted_prob(sm, first + i, base, mx, z1);
      run = (i == 0) ? (apply ? __fadd_rn(exc, x) : x) : __fadd_rn(run, x);
      top_p_remove(sm, s, first + i, base, run, thr);
    }
  }
  __syncthreads();
}

__global__ void __launch_bounds__(kThreads, 1)
samp_sample_kernel(const __nv_bfloat16* __restrict__ logits, int64_t ldl,
                   const float* __restrict__ params, int32_t* __restrict__ history,
                   int32_t* __restrict__ history_len, const uint64_t* __restrict__ seeds,
                   uint64_t* __restrict__ offsets, const int32_t* __restrict__ slot_of_row,
                   const float* __restrict__ noise, int64_t ldn, int32_t* __restrict__ out_tokens,
                   float* __restrict__ out_scores, float* __restrict__ out_probs, unsigned flags) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  const int row = blockIdx.x;
  const int slot = slot_of_row ? slot_of_row[row] : row;
  const int t = threadIdx.x;
  float* s = sm.s;

  // ---- 0. logits (bf16 -> fp32 is exact) ---------------------------------
  const __nv_bfloat16* lrow = logits + static_cast<int64_t>(row) * ldl;
  for (int i = t; i < kN; i += kThreads) s[i] = __bfloat162float(lrow[i]);
  const float* prm = params + static_cast<int64_t>(slot) * KT2N_SAMP_PARAM_COLS;
  int n = history_len[slot];
  n = n < 0 ? 0 : (n > kWindow ? kWindow : n);
  const int32_t* hrow = history + static_cast<int64_t>(slot) * kWindow;
  __syncthreads();

  // ---- 1-2. run penalty, then repetition penalty (warp 0) -----------------
  if (t < 32) {
    if (t == 0 && n > 0) {
      const int last = hrow[kWindow - 1];
      int run = 0;
      for (int e = kWindow - 1; e >= kWindow - n; --e) {
        if (hrow[e] == last) ++run; else break;
      }
      if (last >= kAudioBase && last < kAudioEnd && run >= 10) {
        // excess.to(float32) * 1.3 (python scalar -> float32), scatter_add_(-penalty)
        const float penalty = __fmul_rn(static_cast<float>(run - 9), static_cast<float>(1.3));
        const int col = last - kLow;
        s[col] = __fadd_rn(s[col], -penalty);
      }
    }
    __syncwarp();
    // gathered = s[col]; where(gathered < 0, gathered * 1.1, gathered / 1.1)
    // with torch's CUDA scalar division a * float32(1 / float32(1.1)).
    const float mul11 = static_cast<float>(1.1);
    const float inv11 = __fdiv_rn(1.0f, static_cast<float>(1.1));
    int cols[2];
    float vals[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      const int e = t + 32 * q;
      cols[q] = -1;
      if (e < n) {
        const int col = hrow[kWindow - n + e] - kLow;
        if (col >= 0 && col < kN && col != kExempt0 && col != kExempt1 && col != kExempt2) {
          const float g = s[col];
          cols[q] = col;
          vals[q] = (g < 0.0f) ? __fmul_rn(g, mul11) : __fmul_rn(g, inv11);
        }
      }
    }
    __syncwarp();
#pragma unroll
    for (int q = 0; q < 2; ++q)
      if (cols[q] >= 0) s[cols[q]] = vals[q];  // duplicates write equal values
  }
  __syncthreads();

  // ---- 3. temperature ------------------------------------------------------
  const bool division = (flags & KT2N_SAMP_TEMP_DIVISION) != 0u;
  const float temp = prm[0];
  const float inv_temp = prm[1];
  unsigned int u[kPer];
  float local_max = -CUDART_INF_F;
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int i = t + j * kThreads;
    u[j] = 0u;
    if (i < kN) {
      const float v = division ? __fdiv_rn(s[i], temp) : __fmul_rn(s[i], inv_temp);
      s[i] = v;
      u[j] = ordered_key(v);
      local_max = fmaxf(local_max, v);
    }
  }
  const float mx = block_max(local_max, sm);  // exact; also the max survivor

  // ---- 4. top-k --------------------------------------------------------------
  const float top_k = prm[2];
  if (top_k > 0.0f) {
    int k = static_cast<int>(top_k);  // .to(torch.long): truncation
    // index = clamp(k - 1, 0, keep - 1), keep = min(TOP_K_MAX, width)
    k = k < 1 ? 1 : (k > KT2N_SAMP_TOP_K_MAX ? KT2N_SAMP_TOP_K_MAX : k);
    const float kth = key_to_float(kth_largest_key(u, k, sm));
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const int i = t + j * kThreads;
      if (i < kN && s[i] < kth) s[i] = -CUDART_INF_F;
    }
    __syncthreads();
  }

  // ---- 5. top-p ----------------------------------------------------------------
  const float top_p_thr = prm[3];
  if (top_p_thr >= 0.0f) {
    if (t == 0) sm.bcasti[3] = 0;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const int i = t + j * kThreads;
      if (i < kN && s[i] != -CUDART_INF_F) {
        const int pos = atomicAdd(&sm.bcasti[3], 1);
        sm.keys[pos] = (static_cast<unsigned long long>(sort_key(s[i])) << 32) |
                       static_cast<unsigned long long>(i);
      }
    }
    __syncthreads();
    const int S = sm.bcasti[3];
    int P = 2;
    while (P < S) P <<= 1;
    for (int i = S + t; i < P; i += kThreads) sm.keys[i] = ~0ull;
    __syncthreads();
    bitonic_sort_u64(sm.keys, P);
    // sorted position of survivor r (ascending) is base + r; [0, base) are -inf.
    const int base = kN - S;
    // softmax over the sorted row: torch's per-thread order over positions.
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const int pos = t + j * kThreads;
      if (pos < kN && pos >= base)
        acc = __fadd_rn(acc, exp_shifted(key_to_float(static_cast<unsigned int>(sm.keys[pos - base] >> 32)), mx));
    }
    const float z1 = block_sum_torch(acc, sm);
    if (flags & KT2N_SAMP_SCAN_MULTI_ROW)
      top_p_scan_multi_row(sm, s, base, mx, z1, top_p_thr);
    else
      top_p_scan_one_row(sm, s, base, mx, z1, top_p_thr);
  }

  // ---- 6. min-p -------------------------------------------------------------
  const float min_p = prm[4];
  if (min_p > 0.0f) {
    const float z2 = softmax_denominator(s, mx, sm);
    const float top_prob = __fdiv_rn(exp_shifted(mx, mx), z2);  // = probs.max()
    const float floor_p = __fmul_rn(min_p, top_prob);
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const int i = t + j * kThreads;
      if (i < kN) {
        const float v = s[i];
        const float p = (v == -CUDART_INF_F) ? 0.0f : __fdiv_rn(exp_shifted(v, mx), z2);
        if (p < floor_p) s[i] = -CUDART_INF_F;
      }
    }
    __syncthreads();
  }

  // ---- 7. final softmax and the exponential race ----------------------------
  const float z3 = softmax_denominator(s, mx, sm);
  const uint64_t seed = seeds ? seeds[slot] : 0ull;
  const uint64_t offset = offsets ? offsets[slot] : 0ull;
  const float* nrow = noise ? noise + static_cast<int64_t>(row) * ldn : nullptr;
  float best = -1.0f;
  int best_i = 0x7fffffff;
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int i = t + j * kThreads;
    if (i < kN) {
      const float v = s[i];
      const float p = (v == -CUDART_INF_F) ? 0.0f : __fdiv_rn(exp_shifted(v, mx), z3);
      float q;
      if (nrow) {
        q = __fdiv_rn(p, nrow[i]);
      } else {
        // p == 0 gives 0 / noise = +0 for any (positive, finite) noise.
        q = (p > 0.0f) ? __fdiv_rn(p, torch_exponential(seed, offset, static_cast<uint64_t>(i))) : 0.0f;
      }
      if (out_scores) out_scores[static_cast<int64_t>(row) * kN + i] = v;
      if (out_probs) out_probs[static_cast<int64_t>(row) * kN + i] = p;
      // i ascends within a thread: strict '>' keeps the lowest index on ties.
      if (q > best) {
        best = q;
        best_i = i;
      }
    }
  }
  // argmax combine: greater value, or equal value and lower index (torch).
  {
    const int lane = t & 31;
    const int wid = t >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_down_sync(0xffffffffu, best, off);
      const int oi = __shfl_down_sync(0xffffffffu, best_i, off);
      if (ov > best || (ov == best && oi < best_i)) {
        best = ov;
        best_i = oi;
      }
    }
    __syncthreads();
    if (lane == 0) {
      sm.redf[wid] = best;
      sm.redi[wid] = best_i;
    }
    __syncthreads();
    if (wid == 0) {
      best = sm.redf[lane];
      best_i = sm.redi[lane];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, best, off);
        const int oi = __shfl_down_sync(0xffffffffu, best_i, off);
        if (ov > best || (ov == best && oi < best_i)) {
          best = ov;
          best_i = oi;
        }
      }
      if (lane == 0) sm.bcasti[4] = best_i;
    }
    __syncthreads();
  }
  const int column = sm.bcasti[4];
  const int token = kLow + column;
  if (t == 0) out_tokens[row] = token;

  // ---- 8. state update: history window and the generator offset -----------
  if (flags & KT2N_SAMP_UPDATE_STATE) {
    int32_t* hw = history + static_cast<int64_t>(slot) * kWindow;
    int32_t keep = 0;
    if (t < kWindow - 1) keep = hw[t + 1];
    __syncthreads();
    if (t < kWindow - 1) hw[t] = keep;
    if (t == kWindow - 1) hw[t] = token;
    if (t == 0) {
      history_len[slot] = n + 1 > kWindow ? kWindow : n + 1;
      if (offsets) offsets[slot] = offset + 4ull;
    }
  }
}

__global__ void __launch_bounds__(1024)
samp_exponential_kernel(const uint64_t* __restrict__ seeds, uint64_t* __restrict__ offsets,
                        const int32_t* __restrict__ slot_of_row, float* __restrict__ noise,
                        int64_t ldn, int N, int advance) {
  const int row = blockIdx.x;
  const int slot = slot_of_row ? slot_of_row[row] : row;
  const uint64_t seed = seeds[slot];
  const uint64_t offset = offsets[slot];
  float* out = noise + static_cast<int64_t>(row) * ldn;
  for (int i = threadIdx.x; i < N; i += blockDim.x)
    out[i] = torch_exponential(seed, offset, static_cast<uint64_t>(i));
  if (advance) {
    __syncthreads();
    if (threadIdx.x == 0) offsets[slot] = offset + 4ull;
  }
}

}  // namespace
}  // namespace kt2n_samp

extern "C" int kt2n_samp_sample_smem_bytes(void) {
  return static_cast<int>(sizeof(kt2n_samp::Smem));
}

extern "C" int kt2n_samp_exponential(const uint64_t* seeds, uint64_t* offsets,
                                     const int32_t* slot_of_row, float* noise, int64_t ldn,
                                     int M, int N, int advance, cudaStream_t stream) {
  if (!seeds || !offsets || !noise || M < 1 || N < 1 || ldn < N) return KT2N_SAMP_EARG;
  kt2n_samp::samp_exponential_kernel<<<M, 1024, 0, stream>>>(seeds, offsets, slot_of_row, noise,
                                                              ldn, N, advance);
  return static_cast<int>(cudaPeekAtLastError());
}

extern "C" int kt2n_samp_sample(const void* logits, int64_t ldl, const float* params,
                                int32_t* history, int32_t* history_len, const uint64_t* seeds,
                                uint64_t* offsets, const int32_t* slot_of_row, const float* noise,
                                int64_t ldn, int32_t* out_tokens, float* out_scores,
                                float* out_probs, int M, unsigned flags, cudaStream_t stream) {
  if (!logits || !params || !history || !history_len || !out_tokens || M < 1 ||
      ldl < KT2N_SAMP_AUDIO_N)
    return KT2N_SAMP_EARG;
  if (!noise && (!seeds || !offsets)) return KT2N_SAMP_EARG;
  if (noise && ldn < KT2N_SAMP_AUDIO_N) return KT2N_SAMP_EARG;
  if ((flags & KT2N_SAMP_UPDATE_STATE) && !offsets && !noise) return KT2N_SAMP_EARG;
  const int smem = static_cast<int>(sizeof(kt2n_samp::Smem));
  cudaError_t err = cudaFuncSetAttribute(kt2n_samp::samp_sample_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (err != cudaSuccess) return static_cast<int>(err);
  kt2n_samp::samp_sample_kernel<<<M, kt2n_samp::kThreads, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(logits), ldl, params, history, history_len, seeds, offsets,
      slot_of_row, noise, ldn, out_tokens, out_scores, out_probs, flags);
  return static_cast<int>(cudaPeekAtLastError());
}
