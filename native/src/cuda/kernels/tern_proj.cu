// kt2n k1-ternary: exact W2A16 ternary projections (decode M = 1..16 and prefill M > 16), GPU repack from the
// canonical pack_exact format, decode-back for lossless proofs.  Started from tk_tc4 (+PDL) of the lab kernel
// bake-off; same per-row arithmetic family (K permutation inside 256-chunks, PRMT decode, fixed K split and
// fixed partial-sum order), with a tile-major weight layout, a residual epilogue and an M-tiled prefill kernel
// whose per-row arithmetic is identical to the decode kernel's.
//
// Build standalone: nvcc -O3 -std=c++20 -Xcompiler -fPIC -shared -Iinclude -gencode arch=compute_120,code=sm_120
// (also 90 and 100).  Never --use_fast_math / -ftz / -prec-div=false: the SiLU epilogue must equal torch's.
// -DKT2N_TERN_TUNE=1: decode tuning build (alternative K splits / accumulators / shapes); =2: prefill shapes.
#if __has_include("kt2n/kernels/tern_proj.h")
#include "kt2n/kernels/tern_proj.h"
#else
#include "../../../include/kt2n/kernels/tern_proj.h"
#endif
#include "tern_proj.cuh"
#include "tern_wide_index.cuh"
#include "kt2n/cuda_caps.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef KT2N_TERN_PRODUCTION_ORDER
#define KT2N_TERN_PRODUCTION_ORDER 0
#endif

#ifndef KT2N_TERN_TUNE
#define KT2N_TERN_TUNE 0
#endif

// L3 sprint 3: the >16-row shape experiments (extra M tilings, register caps, one accumulator chain) are built only
// with -DKT2N_TERN_EXPERIMENTS=1 (all measured slower).  Default builds stay lean.
#ifndef KT2N_TERN_EXPERIMENTS
#define KT2N_TERN_EXPERIMENTS 0
#endif

using bf16 = __nv_bfloat16;
using namespace kt2n_tern;

namespace {

// Experimental production scalar order, selected only in explicit proof builds.
// The default kernel retains its established row-invariant arithmetic. Production
// uses a scalar reduction for B1 and Tensor Cores for B>=2; those are distinct
// contracts. This branch has only a B1 retained-input proof until wider checks.
#if KT2N_TERN_PRODUCTION_ORDER
template <int BW>
__device__ __forceinline__ float production_partial(const bf16* x, const uint32_t* codes,
                                                    const uint16_t* scales, int row, int k) {
    const int word = threadIdx.x / 16, sub = threadIdx.x % 16, chunks = k / 256;
    float acc = 0.f;
    for (int base = 0; base < k; base += BW * 16) {
        const int col = base + word * 16 + sub, wi = col / 16;
        const int chunk = wi / 16, t = (wi / 4) % 4, ii = wi % 4;
        const size_t offset = (((size_t)(row / 8) * chunks + chunk) * 32 + (row % 8) * 4 + t) * 4 + ii;
        const uint32_t packed = codes[offset];
        const int code = (packed >> (2 * (sub / 4) + 8 * (sub % 4))) & 3;
        const uint32_t scale = scales[(((size_t)(row / 8) * chunks + chunk) * 8 + row % 8) * 2 + (col / 128) % 2];
        const uint32_t bits = code == 1 ? scale << 16 : code == 2 ? (scale << 16) ^ 0x80000000u : 0;
        acc = __fadd_rn(acc, __fmul_rn(__uint_as_float(bits), __bfloat162float(x[col])));
    }
    for (int delta = 8; delta; delta /= 2)
        acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, delta, 16));
    return acc;
}

template <int BW, int EPI>
__global__ void production_b1_kernel(const bf16* x, const uint32_t* codes, const uint16_t* scales,
                                      bf16* y, const bf16* residual, int k) {
    pdl_wait();
    const int lane = threadIdx.x, sub = lane % 16, word = lane / 16;
    const int output = blockIdx.x, row = EPI == KT2N_TERN_EPI_SILU_MUL ? 2 * output : output;
    float a = production_partial<BW>(x, codes, scales, row, k), b = 0.f;
    if constexpr (EPI == KT2N_TERN_EPI_SILU_MUL)
        b = production_partial<BW>(x, codes, scales, row + 1, k);
    __shared__ float partial[2][BW];
    if (sub == 0) { partial[0][word] = a; partial[1][word] = b; }
    __syncthreads();
    for (int delta = BW / 2; delta; delta /= 2) {
        if (lane < delta) {
            partial[0][lane] = __fadd_rn(partial[0][lane], partial[0][lane + delta]);
            if constexpr (EPI == KT2N_TERN_EPI_SILU_MUL)
                partial[1][lane] = __fadd_rn(partial[1][lane], partial[1][lane + delta]);
        }
        __syncthreads();
    }
    if (lane == 0) {
        if constexpr (EPI == KT2N_TERN_EPI_SILU_MUL) y[output] = silu_mul_bf16(partial[0][0], partial[1][0]);
        else if constexpr (EPI == KT2N_TERN_EPI_RESIDUAL)
            y[output] = __float2bfloat16_rn(__fadd_rn(bf16_round(partial[0][0]), __bfloat162float(residual[output])));
        else y[output] = __float2bfloat16_rn(partial[0][0]);
    }
    // Dependents observe the complete output of each CTA after the store.
    __syncthreads();
    pdl_launch_dependents();
}
#endif

// ------------------------------------------------------------------------------------------------------------
// Family table.  WK (K partial sums) and ACC (accumulators per partial) fix the rounding; they never depend on
// M, on the launch shape or on the decode/prefill path.  Frozen per family (tuned on the RTX PRO 6000).
// ------------------------------------------------------------------------------------------------------------
#ifndef KT2N_TERN_QKV_WK
#define KT2N_TERN_QKV_WK 8
#endif
#ifndef KT2N_TERN_QKV_ACC
#define KT2N_TERN_QKV_ACC 4
#endif
#ifndef KT2N_TERN_O_WK
#define KT2N_TERN_O_WK 8
#endif
#ifndef KT2N_TERN_O_ACC
#define KT2N_TERN_O_ACC 4
#endif
#ifndef KT2N_TERN_GU_WK
#define KT2N_TERN_GU_WK 4
#endif
#ifndef KT2N_TERN_GU_ACC
#define KT2N_TERN_GU_ACC 4
#endif
#ifndef KT2N_TERN_DOWN_WK
#define KT2N_TERN_DOWN_WK 12
#endif
#ifndef KT2N_TERN_DOWN_ACC
#define KT2N_TERN_DOWN_ACC 2
#endif
// Trigger dependents right after the weight loads (1) or after the main loop (0).  Never changes results.
#ifndef KT2N_TERN_EARLY_TRIGGER
#define KT2N_TERN_EARLY_TRIGGER 1
#endif

struct FamilyCfg {
    int k;    // K
    int wk;   // K partial sums (warps splitting K in the decode kernel)
    int acc;  // accumulators per partial (MMA step s -> accumulator s % acc)
};
// qkv / o / gate_up: K = 2048 = 8 chunks; down: K = 6144 = 24 chunks.
constexpr FamilyCfg family_cfg(int fam) {
    return fam == KT2N_TERN_FAM_QKV ? FamilyCfg{2048, KT2N_TERN_QKV_WK, KT2N_TERN_QKV_ACC}
         : fam == KT2N_TERN_FAM_O ? FamilyCfg{2048, KT2N_TERN_O_WK, KT2N_TERN_O_ACC}
         : fam == KT2N_TERN_FAM_GATE_UP ? FamilyCfg{2048, KT2N_TERN_GU_WK, KT2N_TERN_GU_ACC}
         : FamilyCfg{6144, KT2N_TERN_DOWN_WK, KT2N_TERN_DOWN_ACC};
}

#if KT2N_TERN_TUNE
int g_wk_override[4] = {0, 0, 0, 0};
int g_acc_override[4] = {0, 0, 0, 0};
int g_early_override = -1;
#endif

struct Args {
    const bf16* x;
    int ldx;
    int m;
    const uint4* w;
    const uint16_t* s;
    bf16* y;
    int ldy;
    const bf16* r;
    int ldr;
    int n;
    int c;  // K / 256
    bool pdl;
    int early;
    cudaStream_t stream;
    const bf16* lnw = nullptr;  // NORM (L3): RMSNorm weight [K] applied to X in the prologue
    float eps = 0.f;
};

// ------------------------------------------------------------------------------------------------------------
// Epilogue of one 8-column output tile of a 16-row MMA tile: v[0], v[1] = row row0 + g (cols 2t, 2t+1),
// v[2], v[3] = row row0 + 8 + g.  Rows >= M are never written.
// ------------------------------------------------------------------------------------------------------------
template <int EPI>
__device__ __forceinline__ void store_tile(const float (&v)[4], int tile, int row0, int M, int g, int t, bf16* Y,
                                           int ldy, const bf16* R, int ldr) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int row = row0 + h * 8 + g;
        if (row >= M) continue;
        const float c0 = v[2 * h], c1 = v[2 * h + 1];
        if constexpr (EPI == KT2N_TERN_EPI_SILU_MUL) {
            Y[(size_t)row * ldy + tile * 4 + t] = silu_mul_bf16(c0, c1);
        } else {
            const int col = tile * 8 + 2 * t;
            __nv_bfloat162 out;
            if constexpr (EPI == KT2N_TERN_EPI_RESIDUAL)
                out = residual_bf16x2(c0, c1, ld_act_bf16x2(R + (size_t)row * ldr + col));
            else
                out = __floats2bfloat162_rn(c0, c1);
            *reinterpret_cast<__nv_bfloat162*>(Y + (size_t)row * ldy + col) = out;
        }
    }
}

#if KT2N_TERN_PRODUCTION_ORDER
// Production TC regime: contiguous K16 MMA chain, one accumulator,
// family split factors qkv2/o4/gate_up1/down4 and ordered FP32 merge.
__device__ uint16_t production_weight_bits(const uint32_t* codes,const uint16_t* scales,int n,int col,int chunks) {
 const int wi=col/16,sub=col%16,chunk=wi/16,t=(wi/4)%4,ii=wi%4;
 const uint32_t code=(codes[(((size_t)(n/8)*chunks+chunk)*32+(n%8)*4+t)*4+ii]>>(2*(sub/4)+8*(sub%4)))&3;
 const uint16_t scale=scales[(((size_t)(n/8)*chunks+chunk)*8+n%8)*2+(col/128)%2];
 return code==1?scale:code==2?scale^0x8000:0;
}
__device__ uint32_t production_weight_pair(const uint32_t* codes,const uint16_t* scales,int n,int col,int chunks) {
 return (uint32_t)production_weight_bits(codes,scales,n,col,chunks)|((uint32_t)production_weight_bits(codes,scales,n,col+1,chunks)<<16);
}
template<int SPLITS, int EPI>
__global__ void production_tc_kernel(const bf16* x,const uint32_t* codes,const uint16_t* scales,bf16* y,int m,int n,int k,int ldx,int ldy,const bf16* residual,int ldr) {
 (void)n;
 pdl_wait();
 const int lane=threadIdx.x%32,split=threadIdx.x/32,g=lane/4,t=lane%4;
 const int tile=blockIdx.x,row0=blockIdx.y*16,chunks=k/256;
 float c[4]={0,0,0,0};
 for(int col=split*(k/SPLITS);col<(split+1)*(k/SPLITS);col+=16) {
  const int a=col+2*t,b=col+2*t+8;
  const uint32_t a0=row0+g<m?*reinterpret_cast<const uint32_t*>(x+(size_t)(row0+g)*ldx+a):0;
  const uint32_t a1=row0+g+8<m?*reinterpret_cast<const uint32_t*>(x+(size_t)(row0+g+8)*ldx+a):0;
  const uint32_t a2=row0+g<m?*reinterpret_cast<const uint32_t*>(x+(size_t)(row0+g)*ldx+b):0;
  const uint32_t a3=row0+g+8<m?*reinterpret_cast<const uint32_t*>(x+(size_t)(row0+g+8)*ldx+b):0;
  mma_16816(c,a0,a1,a2,a3,production_weight_pair(codes,scales,tile*8+g,a,chunks),production_weight_pair(codes,scales,tile*8+g,b,chunks));
 }
 __shared__ float parts[SPLITS][4][32];
 for(int q=0;q<4;q++)parts[split][q][lane]=c[q];
 __syncthreads();
 if(split)return;
 for(int q=0;q<4;q++)for(int s=1;s<SPLITS;s++)c[q]=__fadd_rn(c[q],parts[s][q][lane]);
 store_tile<EPI>(c,tile,row0,m,g,t,y,ldy,residual,ldr);
 pdl_launch_dependents();
}
#endif

// ------------------------------------------------------------------------------------------------------------
// Decode kernel: M <= 16 rows, one 16-row MMA tile.  Block = WN groups x WK warps; warp (wn, w) owns NT output
// tiles (8 rows of W each) and chunks [w*CPW, (w+1)*CPW).  Inside a warp, MMA step s = 16*i + j (chunk i,
// step j) accumulates into accumulator j % ACC (ACC independent chains: a shorter dependent MMA chain), and the
// warp's partial is ((acc0 + acc1) + ...).  The WK partials are summed in warp order ((p0 + p1) + p2) + ...
// by warp 0.  All weights/scales of the warp are loaded into
// registers before griddepcontrol.wait (PDL overlap); activations after.  `early` = trigger dependents right
// after the weight loads (a dependent then prefetches its weights while this kernel runs; never changes results).
// ------------------------------------------------------------------------------------------------------------
// NORM (L3, 10 Oct; EXPERIMENTAL, off by default: measured slower;
// statistical equality, not bitwise): X is the residual stream h and the kernel computes
// RMSNorm(h) * lnw on the fly: the MMA reads bf16(h_k * lnw_k), the row's sum of squares is accumulated from the
// same loads, and the per-row 1/rms (fp32, mean over K + eps) scales the fp32 accumulators before the epilogue
// (a per-row scalar factors out of the dot product).  Replaces the separate residual + RMSNorm kernel.
template <int WK, int CPW, int ACC, int NT, int WN, bool HI, int EPI, bool NORM = false>
__global__ void __launch_bounds__(WK * WN * 32)
tern_dec_kernel(const bf16* X, int ldx, int M, const uint4* __restrict__ Wt, const uint16_t* __restrict__ St,
                bf16* Y, int ldy, const bf16* R, int ldr, int C, int early, const bf16* __restrict__ lnw = nullptr,
                float eps = 0.f) {
    static_assert(16 % ACC == 0, "ACC must divide 16");
    const int wid = threadIdx.x >> 5;
    const int warp = wid % WK;
    const int wn = wid / WK;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int chunk0 = warp * CPW;
    const int tile0 = (blockIdx.x * WN + wn) * NT;

    uint4 w[NT][CPW];
    uint32_t ta[NT][CPW], tb[NT][CPW];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int i = 0; i < CPW; ++i) {
            const size_t tc = (size_t)(tile0 + nt) * C + chunk0 + i;
            w[nt][i] = ld_weights(Wt + tc * 32 + lane);
            const uint32_t s = __ldg(St + tc * 16 + g * 2 + (t >> 1));
            scale_tables(s, ta[nt][i], tb[nt][i]);
        }
    }
    // NORM: the norm weight (constant, K bf16 = WK * CPW * 32 uint4) is staged in shared memory before the wait,
    // so the prologue costs one shared load per 8 activations and few extra registers (keeping the grid in one
    // wave: register growth halves the blocks per SM).
    // The trigger comes first: a shared store waits for its global load, which would delay the dependents.
    if (early) pdl_launch_dependents();
    __shared__ uint4 s_lnw[NORM ? WK * CPW * 32 : 1];
    if constexpr (NORM) {
        const uint4* lw = reinterpret_cast<const uint4*>(lnw);
        for (int i = threadIdx.x; i < WK * CPW * 32; i += WK * WN * 32) s_lnw[i] = __ldg(lw + i);
    }
    // Everything above reads only constant weights; X / R are written by earlier kernels.
    pdl_wait();
    if constexpr (NORM) __syncthreads();

    const bool r0 = g < M;
    const bool r1 = HI && (g + 8) < M;
    const uint4* xr0 = reinterpret_cast<const uint4*>(X + (size_t)g * ldx);
    const uint4* xr1 = reinterpret_cast<const uint4*>(X + (size_t)(g + 8) * ldx);
    float c[NT][ACC][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int a = 0; a < ACC; ++a) c[nt][a][0] = c[nt][a][1] = c[nt][a][2] = c[nt][a][3] = 0.f;
    float ss0 = 0.f, ss1 = 0.f;  // NORM: sum of squares of rows g, g + 8 over this lane's k
#pragma unroll
    for (int i = 0; i < CPW; ++i) {
        const int base = ((chunk0 + i) * kChunk + t * 64) >> 3;
        uint4 xa[8], xb[8];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            xa[q] = r0 ? ld_act(xr0 + base + q) : make_uint4(0, 0, 0, 0);
            if (HI) xb[q] = r1 ? ld_act(xr1 + base + q) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if constexpr (NORM) {
                if ((j & 1) == 0) {  // just in time, one 16-byte vector at a time (keeps registers low)
                    asm volatile("" ::: "memory");
                    const uint4 wv = s_lnw[base + (j >> 1)];
                    xa[j >> 1] = norm_mul8(xa[j >> 1], wv, ss0);
                    if (HI) xb[j >> 1] = norm_mul8(xb[j >> 1], wv, ss1);
                }
            }
            const uint4 va = xa[j >> 1];
            const uint32_t a0 = (j & 1) ? va.z : va.x;
            const uint32_t a2 = (j & 1) ? va.w : va.y;
            uint32_t a1 = 0u, a3 = 0u;
            if (HI) {
                const uint4 vb = xb[j >> 1];
                a1 = (j & 1) ? vb.z : vb.x;
                a3 = (j & 1) ? vb.w : vb.y;
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                uint32_t b0, b1;
                decode_step(word_of(w[nt][i], j >> 2), j & 3, ta[nt][i], tb[nt][i], b0, b1);
                mma_16816(c[nt][j % ACC], a0, a1, a2, a3, b0, b1);
            }
        }
    }
    if (!early) pdl_launch_dependents();
    float p[NT][4];  // the warp's partial: ((acc0 + acc1) + acc2) + ...
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            float v = c[nt][0][q];
#pragma unroll
            for (int a = 1; a < ACC; ++a) v += c[nt][a][q];
            p[nt][q] = v;
        }
    if constexpr (NORM) {  // the quad (t = 0..3) holds one row's k range of this warp
        ss0 += __shfl_xor_sync(0xffffffffu, ss0, 1);
        ss0 += __shfl_xor_sync(0xffffffffu, ss0, 2);
        if (HI) {
            ss1 += __shfl_xor_sync(0xffffffffu, ss1, 1);
            ss1 += __shfl_xor_sync(0xffffffffu, ss1, 2);
        }
    }
    if constexpr (WK > 1) {
        // Warp 0 finishes: its own partial stays in registers, the others pass theirs through shared memory.
        // (Measured: spreading the finishing work over the warps was slower for M <= 8.)
        __shared__ float red[WN][WK][NT][4][32];  // [q][lane]: conflict-free
        __shared__ float ssred[NORM ? WN : 1][NORM ? WK : 1][2][8];
        if (warp != 0) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int q = 0; q < (HI ? 4 : 2); ++q) red[wn][warp][nt][q][lane] = p[nt][q];
            if constexpr (NORM) {
                if (t == 0) {
                    ssred[wn][warp][0][g] = ss0;
                    ssred[wn][warp][1][g] = ss1;
                }
            }
        }
        __syncthreads();
        if (warp != 0) return;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int q = 0; q < (HI ? 4 : 2); ++q) {
                float s = p[nt][q];
#pragma unroll
                for (int ww = 1; ww < WK; ++ww) s += red[wn][ww][nt][q][lane];
                p[nt][q] = s;
            }
        if constexpr (NORM) {
#pragma unroll
            for (int ww = 1; ww < WK; ++ww) {
                ss0 += ssred[wn][ww][0][g];
                ss1 += ssred[wn][ww][1][g];
            }
        }
    }
    if constexpr (NORM) {
        const float inv_k = 1.0f / static_cast<float>(C * kChunk);
        const float inv0 = rsqrtf(__fadd_rn(__fmul_rn(ss0, inv_k), eps));
        const float inv1 = rsqrtf(__fadd_rn(__fmul_rn(ss1, inv_k), eps));
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            p[nt][0] *= inv0;
            p[nt][1] *= inv0;
            p[nt][2] *= inv1;
            p[nt][3] *= inv1;
        }
    }
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) store_tile<EPI>(p[nt], tile0 + nt, 0, M, g, t, Y, ldy, R, ldr);
}

// ------------------------------------------------------------------------------------------------------------
// Tiled kernel for M > 16 (prefill): the decode kernel's block (WK warps splitting K exactly like the decode
// kernel, same ACC accumulators, same fold and partial-sum order) over an M grid (blockIdx.y = 16*MT rows), each
// warp running MT 16-row tiles with every decoded weight fragment reused across them.  A row's arithmetic is the
// decode kernel's, operation for operation.  (Measured: looping a block over several M blocks with its weights
// resident was slower than one M block per block, so the grid covers M.)
// ------------------------------------------------------------------------------------------------------------
// L3 sprint 7 EXPERIMENT (PS > 0, built only with -DKT2N_TERN_EXPERIMENTS=1, shape code mt = 160 + PS): the
// activation vectors go through a PS-stage cp.async ring in shared memory, one private slot per lane, issued PS - 1
// iterations ahead, instead of plain loads the compiler can hoist only as far as registers allow.  The MMA inputs and
// their order are unchanged (same values, same chains), so the logits stay byte-identical.
template <int WK, int CPW, int ACC, int NT, int WN, int MT, int EPI, int MINB = 1, int PS = 0>
__global__ void __launch_bounds__(WK * WN * 32, MINB)
tern_tile_kernel(const bf16* X, int ldx, int M, const uint4* __restrict__ Wt, const uint16_t* __restrict__ St,
                 bf16* Y, int ldy, const bf16* R, int ldr, int C, int early) {
    static_assert(16 % ACC == 0, "ACC must divide 16");
    static_assert(PS == 0 || PS >= 2, "PS: 0 (off) or >= 2 stages");
    constexpr int ITEMS = NT * MT;
    const int wid = threadIdx.x >> 5;
    const int warp = wid % WK;
    const int wn = wid / WK;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int chunk0 = warp * CPW;
    const int tile0 = (blockIdx.x * WN + wn) * NT;

    uint4 w[NT][CPW];
    uint32_t ta[NT][CPW], tb[NT][CPW];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int i = 0; i < CPW; ++i) {
            const size_t tc = (size_t)(tile0 + nt) * C + chunk0 + i;
            w[nt][i] = ld_weights(Wt + tc * 32 + lane);
            const uint32_t s = __ldg(St + tc * 16 + g * 2 + (t >> 1));
            scale_tables(s, ta[nt][i], tb[nt][i]);
        }
    }
    if (early) pdl_launch_dependents();
    pdl_wait();

    {
        const int m0 = blockIdx.y * 16 * MT;
        const uint4* xr[MT][2];
        bool rv[MT][2];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + mt * 16 + h * 8 + g;
                rv[mt][h] = row < M;
                xr[mt][h] = reinterpret_cast<const uint4*>(X + (size_t)(rv[mt][h] ? row : 0) * ldx);
            }
        float c[NT][MT][ACC][4];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int a = 0; a < ACC; ++a)
                    c[nt][mt][a][0] = c[nt][mt][a][1] = c[nt][mt][a][2] = c[nt][mt][a][3] = 0.f;
        // PS > 0: lane-private ring [warp][stage][2 * mt + h][lane] in dynamic shared memory (a lane only reads
        // what it copied itself, so the copy's own wait_group is the only synchronisation needed).  PS = 0 kernels
        // launch with no dynamic shared memory and never touch it.
        extern __shared__ uint4 tile_ring[];
        [[maybe_unused]] auto slot = [&](int it, int k) -> uint4* {
            return tile_ring + (((size_t)wid * (PS > 0 ? PS : 1) + it % (PS > 0 ? PS : 1)) * (2 * MT) + k) * 32 + lane;
        };
        [[maybe_unused]] auto issue = [&](int it) {
            const int nb = ((chunk0 + (it >> 3)) * kChunk + t * 64) >> 3;
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
#if defined(KT2N_TILE_RING_CA) && KT2N_TILE_RING_CA
                    // .ca: allocate in L1, so CTAs on one SM that read the same activation rows share them in L1
                    const unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(slot(it, 2 * mt + h)));
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(sa),
                                 "l"(xr[mt][h] + nb + (it & 7)), "r"(rv[mt][h] ? 16 : 0) : "memory");
#else
                    cp_async16(slot(it, 2 * mt + h), xr[mt][h] + nb + (it & 7), rv[mt][h] ? 16 : 0);
#endif
                }
        };
        if constexpr (PS > 0) {
#pragma unroll
            for (int s0 = 0; s0 < PS - 1; ++s0) {
                if (s0 < CPW * 8) issue(s0);
                cp_async_commit();
            }
        }
#pragma unroll
        for (int i = 0; i < CPW; ++i) {
            const int base = ((chunk0 + i) * kChunk + t * 64) >> 3;
#pragma unroll
            for (int jp = 0; jp < 8; ++jp) {
                uint4 xa[MT], xb[MT];
                if constexpr (PS > 0) {
                    const int it = i * 8 + jp;
                    if (it + PS - 1 < CPW * 8) issue(it + PS - 1);
                    cp_async_commit();
                    cp_async_wait<PS - 1>();
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        xa[mt] = lds128(slot(it, 2 * mt));
                        xb[mt] = lds128(slot(it, 2 * mt + 1));
                    }
                } else {
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        xa[mt] = rv[mt][0] ? ld_act(xr[mt][0] + base + jp) : make_uint4(0, 0, 0, 0);
                        xb[mt] = rv[mt][1] ? ld_act(xr[mt][1] + base + jp) : make_uint4(0, 0, 0, 0);
                    }
                }
#pragma unroll
                for (int jj = 0; jj < 2; ++jj) {
                    const int j = 2 * jp + jj;
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        uint32_t b0, b1;
                        decode_step(word_of(w[nt][i], j >> 2), j & 3, ta[nt][i], tb[nt][i], b0, b1);
#pragma unroll
                        for (int mt = 0; mt < MT; ++mt) {
                            const uint32_t a0 = jj ? xa[mt].z : xa[mt].x;
                            const uint32_t a2 = jj ? xa[mt].w : xa[mt].y;
                            const uint32_t a1 = jj ? xb[mt].z : xb[mt].x;
                            const uint32_t a3 = jj ? xb[mt].w : xb[mt].y;
                            mma_16816(c[nt][mt][j % ACC], a0, a1, a2, a3, b0, b1);
                        }
                    }
                }
            }
        }
        float p[NT][MT][4];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    float v = c[nt][mt][0][q];
#pragma unroll
                    for (int a = 1; a < ACC; ++a) v += c[nt][mt][a][q];
                    p[nt][mt][q] = v;
                }
        if constexpr (WK > 1) {
            __shared__ float red[WN][WK][ITEMS][4][32];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                    for (int q = 0; q < 4; ++q) red[wn][warp][nt * MT + mt][q][lane] = p[nt][mt][q];
            __syncthreads();
            for (int it = warp; it < ITEMS; it += WK) {
                float v[4];
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    float s = red[wn][0][it][q][lane];
#pragma unroll
                    for (int ww = 1; ww < WK; ++ww) s += red[wn][ww][it][q][lane];
                    v[q] = s;
                }
                store_tile<EPI>(v, tile0 + it / MT, m0 + 16 * (it % MT), M, g, t, Y, ldy, R, ldr);
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) store_tile<EPI>(p[nt][mt], tile0 + nt, m0 + 16 * mt, M, g, t, Y, ldy, R, ldr);
        }
    }
    if (!early) pdl_launch_dependents();
}

// ------------------------------------------------------------------------------------------------------------
// L3 sprint 4: swapped-operand decode kernel for 17..32 rows (shape code mt = 1000 + 10 * MINB + acc2, nt = NP pairs).
// The tiled kernel above runs one CTA per 16-row block, so at 17..32 rows every weight fragment is loaded and decoded
// twice (two CTAs) and half the second MMA tile is padding.  Here the weights are the MMA's A operand (16 output
// columns = two 8-column tiles) and the activations its B operand (8 rows per MMA): one CTA covers all rows (R8 groups
// of 8), each weight fragment is decoded once and feeds R8 MMAs.  The fragments are the same registers as in the
// decode kernel (weight b0/b1 of tiles 2p, 2p+1 -> a0..a3; activation a0/a2 of a row -> b0/b1), K split, ACC chains
// and fold order are the decode kernel's, so with the family's ACC (acc2 = 0) each output is the same dot product in
// the same order; acc2 = 1 uses 2 chains (fewer registers; not the k1 contract).
// EXPERIMENT, measured slower in every family at 20 and 32 rows (lane log, sprint 4): built only with
// -DKT2N_TERN_EXPERIMENTS=1.  (With the family's ACC its logits are byte-identical to the tiled kernel's.)
// ------------------------------------------------------------------------------------------------------------
#if KT2N_TERN_EXPERIMENTS
template <int WK, int CPW, int ACC, int NP, int WN, int R8, int EPI, int MINB = 1>
__global__ void __launch_bounds__(WK * WN * 32, MINB)
tern_swap_kernel(const bf16* X, int ldx, int M, const uint4* __restrict__ Wt, const uint16_t* __restrict__ St,
                 bf16* Y, int ldy, int C, int early) {
    static_assert(16 % ACC == 0, "ACC must divide 16");
    static_assert(EPI != KT2N_TERN_EPI_RESIDUAL, "STORE / SILU_MUL only");
    constexpr int NT = 2 * NP;
    constexpr int ITEMS = NP * R8;
    const int wid = threadIdx.x >> 5;
    const int warp = wid % WK;
    const int wn = wid / WK;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int chunk0 = warp * CPW;
    const int tile0 = (blockIdx.x * WN + wn) * NT;

    uint4 w[NT][CPW];
    uint32_t ta[NT][CPW], tb[NT][CPW];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int i = 0; i < CPW; ++i) {
            const size_t tc = (size_t)(tile0 + nt) * C + chunk0 + i;
            w[nt][i] = ld_weights(Wt + tc * 32 + lane);
            const uint32_t s = __ldg(St + tc * 16 + g * 2 + (t >> 1));
            scale_tables(s, ta[nt][i], tb[nt][i]);
        }
    }
    if (early) pdl_launch_dependents();
    pdl_wait();

    const uint4* xr[R8];
    bool rv[R8];
#pragma unroll
    for (int r = 0; r < R8; ++r) {
        const int row = 8 * r + g;
        rv[r] = row < M;
        xr[r] = reinterpret_cast<const uint4*>(X + (size_t)(rv[r] ? row : 0) * ldx);
    }
    float c[NP][R8][ACC][4];
#pragma unroll
    for (int p = 0; p < NP; ++p)
#pragma unroll
        for (int r = 0; r < R8; ++r)
#pragma unroll
            for (int a = 0; a < ACC; ++a) c[p][r][a][0] = c[p][r][a][1] = c[p][r][a][2] = c[p][r][a][3] = 0.f;
#pragma unroll
    for (int i = 0; i < CPW; ++i) {
        const int base = ((chunk0 + i) * kChunk + t * 64) >> 3;
#pragma unroll
        for (int jp = 0; jp < 8; ++jp) {
            uint4 xv[R8];
#pragma unroll
            for (int r = 0; r < R8; ++r) xv[r] = rv[r] ? ld_act(xr[r] + base + jp) : make_uint4(0, 0, 0, 0);
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const int j = 2 * jp + jj;
#pragma unroll
                for (int p = 0; p < NP; ++p) {
                    uint32_t b0a, b1a, b0b, b1b;
                    decode_step(word_of(w[2 * p][i], j >> 2), j & 3, ta[2 * p][i], tb[2 * p][i], b0a, b1a);
                    decode_step(word_of(w[2 * p + 1][i], j >> 2), j & 3, ta[2 * p + 1][i], tb[2 * p + 1][i], b0b, b1b);
#pragma unroll
                    for (int r = 0; r < R8; ++r) {
                        const uint32_t x0 = jj ? xv[r].z : xv[r].x;  // row 8r + g, logical k (2t, 2t+1)
                        const uint32_t x1 = jj ? xv[r].w : xv[r].y;  // logical k (2t+8, 2t+9)
                        mma_16816(c[p][r][j % ACC], b0a, b0b, b1a, b1b, x0, x1);
                    }
                }
            }
        }
    }
    if (!early) pdl_launch_dependents();
    float pr[NP][R8][4];
#pragma unroll
    for (int p = 0; p < NP; ++p)
#pragma unroll
        for (int r = 0; r < R8; ++r)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                float v = c[p][r][0][q];
#pragma unroll
                for (int a = 1; a < ACC; ++a) v += c[p][r][a][q];
                pr[p][r][q] = v;
            }
    // v[0], v[1]: output column g of tile 2p, rows 8r + 2t, 8r + 2t + 1; v[2], v[3]: the same of tile 2p + 1.
    auto store = [&](const float (&v)[4], int p, int r) {
        const int row0 = 8 * r + 2 * t;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int tile = tile0 + 2 * p + h;
            const float v0 = v[2 * h], v1 = v[2 * h + 1];
            if constexpr (EPI == KT2N_TERN_EPI_SILU_MUL) {
                // gate = even column, up = odd column of the 8-column tile: lanes g and g ^ 1 swap one row each
                const float send = (g & 1) ? v0 : v1;
                const float recv = __shfl_xor_sync(0xffffffffu, send, 4);
                const int row = row0 + (g & 1);
                const float gate = (g & 1) ? recv : v0, up = (g & 1) ? v1 : recv;
                if (row < M) Y[(size_t)row * ldy + tile * 4 + (g >> 1)] = silu_mul_bf16(gate, up);
            } else {
                if (row0 < M) Y[(size_t)row0 * ldy + tile * 8 + g] = __float2bfloat16_rn(v0);
                if (row0 + 1 < M) Y[(size_t)(row0 + 1) * ldy + tile * 8 + g] = __float2bfloat16_rn(v1);
            }
        }
    };
    if constexpr (WK > 1) {
        __shared__ float red[WN][WK][ITEMS][4][32];
#pragma unroll
        for (int p = 0; p < NP; ++p)
#pragma unroll
            for (int r = 0; r < R8; ++r)
#pragma unroll
                for (int q = 0; q < 4; ++q) red[wn][warp][p * R8 + r][q][lane] = pr[p][r][q];
        __syncthreads();
        for (int it = warp; it < ITEMS; it += WK) {  // warp-uniform: the SILU shuffle needs the full warp
            float v[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                float s = red[wn][0][it][q][lane];
#pragma unroll
                for (int ww = 1; ww < WK; ++ww) s += red[wn][ww][it][q][lane];
                v[q] = s;
            }
            store(v, it / R8, it % R8);
        }
    } else {
#pragma unroll
        for (int p = 0; p < NP; ++p)
#pragma unroll
            for (int r = 0; r < R8; ++r) store(pr[p][r], p, r);
    }
}
#endif  // KT2N_TERN_EXPERIMENTS

// ------------------------------------------------------------------------------------------------------------
// Shared-memory GEMM kernel for large M (prefill).  Block = BMW x BNW warps, block tile (16*BMW) rows x
// (8*NTW*BNW) columns, K streamed chunk by chunk (256) through a 2-stage cp.async pipeline (X rows and the
// tile-major codes/scales of the block's columns).  Each warp computes one 16-row tile x NTW column tiles over the
// whole K and reproduces the decode kernel's arithmetic exactly: per K split w (CPW chunks) ACC accumulator chains
// (step j -> chain j % ACC, chunks in order), folded ((a0 + a1) + ...) into p_w, and tot = ((p0 + p1) + ...).
// X smem rows are 512 B (one chunk); 16-B unit u of row r is stored at u ^ ((2*(u>>3) + (r&1)) & 7) so the lanes'
// 16-B reads (unit 8t + q) are conflict-free.
// ------------------------------------------------------------------------------------------------------------
template <int BMW, int BNW, int NTW, int MTW = 1>
struct GemmSmem {
    static constexpr int kRows = 16 * BMW * MTW;
    static constexpr int kTiles = NTW * BNW;
    static constexpr int kX = kRows * 512;      // bytes per stage
    static constexpr int kW = kTiles * 512;     // bytes per stage
    static constexpr int kS = kTiles * 32;      // bytes per stage
    static constexpr int kStage = kX + kW + kS;
    static constexpr int kBytes = 2 * kStage;
};

__device__ __forceinline__ int xunit(int u, int r) { return u ^ ((2 * (u >> 3) + (r & 1)) & 7); }

template <int C, int CPW, int ACC, int BMW, int BNW, int NTW, int MTW, int EPI>
__global__ void __launch_bounds__(BMW * BNW * 32)
tern_gemm_kernel(const bf16* X, int ldx, int M, const uint4* __restrict__ Wt, const uint16_t* __restrict__ St,
                 bf16* Y, int ldy, const bf16* R, int ldr, int early) {
    static_assert(16 % ACC == 0, "ACC must divide 16");
    using L = GemmSmem<BMW, BNW, NTW, MTW>;
    constexpr int NTHR = BMW * BNW * 32;
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x;
    const int wid = tid >> 5, lane = tid & 31;
    const int wm = wid / BNW, wnn = wid % BNW;
    const int g = lane >> 2, t = lane & 3;
    const int m0 = blockIdx.y * L::kRows;
    const int tileb = blockIdx.x * L::kTiles;  // first output tile of the block

    auto load_w = [&](int ch, int stage) {
        unsigned char* sw = smem + stage * L::kStage + L::kX;
        for (int i = tid; i < L::kTiles * 32; i += NTHR) {  // 16-B pieces of codes
            const int tl = i >> 5, ln = i & 31;
            cp_async16(sw + i * 16, Wt + ((size_t)(tileb + tl) * C + ch) * 32 + ln, 16);
        }
        unsigned char* ss = sw + L::kW;
        for (int i = tid; i < L::kTiles * 2; i += NTHR) {  // 16-B pieces of scales (32 B per tile)
            const int tl = i >> 1, h = i & 1;
            cp_async16(ss + i * 16, St + ((size_t)(tileb + tl) * C + ch) * 16 + h * 8, 16);
        }
    };
    auto load_x = [&](int ch, int stage) {
        unsigned char* sx = smem + stage * L::kStage;
        for (int i = tid; i < L::kRows * 32; i += NTHR) {
            const int r = i >> 5, u = i & 31;
            const int row = m0 + r;
            const bool ok = row < M;
            cp_async16(sx + r * 512 + xunit(u, r) * 16, X + (size_t)(ok ? row : 0) * ldx + ch * kChunk + u * 8,
                       ok ? 16 : 0);
        }
    };

    load_w(0, 0);
    if (early > 1) pdl_launch_dependents();  // (the GEMM kernel triggers late by default: its dependents' blocks
    pdl_wait();                               //  would hold shared memory its own later waves need)
    load_x(0, 0);
    cp_async_commit();

    float acc[NTW][MTW][ACC][4];
    float tot[NTW][MTW][4];
#pragma unroll 1
    for (int ch = 0; ch < C; ++ch) {
        const int stage = ch & 1;
        if (ch + 1 < C) {
            load_w(ch + 1, stage ^ 1);
            load_x(ch + 1, stage ^ 1);
        }
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const unsigned char* sx = smem + stage * L::kStage;
        const unsigned char* sw = sx + L::kX;
        const unsigned char* ss = sw + L::kW;
        if (ch % CPW == 0) {
#pragma unroll
            for (int nt = 0; nt < NTW; ++nt)
#pragma unroll
                for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
                    for (int a = 0; a < ACC; ++a)
                        acc[nt][mt][a][0] = acc[nt][mt][a][1] = acc[nt][mt][a][2] = acc[nt][mt][a][3] = 0.f;
        }
        uint4 w[NTW];
        uint32_t ta[NTW], tb[NTW];
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) {
            const int tl = wnn * NTW + nt;
            w[nt] = lds128(sw + (tl * 32 + lane) * 16);
            const uint32_t s = *reinterpret_cast<const uint16_t*>(ss + tl * 32 + (g * 2 + (t >> 1)) * 2);
            scale_tables(s, ta[nt], tb[nt]);
        }
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const int u = 8 * t + q;
            uint4 xa[MTW], xb[MTW];
#pragma unroll
            for (int mt = 0; mt < MTW; ++mt) {
                const int r0 = (wm * MTW + mt) * 16 + g, r1 = r0 + 8;
                xa[mt] = lds128(sx + r0 * 512 + xunit(u, r0) * 16);
                xb[mt] = lds128(sx + r1 * 512 + xunit(u, r1) * 16);
            }
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const int j = 2 * q + jj;
#pragma unroll
                for (int nt = 0; nt < NTW; ++nt) {
                    uint32_t b0, b1;
                    decode_step(word_of(w[nt], j >> 2), j & 3, ta[nt], tb[nt], b0, b1);
#pragma unroll
                    for (int mt = 0; mt < MTW; ++mt) {
                        const uint32_t a0 = jj ? xa[mt].z : xa[mt].x;
                        const uint32_t a2 = jj ? xa[mt].w : xa[mt].y;
                        const uint32_t a1 = jj ? xb[mt].z : xb[mt].x;
                        const uint32_t a3 = jj ? xb[mt].w : xb[mt].y;
                        mma_16816(acc[nt][mt][j % ACC], a0, a1, a2, a3, b0, b1);
                    }
                }
            }
        }
        if (ch % CPW == CPW - 1) {
#pragma unroll
            for (int nt = 0; nt < NTW; ++nt)
#pragma unroll
                for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
                    for (int qq = 0; qq < 4; ++qq) {
                        float v = acc[nt][mt][0][qq];
#pragma unroll
                        for (int a = 1; a < ACC; ++a) v += acc[nt][mt][a][qq];
                        if (ch == CPW - 1)
                            tot[nt][mt][qq] = v;
                        else
                            tot[nt][mt][qq] += v;
                    }
        }
        __syncthreads();  // the stage is overwritten by the next iteration's loads
    }
    if (early <= 1) pdl_launch_dependents();
#pragma unroll
    for (int nt = 0; nt < NTW; ++nt)
#pragma unroll
        for (int mt = 0; mt < MTW; ++mt)
            store_tile<EPI>(tot[nt][mt], tileb + wnn * NTW + nt, m0 + (wm * MTW + mt) * 16, M, g, t, Y, ldy, R, ldr);
}

// ------------------------------------------------------------------------------------------------------------
// lm-proj-wide (11 Oct; DEFAULT OFF): a cluster split-K kernel for 17..32 decode rows.  Only the decode step calls
// it, and only with env KT2N_LM_PROJ_WIDE=1 (src/cuda/kernels/step_o1.cu); every other caller and every default
// run keeps the kernels above.
//
// Tuning rationale: at 17..32 rows the register-tiled kernel is latency-bound.  Each
// warp streams its activation vectors from global memory a few at a time (half of all warp stalls wait on them),
// every CTA re-reads the whole activation matrix for its K range, and 128-162 registers per thread leave only
// 16-32% of the warp slots occupied.  Staging the activations in shared memory would need the CTA's whole K range
// (32 rows x 2048 x 2 B = 128 KB, more than an SM has), so the family's K split is spread over a thread-block
// CLUSTER instead: CTA `rank` of an H-CTA cluster runs partials [rank*PPC, (rank+1)*PPC) (PPC = WK / H) of every
// output tile of the cluster's column group, copies only that K slice of all rows into shared memory once
// (cp.async: no registers, every copy in flight at once), and its warps read their MMA fragments from there.  The
// partials meet in shared memory and are summed across the cluster through distributed shared memory (DSMEM).
//
// Arithmetic: operation for operation the decode kernel's (the k1 contract in tern_proj.cuh):
//   * warp (rank, pl) is decode-kernel warp w = rank * PPC + pl and owns chunks [w*CPW, (w+1)*CPW);
//   * per chunk i and MMA step j (0..15): A = rows g / g+8 of a 16-row tile at k = 256c + 64t + 4j + {0,1}
//     (a0 / a1) and + {2,3} (a2 / a3); B = decode_step(word j>>2 of the lane's code vector, step j&3): the same
//     values in the same fragment slots as tern_tile_kernel (activations read from shared memory instead of
//     global memory: the same bf16 bits);
//   * step s = 16i + j accumulates into chain j % ACC, in the same order (i ascending, then j ascending);
//   * the warp's partial is ((acc0 + acc1) + acc2) + ..., the output ((p0 + p1) + p2) + ... in warp order, and
//     store_tile is the same epilogue.
// So each output element is the same fp32 operations in the same order: bit-identical to tern_dec_kernel /
// tern_tile_kernel (tested by memcmp on random and real weights).  Rows: tile mt holds
// rows 16mt .. 16mt+15 (row 16mt + 8h + g at fragment row 8h + g), as tern_tile_kernel's single M block does;
// rows >= M read as zero and are never written.
// ------------------------------------------------------------------------------------------------------------
namespace wide {

// %cluster_ctarank (sm_90+; every architecture this library is built for has clusters).
__device__ __forceinline__ int cluster_rank() {
    unsigned r = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
#endif
    return static_cast<int>(r);
}

// Cluster barrier with release / acquire: all threads of all CTAs of the cluster arrive; shared-memory writes made
// before it are visible to the other CTAs' DSMEM reads after it.
__device__ __forceinline__ void cluster_sync() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
#endif
}

// 16-byte load of the same shared-memory offset in CTA `rank` of this cluster (DSMEM).
__device__ __forceinline__ float4 ld_cluster_f4(const void* local, int rank) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(local));
    unsigned ra;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(a), "r"(rank));
    asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "r"(ra)
                 : "memory");
#endif
    return v;
}

// cp.async.wait_group N where N is known only after unrolling (N <= 3).
__device__ __forceinline__ void cp_async_wait_n(int n) {
    if (n <= 0)
        cp_async_wait<0>();
    else if (n == 1)
        cp_async_wait<1>();
    else if (n == 2)
        cp_async_wait<2>();
    else
        cp_async_wait<3>();
}

}  // namespace wide

template <int WK, int CPW, int ACC, int H, int NT, int WN, int MT, int EPI, int MINB, bool ONEBUF>
__global__ void __launch_bounds__((WK / H) * WN * 32, MINB)
tern_wide_kernel(const bf16* X, int ldx, int M, const uint4* __restrict__ Wt, const uint16_t* __restrict__ St,
                 bf16* Y, int ldy, const bf16* R, int ldr, int C, int early) {
    static_assert(16 % ACC == 0, "ACC must divide 16");
    static_assert(WK % H == 0, "the cluster must split the family's partials evenly");
    static_assert(CPW >= 1 && CPW <= 4, "one cp.async group per chunk of a partial");
    constexpr int PPC = WK / H;     // K partials (decode-kernel warps) per CTA
    constexpr int NTHR = PPC * WN * 32;
    constexpr int TI = NT * MT;     // 16 x 8 output tiles per warp
    constexpr int ITEMS = WN * TI;  // ... per column group (the same for the H CTAs of a cluster)
    extern __shared__ __align__(16) unsigned char wsm[];
    const int wid = threadIdx.x >> 5;
    const int pl = wid % PPC;
    const int wn = wid / PPC;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int rank = wide::cluster_rank();
    const int chunk0 = kt2n_tern_wide::g_chunk(rank, PPC, pl, 0, CPW);
    const int tile0 = (blockIdx.y * WN + wn) * NT;

    // Weights: as in the decode / tiled kernels (registers, before the PDL wait).
    uint4 w[NT][CPW];
    uint32_t ta[NT][CPW], tb[NT][CPW];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int i = 0; i < CPW; ++i) {
            const size_t tc = (size_t)(tile0 + nt) * C + chunk0 + i;
            w[nt][i] = ld_weights(Wt + tc * 32 + lane);
            const uint32_t s = __ldg(St + tc * 16 + g * 2 + (t >> 1));
            scale_tables(s, ta[nt][i], tb[nt][i]);
        }
    }
    if (early) pdl_launch_dependents();
    pdl_wait();

    // Stage this CTA's K slice (its partials' chunks) of rows 0..M-1 in shared memory, [buffer][row][512 B] with a
    // swizzled 16-byte unit (tern_wide_index.cuh act_off: conflict-free fragment reads, as in the GEMM kernel).
    // Commit group i = chunk i of every partial of the CTA, so the MMAs of chunk 0 start while chunk 1 is still in
    // flight.  ONEBUF (half the shared memory, for more CTAs per SM): one buffer per partial, refilled with chunk i once
    // every warp is done with chunk i - 1.
    const int per_group = PPC * M * 32;
    auto issue = [&](int i) {
        for (int idx = threadIdx.x; idx < per_group; idx += NTHR) {
            const int u = idx & 31, rest = idx >> 5;
            const int r = rest % M, p = rest / M;
            cp_async16(wsm + kt2n_tern_wide::act_off(kt2n_tern_wide::act_buf(p, i, CPW, ONEBUF), M, r, u),
                       X + (size_t)r * ldx + (size_t)kt2n_tern_wide::g_chunk(rank, PPC, p, i, CPW) * kChunk + u * 8, 16);
        }
        cp_async_commit();
    };
#pragma unroll
    for (int i = 0; i < (ONEBUF ? 1 : CPW); ++i) issue(i);

    int rowi[MT][2];
    bool rv[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            rowi[mt][h] = mt * 16 + h * 8 + g;
            rv[mt][h] = rowi[mt][h] < M;
        }
    float c[NT][MT][ACC][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int a = 0; a < ACC; ++a) c[nt][mt][a][0] = c[nt][mt][a][1] = c[nt][mt][a][2] = c[nt][mt][a][3] = 0.f;
#pragma unroll
    for (int i = 0; i < CPW; ++i) {
        if (ONEBUF && i > 0) {
            __syncthreads();  // every warp is done with chunk i - 1: refill the buffers
            issue(i);
        }
        wide::cp_async_wait_n(ONEBUF ? 0 : CPW - 1 - i);
        __syncthreads();  // every thread's copies of group i have landed
        const int buf = kt2n_tern_wide::act_buf(pl, i, CPW, ONEBUF);
#pragma unroll
        for (int jp = 0; jp < 8; ++jp) {
            const int u = 8 * t + jp;
            uint4 xa[MT], xb[MT];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const int r0 = rowi[mt][0], r1 = rowi[mt][1];
                xa[mt] = rv[mt][0] ? lds128(wsm + kt2n_tern_wide::act_off(buf, M, r0, u)) : make_uint4(0, 0, 0, 0);
                xb[mt] = rv[mt][1] ? lds128(wsm + kt2n_tern_wide::act_off(buf, M, r1, u)) : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const int j = 2 * jp + jj;
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    uint32_t b0, b1;
                    decode_step(word_of(w[nt][i], j >> 2), j & 3, ta[nt][i], tb[nt][i], b0, b1);
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        const uint32_t a0 = jj ? xa[mt].z : xa[mt].x;
                        const uint32_t a2 = jj ? xa[mt].w : xa[mt].y;
                        const uint32_t a1 = jj ? xb[mt].z : xb[mt].x;
                        const uint32_t a3 = jj ? xb[mt].w : xb[mt].y;
                        mma_16816(c[nt][mt][j % ACC], a0, a1, a2, a3, b0, b1);
                    }
                }
            }
        }
    }
    float p[NT][MT][4];  // the warp's partial: ((acc0 + acc1) + acc2) + ...
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                float v = c[nt][mt][0][q];
#pragma unroll
                for (int a = 1; a < ACC; ++a) v += c[nt][mt][a][q];
                p[nt][mt][q] = v;
            }
    // The partials reuse the activation slice's shared memory: wait until every warp is done reading it.
    __syncthreads();
    float4* red = reinterpret_cast<float4*>(wsm);  // [WN][PPC][TI][32 lanes]
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            red[kt2n_tern_wide::red_idx(PPC, TI, wn, pl, nt * MT + mt, lane)] =
                make_float4(p[nt][mt][0], p[nt][mt][1], p[nt][mt][2], p[nt][mt][3]);
    wide::cluster_sync();
    // One warp of the cluster sums each output tile of the column group in warp order w = 0 .. WK-1 (partial w sits in
    // CTA w / PPC, slot w % PPC): ((p0 + p1) + p2) + ..., exactly the decode kernel's warp-0 sum.
    for (int it = rank * (PPC * WN) + wid; it < ITEMS; it += H * PPC * WN) {
        const int iw = it / TI, ik = it % TI;
        float4 q[WK];
#pragma unroll
        for (int ww = 0; ww < WK; ++ww)
            q[ww] = wide::ld_cluster_f4(red + kt2n_tern_wide::red_idx(PPC, TI, iw, ww % PPC, ik, lane), ww / PPC);
        float v[4] = {q[0].x, q[0].y, q[0].z, q[0].w};
#pragma unroll
        for (int ww = 1; ww < WK; ++ww) {
            v[0] += q[ww].x;
            v[1] += q[ww].y;
            v[2] += q[ww].z;
            v[3] += q[ww].w;
        }
        store_tile<EPI>(v, (blockIdx.y * WN + iw) * NT + ik / MT, 16 * (ik % MT), M, g, t, Y, ldy, R, ldr);
    }
    wide::cluster_sync();  // a CTA's shared memory must outlive the other CTAs' DSMEM reads of it
    if (!early) pdl_launch_dependents();
}

// ------------------------------------------------------------------------------------------------------------
// Layout kernels
// ------------------------------------------------------------------------------------------------------------
__global__ void repack_codes_kernel(const uint32_t* __restrict__ src, int rows, int kw, uint32_t* __restrict__ dst,
                                    int C, int row_offset, int row_step) {
    const size_t total = (size_t)rows * kw;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t)gridDim.x * blockDim.x) {
        const int i = (int)(idx / kw), wi = (int)(idx % kw);
        const int n = row_offset + i * row_step;
        const int tile = n >> 3, g = n & 7;
        const int chunk = wi >> 4, t = (wi >> 2) & 3, ii = wi & 3;
        dst[(((size_t)tile * C + chunk) * 32 + g * 4 + t) * 4 + ii] = nat_to_sel(src[idx]);
    }
}

__global__ void repack_scales_kernel(const uint16_t* __restrict__ src, int rows, int groups, uint16_t* __restrict__ dst,
                                     int C, int row_offset, int row_step) {
    const size_t total = (size_t)rows * groups;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t)gridDim.x * blockDim.x) {
        const int i = (int)(idx / groups), gr = (int)(idx % groups);
        const int n = row_offset + i * row_step;
        const int tile = n >> 3, g = n & 7;
        dst[(((size_t)tile * C + (gr >> 1)) * 8 + g) * 2 + (gr & 1)] = src[idx];
    }
}

__global__ void unrepack_codes_kernel(const uint32_t* __restrict__ codes, int rows, int kw, uint32_t* __restrict__ dst,
                                      int C, int row_offset, int row_step) {
    const size_t total = (size_t)rows * kw;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t)gridDim.x * blockDim.x) {
        const int i = (int)(idx / kw), wi = (int)(idx % kw);
        const int n = row_offset + i * row_step;
        const int tile = n >> 3, g = n & 7;
        const int chunk = wi >> 4, t = (wi >> 2) & 3, ii = wi & 3;
        dst[idx] = sel_to_nat(codes[(((size_t)tile * C + chunk) * 32 + g * 4 + t) * 4 + ii]);
    }
}

__global__ void unrepack_scales_kernel(const uint16_t* __restrict__ scales, int rows, int groups,
                                       uint16_t* __restrict__ dst, int C, int row_offset, int row_step) {
    const size_t total = (size_t)rows * groups;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t)gridDim.x * blockDim.x) {
        const int i = (int)(idx / groups), gr = (int)(idx % groups);
        const int n = row_offset + i * row_step;
        const int tile = n >> 3, g = n & 7;
        dst[idx] = scales[(((size_t)tile * C + (gr >> 1)) * 8 + g) * 2 + (gr & 1)];
    }
}

// One thread per (tile, chunk, lane): the exact decode path of the projection kernels.
__global__ void decode_dense_kernel(const uint4* __restrict__ Wt, const uint16_t* __restrict__ St, int n, int k,
                                    uint16_t* __restrict__ out) {
    const int C = k / kChunk;
    const size_t total = (size_t)(n / 8) * C * 32;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t)gridDim.x * blockDim.x) {
        const int lane = (int)(idx % 32);
        const size_t tc = idx / 32;
        const int chunk = (int)(tc % C);
        const int tile = (int)(tc / C);
        const int g = lane >> 2, t = lane & 3;
        const uint4 w = Wt[tc * 32 + lane];
        uint32_t ta, tb;
        scale_tables(St[tc * 16 + g * 2 + (t >> 1)], ta, tb);
        uint16_t* row = out + (size_t)(tile * 8 + g) * k + chunk * kChunk + t * 64;
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                uint32_t b0, b1;
                decode_step(word_of(w, i), q, ta, tb, b0, b1);
                const int o = 16 * i + 4 * q;
                row[o + 0] = (uint16_t)(b0 & 0xFFFFu);
                row[o + 1] = (uint16_t)(b0 >> 16);
                row[o + 2] = (uint16_t)(b1 & 0xFFFFu);
                row[o + 3] = (uint16_t)(b1 >> 16);
            }
    }
}

__global__ void silu_mul_test_kernel(const bf16* __restrict__ gate, const bf16* __restrict__ up, bf16* __restrict__ out,
                                     int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = silu_mul_bf16(__bfloat162float(gate[i]), __bfloat162float(up[i]));
}

// ------------------------------------------------------------------------------------------------------------
// Launch plumbing
// ------------------------------------------------------------------------------------------------------------
template <typename Kern, typename... P>
cudaError_t launch_ex_smem(Kern kern, dim3 grid, dim3 block, size_t smem, bool pdl, cudaStream_t stream,
                           P... params) {
    pdl = kt2n::cuda::pdl_enabled(pdl);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    return cudaLaunchKernelEx(&cfg, kern, params...);
}

template <typename Kern, typename... P>
cudaError_t launch_ex(Kern kern, dim3 grid, dim3 block, bool pdl, cudaStream_t stream, P... params) {
    pdl = kt2n::cuda::pdl_enabled(pdl);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    return cudaLaunchKernelEx(&cfg, kern, params...);
}

template <int WK, int CPW, int ACC, int NT, int WN, int EPI, bool NORM = false>
cudaError_t dec_go(const Args& a) {
    // block <= 1024 threads; reduction buffer within the 48 KB static shared-memory limit
    if constexpr (WK * WN * 32 > 1024 || (WK > 1 && WN * WK * NT * 4 * 32 * 4 > 48 * 1024)) {
        return cudaErrorInvalidConfiguration;
    } else {
        const dim3 grid(a.n / (8 * NT * WN)), block(WK * WN * 32);
        if (a.m > 8)
            return launch_ex(tern_dec_kernel<WK, CPW, ACC, NT, WN, true, EPI, NORM>, grid, block, a.pdl, a.stream,
                             a.x, a.ldx, a.m, a.w, a.s, a.y, a.ldy, a.r, a.ldr, a.c, a.early, a.lnw, a.eps);
        return launch_ex(tern_dec_kernel<WK, CPW, ACC, NT, WN, false, EPI, NORM>, grid, block, a.pdl, a.stream, a.x,
                         a.ldx, a.m, a.w, a.s, a.y, a.ldy, a.r, a.ldr, a.c, a.early, a.lnw, a.eps);
    }
}

template <int WK, int CPW, int ACC, int EPI, bool NORM = false>
cudaError_t dec_shape(int nt, int wn, const Args& a) {
#define KT2N_DEC(NT_, WN_) \
    if (nt == NT_ && wn == WN_) return dec_go<WK, CPW, ACC, NT_, WN_, EPI, NORM>(a);
    KT2N_DEC(1, 1) KT2N_DEC(2, 1) KT2N_DEC(4, 1) KT2N_DEC(1, 2) KT2N_DEC(2, 2) KT2N_DEC(1, 4)
#if KT2N_TERN_TUNE != 1
    KT2N_DEC(4, 2)
#endif
#undef KT2N_DEC
    return cudaErrorInvalidConfiguration;
}

template <int WK, int CPW, int ACC, int NT, int WN, int MT, int EPI, int MINB = 1, int PS = 0>
cudaError_t tile_go(const Args& a) {
    // (MINB blocks of WK * WN warps must fit the SM's 1,536 threads, or ptxas ignores the bound with a warning.)
    // Static shared memory: the reduction buffer (+ the PS-stage activation ring) within 48 KB.
    constexpr int kRed = WK > 1 ? WN * WK * NT * MT * 4 * 32 * 4 : 0;
    constexpr int kRing = PS > 0 ? WN * WK * PS * 2 * MT * 32 * 16 : 0;
    if constexpr (WK * WN * 32 > 1024 || kRed + kRing > 48 * 1024 || WK * WN * 32 * MINB > 1536) {
        return cudaErrorInvalidConfiguration;
    } else {
        const int mblocks = (a.m + MT * 16 - 1) / (MT * 16);
        if (mblocks > 65535) return cudaErrorInvalidConfiguration;
        const dim3 grid(a.n / (8 * NT * WN), mblocks), block(WK * WN * 32);
        if constexpr (PS > 0)
            return launch_ex_smem(tern_tile_kernel<WK, CPW, ACC, NT, WN, MT, EPI, MINB, PS>, grid, block, (size_t)kRing,
                                  a.pdl, a.stream, a.x, a.ldx, a.m, a.w, a.s, a.y, a.ldy, a.r, a.ldr, a.c, a.early);
        return launch_ex(tern_tile_kernel<WK, CPW, ACC, NT, WN, MT, EPI, MINB, PS>, grid, block, a.pdl, a.stream, a.x,
                         a.ldx, a.m, a.w, a.s, a.y, a.ldy, a.r, a.ldr, a.c, a.early);
    }
}

#if KT2N_TERN_EXPERIMENTS
template <int WK, int CPW, int ACC, int NP, int WN, int R8, int EPI, int MINB>
cudaError_t swap_go(const Args& a) {
    if constexpr (WK * WN * 32 > 1024 || (WK > 1 && WN * WK * NP * R8 * 4 * 32 * 4 > 48 * 1024) ||
                  WK * WN * 32 * MINB > 1536) {
        return cudaErrorInvalidConfiguration;
    } else {
        if (a.m > 8 * R8 || a.n % (16 * NP * WN)) return cudaErrorInvalidConfiguration;
        const dim3 grid(a.n / (16 * NP * WN)), block(WK * WN * 32);
        return launch_ex(tern_swap_kernel<WK, CPW, ACC, NP, WN, R8, EPI, MINB>, grid, block, a.pdl, a.stream, a.x,
                         a.ldx, a.m, a.w, a.s, a.y, a.ldy, a.c, a.early);
    }
}

// mt = 1000 + 10 * MINB + acc2 (acc2 1 = 2 accumulator chains), nt = NP (16-column pairs per warp), wn = WN; R8 from M.
// Instantiated for the epilogue each family uses in the decode step (gate_up SILU_MUL, the others STORE) only.
template <int C, int CPW, int ACC, int EPI>
cudaError_t swap_shape(int nt, int wn, int mt, const Args& a) {
    constexpr int WK = C / CPW;
    constexpr bool gu = (C == 8 && WK == 4);
    if constexpr ((EPI == KT2N_TERN_EPI_SILU_MUL) != gu || EPI == KT2N_TERN_EPI_RESIDUAL) {
        return cudaErrorInvalidConfiguration;
    } else {
        constexpr int A2 = ACC < 2 ? ACC : 2;
        const int r8 = (a.m + 7) / 8;
        const int code = mt - 1000;
#define KT2N_SWAP(NP_, MINB_, ACC2_)                                                                   \
    if (nt == NP_ && wn == 1 && code == 10 * MINB_ + ACC2_) {                                          \
        if (r8 <= 3) return swap_go<WK, CPW, (ACC2_ ? A2 : ACC), NP_, 1, 3, EPI, MINB_>(a);            \
        if (r8 <= 4) return swap_go<WK, CPW, (ACC2_ ? A2 : ACC), NP_, 1, 4, EPI, MINB_>(a);            \
        return cudaErrorInvalidConfiguration;                                                          \
    }
        KT2N_SWAP(1, 1, 0) KT2N_SWAP(2, 1, 1) KT2N_SWAP(1, 1, 1)
        if constexpr (gu) { KT2N_SWAP(1, 5, 0) }
#undef KT2N_SWAP
        return cudaErrorInvalidConfiguration;
    }
}
#endif  // KT2N_TERN_EXPERIMENTS

template <int C, int CPW, int ACC, int EPI>
cudaError_t pre_shape(int nt, int wn, int mt, const Args& a) {
    constexpr int WK = C / CPW;
#if KT2N_TERN_EXPERIMENTS
    if (mt >= 1000) return swap_shape<C, CPW, ACC, EPI>(nt, wn, mt, a);  // L3 sprint 4: swapped operands
#endif
#define KT2N_TILE(NT_, WN_, MT_) \
    if (nt == NT_ && wn == WN_ && mt == MT_) return tile_go<WK, CPW, ACC, NT_, WN_, MT_, EPI>(a);
    KT2N_TILE(2, 1, 2) KT2N_TILE(2, 2, 1) KT2N_TILE(1, 2, 2) KT2N_TILE(4, 1, 1) KT2N_TILE(4, 2, 1)
#if KT2N_TERN_EXPERIMENTS
    // L3 (10 Oct): decode buckets 17..64 (one M block of 32 rows, sized so the grid fits one wave on 188 SMs)
    KT2N_TILE(1, 1, 1) KT2N_TILE(2, 1, 1) KT2N_TILE(1, 1, 2) KT2N_TILE(2, 2, 2) KT2N_TILE(3, 1, 2)
    // ... and with a register cap (__launch_bounds__ min blocks per SM, MINB) so a 32-row grid fits one wave:
    // mt = MT + 10 * MINB.  The cap changes register allocation only, never the arithmetic.
#define KT2N_TILE_B(NT_, WN_, MT_, MINB_) \
    if (nt == NT_ && wn == WN_ && mt == MT_ + 10 * MINB_) return tile_go<WK, CPW, ACC, NT_, WN_, MT_, EPI, MINB_>(a);
    KT2N_TILE_B(2, 1, 2, 2) KT2N_TILE_B(2, 1, 2, 4) KT2N_TILE_B(2, 1, 2, 5) KT2N_TILE_B(3, 1, 2, 3)
#undef KT2N_TILE_B
    // L3 EXPERIMENT (sprint 3): one accumulator chain per output (ACC = 1) instead of the family's ACC, mt = MT + 100.
    // NOT the k1 arithmetic contract: the K partial sums are added in another order, so rows of these buckets are no
    // longer bit-identical to the same rows at <= 16 rows (statistically equal).  Fewer accumulator registers.
#define KT2N_TILE_A1(NT_, WN_, MT_) \
    if (nt == NT_ && wn == WN_ && mt == MT_ + 100) return tile_go<WK, CPW, 1, NT_, WN_, MT_, EPI>(a);
    KT2N_TILE_A1(4, 1, 1) KT2N_TILE_A1(4, 2, 1) KT2N_TILE_A1(2, 1, 2) KT2N_TILE_A1(4, 1, 2) KT2N_TILE_A1(1, 1, 2)
    KT2N_TILE_A1(2, 2, 2)
#undef KT2N_TILE_A1
    // L3 sprint 7 EXPERIMENT: the PS-stage cp.async activation ring (mt = 160 + PS, MT = 1).
#define KT2N_TILE_P(NT_, WN_, PS_) \
    if (nt == NT_ && wn == WN_ && mt == 160 + PS_) return tile_go<WK, CPW, ACC, NT_, WN_, 1, EPI, 1, PS_>(a);
    KT2N_TILE_P(4, 1, 2) KT2N_TILE_P(4, 1, 3) KT2N_TILE_P(4, 1, 4) KT2N_TILE_P(2, 1, 4) KT2N_TILE_P(4, 2, 2)
#undef KT2N_TILE_P
#endif  // KT2N_TERN_EXPERIMENTS
#if KT2N_TERN_TUNE == 2
    KT2N_TILE(1, 1, 1) KT2N_TILE(2, 1, 1) KT2N_TILE(1, 2, 1) KT2N_TILE(1, 1, 2) KT2N_TILE(1, 1, 4)
    KT2N_TILE(2, 1, 4) KT2N_TILE(2, 2, 2) KT2N_TILE(4, 1, 2) KT2N_TILE(1, 2, 4) KT2N_TILE(1, 4, 1) KT2N_TILE(2, 4, 1)
#endif
#undef KT2N_TILE
    return cudaErrorInvalidConfiguration;
}

template <int C, int CPW, int ACC, int BMW, int BNW, int NTW, int MTW, int EPI>
cudaError_t gemm_go(const Args& a) {
    using L = GemmSmem<BMW, BNW, NTW, MTW>;
    auto kern = tern_gemm_kernel<C, CPW, ACC, BMW, BNW, NTW, MTW, EPI>;
    // Dynamic shared memory above 48 KB needs an opt-in per function (and device).  Idempotent, so a benign race;
    // legal during stream capture (tested).
    static int attr_set[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = 0;
    if (!attr_set[dev]) {
        const cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, L::kBytes);
        if (e != cudaSuccess) return e;
        attr_set[dev] = 1;
    }
    const int mblocks = (a.m + L::kRows - 1) / L::kRows;
    if (mblocks > 65535) return cudaErrorInvalidConfiguration;
    const dim3 grid(a.n / (8 * NTW * BNW), mblocks), block(BMW * BNW * 32);
    return launch_ex_smem(kern, grid, block, (size_t)L::kBytes, a.pdl, a.stream, a.x, a.ldx, a.m, a.w, a.s, a.y,
                          a.ldy, a.r, a.ldr, a.early);
}

// gcode = -mt = BMW + 100 * (MTW - 1)
template <int C, int CPW, int ACC, int EPI>
cudaError_t gemm_shape(int ntw, int bnw, int gcode, const Args& a) {
    const int bmw = gcode % 100, mtw = gcode / 100 + 1;
#define KT2N_GEMM(BMW_, BNW_, NTW_, MTW_) \
    if (bmw == BMW_ && bnw == BNW_ && ntw == NTW_ && mtw == MTW_) return gemm_go<C, CPW, ACC, BMW_, BNW_, NTW_, MTW_, EPI>(a);
    KT2N_GEMM(1, 4, 4, 2) KT2N_GEMM(1, 4, 2, 2) KT2N_GEMM(1, 8, 2, 2) KT2N_GEMM(1, 8, 2, 4)
    KT2N_GEMM(1, 4, 4, 1) KT2N_GEMM(4, 2, 2, 1)
#if KT2N_TERN_TUNE == 2
    KT2N_GEMM(2, 4, 4, 1) KT2N_GEMM(4, 4, 2, 1) KT2N_GEMM(4, 2, 4, 1) KT2N_GEMM(2, 8, 2, 1) KT2N_GEMM(1, 8, 4, 1)
    KT2N_GEMM(2, 2, 4, 1) KT2N_GEMM(2, 4, 4, 2) KT2N_GEMM(2, 4, 2, 2) KT2N_GEMM(2, 2, 4, 2) KT2N_GEMM(2, 2, 2, 2)
    KT2N_GEMM(1, 4, 4, 4) KT2N_GEMM(1, 4, 2, 4)
#endif
#undef KT2N_GEMM
    return cudaErrorInvalidConfiguration;
}

#if KT2N_TERN_PRODUCTION_ORDER
template <int BW>
cudaError_t production_b1_go(int epi, const Args& a) {
    const dim3 grid(epi == KT2N_TERN_EPI_SILU_MUL ? a.n / 2 : a.n), block(BW * 16);
    const uint32_t* codes = reinterpret_cast<const uint32_t*>(a.w);
    switch (epi) {
        case KT2N_TERN_EPI_STORE:
            return launch_ex(production_b1_kernel<BW, KT2N_TERN_EPI_STORE>, grid, block, a.pdl, a.stream,
                             a.x, codes, a.s, a.y, a.r, a.c * 256);
        case KT2N_TERN_EPI_SILU_MUL:
            return launch_ex(production_b1_kernel<BW, KT2N_TERN_EPI_SILU_MUL>, grid, block, a.pdl, a.stream,
                             a.x, codes, a.s, a.y, a.r, a.c * 256);
        case KT2N_TERN_EPI_RESIDUAL:
            return launch_ex(production_b1_kernel<BW, KT2N_TERN_EPI_RESIDUAL>, grid, block, a.pdl, a.stream,
                             a.x, codes, a.s, a.y, a.r, a.c * 256);
        default: return cudaErrorInvalidConfiguration;
    }
}
#endif

#if KT2N_TERN_PRODUCTION_ORDER
template <int SPLITS>
cudaError_t production_tc_go(int epi, const Args& a) {
    const dim3 grid(a.n / 8, (a.m + 15) / 16), block(SPLITS * 32);
    const uint32_t* codes = reinterpret_cast<const uint32_t*>(a.w);
    switch (epi) {
        case KT2N_TERN_EPI_STORE:
            return launch_ex(production_tc_kernel<SPLITS, KT2N_TERN_EPI_STORE>, grid, block, a.pdl, a.stream,
                             a.x, codes, a.s, a.y, a.m, a.n, a.c * 256, a.ldx, a.ldy, a.r, a.ldr);
        case KT2N_TERN_EPI_SILU_MUL:
            return launch_ex(production_tc_kernel<SPLITS, KT2N_TERN_EPI_SILU_MUL>, grid, block, a.pdl, a.stream,
                             a.x, codes, a.s, a.y, a.m, a.n, a.c * 256, a.ldx, a.ldy, a.r, a.ldr);
        case KT2N_TERN_EPI_RESIDUAL:
            return launch_ex(production_tc_kernel<SPLITS, KT2N_TERN_EPI_RESIDUAL>, grid, block, a.pdl, a.stream,
                             a.x, codes, a.s, a.y, a.m, a.n, a.c * 256, a.ldx, a.ldy, a.r, a.ldr);
        default: return cudaErrorInvalidConfiguration;
    }
}
#endif

template <int C, int WK, int ACC>
cudaError_t run_cfg(int epi, bool prefill, int nt, int wn, int mt, const Args& a) {
    constexpr int CPW = C / WK;
    static_assert(CPW * WK == C, "split must tile the chunks");
    if (prefill && mt < 0) {  // shared-memory GEMM kernel: nt = NTW, wn = BNW, -mt = BMW
        switch (epi) {
            case KT2N_TERN_EPI_STORE: return gemm_shape<C, CPW, ACC, KT2N_TERN_EPI_STORE>(nt, wn, -mt, a);
            case KT2N_TERN_EPI_SILU_MUL: return gemm_shape<C, CPW, ACC, KT2N_TERN_EPI_SILU_MUL>(nt, wn, -mt, a);
#if !KT2N_TERN_TUNE
            case KT2N_TERN_EPI_RESIDUAL: return gemm_shape<C, CPW, ACC, KT2N_TERN_EPI_RESIDUAL>(nt, wn, -mt, a);
#endif
            default: return cudaErrorInvalidConfiguration;
        }
    }
    if (prefill) {
        switch (epi) {
            case KT2N_TERN_EPI_STORE: return pre_shape<C, CPW, ACC, KT2N_TERN_EPI_STORE>(nt, wn, mt, a);
            case KT2N_TERN_EPI_SILU_MUL: return pre_shape<C, CPW, ACC, KT2N_TERN_EPI_SILU_MUL>(nt, wn, mt, a);
#if !KT2N_TERN_TUNE  // tune builds time STORE / SILU_MUL only (compile time)
            case KT2N_TERN_EPI_RESIDUAL: return pre_shape<C, CPW, ACC, KT2N_TERN_EPI_RESIDUAL>(nt, wn, mt, a);
#endif
            default: return cudaErrorInvalidConfiguration;
        }
    }
    switch (epi) {
        case KT2N_TERN_EPI_STORE: return dec_shape<WK, CPW, ACC, KT2N_TERN_EPI_STORE>(nt, wn, a);
        case KT2N_TERN_EPI_SILU_MUL: return dec_shape<WK, CPW, ACC, KT2N_TERN_EPI_SILU_MUL>(nt, wn, a);
#if !KT2N_TERN_TUNE
        case KT2N_TERN_EPI_RESIDUAL: return dec_shape<WK, CPW, ACC, KT2N_TERN_EPI_RESIDUAL>(nt, wn, a);
#endif
        default: return cudaErrorInvalidConfiguration;
    }
}

#if KT2N_TERN_TUNE == 1
template <int C, int WK>
cudaError_t run_acc(int acc, int epi, bool prefill, int nt, int wn, int mt, const Args& a) {
    switch (acc) {
        case 1: return run_cfg<C, WK, 1>(epi, prefill, nt, wn, mt, a);
        case 2: return run_cfg<C, WK, 2>(epi, prefill, nt, wn, mt, a);
        case 4: return run_cfg<C, WK, 4>(epi, prefill, nt, wn, mt, a);
        default: return cudaErrorInvalidConfiguration;
    }
}
#endif

cudaError_t run_family(int fam, int wk, int acc, int epi, bool prefill, int nt, int wn, int mt, const Args& a) {
#if KT2N_TERN_TUNE == 1
    if (fam == KT2N_TERN_FAM_DOWN) {
        switch (wk) {
            case 6: return run_acc<24, 6>(acc, epi, prefill, nt, wn, mt, a);
            case 12: return run_acc<24, 12>(acc, epi, prefill, nt, wn, mt, a);
            case 24: return run_acc<24, 24>(acc, epi, prefill, nt, wn, mt, a);
            default: return cudaErrorInvalidConfiguration;
        }
    }
    switch (wk) {
        case 2: return run_acc<8, 2>(acc, epi, prefill, nt, wn, mt, a);
        case 4: return run_acc<8, 4>(acc, epi, prefill, nt, wn, mt, a);
        case 8: return run_acc<8, 8>(acc, epi, prefill, nt, wn, mt, a);
        default: return cudaErrorInvalidConfiguration;
    }
#else
    (void)wk;
    (void)acc;
    // Used only as template arguments, which gcc's -Wunused-but-set-variable does not count as a use.
    [[maybe_unused]] constexpr FamilyCfg q = family_cfg(KT2N_TERN_FAM_QKV);
    [[maybe_unused]] constexpr FamilyCfg o = family_cfg(KT2N_TERN_FAM_O);
    [[maybe_unused]] constexpr FamilyCfg gu = family_cfg(KT2N_TERN_FAM_GATE_UP);
    [[maybe_unused]] constexpr FamilyCfg d = family_cfg(KT2N_TERN_FAM_DOWN);
    switch (fam) {
        case KT2N_TERN_FAM_QKV: return run_cfg<q.k / kChunk, q.wk, q.acc>(epi, prefill, nt, wn, mt, a);
        case KT2N_TERN_FAM_O: return run_cfg<o.k / kChunk, o.wk, o.acc>(epi, prefill, nt, wn, mt, a);
        case KT2N_TERN_FAM_GATE_UP: return run_cfg<gu.k / kChunk, gu.wk, gu.acc>(epi, prefill, nt, wn, mt, a);
        default: return run_cfg<d.k / kChunk, d.wk, d.acc>(epi, prefill, nt, wn, mt, a);
    }
#endif
}

// NORM decode path (L3): qkv with EPI_STORE and gate_up with EPI_SILU_MUL only (the two projections that read a
// normed residual stream), default family split / accumulators.
cudaError_t run_family_norm(int fam, int epi, int nt, int wn, const Args& a) {
    [[maybe_unused]] constexpr FamilyCfg q = family_cfg(KT2N_TERN_FAM_QKV);
    [[maybe_unused]] constexpr FamilyCfg gu = family_cfg(KT2N_TERN_FAM_GATE_UP);
    if (fam == KT2N_TERN_FAM_QKV && epi == KT2N_TERN_EPI_STORE)
        return dec_shape<q.wk, (q.k / kChunk) / q.wk, q.acc, KT2N_TERN_EPI_STORE, true>(nt, wn, a);
    if (fam == KT2N_TERN_FAM_GATE_UP && epi == KT2N_TERN_EPI_SILU_MUL)
        return dec_shape<gu.wk, (gu.k / kChunk) / gu.wk, gu.acc, KT2N_TERN_EPI_SILU_MUL, true>(nt, wn, a);
    return cudaErrorInvalidConfiguration;
}

// ------------------------------------------------------------------------------------------------------------
// lm-proj-wide launch plumbing (default off; see tern_wide_kernel).
// ------------------------------------------------------------------------------------------------------------
// Clears a non-sticky error left by a failed query / attribute call, so a later cudaGetLastError() of another
// launcher does not report it, and reports "not available" (the caller then runs the k1 kernel).
cudaError_t wide_unavailable() {
    (void)cudaGetLastError();
    return cudaErrorInvalidConfiguration;
}

// Cluster launch support (cudaDevAttrClusterLaunch), once per device.
bool wide_cluster_ok(int dev) {
    static int ok[64] = {};  // 0 unknown, 1 yes, -1 no (a benign race: every writer stores the same value)
    if (dev < 0 || dev >= 64) return false;
    if (!ok[dev]) {
        int v = 0;
        const bool yes = cudaDeviceGetAttribute(&v, cudaDevAttrClusterLaunch, dev) == cudaSuccess && v != 0;
        if (!yes) (void)cudaGetLastError();
        ok[dev] = yes ? 1 : -1;
    }
    return ok[dev] > 0;
}

template <int WK, int CPW, int ACC, int H, int NT, int WN, int MT, int EPI, int MINB, bool ONEBUF>
cudaError_t wide_go(const Args& a) {
    constexpr int PPC = WK / H;
    constexpr int NTHR = PPC * WN * 32;
    constexpr int NBUF = ONEBUF ? 1 : CPW;
    constexpr size_t kRed = (size_t)WN * PPC * NT * MT * 32 * 16;       // partials, bytes
    constexpr size_t kActMax = (size_t)PPC * NBUF * (16 * MT) * 512;     // activation buffers at M = 16 * MT
    constexpr size_t kMax = kActMax > kRed ? kActMax : kRed;
    // 99 KB: the largest dynamic shared memory per block on sm_120 (sm_90 / sm_100 allow more).
    if constexpr (WK % H != 0 || NTHR > 1024 || H > 8 || kMax > 99 * 1024 || NTHR * MINB > 1536) {
        return cudaErrorInvalidConfiguration;
    } else {
        if (a.m < 1 || a.m > 16 * MT || a.n % (8 * NT * WN) || a.c != WK * CPW) return cudaErrorInvalidConfiguration;
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return wide_unavailable();
        if (!wide_cluster_ok(dev)) return cudaErrorInvalidConfiguration;
        auto kern = tern_wide_kernel<WK, CPW, ACC, H, NT, WN, MT, EPI, MINB, ONEBUF>;
        // Dynamic shared memory above 48 KB needs a per-function opt-in (idempotent; legal during stream capture,
        // as gemm_go relies on).  Set once per device for the largest size this instantiation ever launches.
        static int attr_set[64] = {};
        if (kMax > 48 * 1024 && dev < 64 && !attr_set[dev]) {
            if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kMax) != cudaSuccess)
                return wide_unavailable();
            attr_set[dev] = 1;
        }
        const size_t act = (size_t)PPC * NBUF * a.m * 512;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(H, a.n / (8 * NT * WN), 1);
        cfg.blockDim = dim3(NTHR, 1, 1);
        cfg.dynamicSmemBytes = act > kRed ? act : kRed;
        cfg.stream = a.stream;
        cudaLaunchAttribute attr[2];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = H;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[1].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        // Can a cluster of this shape be resident at all (cluster size, shared memory)?  Asked once per device at the
        // largest shared memory this instantiation uses; 0 clusters = not available (the k1 kernel runs instead).
        static int fits[64] = {};  // 0 unknown, 1 yes, -1 no
        if (dev < 64 && !fits[dev]) {
            cudaLaunchConfig_t q = cfg;
            q.dynamicSmemBytes = kMax;
            int nc = 0;
            const bool yes = cudaOccupancyMaxActiveClusters(&nc, kern, &q) == cudaSuccess && nc > 0;
            if (!yes) (void)cudaGetLastError();
            fits[dev] = yes ? 1 : -1;
        }
        if (dev >= 64 || fits[dev] < 0) return cudaErrorInvalidConfiguration;
        cfg.numAttrs = a.pdl ? 2 : 1;
        // A failed launch is an error (ECUDA), never the silent "not available" (EINVAL) of the checks above.
        const cudaError_t e =
            cudaLaunchKernelEx(&cfg, kern, a.x, a.ldx, a.m, a.w, a.s, a.y, a.ldy, a.r, a.ldr, a.c, a.early);
        return e == cudaErrorInvalidConfiguration ? cudaErrorLaunchFailure : e;
    }
}

// Shapes {H = CTAs per cluster, NT = column tiles per warp, WN = warps along N per CTA, MINB = __launch_bounds__ min
// blocks per SM (1 = no register cap), ONEBUF} per family, shape s = 1..3 (0 = shape 1).  All rows (<= 32) in one CTA
// as two 16-row tiles (MT = 2).  A shape never changes a result bit (each row's arithmetic is the family's); only the
// grid, shared memory and registers differ.  Chosen so the grid fits one wave on the 188-SM RTX PRO 6000 at 32 rows
// (sm_120 ptxas registers, uncapped: qkv / o 106-108, gate_up NT 2 133, down NT 2 88-101):
//   qkv s1 64 x 8 CTAs of 128 threads, 16 KB of activations (4 CTAs / SM); o s1 32 x 8 of 128;
//   gate_up s1 128 x 4 of 128, NT 3, 32 KB (3 / SM); down s1 32 x 6 of 256, one 32 KB buffer (2 / SM).
// Which shape is fastest is measured at T; the step takes one per family from
// KT2N_LM_PROJ_WIDE_SHAPE.
constexpr int kWideShapes = 3;
constexpr int kWide[4][kWideShapes][5] = {
    {{8, 2, 4, 1, 0}, {8, 2, 2, 1, 0}, {4, 2, 4, 1, 0}},  // qkv     (WK 8, CPW 1, ACC 4)
    {{8, 2, 4, 1, 0}, {8, 2, 2, 1, 0}, {4, 2, 2, 1, 0}},  // o       (WK 8, CPW 1, ACC 4)
    {{4, 3, 4, 3, 0}, {4, 2, 8, 2, 0}, {4, 2, 4, 4, 0}},  // gate_up (WK 4, CPW 2, ACC 4)
    {{6, 2, 4, 2, 1}, {6, 2, 2, 1, 1}, {6, 2, 4, 1, 0}},  // down    (WK 12, CPW 2, ACC 2)
};

template <int C, int WK, int ACC, int EPI, int FAM>
cudaError_t wide_family(int shape, const Args& a) {
    constexpr int CPW = C / WK;
    static_assert(CPW * WK == C, "split must tile the chunks");
#define KT2N_WIDE(S_)                                                                                    \
    wide_go<WK, CPW, ACC, kWide[FAM][S_][0], kWide[FAM][S_][1], kWide[FAM][S_][2], 2, EPI, kWide[FAM][S_][3], \
            kWide[FAM][S_][4] != 0>(a)
    switch (shape) {
        case 0:
        case 1: return KT2N_WIDE(0);
        case 2: return KT2N_WIDE(1);
        case 3: return KT2N_WIDE(2);
        default: return cudaErrorInvalidConfiguration;
    }
#undef KT2N_WIDE
}

// The decode step's epilogue per family only: qkv / o / down STORE, gate_up SILU_MUL (others: not available).
cudaError_t run_wide(int fam, int epi, int shape, const Args& a) {
    [[maybe_unused]] constexpr FamilyCfg q = family_cfg(KT2N_TERN_FAM_QKV);
    [[maybe_unused]] constexpr FamilyCfg o = family_cfg(KT2N_TERN_FAM_O);
    [[maybe_unused]] constexpr FamilyCfg gu = family_cfg(KT2N_TERN_FAM_GATE_UP);
    [[maybe_unused]] constexpr FamilyCfg d = family_cfg(KT2N_TERN_FAM_DOWN);
    switch (fam) {
        case KT2N_TERN_FAM_QKV:
            if (epi != KT2N_TERN_EPI_STORE) return cudaErrorInvalidConfiguration;
            return wide_family<q.k / kChunk, q.wk, q.acc, KT2N_TERN_EPI_STORE, KT2N_TERN_FAM_QKV>(shape, a);
        case KT2N_TERN_FAM_O:
            if (epi != KT2N_TERN_EPI_STORE) return cudaErrorInvalidConfiguration;
            return wide_family<o.k / kChunk, o.wk, o.acc, KT2N_TERN_EPI_STORE, KT2N_TERN_FAM_O>(shape, a);
        case KT2N_TERN_FAM_GATE_UP:
            if (epi != KT2N_TERN_EPI_SILU_MUL) return cudaErrorInvalidConfiguration;
            return wide_family<gu.k / kChunk, gu.wk, gu.acc, KT2N_TERN_EPI_SILU_MUL, KT2N_TERN_FAM_GATE_UP>(shape, a);
        default:
            if (epi != KT2N_TERN_EPI_STORE) return cudaErrorInvalidConfiguration;
            return wide_family<d.k / kChunk, d.wk, d.acc, KT2N_TERN_EPI_STORE, KT2N_TERN_FAM_DOWN>(shape, a);
    }
}

// Decode launch shape {nt, wn} per (family, M = 1..16) (never changes results).  Measured on the RTX PRO 6000 with
// 28-launch graphs, L2 flushed, PDL: codegen of a shape can be slow at some
// M (e.g. qkv {2,2} at M = 7..8), so the table is per M, not a rule.
constexpr unsigned char kDecShape[4][16][2] = {
    // qkv
    {{1, 2}, {1, 2}, {1, 2}, {1, 2}, {1, 2}, {4, 2}, {4, 2}, {4, 2}, {2, 2}, {2, 2}, {4, 2}, {4, 2}, {4, 2}, {4, 2},
     {4, 2}, {4, 2}},
    // o
    {{1, 2}, {1, 2}, {1, 2}, {4, 1}, {4, 1}, {4, 1}, {4, 1}, {4, 1}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2},
     {2, 2}, {2, 2}},
    // gate_up
    {{1, 2}, {2, 2}, {2, 1}, {2, 1}, {2, 2}, {2, 1}, {4, 1}, {4, 1}, {4, 1}, {4, 1}, {4, 1}, {4, 1}, {4, 1}, {4, 1},
     {4, 1}, {4, 1}},
    // down
    {{1, 2}, {1, 2}, {1, 2}, {1, 2}, {1, 2}, {2, 2}, {4, 1}, {4, 1}, {4, 1}, {4, 1}, {4, 1}, {4, 1}, {4, 1}, {4, 1},
     {4, 1}, {4, 1}},
};

void auto_shape(int fam, int m, int n, int& nt, int& wn, int& mt) {
    if (m <= 16) {
        if (nt <= 0) nt = kDecShape[fam][m - 1][0];
        if (wn <= 0) wn = kDecShape[fam][m - 1][1];
        mt = 1;
    } else {
        if (mt == 0) {
            // M > 16 (tune_shapes.py --prefill-only, M = 17..512, RTX PRO 6000): the register-tiled kernel for
            // small M, the shared-memory GEMM kernel above the crossover.  GEMM codes: kmt = -(BMW + 100*(MTW-1)),
            // knt = NTW, kwn = BNW; -101 = one warp row of 2 x 16-row tiles.
            int knt = 4, kwn = 1, kmt = 1;  // register-tiled, 4 column tiles per warp
            switch (fam) {
                case KT2N_TERN_FAM_QKV:
                    if (m <= 48) { kwn = 2; }
                    else if (m <= 256) { knt = 2; kwn = 4; kmt = -101; }
                    else { knt = 4; kwn = 4; kmt = -101; }
                    break;
                case KT2N_TERN_FAM_O:
                    if (m <= 48) { kwn = 1; }
                    else if (m <= 80) { kwn = 2; }
                    else { knt = 2; kwn = 4; kmt = -101; }
                    break;
                case KT2N_TERN_FAM_GATE_UP:
                    if (m > 48) { knt = 4; kwn = 4; kmt = -101; }
                    break;
                default:
                    if (m <= 80) { }
                    else if (m <= 256) { knt = 2; kwn = 4; kmt = -101; }
                    else if (m <= 400) { knt = 2; kwn = 8; kmt = -101; }
                    else { knt = 2; kwn = 8; kmt = -301; }
                    break;
            }
            if (nt <= 0) nt = knt;
            if (wn <= 0) wn = kwn;
            mt = kmt;
        } else if (mt < 0) {  // shared-memory GEMM kernel: NTW = nt, BNW = wn, BMW = -mt
            if (nt <= 0) nt = 4;
            if (wn <= 0) wn = 4;
        } else {  // register-tiled kernel
            if (nt <= 0) nt = 4;
            if (wn <= 0) wn = 1;
        }
    }
    // the grid must tile N exactly
    while (nt > 1 && n % (8 * nt * wn)) nt >>= 1;
    while (wn > 1 && n % (8 * nt * wn)) wn >>= 1;
}

inline bool aligned(const void* p, size_t a) { return (reinterpret_cast<uintptr_t>(p) % a) == 0; }

int grid_for(size_t total) {
    size_t blocks = (total + 255) / 256;
    if (blocks > 65535u * 8u) blocks = 65535u * 8u;
    return (int)(blocks ? blocks : 1);
}

// kt2n_tern_repack's argument rule; the family helpers check every component with it before the first launch.
bool repack_ok(const void* src_codes, const void* src_scales, int rows, int k, const void* dst_codes,
               const void* dst_scales, int n_total, int row_offset, int row_step) {
    return src_codes && src_scales && dst_codes && dst_scales && rows > 0 && k > 0 && k % kChunk == 0 &&
           n_total > 0 && n_total % 8 == 0 && row_offset >= 0 && row_step > 0 &&
           (int64_t)row_offset + (int64_t)(rows - 1) * row_step < n_total;
}

}  // namespace

// ------------------------------------------------------------------------------------------------------------
// C API
// ------------------------------------------------------------------------------------------------------------
extern "C" {

size_t kt2n_tern_codes_bytes(int n, int k) { return (size_t)n * (size_t)k / 4; }
size_t kt2n_tern_scales_bytes(int n, int k) { return (size_t)n * (size_t)(k / kGroup) * 2; }

int kt2n_tern_family_split(int family, int* warps_k, int* chunks_per_split) {
    if (family < 0 || family > 3) return KT2N_TERN_EINVAL;
    const FamilyCfg f = family_cfg(family);
    int wk = f.wk;
#if KT2N_TERN_TUNE == 1
    if (g_wk_override[family] > 0) wk = g_wk_override[family];
#endif
    if (warps_k) *warps_k = wk;
    if (chunks_per_split) *chunks_per_split = (f.k / kChunk) / wk;
    return KT2N_TERN_OK;
}

int kt2n_tern_family_acc(int family) {
    if (family < 0 || family > 3) return -1;
#if KT2N_TERN_TUNE == 1
    if (g_acc_override[family] > 0) return g_acc_override[family];
#endif
    return family_cfg(family).acc;
}

#if KT2N_TERN_TUNE
// Tune builds only (measurement): K split and accumulators (change rounding), trigger position (does not).
int kt2n_tern_tune_set_split(int family, int warps_k) {
    if (family < 0 || family > 3) return KT2N_TERN_EINVAL;
    g_wk_override[family] = warps_k;
    return KT2N_TERN_OK;
}
int kt2n_tern_tune_set_acc(int family, int acc) {
    if (family < 0 || family > 3) return KT2N_TERN_EINVAL;
    g_acc_override[family] = acc;
    return KT2N_TERN_OK;
}
int kt2n_tern_tune_set_early(int early) {
    g_early_override = early;
    return KT2N_TERN_OK;
}
#endif

int kt2n_tern_repack(const int32_t* src_codes, const void* src_scales, int rows, int k, void* dst_codes,
                     void* dst_scales, int n_total, int row_offset, int row_step, cudaStream_t stream) {
    if (!repack_ok(src_codes, src_scales, rows, k, dst_codes, dst_scales, n_total, row_offset, row_step))
        return KT2N_TERN_EINVAL;
    const int C = k / kChunk, kw = k / 16, groups = k / kGroup;
    repack_codes_kernel<<<grid_for((size_t)rows * kw), 256, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(src_codes), rows, kw, static_cast<uint32_t*>(dst_codes), C, row_offset,
        row_step);
    repack_scales_kernel<<<grid_for((size_t)rows * groups), 256, 0, stream>>>(
        static_cast<const uint16_t*>(src_scales), rows, groups, static_cast<uint16_t*>(dst_scales), C, row_offset,
        row_step);
    return cudaGetLastError() == cudaSuccess ? KT2N_TERN_OK : KT2N_TERN_ECUDA;
}

int kt2n_tern_unrepack(const void* codes, const void* scales, int n_total, int k, int row_offset, int row_step,
                       int rows, int32_t* dst_codes, void* dst_scales, cudaStream_t stream) {
    if (!codes || !scales || !dst_codes || !dst_scales || rows <= 0 || k <= 0 || k % kChunk || n_total <= 0 ||
        n_total % 8 || row_offset < 0 || row_step <= 0 ||
        (int64_t)row_offset + (int64_t)(rows - 1) * row_step >= n_total)
        return KT2N_TERN_EINVAL;
    const int C = k / kChunk, kw = k / 16, groups = k / kGroup;
    unrepack_codes_kernel<<<grid_for((size_t)rows * kw), 256, 0, stream>>>(
        static_cast<const uint32_t*>(codes), rows, kw, reinterpret_cast<uint32_t*>(dst_codes), C, row_offset,
        row_step);
    unrepack_scales_kernel<<<grid_for((size_t)rows * groups), 256, 0, stream>>>(
        static_cast<const uint16_t*>(scales), rows, groups, static_cast<uint16_t*>(dst_scales), C, row_offset,
        row_step);
    return cudaGetLastError() == cudaSuccess ? KT2N_TERN_OK : KT2N_TERN_ECUDA;
}

int kt2n_tern_decode_dense(const void* codes, const void* scales, int n, int k, void* dense_out,
                           cudaStream_t stream) {
    if (!codes || !scales || !dense_out || n <= 0 || n % 8 || k <= 0 || k % kChunk || !aligned(codes, 16))
        return KT2N_TERN_EINVAL;
    decode_dense_kernel<<<grid_for((size_t)(n / 8) * (k / kChunk) * 32), 256, 0, stream>>>(
        static_cast<const uint4*>(codes), static_cast<const uint16_t*>(scales), n, k,
        static_cast<uint16_t*>(dense_out));
    return cudaGetLastError() == cudaSuccess ? KT2N_TERN_OK : KT2N_TERN_ECUDA;
}

int kt2n_tern_repack_qkv(const int32_t* q_codes, const void* q_scales, const int32_t* k_codes, const void* k_scales,
                         const int32_t* v_codes, const void* v_scales, int nq, int nkv, int k, void* dst_codes,
                         void* dst_scales, cudaStream_t stream) {
    // Every component is checked before the first launch (a bad k/v argument must not leave q repacked).
    if (nq <= 0 || nkv <= 0 || (int64_t)nq + 2 * (int64_t)nkv > INT32_MAX) return KT2N_TERN_EINVAL;
    const int n_total = nq + 2 * nkv;
    if (!repack_ok(q_codes, q_scales, nq, k, dst_codes, dst_scales, n_total, 0, 1) ||
        !repack_ok(k_codes, k_scales, nkv, k, dst_codes, dst_scales, n_total, nq, 1) ||
        !repack_ok(v_codes, v_scales, nkv, k, dst_codes, dst_scales, n_total, nq + nkv, 1))
        return KT2N_TERN_EINVAL;
    int rc = kt2n_tern_repack(q_codes, q_scales, nq, k, dst_codes, dst_scales, n_total, 0, 1, stream);
    if (rc) return rc;
    rc = kt2n_tern_repack(k_codes, k_scales, nkv, k, dst_codes, dst_scales, n_total, nq, 1, stream);
    if (rc) return rc;
    return kt2n_tern_repack(v_codes, v_scales, nkv, k, dst_codes, dst_scales, n_total, nq + nkv, 1, stream);
}

int kt2n_tern_repack_gate_up(const int32_t* gate_codes, const void* gate_scales, const int32_t* up_codes,
                             const void* up_scales, int inter, int k, void* dst_codes, void* dst_scales,
                             cudaStream_t stream) {
    if (inter <= 0 || inter > INT32_MAX / 2 ||
        !repack_ok(gate_codes, gate_scales, inter, k, dst_codes, dst_scales, 2 * inter, 0, 2) ||
        !repack_ok(up_codes, up_scales, inter, k, dst_codes, dst_scales, 2 * inter, 1, 2))
        return KT2N_TERN_EINVAL;
    int rc = kt2n_tern_repack(gate_codes, gate_scales, inter, k, dst_codes, dst_scales, 2 * inter, 0, 2, stream);
    if (rc) return rc;
    return kt2n_tern_repack(up_codes, up_scales, inter, k, dst_codes, dst_scales, 2 * inter, 1, 2, stream);
}

int kt2n_tern_proj_norm(int family, const void* x, int64_t ldx, int m, const void* ln_w, float eps,
                        const void* codes, const void* scales, int n, int k, void* y, int64_t ldy, int epi,
                        unsigned flags, cudaStream_t stream) {
    if (!((family == KT2N_TERN_FAM_QKV && epi == KT2N_TERN_EPI_STORE) ||
          (family == KT2N_TERN_FAM_GATE_UP && epi == KT2N_TERN_EPI_SILU_MUL)))
        return KT2N_TERN_EINVAL;
    const FamilyCfg f = family_cfg(family);
    if (k != f.k || n <= 0 || n % 8 || m < 1 || m > 16 || !x || !ln_w || !codes || !scales || !y) return KT2N_TERN_EINVAL;
    if (!aligned(x, 16) || !aligned(ln_w, 16) || ldx < k || ldx % 8 || ldx > INT32_MAX) return KT2N_TERN_EINVAL;
    if (!aligned(codes, 16) || !aligned(scales, 2)) return KT2N_TERN_EINVAL;
    if (epi == KT2N_TERN_EPI_SILU_MUL) {
        if (!aligned(y, 2) || ldy < n / 2 || ldy > INT32_MAX) return KT2N_TERN_EINVAL;
    } else {
        if (!aligned(y, 4) || ldy < n || ldy % 2 || ldy > INT32_MAX) return KT2N_TERN_EINVAL;
    }
    int nt = 0, wn = 0, mt = 0;
    auto_shape(family, m, n, nt, wn, mt);
    if (n % (8 * nt * wn)) return KT2N_TERN_EINVAL;
    Args a;
    a.x = static_cast<const bf16*>(x);
    a.ldx = (int)ldx;
    a.m = m;
    a.w = static_cast<const uint4*>(codes);
    a.s = static_cast<const uint16_t*>(scales);
    a.y = static_cast<bf16*>(y);
    a.ldy = (int)ldy;
    a.r = nullptr;
    a.ldr = 0;
    a.n = n;
    a.c = k / kChunk;
    a.pdl = (flags & KT2N_TERN_FLAG_PDL) != 0;
    a.early = KT2N_TERN_EARLY_TRIGGER;
    a.stream = stream;
    a.lnw = static_cast<const bf16*>(ln_w);
    a.eps = eps;
    const cudaError_t err = run_family_norm(family, epi, nt, wn, a);
    if (err == cudaErrorInvalidConfiguration) return KT2N_TERN_EINVAL;
    return err == cudaSuccess ? KT2N_TERN_OK : KT2N_TERN_ECUDA;
}

int kt2n_tern_proj_ex(int family, const void* x, int64_t ldx, int m, const void* codes, const void* scales, int n,
                      int k, void* y, int64_t ldy, int epi, const void* residual, int64_t ldr, unsigned flags,
                      int nt, int wn, int mt, cudaStream_t stream) {
    if (family < 0 || family > 3 || epi < 0 || epi > 2) return KT2N_TERN_EINVAL;
    const FamilyCfg f = family_cfg(family);
    int wk = 0;
    kt2n_tern_family_split(family, &wk, nullptr);
    const int acc = kt2n_tern_family_acc(family);
    if (k != f.k || n <= 0 || n % 8 || m < 1 || !x || !codes || !scales || !y) return KT2N_TERN_EINVAL;
    if (!aligned(x, 16) || ldx < k || ldx % 8 || ldx > INT32_MAX) return KT2N_TERN_EINVAL;
    if (!aligned(codes, 16) || !aligned(scales, 2)) return KT2N_TERN_EINVAL;
    if (epi == KT2N_TERN_EPI_SILU_MUL) {
        if (family != KT2N_TERN_FAM_GATE_UP || !aligned(y, 2) || ldy < n / 2 || ldy > INT32_MAX) return KT2N_TERN_EINVAL;
    } else {
        if (!aligned(y, 4) || ldy < n || ldy % 2 || ldy > INT32_MAX) return KT2N_TERN_EINVAL;
    }
    if (epi == KT2N_TERN_EPI_RESIDUAL) {
        if (!residual || !aligned(residual, 4) || ldr < n || ldr % 2 || ldr > INT32_MAX) return KT2N_TERN_EINVAL;
    }
    const bool prefill = m > 16;
    if (mt < 0 && !aligned(scales, 16)) return KT2N_TERN_EINVAL;  // GEMM kernel copies scales in 16-B pieces
    const bool auto_kernel = mt == 0;
    auto_shape(family, m, n, nt, wn, mt);
    if (auto_kernel && mt < 0 && !aligned(scales, 16)) {  // GEMM needs 16-B aligned scales: tiled kernel instead
        nt = 4;
        wn = 1;
        mt = 1;
    }
    if (n % (8 * nt * wn)) return KT2N_TERN_EINVAL;
    Args a;
    a.x = static_cast<const bf16*>(x);
    a.ldx = (int)ldx;
    a.m = m;
    a.w = static_cast<const uint4*>(codes);
    a.s = static_cast<const uint16_t*>(scales);
    a.y = static_cast<bf16*>(y);
    a.ldy = (int)ldy;
    a.r = static_cast<const bf16*>(residual);
    a.ldr = (int)(epi == KT2N_TERN_EPI_RESIDUAL ? ldr : 0);
    a.n = n;
    a.c = k / kChunk;
    a.pdl = (flags & KT2N_TERN_FLAG_PDL) != 0;
    a.early = KT2N_TERN_EARLY_TRIGGER;
#if KT2N_TERN_TUNE
    if (g_early_override >= 0) a.early = g_early_override;
#endif
    a.stream = stream;
#if KT2N_TERN_PRODUCTION_ORDER
    const cudaError_t err = !kt2n_tern_uses_production_decode(flags)
                                 ? run_family(family, wk, acc, epi, prefill, nt, wn, mt, a)
                                 : m == 1 ? (family == KT2N_TERN_FAM_DOWN ? production_b1_go<32>(epi, a)
                                                                                : production_b1_go<16>(epi, a))
                                 : m <= 16 ? (family == KT2N_TERN_FAM_QKV ? production_tc_go<2>(epi, a)
                                              : family == KT2N_TERN_FAM_GATE_UP ? production_tc_go<1>(epi, a)
                                                                                : production_tc_go<4>(epi, a))
                                           : run_family(family, wk, acc, epi, prefill, nt, wn, mt, a);
#else
    const cudaError_t err = run_family(family, wk, acc, epi, prefill, nt, wn, mt, a);
#endif
    if (err == cudaErrorInvalidConfiguration) return KT2N_TERN_EINVAL;  // shape not instantiated
    return err == cudaSuccess ? KT2N_TERN_OK : KT2N_TERN_ECUDA;
}

int kt2n_tern_proj(int family, const void* x, int64_t ldx, int m, const void* codes, const void* scales, int n,
                   int k, void* y, int64_t ldy, int epi, const void* residual, int64_t ldr, unsigned flags,
                   cudaStream_t stream) {
    return kt2n_tern_proj_ex(family, x, ldx, m, codes, scales, n, k, y, ldy, epi, residual, ldr, flags, 0, 0, 0,
                             stream);
}

int kt2n_tern_proj_wide(int family, const void* x, int64_t ldx, int m, const void* codes, const void* scales, int n,
                        int k, void* y, int64_t ldy, int epi, const void* residual, int64_t ldr, unsigned flags,
                        int shape, cudaStream_t stream) {
    if (family < 0 || family > 3 || epi < 0 || epi > 2 || shape < 0 || shape > kWideShapes) return KT2N_TERN_EINVAL;
    const FamilyCfg f = family_cfg(family);
    if (k != f.k || n <= 0 || n % 8 || m < 1 || m > 32 || !x || !codes || !scales || !y) return KT2N_TERN_EINVAL;
    if (!aligned(x, 16) || ldx < k || ldx % 8 || ldx > INT32_MAX) return KT2N_TERN_EINVAL;
    if (!aligned(codes, 16) || !aligned(scales, 2)) return KT2N_TERN_EINVAL;
    if (epi == KT2N_TERN_EPI_SILU_MUL) {
        if (family != KT2N_TERN_FAM_GATE_UP || !aligned(y, 2) || ldy < n / 2 || ldy > INT32_MAX) return KT2N_TERN_EINVAL;
    } else {
        if (!aligned(y, 4) || ldy < n || ldy % 2 || ldy > INT32_MAX) return KT2N_TERN_EINVAL;
    }
    if (epi == KT2N_TERN_EPI_RESIDUAL) {
        if (!residual || !aligned(residual, 4) || ldr < n || ldr % 2 || ldr > INT32_MAX) return KT2N_TERN_EINVAL;
    }
#if KT2N_TERN_PRODUCTION_ORDER
    // That build gives decode rows 1..16 another arithmetic class (production order); never mix it with this one.
    if (m <= 16 && kt2n_tern_uses_production_decode(flags)) return KT2N_TERN_EINVAL;
#endif
#if KT2N_TERN_TUNE == 1
    // Tune builds may override a family's split / accumulators for the k1 kernels; this kernel has the frozen ones.
    if (g_wk_override[family] > 0 || g_acc_override[family] > 0) return KT2N_TERN_EINVAL;
#endif
    Args a;
    a.x = static_cast<const bf16*>(x);
    a.ldx = (int)ldx;
    a.m = m;
    a.w = static_cast<const uint4*>(codes);
    a.s = static_cast<const uint16_t*>(scales);
    a.y = static_cast<bf16*>(y);
    a.ldy = (int)ldy;
    a.r = static_cast<const bf16*>(residual);
    a.ldr = (int)(epi == KT2N_TERN_EPI_RESIDUAL ? ldr : 0);
    a.n = n;
    a.c = k / kChunk;
    a.pdl = (flags & KT2N_TERN_FLAG_PDL) != 0;
    a.early = KT2N_TERN_EARLY_TRIGGER;
#if KT2N_TERN_TUNE
    if (g_early_override >= 0) a.early = g_early_override;
#endif
    a.stream = stream;
    const cudaError_t err = run_wide(family, epi, shape, a);
    if (err == cudaErrorInvalidConfiguration) return KT2N_TERN_EINVAL;  // not available: nothing was launched
    return err == cudaSuccess ? KT2N_TERN_OK : KT2N_TERN_ECUDA;
}

int kt2n_tern_proj_wide_shape(int family, int shape, int* cluster, int* nt, int* wn) {
    if (family < 0 || family > 3 || shape < 0 || shape > kWideShapes) return KT2N_TERN_EINVAL;
    const int s = shape == 0 ? 0 : shape - 1;
    if (cluster) *cluster = kWide[family][s][0];
    if (nt) *nt = kWide[family][s][1];
    if (wn) *wn = kWide[family][s][2];
    return KT2N_TERN_OK;
}

int kt2n_tern_test_silu_mul(const void* gate, const void* up, void* out, int count, cudaStream_t stream) {
    if (!gate || !up || !out || count <= 0) return KT2N_TERN_EINVAL;
    silu_mul_test_kernel<<<(count + 255) / 256, 256, 0, stream>>>(static_cast<const bf16*>(gate),
                                                                  static_cast<const bf16*>(up), static_cast<bf16*>(out),
                                                                  count);
    return cudaGetLastError() == cudaSuccess ? KT2N_TERN_OK : KT2N_TERN_ECUDA;
}

}  // extern "C"
