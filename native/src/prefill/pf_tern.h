// prof-prefill variant: register-lean multi-stage ternary GEMM for the prefill projections (see pf_tern.cu).
// Internal to the prefill (not part of the public C API); bitwise identical to kt2n_tern_proj for every shape.
#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kt2n_pf_tern_args {
  int family;  // KT2N_TERN_FAM_*
  int epi;     // KT2N_TERN_EPI_* (qkv: STORE, o / down: RESIDUAL, gate_up: SILU_MUL)
  const void* x;
  int64_t ldx;
  int m;
  const void* codes;   // k1 kernel layout (kt2n_tern_repack*), 16-B aligned
  const void* scales;  // 16-B aligned
  int n;
  void* y;
  int64_t ldy;
  const void* residual;
  int64_t ldr;
  int pdl;
  cudaStream_t stream;
} kt2n_pf_tern_args;

// Shape code = WM*100000 + WN*10000 + MT*1000 + NT*100 + ST*10 + P: WM x WN warps per block, each warp MT 16-row x
// NT 8-column tiles, ST pipeline stages (one K split each), P accumulator chains in flight (1 or 2).
// Returns KT2N_TERN_OK, KT2N_TERN_EINVAL (shape not instantiated / bad arguments) or KT2N_TERN_ECUDA.
int kt2n_pf_tern_proj(const kt2n_pf_tern_args* args, int shape_code);

// The prefill's projection shape for (family, M) under KT2N_PF_FLAG_PROJ_TABLE: returns 1 with either *lean (a lean
// shape code) or *lean = 0 and an explicit k1 shape (*nt, *wn, *mt for kt2n_tern_proj_ex); 0 = keep k1's choice.
int kt2n_pf_proj_choice(int family, int m, int* lean, int* nt, int* wn, int* mt);

#ifdef __cplusplus
}
#endif
