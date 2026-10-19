// prof-prefill variant: prefill attention over tiles of <= 8 queries with k3's decode arithmetic (pf_attn.cu).
// Internal to the prefill; bitwise identical to kt2n_attn_prefill / kt2n_attn_decode_* at every position.
#pragma once

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One tile: queries q_start .. q_start + nq - 1 (rows of q / out), logical positions pos0 .. pos0 + nq - 1 (query i
// attends keys 0 .. pos0 + i of slot `slot`, column = logical key).
typedef struct kt2n_pf_q8_tile {
  int32_t q_start;
  int32_t nq;  // 1..8
  int32_t pos0;
  int32_t slot;
} kt2n_pf_q8_tile;

size_t kt2n_pf_q8_workspace_bytes(int tiles, int max_keys);
// Same layout contract as kt2n_attn_prefill (q / out [tokens, 16 heads x 128], K/V slot rows).
// mode 0 = auto (split-parallel when the workspace covers the call and tiles x 8 < KT2N_PF_Q8_SEQ_MIN_CTAS, else
// sequential), 1 = split-parallel (needs the workspace: zeroed once, counters stay zero between launches,
// shareable with stream-ordered k3 launches; <= 128 tiles), 2 = sequential (no workspace).  Same bits in all modes.
#define KT2N_PF_Q8_SEQ_MIN_CTAS 376
int kt2n_pf_q8_attention(const void* q, int64_t q_row_stride, const void* k_cache, const void* v_cache,
                         int64_t kv_slot_stride, int64_t kv_head_stride, const kt2n_pf_q8_tile* tiles, int num_tiles,
                         int max_keys, void* out, int64_t out_row_stride, void* workspace, size_t workspace_bytes,
                         int pdl, int mode, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
