// kt2n native HiFT vocoder + streaming tail (lane L6, 10 Oct 2026). C ABI.
//
// Production reference: kittenml-api ebcdd3f kitten_tts_09/incremental_s3gen.py
// (IncrementalS3Gen._hift_rows :3031, launch :3258 -> the per-row tail :3700-3830,
// splice_published_tail :299), acoustic_graphs.AcousticGraphs._hift_core after
// hift_fast_20260927 (weight norm folded, Snake as one kernel).
//
// One call renders B rectangular rows of T mel frames (production groups only
// equal-width rows too). Per-row streaming state lives in caller-owned device
// buffers named by HiftRow; the planner (kt2n_hift_plan) computes the per-row
// scalars exactly as production's launch() does.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum Kt2nHiftRowFlags {
  KT2N_HIFT_FIRST = 1,        // no held tail yet: apply trim_fade (20 ms zeros + 20 ms cosine rise)
  KT2N_HIFT_SHORT_XFADE = 2,  // overlap < 3840 under G_SHORT_FIRST_COMMIT: whole np.hamming(2*overlap)
  KT2N_HIFT_SPLICE = 4,       // splice_published_tail of the previous first-commit tail
  KT2N_HIFT_HOLD = 8,         // not a final: write held tail, source cache, mel cache
};

// Device-side row descriptor (one per row; 192 bytes).
typedef struct Kt2nHiftRow {
  uint64_t seed;            // HiFT seed = 3,000,000 + seed * 10,000 + committed_tokens (Philox + mt19937)
  const float* src_in;      // previous source cache [src_len] (null when src_len == 0)
  float* src_out;           // new source cache [min(3840, L)] (KT2N_HIFT_HOLD)
  float* mel_cache_out;     // new mel cache [80][8] slot layout, first min(8, T) columns (KT2N_HIFT_HOLD)
  const float* held_in;     // previous speech tail [held_len] (null when FIRST)
  float* held_out;          // new speech tail [tail_len] (KT2N_HIFT_HOLD)
  float* out;               // published samples [out_len], clipped to +-1
  int32_t src_len;
  int32_t held_len;
  int32_t flags;
  int32_t emit_len;         // samples of the (head-processed) waveform that are emitted
  int32_t overlap;          // seam crossfade length (non-FIRST rows)
  int32_t skip;             // splice: head samples already played
  int32_t splice_count;     // splice: linear-ramp length
  int32_t splice_start;     // splice: first sample of held_in that is blended
  int32_t out_len;          // samples written to out
  int32_t tail_len;         // new held tail length (KT2N_HIFT_HOLD)
  int32_t reserved[2];
  float phase[12];          // harmonic phases [9] (phase[0] = 0), from kt2n_hift_phases
  int32_t reserved2[10];
} Kt2nHiftRow;

// Host planner input / output for one row (mirrors IncrementalS3Gen.launch).
typedef struct Kt2nHiftPlanIn {
  int32_t frames;            // T: vocoder input frames (mel cache included)
  int32_t finalize;          // final flush / whole render
  int32_t publish_held_tail; // launch(publish_held_tail=True)
  int32_t preserve_next_commit;
  int32_t first_commit_tail; // S3GEN_FIRST_COMMIT_TAIL (live 1)
  int32_t short_first;       // G_SHORT_FIRST_COMMIT (live 1)
  int32_t row_fade;          // first-commit tail splice samples (live 480)
  int32_t commit_index;      // state.commit_index
  int32_t held_len;          // len(state.speech_tail), 0 = None
  int32_t pending_tail;      // state.published_tail_samples
} Kt2nHiftPlanIn;

typedef struct Kt2nHiftPlanOut {
  int32_t flags, emit_len, overlap, skip, splice_count, splice_start, out_len, tail_len;
  int32_t new_pending_tail;  // state.published_tail_samples after this call
  int32_t new_mel_cache;     // mel-cache frames after this call (0 = unchanged / final)
  int32_t new_src_len;       // source-cache samples after this call
  int32_t error;             // 1 = "incremental waveform shorter than HiFT cache"
} Kt2nHiftPlanOut;

// weights: 319 device pointers (spike order, prod_render.hift_weight_list; stride-1 conv weights
// transposed to [cin][K][cout]). max_frame_rows bounds B * T of any call; max_b bounds B.
void* kt2n_hift_create(const void* const* weights, int n_weights, int max_b, int max_t);
// Same, with the workspace sized for max_frame_rows = max over calls of B * T (<= max_b * max_t).
void* kt2n_hift_create2(const void* const* weights, int n_weights, int max_b, int max_t, int max_frame_rows);
const char* kt2n_hift_last_error(void);
void kt2n_hift_destroy(void* h);
// conv_mode 0 split-kernel only, 1 fp32 SIMT big rows, 2 TF32 mma big rows, 3 3xTF32;
// big_min_tiles: rows with >= this many 64x128 tiles use the big kernel; pdl 0/1; parallel 0/1 (streams).
void kt2n_hift_config(int conv_mode, int big_min_tiles, int pdl, int parallel);
// Per-handle work policy: convs with >= tc_min_k * 1024 outputs per call use the tensor cores (-1: the global
// default 128, tuned for first-commit latency; a BULK handle sharing the GPU should use a lower value, e.g. 16:
// measured to use ~half the GPU work per small call; sms > 0 sizes the split heuristics for that
// many SMs. Call after create, before the first run or capture. Returns 0 on success.
int kt2n_hift_set_policy(void* h, int tc_min_k, int sms);
// Per-handle resblock grouping (v26): 0 = the three resblock branches of a stage on three streams (v21, default);
// 1 = one tensor-core launch per layer for all three branches, on the caller's stream (v21's tiles and splits:
// bitwise equal to 0); 2 = the same, with channel splits balanced by taps (K = 3 / 7 / 11 -> u / 2u / 4u: equal
// work per block, no long K = 11 tail per layer); 3 = uniform tiles / splits sized for the 3x larger grid.
// Layers that do not take the tensor-core path keep the 3-stream path. Call before the first run or capture.
int kt2n_hift_set_group(void* h, int mode);

// Core (spike-compatible): external phase [B,9,1] and noise [B,9,480T]; cache [B,1,cache_len] or null.
int kt2n_hift_run(void* h, int B, int T, int cache_len, const float* mel, const float* cache,
                  const float* phase, const float* noise, float* wave, float* source, void* stream);
// Streaming: rows (device, [B]) carry seeds, caches and the tail plan; noise is drawn natively
// (torch CUDA randn port, Philox), phases come from rows[b].phase. wave / source optional (checks).
int kt2n_hift_run_rows(void* h, int B, int T, const float* mel, const Kt2nHiftRow* rows_dev,
                       float* wave, float* source, void* stream);

// Assemble the HiFT input [B][80][T] = cat(mel cache, new frames): mel_cache[b] -> the row's [80][8] cache
// (first cache_frames[b] columns valid; may be null with cache_frames null), new frames from a frame-major
// packed buffer [total_frames][80] at frame_off[b]. All pointers device; capturable.
int kt2n_hift_pack_mel(int B, int T, const float* frames_fm, const int32_t* frame_off, const float* const* mel_cache,
                       const int32_t* cache_frames, float* out, void* stream);

// Checks / host helpers.
int kt2n_hift_noise(void* h, int B, int L, const uint64_t* seeds_host, float* out, void* stream);
void kt2n_hift_phases(uint64_t seed, float* out9);   // torch CPU mt19937 rand -> U(-pi, pi), [0] = 0
void kt2n_hift_plan(const Kt2nHiftPlanIn* in, Kt2nHiftPlanOut* out);
void kt2n_hift_tables(float* trim960, float* hamming7680);

#ifdef __cplusplus
}
#endif
