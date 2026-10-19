// G0 header compile test: include/kt2n/engine.h (C ABI) and
// include/kt2n/render/ops.h (renderer op-library contract).
//
// Host:  target kt2n_header_tests (ctest -L cpu). It also links tests/cpp/test_engine_h.c, compiled
//        as C11, and checks that C and C++ agree on every engine.h layout.
// nvcc:  nvcc -std=c++20 -x cu -arch=sm_120 -Iinclude tests/cpp/test_headers.cc tests/cpp/test_engine_h.c
//        builds the device half below too (run_tile's signature, the KT2N_HD pool offsets on the
//        device, a Phase A/B tile kernel) with a plain main instead of gtest.
//        Host without gtest: g++ -std=c++20 -DKT2N_HEADERS_NO_GTEST (the same plain main).
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "kt2n/engine.h"
#include "kt2n/render/ops.h"

namespace r = kt2n::render;

// ---------------------------------------------------------------------------
// LP64 layouts. Existing fields are frozen; optional trailing fields use struct_size.
// ---------------------------------------------------------------------------
static_assert(sizeof(void*) == 8, "LP64 only");
static_assert(KT2N_ABI_VERSION == 2u);
static_assert(sizeof(kt2n_event) == 120);
static_assert(sizeof(kt2n_request_params) == 144);  // 136 + control_json (L8b, appended 10 Oct, struct_size-versioned)
static_assert(offsetof(kt2n_request_params, control_json) == 136);
static_assert(sizeof(kt2n_engine_config) == 160);
static_assert(offsetof(kt2n_engine_config, voice_build_dir) == 144);
static_assert(offsetof(kt2n_engine_config, max_custom_voices) == 152);
static_assert(sizeof(kt2n_error) == 504);
static_assert(sizeof(kt2n_stats) == 136);
static_assert(sizeof(kt2n_sampler) == 48);
static_assert(offsetof(kt2n_engine_config, struct_size) == 0);
static_assert(offsetof(kt2n_request_params, struct_size) == 0);
static_assert(offsetof(kt2n_session_params, struct_size) == 0);
static_assert(offsetof(kt2n_voice_build_params, struct_size) == 0);
static_assert(offsetof(kt2n_voice_info, struct_size) == 0);
static_assert(offsetof(kt2n_stats, struct_size) == 0);
static_assert(std::is_trivially_copyable_v<kt2n_event>);
static_assert(std::is_standard_layout_v<kt2n_event>);

// Function signatures (decltype only: nothing is linked).
static_assert(std::is_same_v<decltype(&kt2n_engine_create),
                             int32_t (*)(const kt2n_engine_config*, kt2n_engine**, kt2n_error*)>);
static_assert(std::is_same_v<decltype(&kt2n_submit),
                             int32_t (*)(kt2n_engine*, const kt2n_request_params*, kt2n_request_id*, kt2n_error*)>);
static_assert(std::is_same_v<decltype(&kt2n_poll), int32_t (*)(kt2n_engine*, kt2n_event*, int32_t, int32_t, int32_t*)>);
static_assert(std::is_same_v<decltype(&kt2n_event_fd), int (*)(kt2n_engine*)>);
static_assert(std::is_same_v<decltype(&kt2n_session_text_append),
                             int32_t (*)(kt2n_engine*, kt2n_session_id, const char*, size_t, kt2n_error*)>);
static_assert(std::is_same_v<decltype(&kt2n_voice_build),
                             int32_t (*)(kt2n_engine*, const kt2n_voice_build_params*, kt2n_job_id*, kt2n_error*)>);
static_assert(std::is_same_v<decltype(&kt2n_cancel), int32_t (*)(kt2n_engine*, uint64_t, int32_t)>);
static_assert(std::is_same_v<decltype(&kt2n_error_lookup), const kt2n_error_info* (*)(int32_t)>);

// Public error codes keep their numbers (adapters map them by value).
static_assert(KT2N_ERR_INVALID_REQUEST == 100 && KT2N_ERR_VOICE_NOT_FOUND == 110);
static_assert(KT2N_ERR_INVALID_AUDIO == 200 && KT2N_ERR_CUSTOM_VOICE_SERVICE_UNAVAILABLE == 214);
static_assert(KT2N_ERR_INVALID_REFERENCE_TRANSCRIPT == 215 && KT2N_ERR_INVALID_CONSENT_TRANSCRIPT == 216);
static_assert(KT2N_ERR_SPEECH_GENERATION_FAILED == 300 && KT2N_ERR_INTERNAL == 303);

// ---------------------------------------------------------------------------
// ops.h: layouts and constants
// ---------------------------------------------------------------------------
static_assert(sizeof(r::RowDesc) == 160 && alignof(r::RowDesc) == 16);
static_assert(sizeof(r::OpDesc) == 240 && alignof(r::OpDesc) == 16);
static_assert(std::is_trivially_copyable_v<r::RowDesc> && std::is_trivially_copyable_v<r::CallDesc> &&
              std::is_trivially_copyable_v<r::OpDesc>);
static_assert(std::is_standard_layout_v<r::RowDesc> && std::is_standard_layout_v<r::CallDesc>);
static_assert(std::is_trivially_copyable_v<r::VoiceView> && std::is_trivially_copyable_v<r::StateView>);
// Rows production serves above its chunk cache's 1088 (eager path) fit: kMaxRowTokens is the LM
// slot ceiling (CUDA_GRAPH_MAX_CTX), a whole number of pages and of 64-token whole-render buckets.
static_assert(r::kMaxRowTokens == 4096 && r::kMaxRowTokens % r::kPageTokens == 0 &&
              r::whole_bucket(r::kMaxRowTokens) == r::kMaxRowTokens);
static_assert(250 + 2 * 750 + 3 <= r::kMaxRowTokens && 250 + 1000 + 3 <= r::kMaxRowTokens);
static_assert(r::kPageTokens == 32 && r::kPageBytes == 786432u + 1048576u + 14680064u);
static_assert(r::kEstAttnLayers == (1 + r::kEstMidBlocks + 1) * r::kEstTfPerBlock);
static_assert(r::kSamplesPerFrame == 8 * 5 * 3 * r::kIstftHop);
static_assert(r::kSourceCacheSamples == r::kMelCacheFrames * r::kSamplesPerFrame);
static_assert(r::kMelChunk == r::kTokenChunk * r::kMelPerToken);
static_assert(r::kEstIn == 4 * r::kMel);
static_assert(r::kTileSmemBytes == 81920u);

// Buckets and helpers.
static_assert(r::bucket_of(r::kSteadySuffixBuckets, 1) == 16);
static_assert(r::bucket_of(r::kSteadySuffixBuckets, 53) == 64);
static_assert(r::bucket_of(r::kSteadySuffixBuckets, 81) == -1);
static_assert(r::bucket_of(r::kFirstSuffixBuckets, 28) == 32);
static_assert(r::bucket_of(r::kRowBuckets, 10) == 12);
static_assert(r::bucket_of(r::kRowBuckets, 64) == 64);
static_assert(r::whole_bucket(1) == 64 && r::whole_bucket(64) == 64 && r::whole_bucket(65) == 128);

// RNG formulas (incremental_s3gen.py:90 / :3561, runtime.py:15444).
static_assert(r::hift_seed(5, 25) == 3050025ull);
static_assert(r::hift_seed(1024ull * 0x3FFFFFFFull, 0) == 3000000ull + 1024ull * 0x3FFFFFFFull * 10000ull);
static_assert(r::target_noise_width(400) == 806);
static_assert(r::kFlowNoiseSeedBase == 4000000ull && r::kFlowNoiseExtSeedBase == 5000000ull);

// Stream roles.
static_assert(r::stream_for(r::Phase::kFirstCommit, 0) == r::StreamRole::kSUrgent);
static_assert(r::stream_for(r::Phase::kSteadyCommit, 1) == r::StreamRole::kSUrgent);
static_assert(r::stream_for(r::Phase::kSteadyCommit, 2) == r::StreamRole::kSBulk);
static_assert(r::stream_for(r::Phase::kPreview, 0) == r::StreamRole::kSUrgent);
static_assert(r::stream_for(r::Phase::kFinalFlush, 0) == r::StreamRole::kSBulk);
static_assert(r::stream_for(r::Phase::kWholeRender, 0) == r::StreamRole::kSBulk);

// Pool offsets: the last element of each domain is the domain size - 1.
static_assert(std::is_same_v<decltype(r::enc1_offset(r::PoolView{}, 0, 0, 0, 0)), size_t>);

#if defined(__CUDACC__)
// ---------------------------------------------------------------------------
// Device half (nvcc): the tile signature, KT2N_HD helpers on the device, and the
// Phase A/B kernel shape. A stand-in tile function with run_tile's exact type.
// ---------------------------------------------------------------------------
static_assert(std::is_same_v<decltype(r::run_tile), void(const r::OpDesc&, const r::CallDesc&, int32_t, r::TileCtx&)>);

__device__ void test_tile(const r::OpDesc& op, const r::CallDesc& call, int32_t tile, r::TileCtx& ctx) {
  if (tile >= op.n_tiles || call.n_rows <= 0) return;
  const r::RowDesc& row = call.rows[0];
  const int32_t page = row.n_pages > 0 ? call.page_table[row.page_off] : 0;
  const size_t off = r::est_offset(call.pool, 0, op.layer < 0 ? 0 : op.layer, page, 0);
  if (threadIdx.x == 0 && ctx.smem_bytes == 0xFFFFFFFFu) call.pcm_f32[0] = float(off);
}
static_assert(std::is_same_v<decltype(test_tile), decltype(r::run_tile)>);

__global__ void __launch_bounds__(r::kTileThreads) test_op_kernel(const r::OpDesc* op, const r::CallDesc* call) {
  extern __shared__ unsigned char smem[];
  r::TileCtx ctx{smem, op->smem_bytes, static_cast<int32_t>(blockIdx.x), 0};
  test_tile(*op, *call, static_cast<int32_t>(blockIdx.x), ctx);
}
#endif

// ---------------------------------------------------------------------------
// Runtime checks (gtest on the host build; a plain main for the quick nvcc check)
// ---------------------------------------------------------------------------
extern "C" size_t kt2n_c_layout_probe(int which);

namespace {

bool pool_offsets_ok() {
  r::PoolView p{};
  p.n_pages = 3;
  const size_t enc1_total = size_t(r::kEnc1Layers) * 2 * 3 * r::kPageTokens * r::kEncDim;
  const size_t enc2_total = size_t(r::kEnc2Layers) * 2 * 3 * r::kPageFrames * r::kEncDim;
  const size_t est_total = size_t(r::kFlowSteps) * r::kEstAttnLayers * 3 * r::kPageFrames * 2 * r::kEstInner;
  return r::enc1_offset(p, r::kEnc1Layers - 1, 1, 2, r::kPageTokens - 1) + r::kEncDim == enc1_total &&
         r::enc2_offset(p, r::kEnc2Layers - 1, 1, 2, r::kPageFrames - 1) + r::kEncDim == enc2_total &&
         r::est_offset(p, r::kFlowSteps - 1, r::kEstAttnLayers - 1, 2, r::kPageFrames - 1) + 2 * r::kEstInner ==
             est_total &&
         enc1_total * sizeof(float) == 3 * r::kEnc1PageBytes && enc2_total * sizeof(float) == 3 * r::kEnc2PageBytes &&
         est_total * 2 == 3 * r::kEstPageBytes;
}

bool c_layout_matches() {
  const size_t want[] = {sizeof(kt2n_event),
                         sizeof(kt2n_request_params),
                         sizeof(kt2n_engine_config),
                         sizeof(kt2n_error),
                         sizeof(kt2n_stats),
                         sizeof(kt2n_session_params),
                         sizeof(kt2n_voice_build_params),
                         sizeof(kt2n_voice_info),
                         sizeof(kt2n_sampler),
                         offsetof(kt2n_event, data),
                         offsetof(kt2n_event, name),
                         offsetof(kt2n_event, voice),
                         offsetof(kt2n_request_params, user_data),
                         offsetof(kt2n_request_params, control_json)};
  for (int i = 0; i < static_cast<int>(sizeof(want) / sizeof(want[0])); ++i)
    if (kt2n_c_layout_probe(i) != want[i]) return false;
  return true;
}

// Timeline names are part of the public event format.
bool timeline_names_ok() {
  const char* names[] = {KT2N_TL_SCHEDULER_ADMITTED,      KT2N_TL_LM_PREFILL_FINISHED,
                         KT2N_TL_N_OPENING_SKIP,          KT2N_TL_S3GEN_STARTED,
                         KT2N_TL_S3GEN_PUBLICATION,       KT2N_TL_S3GEN_FINAL_PUBLICATION,
                         KT2N_TL_T9_DEFERRAL_ENDED,       KT2N_TL_LM_SEGMENT_FINISHED,
                         KT2N_TL_LM_TOKENS_FINALIZED,     KT2N_TL_NATIVE_PREFILL,
                         KT2N_TL_LM_DECODE_STEP,          KT2N_TL_STARTUP_HOLD_RELEASED,
                         KT2N_TL_API_WEBSOCKET_RESERVE_RELEASED, KT2N_TL_API_WEBSOCKET_PUBLICATION,
                         KT2N_TL_API_SSE_PUBLICATION,     KT2N_TL_API_ENGINE_PUBLICATION_READY};
  const char* want[] = {"gpu.scheduler_admitted",      "gpu.lm_prefill_finished",
                        "gpu.n_opening_skip",          "gpu.s3gen_started",
                        "gpu.s3gen_publication",       "gpu.s3gen_final_publication",
                        "gpu.t9_acoustic_deferral_ended", "gpu.lm_segment_finished",
                        "gpu.lm_tokens_finalized",     "gpu.kt2n_native_prefill",
                        "gpu.lm_decode_step",          "api.startup_hold_released",
                        "api.websocket_reserve_released", "api.websocket_publication",
                        "api.sse_publication",         "api.engine_publication_ready"};
  for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i)
    if (std::strcmp(names[i], want[i]) != 0) return false;
  return true;
}

}  // namespace

#if defined(KT2N_HEADERS_NO_GTEST) || defined(__CUDACC__)
int main() {
  const bool ok = pool_offsets_ok() && c_layout_matches() && timeline_names_ok();
  return ok ? 0 : 1;
}
#else
#include <gtest/gtest.h>

TEST(Headers, PoolOffsetsCoverEachDomainExactly) { EXPECT_TRUE(pool_offsets_ok()); }
TEST(Headers, CAndCxxAgreeOnEngineLayouts) { EXPECT_TRUE(c_layout_matches()); }
TEST(Headers, TimelineNamesAreProductions) { EXPECT_TRUE(timeline_names_ok()); }
#endif
