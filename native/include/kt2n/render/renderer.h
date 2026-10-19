// L7 Renderer: the whole token -> PCM renderer from one file (s3gen.kt2n), for the conductor (L8).
//   Renderer::create(s3gen.kt2n) loads the encoder (L4 layout, fp32), the estimator (L5 loader) and HiFT (L6
//   319-tensor list), and builds a RenderService whose calls run: [L4 encoder + L7 inputs + L5 estimator] as one
//   captured graph per bucket, then L6 HiFT per equal-width group, then one PCM copy.
//   add_voice(prompt tokens, prompt mel, raw x-vector) computes the speaker conditioning on the GPU with L4's
//   spk_affine (spk_embed_affine_layer(l2normalize(x))) and builds the voice prefix (shared pages) once.
// All other calls are RenderService's (include/kt2n/render/service.h) = L8's RenderBackend.
// CUDA builds only (src/render/renderer.cu).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "kt2n/render/hift_stage.h"
#include "kt2n/render/program.h"
#include "kt2n/render/service.h"

namespace kt2n::render {

struct RendererConfig {
  RenderServiceConfig service;
  FlowProgramConfig flow;
  bool hift = true;
  int32_t hift_max_b = 32;
  int32_t hift_max_t = 1024;            // S_urgent HiFT handle: frames of the widest row (commits 0-1 need <= 34)
  int32_t hift_max_frame_rows = 2048;   // S_urgent: B x T bound per launch (L6: ~750 KB of workspace per frame-row)
  // S_bulk HiFT handle (finals + whole renders): one stream:false row must fit in one launch. 4096 frames covers a
  // 2,045-token row (DEMO budget 1,000 x 2 for the noise extension + 3 + 3 silence = 2,006 tokens = 4,012 frames);
  // ~3.1 GB. Longer whole renders fail their call with an error (never a crash).
  int32_t hift_bulk_max_t = 4096;
  int32_t hift_bulk_max_frame_rows = 4096;
  // Work policy for the S_bulk handle: convs from 16 x 1024
  // outputs on the tensor cores instead of from 128 x 1024. Bulk commits (18-58 frames, B1-B2, what N=20 runs) take
  // 18-36 % less GPU when calls share the GPU and are as fast or faster alone; correctness within production's own
  // TF32 error (oracle 52-65 dB per call). The urgent handles keep L6's latency default (a first commit is 0.55 ms
  // instead of 0.78 alone). OPT-IN (L6 16:59: an engine A/B at N=10/20 decides; it changes bulk PCM within production's
  // TF32 error, so the quality gates see it too): -1 = L6's default on S_bulk (RC0 behaviour); KT2N_L7_HIFT_BULK_TC_MIN=16.
  int32_t hift_bulk_tc_min = -1;
  // S_aux HiFT handle (only with service.caller.aux_max_rows > 0 / KT2N_RENDER_AUX_ROWS): commits only (a 50-token
  // commit with its 8 cached frames is 108 frames wide); ~190 MB at 256 frame-rows.
  int32_t hift_aux_max_t = 128;
  int32_t hift_aux_max_frame_rows = 256;
  bool hift_warm = true;                // run HiFT once per shape class at create (lazy module loading off the live path)
  int32_t hift_prewarm_rows = 20;       // HiFT graphs captured at create (hift_stage.h prewarm); 0 = none
  bool build_prefixes = true;
  // With build_prefixes: add_voice also captures the voice's flow graphs ahead (RenderService::prewarm_voice)
  // for rows buckets <= prewarm_rows and rows up to prewarm_tokens tokens. 0 = off.
  int32_t prewarm_rows = 20;
  int32_t prewarm_tokens = 320;           // voice prefix at add_voice (production: built on first use, G_PROMPT_NOISE=voice)
};

class Renderer {
 public:
  static std::unique_ptr<Renderer> create(const std::string& s3gen_path, const RendererConfig& cfg, std::string* err);
  ~Renderer();
  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;

  // prompt_feat [frames][80] (frames = 2 x tokens), embedding = the raw speaker x-vector [192].
  int32_t add_voice(const std::vector<int32_t>& prompt_tokens, const std::vector<float>& prompt_feat,
                    const std::vector<float>& embedding, std::string* err);
  // RenderService::remove_voice (the renderer keeps no other per-voice state).
  int32_t remove_voice(int32_t voice, std::string* err);
  RenderService& service();
  // HiFT graph-mode counters (hift_stage.h); zeros without HiFT.
  HiftStage::GraphStats hift_graph_stats() const;
  // Totals of the add_voice prewarms (keys seen, graphs captured, host ms).
  RenderService::PrewarmStats prewarm_stats() const;

  struct Impl;

 private:
  Renderer();
  std::unique_ptr<Impl> impl_;
};

}  // namespace kt2n::render
