// L7 render program: the lanes' op libraries composed into one RenderStage per call
// (captured into the call's graph by RenderCaller):
//   1. L4 encoder (enc::launch_ops over the CallDesc)                      -> mu [frames][80]
//   2. L7 estimator inputs: x0 = the rows' numpy noise (kStateNoiseGather), cond = the voice's prompt
//      mel on prompt frames (0 after), spks = the voice's speaker conditioning      [frames][80] / [rows][80]
//   3. L5 estimator (est::Estimator::run, 2 mean-flow steps, 1 for a preview)   -> CallDesc::mel_out
// HiFT (L6) is not in the program yet: its C ABI renders equal-width row groups with its own streams.
// CUDA builds only (src/render/program.cu).
#pragma once

#include <memory>

#include "kt2n/render/call.h"
#include "kt2n/render/encoder.h"
#include "kt2n/render/estimator.h"

namespace kt2n::render {

struct FlowProgramConfig {
  bool pdl = true;
  MathMode enc_conv_math = MathMode::kTf32;  // production's convs run under cuDNN's TF32 default
};

class FlowProgram {
 public:
  // `enc_w` and `est` must outlive the program (device weights).
  FlowProgram(const enc::Weights& enc_w, const est::Estimator& est, const FlowProgramConfig& cfg);
  ~FlowProgram();
  RenderStage stage();
  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace kt2n::render
