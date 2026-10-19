// kt2n text facade for the conductor (lane L9b, for L8): one request's text
// from entry to LM jobs, in production's order.
//
// Production (kittenml-api@ebcdd3f, live settings):
//   api.py:7020        HTTP: text = payload.input.strip(); blank -> 400
//                      (a WebSocket phrase reaches the engine as cut, unstripped)
//   runtime.py:20505   decide_request_emotion(text)    -> RequestPlan()  step 1
//   runtime.py:20509   tagged_demo_chunking / q_plain_dc_mode            step 1
//   runtime.py:20546   plan_normalization(): prepare() (L9a) + plan      step 2-3
//   runtime.py:20581   segment 0: normalize_planned_segment (L9a grammar
//                      + re-split)                       -> add_normalized(0)
//   runtime.py:20643   q_demo_chunks = len(raw_segments) - 1 + len(seg-0 chunks)
//   runtime.py:20648   _submit_cuda_segments(seg-0 chunks): prompt + budget
//   runtime.py:20246   segments 1.. on the request's normaliser thread,
//                      _extend_cuda_segments (returns before the extend when
//                      a segment yields no chunks, :19797)
//                                                    -> add_normalized(1, 2, ...)
//
// What the facade does NOT do (and who does):
//   * prepare() and the grammar (normalize_planned_segment) are L9a's
//     (kt2n::text, OpenFst).  The caller runs them and hands the results in,
//     so this library keeps no third-party dependency.  Pass the grammar the
//     PLANNED segments (segments() as UTF-8) and the L9a PreparedText it came
//     from; L9a's N_SPLIT path is the caller's choice (latency only: the text
//     was identical on every corpus text).
//   * When segment k+1 may be normalised (first-audio wait, look-ahead 1) is
//     L9a IncrementalPolicy + L8's thread.  The facade only checks ORDER.
//   * The voice prefix is L11's; suffix ids here are what follows it.
//
// Not thread safe: one request's plan is used by one thread at a time (L8
// already serialises segment 0 on the request's text task and the rest on its
// normaliser thread).
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "kt2n/textplan.h"

namespace kt2n::textplan {

// L9a's prepare() result (kt2n::text::PreparedText), as code points.
struct Prepared {
  U32 text;
  bool wfst = false;   // the grammar applies (abbreviation merge on)
  Emphasis emphasis;   // the request's (((emphasis))) spans
};
// The passthrough prepare (no L9a compiled in; plumbing tests only, never
// the RC): text unchanged, no grammar, the request's own emphasis spans.
Prepared passthrough_prepare(std::u32string_view request_text);

// stream:true (speech mode none) or stream:false (speech mode DEMO).
enum class BudgetRule { kStream, kDemo };

struct LmChunk {
  U32 text;                      // what the LM is asked to say
  std::vector<int32_t> suffix;   // ids after the voice prefix (empty when no tokenizer)
  int budget = 0;                // max new LM tokens (runtime.py:19395-19415)
  int ordinal = 0;               // request-wide job index (lane.jobs index; seed = salt + ordinal)
  int segment = 0;               // planned segment it came from
};

class RequestPlan {
 public:
  // Steps 1-3.  ``request_text``: as runtime.generate_stream receives it
  // (HTTP: already input.strip()-ed and non-blank; a WS phrase: as cut).
  // ``prepared``: L9a prepare(request_text, language), or passthrough_prepare.
  // ``emotion_preset``: a WS phrase's turn stamp (EmotionTurn::stamp), which
  // wins over the phrase's own text.
  RequestPlan(std::u32string_view request_text, Prepared prepared, const Abbreviations& abbreviations,
              BudgetRule rule, std::optional<bool> emotion_preset = std::nullopt,
              const Settings& settings = {});

  const RequestText& decision() const { return decision_; }
  bool emotion() const { return decision_.emotion; }  // {emo: 1} on every prompt
  int dc_mode() const { return decision_.dc_mode; }   // control.q_dc_mode (1 plain, 0 tagged)
  const Prepared& prepared() const { return prepared_; }
  // The planned (pre-grammar) segments; the grammar's input.
  const std::vector<U32>& segments() const { return segments_; }
  // Billing weights: code points of each planned segment (runtime.py:9494).

  // Steps 4-5 for planned segment next_segment(): its grammar output (L9a
  // SegmentResult::text) -> demo re-split -> LM chunks with prompt suffix and
  // budget.  Throws std::logic_error when called out of order or after the
  // last segment.  ``tokenizer`` nullptr leaves suffix empty (stub LM).
  std::vector<LmChunk> add_normalized(std::size_t index, std::u32string_view normalized,
                                      const Tokenizer* tokenizer);
  std::size_t next_segment() const { return next_; }
  bool sealed() const { return next_ >= segments_.size(); }  // every segment added
  // control.q_demo_chunks (known once segment 0 is added; -1 before).
  int total_chunks() const { return total_chunks_; }
  int chunks_created() const { return chunks_; }

 private:
  Settings settings_;
  BudgetRule rule_;
  RequestText decision_;
  Prepared prepared_;
  std::vector<U32> segments_;
  std::size_t next_ = 0;
  int total_chunks_ = -1;
  int chunks_ = 0;
};

}  // namespace kt2n::textplan
