// kt2n::lm device kernels: stop rules + token ring (post), admission init, bucket row map.  Lane L3, 10 Oct.
//
// Stop rules are production's _CudaDecodeEngine._segment_finished (runtime.py @ ebcdd3f) for the live settings
// (L_VERIFY=0, no lab force ids, no Q lanes), checked in production's order after the token is counted:
//   SPEECH_END -> eos; STOP -> stop; DEMO and new >= budget -> budget; cap and new >= cap -> max_new_tokens;
//   new >= ceiling x budget -> ceiling; new - silent >= budget -> budget;
//   demo-chunked and prompt + new >= ctx_limit - margin -> ceiling;
//   plus the slot guard: the next step could not write its K/V (position >= max_ctx) -> ceiling.
// silent counts audio ids in the sub-gate set (SUB_GATE_AUDIO_IDS) when armed, before the checks (as production).
#include "lm_device.cuh"

namespace kt2n::lm::dev {
namespace {

constexpr int32_t kSpeechEnd = 151670;
constexpr int32_t kStop = 151674;
constexpr int32_t kAudioBase = 151675;
constexpr int32_t kAudioVocab = 6561;
constexpr int32_t kSpeechStart = 151669;

__global__ void __launch_bounds__(kRows) post_kernel(State st, int rows, const int32_t* __restrict__ row_slot,
                                                     const int32_t* __restrict__ samp_tok, int update_rows) {
  __shared__ uint32_t s_seq;
  if (threadIdx.x == 0) s_seq = *st.seq;
  __syncthreads();
  RingEntry* e = st.ring + (s_seq % static_cast<uint32_t>(st.ring_n));
  const int r = threadIdx.x;
  if (r < rows) {
    const int s = row_slot[r];
    const uint32_t f = st.flags[s];
    int32_t token = -1, index = 0, fin = kFinNone;
    if (!(f & kDone)) {
      const int32_t n = st.new_tokens[s] + 1;
      token = samp_tok[r];
      if (st.forced_len[s] >= n) {
        token = st.forced[static_cast<int64_t>(s) * st.forced_cap + (n - 1)];
        st.history[s * kWindow + (kWindow - 1)] = token;  // the sampler appended its own draw; history = forced id
      }
      int32_t sil = st.silent[s];
      const bool audio = token >= kAudioBase && token < kAudioBase + kAudioVocab;
      if (audio && st.silence_armed) {
        const int a = token - kAudioBase;
        if ((st.silent_bits[a >> 5] >> (a & 31)) & 1u) ++sil;
      }
      const int32_t budget = st.budget[s];
      const int32_t cap = st.cap[s];
      const int64_t pos_next = st.pos[s] + 1;  // logical position of this token
      if (token == kSpeechEnd) {
        fin = kFinEos;
      } else if (token == kStop) {
        fin = kFinStop;
      } else if ((f & kDemo) && n >= budget) {
        fin = kFinBudget;
      } else if (cap > 0 && n >= cap) {
        fin = kFinMaxNew;
      } else if (static_cast<int64_t>(n) >= static_cast<int64_t>(budget) * st.ceiling) {
        fin = kFinCeiling;
      } else if (n - sil >= budget) {
        fin = kFinBudget;
      } else if ((f & kChunked) && st.prompt_len[s] + n >= st.ctx_limit - st.margin) {
        fin = kFinCeiling;
      } else if (pos_next >= st.max_ctx) {
        fin = kFinCeiling;  // slot full (production would raise in the slot buffer)
      }
      st.new_tokens[s] = n;
      st.silent[s] = sil;
      index = n;
      if (fin != kFinNone) {
        st.flags[s] = f | kDone;
      } else {
        st.tok[s] = token;
        st.pos[s] = pos_next;
        if (update_rows) {
          st.row_tok[r] = token;
          st.row_pos[r] = pos_next;
        }
      }
    }
    e->token[r] = token;
    e->index[r] = index;
    e->finish[r] = fin;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    e->rows = rows;
    __threadfence_system();
    *reinterpret_cast<volatile uint32_t*>(&e->seq) = s_seq + 1u;
    *st.seq = s_seq + 1u;
  }
}

__global__ void adm_init_kernel(State st, AdmInit a) {
  const int i = blockIdx.x;
  const int s = a.slot[i];
  const int t = threadIdx.x;
  if (t < kWindow) st.history[s * kWindow + t] = 0;
  if (t < kParamCols) st.params[s * kParamCols + t] = a.params[i][t];
  if (t == 0) {
    st.tok[s] = kSpeechStart;  // overwritten by the first sampled token
    st.pos[s] = a.prompt_len[i] - 1;
    st.new_tokens[s] = 0;
    st.silent[s] = 0;
    st.budget[s] = a.budget[i];
    st.cap[s] = a.cap[i];
    st.prompt_len[s] = a.prompt_len[i];
    st.flags[s] = a.flags[i];
    st.forced_len[s] = a.forced_len[i];
    st.history_len[s] = 0;
    st.seeds[s] = a.seed[i];
    st.offsets[s] = 0;
    st.adm_slot[i] = s;
    st.voff[s] = a.voff[i];
    st.vlen[s] = a.vlen[i];
  }
}

__global__ void __launch_bounds__(kRows) rowmap_kernel(State st, RowMap m) {
  const int r = threadIdx.x;
  if (r < m.n) {
    const int s = m.slot[r];
    st.row_slot[r] = s;
    st.row_tok[r] = st.tok[s];
    st.row_pos[r] = st.pos[s];
    st.row_voff[r] = st.voff[s];
    st.row_vlen[r] = st.vlen[s];
  } else if (r < m.b) {
    st.row_slot[r] = m.scratch;
    st.row_tok[r] = kSpeechStart;
    st.row_pos[r] = 0;
    st.row_voff[r] = -1;
    st.row_vlen[r] = 0;
  }
}

}  // namespace

cudaError_t launch_post(const State& st, int rows, const int32_t* row_slot, const int32_t* samp_tok, int update_rows,
                        cudaStream_t stream) {
  post_kernel<<<1, kRows, 0, stream>>>(st, rows, row_slot, samp_tok, update_rows);
  return cudaGetLastError();
}

cudaError_t launch_adm_init(const State& st, const AdmInit& a, cudaStream_t stream) {
  if (a.n < 1 || a.n > kAdmRows) return cudaErrorInvalidValue;
  adm_init_kernel<<<a.n, 64, 0, stream>>>(st, a);
  return cudaGetLastError();
}

cudaError_t launch_rowmap(const State& st, const RowMap& m, cudaStream_t stream) {
  if (m.n < 0 || m.b < m.n || m.b > kRows) return cudaErrorInvalidValue;
  rowmap_kernel<<<1, kRows, 0, stream>>>(st, m);
  return cudaGetLastError();
}

}  // namespace kt2n::lm::dev
