// L11: clone build service (see voice_service.h).
#include "voice_service.h"

#include <chrono>
#include <cstdio>
#include <cstring>

#include "consent_phrase.h"
#include "decode.h"

namespace kt2n::voice {
namespace {

// Python str.isspace() code points (what str.strip() removes).
bool py_space(uint32_t c) {
  return (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}

// UTF-8 -> code points (false on malformed input).
bool decode_utf8(const std::string& s, std::vector<uint32_t>& cps, std::vector<size_t>& offs) {
  size_t i = 0;
  while (i < s.size()) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    uint32_t cp;
    size_t n;
    if (c < 0x80) {
      cp = c;
      n = 1;
    } else if ((c >> 5) == 6) {
      cp = c & 0x1f;
      n = 2;
    } else if ((c >> 4) == 14) {
      cp = c & 0x0f;
      n = 3;
    } else if ((c >> 3) == 30) {
      cp = c & 0x07;
      n = 4;
    } else {
      return false;
    }
    if (i + n > s.size()) return false;
    for (size_t k = 1; k < n; ++k) {
      unsigned char cc = static_cast<unsigned char>(s[i + k]);
      if ((cc >> 6) != 2) return false;
      cp = (cp << 6) | (cc & 0x3f);
    }
    cps.push_back(cp);
    offs.push_back(i);
    i += n;
  }
  offs.push_back(s.size());
  return true;
}

struct BuildFailed {
  VoiceBuildOutcome outcome;
};

const char* code_for(int32_t status) {
  switch (status) {
    case 5: return "cancelled";
    case 200: return "invalid_audio";
    case 201: return "invalid_audio_duration";
    case 202: return "invalid_audio_signal";
    case 203: return "invalid_reference_audio";
    case 204: return "invalid_reference_audio";
    case 206: return "speaker_verification_failed";
    case 207: return "consent_phrase_mismatch";
    case 208: return "unsupported_consent_language";
    case 212: return "custom_voice_creation_busy";
    case 213: return "custom_voice_service_error";
    case 215: return "invalid_reference_transcript";
    case 216: return "invalid_consent_transcript";
    default: return "";
  }
}

// production's error.message for each status (runtime.py ebcdd3f; line numbers in the lane log)
std::string message_for(int32_t status, uint32_t flags) {
  switch (status) {
    case 200: return "The uploaded file could not be decoded as supported audio.";  // :11128
    case 202:
      return "Reference audio must contain clear speech: it is silent, too quiet or distorted. "
             "Send 2 to 30 seconds of one person speaking at a normal level.";  // :11162
    case 203: return "reference prefix length is out of bounds";                   // custom_voices.py :323
    case 204: return "The reference audio could not be aligned for voice cloning."; // :8400
    case 206:
      return (flags & kVbInline) ? "The reference audio does not match the consent speaker."  // :12761
                                 : "The audio sample does not match the consent speaker.";    // :12536
    case 212: return "Custom voice creation is temporarily busy.";                  // :9603
    case 213: return "The speaker encoder returned an invalid result.";             // :11345
    default: return "";
  }
}

VoiceBuildOutcome fail_with(int32_t status, const std::string& param, std::string message = "") {
  VoiceBuildOutcome o;
  o.status = status;
  o.code = code_for(status);
  o.param = param;
  o.message = std::move(message);
  return o;
}

// :11143 (the decode stops just past 30 s, so a long clip's own length is not reported)
std::string duration_message(size_t samples) {
  const double d = double(samples) / 24000.0;
  char buf[160];
  if (d > 30.0)
    std::snprintf(buf, sizeof buf,
                  "Reference audio must be 2 to 30 seconds long; this clip is longer than 30 seconds. Trim it to a "
                  "shorter clip.");
  else
    std::snprintf(buf, sizeof buf, "Reference audio must be 2 to 30 seconds long; this clip is %.1f seconds. Send a "
                  "longer clip.", d);
  return buf;
}

}  // namespace

VoiceService::VoiceService(VoiceBuilder& builder, Tokenize tokenize, int slots, int depth, double wait_s,
                           int inline_lru)
    : builder_(builder), tokenize_(std::move(tokenize)), admission_(slots, depth), wait_s_(wait_s), lru_(inline_lru) {}

std::string VoiceService::validate_transcript(const std::string& raw, std::string* stripped, const char* field) {
  const std::string f = field ? field : "reference_transcript";
  const std::string bad_len = f + " is required and must not exceed 4096 characters or 16 KiB.";
  std::vector<uint32_t> cps;
  std::vector<size_t> offs;
  if (!decode_utf8(raw, cps, offs)) return bad_len;
  size_t b = 0, e = cps.size();
  while (b < e && py_space(cps[b])) ++b;
  while (e > b && py_space(cps[e - 1])) --e;
  std::string t = raw.substr(offs[b], offs[e] - offs[b]);
  if (stripped) *stripped = t;
  if (e == b || e - b > 4096 || t.size() > 16 * 1024) return bad_len;
  for (size_t i = b; i < e; ++i)
    if (cps[i] < 32 && cps[i] != '\n' && cps[i] != '\r' && cps[i] != '\t')
      return f + " contains invalid control characters.";
  return "";
}

VoiceBuildOutcome VoiceService::build(const VoiceBuildRequest& req) {
  // Only a pure inline build uses the LRU. A saved build always runs (its outcome carries the export blob, which a
  // cache hit would not have), and a consent recording is never a voice to reuse. No documented path combines
  // INLINE with SAVED or CONSENT; if a caller does, the saved / consent order and outputs apply.
  if (!(req.flags & kVbInline) || (req.flags & (kVbConsent | kVbSaved))) return build_uncached(req, nullptr, "");
  // Inline (resolve_inline_voice :12664): transcript rules before the cache and the slot.
  std::string transcript;
  if (std::string m = validate_transcript(req.reference_transcript, &transcript); !m.empty())
    return fail_with(215, "reference_transcript", m);
  const std::vector<int32_t> ids = tokenize_(transcript);
  // No identity key, no cache. Production's inline key names the owner, the consent, the upload's and the
  // transcript's sha256, the checkpoint and the S3Gen revision (resolve_inline_voice :12709); "" names none of
  // them. Caching under "" served one upload's voice to a different upload (verifier probe, 10 Oct sprint 3), so a
  // keyless inline build neither reads nor fills the LRU: it builds every time (L8's engine also keeps keyless
  // voices out of its own InlineVoiceIndex).
  if (req.identity_key.empty()) return build_uncached(req, &ids, transcript);
  bool loaded = false;
  VoiceBuildOutcome inner;
  try {
    auto lease = lru_.acquire(req.identity_key, [&]() -> std::shared_ptr<const CloneResult> {
      inner = build_uncached(req, &ids, transcript);
      loaded = true;
      if (inner.status != 0) throw BuildFailed{inner};  // failures are never cached
      return inner.result;
    });
    VoiceBuildOutcome out = loaded ? inner : VoiceBuildOutcome{};
    out.result = *lease;
    out.cache_hit = !loaded;
    if (!loaded) fill_projection(out);  // a hit carries the same projected row / nearest id as the build did
    return out;
  } catch (const BuildFailed& f) {
    return f.outcome;
  } catch (const CacheEntryInvalidated&) {
    return fail_with(110, "voice");  // voice_not_found
  }
}

void VoiceService::fill_projection(VoiceBuildOutcome& out) const {
  if (!catalogue_ || !out.result) return;
  const int k = out.result->nearest_index;
  if (k < 0 || (size_t)k >= catalogue_->size()) return;
  out.projected_speaker = catalogue_->projected.data() + (size_t)k * 2048;
  out.nearest_voice_id = catalogue_->ids[(size_t)k];
}

void VoiceService::set_catalogue(const Catalogue* catalogue) {
  catalogue_ = catalogue;
  if (catalogue) builder_.set_bank(catalogue->raw, catalogue->ids);
}

VoiceBuildOutcome VoiceService::build_uncached(const VoiceBuildRequest& req, const std::vector<int32_t>* text_ids_in,
                                               std::string transcript) {
  VoiceBuildOutcome out;
  const bool consent_create = (req.flags & kVbConsent) != 0;
  const char* field = consent_create ? "recording" : (req.flags & kVbInline) ? "reference_audio" : "audio_sample";
  const char* ref_param = (req.flags & kVbInline) ? "voice.reference_audio" : "audio_sample";
  double waited = 0.0;
  AdmitResult a = admission_.acquire(wait_s_, req.deadline, req.cancelled, false, &waited);
  if (a == AdmitResult::kCancelled) return fail_with(5, "");
  if (a != AdmitResult::kOk) return fail_with(212, "", message_for(212, req.flags));
  struct Release {
    CloneAdmission* a;
    ~Release() { a->release(); }
  } rel{&admission_};
  // Saved create / consent create: the text rules run inside the slot (the management slot wraps the method).
  std::vector<int32_t> own_ids;
  if (consent_create) {
    if (std::string m = validate_transcript(req.reference_transcript, &transcript, "consent_transcript"); !m.empty())
      return fail_with(216, "consent_transcript", m);
    const ConsentTextVerdict cv = check_consent_text(transcript, req.language);
    if (cv.status) return fail_with(cv.status, cv.param, cv.message);
  } else if (!text_ids_in) {
    if (std::string m = validate_transcript(req.reference_transcript, &transcript); !m.empty())
      return fail_with(215, "reference_transcript", m);
    own_ids = tokenize_(transcript);
  }
  const std::vector<int32_t>& text_ids = text_ids_in ? *text_ids_in : own_ids;
  // P5 (saved create with consent): the stored consent's own text is re-checked inside the slot, before the
  // decode (runtime.py :12507). P3 (inline) re-checks it after the decode and its checks (:12747), below.
  const bool recheck = (req.flags & kVbVerifySpeaker) && !req.consent_transcript.empty();
  if (recheck && !(req.flags & kVbInline)) {
    const ConsentTextVerdict cv = check_consent_text(req.consent_transcript, req.consent_language);
    if (cv.status) return fail_with(cv.status, cv.param, cv.message);
  }
  out.admission_wait_ms = waited * 1000.0;
  auto t0 = std::chrono::steady_clock::now();
  std::vector<float> decoded;
  if (req.flags & kVbDecodedPcm) {
    if (req.pcm_rate != 24000 || !req.pcm || req.pcm_samples == 0)
      return fail_with(200, field, "The uploaded file contains no decodable audio.");
    decoded.assign(req.pcm, req.pcm + req.pcm_samples);
  } else {
    std::string err;
    if (!decode_audio(req.audio, req.audio_len, 30.1, decoded, &err)) return fail_with(200, field, message_for(200, 0));
    if (decoded.empty()) return fail_with(200, field, "The uploaded file contains no decodable audio.");  // :11134
  }
  if (recheck && (req.flags & kVbInline)) {
    const AudioVerdict v = check_decoded(decoded.data(), decoded.size());
    if (v != AudioVerdict::kOk) {
      const int32_t st = 200 + static_cast<int>(v) - 1;
      return fail_with(st, field, st == 201 ? duration_message(decoded.size())
                                  : st == 200 ? "The uploaded audio contains non-finite samples."
                                              : message_for(st, req.flags));
    }
    const ConsentTextVerdict cv = check_consent_text(req.consent_transcript, req.consent_language);
    if (cv.status) return fail_with(cv.status, cv.param, cv.message);
  }
  auto t1 = std::chrono::steady_clock::now();
  out.decode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  auto r = std::make_shared<CloneResult>();
  {
    std::lock_guard<std::mutex> g(gpu_mu_);
    if (req.gpu_turn && !req.gpu_turn()) return fail_with(5, "");
    auto tg = std::chrono::steady_clock::now();
    out.gpu_wait_ms = std::chrono::duration<double, std::milli>(tg - t1).count();
    t1 = tg;
    struct Done {
      const std::function<void()>* f;
      ~Done() {
        if (*f) (*f)();
      }
    } done{&req.gpu_done};
    const float* consent = (req.flags & kVbVerifySpeaker) ? req.consent_speaker : nullptr;
    if (req.flags & kVbConsent) {
      // consent create (P6): decode + speaker embedding only (the text checks ran above)
      AudioVerdict v = check_decoded(decoded.data(), decoded.size());
      if (v != AudioVerdict::kOk) {
        r->status = 200 + static_cast<int>(v) - 1;
      } else {
        r->pcm16.resize(decoded.size());
        std::vector<float> wav24(decoded.size());
        canonical_pcm16(decoded.data(), decoded.size(), r->pcm16.data(), wav24.data());
        r->raw_speaker = builder_.speaker_embedding(wav24.data(), wav24.size());
        // production create_voice_consent -> _pyannote_embedding: the same 512 / finite / unit-norm check as the
        // clone paths (else 503); this vector becomes the stored consent speaker every P3/P5 check compares with
        if (!speaker_embedding_ok(r->raw_speaker)) r->status = 213;
      }
    } else {
      *r = builder_.build_clone(decoded.data(), decoded.size(), text_ids.data(), text_ids.size(), consent);
    }
  }
  out.build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
  out.status = r->status;
  if (r->status != 0) {
    out.code = code_for(r->status);
    out.message = r->status == 201   ? duration_message(decoded.size())
                  : r->status == 200 ? "The uploaded audio contains non-finite samples."  // :11153 (decode ok)
                                     : message_for(r->status, req.flags);
    if (r->status >= 200 && r->status <= 202) out.param = field;
    if (r->status == 203 || r->status == 204 || r->status == 206) out.param = ref_param;
    return out;
  }
  out.result = r;
  if (catalogue_ && !(req.flags & kVbConsent) && r->nearest_index >= 0 &&
      (size_t)r->nearest_index < catalogue_->size()) {
    auto te = std::chrono::steady_clock::now();
    fill_projection(out);
    if (req.flags & kVbSaved) {
      // production _build_custom_voice_artifact: identity over the upload, the stored canonical WAV and the
      // stripped transcript. A decoded-PCM input has no upload bytes: its source digest is the canonical WAV's.
      auto wav = pcm16_wav_bytes(r->pcm16.data(), r->pcm16.size(), 24000);
      CloneIdentityInputs in;
      in.audio_sha256 = sha256_hex(wav.data(), wav.size());
      in.source_sha256 = (req.flags & kVbDecodedPcm) ? in.audio_sha256 : sha256_hex(req.audio, req.audio_len);
      in.transcript_sha256 = sha256_hex(transcript.data(), transcript.size());
      in.nearest_voice_id = out.nearest_voice_id;
      out.identity = production_identity(in);
      std::vector<float> projected(out.projected_speaker, out.projected_speaker + 2048);
      std::string id = req.voice_id.empty() ? "clone-" + in.audio_sha256.substr(0, 16) : req.voice_id;
      VoiceRecord rec = make_clone_record(id, "saved", r->lm_prefix, r->raw_speaker, projected,
                                          r->decoder.prompt_token, r->decoder.prompt_feat, r->decoder.xvector, "",
                                          out.identity);
      rec.meta = json::Value::make_object([&] {
        std::vector<std::pair<std::string, json::Value>> m;
        for (const auto& kv : rec.meta.members())
          m.emplace_back(kv.first, kv.first == "signature" ? json::Value::make_string(clone_signature(out.identity, rec))
                                                           : kv.second);
        return m;
      }());
      VoicesBlob blob;
      blob.top = default_top_meta();
      blob.voices.push_back(std::move(rec));
      out.artifact = write_voices_blob(blob);
    }
    out.export_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - te).count();
  }
  return out;
}

}  // namespace kt2n::voice
