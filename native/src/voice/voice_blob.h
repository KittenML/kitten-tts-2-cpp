// L11: voices.kt2n (magic KT2NVOX1) in memory: the built-in catalogue and the one-voice saved-voice artifact that
// engine.h kt2n_voice_load / kt2n_voice_export carry (the container format in tools/kt2n_assets/format.py "voices.kt2n", frozen at G0).
//
// The reader applies the same rules as tools/kt2n_assets/voices.py read(): header, offsets, canonical metadata (the
// parsed metadata re-serialised must equal the stored bytes), zero padding, no overlaps, per-voice fields with
// dtype/shape/value checks, segment sha256 and the copy rule (src_sha256 == data sha256). It never guesses a
// missing field. The writer emits bytes identical to the Python writer for the same content.
#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "../core/json.h"

namespace kt2n::voice {

class VoiceBlobError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

constexpr int kVoiceFields = 6;  // lm_prefix, raw_speaker, projected_speaker, prompt_token, prompt_feat, embedding
extern const char* const kVoiceFieldNames[kVoiceFields];

struct VoiceRecord {
  // Voice metadata object (FORMAT.md): id, public_names, language, hidden, prompt_tokens, prompt_frames,
  // lm_prefix_len, peak_gain_db, signature, kind [, identity]. The writer recomputes the three lengths.
  json::Value meta;
  std::vector<int32_t> lm_prefix;          // [P]
  std::vector<float> raw_speaker;          // [512], or empty for inference-only prepared voices
  std::vector<float> projected_speaker;    // [2048]
  std::vector<int32_t> prompt_token;       // [Tp]
  std::vector<float> prompt_feat;          // [2*Tp][80]
  std::vector<float> embedding;            // [192]
  // Per field: transform ("" = copy) and the sources list (empty = [{"name": <field>, "sha256": <own bytes>}]).
  std::array<std::string, kVoiceFields> transform;
  std::array<json::Value, kVoiceFields> sources;

  std::string id() const;
  std::string kind() const;
};

struct VoicesBlob {
  // Top-level metadata other than "voices" and "tensors": format, version, checkpoint, s3gen_sha256, peak_norm,
  // optional source. Missing format/version are filled by the writer.
  json::Value top;
  std::vector<VoiceRecord> voices;
};

// Throws VoiceBlobError on any rule violation.
VoicesBlob parse_voices_blob(const void* data, size_t len, bool verify_sha256 = true);
std::vector<uint8_t> write_voices_blob(const VoicesBlob& blob);

// json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False), byte for byte.
std::string canonical_json(const json::Value& v);
// Python repr(float) (shortest round-trip digits; fixed when -4 < decpt <= 16, else d.ddde+XX).
std::string python_float_repr(double x);

// The production constants a native clone carries (runtime.py :2872 CHECKPOINT_CACHE_TAG, FORMAT.md s3gen sha,
// peak_norm_gains_20260930.json default_gain_db).
extern const char* const kCheckpointCacheTag;
extern const char* const kS3genSha256;
constexpr double kCustomPeakGainDb = 0.087;
json::Value default_top_meta();  // format, version, checkpoint, s3gen_sha256, peak_norm

// One clone as a voice record. projected_speaker = the nearest catalogue voice's row (production projects the
// nearest voice's raw embedding with the same spk_proj call it uses for that built-in voice; checked bit-identical). identity: production ArtifactIdentity fields or null.
VoiceRecord make_clone_record(const std::string& id, const std::string& kind, const std::vector<int32_t>& lm_prefix,
                              const std::vector<float>& raw_speaker, const std::vector<float>& projected_speaker,
                              const std::vector<int32_t>& prompt_token, const std::vector<float>& prompt_feat,
                              const std::vector<float>& embedding, const std::string& signature,
                              const json::Value& identity = json::Value());

// The built-in catalogue as the clone path needs it: raw rows in file order (= the live bank's voice_ids order, the
// order production's _nearest_flagship_embedding iterates) and each voice's projected row.
struct Catalogue {
  std::vector<std::string> ids;
  std::vector<float> raw;        // [n][512]
  std::vector<float> projected;  // [n][2048]
  std::vector<uint8_t> hidden;
  static Catalogue from_blob(const VoicesBlob& blob);  // prepared voices are excluded from the clone bank
  size_t size() const { return ids.size(); }
  // production _nearest_flagship_embedding: cosine against candidate / ||candidate||, first max wins.
  int nearest(const float* emb512, float* best_sim = nullptr) const;
};

// Source, audio and transcript identity for a prepared clone with nearest-voice speaker projection.
struct CloneIdentityInputs {
  std::string source_sha256;      // sha256 of the upload as sent
  std::string audio_sha256;       // sha256 of the canonical PCM16 WAV (pcm16_wav_bytes)
  std::string transcript_sha256;  // sha256 of the stripped reference transcript (UTF-8)
  std::string nearest_voice_id;
};
json::Value production_identity(const CloneIdentityInputs& in);
// Content digest a native clone record carries as `signature`: sha256 of canonical_json({"identity": ...,
// "tensors": [data sha256 of the six fields in file order]}). (Production's artifact_identity digests its own NPZ
// descriptors; a natively built voice has its own tensors, so it gets its own digest.)
std::string clone_signature(const json::Value& identity, const VoiceRecord& r);

// kt2n_voice_load: production's _load_custom_reference gate (runtime.py :13175) on a one-voice saved artifact.
//   status 0   accepted (the engine serves the STORED projected_speaker, as production's materialize does);
//   status 1   KT2N_ERR_INVALID_ARGUMENT: not exactly one voice, or kind "builtin" (built-ins load from voices_file);
//   status 210 voice_artifact_version_mismatch (409, param "voice"): the identity differs from the running
//              deployment (`mismatched` names every field). Includes the nearest catalogue voice re-derived from
//              the artifact's own raw_speaker against the live catalogue (CLONE_SPEAKER_PROJECTION=nearest_flagship),
//              and a missing identity. Production answers by rebuilding from the stored WAV; the adapter does that
//              with kt2n_voice_build (the engine never guesses).
//   status 213 custom_voice_service_error (503): the blob is corrupt (production CorruptCustomVoiceStore).
struct LoadVerdict {
  int32_t status = 0;
  std::string code;
  std::string param;
  std::vector<std::string> mismatched;
};
LoadVerdict check_saved_voice(const void* data, size_t len, const Catalogue& catalogue, VoicesBlob* parsed = nullptr);

// Canonical PCM16 mono WAV bytes exactly as production's _pcm_wav_bytes writes them (44-byte RIFF header), and
// its sha256 (ArtifactIdentity.audio_sha256).
std::vector<uint8_t> pcm16_wav_bytes(const int16_t* pcm, size_t n, int rate = 24000);
std::string sha256_hex(const void* data, size_t len);

}  // namespace kt2n::voice
