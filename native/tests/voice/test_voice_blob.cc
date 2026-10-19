// L11: voices.kt2n in memory (src/voice/voice_blob.cc): Python float repr, canonical JSON, round trip,
// determinism, strict rejection, production's canonical WAV header. CPU only.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

#include "voice_blob.h"

#include "py_float_repr.inc"

using namespace kt2n::voice;

static int g_fail = 0;
#define CHECK(c)                                                       \
  do {                                                                 \
    if (!(c)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
      ++g_fail;                                                        \
    }                                                                  \
  } while (0)

static VoiceRecord synth(const std::string& id, int tp, int p, unsigned seed, const char* kind = "builtin") {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd;
  std::vector<int32_t> prefix(p), tok(tp);
  for (auto& x : prefix) x = (int32_t)(rng() % 158241);
  for (auto& x : tok) x = (int32_t)(rng() % 6561);
  std::vector<float> raw(512), proj(2048), feat((size_t)tp * 160), emb(192);
  for (auto* v : {&raw, &proj, &feat, &emb})
    for (auto& x : *v) x = nd(rng);
  VoiceRecord r = make_clone_record(id, kind, prefix, raw, proj, tok, feat, emb, "sig-" + id);
  return r;
}

static std::vector<uint8_t> blob_of(std::vector<VoiceRecord> v) {
  VoicesBlob b;
  b.top = default_top_meta();
  b.voices = std::move(v);
  return write_voices_blob(b);
}

static bool rejects(std::vector<uint8_t> bytes) {
  try {
    parse_voices_blob(bytes.data(), bytes.size());
  } catch (const VoiceBlobError&) {
    return true;
  }
  return false;
}

int main() {
  // Public prepared references have a projected speaker but no raw embedding.
  // They round-trip without fabricated tensors and never enter the clone bank.
  {
    auto prepared = synth("Prepared", 3, 8, 42, "prepared");
    prepared.raw_speaker.clear();
    auto bytes = blob_of({prepared});
    auto decoded = parse_voices_blob(bytes.data(), bytes.size());
    CHECK(decoded.voices[0].raw_speaker.empty());
    CHECK(decoded.voices[0].projected_speaker == prepared.projected_speaker);
    CHECK(Catalogue::from_blob(decoded).size() == 0);
    CHECK(write_voices_blob(decoded) == bytes);
  }
  // 1. Python repr(float), exact strings from CPython.
  for (const auto& c : kPyRepr) {
    double d;
    std::memcpy(&d, &c.bits, 8);
    std::string got = python_float_repr(d);
    if (got != c.repr) std::fprintf(stderr, "repr %s != %s\n", got.c_str(), c.repr);
    CHECK(got == c.repr);
  }

  // 2. Canonical JSON: sorted keys, compact, escapes as Python's json (ensure_ascii=False).
  {
    auto v = kt2n::json::parse(R"({"b":1,"a":[true,null,"x\"\\\n\u001fé"],"c":{"z":0.5,"y":-0.0}})");
    CHECK(canonical_json(v) == "{\"a\":[true,null,\"x\\\"\\\\\\n\\u001f\xC3\xA9\"],\"b\":1,\"c\":{\"y\":-0.0,\"z\":0.5}}");
  }

  // 3. Round trip, alignment, determinism.
  std::vector<VoiceRecord> vs = {synth("a", 7, 30, 1), synth("voice_x", 250, 327, 2, "saved")};
  auto blob = blob_of(vs);
  CHECK(blob.size() % 4096 == 0);
  CHECK(std::memcmp(blob.data(), "KT2NVOX1", 8) == 0);
  CHECK(blob_of(vs) == blob);
  auto parsed = parse_voices_blob(blob.data(), blob.size());
  CHECK(parsed.voices.size() == 2);
  CHECK(parsed.voices[1].id() == "voice_x" && parsed.voices[1].kind() == "saved");
  CHECK(parsed.voices[1].prompt_feat == vs[1].prompt_feat && parsed.voices[1].lm_prefix == vs[1].lm_prefix);
  CHECK(parsed.voices[0].projected_speaker == vs[0].projected_speaker);
  CHECK(parsed.voices[1].meta.get("prompt_frames")->as_int() == 500);
  CHECK(write_voices_blob(parsed) == blob);  // parse -> write reproduces the bytes

  // 4. Rejections.
  {
    auto b = blob;
    b[7] = '2';
    CHECK(rejects(b));
    b = blob;
    b[60] = 1;
    CHECK(rejects(b));
    b = blob;
    uint64_t data_off;
    std::memcpy(&data_off, b.data() + 32, 8);
    b[data_off + 9] ^= 1;  // tensor byte -> sha256 mismatch
    CHECK(rejects(b));
    b = blob;
    b.resize(b.size() - 4096);
    CHECK(rejects(b));
    b = blob;
    uint64_t meta_len;
    std::memcpy(&meta_len, b.data() + 24, 8);
    b[64 + meta_len + 2] = 7;  // padding after metadata
    CHECK(rejects(b));
    b = blob;
    b.back() = 1;  // padding at the end
    CHECK(rejects(b));
    b = blob;
    auto pos = std::string(b.begin() + 64, b.begin() + 64 + meta_len).find("\"kind\":\"saved\"");
    CHECK(pos != std::string::npos);
    std::memcpy(b.data() + 64 + pos + 8, "inlin", 5);  // "saved" -> "inlin": unknown kind
    CHECK(rejects(b));
    CHECK(rejects({}));
  }
  {
    auto bad = synth("a", 7, 30, 3);
    bad.prompt_token[0] = 6561;
    bool threw = false;
    try {
      blob_of({bad});
    } catch (const VoiceBlobError&) {
      threw = true;
    }
    CHECK(threw);
    bad = synth("a", 7, 30, 3);
    bad.embedding[5] = NAN;
    threw = false;
    try {
      blob_of({bad});
    } catch (const VoiceBlobError&) {
      threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
      blob_of({synth("a", 7, 30, 1), synth("a", 7, 30, 2)});
    } catch (const VoiceBlobError&) {
      threw = true;
    }
    CHECK(threw);
  }

  // 5. Production's canonical WAV (wave module, 44-byte header).
  {
    int16_t pcm[3] = {0, -32767, 32767};
    auto w = pcm16_wav_bytes(pcm, 3, 24000);
    static const uint8_t want[44] = {'R', 'I', 'F', 'F', 42, 0, 0, 0, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ',
                                     16,  0,   0,   0,   1,  0, 1, 0, 0xC0, 0x5D, 0, 0, 0x80, 0xBB, 0, 0,
                                     2,   0,   16,  0,   'd', 'a', 't', 'a', 6, 0, 0, 0};
    CHECK(w.size() == 50 && std::memcmp(w.data(), want, 44) == 0);
  }

  // 6. kt2n_voice_load gate (production _load_custom_reference).
  {
    std::vector<VoiceRecord> cv = {synth("v0", 7, 30, 10), synth("v1", 7, 30, 11), synth("v2", 7, 30, 12)};
    for (auto& r : cv) {  // unit raw speakers, as the bank holds
      double nn = 0;
      for (float x : r.raw_speaker) nn += double(x) * x;
      for (float& x : r.raw_speaker) x = float(x / std::sqrt(nn));
    }
    Catalogue cat = Catalogue::from_blob(parse_voices_blob(blob_of(cv).data(), blob_of(cv).size()));
    CHECK(cat.size() == 3 && cat.nearest(cv[1].raw_speaker.data()) == 1);
    auto saved = [&](const std::string& nearest_id, bool with_identity, const char* demo = nullptr) {
      VoiceRecord r = synth("voice_s", 9, 40, 20, "saved");
      r.raw_speaker = cv[1].raw_speaker;
      r.raw_speaker[0] += 1e-3f;
      CloneIdentityInputs in{std::string(64, 'a'), std::string(64, 'b'), std::string(64, 'c'), nearest_id};
      kt2n::json::Value id = production_identity(in);
      if (demo) {
        std::vector<std::pair<std::string, kt2n::json::Value>> m;
        for (const auto& kv : id.members())
          m.emplace_back(kv.first, kv.first == "demo_commit" ? kt2n::json::Value::make_string(demo) : kv.second);
        id = kt2n::json::Value::make_object(m);
      }
      VoiceRecord out = make_clone_record("voice_s", "saved", r.lm_prefix, r.raw_speaker, r.projected_speaker,
                                          r.prompt_token, r.prompt_feat, r.embedding, "sig",
                                          with_identity ? id : kt2n::json::Value());
      return blob_of({out});
    };
    auto ok = saved("v1", true);
    CHECK(check_saved_voice(ok.data(), ok.size(), cat).status == 0);
    auto wrong = saved("v0", true);
    LoadVerdict w = check_saved_voice(wrong.data(), wrong.size(), cat);
    CHECK(w.status == 210 && w.code == "voice_artifact_version_mismatch" && w.param == "voice" &&
          w.mismatched.size() == 1 && w.mismatched[0] == "nearest_flagship_voice_id");
    auto demo = saved("v1", true, "0000000000000000000000000000000000000000");
    LoadVerdict d = check_saved_voice(demo.data(), demo.size(), cat);
    CHECK(d.status == 210 && d.mismatched.size() == 1 && d.mismatched[0] == "demo_commit");
    auto noid = saved("v1", false);
    LoadVerdict n = check_saved_voice(noid.data(), noid.size(), cat);
    CHECK(n.status == 210 && n.mismatched.size() == 1 && n.mismatched[0] == "identity");
    auto builtin = blob_of({synth("b", 7, 30, 30)});
    CHECK(check_saved_voice(builtin.data(), builtin.size(), cat).status == 1);
    // A valid prepared inference record may omit raw_speaker. Loading it as a
    // saved clone must reject its kind before nearest-speaker lookup.
    auto prepared = synth("prepared", 7, 30, 31, "prepared");
    prepared.raw_speaker.clear();
    auto prepared_blob = blob_of({prepared});
    CHECK(check_saved_voice(prepared_blob.data(), prepared_blob.size(), cat).status == 1);
    auto with_identity = make_clone_record("prepared", "prepared", prepared.lm_prefix,
        prepared.raw_speaker, prepared.projected_speaker, prepared.prompt_token, prepared.prompt_feat,
        prepared.embedding, "sig", production_identity(CloneIdentityInputs{}));
    prepared_blob = blob_of({with_identity});
    CHECK(check_saved_voice(prepared_blob.data(), prepared_blob.size(), cat).status == 1);
    auto corrupt = ok;
    corrupt[corrupt.size() - 4096 - 1] ^= 0x40;  // inside the last tensor or its padding
    CHECK(check_saved_voice(corrupt.data(), corrupt.size(), cat).status == 213);
    VoicesBlob other;
    other.top = default_top_meta();
    std::vector<std::pair<std::string, kt2n::json::Value>> m;
    for (const auto& kv : other.top.members())
      m.emplace_back(kv.first, kv.first == "checkpoint" ? kt2n::json::Value::make_string(std::string(64, '0')) : kv.second);
    other.top = kt2n::json::Value::make_object(m);
    other.voices = parse_voices_blob(ok.data(), ok.size()).voices;
    auto oldck = write_voices_blob(other);
    LoadVerdict o = check_saved_voice(oldck.data(), oldck.size(), cat);
    CHECK(o.status == 210 && o.mismatched.size() == 1 && o.mismatched[0] == "file.checkpoint");
  }

  std::printf("%s\n", g_fail ? "test_voice_blob: FAIL" : "test_voice_blob: PASS");
  return g_fail ? 1 : 0;
}
