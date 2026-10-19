// L11: voices.kt2n in memory (see voice_blob.h). CPU only.
#include "voice_blob.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <utility>

#include "kt2n/sha256.h"

namespace kt2n::voice {

const char* const kVoiceFieldNames[kVoiceFields] = {"lm_prefix",   "raw_speaker", "projected_speaker",
                                                    "prompt_token", "prompt_feat", "embedding"};
const char* const kCheckpointCacheTag = "459a3024236e49b01dc3b79f926f9363b5a1523045c1cadfb0ec7e1876d14cca";
const char* const kS3genSha256 = "d65cb687a2ed581ee6cc297e919ffefa63386944f42364ae13b78a594945514f";

namespace {

constexpr char kMagic[8] = {'K', 'T', '2', 'N', 'V', 'O', 'X', '1'};
constexpr uint64_t kHeader = 64, kAlign = 4096;
constexpr int kVocab = 158241, kS3Vocab = 6561, kMaxPrefix = 8192;
const char* const kFieldDtype[kVoiceFields] = {"i32", "f32", "f32", "i32", "f32", "f32"};

uint64_t align_up(uint64_t x) { return (x + kAlign - 1) / kAlign * kAlign; }

[[noreturn]] void fail(const std::string& what) { throw VoiceBlobError("voices.kt2n: " + what); }

template <class T>
T rd(const uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}

void dump_string(const std::string& s, std::string& out) {
  out.push_back('"');
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
          out += buf;
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  out.push_back('"');
}

void canon(const json::Value& v, std::string& out) {
  using T = json::Value::Type;
  switch (v.type()) {
    case T::kNull: out += "null"; break;
    case T::kBool: out += v.as_bool() ? "true" : "false"; break;
    case T::kInt: out += std::to_string(v.as_int()); break;
    case T::kDouble: out += python_float_repr(v.as_double()); break;
    case T::kString: dump_string(v.as_string(), out); break;
    case T::kArray:
      out.push_back('[');
      for (size_t i = 0; i < v.items().size(); ++i) {
        if (i) out.push_back(',');
        canon(v.items()[i], out);
      }
      out.push_back(']');
      break;
    case T::kObject: {
      std::vector<const std::pair<std::string, json::Value>*> m;
      for (const auto& kv : v.members()) m.push_back(&kv);
      // Python sorts str keys by code point; UTF-8 byte order is the same order.
      std::sort(m.begin(), m.end(), [](auto* a, auto* b) { return a->first < b->first; });
      out.push_back('{');
      for (size_t i = 0; i < m.size(); ++i) {
        if (i) out.push_back(',');
        dump_string(m[i]->first, out);
        out.push_back(':');
        canon(m[i]->second, out);
      }
      out.push_back('}');
      break;
    }
  }
}

json::Value obj(std::vector<std::pair<std::string, json::Value>> m) { return json::Value::make_object(std::move(m)); }
json::Value str(std::string s) { return json::Value::make_string(std::move(s)); }
json::Value num(int64_t i) { return json::Value::make_int(i); }

json::Value with(const json::Value& o, const std::string& key, json::Value val) {  // copy of o with key set
  std::vector<std::pair<std::string, json::Value>> m;
  bool done = false;
  for (const auto& kv : o.members()) {
    if (kv.first == key) {
      m.emplace_back(key, val);
      done = true;
    } else {
      m.push_back(kv);
    }
  }
  if (!done) m.emplace_back(key, std::move(val));
  return obj(std::move(m));
}

const json::Value& need(const json::Value& o, const char* key, std::string_view ctx) {
  const json::Value* v = o.get(key);
  if (!v) fail(std::string(ctx) + ": missing '" + key + "'");
  return *v;
}

bool is_count(const json::Value& v) { return v.is_int() && v.as_int() >= 0; }

struct FieldView {
  const void* p;
  size_t bytes;
  std::vector<int64_t> shape;
};

FieldView field_view(const VoiceRecord& r, int f) {
  switch (f) {
    case 0: return {r.lm_prefix.data(), r.lm_prefix.size() * 4, {(int64_t)r.lm_prefix.size()}};
    case 1: return {r.raw_speaker.data(), r.raw_speaker.size() * 4, {(int64_t)r.raw_speaker.size()}};
    case 2: return {r.projected_speaker.data(), r.projected_speaker.size() * 4, {(int64_t)r.projected_speaker.size()}};
    case 3: return {r.prompt_token.data(), r.prompt_token.size() * 4, {(int64_t)r.prompt_token.size()}};
    case 4: return {r.prompt_feat.data(), r.prompt_feat.size() * 4, {(int64_t)(r.prompt_feat.size() / 80), 80}};
    default: return {r.embedding.data(), r.embedding.size() * 4, {(int64_t)r.embedding.size()}};
  }
}

void check_record(const VoiceRecord& r, const std::string& id) {
  auto finite = [&](const std::vector<float>& v, const char* n) {
    for (float x : v)
      if (!std::isfinite(x)) fail(id + "/" + n + ": non-finite value");
  };
  if (r.lm_prefix.empty() || r.lm_prefix.size() > (size_t)kMaxPrefix) fail(id + "/lm_prefix: length");
  for (int32_t t : r.lm_prefix)
    if (t < 0 || t >= kVocab) fail(id + "/lm_prefix: id out of vocabulary");
  const bool raw_ok = r.raw_speaker.size() == 512 || (r.kind() == "prepared" && r.raw_speaker.empty());
  if (!raw_ok || r.projected_speaker.size() != 2048 || r.embedding.size() != 192)
    fail(id + ": speaker/embedding shapes");
  if (r.prompt_token.empty() || r.prompt_feat.size() != r.prompt_token.size() * 2 * 80)
    fail(id + ": prompt_feat is not [2*Tp][80]");
  for (int32_t t : r.prompt_token)
    if (t < 0 || t >= kS3Vocab) fail(id + "/prompt_token: value out of range");
  finite(r.raw_speaker, "raw_speaker");
  finite(r.projected_speaker, "projected_speaker");
  finite(r.prompt_feat, "prompt_feat");
  finite(r.embedding, "embedding");
}

}  // namespace

std::string sha256_hex(const void* data, size_t len) {
  kt2n::Sha256 h;
  h.update(data, len);
  return h.hexdigest();
}

std::string python_float_repr(double x) {
  if (!std::isfinite(x)) throw VoiceBlobError("voices.kt2n: non-finite number in metadata");
  if (x == 0.0) return std::signbit(x) ? "-0.0" : "0.0";
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof(buf), x, std::chars_format::scientific);  // shortest round trip
  std::string s(buf, r.ptr);
  bool neg = s[0] == '-';
  if (neg) s.erase(0, 1);
  size_t e = s.find('e');
  std::string digits;
  for (size_t i = 0; i < e; ++i)
    if (s[i] != '.') digits.push_back(s[i]);
  int decpt = std::stoi(s.substr(e + 1)) + 1;  // value = 0.d1d2... x 10^decpt
  std::string out;
  if (decpt > -4 && decpt <= 16) {
    if (decpt <= 0) {
      out = "0." + std::string((size_t)-decpt, '0') + digits;
    } else if (decpt >= (int)digits.size()) {
      out = digits + std::string((size_t)decpt - digits.size(), '0') + ".0";
    } else {
      out = digits.substr(0, (size_t)decpt) + "." + digits.substr((size_t)decpt);
    }
  } else {
    out = digits.substr(0, 1);
    if (digits.size() > 1) out += "." + digits.substr(1);
    int e10 = decpt - 1;
    char eb[16];
    std::snprintf(eb, sizeof(eb), "e%c%02d", e10 < 0 ? '-' : '+', std::abs(e10));
    out += eb;
  }
  return neg ? "-" + out : out;
}

std::string canonical_json(const json::Value& v) {
  std::string out;
  canon(v, out);
  return out;
}

std::string VoiceRecord::id() const {
  const json::Value* v = meta.get("id");
  return v && v->is_string() ? v->as_string() : std::string();
}
std::string VoiceRecord::kind() const {
  const json::Value* v = meta.get("kind");
  return v && v->is_string() ? v->as_string() : std::string();
}

json::Value default_top_meta() {
  json::Value peak = obj({
      {"schema", str("kittentts-peak-norm-gains.v1")},
      {"default_gain_db", json::Value::make_double(kCustomPeakGainDb)},
      {"default_rule", str("catalogue median of the voices' gains: the start gain of a voice with no calibration "
                           "(custom, shared, cloned) under KITTENTTS_09_PEAK_NORM_CUSTOM=median")},
      {"source_sha256", str("0008e338d315f82bf0b5c084db0438b94b328460c0d2db44749c7e249fccaf3e")},
  });
  return obj({{"format", str("kt2n")},
              {"version", num(1)},
              {"checkpoint", str(kCheckpointCacheTag)},
              {"s3gen_sha256", str(kS3genSha256)},
              {"peak_norm", std::move(peak)}});
}

VoiceRecord make_clone_record(const std::string& id, const std::string& kind, const std::vector<int32_t>& lm_prefix,
                              const std::vector<float>& raw_speaker, const std::vector<float>& projected_speaker,
                              const std::vector<int32_t>& prompt_token, const std::vector<float>& prompt_feat,
                              const std::vector<float>& embedding, const std::string& signature,
                              const json::Value& identity) {
  VoiceRecord r;
  std::vector<std::pair<std::string, json::Value>> m = {
      {"id", str(id)},
      {"public_names", json::Value::make_array({})},
      {"language", str("")},
      {"hidden", json::Value::make_bool(false)},
      {"peak_gain_db", json::Value::make_double(kCustomPeakGainDb)},
      {"signature", str(signature)},
      {"kind", str(kind)},
  };
  if (!identity.is_null()) m.emplace_back("identity", identity);
  r.meta = obj(std::move(m));
  r.lm_prefix = lm_prefix;
  r.raw_speaker = raw_speaker;
  r.projected_speaker = projected_speaker;
  r.prompt_token = prompt_token;
  r.prompt_feat = prompt_feat;
  r.embedding = embedding;
  return r;
}

std::vector<uint8_t> write_voices_blob(const VoicesBlob& blob) {
  if (blob.voices.empty()) fail("no voices");
  std::vector<json::Value> vmeta, tensors;
  std::vector<std::pair<uint64_t, FieldView>> segs;
  std::vector<std::string> seen;
  uint64_t cursor = 0;
  for (const VoiceRecord& r : blob.voices) {
    std::string id = r.id(), kind = r.kind();
    if (id.empty() || id.find('/') != std::string::npos) fail("voice id '" + id + "'");
    if (std::find(seen.begin(), seen.end(), id) != seen.end()) fail("duplicate voice id " + id);
    seen.push_back(id);
    if (kind != "builtin" && kind != "inline" && kind != "saved" && kind != "prepared")
      fail(id + ": kind '" + kind + "'");
    check_record(r, id);
    json::Value m = r.meta;
    int64_t tp = (int64_t)r.prompt_token.size();
    m = with(m, "prompt_tokens", num(tp));
    m = with(m, "prompt_frames", num(2 * tp));
    m = with(m, "lm_prefix_len", num((int64_t)r.lm_prefix.size()));
    vmeta.push_back(std::move(m));
    for (int f = 0; f < kVoiceFields; ++f) {
      FieldView fv = field_view(r, f);
      std::string digest = sha256_hex(fv.p, fv.bytes);
      std::vector<json::Value> shape;
      for (int64_t d : fv.shape) shape.push_back(num(d));
      std::string transform = r.transform[f].empty() ? "copy" : r.transform[f];
      json::Value sources = r.sources[f].is_array()
                                ? r.sources[f]
                                : json::Value::make_array({obj({{"name", str(kVoiceFieldNames[f])},
                                                                {"sha256", str(digest)}})});
      std::vector<std::pair<std::string, json::Value>> e = {
          {"name", str("voices/" + id + "/" + kVoiceFieldNames[f])},
          {"kind", str("dense")},
          {"dtype", str(kFieldDtype[f])},
          {"shape", json::Value::make_array(shape)},
          {"data", obj({{"off", num((int64_t)cursor)},
                        {"len", num((int64_t)fv.bytes)},
                        {"shape", json::Value::make_array(shape)},
                        {"dtype", str(kFieldDtype[f])},
                        {"sha256", str(digest)}})},
          {"group", str("voice")},
          {"transform", str(transform)},
          {"sources", sources},
      };
      if (transform == "copy") {
        const json::Value* s0 = sources.items().empty() ? nullptr : sources.items()[0].get("sha256");
        if (!s0 || !s0->is_string() || s0->as_string() != digest) fail(id + ": copy source digest differs");
        e.emplace_back("src_sha256", str(digest));
      }
      tensors.push_back(obj(std::move(e)));
      segs.emplace_back(cursor, fv);
      cursor = align_up(cursor + fv.bytes);
    }
  }
  json::Value top = blob.top.is_object() ? blob.top : default_top_meta();
  if (!top.get("format")) top = with(top, "format", str("kt2n"));
  if (!top.get("version")) top = with(top, "version", num(1));
  top = with(top, "voices", json::Value::make_array(vmeta));
  top = with(top, "tensors", json::Value::make_array(tensors));
  std::string mb = canonical_json(top);
  uint64_t data_off = align_up(kHeader + mb.size()), data_len = align_up(cursor);
  std::vector<uint8_t> out(data_off + data_len, 0);
  std::memcpy(out.data(), kMagic, 8);
  uint32_t ver = 1, flags = 0;
  uint64_t hv[5] = {kHeader, (uint64_t)mb.size(), data_off, data_len, data_off + data_len};
  std::memcpy(out.data() + 8, &ver, 4);
  std::memcpy(out.data() + 12, &flags, 4);
  std::memcpy(out.data() + 16, hv, sizeof(hv));
  std::memcpy(out.data() + kHeader, mb.data(), mb.size());
  for (const auto& [off, fv] : segs) std::memcpy(out.data() + data_off + off, fv.p, fv.bytes);
  return out;
}

VoicesBlob parse_voices_blob(const void* data, size_t len, bool verify_sha256) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  if (!p || len < kHeader) fail("shorter than the 64-byte header");
  if (std::memcmp(p, kMagic, 8) != 0) fail("bad magic");
  if (rd<uint32_t>(p + 8) != 1 || rd<uint32_t>(p + 12) != 0) fail("unsupported version/flags");
  for (int i = 56; i < 64; ++i)
    if (p[i]) fail("reserved header bytes are not zero");
  uint64_t meta_off = rd<uint64_t>(p + 16), meta_len = rd<uint64_t>(p + 24), data_off = rd<uint64_t>(p + 32),
           data_len = rd<uint64_t>(p + 40), file_len = rd<uint64_t>(p + 48);
  if (meta_off != kHeader || meta_len > len || data_off != align_up(kHeader + meta_len) || data_len % kAlign ||
      file_len != len || data_off > file_len || data_off + data_len != file_len)
    fail("bad offsets/lengths");
  std::string_view mb(reinterpret_cast<const char*>(p + meta_off), meta_len);
  json::Value meta;
  try {
    meta = json::parse(mb);
  } catch (const std::exception& e) {
    fail(std::string("metadata: ") + e.what());
  }
  if (!meta.is_object() || canonical_json(meta) != mb) fail("metadata is not canonical JSON");
  for (uint64_t i = meta_off + meta_len; i < data_off; ++i)
    if (p[i]) fail("non-zero padding after the metadata");
  const json::Value& fmt = need(meta, "format", "metadata");
  const json::Value& ver = need(meta, "version", "metadata");
  if (!fmt.is_string() || fmt.as_string() != "kt2n" || !ver.is_int() || ver.as_int() != 1)
    fail("metadata format/version");
  for (const char* k : {"checkpoint", "s3gen_sha256"}) {
    const json::Value& v = need(meta, k, "metadata");
    if (!v.is_string() || v.as_string().size() != 64) fail(std::string("metadata ") + k);
  }
  const json::Value& voices = need(meta, "voices", "metadata");
  const json::Value& tensors = need(meta, "tensors", "metadata");
  if (!voices.is_array() || voices.items().empty() || !tensors.is_array() ||
      tensors.items().size() != voices.items().size() * kVoiceFields)
    fail("tensor list does not match the voices");

  VoicesBlob out;
  {
    std::vector<std::pair<std::string, json::Value>> top;
    for (const auto& kv : meta.members())
      if (kv.first != "voices" && kv.first != "tensors") top.push_back(kv);
    out.top = obj(std::move(top));
  }
  // name -> tensor entry
  std::vector<std::pair<std::string, const json::Value*>> by_name;
  for (const auto& t : tensors.items()) {
    const json::Value& n = need(t, "name", "tensor");
    if (!n.is_string()) fail("tensor name");
    by_name.emplace_back(n.as_string(), &t);
  }
  std::sort(by_name.begin(), by_name.end(), [](auto& a, auto& b) { return a.first < b.first; });
  for (size_t i = 1; i < by_name.size(); ++i)
    if (by_name[i].first == by_name[i - 1].first) fail("duplicate tensor " + by_name[i].first);
  auto find = [&](const std::string& n) -> const json::Value* {
    auto it = std::lower_bound(by_name.begin(), by_name.end(), n, [](auto& a, const std::string& b) { return a.first < b; });
    return it != by_name.end() && it->first == n ? it->second : nullptr;
  };

  static const char* kRequired[] = {"id", "public_names", "language", "hidden", "prompt_tokens", "prompt_frames",
                                    "lm_prefix_len", "peak_gain_db", "signature", "kind"};
  std::vector<std::pair<uint64_t, uint64_t>> spans;
  std::vector<std::string> seen;
  for (const auto& vm : voices.items()) {
    if (!vm.is_object()) fail("voice metadata is not an object");
    for (const auto& kv : vm.members()) {
      bool known = kv.first == "identity";
      for (const char* k : kRequired) known |= kv.first == k;
      if (!known) fail("unknown voice metadata key '" + kv.first + "'");
    }
    for (const char* k : kRequired) need(vm, k, "voice");
    const json::Value& idv = need(vm, "id", "voice");
    const json::Value& kindv = need(vm, "kind", "voice");
    if (!idv.is_string() || idv.as_string().empty() || idv.as_string().find('/') != std::string::npos)
      fail("voice id");
    std::string id = idv.as_string();
    if (std::find(seen.begin(), seen.end(), id) != seen.end()) fail("duplicate voice " + id);
    seen.push_back(id);
    if (!kindv.is_string() || (kindv.as_string() != "builtin" && kindv.as_string() != "inline" &&
                               kindv.as_string() != "saved" && kindv.as_string() != "prepared"))
      fail(id + ": kind");
    if (!need(vm, "hidden", id).is_bool() || !need(vm, "peak_gain_db", id).is_number() ||
        !need(vm, "signature", id).is_string() || !need(vm, "language", id).is_string() ||
        !need(vm, "public_names", id).is_array())
      fail(id + ": metadata types");
    VoiceRecord r;
    r.meta = vm;
    for (int f = 0; f < kVoiceFields; ++f) {
      std::string name = "voices/" + id + "/" + kVoiceFieldNames[f];
      const json::Value* e = find(name);
      if (!e) fail(name + ": missing");
      const json::Value& kindt = need(*e, "kind", name);
      const json::Value& dt = need(*e, "dtype", name);
      const json::Value& sh = need(*e, "shape", name);
      const json::Value& seg = need(*e, "data", name);
      if (!kindt.is_string() || kindt.as_string() != "dense" || !dt.is_string() || dt.as_string() != kFieldDtype[f])
        fail(name + ": kind/dtype");
      const json::Value& sdt = need(seg, "dtype", name);
      const json::Value& ssh = need(seg, "shape", name);
      const json::Value& off = need(seg, "off", name);
      const json::Value& ln = need(seg, "len", name);
      const json::Value& sha = need(seg, "sha256", name);
      if (!sdt.is_string() || sdt.as_string() != kFieldDtype[f] || !ssh.is_array() || !sh.is_array() ||
          canonical_json(ssh) != canonical_json(sh) || !is_count(off) || !is_count(ln) || !sha.is_string())
        fail(name + ": segment fields");
      uint64_t numel = 1;
      size_t rank = ssh.items().size();
      if (rank != (f == 4 ? 2u : 1u)) fail(name + ": rank");
      for (const auto& d : ssh.items()) {
        // bound every dimension by the data size before multiplying, so a hostile shape cannot wrap the product
        if (!is_count(d) || (uint64_t)d.as_int() > data_len) fail(name + ": shape");
        numel *= (uint64_t)d.as_int();
        if (numel > data_len) fail(name + ": shape larger than the data");
      }
      if (f == 4 && ssh.items()[1].as_int() != 80) fail(name + ": prompt_feat width");
      uint64_t o = (uint64_t)off.as_int(), l = (uint64_t)ln.as_int();
      if (o % kAlign || l != numel * 4 || o > data_len || l > data_len - o) fail(name + ": segment bounds");
      const uint8_t* src = p + data_off + o;
      if (verify_sha256 && sha256_hex(src, l) != sha.as_string()) fail(name + ": sha256 mismatch");
      const json::Value& tr = need(*e, "transform", name);
      if (!tr.is_string()) fail(name + ": transform");
      if (tr.as_string() == "copy") {
        const json::Value* s = e->get("src_sha256");
        if (!s || !s->is_string() || s->as_string() != sha.as_string()) fail(name + ": src_sha256");
        r.transform[f] = "";
      } else {
        r.transform[f] = tr.as_string();
      }
      r.sources[f] = need(*e, "sources", name);
      spans.emplace_back(o, o + l);
      switch (f) {
        case 0: r.lm_prefix.resize(numel); std::memcpy(r.lm_prefix.data(), src, l); break;
        case 1: r.raw_speaker.resize(numel); if (l) std::memcpy(r.raw_speaker.data(), src, l); break;
        case 2: r.projected_speaker.resize(numel); std::memcpy(r.projected_speaker.data(), src, l); break;
        case 3: r.prompt_token.resize(numel); std::memcpy(r.prompt_token.data(), src, l); break;
        case 4: r.prompt_feat.resize(numel); std::memcpy(r.prompt_feat.data(), src, l); break;
        default: r.embedding.resize(numel); std::memcpy(r.embedding.data(), src, l); break;
      }
    }
    check_record(r, id);
    const json::Value& pt = need(vm, "prompt_tokens", id);
    const json::Value& pf = need(vm, "prompt_frames", id);
    const json::Value& pl = need(vm, "lm_prefix_len", id);
    if (!pt.is_int() || !pf.is_int() || !pl.is_int() || pt.as_int() != (int64_t)r.prompt_token.size() ||
        pf.as_int() != 2 * pt.as_int() || pl.as_int() != (int64_t)r.lm_prefix.size())
      fail(id + ": metadata lengths disagree with the tensors");
    out.voices.push_back(std::move(r));
  }
  std::sort(spans.begin(), spans.end());
  uint64_t cur = 0;
  for (const auto& [a, b] : spans) {
    if (a < cur) fail("segments overlap");
    for (uint64_t i = cur; i < a; ++i)
      if (p[data_off + i]) fail("non-zero padding between segments");
    cur = b;
  }
  for (uint64_t i = cur; i < data_len; ++i)
    if (p[data_off + i]) fail("non-zero padding at the end");
  return out;
}

Catalogue Catalogue::from_blob(const VoicesBlob& blob) {
  Catalogue c;
  for (const VoiceRecord& r : blob.voices) {
    // Prepared public voices carry their projected vector directly. They are
    // inference inputs, not members of the raw-speaker nearest-neighbour bank.
    if (r.kind() == "prepared") continue;
    if (r.kind() != "builtin") fail("catalogue voice " + r.id() + " is not builtin");
    c.ids.push_back(r.id());
    c.raw.insert(c.raw.end(), r.raw_speaker.begin(), r.raw_speaker.end());
    c.projected.insert(c.projected.end(), r.projected_speaker.begin(), r.projected_speaker.end());
    const json::Value* h = r.meta.get("hidden");
    c.hidden.push_back(h && h->is_bool() && h->as_bool());
  }
  return c;
}

int Catalogue::nearest(const float* emb, float* best_sim) const {
  int best = -1;
  float best_s = -1.f;
  for (size_t i = 0; i < ids.size(); ++i) {
    const float* c = raw.data() + i * 512;
    double nn = 0.0;
    for (int d = 0; d < 512; ++d) nn += double(c[d]) * c[d];
    float norm = (float)std::sqrt(nn);
    if (norm <= 0.f) continue;
    float s = 0.f;
    for (int d = 0; d < 512; ++d) s += emb[d] * (c[d] / norm);
    if (s > best_s) {
      best_s = s;
      best = (int)i;
    }
  }
  if (best_sim) *best_sim = best_s;
  return best;
}

json::Value production_identity(const CloneIdentityInputs& in) {
  return obj({{"checkpoint_revision", str(kCheckpointCacheTag)},
              {"demo_commit", str("5f9e55614f55cd493d5cfa8e7691b6cdd3d195ab")},
              {"pyannote_model", str("pyannote/embedding")},
              {"pyannote_revision", str("4db4899737a38b2d618bbd74350915aa10293cb2")},
              {"s3gen_model", str("ResembleAI/chatterbox-turbo")},
              {"s3gen_revision", str("1e4698ca7cbb41ff030c4185f0927a3b42d76924")},
              {"source_sha256", str(in.source_sha256)},
              {"audio_sha256", str(in.audio_sha256)},
              {"reference_transcript_sha256", str(in.transcript_sha256)},
              {"nearest_flagship_voice_id", str(in.nearest_voice_id)},
              {"vocabulary_size", num(kVocab)},
              {"speaker_projection_policy", str("nearest_flagship")},
              {"use_icl", json::Value::make_bool(true)}});
}

std::string clone_signature(const json::Value& identity, const VoiceRecord& r) {
  std::vector<json::Value> shas;
  for (int f = 0; f < kVoiceFields; ++f) {
    FieldView fv = field_view(r, f);
    shas.push_back(str(sha256_hex(fv.p, fv.bytes)));
  }
  std::string c = canonical_json(obj({{"identity", identity}, {"tensors", json::Value::make_array(shas)}}));
  return sha256_hex(c.data(), c.size());
}

LoadVerdict check_saved_voice(const void* data, size_t len, const Catalogue& catalogue, VoicesBlob* parsed) {
  LoadVerdict v;
  VoicesBlob b;
  try {
    b = parse_voices_blob(data, len, true);
  } catch (const VoiceBlobError&) {
    v.status = 213;
    v.code = "custom_voice_service_error";
    return v;
  }
  if (b.voices.size() != 1 ||
      (b.voices[0].kind() != "saved" && b.voices[0].kind() != "inline") ||
      b.voices[0].raw_speaker.size() != 512) {
    v.status = 1;
    v.code = "internal_error";
    return v;
  }
  const VoiceRecord& r = b.voices[0];
  auto mismatch = [&](const std::string& f) { v.mismatched.push_back(f); };
  const json::Value* ck = b.top.get("checkpoint");
  const json::Value* s3 = b.top.get("s3gen_sha256");
  if (!ck || !ck->is_string() || ck->as_string() != kCheckpointCacheTag) mismatch("file.checkpoint");
  if (!s3 || !s3->is_string() || s3->as_string() != kS3genSha256) mismatch("file.s3gen_sha256");
  const json::Value* ident = r.meta.get("identity");
  if (!ident || !ident->is_object()) {
    mismatch("identity");
  } else {
    int nearest = catalogue.nearest(r.raw_speaker.data());
    CloneIdentityInputs none;
    none.nearest_voice_id = nearest >= 0 ? catalogue.ids[(size_t)nearest] : std::string();
    json::Value expected = production_identity(none);
    for (const auto& kv : expected.members()) {
      if (kv.first == "source_sha256" || kv.first == "audio_sha256" || kv.first == "reference_transcript_sha256")
        continue;  // per-voice digests, not deployment identity
      const json::Value* got = ident->get(kv.first);
      if (!got || canonical_json(*got) != canonical_json(kv.second)) mismatch(kv.first);
    }
  }
  if (!v.mismatched.empty()) {
    v.status = 210;
    v.code = "voice_artifact_version_mismatch";
    v.param = "voice";
  }
  if (parsed) *parsed = std::move(b);
  return v;
}

std::vector<uint8_t> pcm16_wav_bytes(const int16_t* pcm, size_t n, int rate) {
  uint32_t data = (uint32_t)(n * 2);
  std::vector<uint8_t> out(44 + data);
  auto put = [&](size_t at, const void* v, size_t k) { std::memcpy(out.data() + at, v, k); };
  uint32_t riff = 36 + data, fmt_len = 16, sr = (uint32_t)rate, byte_rate = (uint32_t)rate * 2;
  uint16_t pcm_tag = 1, ch = 1, block = 2, bits = 16;
  put(0, "RIFF", 4);
  put(4, &riff, 4);
  put(8, "WAVEfmt ", 8);
  put(16, &fmt_len, 4);
  put(20, &pcm_tag, 2);
  put(22, &ch, 2);
  put(24, &sr, 4);
  put(28, &byte_rate, 4);
  put(32, &block, 2);
  put(34, &bits, 2);
  put(36, "data", 4);
  put(40, &data, 4);
  if (n) put(44, pcm, data);
  return out;
}

}  // namespace kt2n::voice
