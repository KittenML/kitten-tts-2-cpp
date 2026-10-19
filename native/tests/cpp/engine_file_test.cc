// Engine-file reader + verify_all against files written by make_tiny_kt2n.py
// (an independent numpy implementation of the FORMAT and of pack_exact).
#include "kt2n/engine_file.h"

#include <gtest/gtest.h>

#include <bit>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "json.h"
#include "kt2n/bf16.h"
#include "kt2n/ref_kernels.h"
#include "kt2n/sha256.h"
#include "test_util.h"

namespace {

using kt2n::EngineFile;
using kt2n::FormatError;
using kt2n::TensorKind;
using kt2n::test::data_path;

constexpr const char* kUp = "model.layers.0.mlp.up_proj.weight";
constexpr const char* kQ = "model.layers.0.self_attn.q_proj.weight";

std::set<std::pair<std::string, std::string>> error_checks(const kt2n::VerifyReport& r) {
  std::set<std::pair<std::string, std::string>> out;
  for (const auto& e : r.errors) out.emplace(e.tensor, e.check);
  return out;
}

template <typename T>
std::vector<T> copy_segment(const EngineFile& f, const kt2n::Segment& s) {
  std::vector<T> out(s.len / sizeof(T));
  std::memcpy(out.data(), f.bytes(s).data(), s.len);
  return out;
}

TEST(EngineFile, HeaderAndTable) {
  const EngineFile f = EngineFile::open(data_path("tiny.kt2n"));
  const auto& h = f.header();
  EXPECT_EQ(std::string(h.magic, 8), "KT2NENG1");
  EXPECT_EQ(h.version, 1u);
  EXPECT_EQ(h.flags, 0u);
  EXPECT_EQ(h.meta_off, 64u);
  EXPECT_EQ(h.data_off % 4096, 0u);
  EXPECT_EQ(h.file_len, f.size());
  EXPECT_TRUE(f.data_page_aligned());
  ASSERT_EQ(f.tensors().size(), 5u);
  EXPECT_EQ(f.tensors()[0].name, "model.embed_tokens.weight");

  const auto* up = f.find(kUp);
  ASSERT_NE(up, nullptr);
  EXPECT_EQ(up->kind, TensorKind::kTernary2);
  EXPECT_EQ(up->dtype, kt2n::DType::kBF16);
  EXPECT_EQ(up->shape, (std::vector<int64_t>{24, 512}));
  EXPECT_EQ(up->rows(), 24);
  EXPECT_EQ(up->cols(), 512);
  EXPECT_EQ(up->codes.shape, (std::vector<int64_t>{24, 32}));
  EXPECT_EQ(up->codes.dtype, kt2n::DType::kI32);
  EXPECT_EQ(up->codes.len, 24u * 32u * 4u);
  EXPECT_EQ(up->scales.shape, (std::vector<int64_t>{24, 4}));
  EXPECT_EQ(up->scales.len, 24u * 4u * 2u);
  EXPECT_EQ(up->codes.abs_off, h.data_off + up->codes.off);
  EXPECT_EQ(up->codes.ptr, f.base() + up->codes.abs_off);
  EXPECT_EQ(up->codes.abs_off % 4096, 0u);
  // up_proj has -0.0 weights -> zero_signs segment; q_proj has none.
  EXPECT_TRUE(up->has_zero_signs);
  EXPECT_EQ(up->segments().size(), 3u);
  EXPECT_GT(up->zeros, 0);
  EXPECT_GT(up->neg_zero, 0);
  EXPECT_EQ(up->zero_signs.dtype, kt2n::DType::kU8);
  EXPECT_EQ(up->zero_signs.shape, (std::vector<int64_t>{(up->zeros + 7) / 8}));
  const auto* q = f.find(kQ);
  ASSERT_NE(q, nullptr);
  EXPECT_FALSE(q->has_zero_signs);
  EXPECT_EQ(q->segments().size(), 2u);
  EXPECT_EQ(q->neg_zero, 0);
  EXPECT_GT(q->zeros, 0);

  const auto* bias = f.find("spk_proj.0.bias");
  ASSERT_NE(bias, nullptr);
  EXPECT_EQ(bias->kind, TensorKind::kDense);
  EXPECT_EQ(bias->dtype, kt2n::DType::kF32);
  EXPECT_EQ(bias->data.len, 16u * 4u);
  EXPECT_EQ(f.find("nope"), nullptr);
}

TEST(EngineFile, MetadataLookups) {
  const EngineFile f = EngineFile::open(data_path("tiny.kt2n"));
  EXPECT_EQ(f.meta_string("/format"), "kt2n");
  EXPECT_EQ(f.meta_int("/version"), 1);
  EXPECT_EQ(f.meta_int("/config/hidden_size"), 256);
  EXPECT_DOUBLE_EQ(f.meta_number("/config/rms_norm_eps").value(), 1e-6);
  EXPECT_EQ(f.meta_int("/runtime_config/audio_token_start"), 16);
  EXPECT_EQ(f.meta_int("/quant/group"), 128);
  EXPECT_EQ(f.meta_int("/tensors/1/shape/1"), 512);
  EXPECT_FALSE(f.meta_int("/config/missing").has_value());
  EXPECT_FALSE(f.meta_int("/format").has_value());  // wrong type
  EXPECT_TRUE(f.meta_has("/source/seed"));
  EXPECT_EQ(f.meta_dump("/quant/codes"), R"({"0":0,"1":"+1","2":"-1"})");
  EXPECT_FALSE(f.meta_json().empty());
  EXPECT_EQ(f.meta_json().front(), '{');
}

TEST(EngineFile, MoveKeepsMappingValid) {
  EngineFile a = EngineFile::open(data_path("tiny.kt2n"));
  const auto* up = a.find(kUp);
  const std::string want = up->codes.sha256;
  EngineFile b = std::move(a);
  const auto* up2 = b.find(kUp);
  ASSERT_NE(up2, nullptr);
  const auto bytes = b.bytes(up2->codes);
  EXPECT_EQ(kt2n::sha256_hex(bytes.data(), bytes.size()), want);
}

TEST(EngineFile, VerifyAllPassesOnValidFile) {
  const EngineFile f = EngineFile::open(data_path("tiny.kt2n"));
  for (int threads : {0, 1, 3}) {
    kt2n::VerifyOptions opts;
    opts.threads = threads;
    const auto rep = kt2n::verify_all(f, opts);
    EXPECT_TRUE(rep.ok) << "threads " << threads;
    EXPECT_TRUE(rep.errors.empty());
    EXPECT_TRUE(rep.warnings.empty());
    EXPECT_EQ(rep.tensors, 5u);
    EXPECT_EQ(rep.ternary, 2u);
    EXPECT_EQ(rep.dense, 3u);
    EXPECT_EQ(rep.segments, 8u);
    EXPECT_EQ(rep.invalid_codes, 0u);
  }
}

TEST(EngineFile, VerifyAllCatchesFlippedCodeByte) {
  const EngineFile f = EngineFile::open(data_path("tiny_badseg.kt2n"));
  const auto rep = kt2n::verify_all(f);
  EXPECT_FALSE(rep.ok);
  const auto checks = error_checks(rep);
  EXPECT_TRUE(checks.count({kUp, "seg_sha256:codes"})) << rep.errors.size();
  EXPECT_TRUE(checks.count({kUp, "src_sha256"}));
  for (const auto& e : rep.errors) EXPECT_EQ(e.tensor, kUp) << e.check;
}

TEST(EngineFile, VerifyAllCatchesWrongSourceHash) {
  const EngineFile f = EngineFile::open(data_path("tiny_badsrc.kt2n"));
  const auto rep = kt2n::verify_all(f);
  EXPECT_FALSE(rep.ok);
  ASSERT_EQ(rep.errors.size(), 1u);
  EXPECT_EQ(rep.errors[0].tensor, kUp);
  EXPECT_EQ(rep.errors[0].check, "src_sha256");
  kt2n::VerifyOptions no_rebuild;
  no_rebuild.reconstruct = false;
  EXPECT_TRUE(kt2n::verify_all(f, no_rebuild).ok);
}

TEST(EngineFile, NoReconstructChecksSegmentHashesOnly) {
  // tiny_badcode's hashes are recorded over the bad codes: every segment hash
  // matches, so the segment-only mode passes and does no per-weight work.
  const EngineFile f = EngineFile::open(data_path("tiny_badcode.kt2n"));
  kt2n::VerifyOptions no_rebuild;
  no_rebuild.reconstruct = false;
  const auto rep = kt2n::verify_all(f, no_rebuild);
  EXPECT_TRUE(rep.ok);
  EXPECT_TRUE(rep.errors.empty());
  EXPECT_EQ(rep.invalid_codes, 0u);
  uint64_t segment_bytes = 0;
  for (const auto& t : f.tensors()) {
    for (const auto& [name, seg] : t.segments()) segment_bytes += seg->len;
  }
  EXPECT_EQ(rep.bytes_hashed, segment_bytes);  // no rebuilt bf16 bytes
  EXPECT_FALSE(kt2n::verify_all(f).ok);         // the full mode still catches the code
  // A flipped code byte breaks a segment hash, which the segment-only mode reports.
  const EngineFile g = EngineFile::open(data_path("tiny_badseg.kt2n"));
  const auto bad = kt2n::verify_all(g, no_rebuild);
  EXPECT_FALSE(bad.ok);
  EXPECT_TRUE(error_checks(bad).count({kUp, "seg_sha256:codes"}));
}

TEST(EngineFile, VerifyAllCatchesWrongNegZeroCount) {
  const EngineFile f = EngineFile::open(data_path("tiny_badzero.kt2n"));
  const auto rep = kt2n::verify_all(f);
  EXPECT_FALSE(rep.ok);
  ASSERT_EQ(rep.errors.size(), 1u);
  EXPECT_EQ(rep.errors[0].tensor, kUp);
  EXPECT_EQ(rep.errors[0].check, "neg_zero");
}

TEST(EngineFile, VerifyAllCatchesDirtyPadding) {
  auto bytes = kt2n::test::read_bytes(data_path("tiny.kt2n"));
  uint64_t data_off = 0;
  std::memcpy(&data_off, bytes.data() + 32, 8);
  bytes[data_off - 1] = 0x01;  // last byte of the metadata->data padding
  const std::string p = data_path("dirty_padding.kt2n");
  kt2n::test::write_bytes(p, bytes);
  {
    const EngineFile f = EngineFile::open(p);  // structure is still valid
    const auto rep = kt2n::verify_all(f);
    EXPECT_FALSE(rep.ok);
    ASSERT_EQ(rep.errors.size(), 1u);
    EXPECT_EQ(rep.errors[0].check, "layout:padding");
  }
  std::remove(p.c_str());
}

TEST(EngineFile, VerifyAllCatchesInvalidCode) {
  const EngineFile f = EngineFile::open(data_path("tiny_badcode.kt2n"));
  const auto rep = kt2n::verify_all(f);
  EXPECT_FALSE(rep.ok);
  EXPECT_EQ(rep.invalid_codes, 1u);
  EXPECT_TRUE(error_checks(rep).count({kUp, "invalid_code"}));
}

// Cross-language checks against numbers computed by make_tiny_kt2n.py.
class Expected : public ::testing::Test {
 protected:
  void SetUp() override {
    expected_ = kt2n::json::parse(kt2n::test::read_text(data_path("tiny.expected.json")));
  }
  kt2n::json::Value expected_;
};

TEST_F(Expected, UnpackRebuildsSourceBits) {
  const EngineFile f = EngineFile::open(data_path("tiny.kt2n"));
  const auto* q = f.find(kQ);
  ASSERT_NE(q, nullptr);
  const auto codes = copy_segment<int32_t>(f, q->codes);
  const auto scales = copy_segment<uint16_t>(f, q->scales);
  std::vector<uint16_t> w(static_cast<size_t>(q->rows() * q->cols()));
  EXPECT_EQ(kt2n::ref::unpack_ternary(codes.data(), scales.data(), q->rows(), q->cols(), w.data()), 0u);
  const std::string got = kt2n::sha256_hex(w.data(), w.size() * 2);
  EXPECT_EQ(got, expected_.at_pointer("/unpack/sha256_bf16")->as_string());
  EXPECT_EQ(got, q->src_sha256);
  // q_proj has all-zero groups (scale 0): they must decode to +0.0.
  size_t zero_scales = 0;
  for (auto s : scales) zero_scales += s == 0 ? 1 : 0;
  EXPECT_EQ(zero_scales, 3u);
}

TEST_F(Expected, ZeroCountsAndPlusZeroUnpack) {
  const EngineFile f = EngineFile::open(data_path("tiny.kt2n"));
  const auto& z = *expected_.get("zeros");
  const auto* up = f.find(z.get("tensor")->as_string());
  ASSERT_NE(up, nullptr);
  EXPECT_EQ(up->zeros, z.get("zeros")->as_int());
  EXPECT_EQ(up->neg_zero, z.get("neg_zero")->as_int());
  // The reference unpack ignores zero signs (production semantics): every zero is +0.0.
  const auto codes = copy_segment<int32_t>(f, up->codes);
  const auto scales = copy_segment<uint16_t>(f, up->scales);
  std::vector<uint16_t> w(static_cast<size_t>(up->rows() * up->cols()));
  EXPECT_EQ(kt2n::ref::unpack_ternary(codes.data(), scales.data(), up->rows(), up->cols(), w.data()), 0u);
  EXPECT_EQ(kt2n::sha256_hex(w.data(), w.size() * 2), z.get("sha256_plus_zero")->as_string());
  EXPECT_NE(kt2n::sha256_hex(w.data(), w.size() * 2), up->src_sha256);  // the source has -0.0s
}

TEST_F(Expected, GemvMatchesNumpySequentialOrder) {
  const EngineFile f = EngineFile::open(data_path("tiny.kt2n"));
  const auto& g = *expected_.get("gemv");
  const auto* up = f.find(g.get("tensor")->as_string());
  ASSERT_NE(up, nullptr);
  const int64_t M = g.get("M")->as_int(), N = g.get("N")->as_int(), K = g.get("K")->as_int();
  ASSERT_EQ(N, up->rows());
  ASSERT_EQ(K, up->cols());
  std::vector<uint16_t> x;
  for (const auto& v : g.get("x_bf16")->items()) x.push_back(static_cast<uint16_t>(v.as_int()));
  ASSERT_EQ(x.size(), static_cast<size_t>(M * K));
  const auto codes = copy_segment<int32_t>(f, up->codes);
  const auto scales = copy_segment<uint16_t>(f, up->scales);
  std::vector<float> y(static_cast<size_t>(M * N));
  kt2n::ref::ternary_gemv_ref(codes.data(), scales.data(), N, K, x.data(), M, y.data());
  const auto& want = g.get("y_f32_bits")->items();
  ASSERT_EQ(want.size(), y.size());
  for (size_t i = 0; i < y.size(); ++i) {
    ASSERT_EQ(std::bit_cast<uint32_t>(y[i]), static_cast<uint32_t>(want[i].as_int())) << i;
  }
}

TEST_F(Expected, RmsNormMatchesNumpyHfSemantics) {
  const EngineFile f = EngineFile::open(data_path("tiny.kt2n"));
  const auto& r = *expected_.get("rmsnorm");
  const auto* wt = f.find(r.get("weight")->as_string());
  ASSERT_NE(wt, nullptr);
  const int64_t M = r.get("M")->as_int(), D = r.get("D")->as_int();
  const auto w = copy_segment<uint16_t>(f, wt->data);
  ASSERT_EQ(w.size(), static_cast<size_t>(D));
  std::vector<uint16_t> x;
  for (const auto& v : r.get("x_bf16")->items()) x.push_back(static_cast<uint16_t>(v.as_int()));
  std::vector<uint16_t> y(x.size());
  kt2n::ref::rmsnorm_ref(x.data(), w.data(), M, D, static_cast<float>(r.get("eps")->as_double()), y.data());
  const auto& want = r.get("y_bf16")->items();
  ASSERT_EQ(want.size(), y.size());
  for (size_t i = 0; i < y.size(); ++i) ASSERT_EQ(y[i], static_cast<uint16_t>(want[i].as_int())) << i;
}

// ---------------------------------------------------------------------------
// Malformed files: copy tiny.kt2n, damage it, expect a FormatError.
// ---------------------------------------------------------------------------
class Malformed : public ::testing::Test {
 protected:
  void SetUp() override { bytes_ = kt2n::test::read_bytes(data_path("tiny.kt2n")); }
  void expect_format_error(const std::vector<uint8_t>& b, const std::string& needle) {
    const std::string p = data_path(std::string("malformed_") +
                                    ::testing::UnitTest::GetInstance()->current_test_info()->name() + ".kt2n");
    kt2n::test::write_bytes(p, b);
    try {
      EngineFile::open(p);
      ADD_FAILURE() << "expected FormatError containing '" << needle << "'";
    } catch (const FormatError& e) {
      EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
    }
    std::remove(p.c_str());
  }
  void set_u32(size_t off, uint32_t v) { std::memcpy(bytes_.data() + off, &v, 4); }
  void set_u64(size_t off, uint64_t v) { std::memcpy(bytes_.data() + off, &v, 8); }
  uint64_t get_u64(size_t off) const {
    uint64_t v = 0;
    std::memcpy(&v, bytes_.data() + off, 8);
    return v;
  }
  // Replace the first occurrence of `from` in the metadata with a same-length `to`.
  void patch_meta(const std::string& from, const std::string& to) {
    ASSERT_EQ(from.size(), to.size());
    const uint64_t meta_off = get_u64(16), meta_len = get_u64(24);
    std::string meta(reinterpret_cast<const char*>(bytes_.data() + meta_off), meta_len);
    const size_t at = meta.find(from);
    ASSERT_NE(at, std::string::npos) << from;
    std::memcpy(bytes_.data() + meta_off + at, to.data(), to.size());
  }
  std::vector<uint8_t> bytes_;
};

TEST_F(Malformed, BadMagic) {
  bytes_[0] = 'X';
  expect_format_error(bytes_, "magic");
}
TEST_F(Malformed, BadVersion) {
  set_u32(8, 2);
  expect_format_error(bytes_, "version");
}
TEST_F(Malformed, NonZeroFlags) {
  set_u32(12, 1);
  expect_format_error(bytes_, "flags");
}
TEST_F(Malformed, NonZeroReserved) {
  bytes_[63] = 1;
  expect_format_error(bytes_, "reserved");
}
TEST_F(Malformed, Truncated) {
  bytes_.pop_back();
  expect_format_error(bytes_, "file_len");
}
TEST_F(Malformed, TooSmall) {
  bytes_.resize(10);
  expect_format_error(bytes_, "smaller than");
}
TEST_F(Malformed, MetaOutOfBounds) {
  set_u64(24, get_u64(48) + 1);
  expect_format_error(bytes_, "meta region");
}
TEST_F(Malformed, DataOutOfBounds) {
  set_u64(40, get_u64(40) + 1);
  expect_format_error(bytes_, "data region");
}
TEST_F(Malformed, MetaNotJson) {
  bytes_[get_u64(16)] = 'x';
  expect_format_error(bytes_, "metadata");
}
TEST_F(Malformed, MisalignedSegment) {
  patch_meta("\"off\":4096", "\"off\":4100");
  expect_format_error(bytes_, "not a multiple of 4096");
}
TEST_F(Malformed, UnknownKind) {
  patch_meta("\"kind\":\"dense\"", "\"kind\":\"dunce\"");
  expect_format_error(bytes_, "unknown kind");
}
TEST_F(Malformed, UnknownDtype) {
  patch_meta("\"dtype\":\"f32\"", "\"dtype\":\"q32\"");
  expect_format_error(bytes_, "unknown dtype");
}
TEST_F(Malformed, WrongFormatName) {
  patch_meta("\"format\":\"kt2n\"", "\"format\":\"kt3n\"");
  expect_format_error(bytes_, "format");
}
TEST_F(Malformed, DenseDtypeDiffersFromLogical) {
  // spk_proj.0.bias is f32 [16]; its data segment (first "dtype" in the sorted
  // entry) now claims u32, same item size. Used to pass with a verify warning.
  patch_meta("\"dtype\":\"f32\"", "\"dtype\":\"u32\"");
  expect_format_error(bytes_, "like the logical tensor");
}
TEST_F(Malformed, DenseShapeDiffersFromLogical) {
  // model.embed_tokens.weight is bf16 [32,64]; the data segment says [64,32] (same numel).
  patch_meta("\"shape\":[32,64]", "\"shape\":[64,32]");
  expect_format_error(bytes_, "like the logical tensor");
}
TEST_F(Malformed, DuplicateMetadataKey) {
  // The top-level "source" key renamed to a second "format": the first copy is
  // "kt2n" (what get() saw), the last is an object (what json.loads keeps).
  patch_meta("\"source\":", "\"format\":");
  expect_format_error(bytes_, "duplicate object key");
}
TEST_F(Malformed, ZeroSignsWrongLength) {
  // zeros is recorded; claiming 8 more zeros makes zero_signs one byte short.
  const EngineFile f = EngineFile::open(data_path("tiny.kt2n"));
  const auto* up = f.find(kUp);
  const std::string from = "\"zeros\":" + std::to_string(up->zeros);
  const std::string to = "\"zeros\":" + std::to_string(up->zeros + 8);
  if (from.size() != to.size()) GTEST_SKIP() << "digit count changes";
  patch_meta(from, to);
  expect_format_error(bytes_, "zero_signs");
}
TEST_F(Malformed, BadShaHex) {
  const EngineFile f = EngineFile::open(data_path("tiny.kt2n"));
  const std::string sha = f.find(kUp)->src_sha256;
  std::string upper = sha;
  for (auto& c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  if (upper == sha) GTEST_SKIP() << "sha has no letters";
  patch_meta(sha, upper);
  expect_format_error(bytes_, "lowercase hex");
}

TEST(EngineFileIo, MissingFileIsIoError) {
  EXPECT_THROW(EngineFile::open(data_path("does_not_exist.kt2n")), kt2n::IoError);
}

TEST(EngineFileDtypes, NamesAndSizes) {
  EXPECT_EQ(kt2n::parse_dtype("bf16"), kt2n::DType::kBF16);
  EXPECT_EQ(kt2n::parse_dtype("BF16"), kt2n::DType::kBF16);
  EXPECT_EQ(kt2n::parse_dtype("i32"), kt2n::DType::kI32);
  EXPECT_EQ(kt2n::parse_dtype("F32"), kt2n::DType::kF32);
  EXPECT_FALSE(kt2n::parse_dtype("float32").has_value());
  EXPECT_EQ(kt2n::dtype_size(kt2n::DType::kBF16), 2u);
  EXPECT_EQ(kt2n::dtype_size(kt2n::DType::kI64), 8u);
  EXPECT_EQ(kt2n::dtype_name(kt2n::DType::kU16), "u16");
  EXPECT_EQ(kt2n::kind_name(TensorKind::kTernary2), "ternary2");
}

}  // namespace
