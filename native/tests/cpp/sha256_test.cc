#include "kt2n/sha256.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

TEST(Sha256, NistVectors) {
  EXPECT_EQ(kt2n::sha256_hex("", 0), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(kt2n::sha256_hex("abc", 3), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  const std::string m448 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  EXPECT_EQ(kt2n::sha256_hex(m448.data(), m448.size()),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  const std::string m896 =
      "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
  EXPECT_EQ(kt2n::sha256_hex(m896.data(), m896.size()),
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
  const std::string million(1000000, 'a');
  EXPECT_EQ(kt2n::sha256_hex(million.data(), million.size()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256, StreamingMatchesOneShot) {
  std::vector<uint8_t> data(1000);
  for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<uint8_t>((i * 131u + 7u) & 0xFFu);
  for (size_t len : {0u, 1u, 55u, 56u, 63u, 64u, 65u, 119u, 120u, 127u, 128u, 129u, 500u, 1000u}) {
    const std::string want = kt2n::sha256_hex(data.data(), len);
    for (size_t chunk : {1u, 3u, 7u, 63u, 64u, 65u, 1000u}) {
      kt2n::Sha256 h;
      for (size_t off = 0; off < len; off += chunk) h.update(data.data() + off, std::min(chunk, len - off));
      EXPECT_EQ(h.hexdigest(), want) << "len " << len << " chunk " << chunk;
    }
  }
}

TEST(Sha256, ReusableAfterFinish) {
  kt2n::Sha256 h;
  h.update("abc", 3);
  const std::string first = h.hexdigest();
  h.update("abc", 3);
  EXPECT_EQ(h.hexdigest(), first);
}

TEST(Sha256, HexValidation) {
  EXPECT_TRUE(kt2n::is_sha256_hex("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
  EXPECT_FALSE(kt2n::is_sha256_hex("E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855"));
  EXPECT_FALSE(kt2n::is_sha256_hex("e3b0"));
  EXPECT_FALSE(kt2n::is_sha256_hex("g3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
}

}  // namespace
