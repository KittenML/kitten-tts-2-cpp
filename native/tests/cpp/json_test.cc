#include "json.h"

#include <gtest/gtest.h>

namespace {

using kt2n::json::parse;
using kt2n::json::ParseError;

TEST(Json, ParsesScalarsAndContainers) {
  const auto v = parse(R"({"a": 1, "b": [true, false, null], "c": {"d": "x"}, "e": -2.5e-3, "big": 9007199254740993})");
  ASSERT_TRUE(v.is_object());
  EXPECT_EQ(v.get("a")->as_int(), 1);
  EXPECT_TRUE(v.get("b")->items()[0].as_bool());
  EXPECT_TRUE(v.get("b")->items()[2].is_null());
  EXPECT_EQ(v.get("c")->get("d")->as_string(), "x");
  EXPECT_DOUBLE_EQ(v.get("e")->as_double(), -2.5e-3);
  EXPECT_EQ(v.get("big")->as_int(), 9007199254740993LL);  // exact, beyond double precision
  EXPECT_EQ(v.get("missing"), nullptr);
}

TEST(Json, StringEscapes) {
  const auto v = parse(R"(["a\"b\\c\/d\n\t", "\u00e9\u20ac", "\ud83d\ude00"])");
  EXPECT_EQ(v.items()[0].as_string(), "a\"b\\c/d\n\t");
  EXPECT_EQ(v.items()[1].as_string(), "\xC3\xA9\xE2\x82\xAC");
  EXPECT_EQ(v.items()[2].as_string(), "\xF0\x9F\x98\x80");
}

TEST(Json, Pointer) {
  const auto v = parse(R"({"config": {"hidden_size": 2048, "a/b": 1, "m~n": 2}, "list": [10, 20]})");
  EXPECT_EQ(v.at_pointer("/config/hidden_size")->as_int(), 2048);
  EXPECT_EQ(v.at_pointer("/config/a~1b")->as_int(), 1);
  EXPECT_EQ(v.at_pointer("/config/m~0n")->as_int(), 2);
  EXPECT_EQ(v.at_pointer("/list/1")->as_int(), 20);
  EXPECT_EQ(v.at_pointer("/list/2"), nullptr);
  EXPECT_EQ(v.at_pointer("/list/01"), nullptr);
  EXPECT_EQ(v.at_pointer("/nope"), nullptr);
  EXPECT_EQ(v.at_pointer("config"), nullptr);
  EXPECT_EQ(v.at_pointer(""), &v);
}

TEST(Json, DumpRoundTrip) {
  const std::string text = R"({"b":[1,2.5,"x\n"],"a":{"c":null,"d":true}})";
  EXPECT_EQ(parse(text).dump(), text);
}

TEST(Json, RejectsMalformed) {
  for (const char* bad : {"", "{", "[1,]", "{\"a\" 1}", "01", "1.", "-", "tru", "\"abc", "\"\\x\"", "[1] 2",
                          "\"\\ud800\"", "{\"a\":1,}", "1e999"}) {
    EXPECT_THROW(parse(bad), ParseError) << bad;
  }
}

TEST(Json, RejectsDuplicateKeys) {
  // get() would return the first copy while Python's json.loads keeps the last.
  EXPECT_THROW(parse(R"({"a": 1, "a": 2})"), ParseError);
  EXPECT_THROW(parse(R"({"x": {"k": 1, "b": 0, "k": 1}})"), ParseError);
  EXPECT_THROW(parse(R"([{"t": [], "t": {}}])"), ParseError);
  EXPECT_THROW(parse(R"({"\u0061": 1, "a": 2})"), ParseError);  // same name after unescaping
  // The same name in different objects is fine.
  const auto v = parse(R"({"a": {"a": 1}, "b": [{"a": 2}, {"a": 3}]})");
  EXPECT_EQ(v.at_pointer("/a/a")->as_int(), 1);
  EXPECT_EQ(v.at_pointer("/b/1/a")->as_int(), 3);
}

TEST(Json, DepthLimit) {
  std::string deep(300, '[');
  deep += std::string(300, ']');
  EXPECT_THROW(parse(deep), ParseError);
}

}  // namespace
