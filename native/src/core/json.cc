#include "json.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <system_error>
#include <unordered_set>

namespace kt2n::json {

Value Value::make_bool(bool b) {
  Value v;
  v.type_ = Type::kBool;
  v.bool_ = b;
  return v;
}
Value Value::make_int(int64_t i) {
  Value v;
  v.type_ = Type::kInt;
  v.int_ = i;
  return v;
}
Value Value::make_double(double d) {
  Value v;
  v.type_ = Type::kDouble;
  v.double_ = d;
  return v;
}
Value Value::make_string(std::string s) {
  Value v;
  v.type_ = Type::kString;
  v.string_ = std::move(s);
  return v;
}
Value Value::make_array(std::vector<Value> items) {
  Value v;
  v.type_ = Type::kArray;
  v.items_ = std::move(items);
  return v;
}
Value Value::make_object(std::vector<std::pair<std::string, Value>> members) {
  Value v;
  v.type_ = Type::kObject;
  v.members_ = std::move(members);
  return v;
}

const Value* Value::get(std::string_view key) const noexcept {
  if (type_ != Type::kObject) return nullptr;
  for (const auto& [k, v] : members_) {
    if (k == key) return &v;
  }
  return nullptr;
}

const Value* Value::at_pointer(std::string_view pointer) const {
  const Value* cur = this;
  if (pointer.empty()) return cur;
  if (pointer.front() != '/') return nullptr;
  size_t pos = 1;
  while (true) {
    const size_t next = pointer.find('/', pos);
    std::string_view raw = pointer.substr(pos, next == std::string_view::npos ? std::string_view::npos : next - pos);
    std::string token;
    token.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
      if (raw[i] == '~') {
        if (i + 1 >= raw.size()) return nullptr;
        if (raw[i + 1] == '0') {
          token.push_back('~');
        } else if (raw[i + 1] == '1') {
          token.push_back('/');
        } else {
          return nullptr;
        }
        ++i;
      } else {
        token.push_back(raw[i]);
      }
    }
    if (cur->is_object()) {
      cur = cur->get(token);
    } else if (cur->is_array()) {
      if (token.empty() || (token.size() > 1 && token[0] == '0')) return nullptr;
      size_t idx = 0;
      auto [p, ec] = std::from_chars(token.data(), token.data() + token.size(), idx);
      if (ec != std::errc() || p != token.data() + token.size() || idx >= cur->items().size()) return nullptr;
      cur = &cur->items()[idx];
    } else {
      return nullptr;
    }
    if (cur == nullptr) return nullptr;
    if (next == std::string_view::npos) return cur;
    pos = next + 1;
  }
}

namespace {

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

class Parser {
 public:
  explicit Parser(std::string_view text) : text_(text) {}

  Value parse_document() {
    skip_ws();
    Value v = parse_value(0);
    skip_ws();
    if (pos_ != text_.size()) fail("trailing characters after JSON value");
    return v;
  }

 private:
  static constexpr int kMaxDepth = 256;

  [[noreturn]] void fail(const std::string& what) const {
    throw ParseError("JSON parse error at byte " + std::to_string(pos_) + ": " + what);
  }

  void skip_ws() {
    while (pos_ < text_.size()) {
      const char c = text_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++pos_;
      } else {
        break;
      }
    }
  }

  bool consume_literal(std::string_view lit) {
    if (text_.substr(pos_, lit.size()) == lit) {
      pos_ += lit.size();
      return true;
    }
    return false;
  }

  Value parse_value(int depth) {
    if (depth > kMaxDepth) fail("nesting too deep");
    if (pos_ >= text_.size()) fail("unexpected end of input");
    const char c = text_[pos_];
    switch (c) {
      case '{': return parse_object(depth);
      case '[': return parse_array(depth);
      case '"': return Value::make_string(parse_string());
      case 't':
        if (consume_literal("true")) return Value::make_bool(true);
        fail("invalid literal");
      case 'f':
        if (consume_literal("false")) return Value::make_bool(false);
        fail("invalid literal");
      case 'n':
        if (consume_literal("null")) return Value::make_null();
        fail("invalid literal");
      default:
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        fail(std::string("unexpected character '") + c + "'");
    }
  }

  Value parse_object(int depth) {
    ++pos_;  // '{'
    std::vector<std::pair<std::string, Value>> members;
    std::unordered_set<std::string> keys;
    skip_ws();
    if (pos_ < text_.size() && text_[pos_] == '}') {
      ++pos_;
      return Value::make_object(std::move(members));
    }
    while (true) {
      skip_ws();
      if (pos_ >= text_.size() || text_[pos_] != '"') fail("expected object key");
      std::string key = parse_string();
      // Duplicate names are ambiguous (get() would see the first, Python's
      // json.loads keeps the last), so they are an error.
      if (!keys.insert(key).second) fail("duplicate object key \"" + key + "\"");
      skip_ws();
      if (pos_ >= text_.size() || text_[pos_] != ':') fail("expected ':'");
      ++pos_;
      skip_ws();
      Value v = parse_value(depth + 1);
      members.emplace_back(std::move(key), std::move(v));
      skip_ws();
      if (pos_ >= text_.size()) fail("unterminated object");
      if (text_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (text_[pos_] == '}') {
        ++pos_;
        return Value::make_object(std::move(members));
      }
      fail("expected ',' or '}'");
    }
  }

  Value parse_array(int depth) {
    ++pos_;  // '['
    std::vector<Value> items;
    skip_ws();
    if (pos_ < text_.size() && text_[pos_] == ']') {
      ++pos_;
      return Value::make_array(std::move(items));
    }
    while (true) {
      skip_ws();
      items.push_back(parse_value(depth + 1));
      skip_ws();
      if (pos_ >= text_.size()) fail("unterminated array");
      if (text_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (text_[pos_] == ']') {
        ++pos_;
        return Value::make_array(std::move(items));
      }
      fail("expected ',' or ']'");
    }
  }

  unsigned parse_hex4() {
    if (pos_ + 4 > text_.size()) fail("truncated \\u escape");
    unsigned v = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = text_[pos_++];
      v <<= 4;
      if (c >= '0' && c <= '9') {
        v |= static_cast<unsigned>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        v |= static_cast<unsigned>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        v |= static_cast<unsigned>(c - 'A' + 10);
      } else {
        fail("bad hex digit in \\u escape");
      }
    }
    return v;
  }

  static void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
      out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }

  std::string parse_string() {
    ++pos_;  // opening quote
    std::string out;
    while (true) {
      if (pos_ >= text_.size()) fail("unterminated string");
      const unsigned char c = static_cast<unsigned char>(text_[pos_++]);
      if (c == '"') return out;
      if (c < 0x20) fail("control character in string");
      if (c != '\\') {
        out.push_back(static_cast<char>(c));
        continue;
      }
      if (pos_ >= text_.size()) fail("unterminated escape");
      const char e = text_[pos_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          uint32_t cp = parse_hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (pos_ + 2 > text_.size() || text_[pos_] != '\\' || text_[pos_ + 1] != 'u') {
              fail("unpaired high surrogate");
            }
            pos_ += 2;
            const uint32_t lo = parse_hex4();
            if (lo < 0xDC00 || lo > 0xDFFF) fail("bad low surrogate");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            fail("unpaired low surrogate");
          }
          append_utf8(out, cp);
          break;
        }
        default:
          fail("bad escape");
      }
    }
  }

  Value parse_number() {
    const size_t start = pos_;
    bool is_integer = true;
    if (text_[pos_] == '-') ++pos_;
    if (pos_ >= text_.size()) fail("bad number");
    if (text_[pos_] == '0') {
      ++pos_;
    } else if (text_[pos_] >= '1' && text_[pos_] <= '9') {
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
    } else {
      fail("bad number");
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      is_integer = false;
      ++pos_;
      if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') fail("bad fraction");
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
    }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      is_integer = false;
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
      if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') fail("bad exponent");
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_;
    }
    const char* first = text_.data() + start;
    const char* last = text_.data() + pos_;
    if (is_integer) {
      int64_t i = 0;
      auto [p, ec] = std::from_chars(first, last, i);
      if (ec == std::errc() && p == last) return Value::make_int(i);
      // Out of int64 range: fall through to double.
    }
    double d = 0.0;
    auto [p, ec] = std::from_chars(first, last, d);
    if (ec != std::errc() || p != last || !std::isfinite(d)) fail("number out of range");
    return Value::make_double(d);
  }

  std::string_view text_;
  size_t pos_ = 0;
};

}  // namespace

void Value::dump_to(std::string& out) const {
  switch (type_) {
    case Type::kNull: out += "null"; break;
    case Type::kBool: out += bool_ ? "true" : "false"; break;
    case Type::kInt: out += std::to_string(int_); break;
    case Type::kDouble: {
      char buf[64];
      auto [p, ec] = std::to_chars(buf, buf + sizeof(buf), double_);
      if (ec != std::errc()) {
        out += "null";
      } else {
        out.append(buf, p);
      }
      break;
    }
    case Type::kString: dump_string(string_, out); break;
    case Type::kArray: {
      out.push_back('[');
      for (size_t i = 0; i < items_.size(); ++i) {
        if (i) out.push_back(',');
        items_[i].dump_to(out);
      }
      out.push_back(']');
      break;
    }
    case Type::kObject: {
      out.push_back('{');
      for (size_t i = 0; i < members_.size(); ++i) {
        if (i) out.push_back(',');
        dump_string(members_[i].first, out);
        out.push_back(':');
        members_[i].second.dump_to(out);
      }
      out.push_back('}');
      break;
    }
  }
}

std::string Value::dump() const {
  std::string out;
  dump_to(out);
  return out;
}

Value parse(std::string_view text) { return Parser(text).parse_document(); }

}  // namespace kt2n::json
