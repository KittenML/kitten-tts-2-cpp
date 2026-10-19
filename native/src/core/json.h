// Minimal JSON reader for engine-file metadata (private to kt2n_core).
// Strict RFC 8259 parsing; integers that fit int64 are kept exactly; duplicate
// object keys are rejected.
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace kt2n::json {

class ParseError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class Value {
 public:
  enum class Type : uint8_t { kNull, kBool, kInt, kDouble, kString, kArray, kObject };

  Value() = default;
  static Value make_null() { return Value(); }
  static Value make_bool(bool b);
  static Value make_int(int64_t i);
  static Value make_double(double d);
  static Value make_string(std::string s);
  static Value make_array(std::vector<Value> items);
  static Value make_object(std::vector<std::pair<std::string, Value>> members);

  Type type() const noexcept { return type_; }
  bool is_null() const noexcept { return type_ == Type::kNull; }
  bool is_bool() const noexcept { return type_ == Type::kBool; }
  bool is_int() const noexcept { return type_ == Type::kInt; }
  bool is_number() const noexcept { return type_ == Type::kInt || type_ == Type::kDouble; }
  bool is_string() const noexcept { return type_ == Type::kString; }
  bool is_array() const noexcept { return type_ == Type::kArray; }
  bool is_object() const noexcept { return type_ == Type::kObject; }

  bool as_bool() const noexcept { return bool_; }
  int64_t as_int() const noexcept { return int_; }
  double as_double() const noexcept { return type_ == Type::kInt ? static_cast<double>(int_) : double_; }
  const std::string& as_string() const noexcept { return string_; }
  const std::vector<Value>& items() const noexcept { return items_; }
  const std::vector<std::pair<std::string, Value>>& members() const noexcept { return members_; }

  // Object member lookup; nullptr if absent or not an object (parse() rejects duplicate keys).
  const Value* get(std::string_view key) const noexcept;
  // RFC 6901 JSON pointer ("" = this value). nullptr if absent / malformed.
  const Value* at_pointer(std::string_view pointer) const;

  // Compact serialization (objects keep member order).
  std::string dump() const;

 private:
  void dump_to(std::string& out) const;

  Type type_ = Type::kNull;
  bool bool_ = false;
  int64_t int_ = 0;
  double double_ = 0.0;
  std::string string_;
  std::vector<Value> items_;
  std::vector<std::pair<std::string, Value>> members_;
};

// Parses a complete JSON document (trailing whitespace allowed, nothing else).
Value parse(std::string_view text);

}  // namespace kt2n::json
