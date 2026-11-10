/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#include "humanoid_mpc_validation/io/JsonValue.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid::validation {
namespace {

/** The fewest significant digits formatNumber() tries, DBL_DIG, and the most,
 * DBL_DECIMAL_DIG, which reads back any double. */
constexpr int kMinRoundTripDigits = std::numeric_limits<double>::digits10;
constexpr int kMaxRoundTripDigits = std::numeric_limits<double>::max_digits10;

/** The first character a JSON string may hold unescaped: U+0000 to U+001F are
 * control characters (RFC 8259, section 7). */
constexpr int kFirstNonControlCharacter = 0x20;

/** The spaces each nesting level of a serialized array or object is indented
 * by. */
constexpr int kIndentStep = 2;

// UTF-8 (RFC 3629, section 3): the first code point that needs two, three and
// four bytes, the marker of the lead byte of each length, and the continuation
// bytes, which carry 6 bits each under their marker.
constexpr int kFirstTwoByteCodePoint = 0x80;
constexpr int kFirstThreeByteCodePoint = 0x800;
constexpr int kFirstFourByteCodePoint = 0x10000;
constexpr int kTwoByteLeadMarker = 0xc0;
constexpr int kThreeByteLeadMarker = 0xe0;
constexpr int kFourByteLeadMarker = 0xf0;
constexpr int kContinuationMarker = 0x80;
constexpr int kContinuationPayloadMask = 0x3f;
constexpr int kContinuationPayloadBits = 6;
constexpr int kTwoContinuationsPayloadBits = 2 * kContinuationPayloadBits;
constexpr int kThreeContinuationsPayloadBits = 3 * kContinuationPayloadBits;

// A Unicode escape (RFC 8259, section 7): its prefix and four hexadecimal
// digits of 4 bits each, the letters a to f worth 10 to 15.
constexpr absl::string_view kUnicodeEscapePrefix = "\\u";
constexpr int kHexQuadDigits = 4;
constexpr int kBitsPerHexDigit = 4;
constexpr int kHexLetterOffset = 10;

// UTF-16 surrogate pairs: a high surrogate in [0xd800, 0xdc00) and a low one in
// [0xdc00, 0xe000) encode the code point 0x10000 + ((high - 0xd800) << 10) +
// (low - 0xdc00).
constexpr int kHighSurrogateBegin = 0xd800;
constexpr int kLowSurrogateBegin = 0xdc00;
constexpr int kLowSurrogateEnd = 0xe000;
constexpr int kSurrogatePairBase = 0x10000;
constexpr int kSurrogatePayloadBits = 10;

/** The fewest significant digits (15 to 17) that read back to exactly `value`.
 */
std::string formatNumber(double value) {
  std::string text;
  for (int precision = kMinRoundTripDigits; precision <= kMaxRoundTripDigits;
       ++precision) {
    text = absl::StrFormat("%.*g", precision, value);
    if (std::strtod(text.c_str(), /*endptr=*/nullptr) == value) break;
  }
  return text;
}

void appendEscaped(std::string& out, absl::string_view text) {
  out.push_back('"');
  for (const char c : text) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\t':
        out += "\\t";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      default:
        if (static_cast<unsigned char>(c) < kFirstNonControlCharacter) {
          absl::StrAppendFormat(
              &out, "\\u%04x",
              static_cast<unsigned int>(static_cast<unsigned char>(c)));
        } else {
          out.push_back(c);
        }
    }
  }
  out.push_back('"');
}

void appendUtf8(std::string& out, uint32_t codePoint) {
  if (codePoint < kFirstTwoByteCodePoint) {
    out.push_back(static_cast<char>(codePoint));
  } else if (codePoint < kFirstThreeByteCodePoint) {
    out.push_back(static_cast<char>(kTwoByteLeadMarker |
                                    (codePoint >> kContinuationPayloadBits)));
    out.push_back(static_cast<char>(kContinuationMarker |
                                    (codePoint & kContinuationPayloadMask)));
  } else if (codePoint < kFirstFourByteCodePoint) {
    out.push_back(static_cast<char>(
        kThreeByteLeadMarker | (codePoint >> kTwoContinuationsPayloadBits)));
    out.push_back(static_cast<char>(
        kContinuationMarker |
        ((codePoint >> kContinuationPayloadBits) & kContinuationPayloadMask)));
    out.push_back(static_cast<char>(kContinuationMarker |
                                    (codePoint & kContinuationPayloadMask)));
  } else {
    out.push_back(static_cast<char>(
        kFourByteLeadMarker | (codePoint >> kThreeContinuationsPayloadBits)));
    out.push_back(static_cast<char>(
        kContinuationMarker | ((codePoint >> kTwoContinuationsPayloadBits) &
                               kContinuationPayloadMask)));
    out.push_back(static_cast<char>(
        kContinuationMarker |
        ((codePoint >> kContinuationPayloadBits) & kContinuationPayloadMask)));
    out.push_back(static_cast<char>(kContinuationMarker |
                                    (codePoint & kContinuationPayloadMask)));
  }
}

/** Recursive-descent parser over one document. */
class Parser {
 public:
  explicit Parser(absl::string_view text) : text_(text) {}

  absl::StatusOr<JsonValue> parseDocument() {
    absl::StatusOr<JsonValue> value = parseValue(/*depth=*/0);
    if (!value.ok()) return value;
    skipWhitespace();
    if (offset_ != text_.size()) {
      return error("unexpected text after the document");
    }
    return value;
  }

 private:
  static constexpr int kMaxDepth = 256;

  absl::Status error(absl::string_view what) const {
    return absl::InvalidArgumentError(
        absl::StrCat("[JsonValue::parse] ", what, " at byte ", offset_));
  }

  void skipWhitespace() {
    while (offset_ < text_.size() &&
           (text_[offset_] == ' ' || text_[offset_] == '\n' ||
            text_[offset_] == '\t' || text_[offset_] == '\r')) {
      ++offset_;
    }
  }

  bool consumeLiteral(absl::string_view literal) {
    if (text_.substr(offset_, literal.size()) != literal) return false;
    offset_ += literal.size();
    return true;
  }

  absl::StatusOr<JsonValue> parseValue(int depth) {
    if (depth > kMaxDepth) return error("nesting too deep");
    skipWhitespace();
    if (offset_ >= text_.size()) return error("unexpected end of the document");
    const char c = text_[offset_];
    if (c == '{') return parseObject(depth);
    if (c == '[') return parseArray(depth);
    if (c == '"') {
      absl::StatusOr<std::string> text = parseString();
      if (!text.ok()) return text.status();
      return JsonValue::string(*std::move(text));
    }
    if (consumeLiteral("true")) return JsonValue::boolean(/*value=*/true);
    if (consumeLiteral("false")) return JsonValue::boolean(/*value=*/false);
    if (consumeLiteral("null")) return JsonValue();
    if (c == '-' || (c >= '0' && c <= '9')) return parseNumber();
    return error(
        absl::StrCat("unexpected character '", std::string(1, c), "'"));
  }

  absl::StatusOr<JsonValue> parseNumber() {
    const size_t start = offset_;
    if (text_[offset_] == '-') ++offset_;
    const size_t digitsStart = offset_;
    while (offset_ < text_.size() && text_[offset_] >= '0' &&
           text_[offset_] <= '9') {
      ++offset_;
    }
    if (offset_ == digitsStart) return error("a number without digits");
    if (text_[digitsStart] == '0' && offset_ - digitsStart > 1) {
      return error("a number with a leading zero");
    }
    if (offset_ < text_.size() && text_[offset_] == '.') {
      ++offset_;
      const size_t fractionStart = offset_;
      while (offset_ < text_.size() && text_[offset_] >= '0' &&
             text_[offset_] <= '9') {
        ++offset_;
      }
      if (offset_ == fractionStart) {
        return error("a number without digits after the decimal point");
      }
    }
    if (offset_ < text_.size() &&
        (text_[offset_] == 'e' || text_[offset_] == 'E')) {
      ++offset_;
      if (offset_ < text_.size() &&
          (text_[offset_] == '+' || text_[offset_] == '-')) {
        ++offset_;
      }
      const size_t exponentStart = offset_;
      while (offset_ < text_.size() && text_[offset_] >= '0' &&
             text_[offset_] <= '9') {
        ++offset_;
      }
      if (offset_ == exponentStart) {
        return error("a number without exponent digits");
      }
    }
    const std::string literal(text_.substr(start, offset_ - start));
    return JsonValue::number(std::strtod(literal.c_str(), /*endptr=*/nullptr));
  }

  absl::StatusOr<uint32_t> parseHexQuad() {
    if (offset_ + kHexQuadDigits > text_.size()) {
      return error("a truncated \\u escape");
    }
    uint32_t value = 0;
    for (int i = 0; i < kHexQuadDigits; ++i) {
      const char c = text_[offset_++];
      value <<= kBitsPerHexDigit;
      if (c >= '0' && c <= '9') {
        value |= static_cast<uint32_t>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        value |= static_cast<uint32_t>(c - 'a' + kHexLetterOffset);
      } else if (c >= 'A' && c <= 'F') {
        value |= static_cast<uint32_t>(c - 'A' + kHexLetterOffset);
      } else {
        return error("a \\u escape that is not hexadecimal");
      }
    }
    return value;
  }

  absl::StatusOr<std::string> parseString() {
    ++offset_;  // the opening quote
    std::string out;
    while (true) {
      if (offset_ >= text_.size()) return error("an unterminated string");
      const char c = text_[offset_++];
      if (c == '"') return out;
      if (static_cast<unsigned char>(c) < kFirstNonControlCharacter) {
        return error("a control character in a string");
      }
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (offset_ >= text_.size()) return error("an unterminated escape");
      const char escape = text_[offset_++];
      switch (escape) {
        case '"':
        case '\\':
        case '/':
          out.push_back(escape);
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'u': {
          absl::StatusOr<uint32_t> codePoint = parseHexQuad();
          if (!codePoint.ok()) return codePoint.status();
          if (*codePoint >= kHighSurrogateBegin &&
              *codePoint < kLowSurrogateBegin &&
              text_.substr(offset_, kUnicodeEscapePrefix.size()) ==
                  kUnicodeEscapePrefix) {
            offset_ += kUnicodeEscapePrefix.size();
            const absl::StatusOr<uint32_t> low = parseHexQuad();
            if (!low.ok()) return low.status();
            if (*low < kLowSurrogateBegin || *low >= kLowSurrogateEnd) {
              return error("an unpaired surrogate");
            }
            *codePoint =
                kSurrogatePairBase +
                ((*codePoint - kHighSurrogateBegin) << kSurrogatePayloadBits) +
                (*low - kLowSurrogateBegin);
          }
          appendUtf8(out, *codePoint);
          break;
        }
        default:
          return error(absl::StrCat("an unknown escape '\\",
                                    std::string(1, escape), "'"));
      }
    }
  }

  absl::StatusOr<JsonValue> parseArray(int depth) {
    ++offset_;  // [
    JsonValue array = JsonValue::array();
    skipWhitespace();
    if (offset_ < text_.size() && text_[offset_] == ']') {
      ++offset_;
      return array;
    }
    while (true) {
      absl::StatusOr<JsonValue> element = parseValue(depth + 1);
      if (!element.ok()) return element;
      array.append(*std::move(element));
      skipWhitespace();
      if (offset_ >= text_.size()) return error("an unterminated array");
      if (text_[offset_] == ',') {
        ++offset_;
        continue;
      }
      if (text_[offset_] == ']') {
        ++offset_;
        return array;
      }
      return error("expected ',' or ']' in an array");
    }
  }

  absl::StatusOr<JsonValue> parseObject(int depth) {
    ++offset_;  // {
    JsonValue object = JsonValue::object();
    skipWhitespace();
    if (offset_ < text_.size() && text_[offset_] == '}') {
      ++offset_;
      return object;
    }
    while (true) {
      skipWhitespace();
      if (offset_ >= text_.size() || text_[offset_] != '"') {
        return error("expected a member name");
      }
      absl::StatusOr<std::string> key = parseString();
      if (!key.ok()) return key.status();
      if (object.find(*key) != nullptr) {
        return error(absl::StrCat("a duplicate member '", *key, "'"));
      }
      skipWhitespace();
      if (offset_ >= text_.size() || text_[offset_] != ':') {
        return error("expected ':' after a member name");
      }
      ++offset_;
      absl::StatusOr<JsonValue> value = parseValue(depth + 1);
      if (!value.ok()) return value;
      object.set(*key, *std::move(value));
      skipWhitespace();
      if (offset_ >= text_.size()) return error("an unterminated object");
      if (text_[offset_] == ',') {
        ++offset_;
        continue;
      }
      if (text_[offset_] == '}') {
        ++offset_;
        return object;
      }
      return error("expected ',' or '}' in an object");
    }
  }

  // NOLINTNEXTLINE(totw-view-member): a Parser lives only inside JsonValue::parse(), whose argument outlives it.
  absl::string_view text_;
  size_t offset_ = 0;
};

}  // namespace

JsonValue JsonValue::boolean(bool value) {
  JsonValue json;
  json.type_ = Type::kBool;
  json.bool_ = value;
  return json;
}

JsonValue JsonValue::number(double value) {
  JsonValue json;
  json.type_ = Type::kNumber;
  json.number_ = value;
  return json;
}

JsonValue JsonValue::optionalNumber(std::optional<double> value) {
  return value.has_value() ? number(*value) : JsonValue();
}

JsonValue JsonValue::string(std::string value) {
  JsonValue json;
  json.type_ = Type::kString;
  json.string_ = std::move(value);
  return json;
}

JsonValue JsonValue::array() {
  JsonValue json;
  json.type_ = Type::kArray;
  return json;
}

JsonValue JsonValue::object() {
  JsonValue json;
  json.type_ = Type::kObject;
  return json;
}

bool JsonValue::asBool() const {
  CHECK(type_ == Type::kBool) << "[JsonValue] not a bool";
  return bool_;
}

double JsonValue::asNumber() const {
  CHECK(type_ == Type::kNumber) << "[JsonValue] not a number";
  return number_;
}

const std::string& JsonValue::asString() const {
  CHECK(type_ == Type::kString) << "[JsonValue] not a string";
  return string_;
}

size_t JsonValue::size() const {
  return (type_ == Type::kArray || type_ == Type::kObject) ? elements_.size()
                                                           : 0;
}

const JsonValue& JsonValue::at(size_t index) const {
  CHECK(type_ == Type::kArray) << "[JsonValue] not an array";
  CHECK_LT(index, elements_.size());
  return elements_[index];
}

const std::string& JsonValue::keyAt(size_t index) const {
  CHECK(type_ == Type::kObject) << "[JsonValue] not an object";
  CHECK_LT(index, keys_.size());
  return keys_[index];
}

const JsonValue& JsonValue::valueAt(size_t index) const {
  CHECK(type_ == Type::kObject) << "[JsonValue] not an object";
  CHECK_LT(index, elements_.size());
  return elements_[index];
}

JsonValue& JsonValue::set(absl::string_view key, JsonValue value) {
  CHECK(type_ == Type::kObject)
      << "[JsonValue] set() on a value that is not an object";
  for (size_t i = 0; i < keys_.size(); ++i) {
    if (keys_[i] == key) {
      elements_[i] = std::move(value);
      return elements_[i];
    }
  }
  keys_.emplace_back(key);
  elements_.push_back(std::move(value));
  return elements_.back();
}

JsonValue& JsonValue::append(JsonValue value) {
  CHECK(type_ == Type::kArray)
      << "[JsonValue] append() on a value that is not an array";
  elements_.push_back(std::move(value));
  return elements_.back();
}

const JsonValue* absl_nullable JsonValue::find(absl::string_view key) const {
  if (type_ != Type::kObject) return nullptr;
  for (size_t i = 0; i < keys_.size(); ++i) {
    if (keys_[i] == key) return &elements_[i];
  }
  return nullptr;
}

const JsonValue* absl_nullable JsonValue::findPath(
    absl::string_view dottedPath) const {
  const JsonValue* absl_nullable value = this;
  for (const absl::string_view key : absl::StrSplit(dottedPath, '.')) {
    value = value->find(key);
    if (value == nullptr) return nullptr;
  }
  return value;
}

std::string JsonValue::serialize() const {
  std::string out;
  serializeTo(out, /*indent=*/0);
  return out;
}

void JsonValue::serializeTo(std::string& out, int indent) const {
  switch (type_) {
    case Type::kNull:
      out += "null";
      return;
    case Type::kBool:
      out += bool_ ? "true" : "false";
      return;
    case Type::kNumber:
      out += std::isfinite(number_) ? formatNumber(number_) : "null";
      return;
    case Type::kString:
      appendEscaped(out, string_);
      return;
    case Type::kArray: {
      if (elements_.empty()) {
        out += "[]";
        return;
      }
      // An array of scalars stays on one line; one that holds arrays or objects
      // puts an element on each line.
      bool scalars = true;
      for (const JsonValue& element : elements_) {
        scalars = scalars && !element.isArray() && !element.isObject();
      }
      if (scalars) {
        out.push_back('[');
        for (size_t i = 0; i < elements_.size(); ++i) {
          if (i > 0) out += ", ";
          elements_[i].serializeTo(out, indent);
        }
        out.push_back(']');
        return;
      }
      out += "[\n";
      for (size_t i = 0; i < elements_.size(); ++i) {
        out.append(static_cast<size_t>(indent + kIndentStep), ' ');
        elements_[i].serializeTo(out, indent + kIndentStep);
        out += i + 1 < elements_.size() ? ",\n" : "\n";
      }
      out.append(static_cast<size_t>(indent), ' ');
      out.push_back(']');
      return;
    }
    case Type::kObject: {
      if (elements_.empty()) {
        out += "{}";
        return;
      }
      out += "{\n";
      for (size_t i = 0; i < elements_.size(); ++i) {
        out.append(static_cast<size_t>(indent + kIndentStep), ' ');
        appendEscaped(out, keys_[i]);
        out += ": ";
        elements_[i].serializeTo(out, indent + kIndentStep);
        out += i + 1 < elements_.size() ? ",\n" : "\n";
      }
      out.append(static_cast<size_t>(indent), ' ');
      out.push_back('}');
      return;
    }
  }
}

absl::StatusOr<JsonValue> JsonValue::parse(absl::string_view text) {
  Parser parser(text);
  return parser.parseDocument();
}

bool operator==(const JsonValue& lhs, const JsonValue& rhs) {
  if (lhs.type_ != rhs.type_) return false;
  switch (lhs.type_) {
    case JsonValue::Type::kNull:
      return true;
    case JsonValue::Type::kBool:
      return lhs.bool_ == rhs.bool_;
    case JsonValue::Type::kNumber:
      return lhs.number_ == rhs.number_;
    case JsonValue::Type::kString:
      return lhs.string_ == rhs.string_;
    case JsonValue::Type::kArray:
      return lhs.elements_ == rhs.elements_;
    case JsonValue::Type::kObject:
      return lhs.keys_ == rhs.keys_ && lhs.elements_ == rhs.elements_;
  }
  return false;
}

absl::StatusOr<JsonValue> readJsonFile(absl::string_view path) {
  std::ifstream file{std::filesystem::path(path), std::ios::binary};
  if (!file.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("[readJsonFile] cannot read '", path, "'"));
  }
  const std::string text((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
  absl::StatusOr<JsonValue> value = JsonValue::parse(text);
  if (!value.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat(path, ": ", value.status().message()));
  }
  return value;
}

absl::Status writeJsonFile(absl::string_view path, const JsonValue& value) {
  const std::filesystem::path parent =
      std::filesystem::path(path).parent_path();
  std::error_code error;
  if (!parent.empty()) std::filesystem::create_directories(parent, error);
  std::ofstream file{std::filesystem::path(path),
                     std::ios::binary | std::ios::trunc};
  if (!file.is_open()) {
    return absl::InternalError(
        absl::StrCat("[writeJsonFile] cannot write '", path, "'"));
  }
  file << value.serialize() << "\n";
  file.close();
  if (!file) {
    return absl::InternalError(
        absl::StrCat("[writeJsonFile] writing '", path, "' failed"));
  }
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid::validation
