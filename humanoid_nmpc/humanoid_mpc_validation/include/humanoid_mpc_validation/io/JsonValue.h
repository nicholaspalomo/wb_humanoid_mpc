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

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid::validation {

/**
 * A JSON document (RFC 8259), small enough to need no third-party library: the closed-loop metrics and the solve
 * benchmark write their results with it, and the schema test and the band comparison read them back.
 *
 * Objects keep their members in insertion order, so a file written twice from the same values is byte-identical and
 * diffs well. Numbers are doubles, written with the fewest significant digits that read back to the same double; a
 * number that is not finite has no JSON form and is written as null.
 */
class JsonValue {
 public:
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

  /** null. */
  JsonValue() = default;

  static JsonValue boolean(bool value);
  static JsonValue number(double value);
  /** number(*value), or null when there is no value. */
  static JsonValue optionalNumber(std::optional<double> value);
  static JsonValue string(std::string value);
  static JsonValue array();
  static JsonValue object();

  Type type() const { return type_; }
  bool isNull() const { return type_ == Type::kNull; }
  bool isBool() const { return type_ == Type::kBool; }
  bool isNumber() const { return type_ == Type::kNumber; }
  bool isString() const { return type_ == Type::kString; }
  bool isArray() const { return type_ == Type::kArray; }
  bool isObject() const { return type_ == Type::kObject; }

  /** The value of a bool, number or string; CHECK-fails on another type. */
  bool asBool() const;
  double asNumber() const;
  const std::string& asString() const;

  /** Elements of an array or members of an object; 0 for the other types. */
  size_t size() const;
  /** Element `index` of an array; CHECK-fails out of range or on another type. */
  const JsonValue& at(size_t index) const;
  /** Key and value of member `index` of an object, in insertion order; CHECK-fails out of range or on another type. */
  const std::string& keyAt(size_t index) const;
  const JsonValue& valueAt(size_t index) const;

  /** Sets member `key` of an object (CHECK-fails on another type): replaces it in place, or appends it. Returns it. */
  JsonValue& set(absl::string_view key, JsonValue value);
  /** Appends to an array (CHECK-fails on another type). Returns the new element. */
  JsonValue& append(JsonValue value);

  /** Member `key` of an object; nullptr when absent or when this is not an object. */
  const JsonValue* absl_nullable find(absl::string_view key) const;
  /** The value at a dotted path of object keys ("solve_time_ms.total.p99"); nullptr when any part is absent. */
  const JsonValue* absl_nullable findPath(absl::string_view dottedPath) const;

  /** The document with two-space indentation, without a trailing newline. */
  std::string serialize() const;

  /** Parses one JSON document; InvalidArgument with the byte offset of the first error. */
  static absl::StatusOr<JsonValue> parse(absl::string_view text);

  /** Structural equality: same types, numbers equal as doubles, object members equal in the same order. */
  bool operator==(const JsonValue& other) const;
  bool operator!=(const JsonValue& other) const { return !(*this == other); }

 private:
  void serializeTo(std::string& out, int indent) const;

  Type type_ = Type::kNull;
  bool bool_ = false;
  double number_ = 0.0;
  std::string string_;
  std::vector<JsonValue> elements_;  // array elements, or object member values
  std::vector<std::string> keys_;    // object member keys, parallel to elements_
};

/** JsonValue::parse() of the file at `path`; NotFound when it cannot be read. */
absl::StatusOr<JsonValue> readJsonFile(const std::string& path);

/** `value` serialized into `path` with a final newline, creating its directory; Internal when it cannot be written. */
absl::Status writeJsonFile(const std::string& path, const JsonValue& value);

}  // namespace ocs2::humanoid::validation
