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

#include "nproto/Conversions.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/repeated_ptr_field.h"

namespace nproto::internal {
namespace {

bool isNameCharacter(char character) {
  return absl::ascii_isalnum(static_cast<unsigned char>(character)) || character == '_';
}

// Whether `message` starts with a field path followed by ": ", such as "kind: ...", "patch.kind: ..." or
// "positions[\"a b\"]: ...", rather than with the reason of an error, "7 is not ...". A path is field names joined by
// '.', each followed by any number of [index] or ["key"] segments.
bool startsWithPath(absl::string_view message) {
  size_t position = 0;
  bool expectName = message.empty() || message.front() != '[';
  while (position < message.size()) {
    if (expectName) {
      const size_t start = position;
      while (position < message.size() && isNameCharacter(message[position])) {
        ++position;
      }
      if (position == start || absl::ascii_isdigit(static_cast<unsigned char>(message[start]))) {
        return false;
      }
      expectName = false;
    } else if (message[position] == '[') {
      ++position;
      if (position < message.size() && message[position] == '"') {
        ++position;
        while (position < message.size() && message[position] != '"') {
          position += message[position] == '\\' ? 2 : 1;
        }
        ++position;
      } else {
        while (position < message.size() && message[position] != ']') {
          ++position;
        }
      }
      if (position >= message.size() || message[position] != ']') {
        return false;
      }
      ++position;
    } else if (message[position] == '.') {
      ++position;
      expectName = true;
    } else {
      return absl::StartsWith(message.substr(position), ": ");
    }
  }
  return false;
}

// `prefix` joined to the message of `status`: "[2]" + "[1].kind: ..." -> "[2][1].kind: ...", + "kind: ..." ->
// "[2].kind: ...", + "7 is not ..." -> "[2]: 7 is not ...".
absl::Status prefixed(absl::string_view prefix, const absl::Status& status) {
  const absl::string_view message = status.message();
  absl::string_view separator = ": ";
  if (!message.empty() && message.front() == '[') {
    separator = "";
  } else if (startsWithPath(message)) {
    separator = ".";
  }
  return absl::Status(status.code(), absl::StrCat(prefix, separator, message));
}

}  // namespace

absl::Status UnknownEnumValueError(int32_t number, absl::string_view enumName) {
  return absl::InvalidArgumentError(absl::StrCat(number, " is not a value of ", enumName));
}

absl::Status AnnotateError(absl::string_view field, const absl::Status& status) {
  return prefixed(field, status);
}

absl::Status AnnotateElement(int index, const absl::Status& status) {
  return prefixed(absl::StrCat("[", index, "]"), status);
}

absl::Status AnnotateKey(absl::string_view formattedKey, const absl::Status& status) {
  return prefixed(absl::StrCat("[", formattedKey, "]"), status);
}

std::string FormatKey(const std::string& key) {
  return absl::StrCat("\"", absl::CEscape(key), "\"");
}

std::string FormatKey(bool key) {
  return key ? "true" : "false";
}

void StringsFromProto(const google::protobuf::RepeatedPtrField<std::string>& proto, std::vector<std::string>* absl_nonnull value) {
  value->resize(static_cast<size_t>(proto.size()));
  for (int index = 0; index < proto.size(); ++index) {
    (*value)[static_cast<size_t>(index)] = proto.Get(index);
  }
}

void StringsToProto(const std::vector<std::string>& value, google::protobuf::RepeatedPtrField<std::string>* absl_nonnull proto) {
  ResizeRepeatedPtrField(static_cast<int>(value.size()), proto);
  for (size_t index = 0; index < value.size(); ++index) {
    *proto->Mutable(static_cast<int>(index)) = value[index];
  }
}

}  // namespace nproto::internal
