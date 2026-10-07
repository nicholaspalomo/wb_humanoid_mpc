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

#include "nproto/Schema.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "google/protobuf/unknown_field_set.h"

namespace nproto {
namespace {

using google::protobuf::Descriptor;
using google::protobuf::EnumDescriptor;
using google::protobuf::FieldDescriptor;
using google::protobuf::Message;
using google::protobuf::Reflection;

// LINT.IfChange(fnv1a)
constexpr uint64_t kFnvOffsetBasis = 14'695'981'039'346'656'037ULL;
constexpr uint64_t kFnvPrime = 1'099'511'628'211ULL;
// LINT.ThenChange(//tools/nproto/nproto_schema.py:fnv1a)

/** Appends the unknown fields of `message` and of the messages below it, under `path`, to `paths`. */
void collectUnknownFields(const Message& message, const std::string& path, std::vector<std::string>& paths) {
  const Reflection* absl_nonnull const reflection = message.GetReflection();
  const google::protobuf::UnknownFieldSet& unknown = reflection->GetUnknownFields(message);
  for (int i = 0; i < unknown.field_count(); ++i) {
    paths.push_back(absl::StrCat(path, ": field ", unknown.field(i).number()));
  }
  std::vector<const FieldDescriptor* absl_nonnull> fields;
  reflection->ListFields(message, &fields);
  for (const FieldDescriptor* absl_nonnull const field : fields) {
    if (field->cpp_type() != FieldDescriptor::CPPTYPE_MESSAGE) continue;
    const std::string fieldPath = path.empty() ? std::string(field->name()) : absl::StrCat(path, ".", field->name());
    if (!field->is_repeated()) {
      collectUnknownFields(reflection->GetMessage(message, field), fieldPath, paths);
      continue;
    }
    const int size = reflection->FieldSize(message, field);
    for (int index = 0; index < size; ++index) {
      collectUnknownFields(reflection->GetRepeatedMessage(message, field, index), absl::StrCat(fieldPath, "[", index, "]"), paths);
    }
  }
}

/** Every message and enum reachable from `root` through the fields, by full name. */
struct Reachable {
  std::map<std::string, const Descriptor* absl_nonnull> messages;
  std::map<std::string, const EnumDescriptor* absl_nonnull> enums;
};

void collectReachable(const Descriptor& descriptor, Reachable& reachable) {
  if (!reachable.messages.emplace(std::string(descriptor.full_name()), &descriptor).second) return;
  for (int i = 0; i < descriptor.field_count(); ++i) {
    const FieldDescriptor* absl_nonnull const field = descriptor.field(i);
    if (const Descriptor* absl_nullable const type = field->message_type(); type != nullptr) {
      collectReachable(*type, reachable);
    }
    if (const EnumDescriptor* absl_nullable const type = field->enum_type(); type != nullptr) {
      reachable.enums.emplace(std::string(type->full_name()), type);
    }
  }
}

/** The full name of the message or enum type of `field`; "-" for a scalar. */
std::string typeName(const FieldDescriptor& field) {
  if (field.message_type() != nullptr) return std::string(field.message_type()->full_name());
  if (field.enum_type() != nullptr) return std::string(field.enum_type()->full_name());
  return "-";
}

}  // namespace

std::vector<std::string> UnknownFieldPaths(const Message& message) {
  std::vector<std::string> paths;
  collectUnknownFields(message, /*path=*/"", paths);
  return paths;
}

std::string SchemaFingerprintText(const Descriptor& descriptor) {
  Reachable reachable;
  collectReachable(descriptor, reachable);
  std::string text;
  for (const std::pair<const std::string, const Descriptor* absl_nonnull>& entry : reachable.messages) {
    absl::StrAppend(&text, "message ", entry.first, "\n");
    std::vector<const FieldDescriptor* absl_nonnull> fields;
    for (int i = 0; i < entry.second->field_count(); ++i) fields.push_back(entry.second->field(i));
    std::sort(fields.begin(), fields.end(), [](const FieldDescriptor* absl_nonnull lhs, const FieldDescriptor* absl_nonnull rhs) {
      return lhs->number() < rhs->number();
    });
    for (const FieldDescriptor* absl_nonnull const field : fields) {
      absl::StrAppend(&text, "field ", field->number(), " ", field->name(), " ", static_cast<int>(field->type()), " ",
                      field->is_repeated() ? "repeated" : "singular", " ", field->has_presence() ? "presence" : "no_presence", " ",
                      typeName(*field), "\n");
    }
  }
  for (const std::pair<const std::string, const EnumDescriptor* absl_nonnull>& entry : reachable.enums) {
    absl::StrAppend(&text, "enum ", entry.first, "\n");
    std::vector<std::pair<int, std::string>> values;
    for (int i = 0; i < entry.second->value_count(); ++i) {
      values.emplace_back(entry.second->value(i)->number(), std::string(entry.second->value(i)->name()));
    }
    std::sort(values.begin(), values.end());
    for (const std::pair<int, std::string>& value : values) absl::StrAppend(&text, "value ", value.first, " ", value.second, "\n");
  }
  return text;
}

std::string SchemaFingerprint(const Descriptor& descriptor) {
  uint64_t hash = kFnvOffsetBasis;
  for (const char byte : SchemaFingerprintText(descriptor)) {
    hash ^= static_cast<uint64_t>(static_cast<unsigned char>(byte));
    hash *= kFnvPrime;
  }
  return absl::StrFormat("%016x", hash);
}

}  // namespace nproto
