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

#include "tools/nproto/test/ProtoTestValues.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

#include "google/protobuf/descriptor.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

namespace nproto::test_support {
namespace {

using google::protobuf::Descriptor;
using google::protobuf::EnumDescriptor;
using google::protobuf::EnumValueDescriptor;
using google::protobuf::FieldDescriptor;
using google::protobuf::Message;
using google::protobuf::Reflection;

constexpr int kMaximumDepth = 8;

// A number for one value of one field: distinct per seed, field and element, never zero.
std::int64_t variantOf(const TestValueOptions& options, const FieldDescriptor* field, int element) {
  return static_cast<std::int64_t>(options.seed) * 1000 + field->number() * 10 + element + 1;
}

// A string of fixed length (longer than any small-string buffer), so that two seeds give strings of one size.
std::string textOf(std::int64_t variant) {
  return absl::StrFormat("value-%09d-text", variant);
}

// An enum value other than the default when there is one.
const EnumValueDescriptor* enumValueOf(const FieldDescriptor* field, std::int64_t variant) {
  const EnumDescriptor* enumType = field->enum_type();
  const int defaultNumber = field->default_value_enum()->number();
  int candidates = 0;
  for (int index = 0; index < enumType->value_count(); ++index) {
    candidates += enumType->value(index)->number() != defaultNumber ? 1 : 0;
  }
  if (candidates == 0) {
    return field->default_value_enum();
  }
  int choice = static_cast<int>(variant % candidates);
  for (int index = 0; index < enumType->value_count(); ++index) {
    if (enumType->value(index)->number() == defaultNumber) {
      continue;
    }
    if (choice-- == 0) {
      return enumType->value(index);
    }
  }
  return field->default_value_enum();
}

void fill(const TestValueOptions& options, int depth, Message* message);

// Sets the singular field `field` of `message`, or adds an element when it is repeated.
void setValue(const TestValueOptions& options, int depth, const FieldDescriptor* field, std::int64_t variant, Message* message) {
  const Reflection* reflection = message->GetReflection();
  const bool repeated = field->is_repeated();
  switch (field->cpp_type()) {
    case FieldDescriptor::CPPTYPE_INT32:
      repeated ? reflection->AddInt32(message, field, static_cast<std::int32_t>(-variant))
               : reflection->SetInt32(message, field, static_cast<std::int32_t>(-variant));
      break;
    case FieldDescriptor::CPPTYPE_INT64:
      repeated ? reflection->AddInt64(message, field, -variant * 1000003) : reflection->SetInt64(message, field, -variant * 1000003);
      break;
    case FieldDescriptor::CPPTYPE_UINT32:
      repeated ? reflection->AddUInt32(message, field, static_cast<std::uint32_t>(variant))
               : reflection->SetUInt32(message, field, static_cast<std::uint32_t>(variant));
      break;
    case FieldDescriptor::CPPTYPE_UINT64:
      repeated ? reflection->AddUInt64(message, field, static_cast<std::uint64_t>(variant) * 1000003u)
               : reflection->SetUInt64(message, field, static_cast<std::uint64_t>(variant) * 1000003u);
      break;
    case FieldDescriptor::CPPTYPE_DOUBLE:
      repeated ? reflection->AddDouble(message, field, static_cast<double>(variant) + 0.25)
               : reflection->SetDouble(message, field, static_cast<double>(variant) + 0.25);
      break;
    case FieldDescriptor::CPPTYPE_FLOAT:
      repeated ? reflection->AddFloat(message, field, static_cast<float>(variant) + 0.5f)
               : reflection->SetFloat(message, field, static_cast<float>(variant) + 0.5f);
      break;
    case FieldDescriptor::CPPTYPE_BOOL:
      // Singular: true, the non-default value. Repeated: alternating.
      repeated ? reflection->AddBool(message, field, variant % 2 == 0) : reflection->SetBool(message, field, /*value=*/true);
      break;
    case FieldDescriptor::CPPTYPE_ENUM:
      repeated ? reflection->AddEnum(message, field, enumValueOf(field, variant))
               : reflection->SetEnum(message, field, enumValueOf(field, variant));
      break;
    case FieldDescriptor::CPPTYPE_STRING:
      repeated ? reflection->AddString(message, field, textOf(variant)) : reflection->SetString(message, field, textOf(variant));
      break;
    case FieldDescriptor::CPPTYPE_MESSAGE:
      if (depth < kMaximumDepth) {
        fill(options, depth + 1, repeated ? reflection->AddMessage(message, field) : reflection->MutableMessage(message, field));
      }
      break;
  }
}

// Sets the key of a map entry from its index only, so that two seeds give one key set.
void setKey(const FieldDescriptor* key, int index, Message* entry) {
  const Reflection* reflection = entry->GetReflection();
  switch (key->cpp_type()) {
    case FieldDescriptor::CPPTYPE_INT32:
      reflection->SetInt32(entry, key, index - 1);
      break;
    case FieldDescriptor::CPPTYPE_INT64:
      reflection->SetInt64(entry, key, static_cast<std::int64_t>(index) * 10000000000LL - 1);
      break;
    case FieldDescriptor::CPPTYPE_UINT32:
      reflection->SetUInt32(entry, key, static_cast<std::uint32_t>(index) + 5u);
      break;
    case FieldDescriptor::CPPTYPE_UINT64:
      reflection->SetUInt64(entry, key, static_cast<std::uint64_t>(index) * 10000000000ULL + 5u);
      break;
    case FieldDescriptor::CPPTYPE_BOOL:
      reflection->SetBool(entry, key, index % 2 == 0);
      break;
    case FieldDescriptor::CPPTYPE_STRING:
      reflection->SetString(entry, key, absl::StrCat("key ", index));
      break;
    default:
      ADD_FAILURE() << "not a map key type: " << key->full_name();
      break;
  }
}

void fill(const TestValueOptions& options, int depth, Message* message) {
  const Descriptor* descriptor = message->GetDescriptor();
  const Reflection* reflection = message->GetReflection();
  for (int index = 0; index < descriptor->field_count(); ++index) {
    const FieldDescriptor* field = descriptor->field(index);
    if (options.skipEvery > 0 && (field->number() + options.seed) % options.skipEvery == 0) {
      continue;
    }
    if (const google::protobuf::OneofDescriptor* oneof = field->real_containing_oneof(); oneof != nullptr) {
      if (oneof->field(options.oneofChoice % oneof->field_count()) != field) {
        continue;
      }
    }
    if (field->is_map()) {
      const int entries = field->message_type()->map_key()->cpp_type() == FieldDescriptor::CPPTYPE_BOOL ? std::min(options.repeatedSize, 2)
                                                                                                        : options.repeatedSize;
      for (int element = 0; element < entries; ++element) {
        Message* entry = reflection->AddMessage(message, field);
        setKey(field->message_type()->map_key(), element, entry);
        setValue(options, depth, field->message_type()->map_value(), variantOf(options, field, element), entry);
      }
    } else if (field->is_repeated()) {
      for (int element = 0; element < options.repeatedSize; ++element) {
        setValue(options, depth, field, variantOf(options, field, element), message);
      }
    } else {
      setValue(options, depth, field, variantOf(options, field, /*element=*/0), message);
    }
  }
}

}  // namespace

void FillWithTestValues(const TestValueOptions& options, google::protobuf::Message* message) {
  fill(options, /*depth=*/0, message);
}

std::string DeterministicBytes(const google::protobuf::Message& message) {
  std::string bytes;
  {
    google::protobuf::io::StringOutputStream stream(&bytes);
    google::protobuf::io::CodedOutputStream coded(&stream);
    coded.SetSerializationDeterministic(/*value=*/true);
    EXPECT_TRUE(message.SerializePartialToCodedStream(&coded));
  }
  return bytes;
}

::testing::AssertionResult ProtoEquals(const google::protobuf::Message& expected, const google::protobuf::Message& actual) {
  if (expected.GetDescriptor() == actual.GetDescriptor() && DeterministicBytes(expected) == DeterministicBytes(actual)) {
    return ::testing::AssertionSuccess();
  }
  std::string expectedText;
  std::string actualText;
  if (!google::protobuf::TextFormat::PrintToString(expected, &expectedText)) {
    expectedText = "<cannot print>";
  }
  if (!google::protobuf::TextFormat::PrintToString(actual, &actualText)) {
    actualText = "<cannot print>";
  }
  return ::testing::AssertionFailure() << "expected " << expected.GetDescriptor()->full_name() << ":\n"
                                       << expectedText << "\nactual " << actual.GetDescriptor()->full_name() << ":\n"
                                       << actualText;
}

}  // namespace nproto::test_support
