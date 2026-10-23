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

// The helpers the code nproto generates calls (tools/nproto/README.md): the conversions of repeated, map, optional and
// oneof fields between an nproto struct and its protobuf message, and the error messages of FromProto(). Not an API of
// its own; include the generated <file>.nproto.pb.h instead.
//
// Every helper converts INTO the object it is given and reuses what that object already holds, so that converting a
// value into an object of the same shape does not allocate:
//
// - Eigen vectors and std::vector of scalars resize to the same size (a no-op) and copy;
// - std::vector of strings and of structs resize (same size: a no-op) and assign element by element, and std::string
//   assignment reuses its capacity;
// - repeated fields of the message are resized in place, and RepeatedPtrField keeps the elements a shorter value
//   removes, cleared, for the next longer one;
// - a std::map or protobuf Map whose key set is the value's is updated in place; any other key set rebuilds the map,
//   which allocates a node per key;
// - a std::optional or std::variant that already holds the alternative is assigned in place.
//
// FromProto() fails only on an enum number its enum does not define. The errors name the field path,
// "annotations.target_contact_patches[2].kind: 7 is not a value of humanoid_mpc_msgs.TargetContactPatch.Kind", and
// building them allocates, which only an error does.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/map.h"
#include "google/protobuf/repeated_field.h"
#include "google/protobuf/repeated_ptr_field.h"

namespace nproto::internal {

// ---------------------------------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------------------------------

/** The error of FromProto() for an enum number that `enumName` does not define: "7 is not a value of <enumName>". */
absl::Status UnknownEnumValueError(int32_t number, absl::string_view enumName);

/**
 * `status` from converting the field `field`, with the field in front of the path its message names:
 * "kind: 7 is not ..." becomes "patch.kind: 7 is not ...", and "7 is not ..." becomes "kind: 7 is not ...".
 */
absl::Status AnnotateError(absl::string_view field, const absl::Status& status);

/** `status` from converting element `index` of a repeated field: "[3].kind: ..." or "[3]: ...". */
absl::Status AnnotateElement(int index, const absl::Status& status);

/** `status` from converting the value of the map key `formattedKey` (see FormatKey): "[\"x\"]: ...". */
absl::Status AnnotateKey(absl::string_view formattedKey, const absl::Status& status);

/** A map key as an error message writes it: strings quoted, bools as true/false, integers in decimal. */
std::string FormatKey(const std::string& key);
std::string FormatKey(bool key);
template <typename Integer>
std::string FormatKey(Integer key) {
  return absl::StrCat(key);
}

// ---------------------------------------------------------------------------------------------------------------------
// One value: a scalar or string is copied, an enum or message is converted by its generated overload (found by
// argument-dependent lookup in the namespace of its struct).
// ---------------------------------------------------------------------------------------------------------------------

// The helpers pass their output parameter on to the generated ToProto() / FromProto(), whose output parameters are
// `absl_nonnull` too.
// LINT.IfChange(output_parameters)
template <typename T>
absl::Status ValueFromProto(const T& proto, T* absl_nonnull value) {
  *value = proto;
  return absl::OkStatus();
}

template <typename Proto, typename Value>
absl::Status ValueFromProto(const Proto& proto, Value* absl_nonnull value) {
  return FromProto(proto, value);
}

template <typename T>
void ValueToProto(const T& value, T* absl_nonnull proto) {
  *proto = value;
}

template <typename Value, typename Proto>
void ValueToProto(const Value& value, Proto* absl_nonnull proto) {
  ToProto(value, proto);
}
// LINT.ThenChange(//tools/nproto/nproto_generator.py:output_parameters)

// ---------------------------------------------------------------------------------------------------------------------
// Repeated fields
// ---------------------------------------------------------------------------------------------------------------------

/** Resizes `field` to `size`; removed elements stay allocated, cleared, and Add() hands them out again. */
template <typename T>
void ResizeRepeatedPtrField(int size, google::protobuf::RepeatedPtrField<T>* absl_nonnull field) {
  while (field->size() > size) {
    field->RemoveLast();
  }
  while (field->size() < size) {
    static_cast<void>(field->Add());
  }
}

/** repeated double / float -> Eigen::VectorXd / VectorXf. */
template <typename Scalar>
void VectorFromProto(const google::protobuf::RepeatedField<Scalar>& proto, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>* absl_nonnull value) {
  value->resize(proto.size());
  std::copy(proto.begin(), proto.end(), value->data());
}

template <typename Scalar>
void VectorToProto(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& value, google::protobuf::RepeatedField<Scalar>* absl_nonnull proto) {
  proto->Assign(value.data(), value.data() + value.size());
}

/** Other repeated scalars (integers, bool) -> std::vector. */
template <typename T>
void ScalarsFromProto(const google::protobuf::RepeatedField<T>& proto, std::vector<T>* absl_nonnull value) {
  value->assign(proto.begin(), proto.end());
}

template <typename T>
void ScalarsToProto(const std::vector<T>& value, google::protobuf::RepeatedField<T>* absl_nonnull proto) {
  proto->Assign(value.begin(), value.end());
}

/** repeated string / bytes -> std::vector<std::string>. */
void StringsFromProto(const google::protobuf::RepeatedPtrField<std::string>& proto, std::vector<std::string>* absl_nonnull value);
void StringsToProto(const std::vector<std::string>& value, google::protobuf::RepeatedPtrField<std::string>* absl_nonnull proto);

/** repeated enum (a RepeatedField<int> in protobuf's C++ code) -> std::vector of the enum class. */
template <typename ProtoEnum, typename Enum>
absl::Status EnumsFromProto(const google::protobuf::RepeatedField<int>& proto, std::vector<Enum>* absl_nonnull value) {
  value->resize(static_cast<size_t>(proto.size()));
  for (int index = 0; index < proto.size(); ++index) {
    const absl::Status status = FromProto(static_cast<ProtoEnum>(proto.Get(index)), &(*value)[static_cast<size_t>(index)]);
    if (!status.ok()) {
      return AnnotateElement(index, status);
    }
  }
  return absl::OkStatus();
}

template <typename Enum>
void EnumsToProto(const std::vector<Enum>& value, google::protobuf::RepeatedField<int>* absl_nonnull proto) {
  proto->Clear();
  for (const Enum element : value) {
    proto->Add(static_cast<int>(element));
  }
}

/** repeated messages -> std::vector of their structs. */
template <typename Proto, typename Struct>
absl::Status MessagesFromProto(const google::protobuf::RepeatedPtrField<Proto>& proto, std::vector<Struct>* absl_nonnull value) {
  value->resize(static_cast<size_t>(proto.size()));
  for (int index = 0; index < proto.size(); ++index) {
    const absl::Status status = FromProto(proto.Get(index), &(*value)[static_cast<size_t>(index)]);
    if (!status.ok()) {
      return AnnotateElement(index, status);
    }
  }
  return absl::OkStatus();
}

template <typename Struct, typename Proto>
void MessagesToProto(const std::vector<Struct>& value, google::protobuf::RepeatedPtrField<Proto>* absl_nonnull proto) {
  ResizeRepeatedPtrField(static_cast<int>(value.size()), proto);
  for (size_t index = 0; index < value.size(); ++index) {
    ToProto(value[index], proto->Mutable(static_cast<int>(index)));
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Maps
// ---------------------------------------------------------------------------------------------------------------------

/** map<K, V> -> std::map<K, V or its struct>, in key order. */
template <typename Key, typename Proto, typename Value>
absl::Status MapFromProto(const google::protobuf::Map<Key, Proto>& proto, std::map<Key, Value>* absl_nonnull value) {
  bool sameKeys = proto.size() == value->size();
  for (typename google::protobuf::Map<Key, Proto>::const_iterator entry = proto.begin(); sameKeys && entry != proto.end(); ++entry) {
    sameKeys = value->find(entry->first) != value->end();
  }
  if (!sameKeys) {
    value->clear();
  }
  for (typename google::protobuf::Map<Key, Proto>::const_iterator entry = proto.begin(); entry != proto.end(); ++entry) {
    const typename std::map<Key, Value>::iterator target = sameKeys ? value->find(entry->first) : value->try_emplace(entry->first).first;
    const absl::Status status = ValueFromProto(entry->second, &target->second);
    if (!status.ok()) {
      return AnnotateKey(FormatKey(entry->first), status);
    }
  }
  return absl::OkStatus();
}

template <typename Key, typename Value, typename Proto>
void MapToProto(const std::map<Key, Value>& value, google::protobuf::Map<Key, Proto>* absl_nonnull proto) {
  bool sameKeys = proto->size() == value.size();
  for (typename std::map<Key, Value>::const_iterator entry = value.begin(); sameKeys && entry != value.end(); ++entry) {
    sameKeys = proto->find(entry->first) != proto->end();
  }
  if (!sameKeys) {
    proto->clear();
  }
  for (typename std::map<Key, Value>::const_iterator entry = value.begin(); entry != value.end(); ++entry) {
    Proto* absl_nonnull const target = sameKeys ? &proto->find(entry->first)->second : &(*proto)[entry->first];
    ValueToProto(entry->second, target);
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Explicit presence and oneofs
// ---------------------------------------------------------------------------------------------------------------------

/** An optional scalar or string: `present` is the message's has_<field>(). */
template <typename T>
void CopyOptionalFromProto(bool present, const T& proto, std::optional<T>* absl_nonnull value) {
  if (!present) {
    value->reset();
  } else if (value->has_value()) {
    **value = proto;
  } else {
    value->emplace(proto);
  }
}

/** An optional enum or message. */
template <typename Proto, typename Value>
absl::Status ConvertOptionalFromProto(bool present, const Proto& proto, std::optional<Value>* absl_nonnull value) {
  if (!present) {
    value->reset();
    return absl::OkStatus();
  }
  if (!value->has_value()) {
    value->emplace();
  }
  return FromProto(proto, &**value);
}

/** The scalar or string alternative `Index` of a oneof's std::variant. */
template <size_t Index, typename Proto, typename Variant>
void CopyAlternativeFromProto(const Proto& proto, Variant* absl_nonnull value) {
  if (value->index() != Index) {
    value->template emplace<Index>();
  }
  std::get<Index>(*value) = proto;
}

/** The enum or message alternative `Index` of a oneof's std::variant. */
template <size_t Index, typename Proto, typename Variant>
absl::Status ConvertAlternativeFromProto(const Proto& proto, Variant* absl_nonnull value) {
  if (value->index() != Index) {
    value->template emplace<Index>();
  }
  return FromProto(proto, &std::get<Index>(*value));
}

}  // namespace nproto::internal
