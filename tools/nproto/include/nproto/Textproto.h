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

// Strict textproto parsing for the repository's configuration files (tools/nproto/README.md): every configuration file
// is a .textproto of a message, and a file that does not match its schema exactly is an error that names the file, the
// line and the column, never a silently ignored field.
//
//   absl::StatusOr<robot_ipc_proto::NetworkConfig> config =
//       nproto::ParseTextprotoFile<robot_ipc_proto::NetworkConfig>("config/ipc/network.textproto");
//   absl::StatusOr<robot::ipc::msgs::NetworkConfig> plain =
//       nproto::LoadTextprotoFile<robot::ipc::msgs::NetworkConfig, robot_ipc_proto::NetworkConfig>(path);
//
// For start-up and tools: parsing allocates, reads files and must never run on a realtime thread.

#pragma once

#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/message.h"

namespace nproto {

/**
 * Replaces the contents of `message` with the textproto `text`, parsed strictly with google::protobuf::TextFormat: a
 * syntax error, an unknown field or extension, an unknown enum value name, a value of the wrong type or out of range,
 * a non-repeated field given twice and a missing proto2 required field are all errors. The error is InvalidArgument,
 * one line per problem, "<sourceName>:<line>:<column>: <problem>" with 1-based line and column. On error the contents
 * of `message` are unspecified.
 *
 * The schema answers what it can (tools/nproto/README.md, "Retired fields"): an unknown field that the message lists as
 * an (nproto.retired_field) is "'<name>' is retired: <replacement>"; otherwise the unknown-field error says "Did you
 * mean ..." when the snake-cased name is a field, and ends with the message's (nproto.retired_layout_hint). A `text`
 * whose leading comment block names another message (`# proto-message: pkg.Other`) is refused before it is parsed.
 */
absl::Status ParseTextprotoInto(absl::string_view text, absl::string_view sourceName, google::protobuf::Message* absl_nonnull message);

/** The contents of the file at `path`. NotFound (or the error the OS reports) names the path. */
absl::StatusOr<std::string> ReadTextFile(absl::string_view path);

/** The textproto `text` as a `Message` (see ParseTextprotoInto); `sourceName` names it in errors. */
template <typename Message>
absl::StatusOr<Message> ParseTextproto(absl::string_view text, absl::string_view sourceName) {
  Message message;
  absl::Status status = ParseTextprotoInto(text, sourceName, &message);
  if (!status.ok()) {
    return status;
  }
  return message;
}

/** The textproto file at `path` as a `Message`; errors name the path. */
template <typename Message>
absl::StatusOr<Message> ParseTextprotoFile(absl::string_view path) {
  absl::StatusOr<std::string> text = ReadTextFile(path);
  if (!text.ok()) {
    return text.status();
  }
  return ParseTextproto<Message>(*text, path);
}

/**
 * The textproto file at `path` as the nproto struct `Struct` of its message `Message`: ParseTextprotoFile() followed
 * by the generated FromProto() (include the message's <file>.nproto.pb.h). A FromProto() error is prefixed with the
 * path: "config/x.textproto: kind: 7 is not a value of ...".
 */
template <typename Struct, typename Message>
absl::StatusOr<Struct> LoadTextprotoFile(absl::string_view path) {
  absl::StatusOr<Message> message = ParseTextprotoFile<Message>(path);
  if (!message.ok()) {
    return message.status();
  }
  Struct value;
  const absl::Status status = FromProto(*message, &value);
  if (!status.ok()) {
    return absl::Status(status.code(), absl::StrCat(path, ": ", status.message()));
  }
  return value;
}

/**
 * `message` as a textproto, for tools that write configuration files: one field per line, repeated scalars as
 * `field: [1, 2, 3]`, map entries sorted by key, UTF-8 strings unescaped. ParseTextprotoInto() reads it back. Empty if the printer fails,
 * which it does not for a message of a generated type.
 */
std::string WriteTextproto(const google::protobuf::Message& message);

}  // namespace nproto
