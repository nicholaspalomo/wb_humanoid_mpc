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

// The schema a message was built from, at run time (tools/nproto/README.md, "Version skew"): the fields of a received
// message that its receiver's schema does not have, and a fingerprint of a schema's wire layout that a sender puts in
// its message for the receiver to compare with its own. Both allocate: for an IO thread, never a realtime one.
//
//   if (std::vector<std::string> unknown = nproto::UnknownFieldPaths(message); !unknown.empty()) refuse(unknown);
//   message.set_schema_fingerprint(nproto::SchemaFingerprint(*message.GetDescriptor()));

#pragma once

#include <string>
#include <vector>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"

namespace nproto {

/**
 * The fields of `message`, and of every message below it, that the receiver's schema does not know: a sender built
 * from another version of the schema set them, and the parser kept them as unknown fields, which no conversion reads.
 * Each is "<path>: field <number>", the path from `message` with an element of a repeated field by its index
 * ("task.contact_wrench_gate: field 103", "joint_gains[2]: field 9", ": field 40" for one of `message` itself). Empty
 * when `message` carries no unknown field.
 */
std::vector<std::string> UnknownFieldPaths(const google::protobuf::Message& message);

/**
 * The fingerprint of the wire layout of `descriptor` and of every message and enum its fields reach: 16 lowercase hex
 * digits, the 64-bit FNV-1a hash of SchemaFingerprintText(). Two builds give the same fingerprint exactly when they
 * describe these messages alike, so a receiver refuses a payload whose fingerprint is not its own, also one that lacks
 * fields the receiver has, which an unknown-field check cannot see. Options, comments and default values do not take
 * part. The Python twin is nproto_schema.schema_fingerprint().
 */
std::string SchemaFingerprint(const google::protobuf::Descriptor& descriptor);

/**
 * What SchemaFingerprint() hashes: every message reachable from `descriptor` and every enum, in the order of their full
 * names, a line for the message ("message <full name>") and one per field in the order of their numbers ("field
 * <number> <name> <type number> <repeated|singular> <presence|no_presence> <message or enum full name, or ->"), and a
 * line for each enum ("enum <full name>") and one per value ("value <number> <name>"), each line ending in "\n".
 */
std::string SchemaFingerprintText(const google::protobuf::Descriptor& descriptor);

}  // namespace nproto
