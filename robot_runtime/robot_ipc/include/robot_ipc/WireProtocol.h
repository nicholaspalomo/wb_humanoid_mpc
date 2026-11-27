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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"

namespace robot::ipc {

// The Python bus speaks the same protocol (robot_ipc/envelope.py).
// LINT.IfChange(wire_protocol)
/**
 * The version of the bus's wire protocol. Every sample is one envelope, the
 * payload of a Zenoh put, little-endian:
 *
 *   uint32 protocol_version     kWireProtocolVersion
 *   uint64 writer_id            random, one per publisher of a bus
 *   uint64 sequence_number      1, 2, 3, ... per writer
 *   int64  source_timestamp_ns  strictly increasing across a bus's topics
 *   uint32 type_name_length
 *   char   type_name[type_name_length]  e.g. "legged_mpc_msgs.MpcPolicy"
 *   byte   payload[...]         the serialized protobuf message, to the end
 *
 * A bus refuses (counts as rejected and logs) a sample of another version.
 * Version 1 was the four-frame ZeroMQ framing and version 2 the Cyclone DDS
 * envelope (an IDL struct in XCDR) that came before; bump the version with
 * every change of the envelope, on both sides at once.
 */
inline constexpr uint32_t kWireProtocolVersion = 3;
/** The bytes before the type name. */
inline constexpr size_t kEnvelopeHeaderBytes = 32;
// LINT.ThenChange(//robot_runtime/robot_ipc/python/robot_ipc/envelope.py:wire_protocol)

/** The fixed fields of an envelope. */
struct EnvelopeHeader {
  uint32_t protocolVersion = kWireProtocolVersion;
  uint64_t writerId = 0;
  uint64_t sequenceNumber = 0;
  int64_t sourceTimestampNs = 0;
};

/** An envelope decoded in place: the views point into the decoded bytes. */
struct EnvelopeView {
  EnvelopeHeader header;
  absl::string_view
      typeName;  // NOLINT(totw-view-member): views the bytes decodeEnvelope() decoded, which outlive it
  absl::string_view
      payload;  // NOLINT(totw-view-member): views the bytes decodeEnvelope() decoded, which outlive it
};

/** The envelope of `header`, `typeName` and `payload`, ready to put. */
std::string encodeEnvelope(const EnvelopeHeader& header,
                           absl::string_view typeName,
                           absl::string_view payload);

/**
 * The protocol version of the envelope `bytes`, the first field of every
 * version; nullopt for fewer than four bytes. A bus checks it before decoding.
 */
std::optional<uint32_t> envelopeProtocolVersion(absl::string_view bytes);

/**
 * Decodes an envelope of kWireProtocolVersion; nullopt when `bytes` is shorter
 * than its header or its type name. The views of the result point into
 * `bytes`, which must outlive them.
 */
std::optional<EnvelopeView> decodeEnvelope(absl::string_view bytes);

}  // namespace robot::ipc
