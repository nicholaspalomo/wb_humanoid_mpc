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

#include <cstdint>

#include "absl/strings/string_view.h"

namespace robot::ipc {

// The Python bus speaks the same protocol (robot_ipc/envelope.py).
// LINT.IfChange(wire_protocol)
/**
 * The version of the bus's wire protocol. Every bus topic is a DDS topic of the same name whose samples are one
 * robot_ipc::Envelope (idl/Envelope.idl): the protocol version, the writer's sequence number, the full protobuf type
 * name and the serialized message. A bus refuses (counts as rejected and logs) a sample of another version. Version 1
 * was the four-frame ZeroMQ framing that came before; bump the version with every change of the envelope, on both
 * sides at once.
 */
inline constexpr uint32_t kWireProtocolVersion = 2;
/** The DDS type name of the envelope, which both buses register. */
inline constexpr absl::string_view kEnvelopeTypeName = "robot_ipc::Envelope";
// clang-format off
// LINT.ThenChange(//robot_runtime/robot_ipc/python/robot_ipc/envelope.py:wire_protocol, //robot_runtime/robot_ipc/idl/Envelope.idl:envelope)
// clang-format on

}  // namespace robot::ipc
