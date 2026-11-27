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

#include "robot_ipc/WireProtocol.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>

#include "absl/strings/string_view.h"

namespace robot::ipc {
namespace {

// The offsets of the header's fields (WireProtocol.h).
constexpr size_t kVersionOffset = 0;
constexpr size_t kWriterIdOffset = 4;
constexpr size_t kSequenceOffset = 12;
constexpr size_t kTimestampOffset = 20;
constexpr size_t kTypeNameLengthOffset = 28;
constexpr int kBitsPerByte = 8;
constexpr uint64_t kByteMask = 0xff;

template <typename Unsigned>
  requires std::is_unsigned_v<Unsigned>
void appendLittleEndian(std::string& bytes, Unsigned value) {
  for (size_t index = 0; index < sizeof(Unsigned); ++index) {
    bytes.push_back(static_cast<char>(
        (static_cast<uint64_t>(value) >> (kBitsPerByte * index)) & kByteMask));
  }
}

template <typename Unsigned>
  requires std::is_unsigned_v<Unsigned>
Unsigned readLittleEndian(absl::string_view bytes, size_t offset) {
  uint64_t value = 0;
  for (size_t index = 0; index < sizeof(Unsigned); ++index) {
    value |=
        static_cast<uint64_t>(static_cast<unsigned char>(bytes[offset + index]))
        << (kBitsPerByte * index);
  }
  return static_cast<Unsigned>(value);
}

}  // namespace

std::string encodeEnvelope(const EnvelopeHeader& header,
                           absl::string_view typeName,
                           absl::string_view payload) {
  std::string bytes;
  bytes.reserve(kEnvelopeHeaderBytes + typeName.size() + payload.size());
  appendLittleEndian<uint32_t>(bytes, header.protocolVersion);
  appendLittleEndian<uint64_t>(bytes, header.writerId);
  appendLittleEndian<uint64_t>(bytes, header.sequenceNumber);
  appendLittleEndian<uint64_t>(bytes,
                               static_cast<uint64_t>(header.sourceTimestampNs));
  appendLittleEndian<uint32_t>(bytes, static_cast<uint32_t>(typeName.size()));
  bytes.append(typeName);
  bytes.append(payload);
  return bytes;
}

std::optional<uint32_t> envelopeProtocolVersion(absl::string_view bytes) {
  if (bytes.size() < sizeof(uint32_t)) {
    return std::nullopt;
  }
  return readLittleEndian<uint32_t>(bytes, kVersionOffset);
}

std::optional<EnvelopeView> decodeEnvelope(absl::string_view bytes) {
  if (bytes.size() < kEnvelopeHeaderBytes) {
    return std::nullopt;
  }
  EnvelopeView envelope;
  envelope.header.protocolVersion =
      readLittleEndian<uint32_t>(bytes, kVersionOffset);
  envelope.header.writerId = readLittleEndian<uint64_t>(bytes, kWriterIdOffset);
  envelope.header.sequenceNumber =
      readLittleEndian<uint64_t>(bytes, kSequenceOffset);
  envelope.header.sourceTimestampNs =
      static_cast<int64_t>(readLittleEndian<uint64_t>(bytes, kTimestampOffset));
  const size_t typeNameLength =
      readLittleEndian<uint32_t>(bytes, kTypeNameLengthOffset);
  if (typeNameLength > bytes.size() - kEnvelopeHeaderBytes) {
    return std::nullopt;
  }
  envelope.typeName = bytes.substr(kEnvelopeHeaderBytes, typeNameLength);
  envelope.payload = bytes.substr(kEnvelopeHeaderBytes + typeNameLength);
  return envelope;
}

}  // namespace robot::ipc
