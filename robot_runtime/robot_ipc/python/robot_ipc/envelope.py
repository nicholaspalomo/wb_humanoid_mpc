# Copyright (c) 2026, Nicholas Palomo. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""The envelope of every sample of the bus, as robot_ipc/WireProtocol.h encodes it in C++.

Every sample is one envelope, the payload of a Zenoh put, little-endian: the wire protocol's version, the writer's id
and sequence number, the source timestamp, the full protobuf type name and the serialized message.
"""

import dataclasses
import struct

# LINT.IfChange(wire_protocol)
# The version of the wire protocol; version 1 was the ZeroMQ framing and version 2 the Cyclone DDS envelope that came
# before.
WIRE_PROTOCOL_VERSION = 3
# protocol_version, writer_id, sequence_number, source_timestamp_ns, type_name_length; then the type name and the
# payload.
_HEADER = struct.Struct("<IQQqI")
ENVELOPE_HEADER_BYTES = _HEADER.size
# LINT.ThenChange(//robot_runtime/robot_ipc/include/robot_ipc/WireProtocol.h:wire_protocol)
_VERSION = struct.Struct("<I")


class EnvelopeError(ValueError):
    """Bytes that are no envelope of this protocol version: shorter than its header or its type name."""


@dataclasses.dataclass(frozen=True)
class Envelope:
    """One sample of a bus topic.

    Attributes:
        protocol_version: WIRE_PROTOCOL_VERSION of the publisher; a bus refuses another version.
        writer_id: random, one per publisher of a bus.
        sequence_number: 1, 2, 3, ... per writer: a gap is a sample the subscriber missed.
        source_timestamp_ns: strictly increasing across the topics of the publishing bus.
        type_name: the full protobuf type name of the payload, e.g. "legged_mpc_msgs.MpcPolicy".
        payload: the serialized protobuf message.
    """

    protocol_version: int
    writer_id: int
    sequence_number: int
    source_timestamp_ns: int
    type_name: str
    payload: bytes

    def encode(self) -> bytes:
        """The envelope's bytes, ready to put."""
        name = self.type_name.encode("utf-8")
        return (
            _HEADER.pack(
                self.protocol_version,
                self.writer_id,
                self.sequence_number,
                self.source_timestamp_ns,
                len(name),
            )
            + name
            + self.payload
        )


def protocol_version_of(data: bytes) -> int | None:
    """The protocol version of the envelope `data`, its first field in every version; None for fewer than 4 bytes."""
    if len(data) < _VERSION.size:
        return None
    return int(_VERSION.unpack_from(data)[0])


def decode(data: bytes) -> Envelope:
    """Decodes an envelope of WIRE_PROTOCOL_VERSION.

    Args:
        data: The bytes of a sample.

    Returns:
        The envelope.

    Raises:
        EnvelopeError: the bytes are shorter than the header or the type name.
    """
    if len(data) < ENVELOPE_HEADER_BYTES:
        raise EnvelopeError("the data is shorter than an envelope's header")
    version, writer_id, sequence, timestamp, name_length = _HEADER.unpack_from(
        data
    )
    if name_length > len(data) - ENVELOPE_HEADER_BYTES:
        raise EnvelopeError("the type name runs past the end of the data")
    name_end = ENVELOPE_HEADER_BYTES + name_length
    return Envelope(
        protocol_version=version,
        writer_id=writer_id,
        sequence_number=sequence,
        source_timestamp_ns=timestamp,
        type_name=bytes(data[ENVELOPE_HEADER_BYTES:name_end]).decode(
            "utf-8", errors="replace"
        ),
        payload=bytes(data[name_end:]),
    )


__all__ = [
    "ENVELOPE_HEADER_BYTES",
    "WIRE_PROTOCOL_VERSION",
    "Envelope",
    "EnvelopeError",
    "decode",
    "protocol_version_of",
]
