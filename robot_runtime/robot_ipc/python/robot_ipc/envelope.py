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


"""robot_ipc::Envelope, the one DDS type of the bus, in Python: the twin of idl/Envelope.idl.

Every bus topic is a DDS topic of the same name whose samples are one Envelope: the wire protocol's version, the
writer's sequence number, the full protobuf type name and the serialized message (robot_ipc/WireProtocol.h).

The class declares the IDL members for cyclonedds's type information (XTypes), so that a C++ reader and a Python
writer of the same topic match. It serializes itself: cyclonedds's generic machinery handles a sequence<octet> one
element at a time as a Python list, which costs tens of milliseconds for a 0.75 MB policy; this envelope packs the
payload as one bytes object, in the same XCDR encoding.
"""

import dataclasses
import struct
from typing import Any

import cyclonedds.idl
import cyclonedds.idl.annotations
import cyclonedds.idl.types

# LINT.IfChange(wire_protocol)
# The version of the wire protocol; version 1 was the ZeroMQ framing that came before.
WIRE_PROTOCOL_VERSION = 2
# The DDS type name of the envelope.
ENVELOPE_TYPE_NAME = "robot_ipc::Envelope"
# LINT.ThenChange(//robot_runtime/robot_ipc/include/robot_ipc/WireProtocol.h:wire_protocol)

# The encapsulation identifiers of the serialized data's header (DDS-XTypes 7.6.3.1.2): plain CDR (XCDR1) and plain
# CDR2 (XCDR2) of a final struct, big- and little-endian.
_CDR_BE = 0x0000
_CDR_LE = 0x0001
_CDR2_BE = 0x0006
_CDR2_LE = 0x0007
_HEADER = struct.Struct(">HH")
_HEADER_SIZE = _HEADER.size
# XCDR2 aligns 8-byte members to 4, XCDR1 to 8 (from the end of the header).
_XCDR1_UINT64_ALIGNMENT = 8
_XCDR2_UINT64_ALIGNMENT = 4
_UINT32_SIZE = 4
_UINT64_SIZE = 8
_STRING_TERMINATOR = b"\x00"


class EnvelopeError(ValueError):
    """Serialized data that is not an Envelope (a short buffer, an unknown encoding, a malformed string)."""


def _padding(offset: int, alignment: int) -> int:
    return -offset % alignment


# LINT.IfChange(envelope)
@dataclasses.dataclass
@cyclonedds.idl.annotations.final
class Envelope(cyclonedds.idl.IdlStruct, typename=ENVELOPE_TYPE_NAME):  # type: ignore[call-arg]  # IdlMeta takes the IDL type name.
    """One sample of a bus topic.

    Attributes:
        protocol_version: WIRE_PROTOCOL_VERSION of the publisher; a bus refuses another version.
        sequence_number: 1, 2, 3, ... per writer (one bus publishing one topic): a gap is a sample the reader missed.
        type_name: the full protobuf type name of the payload, e.g. "humanoid_mpc_msgs.MpcPolicy".
        payload: the serialized protobuf message. Declared a sequence<octet> for the type information, held as bytes.
    """

    protocol_version: cyclonedds.idl.types.uint32
    sequence_number: cyclonedds.idl.types.uint64
    type_name: str
    payload: cyclonedds.idl.types.sequence[cyclonedds.idl.types.byte]
    # LINT.ThenChange(//robot_runtime/robot_ipc/idl/Envelope.idl:envelope)

    def serialize(
        self,
        buffer: Any = None,
        endianness: Any = None,
        use_version_2: bool | None = None,
    ) -> bytes:
        """The XCDR2 (or, when asked, XCDR1) little-endian encoding of the envelope, with its header.

        Args:
            buffer: Unused: cyclonedds's own buffer, which this encoding does not need.
            endianness: Unused: always little-endian, which every reader accepts.
            use_version_2: False for XCDR1; XCDR2 otherwise.

        Returns:
            The header and the members, as DataWriter.write() hands them to DDS.
        """
        del buffer, endianness  # Unused.
        name = self.type_name.encode("utf-8") + _STRING_TERMINATOR
        payload = bytes(self.payload)
        version_1 = use_version_2 is False
        encapsulation = _CDR_LE if version_1 else _CDR2_LE
        uint64_alignment = (
            _XCDR1_UINT64_ALIGNMENT if version_1 else _XCDR2_UINT64_ALIGNMENT
        )
        parts = [
            _HEADER.pack(encapsulation, 0),
            struct.pack("<I", self.protocol_version),
        ]
        offset = _UINT32_SIZE
        pad = _padding(offset, uint64_alignment)
        parts.append(
            b"\x00" * pad + struct.pack("<QI", self.sequence_number, len(name))
        )
        offset += pad + _UINT64_SIZE + _UINT32_SIZE
        parts.append(name)
        offset += len(name)
        pad = _padding(offset, _UINT32_SIZE)
        parts.append(b"\x00" * pad + struct.pack("<I", len(payload)))
        parts.append(payload)
        return b"".join(parts)

    @classmethod
    def deserialize(
        cls,
        data: bytes,
        has_header: bool = True,
        use_version_2: bool | None = None,
    ) -> "Envelope":
        """Reads an envelope of any of the four plain encodings (XCDR1 or XCDR2, either byte order).

        Args:
            data: The serialized sample, with its header unless has_header is False.
            has_header: False for data without the 4-byte header, encoded as use_version_2 says.
            use_version_2: Without a header: True for XCDR2 (the default), False for XCDR1.

        Returns:
            The envelope; its payload is a bytes object.

        Raises:
            EnvelopeError: the data is not an envelope.
        """
        view = memoryview(data)
        if has_header:
            if len(view) < _HEADER_SIZE:
                raise EnvelopeError("the data is shorter than its header")
            encapsulation = _HEADER.unpack_from(view)[0]
            if encapsulation not in (_CDR_BE, _CDR_LE, _CDR2_BE, _CDR2_LE):
                raise EnvelopeError(f"unknown encapsulation 0x{encapsulation:04x}")
            body = view[_HEADER_SIZE:]
        else:
            encapsulation = _CDR2_LE if use_version_2 is not False else _CDR_LE
            body = view
        order = "<" if encapsulation in (_CDR_LE, _CDR2_LE) else ">"
        uint64_alignment = (
            _XCDR1_UINT64_ALIGNMENT
            if encapsulation in (_CDR_LE, _CDR_BE)
            else _XCDR2_UINT64_ALIGNMENT
        )
        try:
            (protocol_version,) = struct.unpack_from(f"{order}I", body, 0)
            offset = _UINT32_SIZE + _padding(_UINT32_SIZE, uint64_alignment)
            sequence_number, name_size = struct.unpack_from(f"{order}QI", body, offset)
            offset += _UINT64_SIZE + _UINT32_SIZE
            name = bytes(body[offset : offset + name_size])
            if len(name) != name_size or not name.endswith(_STRING_TERMINATOR):
                raise EnvelopeError("the type name is not a terminated string")
            offset += name_size
            offset += _padding(offset, _UINT32_SIZE)
            (payload_size,) = struct.unpack_from(f"{order}I", body, offset)
            offset += _UINT32_SIZE
            payload = bytes(body[offset : offset + payload_size])
        except struct.error as error:
            raise EnvelopeError(f"the data ends early: {error}") from None
        if len(payload) != payload_size:
            raise EnvelopeError("the payload ends early")
        return cls(
            protocol_version=protocol_version,
            sequence_number=sequence_number,
            type_name=name[:-1].decode("utf-8", errors="replace"),
            payload=payload,
        )


__all__ = [
    "ENVELOPE_TYPE_NAME",
    "WIRE_PROTOCOL_VERSION",
    "Envelope",
    "EnvelopeError",
]
