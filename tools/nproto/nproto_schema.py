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

"""The schema a message was built from: the twin of nproto/Schema.h's fingerprint (tools/nproto/README.md).

    import nproto_schema

    update.schema_fingerprint = nproto_schema.schema_fingerprint(update.DESCRIPTOR)

A sender puts the fingerprint of its message's schema in the message, and the receiver, which computes its own with
nproto::SchemaFingerprint(), refuses a message whose fingerprint differs: a sender built from another version of the
schema, whose fields the receiver would drop as unknown or take at their defaults. The fingerprint covers the wire
layout of the message and of every message and enum its fields reach - names, numbers, types, labels, presence - and
nothing else, so options, comments and default values may change without it.
"""

from google.protobuf import descriptor as descriptor_module

# LINT.IfChange(fnv1a)
_FNV_OFFSET_BASIS = 14_695_981_039_346_656_037
_FNV_PRIME = 1_099_511_628_211
# LINT.ThenChange(//tools/nproto/src/Schema.cpp:fnv1a)
_MASK = (1 << 64) - 1


def _collect(
    descriptor: descriptor_module.Descriptor,
    messages: dict[str, descriptor_module.Descriptor],
    enums: dict[str, descriptor_module.EnumDescriptor],
) -> None:
    """Adds `descriptor` and every message and enum its fields reach to `messages` and `enums`, by full name."""
    if descriptor.full_name in messages:
        return
    messages[descriptor.full_name] = descriptor
    for field in descriptor.fields:
        if field.message_type is not None:
            _collect(field.message_type, messages, enums)
        if field.enum_type is not None:
            enums[field.enum_type.full_name] = field.enum_type


def _type_name(field: descriptor_module.FieldDescriptor) -> str:
    if field.message_type is not None:
        return str(field.message_type.full_name)
    if field.enum_type is not None:
        return str(field.enum_type.full_name)
    return "-"


def schema_fingerprint_text(descriptor: descriptor_module.Descriptor) -> str:
    """The text schema_fingerprint() hashes, line for line nproto::SchemaFingerprintText().

    Args:
      descriptor: The message.

    Returns:
      Every message reachable from `descriptor` and every enum, in the order of their full names: "message <full name>"
      and "field <number> <name> <type number> <repeated|singular> <presence|no_presence> <type or ->" per field in the
      order of their numbers, "enum <full name>" and "value <number> <name>" per value; each line ends in a newline.
    """
    messages: dict[str, descriptor_module.Descriptor] = {}
    enums: dict[str, descriptor_module.EnumDescriptor] = {}
    _collect(descriptor, messages, enums)
    lines: list[str] = []
    for name in sorted(messages):
        lines.append(f"message {name}")
        for field in sorted(messages[name].fields, key=lambda field: field.number):
            label = "repeated" if field.is_repeated else "singular"
            presence = "presence" if field.has_presence else "no_presence"
            lines.append(
                f"field {field.number} {field.name} {field.type} {label} {presence} {_type_name(field)}"
            )
    for name in sorted(enums):
        lines.append(f"enum {name}")
        for number, value in sorted(
            (value.number, value.name) for value in enums[name].values
        ):
            lines.append(f"value {number} {value}")
    return "".join(f"{line}\n" for line in lines)


def schema_fingerprint(descriptor: descriptor_module.Descriptor) -> str:
    """The fingerprint of `descriptor`'s schema, as nproto::SchemaFingerprint() computes it: 16 lowercase hex digits.

    Args:
      descriptor: The message.

    Returns:
      The 64-bit FNV-1a hash of schema_fingerprint_text(), in hex.
    """
    value = _FNV_OFFSET_BASIS
    for byte in schema_fingerprint_text(descriptor).encode("utf-8"):
        value ^= byte
        value = (value * _FNV_PRIME) & _MASK
    return f"{value:016x}"
