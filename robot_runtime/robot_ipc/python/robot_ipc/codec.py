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

"""Decodes a bus message from its type name, without a registry of topics (for tools/ipc).

The second frame of every bus message is the full protobuf type name of the third, such as
"humanoid_mpc_msgs.MpcPolicy". The class is found in the protobuf default descriptor pool, which knows every message
whose generated module (`*_pb2`) has been imported: import the modules of the messages you want to decode first.
"""

from google.protobuf import descriptor_pool
from google.protobuf import message
from google.protobuf import message_factory


class UnknownMessageTypeError(KeyError):
    """A type name that no imported generated module defines."""


def _type_name_text(type_name: str | bytes) -> str:
    if isinstance(type_name, bytes):
        return type_name.decode("utf-8", errors="replace")
    return type_name


def message_class(type_name: str | bytes) -> type[message.Message]:
    """The generated class of a full protobuf type name.

    Args:
        type_name: The full type name, such as "humanoid_mpc_msgs.MpcPolicy", as text or as the bus's second frame.

    Returns:
        The generated message class, from the default descriptor pool.

    Raises:
        UnknownMessageTypeError: no imported module defines the type.
    """
    name = _type_name_text(type_name)
    try:
        descriptor = descriptor_pool.Default().FindMessageTypeByName(name)
    except KeyError:
        raise UnknownMessageTypeError(
            f"no imported protobuf module defines '{name}': import its *_pb2 module first"
        ) from None
    return message_factory.GetMessageClass(descriptor)


def decode(type_name: str | bytes, payload: bytes) -> message.Message:
    """Parses `payload` as the message type `type_name` names.

    Raises:
        UnknownMessageTypeError: no imported module defines the type.
        google.protobuf.message.DecodeError: the payload does not parse as that type.
    """
    return message_class(type_name).FromString(payload)
