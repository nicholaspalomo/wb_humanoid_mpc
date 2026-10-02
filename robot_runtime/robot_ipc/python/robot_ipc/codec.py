"""Decodes a bus message from its type name, without a registry of topics (for tools/ipc).

The second frame of every bus message is the full protobuf type name of the third, such as
"humanoid_mpc_msgs.MpcPolicy". The class is found in the protobuf default descriptor pool, which knows every message
whose generated module (`*_pb2`) has been imported: import the modules of the messages you want to decode first.
"""

from typing import Type, Union

from google.protobuf import descriptor_pool
from google.protobuf import message
from google.protobuf import message_factory


class UnknownMessageTypeError(KeyError):
    """A type name that no imported generated module defines."""


def _type_name_text(type_name: Union[str, bytes]) -> str:
    if isinstance(type_name, bytes):
        return type_name.decode("utf-8", errors="replace")
    return type_name


def message_class(type_name: Union[str, bytes]) -> Type[message.Message]:
    """The generated class of a full protobuf type name.

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


def decode(type_name: Union[str, bytes], payload: bytes) -> message.Message:
    """Parses `payload` as the message type `type_name` names.

    Raises:
        UnknownMessageTypeError: no imported module defines the type.
        google.protobuf.message.DecodeError: the payload does not parse as that type.
    """
    return message_class(type_name).FromString(payload)
