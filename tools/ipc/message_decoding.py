"""Decodes bus payloads by their full protobuf type name and prints them as protobuf text format or JSON.

The second frame of every bus message names its type (`humanoid_mpc_msgs.MpcPolicy`), so a payload decodes without a
topic registry: once every generated module of the message package is imported, the default descriptor pool knows
every type, and the message factory builds the class from the descriptor.
"""

import base64
import importlib
import io
import json
import pkgutil
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    TextIO,
    Type,
    Union,
)

from google.protobuf import descriptor
from google.protobuf import descriptor_pool
from google.protobuf import message
from google.protobuf import message_factory
from google.protobuf import text_format

MESSAGE_PACKAGE = "humanoid_mpc_msgs"
GENERATED_MODULE_SUFFIX = "_pb2"


class UnknownMessageTypeError(LookupError):
    """The default descriptor pool has no message of the type name a bus message carries."""


class FieldPathError(ValueError):
    """A --fields path names no field of the message."""


def import_message_modules(package_name: str = MESSAGE_PACKAGE) -> List[str]:
    """Imports every generated `*_pb2` module of a package, so that the default pool knows its messages.

    Returns the full names of the modules, sorted. The package is scanned rather than listed, so a new .proto file is
    decodable without touching this tool.
    """
    package = importlib.import_module(package_name)
    names = sorted(
        {
            f"{package_name}.{module.name}"
            for module in pkgutil.iter_modules(package.__path__)
            if module.name.endswith(GENERATED_MODULE_SUFFIX)
        }
    )
    for name in names:
        importlib.import_module(name)
    return names


def message_class(type_name: str) -> Type[message.Message]:
    """The generated class of a full type name, from the default descriptor pool."""
    try:
        message_descriptor = descriptor_pool.Default().FindMessageTypeByName(type_name)
    except KeyError as error:
        raise UnknownMessageTypeError(
            f"unknown message type '{type_name}' (not in the default descriptor pool; is its module imported?)"
        ) from error
    return message_factory.GetMessageClass(message_descriptor)


def decode(type_name: str, payload: bytes) -> message.Message:
    """Parses a payload as the message type its bus message names."""
    return message_class(type_name).FromString(payload)


def _is_repeated(field: descriptor.FieldDescriptor) -> bool:
    is_repeated = getattr(field, "is_repeated", None)
    if is_repeated is not None:
        return bool(is_repeated)
    return field.label == descriptor.FieldDescriptor.LABEL_REPEATED


def _is_map(field: descriptor.FieldDescriptor) -> bool:
    return (
        field.message_type is not None
        and field.message_type.GetOptions().map_entry
        and _is_repeated(field)
    )


def _scalar_to_python(value: Any, field: descriptor.FieldDescriptor) -> Any:
    if field.message_type is not None:
        return message_to_python(value)
    if field.enum_type is not None:
        enum_value = field.enum_type.values_by_number.get(value)
        return enum_value.name if enum_value is not None else value
    if field.type == descriptor.FieldDescriptor.TYPE_BYTES:
        # The proto3 JSON mapping of bytes.
        return base64.b64encode(value).decode("ascii")
    return value


def field_to_python(value: Any, field: descriptor.FieldDescriptor) -> Any:
    """A field's value as plain Python: messages as dicts, repeated fields as lists, enums by name."""
    if _is_map(field):
        value_field = field.message_type.fields_by_name["value"]
        return {key: _scalar_to_python(value[key], value_field) for key in value}
    if _is_repeated(field):
        return [_scalar_to_python(element, field) for element in value]
    return _scalar_to_python(value, field)


def message_to_python(msg: message.Message) -> Dict[str, Any]:
    """Every field of a message, set or not, as plain Python, in declaration order.

    Fields at their default value are included, so that a zero or a false reads as such rather than as a missing
    entry.
    """
    return {
        field.name: field_to_python(getattr(msg, field.name), field)
        for field in msg.DESCRIPTOR.fields
    }


# A component of a resolved field path: a field name, or a non-negative index into a repeated field.
PathStep = Union[str, int]


class _ResolvedPath(NamedTuple):
    steps: List[PathStep]
    value: Any
    field: descriptor.FieldDescriptor
    # The path ends at a repeated or map field itself rather than at one of its elements.
    ends_at_container: bool


def _resolve_path(msg: message.Message, path: str) -> _ResolvedPath:
    """Walks a dotted field path; raises FieldPathError naming what is wrong."""
    components = path.split(".")
    if not path or any(not component for component in components):
        raise FieldPathError(f"'{path}' is not a dotted field path")

    steps: List[PathStep] = []
    current: Any = msg
    field: Optional[descriptor.FieldDescriptor] = None
    indexable = False
    for position, component in enumerate(components):
        prefix = ".".join(components[:position]) or "<message>"
        if indexable:
            try:
                index = int(component)
            except ValueError as error:
                raise FieldPathError(
                    f"{path}: {prefix} is a repeated field; expected an index, got '{component}'"
                ) from error
            if not -len(current) <= index < len(current):
                raise FieldPathError(
                    f"{path}: index {index} is out of range for {prefix}, which has {len(current)} element(s)"
                )
            steps.append(index % len(current))
            current = current[index]
            indexable = False
            continue
        if not isinstance(current, message.Message):
            raise FieldPathError(
                f"{path}: {prefix} is a scalar and has no field '{component}'"
            )
        fields = current.DESCRIPTOR.fields_by_name
        if component not in fields:
            raise FieldPathError(
                f"{path}: {current.DESCRIPTOR.full_name} has no field '{component}'; its fields are: "
                f"{', '.join(f.name for f in current.DESCRIPTOR.fields)}"
            )
        field = fields[component]
        steps.append(component)
        current = getattr(current, component)
        indexable = _is_repeated(field) and not _is_map(field)

    assert field is not None  # A valid path names at least one field.
    return _ResolvedPath(steps, current, field, indexable or _is_map(field))


def select_field(msg: message.Message, path: str) -> Any:
    """The value at a dotted field path, as plain Python.

    Components name fields; a component after a repeated field is an index (negative counts from the end), so
    `state_trajectory.0.data` is the first state of a policy and `solver_status.healthy` a nested scalar.
    """
    resolved = _resolve_path(msg, path)
    if resolved.ends_at_container:
        return field_to_python(resolved.value, resolved.field)
    return _scalar_to_python(resolved.value, resolved.field)


# What to print of a message or a repeated field: the field names of a message, or the indices of a repeated field,
# each mapped to what to print of it. None prints all of it.
Selection = Optional[Dict[PathStep, "Selection"]]


def selection_of(msg: message.Message, paths: Sequence[str]) -> Selection:
    """The selection the dotted field paths make of a message (see select_field); None, all of it, without paths.

    A path that selects a field whole wins over the paths into it.
    """
    if not paths:
        return None
    selection: Dict[PathStep, Selection] = {}
    for path in paths:
        steps = _resolve_path(msg, path).steps
        node = selection
        for position, step in enumerate(steps):
            if step in node and node[step] is None:
                break
            if position == len(steps) - 1:
                node[step] = None
            else:
                node = node.setdefault(step, {})
    return selection


_TEXT_INDENT = 2


def _text_scalar(field: descriptor.FieldDescriptor, value: Any) -> str:
    """One scalar or enum value as the protobuf text format writes it: strings quoted and escaped, enums by name."""
    out = io.StringIO()
    text_format.PrintFieldValue(field, value, out, as_utf8=True)
    return out.getvalue()


def _indices_comment(selection: Selection, size: int) -> str:
    if selection is None:
        return ""
    return f"  # [{', '.join(str(index) for index in sorted(selection))}] of {size}"


def _write_field_text(
    field: descriptor.FieldDescriptor,
    value: Any,
    selection: Selection,
    indent: int,
    out: TextIO,
) -> None:
    pad = " " * indent
    if _is_map(field):
        if not value:
            out.write(f"{pad}{field.name}: []\n")
            return
        entry_class = value.GetEntryClass()
        for key in sorted(value):
            out.write(f"{pad}{field.name} {{\n")
            _write_message_text(
                entry_class(key=key, value=value[key]),
                None,
                indent + _TEXT_INDENT,
                out,
            )
            out.write(f"{pad}}}\n")
    elif _is_repeated(field):
        indices = list(range(len(value))) if selection is None else sorted(selection)
        if field.message_type is None:
            elements = ", ".join(_text_scalar(field, value[i]) for i in indices)
            out.write(
                f"{pad}{field.name}: [{elements}]{_indices_comment(selection, len(value))}\n"
            )
            return
        if not indices:
            out.write(f"{pad}{field.name}: []\n")
        for index in indices:
            comment = "" if selection is None else f"  # [{index}] of {len(value)}"
            out.write(f"{pad}{field.name} {{{comment}\n")
            _write_message_text(
                value[index],
                None if selection is None else selection[index],
                indent + _TEXT_INDENT,
                out,
            )
            out.write(f"{pad}}}\n")
    elif field.message_type is not None:
        out.write(f"{pad}{field.name} {{\n")
        _write_message_text(value, selection, indent + _TEXT_INDENT, out)
        out.write(f"{pad}}}\n")
    else:
        out.write(f"{pad}{field.name}: {_text_scalar(field, value)}\n")


def _write_message_text(
    msg: message.Message, selection: Selection, indent: int, out: TextIO
) -> None:
    for field in msg.DESCRIPTOR.fields:
        if selection is not None and field.name not in selection:
            continue
        if field.containing_oneof is not None and not msg.HasField(field.name):
            # A member of a oneof, a proto3 `optional` field included, tracks presence: printing its default would
            # set it when the text is parsed back, and set two members of one oneof.
            out.write(f"{' ' * indent}# {field.name}: not set\n")
            continue
        _write_field_text(
            field,
            getattr(msg, field.name),
            None if selection is None else selection[field.name],
            indent,
            out,
        )


def message_to_text(msg: message.Message, selection: Selection = None) -> str:
    """A message in protobuf text format, with every field, in declaration order.

    Unlike text_format.MessageToString, fields at their default value are printed too (`healthy: false`, `data: []`,
    an unset submessage as its defaults), so that a zero or a false reads as such rather than as a missing line. Repeated scalars are printed as one list, `data: [0.0, 0.5]`. A member of a oneof
    that is not set is a comment. The text parses back with text_format.Parse into a message of equal values; only
    the presence of the submessages that were unset differs.

    With a selection (selection_of), only the selected fields are printed, so that the text is still a textproto of
    the message's type; a repeated field of which only some elements are selected carries a comment with their
    indices.
    """
    out = io.StringIO()
    _write_message_text(msg, selection, 0, out)
    return out.getvalue().rstrip("\n")


def _format_text(msg: message.Message, paths: Sequence[str]) -> str:
    return message_to_text(msg, selection_of(msg, paths))


def _format_json(msg: message.Message, paths: Sequence[str]) -> str:
    if not paths:
        return json.dumps(message_to_python(msg), indent=2)
    return json.dumps({path: select_field(msg, path) for path in paths}, indent=2)


# LINT.IfChange(output_formats)
OUTPUT_FORMATS: Dict[str, Callable[[message.Message, Sequence[str]], str]] = {
    "text": _format_text,
    "json": _format_json,
}
# LINT.ThenChange(//tools/ipc/README.md:output_formats)


def format_fields(
    msg: message.Message, paths: Sequence[str], output_format: str
) -> str:
    """The message, or only the fields at `paths`, printed in one of OUTPUT_FORMATS.

    Raises FieldPathError when a path is not a field of the message, and ValueError for an unknown format.
    """
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(
            f"unknown output format '{output_format}'; the formats are: {', '.join(OUTPUT_FORMATS)}"
        )
    return OUTPUT_FORMATS[output_format](msg, paths)
