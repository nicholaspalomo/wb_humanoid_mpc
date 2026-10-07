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


"""Edits of a configuration file: applied without losing a byte they do not touch, checked, and saved atomically.

    from config_textproto import textproto_save

    result = textproto_save.save(
        "config/mpc/task.textproto",
        task_file_pb2.TaskFile,
        [textproto_save.SetValue("state_weights.scaling", 90.0)],
    )
    result.message  # the file as it now is, parsed strictly

Every edit is made twice: to the text, through textproto_document (a value replaces only its token; a field the file
does not have is inserted at the end of its block, with every block and keyed element it needs; a removed field takes
its line), and to the file's parsed message, through protobuf reflection. The edited text is then parsed strictly
(nproto_textproto) and must be the edited message exactly, field for field and bit for bit; otherwise the edit is
refused with the difference and nothing is written. A value is written as its field's type spells it: a value the file
already holds keeps the file's spelling (5 stays 5, 5e-5 stays 5e-5); a double gets the shortest digits that read back
to it (repr), a float the shortest that read back to the same float; integers are decimal, booleans true or false,
enums by name, strings quoted and escaped.

save() writes a temporary file next to the file, flushes and fsyncs it and renames it over the file, so the 1 Hz file
watchers of the MPC and the robot never read half a file. ensure_backup() keeps the file as it was before the first
save, as the GUI's tabs have always done.
"""

from collections.abc import Mapping, Sequence
import contextlib
import dataclasses
import difflib
import math
import os
import shutil
import stat
import struct
import tempfile
from typing import Any, TypeAlias, TypeVar

from google.protobuf import descriptor as descriptor_module
from google.protobuf import message as message_module
from google.protobuf import text_format

from config_textproto import textproto_document
import nproto_textproto

MessageT = TypeVar("MessageT", bound=message_module.Message)
FieldDescriptor: TypeAlias = descriptor_module.FieldDescriptor

ScalarValue: TypeAlias = bool | int | float | str | bytes

# The integer field types and the values they hold.
_INTEGER_RANGES = {
    FieldDescriptor.TYPE_INT32: (-(2**31), 2**31 - 1),
    FieldDescriptor.TYPE_SINT32: (-(2**31), 2**31 - 1),
    FieldDescriptor.TYPE_SFIXED32: (-(2**31), 2**31 - 1),
    FieldDescriptor.TYPE_UINT32: (0, 2**32 - 1),
    FieldDescriptor.TYPE_FIXED32: (0, 2**32 - 1),
    FieldDescriptor.TYPE_INT64: (-(2**63), 2**63 - 1),
    FieldDescriptor.TYPE_SINT64: (-(2**63), 2**63 - 1),
    FieldDescriptor.TYPE_SFIXED64: (-(2**63), 2**63 - 1),
    FieldDescriptor.TYPE_UINT64: (0, 2**64 - 1),
    FieldDescriptor.TYPE_FIXED64: (0, 2**64 - 1),
}
_FLOATING_TYPES = (FieldDescriptor.TYPE_DOUBLE, FieldDescriptor.TYPE_FLOAT)
# The permissions of a file write_atomically() creates (mkstemp's own are 0600).
_NEW_FILE_MODE = 0o644
# The most lines of the difference a refused edit shows.
_DIFFERENCE_LINES = 40


class SaveError(ValueError):
    """An edit that the file or its schema cannot take, or an edited text that would not be the edited message."""


@dataclasses.dataclass(frozen=True)
class SetValue:
    """Sets the scalar at `path` to `value`; a field the file lacks is inserted, with the blocks and elements it needs.

    Attributes:
      path: A textproto_document path; an element of a repeated scalar by its index (the next index appends).
      value: The value, in the field's type: a bool, an int, a float (an int for a double or a float field too), a str
        (a string, or an enum value's name), bytes, or an enum value's number.
    """

    path: str
    value: ScalarValue


@dataclasses.dataclass(frozen=True)
class AppendValue:
    """Adds `value` after the last element of the repeated field `path` (no selector on its last segment).

    Attributes:
      path: The repeated field.
      value: A scalar, or a message as a mapping of its fields to their values (nested mappings for blocks, sequences
        for repeated fields).
    """

    path: str
    value: ScalarValue | Mapping[str, object]


@dataclasses.dataclass(frozen=True)
class RemoveField:
    """Removes the field or element at `path`, or every element of a repeated field named without a selector.

    The schema default applies to a removed field again. A path the file does not have is a no-op.

    Attributes:
      path: The field or element.
    """

    path: str


Edit: TypeAlias = SetValue | AppendValue | RemoveField


@dataclasses.dataclass(frozen=True)
class EditResult:
    """What applying edits to a file gives.

    Attributes:
      text: The edited text.
      message: The message the edited text parses into, which is the file's message with the edits made.
      changed: Whether the text differs from the file's.
    """

    text: str
    message: Any
    changed: bool


def apply_edits(
    text: str, message_class: type[MessageT], edits: Sequence[Edit], source: str
) -> EditResult:
    """Applies `edits` to the textproto `text` of a `message_class`, checking the result (the module docstring).

    Args:
      text: The file's text; it must parse strictly.
      message_class: The generated class of its schema.
      edits: The edits, in order.
      source: Names the text in errors, usually its path.

    Returns:
      The edited text and its message.

    Raises:
      SaveError: `text` does not parse, an edit does not fit the schema or the file, or the edited text is not the
        edited message.
    """
    message = message_class()
    try:
        nproto_textproto.parse_textproto(text, message, source)
        document = textproto_document.parse(text, source)
    except (nproto_textproto.TextprotoError, textproto_document.DocumentError) as error:
        raise SaveError(str(error)) from error
    return edit(document, message, edits)


def edit(
    document: textproto_document.TextprotoDocument, message: Any, edits: Sequence[Edit]
) -> EditResult:
    """Applies `edits` to a file already parsed: its document and its message, neither of which changes.

    What a tab that keeps the file it shows does on every change: apply the edits made so far, publish the message,
    and save the text when asked.

    Args:
      document: The file's document.
      message: The message its text parses into (strictly).
      edits: The edits, in order.

    Returns:
      The edited text and its message.

    Raises:
      SaveError: An edit does not fit the schema or the file, or the edited text is not the edited message.
    """
    edited_document = document.copy()
    intended = type(message)()
    intended.CopyFrom(message)
    batch = _ValueBatch(edited_document)
    for one_edit in edits:
        try:
            if not (isinstance(one_edit, SetValue) and batch.add(intended, one_edit)):
                batch.write()
                _apply(edited_document, intended, one_edit, document.source)
        except textproto_document.DocumentError as error:
            raise SaveError(str(error)) from error
    try:
        batch.write()
    except textproto_document.DocumentError as error:
        raise SaveError(str(error)) from error
    text = edited_document.text()
    if text == document.text() and _serialized(intended) == _serialized(message):
        # The text that parsed into `message`, unchanged: no need to parse it again.
        return EditResult(text, intended, changed=False)
    return EditResult(text, _check(text, intended, document.source), changed=True)


class _ValueBatch:
    """Values of a document that its file has, written together with one parse of the text.

    A batched value is set in the message at once and written to the text by write(), which every other edit calls
    first. An edit that selects an element by a field a batched edit sets is not batched, since the text does not yet
    show the field's new value; the edits then have the meaning they have one after another.
    """

    def __init__(self, document: textproto_document.TextprotoDocument) -> None:
        self._document = document
        self._values: list[tuple[str, textproto_document.ScalarValue]] = []
        self._fields: set[str] = set()

    def add(self, message: Any, change: SetValue) -> bool:
        """Sets `change` in `message` and keeps its value for the text; False (nothing done) when it cannot wait."""
        segments = textproto_document.parse_path(change.path)
        if not segments or any(
            segment.key is not None and segment.key[0] in self._fields
            for segment in segments
        ):
            return False
        old = self._document.find(change.path)
        if not isinstance(old, textproto_document.ScalarNode):
            return False
        where = _Where(self._document.source, change.path)
        field, converted = _set_in_message(message, segments, change.value, where)
        self._values.append((change.path, _literal(field, converted, old)))
        self._fields.add(segments[-1].name)
        return True

    def write(self) -> None:
        """Writes the batched values to the document (DocumentError as set_scalars()), and starts a new batch."""
        values, self._values = self._values, []
        self._fields = set()
        self._document.set_scalars(values)


def save(path: str, message_class: type[MessageT], edits: Sequence[Edit]) -> EditResult:
    """Applies `edits` to the file at `path` (apply_edits()) and writes the result atomically when the text changed.

    Args:
      path: The file.
      message_class: The generated class of its schema.
      edits: The edits, in order.

    Returns:
      The edited text and its message.

    Raises:
      OSError: The file cannot be read or written.
      SaveError: As apply_edits(); the file is then untouched.
    """
    with open(path, encoding="utf-8", newline="") as file:
        text = file.read()
    result = apply_edits(text, message_class, edits, path)
    if result.changed:
        write_atomically(path, result.text)
    return result


def write_atomically(path: str, text: str) -> None:
    """Replaces the file at `path` (through a symbolic link: its target) with `text` in one rename.

    The text goes to a temporary file in the same directory, which is flushed, fsynced, given the file's permissions
    and renamed over it; the directory is fsynced after the rename. A reader sees the old file or the new one, never
    part of either. On an error the temporary file is removed and the file is untouched.

    Args:
      path: The file; a new one, with permissions 0644, when it does not exist.
      text: Its new content, written as UTF-8 with its line endings as they are.

    Raises:
      OSError: The directory cannot be written.
    """
    target = os.path.realpath(path)
    directory = os.path.dirname(target)
    mode = (
        stat.S_IMODE(os.stat(target).st_mode)
        if os.path.exists(target)
        else _NEW_FILE_MODE
    )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
    directory_descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def ensure_backup(path: str) -> str:
    """Copies the file at `path` to `<path>.bak` unless that exists, keeping the file as it was before the first save.

    Args:
      path: The file.

    Returns:
      The backup's path.

    Raises:
      OSError: The copy cannot be made.
    """
    backup = f"{path}.bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    return backup


# ---- One edit, in the text and in the message ----


def _apply(
    document: textproto_document.TextprotoDocument,
    message: Any,
    change: Edit,
    source: str,
) -> None:
    """Makes one edit in `document` and in `message`."""
    segments = textproto_document.parse_path(change.path)
    where = _Where(source, change.path)
    if not segments:
        raise where.error("an edit names a field, not the file")
    if isinstance(change, SetValue):
        _set(document, message, segments, change.value, where)
    elif isinstance(change, AppendValue):
        _append(document, message, segments, change.value, where)
    else:
        _remove(document, message, segments, where)


@dataclasses.dataclass(frozen=True)
class _Where:
    """The file and the path of an edit, which its errors name."""

    source: str
    path: str

    def error(self, problem: str) -> SaveError:
        return SaveError(f"{self.source}: {self.path}: {problem}")


def _set(
    document: textproto_document.TextprotoDocument,
    message: Any,
    segments: Sequence[textproto_document.PathSegment],
    value: ScalarValue,
    where: _Where,
) -> None:
    """Makes a SetValue edit in `document` and in `message`."""
    old = document.find(where.path)
    field, converted = _set_in_message(message, segments, value, where)
    document.set_or_insert(where.path, _literal(field, converted, old))


def _set_in_message(
    message: Any,
    segments: Sequence[textproto_document.PathSegment],
    value: ScalarValue,
    where: _Where,
) -> tuple[FieldDescriptor, Any]:
    """Sets the value at `segments` of `message`, creating a keyed element it needs; (its field, the value set)."""
    parent, field, last = _navigate(message, segments, where, create=True)
    if field.message_type is not None:
        raise where.error(f"{field.name} is a block; set the values in it")
    converted = _convert(field, value, where)
    if field.is_repeated:
        container = getattr(parent, field.name)
        if last.index is None or last.key is not None:
            raise where.error(
                f"{field.name} is repeated; select an element with [index]"
            )
        if last.index > len(container):
            raise where.error(
                f"{field.name} has {len(container)} elements; [{last.index}] is not one of them or the next"
            )
        if last.index == len(container):
            container.append(converted)
        else:
            container[last.index] = converted
    else:
        if last.index is not None or last.key is not None:
            raise where.error(f"{field.name} is not repeated")
        setattr(parent, field.name, converted)
    return field, converted


def _append(
    document: textproto_document.TextprotoDocument,
    message: Any,
    segments: Sequence[textproto_document.PathSegment],
    value: ScalarValue | Mapping[str, object],
    where: _Where,
) -> None:
    """Makes an AppendValue edit in `document` and in `message`."""
    parent, field, last = _navigate(message, segments, where, create=True)
    if last.index is not None or last.key is not None:
        raise where.error("append names the repeated field, without a selector")
    if not field.is_repeated:
        raise where.error(f"{field.name} is not repeated")
    container = getattr(parent, field.name)
    if field.message_type is None:
        if isinstance(value, Mapping):
            raise where.error(f"{field.name} holds values, not blocks")
        converted = _convert(field, value, where)
        container.append(converted)
        document.append(where.path, _literal(field, converted, None))
        return
    if not isinstance(value, Mapping):
        raise where.error(
            f"{field.name} holds blocks: append a mapping of their fields"
        )
    document.append(where.path, _fill(container.add(), value, where))


def _remove(
    document: textproto_document.TextprotoDocument,
    message: Any,
    segments: Sequence[textproto_document.PathSegment],
    where: _Where,
) -> None:
    """Makes a RemoveField edit in `document` and in `message`; nothing when the file does not have the field."""
    if document.remove(where.path) == 0:
        return
    parent, field, last = _navigate(message, segments, where, create=False)
    if not field.is_repeated:
        parent.ClearField(field.name)
        return
    container = getattr(parent, field.name)
    if last.index is not None:
        del container[last.index]
    elif last.key is not None:
        del container[_key_index(container, field, last.key, where)]
    else:
        parent.ClearField(field.name)


def _navigate(
    message: Any,
    segments: Sequence[textproto_document.PathSegment],
    where: _Where,
    create: bool,
) -> tuple[Any, FieldDescriptor, textproto_document.PathSegment]:
    """(the message that holds the last segment's field, that field, the last segment) of a path through `message`."""
    current = message
    for segment in segments[:-1]:
        field = _field(current, segment, where)
        if field.message_type is None:
            raise where.error(f"{segment.name} is a value, not a block")
        if field.is_repeated:
            current = _element(
                getattr(current, segment.name), field, segment, where, create
            )
        elif segment.index is not None or segment.key is not None:
            raise where.error(f"{segment.name} is not repeated")
        else:
            current = getattr(current, segment.name)
    return current, _field(current, segments[-1], where), segments[-1]


def _field(
    message: Any, segment: textproto_document.PathSegment, where: _Where
) -> FieldDescriptor:
    descriptor = message.DESCRIPTOR
    field = descriptor.fields_by_name.get(segment.name)
    if field is None:
        raise where.error(f"{descriptor.full_name} has no field {segment.name!r}")
    if field.message_type is not None and field.message_type.GetOptions().map_entry:
        raise where.error(
            f"{segment.name} is a map; the editor edits repeated messages with a key field instead"
        )
    return field


def _key_matches(element: Any, key_field: FieldDescriptor, text: str) -> bool:
    """Whether `element`'s key field `key_field` has the value `text`, compared as textproto_document compares a key."""
    if key_field.has_presence and not element.HasField(key_field.name):
        return False
    value = getattr(element, key_field.name)
    if key_field.type == FieldDescriptor.TYPE_STRING:
        return bool(value == text)
    if key_field.type == FieldDescriptor.TYPE_ENUM:
        enum_value = key_field.enum_type.values_by_name.get(text)
        return enum_value is not None and bool(value == enum_value.number)
    if key_field.type == FieldDescriptor.TYPE_BOOL:
        return text == ("true" if value else "false")
    try:
        return bool(textproto_document.parse_number(text) == value)
    except ValueError:
        return False


def _key_field(
    field: FieldDescriptor, key: tuple[str, str], where: _Where
) -> FieldDescriptor:
    """The field of `field`'s element type that `key` selects its elements by; a scalar, else SaveError."""
    key_field = field.message_type.fields_by_name.get(key[0])
    if key_field is None or key_field.message_type is not None or key_field.is_repeated:
        raise where.error(
            f"{key[0]} is not a value of {field.message_type.full_name} to select {field.name} by"
        )
    return key_field


def _key_index(
    container: Any, field: FieldDescriptor, key: tuple[str, str], where: _Where
) -> int:
    """The index of the one element of `container` that `key` selects."""
    key_field = _key_field(field, key, where)
    matches = [
        index
        for index, element in enumerate(container)
        if _key_matches(element, key_field, key[1])
    ]
    if len(matches) != 1:
        raise where.error(
            f"{len(matches)} elements of {field.name} have {key[0]} {key[1]}"
        )
    return matches[0]


def _element(
    container: Any,
    field: FieldDescriptor,
    segment: textproto_document.PathSegment,
    where: _Where,
    create: bool,
) -> Any:
    """The element of the repeated message field `container` that `segment` selects, created when `create` allows."""
    if segment.index is not None:
        if segment.index < len(container):
            return container[segment.index]
        if create and segment.index == len(container):
            return container.add()
        raise where.error(
            f"{field.name} has {len(container)} elements, not [{segment.index}]"
        )
    if segment.key is not None:
        key = segment.key
        key_field = _key_field(field, key, where)
        matches = [
            element for element in container if _key_matches(element, key_field, key[1])
        ]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise where.error(
                f"{len(matches)} elements of {field.name} have {key[0]} {key[1]}"
            )
        if not create:
            raise where.error(f"no element of {field.name} has {key[0]} {key[1]}")
        if key_field.type != FieldDescriptor.TYPE_STRING:
            raise where.error(
                f"a new element of {field.name} is selected by a string key, not by {key_field.name}"
            )
        element = container.add()
        setattr(element, key_field.name, key[1])
        return element
    if len(container) == 1:
        return container[0]
    if create and not container:
        return container.add()
    raise where.error(
        f"{field.name} has {len(container)} elements; select one with [index] or [field=value]"
    )


def _fill(
    message: Any, fields: Mapping[str, object], where: _Where
) -> textproto_document.NewBlock:
    """Sets the `fields` of the new, empty `message` and returns the block that writes them, in the same order."""
    message.SetInParent()
    block: list[tuple[str, textproto_document.Value]] = []
    for name, value in fields.items():
        field = _field(message, textproto_document.PathSegment(name), where)
        values = value if field.is_repeated else [value]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise where.error(f"{name} is repeated: give a sequence of its values")
        for item in values:
            if field.message_type is not None:
                if not isinstance(item, Mapping):
                    raise where.error(
                        f"{name} is a block: give a mapping of its fields"
                    )
                child = (
                    getattr(message, name).add()
                    if field.is_repeated
                    else getattr(message, name)
                )
                block.append((name, _fill(child, item, where)))
                continue
            converted = _convert(field, item, where)
            if field.is_repeated:
                getattr(message, name).append(converted)
            else:
                setattr(message, name, converted)
            block.append((name, _literal(field, converted, None)))
    return textproto_document.NewBlock(tuple(block))


# ---- Values ----


def _convert(field: FieldDescriptor, value: object, where: _Where) -> Any:
    """`value` as reflection stores it in `field`: checked against the field's type and range."""
    kind = field.type
    if kind == FieldDescriptor.TYPE_BOOL:
        if not isinstance(value, bool):
            raise where.error(f"{field.name} is a bool, not {value!r}")
        return value
    if kind in _INTEGER_RANGES:
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise where.error(f"{field.name} is an integer, not {value!r}")
        low, high = _INTEGER_RANGES[kind]
        if not low <= value <= high:
            raise where.error(
                f"{value} is out of the range of {field.name}, [{low}, {high}]"
            )
        return value
    if kind in _FLOATING_TYPES:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise where.error(f"{field.name} is a number, not {value!r}")
        number = float(value)
        if kind == FieldDescriptor.TYPE_FLOAT:
            try:
                return _to_float32(number)
            except OverflowError as error:
                raise where.error(
                    f"{value} is out of the range of the float {field.name}"
                ) from error
        return number
    if kind == FieldDescriptor.TYPE_STRING:
        if not isinstance(value, str):
            raise where.error(f"{field.name} is a string, not {value!r}")
        return value
    if kind == FieldDescriptor.TYPE_BYTES:
        if not isinstance(value, bytes):
            raise where.error(f"{field.name} is bytes, not {value!r}")
        return value
    if kind == FieldDescriptor.TYPE_ENUM:
        return _enum_number(field, value, where)
    raise where.error(f"{field.name} has a type the editor does not write ({kind})")


def _enum_number(field: FieldDescriptor, value: object, where: _Where) -> int:
    enum_type = field.enum_type
    if isinstance(value, str) and value in enum_type.values_by_name:
        return int(enum_type.values_by_name[value].number)
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value in enum_type.values_by_number
    ):
        return value
    names = ", ".join(enum_value.name for enum_value in enum_type.values)
    raise where.error(f"{value!r} is not a value of {enum_type.full_name}: {names}")


def _to_float32(value: float) -> float:
    """`value` rounded to the nearest float; OverflowError for a finite value beyond the float range."""
    return float(struct.unpack("<f", struct.pack("<f", value))[0])


def _same_float32(first: float, second: float) -> bool:
    if math.isnan(first) or math.isnan(second):
        return math.isnan(first) and math.isnan(second)
    try:
        return struct.pack("<f", first) == struct.pack("<f", second)
    except OverflowError:
        return False


def format_float32(value: float) -> str:
    """The shortest text that reads back to the float `value` (a float32), with inf, -inf and nan by name."""
    if math.isnan(value) or math.isinf(value):
        return textproto_document.format_double(value)
    for digits in range(1, 10):
        text = f"{value:.{digits}g}"
        if _same_float32(float(text), value):
            return text
    return repr(value)


def _literal(
    field: FieldDescriptor, converted: Any, old: object
) -> textproto_document.ScalarValue:
    """What the document writes for the value `converted` of `field`; `old` is the file's value node, if any."""
    old_node = old if isinstance(old, textproto_document.ScalarNode) else None
    if field.type == FieldDescriptor.TYPE_FLOAT:
        if old_node is not None and old_node.kind != textproto_document.STRING:
            try:
                if _same_float32(old_node.float_value(), converted):
                    return textproto_document.Literal(old_node.text)
            except ValueError:
                pass
        return textproto_document.Literal(format_float32(converted))
    if field.type == FieldDescriptor.TYPE_ENUM:
        if old_node is not None and old_node.kind == textproto_document.NUMBER:
            with contextlib.suppress(ValueError):
                if old_node.number_value() == converted:
                    return textproto_document.Literal(old_node.text)
        return textproto_document.Identifier(
            field.enum_type.values_by_number[converted].name
        )
    if field.type == FieldDescriptor.TYPE_DOUBLE:
        return float(converted)
    if isinstance(converted, (bool, int, str, bytes)):
        return converted
    raise TypeError(f"{field.name}: {converted!r} is not a scalar")


# ---- The check ----


def _serialized(message: Any) -> bytes:
    return bytes(message.SerializeToString(deterministic=True))


def _check(text: str, intended: Any, source: str) -> Any:
    """The message `text` parses into, which must be `intended` bit for bit; SaveError with the difference otherwise."""
    reparsed = type(intended)()
    try:
        nproto_textproto.parse_textproto(text, reparsed, source)
    except nproto_textproto.TextprotoError as error:
        raise SaveError(
            f"the edited text does not parse, so nothing is written: {error}"
        ) from error
    if reparsed.SerializeToString(deterministic=True) != intended.SerializeToString(
        deterministic=True
    ):
        raise SaveError(
            f"{source}: the edited text is not the edited message, so nothing is written:\n"
            f"{_difference(intended, reparsed)}"
        )
    return reparsed


def _difference(intended: Any, reparsed: Any) -> str:
    """The lines in which the text formats of the two messages differ."""
    lines = list(
        difflib.unified_diff(
            text_format.MessageToString(intended).splitlines(),
            text_format.MessageToString(reparsed).splitlines(),
            "edited message",
            "edited text",
            lineterm="",
        )
    )
    if len(lines) > _DIFFERENCE_LINES:
        lines = lines[:_DIFFERENCE_LINES] + [
            f"... {len(lines) - _DIFFERENCE_LINES} more lines"
        ]
    return (
        "\n".join(lines)
        if lines
        else "(the two serialize differently, e.g. 0.0 against -0.0 or NaN payloads)"
    )
