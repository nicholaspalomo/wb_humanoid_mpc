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


"""A configuration file the operator tunes: the file as last loaded or saved, and the values changed since, without Tk.

    from remote_control import tuned_file

    task = tuned_file.TunedFile("config/mpc/task.textproto", task_file_pb2.TaskFile)
    for spec in task.parameters():                       # config_schema.ParameterSpec, from the schema
        ...
    task.set("state_weights.scaling", 90.0)              # a slider moved: nothing is written
    task.edited().message                                # the file with the change, parsed strictly: what is published
    task.save()                                          # written in place, every other byte kept

The tuning tabs are thin shells over this class: what a slider, a checkbox or a drop-down shows is a parameter of the
file's schema (config_schema), and what it changes is a value here, by the parameter's path. A change is held, not
written: the edited file is made on demand (textproto_save.edit(), which re-parses the edited text strictly and checks
it against the edited message), published from there, and written only by save(), which re-reads the file, makes the
same edits to it and replaces it atomically (textproto_save.save()). The file as loaded or last saved is the
checkpoint that reset() returns to.
"""

from collections.abc import Mapping, Sequence
import os
from typing import Any

from google.protobuf import message as message_module

from config_textproto import textproto_document
from config_textproto import textproto_save
import nproto_textproto
from remote_control import config_schema

# The kinds whose value a widget sets.
EDITABLE_KINDS = (
    config_schema.Kind.NUMBER,
    config_schema.Kind.INTEGER,
    config_schema.Kind.BOOL,
    config_schema.Kind.CHOICE,
)


class TunedFileError(ValueError):
    """A file that cannot be tuned: it does not parse into its schema, or an edit of it does not."""


class TunedFile:
    """One configuration file: its parameters, the changes the operator made to them, the edited file and saving.

    Args:
      path: The file, a textproto of `message_class`.
      message_class: The generated class of the file's schema.
      registries: The names of each registry, for the choices of registry strings (config_schema.parameters());
        None: unknown.

    Raises:
      OSError: The file cannot be read.
      TunedFileError: It does not parse strictly into `message_class`; the message names the line and column.
    """

    def __init__(
        self,
        path: str,
        message_class: type[message_module.Message],
        registries: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._path = os.path.abspath(path)
        self._message_class = message_class
        self._registries = registries
        self._changes: dict[str, Any] = {}
        self._message: Any = message_class()
        self._document = textproto_document.parse("", self._path)
        self._parameters: list[config_schema.ParameterSpec] = []
        self._by_path: dict[str, config_schema.ParameterSpec] = {}
        self.reload()

    @property
    def path(self) -> str:
        return self._path

    @property
    def message_class(self) -> type[message_module.Message]:
        return self._message_class

    @property
    def message(self) -> Any:
        """The file as loaded or last saved (the checkpoint), parsed."""
        return self._message

    @property
    def document(self) -> textproto_document.TextprotoDocument:
        """The text of the file as loaded or last saved."""
        return self._document

    def reload(self) -> None:
        """Reads the file again: it becomes the checkpoint, and every change is dropped.

        Raises:
          OSError: The file cannot be read.
          TunedFileError: It does not parse strictly into its schema.
        """
        with open(self._path, encoding="utf-8", newline="") as file:
            text = file.read()
        message = self._message_class()
        try:
            nproto_textproto.parse_textproto(text, message, self._path)
            document = textproto_document.parse(text, self._path)
        except (
            nproto_textproto.TextprotoError,
            textproto_document.DocumentError,
        ) as error:
            raise TunedFileError(str(error)) from error
        self._checkpoint(message, document)

    def _checkpoint(
        self, message: Any, document: textproto_document.TextprotoDocument
    ) -> None:
        self._message = message
        self._document = document
        self._changes = {}
        self._parameters = config_schema.parameters(message, document, self._registries)
        self._by_path = {spec.path: spec for spec in self._parameters}

    def parameters(self) -> list[config_schema.ParameterSpec]:
        """Every parameter of the file as loaded or last saved (config_schema.parameters()), in schema order."""
        return list(self._parameters)

    def rendered(self) -> list[config_schema.ParameterSpec]:
        """The parameters the GUI gives a widget (ParameterSpec.renders)."""
        return [spec for spec in self._parameters if spec.renders]

    def spec(self, path: str) -> config_schema.ParameterSpec:
        """The parameter at `path`; KeyError when the file has none there."""
        return self._by_path[path]

    def has(self, path: str) -> bool:
        return path in self._by_path

    def saved_value(self, path: str) -> Any:
        """The value of the parameter at `path` in the file as loaded or last saved."""
        return self.spec(path).value

    def value(self, path: str) -> Any:
        """The value of the parameter at `path`: the operator's, or the file's."""
        if path in self._changes:
            return self._changes[path]
        return self.saved_value(path)

    def set(self, path: str, value: Any) -> None:
        """Changes the parameter at `path` to `value`; setting it back to the file's value drops the change.

        Args:
          path: A parameter of the file.
          value: Its new value, in its kind: a number (an int for an integer, which a float is rounded to), a bool, or
            the name of a choice; None: the file's value again, which for a parameter the file leaves unset is no value
            at all (a row reset to unset), as discard() does.

        Raises:
          KeyError: The file has no parameter at `path`.
          ValueError: The parameter is not one a widget edits (a name list or text).
        """
        spec = self.spec(path)
        if spec.kind not in EDITABLE_KINDS:
            raise ValueError(
                f"{path} is a {spec.kind.value}, which the GUI does not edit"
            )
        if value is None:
            self.discard(path)
            return
        converted = _converted(spec, value)
        if spec.value is not None and _same(spec.value, converted):
            # Back to the file's value, or a default left at the default: nothing to write into the file.
            self._changes.pop(path, None)
        else:
            self._changes[path] = converted

    def changes(self) -> dict[str, Any]:
        """The parameters the operator changed, by path, with their new values, in the order they were first changed."""
        return dict(self._changes)

    def discard(self, path: str) -> None:
        """Drops the change of the parameter at `path`, if any: its value is the file's again."""
        self._changes.pop(path, None)

    def reset(self) -> None:
        """Drops every change: every value is the file's as loaded or last saved."""
        self._changes = {}

    def edits(self) -> list[textproto_save.Edit]:
        """The changes, as the edits of textproto_save."""
        return [
            textproto_save.SetValue(path, value)
            for path, value in self._changes.items()
        ]

    def edited(self) -> textproto_save.EditResult:
        """The file with the changes made: its text, and its message parsed strictly from that text.

        Raises:
          TunedFileError: The edited text would not parse into the edited message (textproto_save.SaveError).
        """
        try:
            return textproto_save.edit(self._document, self._message, self.edits())
        except textproto_save.SaveError as error:
            raise TunedFileError(str(error)) from error

    def save(self) -> textproto_save.EditResult:
        """Writes the changes into the file, re-read for it, atomically; the result becomes the checkpoint.

        Every byte the changes do not touch - comments, blank lines, LINT directives, values the file spells its own way
        - is kept, and an edit of the file made since it was loaded is kept too, unless it is to one of the changed
        values. The file keeps its first version as `<file>.bak` (textproto_save.ensure_backup()).

        Returns:
          The text and its message. The text is exactly what the file holds after the save, read back as UTF-8 with its
          line endings as they are: the text written, or the file's own when there are no changes. The tabs send it to
          the robot's copy (robot_config_save.py).

        Raises:
          OSError: The file cannot be read or written.
          TunedFileError: The changes do not fit the file (a parameter it no longer has, say); nothing is written.
        """
        if self._changes:
            textproto_save.ensure_backup(self._path)
        try:
            result = textproto_save.save(self._path, self._message_class, self.edits())
        except textproto_save.SaveError as error:
            raise TunedFileError(str(error)) from error
        self._checkpoint(
            result.message, textproto_document.parse(result.text, self._path)
        )
        return result


def _converted(spec: config_schema.ParameterSpec, value: Any) -> Any:
    """`value` in the type of the parameter's kind."""
    if spec.kind == config_schema.Kind.INTEGER:
        return int(round(float(value)))
    if spec.kind == config_schema.Kind.NUMBER:
        return float(value)
    if spec.kind == config_schema.Kind.BOOL:
        return bool(value)
    return str(value)


def _same(first: Any, second: Any) -> bool:
    if isinstance(first, float) or isinstance(second, float):
        try:
            return float(first) == float(second)
        except (TypeError, ValueError):
            return False
    return bool(first == second)


def group_of(spec: config_schema.ParameterSpec) -> str:
    """The group a parameter's row is shown in: the path of the block that holds it.

    The elements of a repeated field that each hold one value (`joint_positions { joint: "x" value: 1 }`) share their
    field's group, so that a joint list is one group rather than a group per joint; an element with several values
    (`contact_wrenches[contact=l_foot].force`) is a group of its own.

    Args:
      spec: The parameter.

    Returns:
      The block's path; empty for a top-level scalar.
    """
    segments = textproto_document.parse_path(spec.path)[:-1]
    element = segments[-1] if segments else None
    selects = element is not None and (
        element.key is not None or element.index is not None
    )
    # An element's only value is labeled by the element's key alone (config_schema), not by its field's name.
    if (
        selects
        and element is not None
        and not spec.label.endswith("." + spec.field.name)
    ):
        segments = (*segments[:-1], textproto_document.PathSegment(element.name))
    return textproto_document.format_path(segments)
