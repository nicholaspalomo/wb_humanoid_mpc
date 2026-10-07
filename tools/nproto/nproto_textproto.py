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


"""Strict textproto parsing for Python: the twin of nproto/Textproto.h (tools/nproto/README.md).

    import nproto_textproto

    task = nproto_textproto.load_textproto("config/mpc/task.textproto", task_file_pb2.TaskFile)

A configuration file that does not match its schema exactly is a TextprotoError, `<source>:<line>:<column>: <problem>`
with 1-based positions at the token at fault, as the C++ parser reports it: a syntax error, an unknown field or
extension, an unknown enum value name, a value of the wrong type or out of range, a non-repeated field given twice, two
alternatives of one oneof, a field number instead of a name, a deprecated field and a missing proto2 required field.
The schema answers what it can, with the C++ parser's words: a field the message lists as an (nproto.retired_field) is
"'<name>' is retired: <replacement>"; otherwise an unknown field says "Did you mean ..." when its snake-cased name is a
field, and ends with the message's (nproto.retired_layout_hint). A text whose leading comment block names another
message (`# proto-message: pkg.Other`) is refused before it is parsed.

The C++ parser reports every problem of a text; this one, built on google.protobuf.text_format, reports the first.
"""

import dataclasses
import re
from typing import TypeVar

from google.protobuf import descriptor as descriptor_module
from google.protobuf import descriptor_pb2
from google.protobuf import descriptor_pool
from google.protobuf import message as message_module
from google.protobuf import text_format
from nproto import retired_field_options_pb2

MessageT = TypeVar("MessageT", bound=message_module.Message)

# The header lines a configuration file names its schema with, in its leading comment block.
# LINT.IfChange(header_lines)
PROTO_FILE_HEADER = "proto-file:"
PROTO_MESSAGE_HEADER = "proto-message:"
# LINT.ThenChange(//tools/nproto/src/Textproto.cpp:header_lines, //tools/hooks/textproto_headers.py:header_lines)

_UNKNOWN_FIELD = re.compile(r'^Message type "([^"]+)" has no field named "([^"]+)"\.')
# text_format reports a field given twice at its value; the C++ parser, and so this one, at its name.
_REPEATED_FIELD = re.compile(
    r'^Message type "[^"]+" should not have multiple "([^"]+)" (?:fields|extensions)\.'
)
_TOKEN = re.compile(
    r"""(?P<space>\s+)
      | (?P<comment>\#[^\n]*)
      | (?P<string>"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')
      | (?P<word>[A-Za-z0-9_.+-]+)
      | (?P<symbol>.)""",
    re.VERBOSE,
)


class TextprotoError(ValueError):
    """A textproto that does not match its schema; str() is `<source>:<line>:<column>: <problem>`."""


def snake_case(name: str) -> str:
    """`name` snake-cased and lower-cased: "useDcmTerminalCost" -> "use_dcm_terminal_cost", "HTTPRequest" -> "http_request".

    The form in which a retired name and an unknown field name are compared, as the C++ parser and the generator compare
    them.

    Args:
      name: A field name in any spelling.

    Returns:
      The name in snake case, lower case.
    """
    words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", words)
    return words.lower()


def header_message(text: str) -> tuple[int, str] | None:
    """(1-based line, full name) of the `# proto-message:` line in the leading comment block of `text`, or None."""
    for number, line in enumerate(text.split("\n"), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            return None  # The leading comment block has ended.
        comment = stripped[1:].lstrip()
        if comment.startswith(PROTO_MESSAGE_HEADER):
            return number, comment[len(PROTO_MESSAGE_HEADER) :].strip()
    return None


def _message_options(
    descriptor: descriptor_module.Descriptor,
) -> descriptor_pb2.MessageOptions:
    """The options of `descriptor`, parsed again now that nproto's retired-field extensions are registered."""
    return descriptor_pb2.MessageOptions.FromString(
        descriptor.GetOptions().SerializeToString()
    )


# LINT.IfChange(retired_options)
def retired_fields(descriptor: descriptor_module.Descriptor) -> list[tuple[str, str]]:
    """(name, replacement) of every (nproto.retired_field) option of the message `descriptor`, in declaration order."""
    entries = _message_options(descriptor).Extensions[
        retired_field_options_pb2.retired_field
    ]
    return [(entry.name, entry.replacement) for entry in entries]


def layout_hint(descriptor: descriptor_module.Descriptor) -> str:
    """The (nproto.retired_layout_hint) of the message `descriptor`; empty when it has none."""
    return str(
        _message_options(descriptor).Extensions[
            retired_field_options_pb2.retired_layout_hint
        ]
    )


def explain_problem(pool: descriptor_pool.DescriptorPool, problem: str) -> str:
    """`problem`, an error of the text parser, answered from the schema where it names an unknown field.

    Args:
      pool: The pool of the message being parsed, which holds the message types the problem names.
      problem: The parser's message, without its position.

    Returns:
      "'x' is retired: <replacement>" for a retired name; otherwise the problem, with "Did you mean "y"?" when the
      snake-cased name is a field and the message's layout hint.
    """
    match = _UNKNOWN_FIELD.match(problem)
    if match is None:
        return problem
    type_name, field_name = match.group(1), match.group(2)
    try:
        descriptor = pool.FindMessageTypeByName(type_name)
    except KeyError:
        return problem
    key = snake_case(field_name)
    for name, replacement in retired_fields(descriptor):
        if snake_case(name) == key:
            return f"'{field_name}' is retired: {replacement}"
    explained = problem
    if key != field_name and key in descriptor.fields_by_name:
        explained += f' Did you mean "{key}"?'
    hint = layout_hint(descriptor)
    if hint:
        explained += f" {hint}"
    return explained


# LINT.ThenChange(//tools/nproto/retired_field_options.proto:retired_options, //tools/nproto/src/Textproto.cpp:retired_options)


@dataclasses.dataclass(frozen=True)
class _Token:
    kind: str  # "string", "word" or "symbol"
    text: str
    line: int  # 1-based
    column: int  # 1-based


def _tokens(text: str) -> list[_Token]:
    """The tokens of `text` without whitespace and comments, with their 1-based positions."""
    tokens: list[_Token] = []
    line, line_start = 1, 0
    for match in _TOKEN.finditer(text):
        kind = match.lastgroup or "symbol"
        if kind not in ("space", "comment"):
            tokens.append(
                _Token(kind, match.group(), line, match.start() - line_start + 1)
            )
        newlines = match.group().count("\n")
        if newlines:
            line += newlines
            line_start = match.start() + match.group().rfind("\n") + 1
    return tokens


class _DeprecatedFieldFinder:
    """Walks the tokens of a text that parsed and finds the deprecated fields it sets, with their positions."""

    def __init__(self, tokens: list[_Token]) -> None:
        self.tokens = tokens
        self.found: list[tuple[_Token, descriptor_module.FieldDescriptor]] = []

    def _text(self, index: int) -> str:
        return self.tokens[index].text if index < len(self.tokens) else ""

    def message(
        self, index: int, descriptor: descriptor_module.Descriptor | None, closing: str
    ) -> int:
        """Walks the fields of a message body from `index` up to `closing` (empty: the end); returns the index after it."""
        while index < len(self.tokens) and self._text(index) != closing:
            field: descriptor_module.FieldDescriptor | None = None
            if self._text(index) == "[":  # An extension or an Any type URL.
                while index < len(self.tokens) and self._text(index) != "]":
                    index += 1
                index += 1
            else:
                name = self.tokens[index]
                field = (
                    descriptor.fields_by_name.get(name.text)
                    if descriptor is not None
                    else None
                )
                if field is not None and field.GetOptions().deprecated:
                    self.found.append((name, field))
                index += 1
            if self._text(index) == ":":
                index += 1
            index = self.value(index, field)
            if self._text(index) in (",", ";"):
                index += 1
        return index + 1

    def value(self, index: int, field: descriptor_module.FieldDescriptor | None) -> int:
        """Walks one value of `field` (a scalar, a message or a list) from `index`; returns the index after it."""
        text = self._text(index)
        if text in ("{", "<"):
            message_type = field.message_type if field is not None else None
            return self.message(index + 1, message_type, "}" if text == "{" else ">")
        if text == "[":
            index += 1
            while index < len(self.tokens) and self._text(index) != "]":
                index = self.value(index, field)
                if self._text(index) == ",":
                    index += 1
            return index + 1
        if text in ("-", "+"):
            index += 1
        index += 1
        while (
            index < len(self.tokens)
            and self.tokens[index].kind == "string"
            and self.tokens[index - 1].kind == "string"
        ):
            index += 1  # Adjacent strings are one value.
        return index


def _parse_error_problem(error: text_format.ParseError) -> str:
    """The message of a text_format.ParseError without the "<line>:<column> : " prefix the exception adds."""
    message = str(error)
    line, column = error.GetLine(), error.GetColumn()
    if line is not None:
        prefix = f"{line}:{column} : " if column is not None else f"{line} : "
        if message.startswith(prefix):
            return message[len(prefix) :]
    return message


def _field_name_position(
    text: str, line: int, column: int, field_name: str
) -> tuple[int, int]:
    """The position of the last token `field_name` of `text` at or before (`line`, `column`); the position itself if none."""
    position = (line, column)
    for token in _tokens(text):
        if (token.line, token.column) > (line, column):
            break
        if token.kind == "word" and token.text == field_name:
            position = (token.line, token.column)
    return position


def parse_textproto(text: str, message: message_module.Message, source: str) -> None:
    """Replaces the contents of `message` with the textproto `text`, parsed strictly (the module docstring).

    Args:
      text: The textproto.
      message: The message to parse into; on error its contents are unspecified.
      source: Names the text in errors, usually its path.

    Raises:
      TextprotoError: `text` does not match the schema of `message`, or its header names another message.
    """
    full_name = message.DESCRIPTOR.full_name
    header = header_message(text)
    if header is not None and header[1] != full_name:
        raise TextprotoError(
            f"{source}:{header[0]}:1: {source} is a {header[1]} (its '# proto-message:' header), not a {full_name}"
        )
    pool = message.DESCRIPTOR.file.pool
    message.Clear()
    try:
        text_format.Parse(text, message)
    except text_format.ParseError as error:
        original = _parse_error_problem(error)
        problem = explain_problem(pool, original)
        line = error.GetLine()
        if line is None:
            raise TextprotoError(f"{source}: {problem}") from error
        column = error.GetColumn() or 1
        repeated = _REPEATED_FIELD.match(original)
        if repeated is not None:
            line, column = _field_name_position(text, line, column, repeated.group(1))
        raise TextprotoError(f"{source}:{line}:{column}: {problem}") from error
    missing = message.FindInitializationErrors()
    if missing:
        raise TextprotoError(
            f"{source}: Message missing required fields: {', '.join(missing)}"
        )
    finder = _DeprecatedFieldFinder(_tokens(text))
    finder.message(0, message.DESCRIPTOR, closing="")
    if finder.found:
        token, field = finder.found[0]
        raise TextprotoError(
            f'{source}:{token.line}:{token.column}: text format contains deprecated field "{field.name}"'
        )


def load_textproto(path: str, message_class: type[MessageT]) -> MessageT:
    """The textproto file at `path` as a `message_class`, parsed strictly; errors name the path.

    Args:
      path: The file.
      message_class: The generated message class of its schema.

    Returns:
      The message.

    Raises:
      OSError: The file cannot be read.
      TextprotoError: It does not match the schema (parse_textproto()).
    """
    with open(path, encoding="utf-8") as file:
        text = file.read()
    message = message_class()
    parse_textproto(text, message, path)
    return message
