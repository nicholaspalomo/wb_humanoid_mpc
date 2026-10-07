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


"""A textproto as a concrete syntax tree that reproduces its file byte for byte, and edits that change only their value.

    from config_textproto import textproto_document

    document = textproto_document.load("config/mpc/task.textproto")
    document.value("state_weights.scaling")                         # 85
    document.trailing_comment("telemetry_frequency")                # "[Hz] robot/state, decimated from ..."
    document.set_scalar("state_weights.joint_positions[joint=l_leg_aky].value", 12.5)
    document.set_or_insert("contacts.contact_rectangle.length", 0.2)  # creates the blocks it needs
    text = document.text()

The tokenizer gives every character of the text a token: identifiers, numbers (exponents, an `f` suffix, hex and octal),
strings (both quote kinds, escapes, adjacent strings), the symbols `{ } < > [ ] : , ; - + . /`, `#` comments and
whitespace. The tree's fields hold the offsets of their tokens, so the text of an unedited document is the file, and an
edit splices only the characters it changes:

- a value is replaced token for token, and a value the file already spells (5 for 5.0, 5e-5 for 0.00005) is kept;
- a field the file does not have goes after the last field of the same name, or at the end of its block, at the
  indentation of its siblings (or at the end of the file for a top-level field), with every block it needs;
- a field is removed with its line and its trailing comment when it has the line to itself.

Comments, blank lines and LINT directives outside an edit therefore survive byte for byte.

A path names a value from the top of the file: field names joined by dots, each one optionally selecting an element of
a repeated field by its position, `telemetry_sinks[0]`, or by the value of one of its fields, the key,
`joint_positions[joint=l_leg_aky]`. A key value that is not a plain word is quoted: `[name="a b"]`. The empty path is
the file.

The module reads syntax, not schemas: it accepts a number where the schema wants a block. The schema's strictness is
textproto_save's, which re-parses every edited text with nproto_textproto, and the C++ parser's. It needs the standard
library only, so that a Python without protobuf (tools/locomotion_heuristics runs on the system Python) can read the
configuration files.
"""

import bisect
from collections.abc import Iterator, Sequence
import copy
import dataclasses
import math
import re
import struct
from typing import NamedTuple, TypeAlias

# The kinds of tokens.
SPACE = "space"
COMMENT = "comment"
STRING = "string"
NUMBER = "number"
IDENTIFIER = "identifier"
SYMBOL = "symbol"

_TRIVIA = (SPACE, COMMENT)
_TOKEN_PATTERN = re.compile(
    r"""(?P<space>[ \t\r\n\f\v]+)
      | (?P<comment>\#[^\n]*)
      | (?P<string>"(?:[^"\\\n]|\\[^\n])*"|'(?:[^'\\\n]|\\[^\n])*')
      | (?P<number>(?:[0-9]|\.[0-9])(?:[0-9A-Za-z_.]|(?<=[eE])[+-])*)
      | (?P<identifier>[A-Za-z_][A-Za-z0-9_]*)
      | (?P<symbol>[{}<>\[\]:,;+./-])""",
    re.VERBOSE,
)
_CLOSING = {"{": "}", "<": ">"}
_SIGNS = ("-", "+")
# Symbols that may stand between a value and the comment that describes it on its line: `value: 5 }  # comment`.
_CLOSERS_AND_SEPARATORS = ("}", ">", "]", ",", ";")

# What text_format reads as the floats with names, and as booleans.
_INFINITY = re.compile(r"inf(?:inity)?f?", re.IGNORECASE)
_NAN = re.compile(r"nanf?", re.IGNORECASE)
_TRUE = ("true", "True", "t")
_FALSE = ("false", "False", "f")
_HEX_INTEGER = re.compile(r"0[xX][0-9A-Fa-f]+")
_OCTAL_INTEGER = re.compile(r"0[0-7]+")
_DECIMAL_INTEGER = re.compile(r"[0-9]+")
_FLOAT = re.compile(r"(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?[fF]?")

_SIMPLE_ESCAPES = {
    "a": 0x07,
    "b": 0x08,
    "f": 0x0C,
    "n": 0x0A,
    "r": 0x0D,
    "t": 0x09,
    "v": 0x0B,
    "\\": 0x5C,
    "'": 0x27,
    '"': 0x22,
    "?": 0x3F,
}
# An escape of a string literal: a simple one, octal, hex, \\u and \\U code points, or one text_format does not know.
_ESCAPE = re.compile(
    r"\\(?:([abfnrtv\\'\"?])|([0-7]{1,3})|[xX]([0-9A-Fa-f]{1,2})|u([0-9A-Fa-f]{4})|U([0-9A-Fa-f]{8})|(.))"
)
_QUOTED_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t", '"': '\\"', "\\": "\\\\"}

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INDEX = re.compile(r"[0-9]+")
_BARE_KEY_VALUE = re.compile(r"[A-Za-z0-9_.+/-]+")
_IDENTIFIER_VALUE = re.compile(r"[-+]?[A-Za-z_][A-Za-z0-9_]*")
_DEFAULT_INDENT_UNIT = "  "


class DocumentError(ValueError):
    """A text the document cannot hold, or an edit it cannot make; the message names the source."""


class TextprotoSyntaxError(DocumentError):
    """A text that is not textproto syntax: `<source>:<line>:<column>: <problem>`, 1-based, at the token at fault."""


class PathError(DocumentError):
    """A path that is malformed, names nothing, or names several values where it must name one."""


class Token(NamedTuple):
    """One token of a text: its kind (SPACE, COMMENT, STRING, NUMBER, IDENTIFIER or SYMBOL), its characters, its offset."""

    kind: str
    text: str
    start: int

    @property
    def end(self) -> int:
        """The offset after the token."""
        return self.start + len(self.text)


@dataclasses.dataclass(frozen=True)
class ScalarNode:
    """A scalar value: a number or a name, with its sign, or one or more adjacent strings.

    Attributes:
      start: The offset of its first token.
      end: The offset after its last token.
      parts: Its tokens, without the whitespace and comments between them.
      text: Its characters, as the file spells them.
    """

    start: int
    end: int
    parts: tuple[Token, ...]
    text: str

    @property
    def kind(self) -> str:
        """STRING, NUMBER or IDENTIFIER: the kind of its last token."""
        return self.parts[-1].kind

    @property
    def signed(self) -> bool:
        """Whether it starts with a sign, as `-5` and `-inf` do."""
        return self.parts[0].text in _SIGNS

    def bytes_value(self) -> bytes:
        """The bytes of a string value, its parts concatenated and unescaped.

        Raises:
          DocumentError: It is not a string, or holds an unknown escape.
        """
        if self.kind != STRING:
            raise DocumentError(f"{self.text} is not a string")
        return b"".join(_unescape(part.text[1:-1]) for part in self.parts)

    def string_value(self) -> str:
        """The string a string value holds (its bytes as UTF-8; an invalid sequence reads as U+FFFD).

        Raises:
          DocumentError: It is not a string, or holds an unknown escape.
        """
        return self.bytes_value().decode("utf-8", errors="replace")

    def number_value(self) -> int | float:
        """The number a numeric value spells: an int for an integer literal, a float otherwise (inf and nan included).

        Raises:
          ValueError: It is not a number.
        """
        if self.kind == STRING:
            raise ValueError(f"{self.text} is not a number")
        number = number_text_value(self.parts[-1].text)
        return -number if self.parts[0].text == "-" else number

    def float_value(self) -> float:
        """The number a numeric value spells, as a double: `-0` is -0.0, as text_format reads it for a double field.

        Raises:
          ValueError: It is not a number.
        """
        if self.kind == STRING:
            raise ValueError(f"{self.text} is not a number")
        number = float(number_text_value(self.parts[-1].text))
        return -number if self.parts[0].text == "-" else number

    def value(self) -> str | int | float | bool:
        """The value as Python reads it without a schema: a string, a number, true/false, or the name of an enum value."""
        if self.kind == STRING:
            return self.string_value()
        if self.kind == IDENTIFIER and not self.signed:
            if self.text in ("true", "True"):
                return True
            if self.text in ("false", "False"):
                return False
        try:
            return self.number_value()
        except ValueError:
            return self.text


@dataclasses.dataclass(frozen=True)
class BlockNode:
    """A message value, `{ ... }` or `< ... >`, or the file itself.

    Attributes:
      open: The offset of its opening delimiter; None for the file.
      close: The offset of its closing delimiter; None for the file.
      fields: Its fields, in file order.
    """

    open: int | None
    close: int | None
    fields: tuple["FieldNode", ...]

    @property
    def end(self) -> int:
        """The offset after its closing delimiter (the file: -1, as it has none)."""
        return -1 if self.close is None else self.close + 1


@dataclasses.dataclass(frozen=True)
class ListNode:
    """A list value, `[a, b]`, of scalars or of blocks.

    Attributes:
      open: The offset of `[`.
      close: The offset of `]`.
      elements: Its values, in order.
      commas: The offsets of the commas; commas[i] follows elements[i].
    """

    open: int
    close: int
    elements: tuple[ScalarNode | BlockNode, ...]
    commas: tuple[int, ...]

    @property
    def end(self) -> int:
        """The offset after `]`."""
        return self.close + 1


ValueNode: TypeAlias = ScalarNode | BlockNode | ListNode


@dataclasses.dataclass(frozen=True)
class FieldNode:
    """One field as the file writes it: `name: value`, `name { ... }`, `name: [ ... ]`, with an optional `,` or `;`.

    Attributes:
      name: The field name (an extension's as `[pkg.name]`).
      start: The offset of the name.
      colon: The offset of the colon; None without one.
      value: The value.
      end: The offset after the value, or after the separator.
    """

    name: str
    start: int
    colon: int | None
    value: ValueNode
    end: int


@dataclasses.dataclass(frozen=True)
class Identifier:
    """A value to write as a name rather than as a quoted string: an enum value, or `inf`, `-inf`, `nan`.

    Attributes:
      name: The name, with an optional sign.
    """

    name: str

    def __post_init__(self) -> None:
        if not _IDENTIFIER_VALUE.fullmatch(self.name):
            raise DocumentError(f"{self.name!r} is not a name")


@dataclasses.dataclass(frozen=True)
class Literal:
    """A value to write exactly as given, already formatted: `0.1` for a float field's float32 0.1, say.

    Attributes:
      text: The value's characters, one textproto scalar.
    """

    text: str


@dataclasses.dataclass(frozen=True)
class NewBlock:
    """A message value to insert, `{ ... }`.

    Attributes:
      fields: Its fields, in order: (name, value) pairs, a repeated field as several pairs of the same name.
    """

    fields: tuple[tuple[str, "Value"], ...] = ()


ScalarValue: TypeAlias = bool | int | float | str | bytes | Identifier | Literal
Value: TypeAlias = ScalarValue | NewBlock


@dataclasses.dataclass(frozen=True)
class PathSegment:
    """One step of a path: a field, and which element of a repeated field it selects.

    Attributes:
      name: The field name.
      index: The element at this position, or None.
      key: (field, value): the element whose field `field` has the value `value`, or None.
    """

    name: str
    index: int | None = None
    key: tuple[str, str] | None = None


@dataclasses.dataclass(frozen=True)
class _Element:
    """A value of a field: the field, the value, and its position in the field's list (None: the field's own value)."""

    field: FieldNode
    node: ScalarNode | BlockNode
    list_index: int | None

    @property
    def start(self) -> int:
        return self.field.start if self.list_index is None else _value_start(self.node)

    @property
    def end(self) -> int:
        return self.field.end if self.list_index is None else self.node.end


def _line_column(text: str, offset: int) -> tuple[int, int]:
    """The 1-based line and column of `offset` in `text`."""
    line = text.count("\n", 0, offset) + 1
    return line, offset - (text.rfind("\n", 0, offset) + 1) + 1


def _syntax_error(
    text: str, source: str, offset: int, problem: str
) -> TextprotoSyntaxError:
    """The syntax error `problem` at `offset` of `text`."""
    line, column = _line_column(text, offset)
    return TextprotoSyntaxError(f"{source}:{line}:{column}: {problem}")


def tokenize(text: str, source: str = "<text>") -> list[Token]:
    """Every token of `text`, whitespace and comments included: their texts concatenated are `text`.

    Args:
      text: The textproto.
      source: Names the text in errors.

    Returns:
      The tokens, in order.

    Raises:
      TextprotoSyntaxError: An unterminated string, or a character no token starts with.
    """
    tokens: list[Token] = []
    position = 0
    for match in _TOKEN_PATTERN.finditer(text):
        if match.start() != position:
            break
        tokens.append(Token(match.lastgroup or SYMBOL, match.group(), position))
        position = match.end()
    if position < len(text):
        character = text[position]
        problem = (
            "unterminated string"
            if character in "\"'"
            else f"unexpected character {character!r}"
        )
        raise _syntax_error(text, source, position, problem)
    return tokens


def number_text_value(text: str) -> int | float:
    """The number one token spells, as text_format reads it: hex, octal and decimal integers, floats, inf and nan.

    Args:
      text: The token, without a sign.

    Returns:
      An int for an integer literal, otherwise a float.

    Raises:
      ValueError: `text` is not a number.
    """
    if _INFINITY.fullmatch(text):
        return math.inf
    if _NAN.fullmatch(text):
        return math.nan
    if _HEX_INTEGER.fullmatch(text):
        return int(text, 16)
    if _OCTAL_INTEGER.fullmatch(text):
        return int(text, 8)
    if _DECIMAL_INTEGER.fullmatch(text):
        return int(text)
    if _FLOAT.fullmatch(text):
        return float(text.rstrip("fF"))
    raise ValueError(f"{text!r} is not a number")


def parse_number(text: str) -> int | float:
    """The number `text` spells with an optional sign: `-5`, `+0x10`, `-inf` (number_text_value() after the sign).

    Raises:
      ValueError: `text` is not a number.
    """
    if text[:1] in _SIGNS:
        number = number_text_value(text[1:])
        return -number if text[0] == "-" else number
    return number_text_value(text)


def _unescape(body: str) -> bytes:
    """The bytes of a string literal's body (between its quotes), its escapes resolved as text_format resolves them."""
    out = bytearray()
    position = 0
    for escape in _ESCAPE.finditer(body):
        out += body[position : escape.start()].encode("utf-8")
        position = escape.end()
        simple, octal, hexadecimal, short_unicode, long_unicode, unknown = (
            escape.groups()
        )
        if simple is not None:
            out.append(_SIMPLE_ESCAPES[simple])
        elif octal is not None:
            out.append(int(octal, 8) & 0xFF)
        elif hexadecimal is not None:
            out.append(int(hexadecimal, 16))
        elif unknown is not None:
            raise DocumentError(f"unknown escape \\{unknown} in {body!r}")
        else:
            code_point = int(short_unicode or long_unicode, 16)
            if code_point > 0x10FFFF:
                raise DocumentError(
                    f"\\U{long_unicode} is not a code point, in {body!r}"
                )
            out += chr(code_point).encode("utf-8", errors="surrogatepass")
    out += body[position:].encode("utf-8")
    return bytes(out)


def quote_string(value: str) -> str:
    """`value` as a double-quoted textproto string: quotes, backslashes and control characters escaped, UTF-8 kept."""
    out = ['"']
    for character in value:
        if character in _QUOTED_ESCAPES:
            out.append(_QUOTED_ESCAPES[character])
        elif ord(character) < 0x20 or ord(character) == 0x7F:
            out.append(f"\\{ord(character):03o}")
        else:
            out.append(character)
    out.append('"')
    return "".join(out)


def quote_bytes(value: bytes) -> str:
    """`value` as a double-quoted textproto string: printable ASCII as it is, every other byte in octal."""
    out = ['"']
    for byte in value:
        character = chr(byte)
        if character in _QUOTED_ESCAPES:
            out.append(_QUOTED_ESCAPES[character])
        elif 0x20 <= byte < 0x7F:
            out.append(character)
        else:
            out.append(f"\\{byte:03o}")
    out.append('"')
    return "".join(out)


def format_double(value: float) -> str:
    """The shortest text that reads back to the double `value` (repr), with inf, -inf and nan by name."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    return repr(float(value))


def _check_literal(text: str) -> None:
    """Raises DocumentError unless `text` is exactly one scalar value."""
    try:
        parser = _Parser(text, "<value>")
        parser.scalar()
        trailing = parser.peek()
    except TextprotoSyntaxError as error:
        raise DocumentError(f"{text!r} is not a textproto value: {error}") from error
    if trailing is not None:
        raise DocumentError(f"{text!r} is not one textproto value")


def format_scalar(value: ScalarValue) -> str:
    """The textproto text of a scalar: true/false, an integer in decimal, a double by format_double(), a string quoted.

    Args:
      value: The value; an Identifier is written as its name and a Literal as its text.

    Returns:
      The text.

    Raises:
      DocumentError: A Literal that is not one scalar.
      TypeError: A value of another type.
    """
    if isinstance(value, Literal):
        _check_literal(value.text)
        return value.text
    if isinstance(value, Identifier):
        return value.name
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_double(value)
    if isinstance(value, str):
        return quote_string(value)
    if isinstance(value, bytes):
        return quote_bytes(value)
    raise TypeError(f"{value!r} is not a textproto scalar")


def _same_double(first: float, second: float) -> bool:
    """Whether two doubles are the same bits (NaN equal to NaN, 0.0 unequal to -0.0)."""
    if math.isnan(first) or math.isnan(second):
        return math.isnan(first) and math.isnan(second)
    return struct.pack("<d", first) == struct.pack("<d", second)


def same_value(node: ScalarNode, value: ScalarValue) -> bool:
    """Whether the file's value `node` already spells `value`, so that writing `value` would change nothing it means.

    A number compares by value (5 spells 5.0, 0x10 spells 16; a double bit for bit, so 0 does not spell -0.0), a
    string by its bytes, a bool by the names text_format reads, an Identifier by name, a Literal by its text.

    Args:
      node: The value in the file.
      value: The value to write.

    Returns:
      True when the file's text can stay.
    """
    if isinstance(value, Literal):
        return node.text == value.text
    if isinstance(value, Identifier):
        return node.kind == IDENTIFIER and node.text == value.name
    if isinstance(value, bool):
        names = _TRUE if value else _FALSE
        return node.kind == IDENTIFIER and not node.signed and node.text in names
    if isinstance(value, (str, bytes)):
        if node.kind != STRING:
            return False
        try:
            encoded = value.encode("utf-8") if isinstance(value, str) else value
            return node.bytes_value() == encoded
        except (DocumentError, UnicodeEncodeError):
            return False
    try:
        if isinstance(value, int):
            number = node.number_value()
            return isinstance(number, int) and number == value
        return _same_double(node.float_value(), value)
    except ValueError:
        return False


class _Parser:
    """Builds the tree of a text from its tokens, recursive descent on the textproto grammar."""

    def __init__(self, text: str, source: str) -> None:
        self._text = text
        self._source = source
        self.tokens = tokenize(text, source)
        self._significant = [
            token for token in self.tokens if token.kind not in _TRIVIA
        ]
        self._index = 0

    def peek(self) -> Token | None:
        """The next significant token, or None at the end."""
        if self._index < len(self._significant):
            return self._significant[self._index]
        return None

    def _next(self, expected: str) -> Token:
        """The next significant token, consumed; at the end, a syntax error saying `expected` is missing."""
        token = self.peek()
        if token is None:
            raise self._error(None, f"expected {expected} before the end of the text")
        self._index += 1
        return token

    def _is_symbol(self, token: Token | None, *symbols: str) -> bool:
        return token is not None and token.kind == SYMBOL and token.text in symbols

    def _take_symbol(self, *symbols: str) -> Token | None:
        """The next significant token, consumed, when it is one of `symbols`; otherwise None, and nothing consumed."""
        token = self.peek()
        if token is None or not self._is_symbol(token, *symbols):
            return None
        self._index += 1
        return token

    def _error(self, token: Token | None, problem: str) -> TextprotoSyntaxError:
        offset = len(self._text) if token is None else token.start
        return _syntax_error(self._text, self._source, offset, problem)

    def document(self) -> BlockNode:
        """The tree of the whole text."""
        return BlockNode(None, None, self._fields(closing=None))

    def _fields(self, closing: str | None) -> tuple[FieldNode, ...]:
        """The fields up to the delimiter `closing` (not consumed), or up to the end of the text when it is None."""
        fields: list[FieldNode] = []
        while True:
            token = self.peek()
            if token is None:
                if closing is not None:
                    raise self._error(
                        None, f'expected "{closing}" before the end of the text'
                    )
                return tuple(fields)
            if closing is not None and self._is_symbol(token, closing):
                return tuple(fields)
            fields.append(self._field())

    def _field(self) -> FieldNode:
        """One field: its name, an optional colon, its value and an optional separator."""
        name_token = self._next("a field name")
        if name_token.kind == IDENTIFIER:
            name = name_token.text
        elif self._is_symbol(name_token, "["):
            name = self._extension_name()
        else:
            raise self._error(
                name_token, f"expected a field name, found {name_token.text!r}"
            )
        colon = self._take_symbol(":")
        value = self._value(name, colon is not None)
        separator = self._take_symbol(",", ";")
        end = value.end if separator is None else separator.end
        return FieldNode(
            name, name_token.start, None if colon is None else colon.start, value, end
        )

    def _extension_name(self) -> str:
        """The rest of `[pkg.extension]` or `[type.googleapis.com/pkg.Message]`, after its `[`."""
        parts = ["["]
        while True:
            token = self._next('"]"')
            if self._is_symbol(token, "]"):
                parts.append("]")
                return "".join(parts)
            if token.kind != IDENTIFIER and not self._is_symbol(token, ".", "/"):
                raise self._error(
                    token, f"expected an extension name, found {token.text!r}"
                )
            parts.append(token.text)

    def _value(self, name: str, has_colon: bool) -> ValueNode:
        """The value of the field `name`: a block, a list, or a scalar (which needs the colon)."""
        token = self.peek()
        if token is None:
            raise self._error(
                None, f"expected a value of {name} before the end of the text"
            )
        if self._is_symbol(token, "{", "<"):
            return self._block()
        if self._is_symbol(token, "["):
            return self._list()
        if not has_colon:
            raise self._error(
                token,
                f'expected ":" between {name} and its value, found {token.text!r}',
            )
        return self.scalar()

    def _block(self) -> BlockNode:
        opening = self._next('"{"')
        closing = _CLOSING[opening.text]
        fields = self._fields(closing)
        close = self._next(f'"{closing}"')
        return BlockNode(opening.start, close.start, fields)

    def _list(self) -> ListNode:
        """A list, `[a, b]`, of scalars or blocks."""
        opening = self._next('"["')
        elements: list[ScalarNode | BlockNode] = []
        commas: list[int] = []
        if self._is_symbol(self.peek(), "]"):
            return ListNode(opening.start, self._next('"]"').start, (), ())
        while True:
            token = self.peek()
            elements.append(
                self._block() if self._is_symbol(token, "{", "<") else self.scalar()
            )
            separator = self._next('"," or "]"')
            if self._is_symbol(separator, "]"):
                return ListNode(
                    opening.start, separator.start, tuple(elements), tuple(commas)
                )
            if not self._is_symbol(separator, ","):
                raise self._error(
                    separator, f'expected "," or "]", found {separator.text!r}'
                )
            commas.append(separator.start)

    def scalar(self) -> ScalarNode:
        """A scalar value: a signed number or name, or adjacent strings."""
        first = self._next("a value")
        parts = [first]
        if self._is_symbol(first, *_SIGNS):
            number = self._next("a number after the sign")
            if number.kind not in (NUMBER, IDENTIFIER):
                raise self._error(
                    number,
                    f"expected a number after {first.text!r}, found {number.text!r}",
                )
            parts.append(number)
        elif first.kind == STRING:
            while (following := self.peek()) is not None and following.kind == STRING:
                parts.append(following)
                self._index += 1
        elif first.kind not in (NUMBER, IDENTIFIER):
            raise self._error(first, f"expected a value, found {first.text!r}")
        return ScalarNode(
            first.start,
            parts[-1].end,
            tuple(parts),
            self._text[first.start : parts[-1].end],
        )


def parse_path(path: str) -> tuple[PathSegment, ...]:
    """The segments of a path (the module docstring); the empty path is the file and has none.

    Args:
      path: e.g. "state_weights.joint_positions[joint=l_leg_aky].value".

    Returns:
      Its segments.

    Raises:
      PathError: `path` is malformed.
    """
    if not path:
        return ()
    segments: list[PathSegment] = []
    position = 0
    while True:
        name = _NAME.match(path, position)
        if name is None:
            raise PathError(f"{path!r}: expected a field name at column {position + 1}")
        position = name.end()
        index: int | None = None
        key: tuple[str, str] | None = None
        if position < len(path) and path[position] == "[":
            index, key, position = _parse_selector(path, position + 1)
        segments.append(PathSegment(name.group(), index, key))
        if position == len(path):
            return tuple(segments)
        if path[position] != ".":
            raise PathError(f"{path!r}: expected '.' or '[' at column {position + 1}")
        position += 1


def _parse_selector(
    path: str, position: int
) -> tuple[int | None, tuple[str, str] | None, int]:
    """(index, key, the position after `]`) of the selector of `path` that starts at `position`, after its `[`."""
    digits = _INDEX.match(path, position)
    if digits is not None and path[digits.end() : digits.end() + 1] == "]":
        return int(digits.group()), None, digits.end() + 1
    field = _NAME.match(path, position)
    if field is None or path[field.end() : field.end() + 1] != "=":
        raise PathError(
            f"{path!r}: expected an index or field=value at column {position + 1}"
        )
    position = field.end() + 1
    if path[position : position + 1] == '"':
        value, position = _parse_quoted(path, position + 1)
    else:
        bare = _BARE_KEY_VALUE.match(path, position)
        if bare is None:
            raise PathError(f"{path!r}: expected a key value at column {position + 1}")
        value, position = bare.group(), bare.end()
    if path[position : position + 1] != "]":
        raise PathError(f"{path!r}: expected ']' at column {position + 1}")
    return None, (field.group(), value), position + 1


def _parse_quoted(path: str, position: int) -> tuple[str, int]:
    """(the value, the position after its closing quote) of a quoted key value whose body starts at `position`."""
    out: list[str] = []
    while position < len(path):
        character = path[position]
        if character == '"':
            return "".join(out), position + 1
        if character == "\\" and position + 1 < len(path):
            out.append(path[position + 1])
            position += 2
            continue
        out.append(character)
        position += 1
    raise PathError(f"{path!r}: unterminated quoted key value")


def _format_key_value(value: str) -> str:
    if _BARE_KEY_VALUE.fullmatch(value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def format_segment(segment: PathSegment) -> str:
    """The text of one segment: `name`, `name[3]` or `name[key=value]`."""
    if segment.index is not None:
        return f"{segment.name}[{segment.index}]"
    if segment.key is not None:
        return f"{segment.name}[{segment.key[0]}={_format_key_value(segment.key[1])}]"
    return segment.name


def format_path(segments: Sequence[PathSegment]) -> str:
    """The path of `segments`, which parse_path() reads back to them."""
    return ".".join(format_segment(segment) for segment in segments)


def _elements(block: BlockNode, name: str) -> list[_Element]:
    """Every value of the field `name` of `block`, in file order: a field given several times, or a list, or both."""
    found: list[_Element] = []
    for field in block.fields:
        if field.name != name:
            continue
        if isinstance(field.value, ListNode):
            found += [
                _Element(field, element, index)
                for index, element in enumerate(field.value.elements)
            ]
        else:
            found.append(_Element(field, field.value, None))
    return found


def _key_matches(node: ScalarNode, text: str) -> bool:
    """Whether the key value `node` is `text`: a string by its value, a name by its spelling, a number by its value."""
    if node.kind == STRING:
        try:
            return node.string_value() == text
        except DocumentError:
            return False
    if node.text == text:
        return True
    try:
        return node.number_value() == parse_number(text)
    except ValueError:
        return False


def _has_key(block: BlockNode, key: tuple[str, str]) -> bool:
    """Whether `block` is the element the key (field, value) selects."""
    values = _elements(block, key[0])
    return (
        len(values) == 1
        and isinstance(values[0].node, ScalarNode)
        and _key_matches(values[0].node, key[1])
    )


class TextprotoDocument:
    """A textproto as a tree of its tokens: what it holds, where, and edits that change only what they edit.

    Every edit splices the text and parses it again, so the tree always describes the text, and a failed edit leaves
    both as they were.
    """

    def __init__(self, text: str, source: str = "<text>") -> None:
        """Parses `text`.

        Args:
          text: The textproto.
          source: Names the text in errors, usually its path.

        Raises:
          TextprotoSyntaxError: `text` is not textproto syntax.
        """
        self._source = source
        self._text = ""
        self._tokens: list[Token] = []
        self._token_starts: list[int] = []
        self._root = BlockNode(None, None, ())
        self._line_starts = [0]
        self._set_text(text)

    def _set_text(self, text: str) -> None:
        parser = _Parser(text, self._source)
        root = parser.document()
        self._text = text
        self._tokens = parser.tokens
        self._token_starts = [token.start for token in parser.tokens]
        self._root = root
        self._line_starts = [0] + [match.end() for match in re.finditer("\n", text)]

    def copy(self) -> "TextprotoDocument":
        """A document of the same text, to edit apart from this one (cheap: the tree is shared, never changed)."""
        return copy.copy(
            self
        )  # Every edit replaces the text, the tokens and the tree; none is changed in place.

    @property
    def source(self) -> str:
        """What the document's errors name it."""
        return self._source

    @property
    def root(self) -> BlockNode:
        """The tree of the whole text."""
        return self._root

    def text(self) -> str:
        """The text, with every edit made so far."""
        return self._text

    # ---- Reading ----

    def _path_error(self, path: str, problem: str) -> PathError:
        return PathError(f"{self._source}: {path}: {problem}")

    def _select(
        self, block: BlockNode, segment: PathSegment, path: str
    ) -> _Element | None:
        """The element of `block` that `segment` selects; None when there is none; PathError when there are several."""
        elements = _elements(block, segment.name)
        if segment.index is not None:
            return elements[segment.index] if segment.index < len(elements) else None
        if segment.key is not None:
            key = segment.key
            matches = [
                element
                for element in elements
                if isinstance(element.node, BlockNode) and _has_key(element.node, key)
            ]
            if len(matches) > 1:
                raise self._path_error(
                    path,
                    f"{len(matches)} elements of {segment.name} have {key[0]} {key[1]}",
                )
            return matches[0] if matches else None
        if len(elements) > 1:
            raise self._path_error(
                path,
                f"{segment.name} is given {len(elements)} times; select one with [index] or [field=value]",
            )
        return elements[0] if elements else None

    def _walk(
        self, segments: Sequence[PathSegment], path: str
    ) -> tuple[BlockNode, int, _Element | None]:
        """How far `segments` lead: (the deepest block reached, how many segments were found, the last one found)."""
        block = self._root
        element: _Element | None = None
        for count, segment in enumerate(segments):
            if element is not None:
                if not isinstance(element.node, BlockNode):
                    shown = format_path(segments[:count])
                    raise self._path_error(path, f"{shown} is a value, not a block")
                block = element.node
            found = self._select(block, segment, path)
            if found is None:
                return block, count, element
            element = found
        return block, len(segments), element

    def _element(self, path: str) -> _Element:
        """The element `path` names; PathError when it names nothing (or the file)."""
        segments = parse_path(path)
        if not segments:
            raise self._path_error(path, "names the file, not a field")
        _, found, element = self._walk(segments, path)
        if found < len(segments) or element is None:
            raise self._path_error(path, "no such field in the file")
        return element

    def find(self, path: str) -> ScalarNode | BlockNode | None:
        """The value `path` names, or None when the file does not have it.

        Args:
          path: The value; the empty path is the file.

        Returns:
          The node of the value, or None.

        Raises:
          PathError: `path` is malformed, crosses a scalar, or names several values.
        """
        segments = parse_path(path)
        if not segments:
            return self._root
        _, found, element = self._walk(segments, path)
        if found < len(segments) or element is None:
            return None
        return element.node

    def has(self, path: str) -> bool:
        """Whether the file has the value `path` names."""
        return self.find(path) is not None

    def get(self, path: str) -> ScalarNode | BlockNode:
        """The value `path` names.

        Raises:
          PathError: The file does not have it, or `path` is malformed or names several values.
        """
        node = self.find(path)
        if node is None:
            raise self._path_error(path, "no such field in the file")
        return node

    def scalar(self, path: str) -> ScalarNode:
        """The scalar `path` names; PathError when it is a block or absent."""
        node = self.get(path)
        if not isinstance(node, ScalarNode):
            raise self._path_error(path, "is a block, not a value")
        return node

    def value(self, path: str) -> str | int | float | bool:
        """The scalar `path` names, as ScalarNode.value() reads it."""
        return self.scalar(path).value()

    def count(self, path: str) -> int:
        """How many values the repeated field `path` (no selector on its last segment) has; 0 when its block is absent."""
        segments = parse_path(path)
        if (
            not segments
            or segments[-1].index is not None
            or segments[-1].key is not None
        ):
            raise self._path_error(path, "count() takes a field without a selector")
        parent = self.find(format_path(segments[:-1]))
        if not isinstance(parent, BlockNode):
            return 0
        return len(_elements(parent, segments[-1].name))

    def field_names(self, path: str = "") -> list[str]:
        """The names of the fields of the block `path` names, in file order, each once."""
        block = self.get(path) if path else self._root
        if not isinstance(block, BlockNode):
            raise self._path_error(path, "is a value, not a block")
        return list(dict.fromkeys(field.name for field in block.fields))

    def span(self, path: str) -> tuple[int, int]:
        """The offsets of the field (or list element) `path` names: its start and the offset after it."""
        element = self._element(path)
        return element.start, element.end

    def position(self, path: str) -> tuple[int, int]:
        """The 1-based line and column of the field (or list element) `path` names."""
        return self._line_column(self._element(path).start)

    def _line_column(self, offset: int) -> tuple[int, int]:
        line = bisect.bisect_right(self._line_starts, offset)
        return line, offset - self._line_starts[line - 1] + 1

    def _tokens_from(self, offset: int) -> Iterator[Token]:
        """The tokens that start at `offset` or after it, in order."""
        for index in range(
            bisect.bisect_left(self._token_starts, offset), len(self._tokens)
        ):
            yield self._tokens[index]

    def _comment_after(self, offset: int) -> str | None:
        """The comment on the line of `offset` after it, with only closing delimiters and separators in between."""
        for token in self._tokens_from(offset):
            if token.kind == SPACE:
                if "\n" in token.text:
                    return None
                continue
            if token.kind == COMMENT:
                return token.text[1:].strip()
            if token.kind != SYMBOL or token.text not in _CLOSERS_AND_SEPARATORS:
                return None
        return None

    def trailing_comment(self, path: str) -> str | None:
        """The comment at the end of the line of the value `path` names, without its `#`; None without one.

        The comment after a value belongs to it when nothing but closing delimiters and separators stand between them,
        so `joint_positions { joint: "x" value: 5 }  # hip` gives "hip" for both the element and its value. A block
        without one after its closing delimiter gives the one after its opening delimiter.

        Args:
          path: The value.

        Returns:
          The comment's text, stripped; None when the line has none.

        Raises:
          PathError: The file does not have the value.
        """
        element = self._element(path)
        comment = self._comment_after(element.end)
        if (
            comment is None
            and isinstance(element.node, BlockNode)
            and element.node.open is not None
        ):
            comment = self._comment_after(element.node.open + 1)
        return comment

    def leading_comments(self, path: str) -> list[str]:
        """The comment lines directly above the field `path` names, without their `#` (none past a blank line).

        Args:
          path: The field; one that does not start its line has none.

        Returns:
          The comments, top to bottom, stripped.

        Raises:
          PathError: The file does not have the field.
        """
        start = self._element(path).start
        if not self._starts_line(start):
            return []
        comments: list[str] = []
        line = bisect.bisect_right(self._line_starts, start) - 1
        while line > 0:
            line -= 1
            text = self._text[
                self._line_starts[line] : self._line_starts[line + 1]
            ].strip()
            if not text.startswith("#"):
                break
            comments.append(text[1:].strip())
        return comments[::-1]

    # ---- Layout ----

    def _newline(self) -> str:
        return "\r\n" if "\r\n" in self._text else "\n"

    def _line_start(self, offset: int) -> int:
        return self._line_starts[bisect.bisect_right(self._line_starts, offset) - 1]

    def _line_indent(self, offset: int) -> str:
        """The leading whitespace of the line of `offset`."""
        start = self._line_start(offset)
        end = start
        while end < len(self._text) and self._text[end] in " \t":
            end += 1
        return self._text[start:end]

    def _starts_line(self, offset: int) -> bool:
        """Whether only whitespace precedes `offset` on its line."""
        return not self._text[self._line_start(offset) : offset].strip()

    def _end_of_line(self, offset: int) -> int | None:
        """Where the line of `offset` ends, when only whitespace and a comment follow `offset` on it.

        Args:
          offset: A token boundary.

        Returns:
          The offset of the line's newline (len(text) on a last line without one), or None when a token follows.
        """
        for token in self._tokens_from(offset):
            if token.kind == COMMENT:
                continue
            if token.kind == SPACE:
                newline = token.text.find("\n")
                if newline >= 0:
                    return token.start + newline
                continue
            return None
        return len(self._text)

    def _is_multiline(self, block: BlockNode) -> bool:
        """Whether a block spans several lines (the file always does)."""
        if block.open is None or block.close is None:
            return True
        return "\n" in self._text[block.open : block.close]

    def _indent_unit(self) -> str:
        """How much deeper the fields of a block are indented than the block, the smallest step the file uses ("  ")."""
        steps: list[str] = []
        pending = [self._root]
        while pending:
            block = pending.pop()
            for field in block.fields:
                values = (
                    field.value.elements
                    if isinstance(field.value, ListNode)
                    else (field.value,)
                )
                for value in values:
                    if not isinstance(value, BlockNode) or not value.fields:
                        continue
                    pending.append(value)
                    child = value.fields[0].start
                    if not self._is_multiline(value) or not self._starts_line(child):
                        continue
                    indent, child_indent = self._line_indent(
                        field.start
                    ), self._line_indent(child)
                    if child_indent.startswith(indent) and len(child_indent) > len(
                        indent
                    ):
                        steps.append(child_indent[len(indent) :])
        return min(steps, key=len) if steps else _DEFAULT_INDENT_UNIT

    def _render_block(self, block: NewBlock, indent: str, inline: bool) -> str:
        """`{ a: 1 }` on one line, or a block whose fields are one level deeper than `indent`, its `}` at `indent`."""
        if not block.fields:
            return "{}"
        if inline:
            fields = " ".join(
                self._render(name, value, "", inline=True)
                for name, value in block.fields
            )
            return f"{{ {fields} }}"
        unit = self._indent_unit()
        lines = ["{"]
        for name, value in block.fields:
            lines.append(
                indent + unit + self._render(name, value, indent + unit, inline=False)
            )
        lines.append(f"{indent}}}")
        return self._newline().join(lines)

    def _render(self, name: str, value: Value, indent: str, inline: bool) -> str:
        """`name: value`, or `name { ... }` (_render_block()); the first line without indentation."""
        if isinstance(value, NewBlock):
            return f"{name} {self._render_block(value, indent, inline)}"
        return f"{name}: {format_scalar(value)}"

    # ---- Edits ----

    def _replace(self, start: int, end: int, replacement: str) -> None:
        self._set_text(self._text[:start] + replacement + self._text[end:])

    def set_scalar(self, path: str, value: ScalarValue) -> bool:
        """Writes `value` over the scalar `path` names, unless the file already spells it (same_value()).

        Args:
          path: The value; it must be in the file.
          value: The new value (format_scalar()).

        Returns:
          Whether the text changed.

        Raises:
          PathError: The file does not have the value, or it is a block.
          DocumentError: `value` is not one scalar.
        """
        return self.set_scalars([(path, value)]) > 0

    def set_scalars(self, values: Sequence[tuple[str, ScalarValue]]) -> int:
        """set_scalar() for many values at once, with one parse of the edited text however many there are.

        Args:
          values: (path, value) pairs; every path must name a scalar in the file. A path given twice takes its last value.

        Returns:
          How many values the text changed.

        Raises:
          PathError: The file does not have one of the values, or it is a block; nothing is changed then.
          DocumentError: A value is not one scalar; nothing is changed then.
        """
        replacements: dict[int, tuple[ScalarNode, str | None]] = {}
        for path, value in values:
            node = self.scalar(path)
            replacements[node.start] = (
                node,
                None if same_value(node, value) else format_scalar(value),
            )
        splices = sorted(
            (
                (node.start, node.end, text)
                for node, text in replacements.values()
                if text is not None
            ),
            reverse=True,
        )
        if not splices:
            return 0
        text = self._text
        for start, end, replacement in splices:
            text = text[:start] + replacement + text[end:]
        self._set_text(text)
        return len(splices)

    def set_or_insert(self, path: str, value: ScalarValue) -> bool:
        """set_scalar() when the file has the value; otherwise inserts its field and every block on the way to it.

        A missing element selected by key is created with its key field first, as a string; one selected by index only
        when the index is the field's length (the element after the last).

        Args:
          path: The value.
          value: Its new value.

        Returns:
          Whether the text changed.

        Raises:
          PathError: `path` names a block, crosses a value, or selects an element that cannot be created.
        """
        segments = parse_path(path)
        if not segments:
            raise self._path_error(path, "names the file, not a field")
        block, found, _ = self._walk(segments, path)
        if found == len(segments):
            return self.set_scalar(path, value)
        self._create(block, segments[found:], value, path)
        return True

    def insert_field(self, parent_path: str, name: str, value: Value) -> None:
        """Adds the field `name` with `value` to the block `parent_path` names, creating the blocks it is missing.

        The field goes after the block's last field of the same name; else at the block's end, before its closing
        delimiter at the indentation of its fields, or at the end of the file for a top-level field.

        Args:
          parent_path: The block; the empty path is the file.
          name: The field's name.
          value: A scalar, or a NewBlock.

        Raises:
          PathError: `parent_path` names a value or selects an element that cannot be created.
        """
        segments = parse_path(parent_path)
        block, found, element = self._walk(segments, parent_path)
        if found == len(segments) and element is not None:
            if not isinstance(element.node, BlockNode):
                raise self._path_error(parent_path, "is a value, not a block")
            block = element.node
        self._create(block, (*segments[found:], PathSegment(name)), value, parent_path)

    def append(self, path: str, value: Value) -> None:
        """Adds `value` after the last value of the repeated field `path` (no selector), creating missing blocks."""
        segments = parse_path(path)
        if (
            not segments
            or segments[-1].index is not None
            or segments[-1].key is not None
        ):
            raise self._path_error(path, "append() takes a field without a selector")
        self.insert_field(format_path(segments[:-1]), segments[-1].name, value)

    def _create(
        self,
        block: BlockNode,
        remaining: Sequence[PathSegment],
        value: Value,
        path: str,
    ) -> None:
        """Inserts into `block` the field of remaining[0], holding a new block for each further segment and `value`."""
        last = remaining[-1]
        if last.key is not None:
            if not isinstance(value, NewBlock):
                raise self._path_error(
                    path, f"{format_segment(last)} names a block, not a value"
                )
            value = NewBlock(_key_fields(last) + value.fields)
        first = remaining[0]
        existing = len(_elements(block, first.name))
        if first.index is not None and first.index != existing:
            raise self._path_error(
                path,
                f"{first.name} has {existing} values; only [{existing}] can be added",
            )
        nested = value
        for depth in range(len(remaining) - 1, 0, -1):
            segment = remaining[depth]
            if segment.index not in (None, 0):
                raise self._path_error(
                    path,
                    f"a new block has no element [{segment.index}] of {segment.name}",
                )
            nested = NewBlock(
                _key_fields(remaining[depth - 1]) + ((segment.name, nested),)
            )
        self._insert(block, first.name, nested)

    def _insert(self, block: BlockNode, name: str, value: Value) -> None:
        siblings = [field for field in block.fields if field.name == name]
        if siblings:
            last = siblings[-1]
            if isinstance(last.value, ListNode):
                self._insert_into_list(last.value, value)
            else:
                self._insert_after(last, name, value)
        elif block.open is None:
            self._insert_at_file_end(name, value)
        else:
            self._insert_at_block_end(block, name, value)

    def _insert_into_list(self, node: ListNode, value: Value) -> None:
        if isinstance(value, NewBlock):
            rendered = self._render_block(value, "", inline=True)
        else:
            rendered = format_scalar(value)
        if node.elements:
            end = node.elements[-1].end
            self._replace(end, end, f", {rendered}")
        else:
            self._replace(node.close, node.close, rendered)

    def _insert_after(self, field: FieldNode, name: str, value: Value) -> None:
        """Inserts after `field`: on a line of its own when `field` has its line to itself, else on the same line."""
        end_of_line = self._end_of_line(field.end)
        if not self._starts_line(field.start) or end_of_line is None:
            self._replace(
                field.end, field.end, " " + self._render(name, value, "", inline=True)
            )
            return
        indent = self._line_indent(field.start)
        inline = isinstance(field.value, BlockNode) and not self._is_multiline(
            field.value
        )
        rendered = indent + self._render(name, value, indent, inline)
        newline = self._newline()
        if end_of_line == len(self._text):
            self._replace(end_of_line, end_of_line, newline + rendered)
        else:
            self._replace(end_of_line + 1, end_of_line + 1, rendered + newline)

    def _insert_at_file_end(self, name: str, value: Value) -> None:
        indent = ""
        for field in reversed(self._root.fields):
            if self._starts_line(field.start):
                indent = self._line_indent(field.start)
                break
        newline = self._newline()
        prefix = "" if not self._text or self._text.endswith("\n") else newline
        rendered = (
            prefix + indent + self._render(name, value, indent, inline=False) + newline
        )
        self._replace(len(self._text), len(self._text), rendered)

    def _insert_at_block_end(self, block: BlockNode, name: str, value: Value) -> None:
        """Inserts before the closing delimiter of `block`: on a line of its own, or on the block's one line."""
        if block.open is None or block.close is None:
            self._insert_at_file_end(name, value)
            return
        close = block.close
        if self._is_multiline(block) and self._starts_line(close):
            indent = self._line_indent(block.open) + self._indent_unit()
            for field in reversed(block.fields):
                if self._starts_line(field.start):
                    indent = self._line_indent(field.start)
                    break
            rendered = (
                indent
                + self._render(name, value, indent, inline=False)
                + self._newline()
            )
            line_start = self._line_start(close)
            self._replace(line_start, line_start, rendered)
            return
        rendered = self._render(name, value, "", inline=True)
        if close - 1 == block.open:
            self._replace(close, close, f" {rendered} ")
        elif self._text[close - 1] in " \t":
            self._replace(close, close, f"{rendered} ")
        else:
            self._replace(close, close, f" {rendered}")

    def remove(self, path: str) -> int:
        """Removes the value `path` names; without a selector on its last segment, every value of that field.

        A field that has its line to itself goes with the line, its trailing comment included; a field that shares its
        line goes with the whitespace after it (or before it, at the end of the line). A list element goes with its comma
        (the one before it, for the last element) and, when it has its line to itself, with that line; the comment lines
        between the elements - LINT directives among them - stay.

        Args:
          path: The value or field.

        Returns:
          How many values were removed: 0 when the file has none.

        Raises:
          PathError: `path` is malformed, crosses a value, or names several elements.
        """
        segments = parse_path(path)
        if not segments:
            raise self._path_error(path, "names the file, not a field")
        parent = self.find(format_path(segments[:-1]))
        if parent is None:
            return 0
        if not isinstance(parent, BlockNode):
            raise self._path_error(
                path, f"{format_path(segments[:-1])} is a value, not a block"
            )
        last = segments[-1]
        if last.index is None and last.key is None:
            elements = _elements(parent, last.name)
            fields = {element.field.start: element.field for element in elements}
            self._remove_spans([self._removal_span(field) for field in fields.values()])
            return len(elements)
        element = self._select(parent, last, path)
        if element is None:
            return 0
        if isinstance(element.field.value, ListNode) and element.list_index is not None:
            self._remove_spans(
                self._list_removal_spans(element.field.value, element.list_index)
            )
        else:
            self._remove_spans([self._removal_span(element.field)])
        return 1

    def _remove_spans(self, spans: list[tuple[int, int]]) -> None:
        """Deletes the characters of `spans`, overlapping ones merged."""
        kept: list[str] = []
        position = 0
        for start, end in sorted(spans):
            start = max(start, position)
            kept.append(self._text[position:start])
            position = max(position, end)
        kept.append(self._text[position:])
        self._set_text("".join(kept))

    def _removal_span(self, field: FieldNode) -> tuple[int, int]:
        """The characters that removing `field` deletes (remove())."""
        end_of_line = self._end_of_line(field.end)
        if self._starts_line(field.start) and end_of_line is not None:
            return self._line_start(field.start), min(end_of_line + 1, len(self._text))
        after = field.end
        while after < len(self._text) and self._text[after] in " \t":
            after += 1
        if after < len(self._text) and self._text[after] not in "\r\n":
            return field.start, after
        before = field.start
        while before > 0 and self._text[before - 1] in " \t":
            before -= 1
        return before, field.end

    def _list_removal_spans(self, node: ListNode, index: int) -> list[tuple[int, int]]:
        """The characters that removing element `index` of the list `node` deletes (remove()).

        Args:
          node: The list.
          index: The element.

        Returns:
          The spans: the element and its comma - the one after it, or for the last element the one before it - and the
          element's line when it has the line to itself; never a comment line between two elements.
        """
        element = node.elements[index]
        start, end = _value_start(element), element.end
        last = index + 1 == len(node.elements)
        spans: list[tuple[int, int]] = []
        if not last:
            end = node.commas[index] + 1
        elif index > 0:
            # Only the comma itself: a comment between it and the element is the previous element's or the list's.
            comma = node.commas[index - 1]
            spans.append((comma, comma + 1))
        end_of_line = self._end_of_line(end)
        if self._starts_line(start) and end_of_line is not None:
            spans.append(
                (self._line_start(start), min(end_of_line + 1, len(self._text)))
            )
        elif not last:
            after = end
            while after < len(self._text) and self._text[after] in " \t":
                after += 1
            spans.append((start, after))
        else:
            before = start
            while before > 0 and self._text[before - 1] in " \t":
                before -= 1
            spans.append((before, end))
        return spans


def _value_start(node: ScalarNode | BlockNode) -> int:
    """The offset of a list element's first character."""
    if isinstance(node, ScalarNode):
        return node.start
    return 0 if node.open is None else node.open


def _key_fields(segment: PathSegment) -> tuple[tuple[str, Value], ...]:
    """The key field a new element selected by `segment` starts with: (field, value as a string), or none."""
    return ((segment.key[0], segment.key[1]),) if segment.key is not None else ()


def parse(text: str, source: str = "<text>") -> TextprotoDocument:
    """The document of `text`; TextprotoSyntaxError when it is not textproto syntax."""
    return TextprotoDocument(text, source)


def load(path: str) -> TextprotoDocument:
    """The document of the file at `path`, read as UTF-8 with its line endings kept; its errors name the path."""
    with open(path, encoding="utf-8", newline="") as file:
        return TextprotoDocument(file.read(), path)
