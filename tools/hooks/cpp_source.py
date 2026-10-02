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

"""The C++ tokenizer and masker every C++ check in tools/hooks shares.

Two views of a C++ source:

- `tokenize()` splits it into tokens - identifiers, numbers, punctuators, comments and string or character literals -
  each with its line, its column and whether whitespace precedes it (`ws_before`). Preprocessor directives are skipped
  with their continuation lines, or kept as one `preprocessor` token each. It handles raw strings (`R"d(...)d"`),
  digit separators (`1'000`), `//` comments continued by a backslash and directives continued the same way.
- `mask()` returns the source with its comments blanked out, and with its comments and the text of its string and
  character literals blanked out. Both keep the length and the line structure of the source, so an offset or a
  line/column in a mask is one in the source; regular expressions then run over code only.

Both are pure lexers: they know nothing of types or templates. They are cached on the source text, since every check
of a file reads the same one.
"""

import functools
import re
from typing import NamedTuple

IDENTIFIER = "ident"
NUMBER = "number"
PUNCTUATOR = "punct"
COMMENT = "comment"
STRING = "string"
PREPROCESSOR = "preprocessor"


class Token(NamedTuple):
    """A C++ token: its kind (IDENTIFIER, NUMBER, ...), text, 1-based line and column, and preceding whitespace."""

    kind: str
    text: str
    line: int
    col: int
    # Whitespace, a line break or a comment comes directly before the token (or it starts the file).
    ws_before: bool = False


_NUMBER = re.compile(
    r"""
    (?:0[xX][0-9a-fA-F']+(?:\.[0-9a-fA-F']*)?(?:[pP][+-]?\d+)?   # hexadecimal (and hex float)
      |0[bB][01']+                                               # binary
      |(?:\d[\d']*\.?[\d']*|\.\d[\d']*)(?:[eE][+-]?\d+)?         # decimal and floating point
    )[a-zA-Z_]*                                                   # suffix (u, l, f, ...) or user-defined literal
    """,
    re.VERBOSE,
)
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_RAW_STRING = re.compile(r'(?:u8|u|U|L)?R"([^()\\\s]{0,16})\(')
_QUOTE = re.compile(r"(?:u8|u|U|L)?[\"']")
_PUNCTUATORS3 = ("->*", "...", "<=>", "<<=", ">>=")
_PUNCTUATORS2 = frozenset(
    {
        "::",
        "->",
        "<<",
        ">>",
        "<=",
        ">=",
        "==",
        "!=",
        "&&",
        "||",
        "++",
        "--",
        "+=",
        "-=",
        "*=",
        "/=",
        "%=",
        "&=",
        "|=",
        "^=",
        ".*",
        "##",
    }
)


def _line_comment_end(source: str, start: int) -> int:
    """The end of the `//` comment at `start`: its line, and the lines a backslash at the end of a line continues it on."""
    n = len(source)
    end = start
    while True:
        newline = source.find("\n", end)
        if newline < 0:
            return n
        last = newline - 1
        if last > start and source[last] == "\r":
            last -= 1
        if last > start and source[last] == "\\":
            end = newline + 1
            continue
        return newline


def _block_comment_end(source: str, start: int) -> int:
    end = source.find("*/", start + 2)
    return len(source) if end < 0 else end + 2


def _directive_end(source: str, start: int) -> int:
    """The end of the preprocessor directive at `start`: its continuation lines and the comments in it included."""
    n = len(source)
    i = start
    while i < n:
        c = source[i]
        if c == "\\" and source.startswith("\n", i + 1):
            i += 2
            continue
        if c == "\\" and source.startswith("\r\n", i + 1):
            i += 3
            continue
        if c == "\n":
            return i
        if source.startswith("/*", i):
            i = _block_comment_end(source, i)
            continue
        if source.startswith("//", i):
            return _line_comment_end(source, i)
        i += 1
    return n


def _quoted_end(source: str, start: int, quote: str) -> int:
    """The end of the string or character literal whose opening quote is at `start`; an unterminated one ends its line."""
    n = len(source)
    j = start + 1
    while j < n and source[j] != quote and source[j] != "\n":
        j += 2 if source[j] == "\\" else 1
    j = min(j, n)
    return j + 1 if j < n and source[j] == quote else j


@functools.lru_cache(maxsize=16)
def _tokenize(source: str, keep_preprocessor: bool) -> tuple[Token, ...]:
    """The tokens of `source` (tokenize() is the cached public entry point)."""
    tokens: list[Token] = []
    n = len(source)
    i = 0
    line = 1
    line_start = 0  # offset of the first character of the current line
    at_line_start = True
    whitespace = True

    def emit(kind: str, start: int, end: int) -> None:
        tokens.append(
            Token(kind, source[start:end], line, start - line_start + 1, whitespace)
        )

    def move_to(end: int) -> None:
        # Advances over source[i:end], counting the line breaks in it.
        nonlocal i, line, line_start
        breaks = source.count("\n", i, end)
        if breaks:
            line += breaks
            line_start = source.rfind("\n", i, end) + 1
        i = end

    while i < n:
        c = source[i]
        if c == "\n":
            at_line_start = True
            whitespace = True
            move_to(i + 1)
            continue
        if c in " \t\r\f\v":
            whitespace = True
            i += 1
            continue
        if c == "\\" and (
            source.startswith("\n", i + 1) or source.startswith("\r\n", i + 1)
        ):
            # A line splice outside a directive.
            whitespace = True
            move_to(i + (2 if source[i + 1] == "\n" else 3))
            continue
        if c == "#" and at_line_start:
            end = _directive_end(source, i)
            if keep_preprocessor:
                emit(PREPROCESSOR, i, end)
            move_to(end)
            whitespace = True
            continue
        at_line_start = False
        if source.startswith("//", i):
            end = _line_comment_end(source, i)
            emit(COMMENT, i, end)
            move_to(end)
            whitespace = True
            continue
        if source.startswith("/*", i):
            end = _block_comment_end(source, i)
            emit(COMMENT, i, end)
            move_to(end)
            whitespace = True
            continue
        if c in "uULR":
            raw = _RAW_STRING.match(source, i)
            if raw:
                terminator = ")" + raw.group(1) + '"'
                end = source.find(terminator, raw.end())
                end = n if end < 0 else end + len(terminator)
                emit(STRING, i, end)
                move_to(end)
                whitespace = False
                continue
        if c in "\"'uUL":
            quote = _QUOTE.match(source, i)
            if quote:
                end = _quoted_end(source, quote.end() - 1, quote.group(0)[-1])
                emit(STRING, i, end)
                move_to(end)
                whitespace = False
                continue
        if c.isdigit() or (c == "." and i + 1 < n and source[i + 1].isdigit()):
            number = _NUMBER.match(source, i)
            end = number.end() if number else i + 1
            emit(NUMBER, i, end)
            i = end
            whitespace = False
            continue
        if c.isalpha() or c == "_":
            word = _IDENTIFIER.match(source, i)
            end = word.end() if word else i + 1
            emit(IDENTIFIER, i, end)
            i = end
            whitespace = False
            continue
        three = source[i : i + 3]
        if three in _PUNCTUATORS3:
            emit(PUNCTUATOR, i, i + 3)
            i += 3
        elif source[i : i + 2] in _PUNCTUATORS2:
            emit(PUNCTUATOR, i, i + 2)
            i += 2
        else:
            emit(PUNCTUATOR, i, i + 1)
            i += 1
        whitespace = False
    return tuple(tokens)


def tokenize(source: str, keep_preprocessor: bool = False) -> tuple[Token, ...]:
    """Splits C++ `source` into tokens; comments and literals are tokens of their own.

    Args:
      source: The C++ source.
      keep_preprocessor: Whether each preprocessor directive, with its continuation lines, becomes one PREPROCESSOR
        token. By default directives are skipped.

    Returns:
      The tokens, in order.
    """
    return _tokenize(source, keep_preprocessor)


def code_tokens(source: str) -> tuple[Token, ...]:
    """The tokens of `source` without comments and preprocessor directives."""
    return tuple(token for token in tokenize(source) if token.kind != COMMENT)


_RAW_STRING_PREFIXES = frozenset({"R", "LR", "uR", "UR", "u8R"})
_RAW_STRING_OPENING = re.compile(r'"([^()\\\s]{0,16})\(')
_WORD = re.compile(r"\w+")
# A preprocessing number, digit separators (1'000'000) and exponents (1e-3, 0x1p+4) included.
_PP_NUMBER = re.compile(r"\.?\d(?:[eEpP][+-]|'\w|[\w.])*")


def _blank(chars: list[str], start: int, end: int) -> None:
    """Overwrites chars[start:end] with spaces, keeping the newlines so that lines and columns stay where they were."""
    for k in range(start, end):
        if chars[k] != "\n":
            chars[k] = " "


@functools.lru_cache(maxsize=16)
def mask(source: str) -> tuple[str, str]:
    """`source` with its comments blanked out, and with its comments and its string and character literals blanked out.

    Both have the length and the line structure of `source`, so an offset into either is an offset into `source`.
    Preprocessor directives are kept; the header name of an `#include "..."` is a literal and is blanked in the second.

    Args:
      source: The C++ source.

    Returns:
      (without_comments, code).
    """
    without_comments = list(source)
    code = list(source)
    n = len(source)
    i = 0
    while i < n:
        c = source[i]
        if source.startswith("//", i):
            end = _line_comment_end(source, i)
            _blank(without_comments, i, end)
            _blank(code, i, end)
            i = end
            continue
        if source.startswith("/*", i):
            end = _block_comment_end(source, i)
            _blank(without_comments, i, end)
            _blank(code, i, end)
            i = end
            continue
        word = _WORD.match(source, i) if (c.isalpha() or c == "_") else None
        if word:
            j = word.end()
            if j < n and source[j] == '"' and word.group(0) in _RAW_STRING_PREFIXES:
                opening = _RAW_STRING_OPENING.match(source, j)
                if opening:
                    terminator = ")" + opening.group(1) + '"'
                    end = source.find(terminator, opening.end())
                    end = n if end < 0 else end + len(terminator)
                    _blank(code, j, end)
                    i = end
                    continue
            # Any other prefix (u8"...", L'x') is a word; the literal after it is read next.
            i = j
            continue
        number = _PP_NUMBER.match(source, i) if c.isdigit() or c == "." else None
        if number:
            i = number.end()
            continue
        if c in "\"'":
            end = _quoted_end(source, i, c)
            _blank(code, i, end)
            i = end
            continue
        i += 1
    return "".join(without_comments), "".join(code)


def comment_text(source: str, without_comments: str) -> str:
    """The comments of `source` where they stand, with its code and the text of its literals blanked out.

    Args:
      source: The C++ source.
      without_comments: The first mask of `source` (mask()).

    Returns:
      A string of the length and line structure of `source`.
    """
    return "".join(
        character if blanked != character or character == "\n" else " "
        for character, blanked in zip(source, without_comments)
    )


def line_starts(text: str) -> list[int]:
    """The offset of the first character of every line of `text`; index k is line k + 1."""
    starts = [0]
    for match in re.finditer("\n", text):
        starts.append(match.end())
    return starts


def offsets(source: str, tokens: tuple[Token, ...]) -> list[int]:
    """The offset in `source` of each token of `tokens` (which tokenize() made from `source`)."""
    starts = line_starts(source)
    return [starts[token.line - 1] + token.col - 1 for token in tokens]


_OPENING = {"(": ")", "[": "]", "{": "}"}
_CLOSING = frozenset(_OPENING.values())


def matching(tokens: tuple[Token, ...], index: int) -> int:
    """The index of the bracket that closes the `(`, `[` or `{` at `index`, or len(tokens) when it is never closed."""
    stack = []
    for k in range(index, len(tokens)):
        text = tokens[k].text
        if text in _OPENING:
            stack.append(_OPENING[text])
        elif text in _CLOSING:
            while stack and stack[-1] != text:
                stack.pop()
            if stack:
                stack.pop()
            if not stack:
                return k
    return len(tokens)


def matching_angle(tokens: tuple[Token, ...], index: int) -> int:
    """Finds the end of the template argument list whose `<` is tokens[index].

    Args:
      tokens: Code tokens (code_tokens()).
      index: The index of the `<`.

    Returns:
      The index of the `>` (or of the `>>` holding it) that closes the list, or len(tokens) when the `<` is not closed
      before a `;`, `{` or `}`: it was a comparison.
    """
    depth = 0
    k = index
    while k < len(tokens):
        text = tokens[k].text
        if text in ("(", "["):
            k = matching(tokens, k) + 1
            continue
        if text == "<":
            depth += 1
        elif text == ">":
            depth -= 1
        elif text == ">>":
            depth -= 2
        elif text in (";", "{", "}"):
            return len(tokens)
        if depth <= 0 and k > index:
            return k
        k += 1
    return len(tokens)


# The kinds of brace-delimited scope.
NAMESPACE_SCOPE = "namespace"  # namespace X { and extern "C" {
CLASS_SCOPE = "class"  # class, struct and union bodies
ENUM_SCOPE = "enum"
CODE_SCOPE = "code"  # function bodies, lambdas and the blocks inside them
INIT_SCOPE = "init"  # braced initializers

_CLASS_KEYS = frozenset({"class", "struct", "union"})
_FUNCTION_QUALIFIERS = frozenset(
    {"const", "override", "final", "noexcept", "mutable", "volatile", "try", "&", "&&"}
)


def _brace_kind(tokens: tuple[Token, ...], index: int, kinds: dict[int, str]) -> str:
    """The kind of scope the `{` at tokens[index] opens, judged from the tokens before it."""
    if index == 0:
        return CODE_SCOPE
    previous = tokens[index - 1]
    if previous.text in ("=", "(", ",", "{", "return", "[", "?", ":", "<<", "+", "-"):
        if previous.text == ":" and _is_label(tokens, index - 1):
            return CODE_SCOPE
        if previous.text == "{" and kinds.get(index - 1) in (
            NAMESPACE_SCOPE,
            CLASS_SCOPE,
            CODE_SCOPE,
        ):
            return CODE_SCOPE if kinds.get(index - 1) == CODE_SCOPE else INIT_SCOPE
        return INIT_SCOPE
    if previous.text in ("else", "do", "try"):
        return CODE_SCOPE
    if previous.text == "}":
        return CODE_SCOPE  # a constructor body after a braced member initializer
    if previous.text == ";":
        return CODE_SCOPE
    # The header: the tokens since the previous statement boundary.
    start = index - 1
    while start > 0 and tokens[start - 1].text not in (";", "{", "}"):
        start -= 1
    header = tokens[start:index]
    texts = [t.text for t in header]
    if "namespace" in texts and "using" not in texts:
        return NAMESPACE_SCOPE
    if texts[:1] == ["extern"] and len(texts) == 2 and header[1].kind == STRING:
        return NAMESPACE_SCOPE
    if "enum" in texts:
        return ENUM_SCOPE
    has_parenthesis = "(" in texts
    for k, token in enumerate(header):
        if token.text in _CLASS_KEYS and (
            k == 0 or header[k - 1].text not in ("<", ",", "enum", "(", "friend")
        ):
            if "=" not in texts and (
                not has_parenthesis or "alignas" in texts or "decltype" in texts
            ):
                return CLASS_SCOPE
    if (
        previous.text == ")"
        or previous.text in _FUNCTION_QUALIFIERS
        or previous.text == "]"
    ):
        return CODE_SCOPE
    if previous.kind == IDENTIFIER or previous.text in (">", ">>"):
        # `auto f() -> T {` is a body; `Type name{...}` and `T{...}` are braced initializers.
        return CODE_SCOPE if "->" in texts else INIT_SCOPE
    return CODE_SCOPE


def _is_label(tokens: tuple[Token, ...], colon: int) -> bool:
    """True for the `:` of `case X:`, `default:` or `public:` (before a `{` it opens a block, not an initializer)."""
    k = colon - 1
    while k >= 0 and tokens[k].text not in (
        ";",
        "{",
        "}",
        "case",
        "default",
        "public",
        "private",
        "protected",
    ):
        k -= 1
    return k >= 0 and tokens[k].text in (
        "case",
        "default",
        "public",
        "private",
        "protected",
    )


def _is_access_label(tokens: tuple[Token, ...], start: int, colon: int) -> bool:
    """True when tokens[start:colon] is `public`, `private` or `protected`, the label the `:` at `colon` ends."""
    return colon - start == 1 and tokens[start].text in (
        "public",
        "private",
        "protected",
    )


class Statement(NamedTuple):
    """A run of tokens that one `;` or one opening `{` of a body ends, with the scopes it sits in."""

    start: int  # index of its first token
    end: int  # index of its terminator, the `;` or the `{` (len(tokens) when there is none)
    scopes: tuple[str, ...]  # the kinds of the enclosing scopes, outermost first
    body: str | None  # the kind of scope its `{` opens, or None when a `;` ends it


@functools.lru_cache(maxsize=16)
def _structure(
    source: str,
) -> tuple[tuple[tuple[str, ...], ...], tuple[Statement, ...]]:
    """The enclosing scopes of every code token of `source`, and its statements."""
    tokens = code_tokens(source)
    kinds: dict[int, str] = {}
    scope_of: list[tuple[str, ...]] = []
    found: list[Statement] = []
    stack: list[str] = []
    start: int | None = None
    k = 0
    n = len(tokens)
    while k < n:
        token = tokens[k]
        scope_of.append(tuple(stack))
        text = token.text
        if text == "{":
            kind = _brace_kind(tokens, k, kinds)
            kinds[k] = kind
            if kind == INIT_SCOPE:
                # The initializer belongs to the statement around it; skip over it.
                close = matching(tokens, k)
                for _ in range(k + 1, min(close + 1, n)):
                    scope_of.append(tuple(stack) + (INIT_SCOPE,))
                if start is None:
                    start = k
                k = close + 1
                continue
            found.append(
                Statement(k if start is None else start, k, tuple(stack), kind)
            )
            start = None
            stack.append(kind)
        elif text == "}":
            if start is not None:
                found.append(Statement(start, k, tuple(stack), None))
                start = None
            if stack:
                stack.pop()
            scope_of[-1] = tuple(stack)
        elif text == ";":
            found.append(
                Statement(k if start is None else start, k, tuple(stack), None)
            )
            start = None
        elif text == ":" and start is not None and _is_access_label(tokens, start, k):
            start = None
        elif start is None:
            start = k
        k += 1
    if start is not None:
        found.append(Statement(start, n, tuple(stack), None))
    return tuple(scope_of), tuple(found)


def scopes(source: str) -> tuple[tuple[str, ...], ...]:
    """For every code token of `source` (code_tokens()), the kinds of the scopes enclosing it, outermost first."""
    return _structure(source)[0]


def statements(source: str) -> tuple[Statement, ...]:
    """The statements of `source`, over its code tokens (code_tokens()), braced initializers inside them."""
    return _structure(source)[1]


def at_namespace_scope(scope: tuple[str, ...]) -> bool:
    """True when every enclosing scope is a namespace (or there is none)."""
    return all(kind == NAMESPACE_SCOPE for kind in scope)
