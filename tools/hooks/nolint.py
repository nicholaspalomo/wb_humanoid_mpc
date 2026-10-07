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

"""NOLINT markers: how a line says that a rule is wrong for it, and why.

A finding that is genuinely wrong for one line is suppressed where it is, with the check's name and a reason, in a
comment:

    // NOLINT(<check>): <reason>                exempts its own line
    // NOLINTNEXTLINE(<check>): <reason>        exempts the line after it
    // NOLINTBEGIN(<check>): <reason>           exempts the lines from it ...
    // NOLINTEND(<check>)                       ... to this one

Several checks may share one marker, `NOLINT(a, b): <reason>`. In Python, Starlark, shell and .bazelrc files the comment
is a `#` comment; in C++ and .proto files a `//` or `/* */` one. A marker is read from comments only: one inside a string
literal is text, not a justification. A marker written as prose, inside backticks or quotes (`` `NOLINT(boost)` ``), is
not a marker either. In other text files (Markdown, YAML) a marker anywhere on the line counts.

A bare NOLINT, without a parenthesized list, is a marker in C++ only, where cpplint and clang-tidy would honor it; in
other files the word is prose. A marker without a reason suppresses nothing. The registry (tools/hooks/checks.py)
reports the markers themselves: `nolint-category` (a bare NOLINT, which would silence every cpplint and clang-tidy check
on the line), `nolint-reason` (no reason after the colon), `nolint-unknown` (a category no tool knows), `nolint-unused`
(a marker for one of the repository's own checks that suppresses nothing) and `nolint-unbalanced` (a NOLINTBEGIN never
closed, or a NOLINTEND that closes nothing).
"""

from collections.abc import Callable
import io
import re
import tokenize
from typing import NamedTuple

from tools.hooks import check_types
from tools.hooks import cpp_source

# How a file's comments are found.
CPP_COMMENTS = "cpp"  # `//` and `/* */`, C++ and .proto
PYTHON_COMMENTS = "python"  # `#`, found with the tokenize module
HASH_COMMENTS = "hash"  # `#` outside quotes, Starlark, shell, .bazelrc
WHOLE_LINE = "line"  # no comment syntax: Markdown, YAML, text

_MARKER = re.compile(
    r"\bNOLINT(?P<variant>NEXTLINE|BEGIN|END)?\b(?:\((?P<categories>[^)]*)\))?(?P<rest>[^\n]*)"
)
_REASON = re.compile(r"^\s*:\s*(?P<reason>\S.*)$")


class Marker(NamedTuple):
    """One NOLINT marker in a file."""

    line: int  # 1-based
    column: int  # 1-based
    variant: str  # "" (NOLINT), "NEXTLINE", "BEGIN" or "END"
    categories: tuple[str, ...]  # empty for a bare NOLINT
    reason: str  # the text after the colon, without the closing `*/`; empty when there is none
    bare: bool  # written without a parenthesized category list

    @property
    def text(self) -> str:
        """The marker as written, without its reason."""
        categories = "" if self.bare else "(" + ", ".join(self.categories) + ")"
        return f"NOLINT{self.variant}{categories}"


_COMMENT_STYLES = {
    check_types.Language.CPP: CPP_COMMENTS,
    check_types.Language.PROTO: CPP_COMMENTS,
    check_types.Language.PYTHON: PYTHON_COMMENTS,
    check_types.Language.STARLARK: HASH_COMMENTS,
    check_types.Language.BAZELRC: HASH_COMMENTS,
    check_types.Language.SHELL: HASH_COMMENTS,
}
CODE_LANGUAGES = frozenset(_COMMENT_STYLES)


def comment_style(path: str) -> str:
    """How the comments of the file at `path` are found: by its language, or WHOLE_LINE for other text files."""
    language = check_types.language_of(path)
    return _COMMENT_STYLES.get(language, WHOLE_LINE) if language else WHOLE_LINE


def _cpp_comment_lines(source: str) -> list[str]:
    """The `//` and `/* */` comments of C++ `source` where they stand."""
    without_comments, _ = cpp_source.mask(source)
    return cpp_source.comment_text(source, without_comments).split("\n")


def _python_comment_lines(source: str) -> list[str]:
    """The `#` comments of Python `source`, found by the tokenize module, where they stand."""
    lines = [" " * len(line) for line in source.split("\n")]
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                row, column = token.start
                line = lines[row - 1]
                lines[row - 1] = (
                    line[:column] + token.string + line[column + len(token.string) :]
                )
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return [hash_comment_text(line) for line in source.split("\n")]
    return lines


def hash_comment_start(line: str) -> int | None:
    """The column of the `#` that starts the comment of `line`, or None. A `#` inside a quoted string does not."""
    quote = None
    i = 0
    while i < len(line):
        if quote:
            if line[i] == "\\":
                i += 2
            elif line.startswith(quote, i):
                i += len(quote)
                quote = None
            else:
                i += 1
        elif line.startswith(('"""', "'''"), i):
            quote = line[i : i + 3]
            i += 3
        elif line[i] in "\"'":
            quote = line[i]
            i += 1
        elif line[i] == "#":
            return i
        else:
            i += 1
    return None


def hash_comment_text(line: str) -> str:
    """The `#` comment of `line` where it stands, with everything else blanked out."""
    start = hash_comment_start(line)
    return " " * len(line) if start is None else " " * start + line[start:]


def comment_lines(source: str, style: str) -> list[str]:
    """The comments of `source` where they stand, line by line, with everything else blanked out.

    Args:
      source: The file's content.
      style: How its comments are found: CPP_COMMENTS, PYTHON_COMMENTS, HASH_COMMENTS or WHOLE_LINE.

    Returns:
      One string per line of `source`.
    """
    if style == CPP_COMMENTS:
        return _cpp_comment_lines(source)
    if style == PYTHON_COMMENTS:
        return _python_comment_lines(source)
    if style == HASH_COMMENTS:
        return [hash_comment_text(line) for line in source.split("\n")]
    if style == WHOLE_LINE:
        return source.split("\n")
    raise ValueError(f"unknown comment style {style!r}")


def _is_prose(text: str, start: int) -> bool:
    """True when the marker at text[start] is quoted - a backtick or quote directly before it - and so prose."""
    return start > 0 and text[start - 1] in "`'\""


def parse(source: str, style: str) -> list[Marker]:
    """Every NOLINT marker in the comments of `source`.

    Args:
      source: The file's content.
      style: How its comments are found (comment_lines()).

    Returns:
      The markers, in order.
    """
    markers = []
    for number, text in enumerate(comment_lines(source, style), start=1):
        for match in _MARKER.finditer(text):
            if _is_prose(text, match.start()):
                continue
            categories_text = match.group("categories")
            bare = categories_text is None
            if bare and style != CPP_COMMENTS:
                # Only cpplint and clang-tidy read a bare NOLINT, and they read C++: elsewhere the word is prose.
                continue
            categories = tuple(
                c.strip() for c in (categories_text or "").split(",") if c.strip()
            )
            rest = match.group("rest").split("*/", 1)[0].rstrip()
            reason_match = _REASON.match(rest)
            reason = reason_match.group("reason").strip() if reason_match else ""
            markers.append(
                Marker(
                    line=number,
                    column=match.start() + 1,
                    variant=match.group("variant") or "",
                    categories=categories,
                    reason=reason,
                    bare=bare,
                )
            )
    return markers


class Problem(NamedTuple):
    """A marker that is itself wrong: the name of the check that reports it, where, and why."""

    check: str
    line: int
    column: int
    message: str


class Suppressions:
    """Which lines the NOLINT markers of one file exempt, for which checks, and which markers were used."""

    def __init__(self, markers: list[Marker]) -> None:
        # (category, line) -> the indices of the markers that exempt it
        self._exempt: dict[tuple[str, int], list[int]] = {}
        self._used: set[int] = set()
        self.markers = markers
        self.unbalanced: list[Problem] = []
        open_blocks: dict[str, list[int]] = {}
        for index, marker in enumerate(markers):
            if marker.bare:
                continue
            for category in marker.categories:
                if marker.variant == "":
                    self._add(category, marker.line, index)
                elif marker.variant == "NEXTLINE":
                    self._add(category, marker.line + 1, index)
                elif marker.variant == "BEGIN":
                    open_blocks.setdefault(category, []).append(index)
                elif open_blocks.get(category):
                    begin = open_blocks[category].pop()
                    for line in range(markers[begin].line, marker.line + 1):
                        self._add(category, line, begin)
                    self._used.add(index)  # an END is used when its BEGIN is
                else:
                    self.unbalanced.append(
                        Problem(
                            "nolint-unbalanced",
                            marker.line,
                            marker.column,
                            f"`NOLINTEND({category})` closes no NOLINTBEGIN({category}).",
                        )
                    )
        for category, indices in open_blocks.items():
            for index in indices:
                marker = markers[index]
                self.unbalanced.append(
                    Problem(
                        "nolint-unbalanced",
                        marker.line,
                        marker.column,
                        f"`NOLINTBEGIN({category})` is never closed by a NOLINTEND({category}), so it exempts "
                        "nothing.",
                    )
                )

    def _add(self, category: str, line: int, index: int) -> None:
        self._exempt.setdefault((category, line), []).append(index)

    def suppresses(self, check: str, line: int) -> bool:
        """True when a marker with a reason exempts `line` from `check`; that marker then counts as used."""
        indices = [
            i for i in self._exempt.get((check, line), []) if self.markers[i].reason
        ]
        self._used.update(indices)
        return bool(indices)

    def unused(self, categories: frozenset[str]) -> list[Marker]:
        """The markers naming one of `categories` that exempted no finding (NOLINTEND markers excepted)."""
        return [
            marker
            for index, marker in enumerate(self.markers)
            if index not in self._used
            and marker.variant != "END"
            and not marker.bare
            and any(c in categories for c in marker.categories)
        ]


def marker_problems(
    markers: list[Marker], known: Callable[[str], bool]
) -> list[Problem]:
    """The problems of the markers themselves, other than unused and unbalanced ones.

    Args:
      markers: The markers of one file (parse()).
      known: Says whether a category is a check some tool knows.

    Returns:
      nolint-category, nolint-reason and nolint-unknown problems.
    """
    problems = []
    for marker in markers:
        if marker.bare or not marker.categories:
            problems.append(
                Problem(
                    "nolint-category",
                    marker.line,
                    marker.column,
                    f"`{marker.text}` names no check, so it would silence every cpplint and clang-tidy check on the "
                    "line: write `NOLINT(<check>): <reason>`.",
                )
            )
            continue
        if marker.variant != "END" and not marker.reason:
            problems.append(
                Problem(
                    "nolint-reason",
                    marker.line,
                    marker.column,
                    f"`{marker.text}` gives no reason: write `{marker.text}: <why the rule is wrong here>`. A marker "
                    "without a reason exempts nothing.",
                )
            )
        for category in marker.categories:
            if not known(category):
                problems.append(
                    Problem(
                        "nolint-unknown",
                        marker.line,
                        marker.column,
                        f"`{category}` in `{marker.text}` is no check of tools/hooks/checks.py, no cpplint category "
                        "and no clang-tidy check of .clang-tidy.",
                    )
                )
    return problems
