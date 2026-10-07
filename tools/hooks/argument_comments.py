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

"""Flags bare literal arguments that need a Google-style argument comment.

The Google C++ style guide asks for the meaning of a non-obvious function argument to be visible at the call site, and
this repository writes that as an argument comment carrying the parameter's name:

    ContactRectangle::loadContactRectangle(taskFile_, modelSettings_, /*contactIndex=*/0, verbose_);

A bare `0` there says nothing. This check finds every argument that is a lone numeric, boolean, `nullptr` or
character literal, or an empty value (`{}`, `""`, `std::nullopt`), without a `/*name=*/` comment directly in front of
it. The name itself is not verified - that needs a compiler (clang-tidy's bugprone-argument-comment) - so copy it from
the callee's declaration.

It is a syntactic check, so it cannot tell a function call from every other use of parentheses, and some literals are
conventional enough that a comment would be noise. Those are exempt:
  - calls whose arguments are ALL numeric literals: element access `R(2, 2)`, coordinates `vector3_t(0.0, 0.0, 1.0)`;
  - one-argument calls, where the function's name says what its only argument is, except a `true`, `false` or
    `nullptr` argument: that one needs its comment unless the callee is a setter (`set*`, `enable*`, `disable*`, a
    protobuf `add_*`), a member or variable being initialized (`stopped_(false)`), or conventional;
  - the callees in CONVENTIONAL_CALLEES and the types matched by CONVENTIONAL_TYPE_PATTERNS (std::max, block, resize,
    StrCat, EXPECT_NEAR, Eigen vectors, ...), where the position of a literal is its meaning;
  - a line carrying `// NOLINT(argument-comment)`, or following a line carrying `// NOLINTNEXTLINE(argument-comment)`.
Preprocessor lines (#define bodies included) are not checked.
"""

import argparse
from collections.abc import Iterable
import os
import re
import sys
from typing import NamedTuple

from tools.hooks import check_types
from tools.hooks import cpp_source
from tools.hooks import lint_files

NAME = "argument-comment"
NOLINT = "NOLINT(argument-comment)"
NOLINTNEXTLINE = "NOLINTNEXTLINE(argument-comment)"

# Words that are followed by a parenthesis without being a call.
KEYWORDS = frozenset(
    {
        "alignas",
        "alignof",
        "catch",
        "co_await",
        "co_return",
        "co_yield",
        "const_cast",
        "decltype",
        "defined",
        "delete",
        "dynamic_cast",
        "for",
        "if",
        "new",
        "noexcept",
        "operator",
        "reinterpret_cast",
        "requires",
        "return",
        "sizeof",
        "static_assert",
        "static_cast",
        "switch",
        "throw",
        "typeid",
        "while",
        "__attribute__",
        "__declspec",
    }
)

# Callees whose literal arguments are conventional: the position says what the literal is.
CONVENTIONAL_CALLEES = frozenset(
    {
        # <algorithm>, <cmath>, <numeric>
        "abs",
        "accumulate",
        "atan2",
        "clamp",
        "copysign",
        "fill",
        "fill_n",
        "fmax",
        "fmin",
        "fmod",
        "hypot",
        "iota",
        "max",
        "min",
        "pow",
        # <cstring>: glibc leaves the fill byte of memset unnamed, so no argument comment could name it.
        "memset",
        # containers and strings
        "assign",
        "at",
        "compare",
        "emplace",
        "emplace_back",
        "erase",
        "find",
        "insert",
        "make_pair",
        "push_back",
        "replace",
        "reserve",
        "resize",
        "rfind",
        "substr",
        "append",
        "find_first_of",
        "find_first_not_of",
        "find_last_of",
        "find_last_not_of",
        "value_or",
        # <atomic>: the value stored or exchanged is the call's whole meaning.
        "store",
        "exchange",
        "fetch_add",
        "fetch_sub",
        # <ctime>: time(nullptr) is the C API's spelling of "now".
        "time",
        # this repository's fill: a feet_array_t holding one value per foot (humanoid_common_mpc/common/Types.h)
        "makeFeetArray",
        # Eigen
        "block",
        "bottomLeftCorner",
        "bottomRightCorner",
        "bottomRows",
        "coeff",
        "coeffRef",
        "col",
        "conservativeResize",
        "Constant",
        "head",
        "Identity",
        "isApprox",
        "isApproxToConstant",
        "isMuchSmallerThan",
        "isZero",
        "leftCols",
        "LinSpaced",
        "middleCols",
        "middleRows",
        "Ones",
        "Random",
        "rightCols",
        "row",
        "segment",
        "setConstant",
        "setIdentity",
        "setOnes",
        "setZero",
        "tail",
        "topLeftCorner",
        "topRightCorner",
        "topRows",
        "Zero",
        # Abseil strings: the literal is part of the text being built.
        "StrAppend",
        "StrCat",
        "StrFormat",
        "StrJoin",
        "StrSplit",
        "StrContains",
        "StrReplaceAll",
        "StartsWith",
        "EndsWith",
        "ConsumePrefix",
        "ConsumeSuffix",
        "StripPrefix",
        "StripSuffix",
        "MaxSplits",
        "ByChar",
        "ByString",
        "ByAnyChar",
        "Substitute",
        "printf",
        "snprintf",
        "fprintf",
    }
)

# Macros and types whose literal arguments are conventional, matched on the callee's (or declared type's) name.
CONVENTIONAL_TYPE_PATTERNS = [
    re.compile(
        r"^(ABSL_)?(EXPECT|ASSERT|CHECK|DCHECK|QCHECK)_\w+$"
    ),  # gtest and Abseil check macros
    re.compile(
        r"^(LOG|VLOG|PLOG|LOG_EVERY_N|LOG_FIRST_N|LOG_IF|DLOG)\w*$"
    ),  # Abseil logging
    re.compile(r"^ABSL_(RETIRED_)?FLAG$"),  # ABSL_FLAG(type, name, default, help)
    re.compile(
        r"^(Vector|Matrix|Array|Quaternion|AngleAxis|Translation|RowVector)\w*$"
    ),  # Eigen types
    re.compile(
        r"^\w*(vector|matrix|quaternion|array)\d*[a-z]*_t$"
    ),  # this repository's Eigen typedefs
    re.compile(
        r"^(vector|array|pair|tuple|basic_string|string|duration|optional)$"
    ),  # std containers, as declared types
]


Token = cpp_source.Token


class Violation(NamedTuple):
    path: str
    line: int
    col: int
    callee: str
    literal: str
    position: int

    def __str__(self) -> str:
        return (
            f"{self.path}:{self.line}:{self.col}: bare literal `{self.literal}` as argument {self.position} of "
            f"`{self.callee}(...)`; name it with an argument comment, `/*parameter_name=*/{self.literal}` (Google "
            f"C++ style), or add `// {NOLINT}` if it is self-evident."
        )


_STANDARD_SUFFIX = re.compile(
    r"^(?:[uU](?:ll|LL|l|L|z|Z)?|(?:ll|LL|l|L|z|Z)[uU]?|[fFlL])?$"
)
_ARGUMENT_COMMENT = re.compile(r"^/\*\s*[A-Za-z_]\w*\s*=\s*\*/$")
_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = frozenset(_OPEN.values())


def _callee_before(code: list[Token], paren: int) -> tuple[str, int] | None:
    """The name a `(` at code[paren] calls and its index, or None when the parenthesis is not a call."""
    k = paren - 1
    if k < 0:
        return None
    if code[k].text in (
        ">",
        ">>",
    ):  # a template argument list: name<...>( or name<A<B>>(
        depth = 0
        while k >= 0:
            if code[k].text == ">":
                depth += 1
            elif code[k].text == ">>":
                depth += 2
            elif code[k].text == "<":
                depth -= 1
                if depth == 0:
                    break
            k -= 1
        k -= 1
        if k < 0:
            return None
    if code[k].kind != "ident" or code[k].text in KEYWORDS:
        return None
    return code[k].text, k


def _declared_type_before(code: list[Token], name_index: int) -> str | None:
    """For `Type name(...)`, the type's own name; None when the name is not preceded by a type.

    A later declarator of a list, the `b` of `Type a(...), b(...)`, has the type of the declaration: the walk goes back
    over the earlier `name(...)` declarators to it.

    Args:
        code: The tokens of the file, comments removed.
        name_index: The index in `code` of the declared name.

    Returns:
        The last identifier of the declared type (`Foo` for `ns::Foo`), or None.
    """
    k = name_index - 1
    while k >= 1 and code[k].text == "," and code[k - 1].text == ")":
        depth = 0
        j = k - 1
        while j >= 0:
            if code[j].text == ")":
                depth += 1
            elif code[j].text == "(":
                depth -= 1
                if depth == 0:
                    break
            j -= 1
        if j < 1 or code[j - 1].kind != "ident" or code[j - 1].text in KEYWORDS:
            return None
        k = j - 2  # the token in front of the earlier declarator's name
    if k < 0:
        return None
    if code[k].text in (">", ">>"):
        depth = 0
        while k >= 0:
            if code[k].text == ">":
                depth += 1
            elif code[k].text == ">>":
                depth += 2
            elif code[k].text == "<":
                depth -= 1
                if depth == 0:
                    break
            k -= 1
        k -= 1
    if k >= 0 and code[k].kind == "ident" and code[k].text not in KEYWORDS:
        return code[k].text
    return None


def _is_conventional_type(name: str) -> bool:
    return any(pattern.match(name) for pattern in CONVENTIONAL_TYPE_PATTERNS)


def _is_conventional(name: str | None) -> bool:
    return name is not None and (
        name in CONVENTIONAL_CALLEES or _is_conventional_type(name)
    )


# The literals whose meaning a call site cannot show: besides numbers, these (and character literals).
_BOOL_OR_NULL = frozenset({"true", "false", "nullptr"})
_EMPTY_VALUES = frozenset({"{}", '""', "std::nullopt", "nullopt"})
_CHARACTER = re.compile(r"^(?:u8|u|U|L)?'(?:[^'\\]|\\.){1,4}'$")


def _literal_text(argument: list[Token]) -> str | None:
    """The literal an argument consists of, or None.

    Numeric (optionally signed), `true` / `false` / `nullptr`, a character literal, and the empty values `{}`, `""` and
    `std::nullopt`, whose meaning a call site shows no more than a number's.

    Args:
      argument: The tokens of one argument, comments included.

    Returns:
      The literal's text (`-1`, `{}`, `std::nullopt`), or None when the argument is anything else.
    """
    body = [t for t in argument if t.kind != "comment"]
    if len(body) == 2 and body[0].text in ("-", "+") and body[1].kind == "number":
        body = [Token("number", body[0].text + body[1].text, body[0].line, body[0].col)]
    joined = "".join(t.text for t in body)
    if joined in _EMPTY_VALUES and len(body) in (1, 2, 3):
        return joined
    if len(body) != 1:
        return None
    token = body[0]
    if token.kind == "ident" and token.text in _BOOL_OR_NULL:
        return token.text
    if token.kind == "string" and _CHARACTER.match(token.text):
        return token.text
    if token.kind == "number":
        suffix = re.sub(
            r"^[-+]?(?:0[xX][0-9a-fA-F'.]+(?:[pP][+-]?\d+)?|0[bB][01']+|[\d'.]+(?:[eE][+-]?\d+)?)",
            "",
            token.text,
        )
        # A user-defined literal (100ms, 2s) carries its unit, which is the explanation.
        return token.text if _STANDARD_SUFFIX.match(suffix) else None
    return None


def _is_number(text: str | None) -> bool:
    """True for the text of a numeric literal (not a bool, null, character or empty value)."""
    return (
        text is not None
        and text not in _BOOL_OR_NULL
        and text not in _EMPTY_VALUES
        and not text.endswith("'")
    )


# A one-argument call whose name already says what its argument is: a setter, or a protobuf repeated-field adder.
_SETTER = re.compile(r"^(?:(?:set|Set|enable|Enable|disable|Disable)[A-Z_]|add_[a-z])")


def _single_argument_needs_comment(
    callee: str, literal: str | None, declared_type: str | None
) -> bool:
    """True when the one argument of `callee(literal)` needs a comment: a bool or null the name does not explain.

    A one-argument call's name usually says what its argument is (`resize(3)`, `push_back('x')`); a `true`, `false` or
    `nullptr` is the exception (`reset(true)`), unless the callee is a setter (`setVerbose(true)`), a member or variable
    being initialized (`stopped_(false)`, `std::atomic<bool> stopped(false)`), or conventional.

    Args:
      callee: The name the call calls.
      literal: Its argument's literal (_literal_text()), or None.
      declared_type: The type of the declaration the call constructs (`Type name(...)`), or None.

    Returns:
      Whether the argument needs a `/*name=*/` comment.
    """
    if literal not in _BOOL_OR_NULL:
        return False
    if _SETTER.match(callee) or callee.endswith("_") or declared_type is not None:
        return False
    return not _is_conventional(callee)


def _has_argument_comment(argument: list[Token]) -> bool:
    """True when a `/*name=*/` comment stands directly in front of the argument's value."""
    for token in argument:
        if token.kind != "comment":
            return False
        if _ARGUMENT_COMMENT.match(token.text):
            return True
    return False


def unsuppressed_violations(source: str, path: str = "<source>") -> list[Violation]:
    """Every bare literal argument in `source`, whatever NOLINT markers it carries (the registry applies those)."""
    tokens = cpp_source.tokenize(source)
    # Line comments are irrelevant to the analysis; block comments stay, to see argument comments.
    code = [t for t in tokens if not (t.kind == "comment" and t.text.startswith("//"))]
    violations: list[Violation] = []
    for index, token in enumerate(code):
        if token.text != "(":
            continue
        found = _callee_before(code, index)
        if found is None:
            continue
        callee, callee_index = found
        # Split the argument list at the commas of this parenthesis level. The template argument list of a conventional
        # type is a level of its own: the sizes of `Eigen::Matrix<SCALAR, 3, 3>` are not arguments of the call around it.
        arguments: list[list[Token]] = [[]]
        stack: list[str] = []
        k = index + 1
        while k < len(code):
            t = code[k]
            if t.text in _CLOSE:
                # A `<` that never closed was a comparison (`array < 3`), not a template argument list.
                while stack and stack[-1] == ">":
                    stack.pop()
            if t.text in _OPEN:
                stack.append(_OPEN[t.text])
            elif (
                t.text == "<"
                and code[k - 1].kind == "ident"
                and _is_conventional_type(code[k - 1].text)
            ):
                stack.append(">")
            elif t.text == ">>" and stack and stack[-1] == ">":
                stack.pop()
                if stack and stack[-1] == ">":
                    stack.pop()
            elif stack and t.text == stack[-1]:
                stack.pop()
            elif not stack and t.text == ")":
                break
            elif not stack and t.text == ",":
                arguments.append([])
                k += 1
                continue
            arguments[-1].append(t)
            k += 1
        arguments = [a for a in arguments if a] if any(arguments) else []
        if not arguments:
            continue
        literals = [_literal_text(a) for a in arguments]
        declared_type = _declared_type_before(code, callee_index)
        if len(arguments) == 1 and not _single_argument_needs_comment(
            callee, literals[0], declared_type
        ):
            continue
        if len(arguments) > 1 and all(_is_number(text) for text in literals):
            continue  # element access and coordinates: R(2, 2), vector3_t(0.0, 0.0, 1.0)
        if _is_conventional(callee) or _is_conventional(declared_type):
            continue
        for position, (argument, text) in enumerate(zip(arguments, literals), start=1):
            if text is None or _has_argument_comment(argument):
                continue
            first = next(t for t in argument if t.kind != "comment")
            violations.append(
                Violation(path, first.line, first.col, callee, text, position)
            )
    return violations


def check_source(source: str, path: str = "<source>") -> list[Violation]:
    """The violations of `source` that no `NOLINT(argument-comment)` or `NOLINTNEXTLINE(argument-comment)` exempts."""
    suppressed = set()
    for number, text in enumerate(source.splitlines(), start=1):
        if NOLINT in text:
            suppressed.add(number)
        if NOLINTNEXTLINE in text:
            suppressed.add(number + 1)
    return [
        v for v in unsuppressed_violations(source, path) if v.line not in suppressed
    ]


def check_files(paths: Iterable[str], root: str) -> list[Violation]:
    violations: list[Violation] = []
    for path in paths:
        with open(path, encoding="utf-8", errors="ignore") as f:
            violations += check_source(f.read(), os.path.relpath(path, root))
    return violations


def check_staged(repository: str) -> list[Violation]:
    """Checks the STAGED version of every first-party C++ file staged for commit - the index, not the working tree.

    This is what the pre-commit hook runs: it judges exactly what is about to be committed, a partial `git add -p`
    included, and only the files the commit touches, so a commit is never blocked by a file it does not change.

    Args:
        repository: The root of the git checkout.

    Returns:
        The violations of the staged files, file by file.
    """
    violations: list[Violation] = []
    for path in lint_files.staged_files(repository):
        if not path.endswith(
            lint_files.CPP_EXTENSIONS
        ) or not lint_files.is_first_party(path):
            continue
        violations += check_source(lint_files.staged_source(repository, path), path)
    return violations


def _findings(source: str, path: str) -> list[check_types.Finding]:
    return [
        check_types.Finding(
            v.path,
            v.line,
            v.col,
            NAME,
            f"bare literal `{v.literal}` as argument {v.position} of `{v.callee}(...)`: name it with an argument "
            f"comment, `/*parameterName=*/{v.literal}`, with the name from the callee's declaration (Google C++ style, "
            "Function argument comments).",
        )
        for v in unsuppressed_violations(source, path)
    ]


CHECKS = [
    check_types.Check(
        name=NAME,
        languages=frozenset({check_types.Language.CPP}),
        scope=lint_files.Scope.FIRST_PARTY,
        check_source=_findings,
        description="literal arguments carry a /*parameterName=*/ comment (AGENTS.md; Google C++ style).",
        hint="Write /*parameterName=*/ in front of each literal, with the name from the callee's declaration "
        "(tools/hooks/argument_comments.py says what is exempt).",
    )
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="C++ files to check")
    parser.add_argument(
        "--git-staged",
        action="store_true",
        help="check the staged version of every C++ file staged for commit (the pre-commit hook)",
    )
    args = parser.parse_args(argv)
    if args.git_staged == bool(args.paths):
        parser.error("give either C++ paths or --git-staged")
    if args.git_staged:
        violations = check_staged(os.getcwd())
    else:
        violations = check_files(args.paths, os.getcwd())
    for violation in violations:
        print(violation)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
