#!/usr/bin/env python3
"""Flags bare literal arguments that need a Google-style argument comment.

The Google C++ style guide asks for the meaning of a non-obvious function argument to be visible at the call site, and
this repository writes that as an argument comment carrying the parameter's name:

    ContactRectangle::loadContactRectangle(taskFile_, modelSettings_, /*contactIndex=*/0, verbose_);

A bare `0` there says nothing. This check finds every argument that is a lone numeric, boolean or `nullptr` literal
without a `/*name=*/` comment directly in front of it, in a call with two or more arguments. The name itself is not
verified - that needs a compiler (clang-tidy's bugprone-argument-comment) - so copy it from the callee's declaration.

It is a syntactic check, so it cannot tell a function call from every other use of parentheses, and some literals are
conventional enough that a comment would be noise. Those are exempt:
  - calls whose arguments are ALL numeric literals: element access `R(2, 2)`, coordinates `vector3_t(0.0, 0.0, 1.0)`;
  - one-argument calls: the function's name says what its only argument is;
  - the callees in CONVENTIONAL_CALLEES and the types matched by CONVENTIONAL_TYPE_PATTERNS (std::max, block, resize,
    StrCat, EXPECT_NEAR, Eigen vectors, ...), where the position of a literal is its meaning;
  - a line carrying `// NOLINT(argument-comment)`, or following a line carrying `// NOLINTNEXTLINE(argument-comment)`.
Preprocessor lines (#define bodies included) are not checked.
"""

import argparse
import os
import re
import subprocess
import sys
from typing import Iterable, List, NamedTuple, Optional, Tuple

NOLINT = "NOLINT(argument-comment)"

CPP_EXTENSIONS = (".cpp", ".cc", ".cxx", ".h", ".hh", ".hpp", ".hxx")
# Third-party code this repository vendors: not ours to restyle.
# LINT.IfChange(vendored_dirs)
VENDORED_DIRS = ("lib/ocs2/", "lib/mujoco_vendor/", "tools/ifttt-lint/")
# LINT.ThenChange(//tools/hooks/lint_code.py:vendored_dirs)
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
        "Substitute",
        "printf",
        "snprintf",
        "fprintf",
    }
)

# Macros and types whose literal arguments are conventional, matched on the callee's (or declared type's) name.
CONVENTIONAL_TYPE_PATTERNS = [
    re.compile(
        r"^(EXPECT|ASSERT|CHECK|DCHECK|QCHECK)_\w+$"
    ),  # gtest and Abseil check macros
    re.compile(
        r"^(LOG|VLOG|PLOG|LOG_EVERY_N|LOG_FIRST_N|LOG_IF|DLOG)\w*$"
    ),  # Abseil logging
    re.compile(
        r"^(Vector|Matrix|Array|Quaternion|AngleAxis|Translation|RowVector)\w*$"
    ),  # Eigen types
    re.compile(
        r"^\w*(vector|matrix|quaternion|array)\d*[a-z]*_t$"
    ),  # this repository's Eigen typedefs
    re.compile(
        r"^(vector|array|pair|tuple|basic_string|string|duration)$"
    ),  # std containers, as declared types
]


class Token(NamedTuple):
    kind: str  # "ident", "number", "punct", "comment", "string"
    text: str
    line: int
    col: int


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


_NUMBER = re.compile(
    r"""
    (?:0[xX][0-9a-fA-F']+(?:\.[0-9a-fA-F']*)?(?:[pP][+-]?\d+)?   # hexadecimal (and hex float)
      |0[bB][01']+                                               # binary
      |(?:\d[\d']*\.?[\d']*|\.\d[\d']*)(?:[eE][+-]?\d+)?         # decimal and floating point
    )[a-zA-Z_]*                                                   # suffix (u, l, f, ...) or user-defined literal
    """,
    re.VERBOSE,
)
_STANDARD_SUFFIX = re.compile(
    r"^(?:[uU](?:ll|LL|l|L|z|Z)?|(?:ll|LL|l|L|z|Z)[uU]?|[fFlL])?$"
)
_ARGUMENT_COMMENT = re.compile(r"^/\*\s*[A-Za-z_]\w*\s*=\s*\*/$")
_PUNCT2 = (
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
)


def tokenize(source: str) -> List[Token]:
    """Splits C++ source into tokens, skipping preprocessor lines and keeping block comments."""
    tokens: List[Token] = []
    i, line, col = 0, 1, 1
    n = len(source)
    at_line_start = True

    def advance(count: int) -> None:
        nonlocal i, line, col
        for _ in range(count):
            if source[i] == "\n":
                line += 1
                col = 1
            else:
                col += 1
            i += 1

    while i < n:
        c = source[i]
        if c == "\n":
            at_line_start = True
            advance(1)
            continue
        if c in " \t\r\f\v":
            advance(1)
            continue
        # Preprocessor directive: skip it and its continuation lines.
        if c == "#" and at_line_start:
            while i < n:
                if source[i] == "\\" and i + 1 < n and source[i + 1] == "\n":
                    advance(2)
                    continue
                if source[i] == "\n":
                    break
                advance(1)
            continue
        at_line_start = False
        start_line, start_col = line, col
        if source.startswith("//", i):
            end = source.find("\n", i)
            end = n if end < 0 else end
            tokens.append(Token("comment", source[i:end], start_line, start_col))
            advance(end - i)
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            end = n if end < 0 else end + 2
            tokens.append(Token("comment", source[i:end], start_line, start_col))
            advance(end - i)
            continue
        raw = re.match(r'(?:u8|u|U|L)?R"([^()\\\s]{0,16})\(', source[i:])
        if raw:
            terminator = ")" + raw.group(1) + '"'
            end = source.find(terminator, i + raw.end())
            end = n if end < 0 else end + len(terminator)
            tokens.append(Token("string", source[i:end], start_line, start_col))
            advance(end - i)
            continue
        prefix = re.match(r"(?:u8|u|U|L)?[\"']", source[i:])
        if prefix:
            quote = prefix.group(0)[-1]
            j = i + prefix.end()
            while j < n and source[j] != quote:
                j += 2 if source[j] == "\\" else 1
            end = min(j + 1, n)
            tokens.append(Token("string", source[i:end], start_line, start_col))
            advance(end - i)
            continue
        if c.isdigit() or (c == "." and i + 1 < n and source[i + 1].isdigit()):
            match = _NUMBER.match(source, i)
            text = match.group(0) if match else c
            tokens.append(Token("number", text, start_line, start_col))
            advance(len(text))
            continue
        if c.isalpha() or c == "_":
            match = re.match(r"[A-Za-z_]\w*", source[i:])
            tokens.append(Token("ident", match.group(0), start_line, start_col))
            advance(match.end())
            continue
        two = source[i : i + 2]
        if two in _PUNCT2:
            tokens.append(Token("punct", two, start_line, start_col))
            advance(2)
            continue
        tokens.append(Token("punct", c, start_line, start_col))
        advance(1)
    return tokens


_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = frozenset(_OPEN.values())


def _callee_before(code: List[Token], paren: int) -> Optional[Tuple[str, int]]:
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


def _declared_type_before(code: List[Token], name_index: int) -> Optional[str]:
    """For `Type name(...)`, the type's own name; None when the name is not preceded by a type.

    A later declarator of a list, the `b` of `Type a(...), b(...)`, has the type of the declaration: the walk goes back
    over the earlier `name(...)` declarators to it.
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


def _is_conventional(name: Optional[str]) -> bool:
    return name is not None and (
        name in CONVENTIONAL_CALLEES or _is_conventional_type(name)
    )


def _literal_text(argument: List[Token]) -> Optional[str]:
    """The literal an argument consists of - numeric (optionally signed), bool or nullptr - or None."""
    body = [t for t in argument if t.kind != "comment"]
    if len(body) == 2 and body[0].text in ("-", "+") and body[1].kind == "number":
        body = [Token("number", body[0].text + body[1].text, body[0].line, body[0].col)]
    if len(body) != 1:
        return None
    token = body[0]
    if token.kind == "ident" and token.text in ("true", "false", "nullptr"):
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


def _has_argument_comment(argument: List[Token]) -> bool:
    """True when a `/*name=*/` comment stands directly in front of the argument's value."""
    for token in argument:
        if token.kind != "comment":
            return False
        if _ARGUMENT_COMMENT.match(token.text):
            return True
    return False


def check_source(source: str, path: str = "<source>") -> List[Violation]:
    tokens = tokenize(source)
    lines = source.splitlines()
    suppressed = set()
    for number, text in enumerate(lines, start=1):
        if NOLINT in text:
            suppressed.add(number)
        if NOLINTNEXTLINE in text:
            suppressed.add(number + 1)

    # Line comments are irrelevant to the analysis; block comments stay, to see argument comments.
    code = [t for t in tokens if not (t.kind == "comment" and t.text.startswith("//"))]
    violations: List[Violation] = []
    for index, token in enumerate(code):
        if token.text != "(":
            continue
        found = _callee_before(code, index)
        if found is None:
            continue
        callee, callee_index = found
        # Split the argument list at the commas of this parenthesis level. The template argument list of a conventional
        # type is a level of its own: the sizes of `Eigen::Matrix<SCALAR, 3, 3>` are not arguments of the call around it.
        arguments: List[List[Token]] = [[]]
        stack = []
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
        if len(arguments) < 2:
            continue
        literals = [_literal_text(a) for a in arguments]
        if all(
            text is not None and text not in ("true", "false", "nullptr")
            for text in literals
        ):
            continue  # element access and coordinates: R(2, 2), vector3_t(0.0, 0.0, 1.0)
        if _is_conventional(callee) or _is_conventional(
            _declared_type_before(code, callee_index)
        ):
            continue
        for position, (argument, text) in enumerate(zip(arguments, literals), start=1):
            if text is None or _has_argument_comment(argument):
                continue
            first = next(t for t in argument if t.kind != "comment")
            if first.line in suppressed:
                continue
            violations.append(
                Violation(path, first.line, first.col, callee, text, position)
            )
    return violations


def check_files(paths: Iterable[str], root: str) -> List[Violation]:
    violations: List[Violation] = []
    for path in paths:
        with open(path, encoding="utf-8", errors="ignore") as f:
            violations += check_source(f.read(), os.path.relpath(path, root))
    return violations


def check_staged(repository: str) -> List[Violation]:
    """Checks the STAGED version of every first-party C++ file staged for commit - the index, not the working tree.

    This is what the pre-commit hook runs: it judges exactly what is about to be committed, a partial `git add -p`
    included, and only the files the commit touches, so a commit is never blocked by a file it does not change.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", repository, *args], capture_output=True, text=True, check=True
        ).stdout

    staged = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").split(
        "\0"
    )
    violations: List[Violation] = []
    for path in staged:
        if not path.endswith(CPP_EXTENSIONS) or path.startswith(VENDORED_DIRS):
            continue
        violations += check_source(git("show", ":" + path), path)
    return violations


def main(argv: Optional[List[str]] = None) -> int:
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
