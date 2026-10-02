#!/usr/bin/env python3
"""Flags Boost in the project's C++ and Bazel files, so that it cannot creep back.

The project does not use Boost: ocs2::PropertyTree replaced boost::property_tree, the integrators are native ports of
the odeint loops, and the lib/ocs2 fork is Boost-free. Boost stays installed only because Pinocchio's public headers
include it, so the @pinocchio target in bazel/system_libs.bzl is the one target that depends on @boost and the one
that carries Boost's configuration defines. This check flags, in code only:

  C++ files, lib/ocs2 included. Only the vendored CppAD headers in lib/ocs2/thirdparty/ are exempt: their disabled
  uBLAS test-vector option still names Boost.
    - an #include (or #include_next) of a header under boost/, in angle brackets or quotes;
    - a header under boost/ named in another directive: a macro an #include expands (#define INC <boost/variant.hpp>),
      or an #if / #elif __has_include(<boost/optional.hpp>), continuation lines included;
    - the boost namespace: `boost::`, `::boost::`, `namespace boost`, and an alias of it (`namespace bst = boost;`),
      whose uses would otherwise all pass, since Pinocchio's headers bring Boost in without an #include of their own;
    - a Boost macro, that is any identifier that begins with BOOST_ (BOOST_FOREACH, #define BOOST_VARIANT_LIMIT_TYPES).
  Bazel files (BUILD, BUILD.bazel, *.bzl, MODULE.bazel, WORKSPACE):
    - a label in a Boost repository: @boost, @@boost, @boost//..., and the Bazel Central Registry's @boost.<library>;
    - a Boost define: -DBOOST_... in a flag (copts, cxxopts), or a string that begins with BOOST_ (a `defines` entry);
    - a Boost library linked: -lboost_..., -l:libboost_..., or a libboost_... file (linkopts, srcs).
  .bazelrc files:
    - a Boost define, -DBOOST_... (--copt, --cxxopt, --per_file_copt, ...), a Boost library linked (--linkopt), and a
      @boost label.

Only code is checked, so the odeint license notice and the notes that say what replaced Boost stay as they are. In C++,
`//` and `/* */` comments and the text of string and character literals are skipped; the header name of an #include
is still read. In Bazel files, `#` comments are skipped, and a `#` inside a quoted string on the same line does not
start one. Bazel strings ARE checked, because labels and flags are strings. A BUILD file that a .bzl writes from a
multi-line string is checked as the BUILD file it is, with its own `#` comment lines skipped.

A justified exception is marked where it is, with its reason, in a comment, as the @pinocchio target marks its own (the
text of a string literal is not a comment, so a marker there exempts nothing):
  - `NOLINT(boost): <reason>` in a comment exempts its own line;
  - `NOLINTNEXTLINE(boost): <reason>` exempts the line after it;
  - `NOLINTBEGIN(boost): <reason>` ... `NOLINTEND(boost)` exempts the lines from the one to the other.
A marker without a reason exempts nothing. A NOLINTBEGIN(boost) that is never closed, or a NOLINTEND(boost) that
closes nothing, is reported in its own right.
"""

import argparse
import bisect
import os
import re

# test_boost_usage patches boost_usage.subprocess.run to take git away from lint_files.repository_files().
import subprocess  # pylint: disable=unused-import
import sys
from typing import Iterable, List, NamedTuple, Optional, Set, Tuple

from tools.hooks import check_types
from tools.hooks import cpp_source
from tools.hooks import lint_files
from tools.hooks import nolint

NAME = "boost"

# The kinds of file this check reads, as kind_of() names them.
CPP = "C++"
STARLARK = "Starlark"
BAZELRC = "bazelrc"

CPP_EXTENSIONS = lint_files.CPP_EXTENSIONS
# The vendored CppAD headers, whose disabled uBLAS test-vector option (CPPAD_TESTVECTOR) names Boost. This is narrower
# than the vendored directories lint_code.py skips on purpose: the rest of lib/ocs2 is this project's own Boost-free
# fork of OCS2, and it is checked.
VENDORED_CPP_DIRS = ("lib/ocs2/thirdparty/",)

# What each kind of finding says, and what to do about it: C++ code has a replacement, a Bazel file has none.
_CPP_ADVICE = (
    " The project does not use Boost: use the standard library, Abseil or ocs2::PropertyTree. A justified exception "
    "says why, with `// NOLINT(boost): <reason>` on the line (tools/hooks/boost_usage.py)."
)
_BAZEL_ADVICE = " A justified exception says why, with `# NOLINT(boost): <reason>` on the line (tools/hooks/boost_usage.py)."
_MESSAGES = {
    "include": ("`{text}` includes a Boost header.", _CPP_ADVICE),
    "header": ("`{text}` names a Boost header in a directive.", _CPP_ADVICE),
    "namespace": ("`{text}` uses the boost namespace.", _CPP_ADVICE),
    "macro": ("`{text}` is a Boost macro.", _CPP_ADVICE),
    "label": (
        "`{text}` names a Boost repository. Only @pinocchio depends on Boost, because Pinocchio's headers include it "
        "(bazel/system_libs.bzl).",
        _BAZEL_ADVICE,
    ),
    "define": (
        "`{text}` is a Boost define. Boost's configuration is on the @pinocchio target's `defines` "
        "(bazel/system_libs.bzl) and on no other target.",
        _BAZEL_ADVICE,
    ),
    "link": (
        "`{text}` links a Boost library. The project links none: Pinocchio's headers need only Boost's headers, which "
        "@pinocchio carries (bazel/system_libs.bzl).",
        _BAZEL_ADVICE,
    ),
    "unclosed": (
        "`{text}` is never closed by a NOLINTEND(boost), so it exempts nothing.",
        "",
    ),
    "unopened": ("`{text}` closes no NOLINTBEGIN(boost).", ""),
}


class Violation(NamedTuple):
    path: str
    line: int
    col: int
    kind: str  # a key of _MESSAGES
    text: str

    def __str__(self) -> str:
        finding, advice = _MESSAGES[self.kind]
        return f"{self.path}:{self.line}:{self.col}: {finding.replace('{text}', self.text)}{advice}"


def kind_of(path: str) -> Optional[str]:
    """The kind of file `path` (relative to the repository root) is for this check, or None when it is not checked."""
    path = path.replace(os.sep, "/")
    name = path.rsplit("/", 1)[-1]
    if name.endswith(CPP_EXTENSIONS):
        return None if path.startswith(VENDORED_CPP_DIRS) else CPP
    if name in ("BUILD", "WORKSPACE") or name.endswith((".bazel", ".bzl", ".BUILD")):
        return STARLARK
    if name.endswith(".bazelrc"):
        return BAZELRC
    return None


# ----------------------------------------------------------------------------------------------------------------------
# C++
# ----------------------------------------------------------------------------------------------------------------------
_INCLUDE_DIRECTIVE = re.compile(r"[ \t]*#[ \t]*(?:include_next|include|import)\b")
_BOOST_HEADER = re.compile(r'[ \t]*([<"]boost/[^>"\n]*[>"]?)')
# The directives other than #include that can name a header: a macro an #include expands, and __has_include.
_HEADER_NAMING_DIRECTIVE = re.compile(r"[ \t]*#[ \t]*(?:define|if|elif)\b")
_BOOST_HEADER_ANYWHERE = re.compile(r'[<"]boost/[^>"\n]*[>"]?')
_CPP_CODE = re.compile(
    r"\bnamespace\s+boost\b"
    # An alias of the namespace itself; an alias of a nested one (= boost::python) is reported at its `boost::`.
    r"|\bnamespace\s+\w+\s*=\s*(?:::\s*)?boost\b(?!\s*::)"
    r"|\bboost\s*::|\bBOOST_\w*"
)


def mask_cpp(source: str) -> Tuple[str, str]:
    """`source` with its comments blanked out, and with its comments and its string and character literals blanked out.

    Both have the length and the line structure of `source`, so an offset into either is an offset into `source`.
    """
    return cpp_source.mask(source)


def _continues(code_line: str) -> bool:
    """True when the code of a preprocessor line ends with the backslash that continues the directive on the next."""
    return code_line.rstrip(" \t\r").endswith("\\")


def _check_cpp(
    source: str, without_comments: str, code: str
) -> List[Tuple[int, int, str, str]]:
    """The (line, column, kind, text) of every Boost token in C++ `source`, given its masks (mask_cpp())."""
    findings = []
    code_lines = code.split("\n")
    in_header_naming_directive = (
        False  # on a continuation line of a #define, #if or #elif
    )
    for number, (code_line, text_line) in enumerate(
        zip(code_lines, without_comments.split("\n")), start=1
    ):
        # The directive must be code, but a quoted header name is a literal, so it is read from the text with literals.
        directive = _INCLUDE_DIRECTIVE.match(code_line)
        if directive:
            header = _BOOST_HEADER.match(text_line, directive.end())
            if header:
                findings.append(
                    (
                        number,
                        header.start(1) + 1,
                        "include",
                        re.sub(r"\s+", "", directive.group(0)) + " " + header.group(1),
                    )
                )
        naming = _HEADER_NAMING_DIRECTIVE.match(code_line)
        if naming or in_header_naming_directive:
            for header in _BOOST_HEADER_ANYWHERE.finditer(
                text_line, naming.end() if naming else 0
            ):
                findings.append((number, header.start() + 1, "header", header.group(0)))
            in_header_naming_directive = _continues(code_line)
    line_starts = [0]
    for line in code_lines[:-1]:
        line_starts.append(line_starts[-1] + len(line) + 1)
    for match in _CPP_CODE.finditer(code):
        number = bisect.bisect_right(line_starts, match.start())
        kind = "macro" if match.group(0).startswith("BOOST_") else "namespace"
        text = re.sub(r"\s+", " ", match.group(0))
        if text.startswith("boost"):
            text = text.replace(
                " ::", "::"
            )  # `boost  ::` is reported as the `boost::` it is
        findings.append(
            (number, match.start() - line_starts[number - 1] + 1, kind, text)
        )
    return findings


# ----------------------------------------------------------------------------------------------------------------------
# Bazel: BUILD and .bzl files, and .bazelrc
# ----------------------------------------------------------------------------------------------------------------------
_LABEL = re.compile(r"(?<![\w@])@@?boost(?![\w-])[\w./:+\-]*")
_DEFINE_FLAG = re.compile(r"(?<![\w-])-D[ \t]*BOOST_\w*(?:=[^\s\"']*)?")
_DEFINE_STRING = re.compile(r"(?<=[\"'])BOOST_\w*(?:=[^\s\"']*)?")
# A Boost library linked: -lboost_system, -l:libboost_system.so.1.83.0, or the file itself in a path.
_LINK_FLAG = re.compile(r"(?<![\w-])-l[ \t]*(?::[ \t]*)?(?:lib)?boost_\w+[\w.]*")
_LIBRARY_FILE = re.compile(r"(?<![\w:])libboost_\w+[\w.]*")


def strip_hash_comment(line: str) -> str:
    """`line` with its `#` comment blanked out. A `#` inside a quoted string that opens on the line does not start one."""
    start = nolint.hash_comment_start(line)
    return line if start is None else line[:start] + " " * (len(line) - start)


def _check_bazel(source: str, kind: str) -> List[Tuple[int, int, str, str]]:
    """The (line, column, kind, text) of every Boost label and define in a Starlark or .bazelrc `source`."""
    rules = [
        (_LABEL, "label"),
        (_DEFINE_FLAG, "define"),
        (_LINK_FLAG, "link"),
        (_LIBRARY_FILE, "link"),
    ]
    if kind == STARLARK:
        rules.append((_DEFINE_STRING, "define"))
    findings = []
    for number, line in enumerate(source.split("\n"), start=1):
        code = strip_hash_comment(line)
        for pattern, finding_kind in rules:
            for match in pattern.finditer(code):
                findings.append(
                    (number, match.start() + 1, finding_kind, match.group(0))
                )
    return findings


# ----------------------------------------------------------------------------------------------------------------------
# Exemptions
# ----------------------------------------------------------------------------------------------------------------------
_MARKER = re.compile(r"\bNOLINT(NEXTLINE|BEGIN|END)?\(boost\)([^\n]*)")


def _has_reason(rest: str) -> bool:
    """True when the comment text after a NOLINT marker gives a reason, that is any word before the comment ends."""
    return re.search(r"\w", rest.split("*/", 1)[0]) is not None


def _exempt_lines(lines: List[str]) -> Tuple[Set[int], List[Tuple[int, int, str, str]]]:
    """The line numbers the NOLINT(boost) markers in `lines` exempt, and the markers that are themselves wrong."""
    exempt: Set[int] = set()
    problems = []
    open_blocks: List[Tuple[int, int, bool]] = []  # line, column, has a reason
    for number, line in enumerate(lines, start=1):
        for marker in _MARKER.finditer(line):
            variant = marker.group(1)
            reason = _has_reason(marker.group(2))
            if variant is None:
                if reason:
                    exempt.add(number)
            elif variant == "NEXTLINE":
                if reason:
                    exempt.add(number + 1)
            elif variant == "BEGIN":
                open_blocks.append((number, marker.start() + 1, reason))
            elif open_blocks:
                begin, _, begin_reason = open_blocks.pop()
                if begin_reason:
                    exempt.update(range(begin, number + 1))
            else:
                problems.append(
                    (number, marker.start() + 1, "unopened", "NOLINTEND(boost)")
                )
    for begin, column, _ in open_blocks:
        problems.append((begin, column, "unclosed", "NOLINTBEGIN(boost)"))
    return exempt, problems


def _raw_findings(source: str, kind: str) -> List[Tuple[int, int, str, str]]:
    """The (line, column, kind, text) of every Boost use in `source`, a file of the given kind, before any marker."""
    if kind == CPP:
        without_comments, code = mask_cpp(source)
        return _check_cpp(source, without_comments, code)
    if kind in (STARLARK, BAZELRC):
        return _check_bazel(source, kind)
    raise ValueError(f"not a kind of file this check reads: {kind!r}")


def _comment_lines(source: str, kind: str) -> List[str]:
    if kind == CPP:
        without_comments, _ = mask_cpp(source)
        return cpp_source.comment_text(source, without_comments).split("\n")
    return [nolint.hash_comment_text(line) for line in source.split("\n")]


def check_source(source: str, kind: str, path: str = "<source>") -> List[Violation]:
    """Every Boost use in `source`, a file of the given kind (CPP, STARLARK or BAZELRC), that no marker exempts.

    The markers are read from the comments alone: one in a string literal is text, not a justification.
    """
    findings = _raw_findings(source, kind)
    exempt, problems = _exempt_lines(_comment_lines(source, kind))
    findings = [f for f in findings if f[0] not in exempt] + problems
    return [
        Violation(path, line, column, finding_kind, text)
        for line, column, finding_kind, text in sorted(findings)
    ]


def check_files(paths: Iterable[str], root: str) -> List[Violation]:
    """Checks every file of `paths` this check reads, by its path relative to `root`; the others are skipped."""
    violations: List[Violation] = []
    for path in paths:
        relative = os.path.relpath(path, root)
        kind = kind_of(relative)
        if kind is None:
            continue
        with open(path, encoding="utf-8", errors="ignore") as f:
            violations += check_source(f.read(), kind, relative)
    return violations


def repository_files(root: str) -> List[str]:
    """The absolute paths of the files in the repository at `root` that this check reads.

    These are the files git tracks or would track, which leaves out Bazel's output trees and git-ignored files. When git
    cannot list them, the tree is walked instead, without .git and the top-level build output directories.
    """
    return [
        os.path.join(root, path)
        for path in lint_files.repository_files(root)
        if kind_of(path)
    ]


def check_staged(repository: str) -> List[Violation]:
    """Checks the STAGED version of every file of a checked kind staged for commit - the index, not the working tree.

    This is what the pre-commit hook runs: it judges exactly what is about to be committed, a partial `git add -p`
    included, and only the files the commit touches, so a commit is never blocked by a file it does not change.
    """
    violations: List[Violation] = []
    for path in lint_files.staged_files(repository):
        kind = kind_of(path)
        if kind is not None:
            violations += check_source(
                lint_files.staged_source(repository, path), kind, path
            )
    return violations


_LANGUAGE_KINDS = {
    check_types.Language.CPP: CPP,
    check_types.Language.STARLARK: STARLARK,
    check_types.Language.BAZELRC: BAZELRC,
}


def _findings(source: str, path: str) -> List[check_types.Finding]:
    kind = kind_of(path)
    if kind is None:
        return []
    return [
        check_types.Finding(
            path,
            line,
            column,
            NAME,
            _MESSAGES[finding_kind][0].replace("{text}", text)
            + _MESSAGES[finding_kind][1],
        )
        for line, column, finding_kind, text in sorted(_raw_findings(source, kind))
    ]


CHECKS = [
    check_types.Check(
        name=NAME,
        languages=frozenset(_LANGUAGE_KINDS),
        scope=lint_files.Scope.NOT_THIRDPARTY,
        check_source=_findings,
        description="the project is Boost-free: only the @pinocchio target depends on Boost (AGENTS.md).",
        hint="Use the standard library, Abseil or ocs2::PropertyTree; a justified exception says why, with "
        "NOLINT(boost): <reason> on its line.",
    )
]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths", nargs="*", help="C++ and Bazel files to check (others are skipped)"
    )
    parser.add_argument(
        "--git-staged",
        action="store_true",
        help="check the staged version of every C++ and Bazel file staged for commit (the pre-commit hook)",
    )
    args = parser.parse_args(argv)
    if args.git_staged == bool(args.paths):
        parser.error("give either file paths or --git-staged")
    if args.git_staged:
        violations = check_staged(os.getcwd())
    else:
        violations = check_files(args.paths, os.getcwd())
    for violation in violations:
        print(violation)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
