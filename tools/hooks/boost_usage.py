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

"""Flags Boost in the project's C++ and Bazel files, so that it cannot creep back.

The project does not use Boost: the configuration is typed textprotos, the integrators are native ports of the odeint
loops, and the lib/ocs2 fork is Boost-free. Boost stays installed only because Pinocchio's public headers include it,
so the @pinocchio target in bazel/system_libs.bzl is the one target that depends on @boost and the one that carries
Boost's configuration defines. This check flags, in code only:

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

A justified exception is marked where it is, with its reason, in a comment, as the @pinocchio target marks its own; the
registry engine applies the markers (tools/hooks/checks.py, tools/hooks/nolint.py):
  - `NOLINT(boost): <reason>` in a comment exempts its own line;
  - `NOLINTNEXTLINE(boost): <reason>` exempts the line after it;
  - `NOLINTBEGIN(boost): <reason>` ... `NOLINTEND(boost)` exempts the lines from the one to the other.
"""

import bisect
import os
import re

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
    " The project does not use Boost: use the standard library or Abseil. A justified exception "
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
}


def kind_of(path: str) -> str | None:
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


def mask_cpp(source: str) -> tuple[str, str]:
    """`source` with its comments blanked out, and with its comments and its string and character literals blanked out.

    Both have the length and the line structure of `source`, so an offset into either is an offset into `source`.
    """
    return cpp_source.mask(source)


def _continues(code_line: str) -> bool:
    """True when the code of a preprocessor line ends with the backslash that continues the directive on the next."""
    return code_line.rstrip(" \t\r").endswith("\\")


def _check_cpp(without_comments: str, code: str) -> list[tuple[int, int, str, str]]:
    """The (line, column, kind, text) of every Boost token of a C++ file, given its masks (mask_cpp())."""
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


def _check_bazel(source: str, kind: str) -> list[tuple[int, int, str, str]]:
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


def _raw_findings(source: str, kind: str) -> list[tuple[int, int, str, str]]:
    """The (line, column, kind, text) of every Boost use in `source`, a file of the given kind, before any marker."""
    if kind == CPP:
        without_comments, code = mask_cpp(source)
        return _check_cpp(without_comments, code)
    if kind in (STARLARK, BAZELRC):
        return _check_bazel(source, kind)
    raise ValueError(f"not a kind of file this check reads: {kind!r}")


_LANGUAGE_KINDS = {
    check_types.Language.CPP: CPP,
    check_types.Language.STARLARK: STARLARK,
    check_types.Language.BAZELRC: BAZELRC,
}


def _findings(source: str, path: str) -> list[check_types.Finding]:
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
        hint="Use the standard library or Abseil; a justified exception says why, with "
        "NOLINT(boost): <reason> on its line.",
    )
]
