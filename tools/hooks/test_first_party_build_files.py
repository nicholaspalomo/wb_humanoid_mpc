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

"""First-party BUILD files: headers at their source paths, and the shared warning flags on every C++ target.

Headers: no `strip_include_prefix`, no `include_prefix`. A .cpp action of the clang-tidy aspect shows the diagnostics of
the first-party headers it includes by their path (tools/clang_tidy/clang_tidy.bzl:header_filter), among them a
rename's declaration and a member a constructor in the .cpp initializes. A header that a target re-roots with
`strip_include_prefix` or `include_prefix` is reached under bazel-out/.../_virtual_includes/ instead, which the filter
does not match, so those diagnostics would be lost. List the directory in `includes = [...]` instead, which keeps the
source path.

Warnings (AGENTS.md, C++ "Warnings"): every cc_library, cc_binary and cc_test of a first-party BUILD file compiles with
FIRST_PARTY_COPTS from bazel/copts.bzl, directly or through a module-level list built from it, and no BUILD or .bzl
file spells a warning flag of its own. Two forms are allowed: `-Werror`, which only makes the shared warnings errors,
and a `-Wno-...` that names its reason in a comment on its line, for a third-party header that one target includes.
Neither form may switch off what the shared list makes an error: `-w` (no warnings at all), `-fpermissive` (errors
demoted to warnings), a bare `-Wno-error`, and `-Wno-error=<flag>` or `-Wno-<flag>` for a `-Werror=<flag>` of
FIRST_PARTY_COPTS are refused with or without a reason.
The flags themselves live in bazel/copts.bzl; tools/clang_tidy/clang_tidy.bzl names the clang warnings behind the
`clang-diagnostic-*` checks of .clang-tidy, which no compiler sees.
"""

import ast
import os
import re
import unittest

from tools.hooks import check_test_support
from tools.hooks import lint_files

_ATTRIBUTE = re.compile(r"^\s*(strip_include_prefix|include_prefix)\s*=")
_BUILD_FILES = ("BUILD", "BUILD.bazel")

SHARED_COPTS = "FIRST_PARTY_COPTS"
_CC_RULES = frozenset({"cc_binary", "cc_library", "cc_test"})

# The files that define warning flags rather than add them to a target (the module docstring).
WARNING_FLAG_DEFINITIONS = frozenset(
    {"bazel/copts.bzl", "tools/clang_tidy/clang_tidy.bzl"}
)
# -Wl, -Wa, and -Wp, pass options through to the linker, the assembler and the preprocessor: they are not warnings.
_WARNING_FLAG = re.compile(r"^-W(?![alp],)")
_REASON = re.compile(r"#\s*\S")
# Flags that switch off every warning or demote errors: never allowed, with or without a reason.
_DISABLING_FLAGS = frozenset({"-w", "-fpermissive", "-Wno-error"})
_SHARED_ERROR = re.compile(r'^\s*"-Werror=(?P<flag>[\w+-]+)"', re.MULTILINE)
_DEMOTION = re.compile(r"^-Wno-(?:error=)?(?P<flag>[\w+-]+)$")


def shared_errors(copts_source: str) -> frozenset[str]:
    """The warnings that FIRST_PARTY_COPTS makes errors (`-Werror=<flag>`), read from bazel/copts.bzl's source."""
    return frozenset(
        match.group("flag") for match in _SHARED_ERROR.finditer(copts_source)
    )


def prefix_attributes(source: str) -> list[tuple[int, str]]:
    """The `strip_include_prefix` and `include_prefix` attributes of a BUILD file, outside comments.

    Args:
      source: The BUILD file.

    Returns:
      (line number, attribute name) for each.
    """
    found = []
    for number, line in enumerate(source.splitlines(), start=1):
        match = _ATTRIBUTE.match(line.split("#", 1)[0])
        if match:
            found.append((number, match.group(1)))
    return found


def _mentions_shared_copts(
    expression: ast.expr, assignments: dict[str, ast.expr], seen: frozenset[str]
) -> bool:
    """True when `expression` uses SHARED_COPTS, itself or through the module-level names it reads."""
    for node in ast.walk(expression):
        if not isinstance(node, ast.Name):
            continue
        if node.id == SHARED_COPTS:
            return True
        if node.id in assignments and node.id not in seen:
            if _mentions_shared_copts(
                assignments[node.id], assignments, seen | {node.id}
            ):
                return True
    return False


def targets_without_shared_copts(source: str) -> list[tuple[int, str]]:
    """The C++ targets of a BUILD file whose `copts` do not include SHARED_COPTS.

    Args:
      source: The BUILD file, which Python's parser reads as it reads Starlark.

    Returns:
      (line number, target name) for each.
    """
    tree = ast.parse(source)
    assignments: dict[str, ast.expr] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = statement.value
    found = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _CC_RULES
        ):
            continue
        arguments = {keyword.arg: keyword.value for keyword in node.keywords}
        name = arguments.get("name")
        label = name.value if isinstance(name, ast.Constant) else "<unnamed>"
        copts = arguments.get("copts")
        if copts is None or not _mentions_shared_copts(copts, assignments, frozenset()):
            found.append((node.lineno, str(label)))
    return found


def warning_flag_problems(
    source: str, errors: frozenset[str] = frozenset()
) -> list[tuple[int, str]]:
    """The warning flags a BUILD or .bzl file spells that it may not (the module docstring).

    Args:
      source: The file.
      errors: The warnings FIRST_PARTY_COPTS makes errors (shared_errors()), which no target may demote or switch off.

    Returns:
      (line number, flag) for each string literal that is a warning flag other than `-Werror` or a `-Wno-...` with a
      reason comment after it on its line, and for each flag that switches warnings off or demotes a shared error.
    """
    lines = source.splitlines()
    found = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if node.value in _DISABLING_FLAGS:
            found.append((node.lineno, node.value))
            continue
        if not _WARNING_FLAG.match(node.value):
            continue
        if node.value == "-Werror":
            continue
        demotion = _DEMOTION.match(node.value)
        if demotion and demotion.group("flag") in errors:
            found.append((node.lineno, node.value))
            continue
        rest = (
            lines[node.lineno - 1][node.end_col_offset :]
            if node.end_lineno == node.lineno
            else ""
        )
        if node.value.startswith("-Wno-") and _REASON.search(rest):
            continue
        found.append((node.lineno, node.value))
    return sorted(found)


def first_party_build_files(root: str) -> list[str]:
    """The first-party BUILD files of the checkout at `root` (lint_files.is_first_party), relative to it."""
    return [
        path
        for path in lint_files.repository_files(root)
        if os.path.basename(path) in _BUILD_FILES and lint_files.is_first_party(path)
    ]


def first_party_bzl_files(root: str) -> list[str]:
    """The first-party .bzl files of the checkout at `root`, relative to it."""
    return [
        path
        for path in lint_files.repository_files(root)
        if path.endswith(".bzl") and lint_files.is_first_party(path)
    ]


class PrefixAttributesTest(unittest.TestCase):
    def test_both_attributes_are_found_and_comments_are_not(self):
        source = 'cc_library(\n    strip_include_prefix = "test",\n    include_prefix = "x",\n    # include_prefix = "y",\n)\n'
        self.assertEqual(
            prefix_attributes(source),
            [(2, "strip_include_prefix"), (3, "include_prefix")],
        )
        self.assertEqual(
            prefix_attributes('cc_library(\n    includes = ["test"],\n)\n'), []
        )


class SharedCoptsTest(unittest.TestCase):
    def test_the_shared_list_directly_or_extended(self):
        source = (
            'cc_library(name = "a", copts = FIRST_PARTY_COPTS)\n'
            'cc_test(name = "b", copts = FIRST_PARTY_COPTS + ["-Werror"])\n'
            'cc_binary(name = "c", copts = select({"//x": FIRST_PARTY_COPTS, "//conditions:default": FIRST_PARTY_COPTS}))\n'
        )
        self.assertEqual(targets_without_shared_copts(source), [])

    def test_through_a_module_level_list(self):
        source = (
            '_BASE = FIRST_PARTY_COPTS\n_EXTRA = _BASE + ["-Wno-error=foo"]  # third_party/foo.h\n'
            'cc_library(name = "a", copts = _EXTRA)\n'
        )
        self.assertEqual(targets_without_shared_copts(source), [])

    def test_a_target_without_them(self):
        source = (
            '_LOCAL = ["-Wall", "-Wextra"]\n'
            'cc_library(name = "a")\n'
            'cc_test(\n    name = "b",\n    copts = _LOCAL,\n)\n'
            'cc_binary(name = "c", copts = ["-O2"])\n'
            'py_library(name = "d")\n'
        )
        self.assertEqual(
            targets_without_shared_copts(source), [(2, "a"), (3, "b"), (7, "c")]
        )


class WarningFlagsTest(unittest.TestCase):
    def test_a_target_may_not_add_a_warning(self):
        source = 'X = FIRST_PARTY_COPTS + [\n    "-Wall",\n    "-Werror=shadow",  # stricter\n    "-Wshadow",\n]\n'
        self.assertEqual(
            warning_flag_problems(source),
            [(2, "-Wall"), (3, "-Werror=shadow"), (4, "-Wshadow")],
        )

    def test_werror_and_a_reasoned_wno_are_allowed(self):
        source = (
            'X = FIRST_PARTY_COPTS + [\n    "-Werror",\n'
            '    "-Wno-error=implicit-fallthrough",  # cppad/local/optimize/*.hpp (third-party)\n]\n'
        )
        self.assertEqual(warning_flag_problems(source), [])

    def test_a_wno_needs_its_reason_on_its_line(self):
        source = 'X = [\n    # third_party/foo.h\n    "-Wno-unused",\n    "-Wno-shadow",  #\n]\n'
        self.assertEqual(
            warning_flag_problems(source), [(3, "-Wno-unused"), (4, "-Wno-shadow")]
        )

    def test_pass_through_options_and_comments_are_not_warnings(self):
        source = '# copts = ["-Wall"]\nX = ["-Wl,-rpath,/opt", "-Wa,--noexecstack", "-Wp,-DX"]\n'
        self.assertEqual(warning_flag_problems(source), [])

    def test_switching_warnings_off_is_refused_even_with_a_reason(self):
        source = (
            'cc_library(name = "a", copts = FIRST_PARTY_COPTS + [\n'
            '    "-w",  # third-party header\n'
            '    "-Wno-error",  # third-party header\n'
            '    "-fpermissive",  # third-party header\n'
            '    "-Wno-error=return-type",  # third-party header\n'
            '    "-Wno-switch",  # third-party header\n'
            '    "-Wno-error=unused-parameter",  # third-party header\n'
            "])\n"
        )
        self.assertEqual(
            warning_flag_problems(source, frozenset({"return-type", "switch"})),
            [
                (2, "-w"),
                (3, "-Wno-error"),
                (4, "-fpermissive"),
                (5, "-Wno-error=return-type"),
                (6, "-Wno-switch"),
            ],
        )
        self.assertEqual(targets_without_shared_copts(source), [])

    def test_the_shared_errors_are_read_from_copts_bzl(self):
        errors = shared_errors(
            check_test_support.read_repository_file("bazel/copts.bzl")
        )
        self.assertIn("return-type", errors)
        self.assertIn("implicit-fallthrough", errors)
        self.assertNotIn("maybe-uninitialized", errors)


class FirstPartyBuildFilesTest(unittest.TestCase):
    def setUp(self):
        self.root = check_test_support.source_tree_root()
        self.files = first_party_build_files(self.root)

    def _read(self, path: str) -> str:
        with open(os.path.join(self.root, path), encoding="utf-8") as f:
            return f.read()

    def test_the_tree_is_found(self):
        self.assertIn("tools/hooks/BUILD.bazel", self.files)
        self.assertNotIn("lib/ocs2/BUILD.bazel", self.files)
        self.assertNotIn("tools/clang_tidy/testdata/BUILD.bazel", self.files)

    def test_no_first_party_target_reroots_its_headers(self):
        for path in self.files:
            with self.subTest(path=path):
                found = prefix_attributes(self._read(path))
                self.assertEqual(
                    found,
                    [],
                    f"{path}: use `includes = [...]` (this test's docstring says why)",
                )

    def test_every_first_party_cpp_target_compiles_with_the_shared_copts(self):
        targets = 0
        for path in self.files:
            source = self._read(path)
            targets += source.count("cc_library(") + source.count("cc_test(")
            with self.subTest(path=path):
                self.assertEqual(
                    targets_without_shared_copts(source),
                    [],
                    f"{path}: give each C++ target `copts = {SHARED_COPTS}` (bazel/copts.bzl)",
                )
        # A parser that silently found no target would make the test pass.
        self.assertGreater(targets, 100)

    def test_no_build_or_bzl_file_adds_a_warning_flag(self):
        files = self.files + first_party_bzl_files(self.root)
        errors = shared_errors(self._read("bazel/copts.bzl"))
        self.assertGreater(len(errors), 5, "bazel/copts.bzl lists its -Werror= flags")
        for definition in sorted(WARNING_FLAG_DEFINITIONS):
            self.assertIn(definition, files, "a stale exemption")
        for path in files:
            if path in WARNING_FLAG_DEFINITIONS:
                continue
            with self.subTest(path=path):
                self.assertEqual(
                    warning_flag_problems(self._read(path), errors),
                    [],
                    f"{path}: add the flag to bazel/copts.bzl, or give a -Wno-... its reason on its line "
                    "(this test's docstring)",
                )


if __name__ == "__main__":
    unittest.main()
