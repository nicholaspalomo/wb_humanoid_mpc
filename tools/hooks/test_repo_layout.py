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

"""Checks that every repository path the checks and the build scope by names a directory or file that exists.

A moved directory leaves a stale prefix wherever a path is spelled out, and a stale prefix fails silently: a check whose
scope names a directory that is gone reads nothing and reports nothing. This test reads, in the checkout:

- every path of repo_layout.all_scoped_paths();
- every string constant of the non-test modules of tools/hooks that looks like a repository path (`_is_path_like()`),
  with `+` chains of constants and repo_layout attributes evaluated first;
- every `//<package>/...` pattern of the CI workflows;
- the directories of clang-tidy's FIRST_PARTY_HEADER_FILTER and its GENERATED_FILES (tools/clang_tidy/clang_tidy.bzl);

and fails on each one that is missing. A path ending in `/` must be a directory, any other a file.
"""

import ast
import glob
import os
import re
import unittest

from tools.hooks import check_test_support
from tools.hooks import repo_layout

# The suffixes of the file paths the checks name.
_PATH_SUFFIXES = (".h", ".cpp", ".py", ".md", ".proto", ".textproto")

# A component of a repository path: a file or directory name with a letter or digit in it (not `*/` or `//`).
_PATH_COMPONENT = re.compile(r"[A-Za-z0-9_.+-]*[A-Za-z0-9][A-Za-z0-9_.+-]*")

# String constants of tools/hooks that look like repository paths but are not, each with why.
_NOT_REPOSITORY_PATHS = frozenset(
    {
        # include_style.py: the prefix of an Abseil include ("absl/..."), not a directory of this repository.
        "absl/",
        # pointer_nullability.py: the Abseil header the annotations come from, as an #include spells it.
        "absl/base/nullability.h",
        # proto_file_layout.py: the nproto options as a .proto imports them, relative to tools/ (tools/nproto).
        "nproto/options.proto",
    }
)

# LINT.IfChange(repo_layout_data)
# The CI workflows whose Bazel target patterns name packages.
WORKFLOWS = (
    ".github/workflows/build_test.yml",
    ".github/workflows/robot_arm64.yml",
)
CLANG_TIDY_BZL = "tools/clang_tidy/clang_tidy.bzl"
# LINT.ThenChange(//tools/hooks/BUILD.bazel:repo_layout_data)

# A Bazel target pattern below a package (`//pkg/...`, `-//pkg/...`); `//...` names the whole tree and no package.
_PACKAGE_PATTERN = re.compile(
    r"(?<!\S)-?//([A-Za-z0-9_][A-Za-z0-9_./-]*?)/\.\.\.(?!\S)"
)
# FIRST_PARTY_HEADER_FILTER's form: "^(\./)?(<dir>|<dir>|...)/".
_HEADER_FILTER = re.compile(r"\^\(\\\./\)\?\((?P<dirs>[A-Za-z0-9_|]+)\)/")


def _missing(root: str, paths: list[str] | tuple[str, ...]) -> list[str]:
    """Returns the paths that do not exist under `root`: a directory for a trailing `/`, a file otherwise."""
    missing = []
    for path in paths:
        full = os.path.join(root, path)
        exists = (
            os.path.isdir(full) if path.endswith("/") else os.path.isfile(full)
        )
        if not exists:
            missing.append(path)
    return missing


def _is_path_like(text: str) -> bool:
    """True for a relative path of name components, ending in `/` or in a suffix of _PATH_SUFFIXES."""
    if "/" not in text or not text.endswith(("/",) + _PATH_SUFFIXES):
        return False
    components = text.removesuffix("/").split("/")
    return all(_PATH_COMPONENT.fullmatch(component) for component in components)


def _value(node: ast.expr, names: dict[str, str]) -> str | None:
    """Returns the string `node` evaluates to from constants, `names` and repo_layout attributes, or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "repo_layout"
    ):
        attribute = getattr(repo_layout, node.attr, None)
        return attribute if isinstance(attribute, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _value(node.left, names)
        right = _value(node.right, names)
        if left is not None and right is not None:
            return left + right
    return None


def _module_names(tree: ast.Module) -> dict[str, str]:
    """Returns the module-level names of `tree` assigned a string that _value() can evaluate, in order."""
    names: dict[str, str] = {}
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            value = _value(statement.value, names)
            if value is not None:
                names[statement.targets[0].id] = value
    return names


def _string_values(source: str) -> list[tuple[int, str]]:
    """Returns (line, value) for each string of `source` in source order: `+` chains evaluated, f-string parts skipped."""
    tree = ast.parse(source)
    names = _module_names(tree)
    consumed: set[int] = set()
    values: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if id(node) in consumed:
            continue
        if isinstance(node, ast.JoinedStr):
            consumed.update(id(part) for part in ast.walk(node))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            value = _value(node, names)
            if value is not None:
                values.append((node.lineno, node.col_offset, value))
                consumed.update(id(part) for part in ast.walk(node))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            values.append((node.lineno, node.col_offset, node.value))
    return [(line, value) for line, _, value in sorted(values)]


def _path_literals(source: str) -> list[tuple[int, str]]:
    """Returns (line, path) for each string of `source` that looks like a repository path and is not allowlisted."""
    return [
        (line, value)
        for line, value in _string_values(source)
        if _is_path_like(value) and value not in _NOT_REPOSITORY_PATHS
    ]


def _workflow_packages(workflow: str) -> list[str]:
    """Returns the package of each `//<package>/...` pattern of `workflow`, as a directory path ending in `/`."""
    return [
        f"{match.group(1)}/" for match in _PACKAGE_PATTERN.finditer(workflow)
    ]


def _bzl_strings(source: str) -> dict[str, str | list[str]]:
    """Returns the module-level strings and lists of strings of a Starlark file, which Python's parser reads."""
    strings: dict[str, str | list[str]] = {}
    for statement in ast.parse(source).body:
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            continue
        name = statement.targets[0].id
        value = statement.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            strings[name] = value.value
        elif isinstance(value, ast.List) and all(
            isinstance(element, ast.Constant) and isinstance(element.value, str)
            for element in value.elts
        ):
            strings[name] = [
                element.value
                for element in value.elts
                if isinstance(element, ast.Constant)
                and isinstance(element.value, str)
            ]
    return strings


def _header_filter_dirs(header_filter: str) -> list[str] | None:
    """Returns the directories of a header filter of _HEADER_FILTER's form, ending in `/`, or None for another form."""
    match = _HEADER_FILTER.fullmatch(header_filter)
    if match is None:
        return None
    return [f"{directory}/" for directory in match.group("dirs").split("|")]


class RepoLayoutTest(unittest.TestCase):
    def setUp(self):
        self.root = check_test_support.source_tree_root()

    def _read(self, path: str) -> str:
        with open(os.path.join(self.root, path), encoding="utf-8") as f:
            return f.read()

    def test_every_scoped_path_exists(self):
        self.assertEqual(
            _missing(self.root, repo_layout.all_scoped_paths()), []
        )

    def test_every_constant_of_the_module_is_a_scoped_path(self):
        constants = {
            value
            for name, value in vars(repo_layout).items()
            if name.isupper() and isinstance(value, str) and "/" in value
        }
        self.assertEqual(constants, set(repo_layout.all_scoped_paths()))

    def test_every_path_literal_of_the_checks_exists(self):
        modules = sorted(
            glob.glob(os.path.join(self.root, "tools", "hooks", "*.py"))
        )
        self.assertIn(
            os.path.join(self.root, "tools", "hooks", "euler_boundary.py"),
            modules,
        )
        found = 0
        for module in modules:
            name = os.path.basename(module)
            if name.startswith("test_"):
                continue
            path = f"tools/hooks/{name}"
            literals = _path_literals(self._read(path))
            found += len(literals)
            for line, literal in literals:
                with self.subTest(path=path, line=line):
                    self.assertEqual(_missing(self.root, [literal]), [])
        # The scan reads the scopes of the checks: euler_boundary.FORMULATION_FILES alone has eight paths.
        self.assertGreater(found, 20)

    def test_every_ci_target_pattern_names_a_package(self):
        for workflow in WORKFLOWS:
            packages = _workflow_packages(self._read(workflow))
            self.assertNotEqual(packages, [], workflow)
            with self.subTest(workflow=workflow):
                self.assertEqual(_missing(self.root, packages), [])

    def test_the_clang_tidy_header_filter_and_generated_files_exist(self):
        strings = _bzl_strings(self._read(CLANG_TIDY_BZL))
        header_filter = strings["FIRST_PARTY_HEADER_FILTER"]
        assert isinstance(header_filter, str)
        directories = _header_filter_dirs(header_filter)
        self.assertIsNotNone(directories, header_filter)
        assert directories is not None
        self.assertEqual(_missing(self.root, directories), [])
        generated = strings["GENERATED_FILES"]
        assert isinstance(generated, list)
        self.assertNotEqual(generated, [])
        self.assertEqual(_missing(self.root, generated), [])


class StalePathTest(unittest.TestCase):
    """The helpers report a path that does not exist, so a stale prefix would fail the tests above."""

    def setUp(self):
        self.root = check_test_support.source_tree_root()

    def test_a_missing_directory_or_file_is_reported(self):
        self.assertEqual(
            _missing(
                self.root,
                [
                    "no_such_package/",
                    "no_such_package/Foo.h",
                    "tools/hooks/",
                    "MODULE.bazel",
                ],
            ),
            ["no_such_package/", "no_such_package/Foo.h"],
        )

    def test_a_file_named_as_a_directory_is_reported(self):
        self.assertEqual(
            _missing(self.root, ["MODULE.bazel/"]), ["MODULE.bazel/"]
        )

    def test_a_stale_prefix_in_a_check_module_is_reported(self):
        source = (
            'OLD_DIR = "no_such_package/"\n'
            'FILES = (OLD_DIR + "include/Foo.h", "tools/hooks/checks.py")\n'
            "PREFIX = repo_layout.NMPC_DIR + 'no_such_package/'\n"
        )
        literals = [path for _, path in _path_literals(source)]
        self.assertEqual(
            literals,
            [
                "no_such_package/",
                "no_such_package/include/Foo.h",
                "tools/hooks/checks.py",
                repo_layout.NMPC_DIR + "no_such_package/",
            ],
        )
        self.assertEqual(
            _missing(self.root, literals),
            [
                "no_such_package/",
                "no_such_package/include/Foo.h",
                repo_layout.NMPC_DIR + "no_such_package/",
            ],
        )

    def test_messages_fragments_and_allowlisted_strings_are_not_paths(self):
        source = (
            'MESSAGE = "see no_such_package/README.md for the rule"\n'
            'NAME = f"no_such_package/{name}"\n'
            'HEADER = "absl/"\n'
            'SEPARATOR = "/"\n'
            'COMMENT_END = "*/"\n'
            'ROOT = "/usr/include/"\n'
            'CATEGORY = "build/include"\n'
        )
        self.assertEqual(_path_literals(source), [])

    def test_a_stale_ci_target_pattern_is_reported(self):
        workflow = (
            "targets: //:all //no_such_package/... //tools/...\n"
            "  //... -//no_such_dir/sub/... -//:all\n"
        )
        packages = _workflow_packages(workflow)
        self.assertEqual(
            packages, ["no_such_package/", "tools/", "no_such_dir/sub/"]
        )
        self.assertEqual(
            _missing(self.root, packages),
            ["no_such_package/", "no_such_dir/sub/"],
        )

    def test_a_stale_header_filter_directory_is_reported(self):
        directories = _header_filter_dirs(
            r"^(\./)?(bazel|no_such_package|tools)/"
        )
        self.assertEqual(directories, ["bazel/", "no_such_package/", "tools/"])
        assert directories is not None
        self.assertEqual(_missing(self.root, directories), ["no_such_package/"])
        self.assertIsNone(_header_filter_dirs("^(bazel|tools)/"))


if __name__ == "__main__":
    unittest.main()
