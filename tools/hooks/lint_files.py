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

"""Which files the repository's lint checks read: the one list of vendored, generated and fixture files.

Every check and formatter in tools/hooks takes its file set from here, so that a vendored directory is skipped
everywhere or nowhere:

- `repository_files()` lists what git tracks or would track (ignored files and Bazel's output trees are never ours);
- `VENDORED_DIRS` is third-party code, left as it comes (AGENTS.md);
- `GENERATED_FILES` are generated sources that live in the tree; their generator is held to the rules instead;
- `FIXTURE_DIRS` hold deliberately bad code for the tools' own tests: skipped by every style check, not by the spelling
  check;
- the scopes (`in_scope()`) name the file sets a check covers, `FIRST_PARTY` being the usual one.

It also derives what the Python tools need from the BUILD files: the Bazel import roots (`python_import_roots()`) that
pylint and mypy resolve first-party modules against, and the first-party top-level packages isort groups.
"""

import enum
import os
import re
import subprocess

# Third-party code the repository vendors. lib/ocs2 is this project's own fork of OCS2: the nullability rules and the
# Boost check cover it (FIRST_PARTY_AND_OCS2), the style checks do not.
# LINT.IfChange(vendored_dirs)
VENDORED_DIRS = ("lib/", "tools/ifttt-lint/")
# LINT.ThenChange(//tools/clang_tidy/clang_tidy.bzl:excluded_packages)
# The CppAD, CppADCodeGen and iit copies inside the OCS2 fork, which no rule of this repository covers.
OCS2_DIR = "lib/ocs2/"
OCS2_THIRDPARTY_DIR = "lib/ocs2/thirdparty/"

# Generated sources checked into the tree: the ACoM SIREN weight headers (humanoid_learning/acom writes them).
# LINT.IfChange(generated_files)
GENERATED_FILES = frozenset(
    {
        "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/acom/AcomSirenWeightsAtlas.h",
        "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/acom/AcomSirenWeightsG1.h",
        "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/acom/AcomSirenWeightsSa01.h",
    }
)
# LINT.ThenChange(//tools/clang_tidy/clang_tidy.bzl:generated_files)

# Deliberately bad code that the tools' own tests run on (the clang-tidy aspect's self-test fixtures).
# LINT.IfChange(fixture_dirs)
FIXTURE_DIRS = ("tools/clang_tidy/testdata/",)
# LINT.ThenChange(//tools/clang_tidy/clang_tidy.bzl:excluded_packages)

# Top-level directories that hold build output rather than sources, for when git cannot list the files.
_OUTPUT_DIRS = frozenset(
    {".git", ".bazel", ".bazel_ros_install", "build", "install", "log"}
)

CPP_EXTENSIONS = (
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".inl",
    ".ipp",
    ".tpp",
)
CPP_HEADER_EXTENSIONS = (".h", ".hh", ".hpp", ".hxx", ".inl", ".ipp", ".tpp")


class Scope(enum.Enum):
    """The file set a check covers, by path relative to the repository root."""

    # Everything except vendored code, generated files and fixtures.
    FIRST_PARTY = "first_party"
    # FIRST_PARTY without tests (is_test_path()).
    FIRST_PARTY_NON_TEST = "first_party_non_test"
    # FIRST_PARTY plus the OCS2 fork in lib/ocs2, without its lib/ocs2/thirdparty copies: the nullability scope.
    FIRST_PARTY_AND_OCS2 = "first_party_and_ocs2"
    # Every file except vendored code: the spelling and inclusive-language checks, which read fixtures and generated
    # files too.
    TEXT = "text"
    # The whole repository except lib/ocs2/thirdparty and tools/ifttt-lint: the Boost check, which also reads the BUILD
    # files of lib/.
    NOT_THIRDPARTY = "not_thirdparty"


def normalize(path: str) -> str:
    """`path` with forward slashes and without a leading `./`."""
    path = path.replace(os.sep, "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def is_vendored(path: str) -> bool:
    """True for a path under one of the VENDORED_DIRS."""
    return normalize(path).startswith(VENDORED_DIRS)


def is_generated(path: str) -> bool:
    """True for one of the GENERATED_FILES."""
    return normalize(path) in GENERATED_FILES


def is_fixture(path: str) -> bool:
    """True for a path under one of the FIXTURE_DIRS."""
    return normalize(path).startswith(FIXTURE_DIRS)


def is_ocs2(path: str) -> bool:
    """True for the OCS2 fork in lib/ocs2, not counting its lib/ocs2/thirdparty copies."""
    path = normalize(path)
    return path.startswith(OCS2_DIR) and not path.startswith(OCS2_THIRDPARTY_DIR)


def is_first_party(path: str) -> bool:
    """True for code this repository owns and styles: not vendored, generated or a fixture."""
    return not (is_vendored(path) or is_generated(path) or is_fixture(path))


_TEST_DIRECTORIES = frozenset({"test", "tests", "testdata", "testing"})
_TEST_BASENAME = re.compile(
    r"^(test[A-Z_.]|conftest\.py$)|_test\.\w+$|Tests?\.\w+$|_test_support\.\w+$"
)


def is_test_path(path: str) -> bool:
    """True for a test or test support file.

    That is a file in a `test`, `tests`, `testdata` or `testing` directory or in a package directory ending in `_test`
    (humanoid_centroidal_mpc_test), or one named like a test: testFoo.cpp, test_foo.py, FooTest.cpp, foo_test.py,
    conftest.py.

    Args:
      path: The path, relative to the repository root.

    Returns:
      Whether the file is test code.
    """
    parts = normalize(path).split("/")
    if any(part in _TEST_DIRECTORIES or part.endswith("_test") for part in parts[:-1]):
        return True
    return _TEST_BASENAME.search(parts[-1]) is not None


def in_scope(path: str, scope: Scope) -> bool:
    """True when `path` (relative to the repository root) belongs to `scope`."""
    if scope == Scope.FIRST_PARTY:
        return is_first_party(path)
    if scope == Scope.FIRST_PARTY_NON_TEST:
        return is_first_party(path) and not is_test_path(path)
    if scope == Scope.FIRST_PARTY_AND_OCS2:
        return (is_first_party(path) or is_ocs2(path)) and not (
            is_generated(path) or is_fixture(path)
        )
    if scope == Scope.TEXT:
        return not is_vendored(path)
    if scope == Scope.NOT_THIRDPARTY:
        path = normalize(path)
        return not path.startswith((OCS2_THIRDPARTY_DIR, "tools/ifttt-lint/"))
    raise ValueError(f"unknown scope {scope!r}")


def _git(repository: str, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", repository, *args], capture_output=True, text=True, check=True
    ).stdout


def repository_files(root: str) -> list[str]:
    """The files of the repository at `root`, relative to it and sorted: what git tracks or would track.

    That leaves out Bazel's output trees and git-ignored files. Files git still lists but that were deleted in the
    working tree, and submodules, are left out too. When git cannot list the files, the tree is walked instead, without
    .git and the top-level build output directories.

    Args:
      root: The repository root.

    Returns:
      The relative paths, with forward slashes.
    """
    try:
        listed = _git(
            root, "ls-files", "--cached", "--others", "--exclude-standard", "-z"
        ).split("\0")
    except (OSError, subprocess.CalledProcessError):
        listed = []
        for directory, subdirectories, files in os.walk(root):
            if directory == root:
                subdirectories[:] = [
                    d
                    for d in subdirectories
                    if d not in _OUTPUT_DIRS and not d.startswith("bazel-")
                ]
            for name in files:
                listed.append(os.path.relpath(os.path.join(directory, name), root))
    return sorted(
        {
            normalize(path)
            for path in listed
            if path and os.path.isfile(os.path.join(root, path))
        }
    )


def staged_files(repository: str) -> list[str]:
    """The paths staged for commit (added, copied, modified or renamed; deletions are not), relative to the root."""
    staged = _git(
        repository, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"
    ).split("\0")
    return [normalize(path) for path in staged if path]


def staged_source(repository: str, path: str) -> str:
    """The STAGED content of `path` - the index, not the working tree - which is what a commit records."""
    return _git(repository, "show", ":" + path)


_IMPORTS_ATTRIBUTE = re.compile(r"\bimports\s*=\s*\[([^\]]*)\]")
_STRING = re.compile(r"[\"']([^\"']*)[\"']")


def python_import_roots(root: str) -> list[str]:
    """The Bazel import roots of the repository's Python code, relative to `root`.

    These are the directories every `imports = [...]` attribute of a BUILD file names, joined to its package. Bazel
    puts them on the path of every target that depends on the library, so pylint and mypy resolve first-party modules
    against the same directories. Found with a regular expression, so a computed `imports` is not seen.

    Args:
      root: The repository root.

    Returns:
      The roots, sorted, with forward slashes; the repository root itself is not among them.
    """
    roots = set()
    for path in repository_files(root):
        name = path.rsplit("/", 1)[-1]
        if name not in ("BUILD", "BUILD.bazel") or not is_first_party(path):
            continue
        package = path.rsplit("/", 1)[0] if "/" in path else ""
        with open(os.path.join(root, path), encoding="utf-8", errors="ignore") as f:
            source = f.read()
        for attribute in _IMPORTS_ATTRIBUTE.finditer(source):
            for entry in _STRING.finditer(attribute.group(1)):
                joined = os.path.normpath(os.path.join(package, entry.group(1)))
                if joined != ".":
                    roots.add(normalize(joined))
    return sorted(roots)


def python_first_party_packages(root: str) -> list[str]:
    """The first-party top-level Python packages and modules: what isort groups as first party.

    These are the packages and modules directly under each Bazel import root, plus `humanoid_learning` and `tools`,
    which are imported from the repository root.

    Args:
      root: The repository root.

    Returns:
      The sorted names.
    """
    names = {"humanoid_learning", "tools"}
    for import_root in python_import_roots(root):
        directory = os.path.join(root, import_root)
        if not os.path.isdir(directory):
            continue
        for entry in os.listdir(directory):
            full = os.path.join(directory, entry)
            if entry.endswith(".py") and os.path.isfile(full):
                names.add(entry[: -len(".py")])
            elif (
                os.path.isdir(full)
                and entry.isidentifier()
                and any(name.endswith(".py") for name in os.listdir(full))
            ):
                names.add(entry)
    return sorted(names)


def isort_package_arguments(root: str) -> list[str]:
    """isort's `-p <package>` arguments for the first-party packages (python_first_party_packages())."""
    arguments = []
    for package in python_first_party_packages(root):
        arguments += ["-p", package]
    return arguments


def cpp_top_level_dirs(root: str) -> list[str]:
    """The top-level directories that hold first-party C++ (the clang-tidy header filter must cover each)."""
    directories = set()
    for path in repository_files(root):
        if "/" in path and path.endswith(CPP_EXTENSIONS) and is_first_party(path):
            directories.add(path.split("/", 1)[0])
    return sorted(directories)


def bazel_package(path: str, root: str) -> str:
    """The Bazel package that holds `path`: the nearest directory with a BUILD file, as `//pkg` (`//` at the root)."""
    directory = os.path.dirname(normalize(path))
    while directory:
        if os.path.isfile(
            os.path.join(root, directory, "BUILD.bazel")
        ) or os.path.isfile(os.path.join(root, directory, "BUILD")):
            return "//" + directory
        directory = os.path.dirname(directory)
    return "//"
