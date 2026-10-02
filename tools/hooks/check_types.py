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

"""The types the lint checks of tools/hooks share: a check, its finding, and the languages it reads.

They live apart from the registry (tools/hooks/checks.py) because every check module uses them and the registry imports
every check module.
"""

from collections.abc import Callable
import dataclasses
import enum
import os

from tools.hooks import lint_files


class Language(enum.Enum):
    """The kind of source a file is, from its name (language_of())."""

    CPP = "C++"
    PYTHON = "Python"
    PROTO = "proto"
    STARLARK = "Starlark"
    BAZELRC = "bazelrc"
    SHELL = "shell"
    # Every text file, whatever else it is: a check that reads TEXT reads them all.
    TEXT = "text"


# The text files: what the whitespace, spelling and inclusive-language checks read.
TEXT_EXTENSIONS = frozenset(
    {
        ".bash",
        ".bazel",
        ".bzl",
        ".c",
        ".cc",
        ".cfg",
        ".cpp",
        ".cxx",
        ".h",
        ".hh",
        ".hpp",
        ".hxx",
        ".ini",
        ".inl",
        ".ipp",
        ".json",
        ".md",
        ".mjcf",
        ".proto",
        ".py",
        ".sh",
        ".textproto",
        ".toml",
        ".tpp",
        ".txt",
        ".urdf",
        ".xacro",
        ".xml",
        ".yaml",
        ".yml",
    }
)
TEXT_FILE_NAMES = frozenset(
    {
        ".bazelignore",
        ".bazelrc",
        ".bazelversion",
        ".clang-format",
        ".clang-tidy",
        ".gitignore",
        ".pylintrc",
        "BUILD",
        "BUILD.bazel",
        "CPPLINT.cfg",
        "MODULE.bazel",
        "Makefile",
        "WORKSPACE",
    }
)
# Robot model files are upstream data: whitespace is checked, wording is not.
MODEL_EXTENSIONS = frozenset({".mjcf", ".urdf", ".xacro", ".xml"})


def is_text(path: str) -> bool:
    """True for a file the text checks read."""
    name = path.replace(os.sep, "/").rsplit("/", 1)[-1]
    return (
        name in TEXT_FILE_NAMES or os.path.splitext(name)[1].lower() in TEXT_EXTENSIONS
    )


def language_of(path: str) -> Language | None:
    """The language of the file at `path` other than TEXT, or None when it is none of them."""
    name = path.replace(os.sep, "/").rsplit("/", 1)[-1]
    if name.endswith(lint_files.CPP_EXTENSIONS):
        return Language.CPP
    if name.endswith(".py"):
        return Language.PYTHON
    if name.endswith(".proto"):
        return Language.PROTO
    if name in ("BUILD", "WORKSPACE") or name.endswith((".bazel", ".bzl", ".BUILD")):
        return Language.STARLARK
    if name.endswith(".bazelrc"):
        return Language.BAZELRC
    if name.endswith((".sh", ".bash")):
        return Language.SHELL
    return None


def languages_of(path: str) -> frozenset[Language]:
    """Every language the file at `path` counts as, TEXT included for a text file."""
    languages = set()
    language = language_of(path)
    if language is not None:
        languages.add(language)
    if is_text(path) or language is not None:
        languages.add(Language.TEXT)
    return frozenset(languages)


@dataclasses.dataclass(frozen=True, order=True)
class Finding:
    """One finding of one check, printed `path:line:column: message [check]`."""

    path: str
    line: int
    column: int
    check: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.message} [{self.check}]"


# check_source(source, path) -> the findings of the check in `source`, the content of the file at `path` (relative to
# the repository root), before any NOLINT marker is applied: tools/hooks/checks.py applies the markers.
CheckFunction = Callable[[str, str], list[Finding]]
# fix_source(source, path) -> `source` with the check's findings fixed.
FixFunction = Callable[[str, str], str]


@dataclasses.dataclass(frozen=True)
class Check:
    """A lint check of the registry (tools/hooks/checks.py).

    Attributes:
      name: The check's name, kebab-case. It is also its NOLINT category, `// NOLINT(<name>): <reason>`, and its name
        for `lint_code --only`.
      languages: The languages it reads. TEXT means every text file.
      scope: The file set it covers.
      check_source: Finds the check's findings in one file.
      description: One line saying what the check enforces, and where the rule comes from.
      fix_source: Rewrites a file so that the check passes, or None when the check cannot fix its findings.
      fixed_by_format: True when `make format` applies fix_source: a rewrite that cannot change behavior.
      hint: What to do about a finding, printed once under the findings of the check.
    """

    name: str
    languages: frozenset[Language]
    scope: lint_files.Scope
    check_source: CheckFunction
    description: str
    fix_source: FixFunction | None = None
    fixed_by_format: bool = False
    hint: str = ""

    def applies_to(self, path: str) -> bool:
        """True when this check reads the file at `path` (relative to the repository root)."""
        if not lint_files.in_scope(path, self.scope):
            return False
        return bool(self.languages & languages_of(path))
