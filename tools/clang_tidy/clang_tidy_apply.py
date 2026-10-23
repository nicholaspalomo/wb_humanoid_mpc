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

"""Applies the fix-its of a clang-tidy build to the source tree (tools/clang_tidy/README.md).

    bazel build --config=clang-tidy-fix --keep_going --build_event_json_file=.bazel/clang_tidy_bep.json //...
    python3 -m tools.clang_tidy.clang_tidy_apply .bazel/clang_tidy_bep.json --checks 'modernize-use-override' \
        [--paths DIR ...]

The fix-its are the `--export-fixes` files of the aspect's actions (output group `clang_tidy_fixes`). A header's
diagnostic comes from the header's own action and from every .cpp that includes it, so replacements are deduplicated by
(file, offset, length, text). A diagnostic - identified by its check, location and message - is applied all or
nothing, so that a rename never lands half-way:

- a replacement outside the source tree (a generated file under bazel-out/, an external repository) is never applied,
  and nor is any other replacement of its diagnostic: rename such an identifier through its generator by hand;
- when one translation unit reported a diagnostic without fix-its (the renamer's "cannot be fixed because it is inside
  a macro"), none of its fix-its is applied anywhere;
- with `--paths`, a diagnostic located in the given directories is applied only when all of its replacements are too,
  so that a work package never edits another's files;
- a diagnostic whose replacements overlap another's is left alone.

Each diagnostic left alone is listed, for a manual fix. The edited files are then formatted with clang-format. The exit
status is 1 when any diagnostic was left alone.
"""

import argparse
import collections
import dataclasses
import fnmatch
import os
import shutil
import subprocess
import sys

import yaml

from tools.clang_tidy import clang_tidy_report

OUTPUT_GROUP = "clang_tidy_fixes"
_OUTSIDE_SOURCE_TREE = ("bazel-out/", "external/")


@dataclasses.dataclass(frozen=True, order=True)
class Replacement:
    """One edit of a file: `length` bytes at byte `offset` become `text`."""

    path: str
    offset: int
    length: int
    text: str

    def overlaps(self, other: "Replacement") -> bool:
        """True when the two edit the same bytes, or insert different text at the same place."""
        if self.path != other.path or self == other:
            return False
        if self.offset == other.offset:
            return True
        return (
            self.offset < other.offset + other.length
            and other.offset < self.offset + self.length
        )


@dataclasses.dataclass(frozen=True)
class Diagnostic:
    """A diagnostic of one translation unit, with the replacements of its fix-its."""

    check: str
    path: str
    offset: int
    message: str
    replacements: tuple[Replacement, ...]

    @property
    def identity(self) -> tuple[str, str, int, str]:
        """What identifies the diagnostic across translation units."""
        return (self.check, self.path, self.offset, self.message)


def parse_fixes(text: str) -> list[Diagnostic]:
    """The diagnostics of one `--export-fixes` file.

    Args:
      text: The YAML clang-tidy wrote; empty for a file without findings.

    Returns:
      The diagnostics, with repository-relative paths.
    """
    document = yaml.safe_load(text) if text.strip() else None
    if not isinstance(document, dict):
        return []
    diagnostics = []
    for entry in document.get("Diagnostics") or []:
        message = entry.get("DiagnosticMessage") or {}
        replacements = tuple(
            Replacement(
                path=clang_tidy_report.repository_path(
                    str(replacement.get("FilePath", ""))
                ),
                offset=int(replacement.get("Offset", 0)),
                length=int(replacement.get("Length", 0)),
                text=str(replacement.get("ReplacementText") or ""),
            )
            for replacement in message.get("Replacements") or []
        )
        diagnostics.append(
            Diagnostic(
                check=str(entry.get("DiagnosticName", "")),
                path=clang_tidy_report.repository_path(
                    str(message.get("FilePath", ""))
                ),
                offset=int(message.get("FileOffset", 0)),
                message=str(message.get("Message", "")),
                replacements=replacements,
            )
        )
    return diagnostics


@dataclasses.dataclass
class Plan:
    """What applying the fix-its would do: the accepted replacements per file and the diagnostics left alone."""

    replacements: dict[str, set[Replacement]] = dataclasses.field(default_factory=dict)
    refused: list[tuple[tuple[str, str, int, str], str]] = dataclasses.field(
        default_factory=list
    )
    applied: int = 0
    without_fix: int = 0


def _matches(check: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(check, glob) for glob in globs)


def _refusal(replacements: set[Replacement], root: str, paths: list[str]) -> str | None:
    for replacement in sorted(replacements):
        path = replacement.path
        if os.path.isabs(path) or path.startswith(_OUTSIDE_SOURCE_TREE):
            return f"edits {path}, which is outside the source tree"
        if not os.path.isfile(os.path.join(root, path)):
            return f"edits {path}, which is not a file of the source tree"
        if not clang_tidy_report.in_paths(path, paths):
            return f"edits {path}, which is outside --paths"
    return None


def plan(
    diagnostics: list[Diagnostic], globs: list[str], paths: list[str], root: str
) -> Plan:
    """Decides which fix-its to apply, all or nothing per diagnostic.

    Args:
      diagnostics: The diagnostics of every translation unit.
      globs: The checks to apply, as fnmatch patterns (`modernize-*`).
      paths: Only diagnostics located in these directories are considered; all for none.
      root: The repository root.

    Returns:
      The plan.
    """
    occurrences: dict[tuple[str, str, int, str], list[Diagnostic]] = (
        collections.defaultdict(list)
    )
    for diagnostic in diagnostics:
        if _matches(diagnostic.check, globs) and clang_tidy_report.in_paths(
            diagnostic.path, paths
        ):
            occurrences[diagnostic.identity].append(diagnostic)

    result = Plan()
    for identity in sorted(occurrences):
        reported = occurrences[identity]
        replacements: set[Replacement] = set()
        for diagnostic in reported:
            replacements.update(diagnostic.replacements)
        if not replacements:
            result.without_fix += 1
            continue
        reason = None
        if any(not diagnostic.replacements for diagnostic in reported):
            reason = "a translation unit reported it without a fix-it (inside a macro?)"
        reason = reason or _refusal(replacements, root, paths)
        if reason is None:
            for replacement in replacements:
                if any(
                    replacement.overlaps(other)
                    for other in result.replacements.get(replacement.path, ())
                ):
                    reason = f"its edit of {replacement.path} overlaps another fix-it"
                    break
        if reason is not None:
            result.refused.append((identity, reason))
            continue
        for replacement in replacements:
            result.replacements.setdefault(replacement.path, set()).add(replacement)
        result.applied += 1
    return result


def apply(result: Plan, root: str) -> list[str]:
    """Writes the plan's replacements into the files under `root`.

    Args:
      result: The plan.
      root: The directory the replacements' paths are relative to.

    Returns:
      The edited files, repository-relative and sorted.
    """
    for path, replacements in result.replacements.items():
        full_path = os.path.join(root, path)
        with open(full_path, "rb") as f:
            content = f.read()
        for replacement in sorted(
            replacements, key=lambda r: (r.offset, r.length), reverse=True
        ):
            content = (
                content[: replacement.offset]
                + replacement.text.encode("utf-8")
                + content[replacement.offset + replacement.length :]
            )
        with open(full_path, "wb") as f:
            f.write(content)
    return sorted(result.replacements)


def load_diagnostics(fixes_files: list[str]) -> list[Diagnostic]:
    """The diagnostics of every `--export-fixes` file that exists.

    Args:
      fixes_files: Absolute paths of the YAML files.

    Returns:
      The diagnostics, in file order.
    """
    diagnostics = []
    for path in fixes_files:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                diagnostics += parse_fixes(f.read())
    return diagnostics


def main(argv: list[str] | None = None) -> int:
    """Applies the selected fix-its of a clang-tidy build and returns 1 when any diagnostic was left alone."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "bep", help="the --build_event_json_file of the clang-tidy build"
    )
    parser.add_argument(
        "--checks",
        required=True,
        help="comma-separated fnmatch patterns of the checks to apply",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=[],
        help="apply only diagnostics located in these directories",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="the repository root (default: the working directory)",
    )
    parser.add_argument(
        "--no-format",
        action="store_true",
        help="do not run clang-format on the edited files",
    )
    arguments = parser.parse_args(argv)

    globs = [glob.strip() for glob in arguments.checks.split(",") if glob.strip()]
    diagnostics = load_diagnostics(
        clang_tidy_report.output_group_files(arguments.bep, OUTPUT_GROUP)
    )
    result = plan(diagnostics, globs, arguments.paths, arguments.root)
    edited = apply(result, arguments.root)
    if edited and not arguments.no_format and shutil.which("clang-format"):
        subprocess.run(["clang-format", "-i", *edited], cwd=arguments.root, check=True)
    for path in edited:
        print(f"edited {path}")
    for (check, path, offset, message), reason in result.refused:
        print(f"left alone: {path} (byte {offset}): {message} [{check}]: {reason}")
    print(
        f"clang-tidy fixes: {result.applied} applied in {len(edited)} files, {len(result.refused)} left alone, "
        f"{result.without_fix} without a fix-it."
    )
    return 1 if result.refused else 0


if __name__ == "__main__":
    sys.exit(main())
