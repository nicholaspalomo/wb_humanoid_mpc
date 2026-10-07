#!/usr/bin/env python3
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

"""The repository's formatter (make format): every rewrite that cannot change behavior.

Run it from the repository root as a module, `python3 -m tools.hooks.format_code`. It

- trims trailing whitespace and ends every text file with exactly one newline;
- applies the fixes of the registry checks that `make format` owns (tools/hooks/checks.py,
  `fixed_by_format`): rewrites such as floating-point radix points that cannot change behavior;
- formats C++ with clang-format (.clang-format);
- orders Python imports with isort (.isort.cfg);
- formats Python with black.

The file set is tools/hooks/lint_files.py's: what git tracks or would track, without vendored code (lib/) and fixtures.

A formatter that is missing, or of another version than the one the lint pins (tools/hooks/lint_code.py:
pinned_version()), is not run: another clang-format or black formats differently, so `make lint` and CI would then
disagree with the result. Each one skipped is reported as a warning (`make lint` with CI=true fails on it). A formatter
that runs and fails is an error: the run exits non-zero.
"""

import argparse
from collections.abc import Callable, Sequence
import functools
import os
import subprocess
import sys
from typing import NamedTuple

from tools.hooks import check_types
from tools.hooks import checks
from tools.hooks import lint_code
from tools.hooks import lint_files

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ISORT_CONFIG = ".isort.cfg"


def fix_whitespace(content: str) -> str:
    """`content` without trailing whitespace and ending in exactly one newline (an empty file stays empty)."""
    if not content:
        return content
    return "\n".join(line.rstrip() for line in content.splitlines()) + "\n"


def _formatted_files(root: str) -> list[str]:
    return [
        path
        for path in lint_files.repository_files(root)
        if lint_files.in_scope(path, lint_files.Scope.TEXT)
        and not lint_files.is_fixture(path)
    ]


def _rewrite(root: str, path: str, transform: Callable[[str], str]) -> bool:
    """Rewrites the file at `path` with `transform` and returns whether that changed it."""
    full = os.path.join(root, path)
    try:
        with open(full, encoding="utf-8") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        return False
    rewritten = transform(content)
    if rewritten == content:
        return False
    with open(full, "w", encoding="utf-8") as f:
        f.write(rewritten)
    return True


def _fix(content: str, path: str, fixes: frozenset[str]) -> str:
    """`content` of the file at `path` with the fixes of the checks `fixes` applied."""
    return checks.fix_file(content, path, fixes)


class FormatterOutcome(NamedTuple):
    """What one formatter did: a warning when it was not run, an error when it failed (each None otherwise)."""

    warning: str | None = None
    error: str | None = None


def _run_formatter(command: list[str], root: str) -> FormatterOutcome:
    """Runs a formatter when it is installed at its pinned version (lint_code.pinned_version())."""
    tool = command[0]
    problem = lint_code.tool_problem(root, tool)
    if problem is not None:
        return FormatterOutcome(warning=f"{problem}: {tool} was not run")
    result = subprocess.run(
        command,
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return FormatterOutcome(
            error=f"{tool} failed with status {result.returncode}: {result.stderr.strip()[:2000]}"
        )
    return FormatterOutcome()


def format_repository(root: str) -> tuple[list[str], list[FormatterOutcome]]:
    """Formats the repository at `root` in place.

    Args:
      root: The repository root.

    Returns:
      The files whose whitespace or registry fixes changed, and the outcome of each formatter that was not run or
      failed.
    """
    files = _formatted_files(root)
    modified = []
    fixes = checks.format_fixes()
    for path in files:
        changed = False
        if check_types.is_text(path):
            changed = _rewrite(root, path, fix_whitespace) or changed
        if fixes:
            changed = (
                _rewrite(root, path, functools.partial(_fix, path=path, fixes=fixes))
                or changed
            )
        if changed:
            modified.append(path)
    cpp = [
        path
        for path in files
        if path.endswith(lint_files.CPP_EXTENSIONS) and lint_files.is_first_party(path)
    ]
    python = [
        path
        for path in files
        if path.endswith(".py") and lint_files.is_first_party(path)
    ]
    commands = []
    if cpp:
        commands.append(["clang-format", "-i", *cpp])
    if python:
        commands.append(
            [
                "isort",
                "--quiet",
                "--settings-path",
                os.path.join(root, ISORT_CONFIG),
                *lint_files.isort_package_arguments(root),
                *python,
            ]
        )
        commands.append(["black", "--quiet", *python])
    outcomes = [_run_formatter(command, root) for command in commands]
    return modified, [o for o in outcomes if o.warning or o.error]


def main(argv: Sequence[str] | None = None) -> int:
    """Formats the repository."""
    parser = argparse.ArgumentParser(
        description="The repository's formatter (make format)."
    )
    parser.parse_args(argv)
    modified, problems = format_repository(REPO_ROOT)
    if modified:
        print(f"✨ Formatted and ensured trailing newlines on {len(modified)} file(s).")
    else:
        print(
            "✅ All files properly formatted with required blank lines and trailing newlines."
        )
    for outcome in problems:
        if outcome.warning:
            print(f"⚠️ Warning: {outcome.warning}.", file=sys.stderr)
        if outcome.error:
            print(f"❌ {outcome.error}", file=sys.stderr)
    return 1 if any(outcome.error for outcome in problems) else 0


if __name__ == "__main__":
    sys.exit(main())
