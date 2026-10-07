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

"""What every test of a registry check asserts: that it is wired in, honors NOLINT, reads the index and keeps its scope.

A check's own test covers what it detects; `assert_check_behaves()` covers the rest, the same way for every check:

- it is in the registry (tools/hooks/checks.py), and lint_code runs it by default;
- `NOLINT(<check>): <reason>` and `NOLINTNEXTLINE(<check>): <reason>` suppress a finding, and a marker without a
  reason suppresses nothing (and is itself reported as `nolint-reason`);
- `lint_code --git-staged` judges the staged version of a file, not the working tree;
- vendored files (lib/) are out of scope;
- a fix, if the check has one, is a fixed point that makes the check pass.
"""

import os
import subprocess
import tempfile
import unittest

from tools.hooks import check_types
from tools.hooks import checks
from tools.hooks import lint_code
from tools.hooks import lint_files


def repository_file(relative_path: str) -> str:
    """A repository file, from the Bazel runfiles when run by Bazel and from the source tree otherwise."""
    roots = []
    if "TEST_SRCDIR" in os.environ:
        roots.append(os.path.join(os.environ["TEST_SRCDIR"], "_main"))
    roots.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    for root in roots:
        candidate = os.path.join(root, relative_path)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(relative_path)


def read_repository_file(relative_path: str) -> str:
    """The content of a repository file (repository_file())."""
    with open(repository_file(relative_path), encoding="utf-8") as f:
        return f.read()


def source_tree_root() -> str:
    """The checkout's root, for a test that has to see every file of the tree, not only its own data.

    Bazel's runfiles hold only a test's data, as links to the source files, so the checkout is where the link of
    MODULE.bazel points; the test needs `//:MODULE.bazel` in its `data`. Outside Bazel it is the tree this file is in.

    Returns:
      The absolute path of the checkout's root.
    """
    return os.path.dirname(os.path.realpath(repository_file("MODULE.bazel")))


def findings(
    name: str, source: str, path: str, root: str = "/nonexistent"
) -> list[check_types.Finding]:
    """The findings the registry engine reports for check `name` on `source`, the file at `path`."""
    return checks.run_file(source, path, frozenset({name}), root)


def _comment(path: str) -> str:
    language = check_types.language_of(path)
    if language in (check_types.Language.CPP, check_types.Language.PROTO):
        return "//"
    return "#"


def _with_marker(source: str, line: int, marker: str, path: str) -> str:
    lines = source.split("\n")
    lines[line - 1] = f"{lines[line - 1]}  {_comment(path)} {marker}"
    return "\n".join(lines)


def _with_marker_above(source: str, line: int, marker: str, path: str) -> str:
    lines = source.split("\n")
    indent = len(lines[line - 1]) - len(lines[line - 1].lstrip())
    lines.insert(line - 1, " " * indent + f"{_comment(path)} {marker}")
    return "\n".join(lines)


class StagedRepository:
    """A temporary git repository, for the staged mode of the linter."""

    def __init__(self) -> None:
        # pylint: disable-next=consider-using-with  # cleanup() deletes it.
        self._directory = tempfile.TemporaryDirectory()
        self.root = self._directory.name
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "test")

    def git(self, *args: str) -> str:
        """Runs git in the repository and returns what it printed."""
        return subprocess.run(
            ["git", "-C", self.root, *args], check=True, capture_output=True, text=True
        ).stdout

    def write(self, path: str, text: str) -> None:
        """Writes a file of the working tree."""
        full = os.path.join(self.root, path)
        os.makedirs(os.path.dirname(full) or self.root, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(text)

    def staged_findings(self, name: str) -> list[check_types.Finding]:
        """The findings of check `name` on the staged files, as `lint_code --git-staged` reads them."""
        context = lint_code.Context(
            root=self.root,
            files=lint_files.staged_files(self.root),
            staged=True,
            selected=frozenset({name}),
            steps=frozenset({"token-checks"}),
            summary=False,
            ci=False,
        )
        return lint_code._token_findings(context)

    def cleanup(self) -> None:
        """Deletes the repository."""
        self._directory.cleanup()


def assert_registered(test: unittest.TestCase, name: str) -> checks.Check:
    """Asserts that check `name` is in the registry and that lint_code runs it by default."""
    check = checks.by_name(name)
    test.assertTrue(check.description, f"{name} has no description")
    test.assertIn("token-checks", lint_code.STEP_NAMES)
    selected, steps = lint_code._parse_only(None)
    test.assertIn(name, selected)
    test.assertIn("token-checks", steps)
    return check


def assert_hook_runs_the_linter(test: unittest.TestCase) -> None:
    """Asserts that the pre-commit hook runs the linter on the staged files after staging the formatter's changes."""
    hook = read_repository_file("tools/hooks/pre-commit")
    command = "python3 -m tools.hooks.lint_code --git-staged"
    test.assertIn("if ! " + command + "; then", hook)
    test.assertLess(hook.index('git add "$file"'), hook.index("if ! " + command))
    after = hook[hook.index("if ! " + command) :]
    test.assertLess(after.index("exit 1"), after.index("completed successfully"))


def assert_check_behaves(
    test: unittest.TestCase,
    name: str,
    flagged: str,
    path: str,
    *,
    clean: str | None = None,
) -> None:
    """Asserts the behavior every registry check shares (the module docstring).

    Args:
      test: The calling test.
      name: The check's name.
      flagged: A source with exactly one finding of the check, on a line that can carry a trailing comment.
      path: The first-party path the source is checked as.
      clean: A source the check accepts, if not just the fixed `flagged`.
    """
    check = assert_registered(test, name)
    test.assertTrue(check.applies_to(path), f"{name} does not read {path}")
    found = findings(name, flagged, path)
    test.assertEqual(len(found), 1, f"{name} on {path}: {found}")
    line = found[0].line
    test.assertEqual(found[0].check, name)
    test.assertIn(f"[{name}]", str(found[0]))
    if clean is not None:
        test.assertEqual(
            findings(name, clean, path), [], f"{name} flags the clean case"
        )

    # NOLINT with a reason on the line, and NOLINTNEXTLINE above it, suppress the finding.
    suppressed = _with_marker(flagged, line, f"NOLINT({name}): a test", path)
    test.assertEqual(findings(name, suppressed, path), [])
    above = _with_marker_above(flagged, line, f"NOLINTNEXTLINE({name}): a test", path)
    test.assertEqual(findings(name, above, path), [])
    # A marker without a reason suppresses nothing, and is reported.
    reasonless = _with_marker(flagged, line, f"NOLINT({name})", path)
    test.assertEqual(len(findings(name, reasonless, path)), 1)
    if check_types.language_of(path) is not None and lint_files.is_first_party(path):
        test.assertEqual(
            [f.check for f in findings("nolint-reason", reasonless, path)],
            ["nolint-reason"],
        )

    # Vendored code is out of scope.
    if check.scope != lint_files.Scope.NOT_THIRDPARTY:
        test.assertEqual(findings(name, flagged, "lib/vendored/" + path), [])

    # The staged version is what counts.
    repository = StagedRepository()
    try:
        repository.write(path, flagged)
        repository.git("add", path)
        test.assertEqual(
            [(f.path, f.check) for f in repository.staged_findings(name)],
            [(path, name)],
        )
        repository.write(path, clean if clean is not None else flagged + "\n")
        test.assertEqual(len(repository.staged_findings(name)), 1)
    finally:
        repository.cleanup()

    # A fix is a fixed point that makes the check pass.
    if check.fix_source is not None:
        fixed = check.fix_source(flagged, path)
        test.assertEqual(
            findings(name, fixed, path), [], f"{name}'s fix leaves findings"
        )
        test.assertEqual(
            check.fix_source(fixed, path), fixed, f"{name}'s fix is not a fixed point"
        )
