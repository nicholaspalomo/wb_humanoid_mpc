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

"""Every lint rule is enforced: no pending list, no sweep configuration, no baseline (AGENTS.md, "Style guides").

While the style rules were rolled out, a rule with findings left in the tree could wait: a token check in a PENDING
set of tools/hooks/checks.py, a cpplint category or pylint message in a SWEEP block of CPPLINT.cfg or .pylintrc, a
relaxed module section in mypy.ini, and a clang-tidy check only in tools/clang_tidy/sweep.clang-tidy, with its own
aspect, .bazelrc configuration and `make lint-tidy-sweep`. All of that is gone, and this test keeps it gone. It also
checks the other half of the promise: that `make lint`, the pre-commit hook and CI's format job run every check of the
registry and every step, with nothing narrowed. A justified exception is a NOLINT marker with its reason, on its line.
"""

import configparser
import contextlib
import inspect
import io
import os
import re
import unittest
from unittest import mock

from tools.hooks import check_test_support
from tools.hooks import check_types
from tools.hooks import checks
from tools.hooks import lint_code
from tools.hooks import lint_files

# The lint configurations, which no rollout marker may be left in.
CONFIGS = (
    ".bazelrc",
    ".clang-tidy",
    ".isort.cfg",
    ".pylintrc",
    "CPPLINT.cfg",
    "Makefile",
    "mypy.ini",
    "tools/clang_tidy/clang_tidy.bzl",
)

# What the rollout's mechanisms were called: a marker block of a configuration, and the second clang-tidy configuration
# with everything named after it.
_SWEEP_MARKER = re.compile(r"\bSWEEP(_BEGIN|_END)?\b")
_SWEEP_NAMES = re.compile(
    r"sweep\.clang-tidy|clang_tidy_sweep_aspect|clang-tidy-sweep|lint-tidy-sweep"
)
# The files that name the old mechanism on purpose: this test, and the test that a stray sweep.clang-tidy enables
# nothing.
_NAMING_THE_SWEEP = frozenset(
    {"tools/hooks/test_lint_enforcement.py", "tools/hooks/test_checks.py"}
)
# A file that holds findings or exemptions instead of the code holding NOLINT markers.
_BASELINE_FILE = re.compile(
    r"(baseline|allowlist|suppressions?|pending|sweep)", re.IGNORECASE
)
# The directories of the lint machinery, where such a file would live; the fixtures are deliberately bad code.
_LINT_DIRECTORIES = ("tools/hooks/", "tools/clang_tidy/")

# mypy's options that, set in a module's own section, would hold that module to less than the others.
_RELAXING_MYPY_OPTIONS = {
    "ignore_errors": "true",
    "disallow_untyped_defs": "false",
    "disallow_incomplete_defs": "false",
    "check_untyped_defs": "false",
}


def _first_party_files(root: str) -> list[str]:
    return [
        path
        for path in lint_files.repository_files(root)
        if lint_files.is_first_party(path)
    ]


def _context_of(argv: list[str]) -> lint_code.Context:
    """The context lint_code.main() builds for `argv`, without linting anything."""
    with mock.patch.object(lint_code, "run", return_value=0) as run, mock.patch.object(
        lint_code, "_select_files", return_value=[]
    ), contextlib.redirect_stdout(io.StringIO()):
        lint_code.main(argv)
    run.assert_called_once()
    context = run.call_args.args[0]
    assert isinstance(context, lint_code.Context)
    return context


class NoPendingListTest(unittest.TestCase):
    def test_the_registry_has_no_pending_set(self):
        self.assertFalse(hasattr(checks, "PENDING"))
        self.assertFalse(
            hasattr(checks, "enforced"), "every check is enforced: use checks.names()"
        )

    def test_no_lint_module_or_test_keeps_one(self):
        root = check_test_support.source_tree_root()
        assignment = re.compile(r"^\s*PENDING\b\s*(:[^=]*)?=", re.MULTILINE)
        for path in _first_party_files(root):
            if not (path.startswith(_LINT_DIRECTORIES) and path.endswith(".py")):
                continue
            with self.subTest(path=path):
                with open(os.path.join(root, path), encoding="utf-8") as f:
                    self.assertIsNone(assignment.search(f.read()))

    def test_a_check_test_cannot_declare_its_check_pending(self):
        for helper in (
            check_test_support.assert_registered,
            check_test_support.assert_check_behaves,
        ):
            with self.subTest(helper=helper.__name__):
                self.assertNotIn("pending", inspect.signature(helper).parameters)


class NoSweepConfigurationTest(unittest.TestCase):
    def test_no_configuration_has_a_sweep_block(self):
        for config in CONFIGS:
            with self.subTest(config=config):
                text = check_test_support.read_repository_file(config)
                self.assertIsNone(_SWEEP_MARKER.search(text))

    def test_nothing_names_the_sweep_configuration_any_more(self):
        root = check_test_support.source_tree_root()
        self.assertFalse(
            os.path.exists(os.path.join(root, "tools/clang_tidy/sweep.clang-tidy"))
        )
        for path in _first_party_files(root):
            if path in _NAMING_THE_SWEEP or not check_types.is_text(path):
                continue
            with open(os.path.join(root, path), encoding="utf-8", errors="ignore") as f:
                found = _SWEEP_NAMES.findall(f.read())
            with self.subTest(path=path):
                self.assertEqual(found, [])

    def test_clang_tidy_has_one_configuration(self):
        root = check_test_support.source_tree_root()
        configurations = [
            path
            for path in lint_files.repository_files(root)
            if os.path.basename(path).endswith(".clang-tidy")
            and not lint_files.is_vendored(path)
        ]
        self.assertEqual(configurations, [".clang-tidy"])
        self.assertEqual(checks.CLANG_TIDY_CONFIGS, (".clang-tidy",))

    def test_every_clang_tidy_build_uses_the_one_aspect(self):
        bazelrc = check_test_support.read_repository_file(".bazelrc")
        aspects = re.findall(r"^build:(\S+) --aspects=(\S+)$", bazelrc, re.MULTILINE)
        self.assertEqual(
            sorted(aspects),
            [
                ("clang-tidy", "//tools/clang_tidy:clang_tidy.bzl%clang_tidy_aspect"),
                (
                    "clang-tidy-fix",
                    "//tools/clang_tidy:clang_tidy.bzl%clang_tidy_aspect",
                ),
            ],
        )
        bzl = check_test_support.read_repository_file("tools/clang_tidy/clang_tidy.bzl")
        self.assertEqual(
            re.findall(r"^(\w+) = aspect\(", bzl, re.MULTILINE), ["clang_tidy_aspect"]
        )

    def test_the_lint_machinery_keeps_no_baseline_file(self):
        root = check_test_support.source_tree_root()
        for path in _first_party_files(root):
            top_level = "/" not in path
            if not (top_level or path.startswith(_LINT_DIRECTORIES)):
                continue
            with self.subTest(path=path):
                self.assertIsNone(_BASELINE_FILE.search(os.path.basename(path)))


class NoRelaxedConfigurationTest(unittest.TestCase):
    def test_mypy_holds_every_module_to_the_global_options(self):
        parser = configparser.ConfigParser()
        parser.read(check_test_support.repository_file("mypy.ini"), encoding="utf-8")
        self.assertTrue(parser.getboolean("mypy", "disallow_untyped_defs"))
        self.assertTrue(parser.getboolean("mypy", "disallow_incomplete_defs"))
        self.assertTrue(parser.getboolean("mypy", "check_untyped_defs"))
        for section in parser.sections():
            if not section.startswith("mypy-"):
                continue
            for option, relaxed in _RELAXING_MYPY_OPTIONS.items():
                with self.subTest(section=section, option=option):
                    self.assertNotEqual(
                        parser.get(section, option, fallback="").strip().lower(),
                        relaxed,
                    )
        # lint_code's mypy step fails on any module section at all.
        self.assertEqual(
            lint_code.mypy_config_problems(check_test_support.source_tree_root()), []
        )

    def test_every_cpplint_filter_says_why(self):
        lines = check_test_support.read_repository_file("CPPLINT.cfg").splitlines()
        filters = [
            number for number, line in enumerate(lines) if line.startswith("filter=")
        ]
        self.assertTrue(filters)
        for number in filters:
            with self.subTest(line=lines[number]):
                self.assertGreater(number, 0)
                self.assertTrue(
                    lines[number - 1].startswith("# "),
                    "a filter needs the comment above it",
                )


class EveryCheckRunsTest(unittest.TestCase):
    def test_make_lint_runs_every_check_and_step(self):
        context = _context_of([])
        self.assertEqual(context.selected, checks.names())
        self.assertEqual(context.steps, lint_code.STEP_NAMES)
        self.assertFalse(context.staged)

    def test_the_staged_run_runs_every_check_and_step(self):
        context = _context_of(["--git-staged"])
        self.assertEqual(context.selected, checks.names())
        self.assertEqual(context.steps, lint_code.STEP_NAMES)
        self.assertTrue(context.staged)

    def test_the_callers_do_not_narrow_the_run(self):
        # Each caller's command, with no --only and no --paths: the Makefile's `lint` recipe, the pre-commit hook and
        # CI's format job.
        callers = {
            "Makefile": r"^lint:\n\t@?python3 -m tools\.hooks\.lint_code$",
            "tools/hooks/pre-commit": r"^if ! python3 -m tools\.hooks\.lint_code --git-staged; then$",
            ".github/workflows/format_test.yml": r"^\s+run: python3 -m tools\.hooks\.lint_code$",
        }
        for path, command in callers.items():
            with self.subTest(caller=path):
                self.assertRegex(
                    check_test_support.read_repository_file(path),
                    re.compile(command, re.MULTILINE),
                )


if __name__ == "__main__":
    unittest.main()
