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

"""Tests for checks.py: the registry of token checks and the engine that applies NOLINT markers."""

import os
import re
import tempfile
import unittest

from tools.hooks import check_test_support
from tools.hooks import check_types
from tools.hooks import checks
from tools.hooks import lint_code

_BRITISH = "the centre\n"  # NOLINT(american-spelling): the British spelling is the test's input
_AMERICAN = "the center\n"
_MARKED = "the centre  NOLINT(american-spelling): a title\n"  # NOLINT(american-spelling): the test's input


def _run(
    source: str, path: str, *names: str, root: str = "/nonexistent"
) -> list[tuple[str, int]]:
    return [
        (f.check, f.line) for f in checks.run_file(source, path, frozenset(names), root)
    ]


class RegistryTest(unittest.TestCase):
    def test_names_are_unique_kebab_case_and_described(self):
        names = [check.name for check in checks.REGISTRY]
        self.assertEqual(
            len(names), len(set(names)), "a check name is registered twice"
        )
        for check in checks.REGISTRY:
            with self.subTest(check=check.name):
                self.assertRegex(check.name, r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
                self.assertTrue(check.description)
                self.assertTrue(check.languages)
                if check.fixed_by_format:
                    self.assertIsNotNone(check.fix_source)

    def test_no_check_shadows_a_step_or_a_cpplint_category(self):
        self.assertEqual(checks.names() & lint_code.STEP_NAMES, frozenset())
        self.assertEqual(checks.names() & checks.CPPLINT_CATEGORIES, frozenset())

    def test_pending_names_checks_or_steps(self):
        self.assertEqual(
            checks.PENDING - checks.names() - lint_code.STEP_NAMES, frozenset()
        )
        self.assertEqual(checks.enforced() & checks.PENDING, frozenset())

    def test_every_check_module_is_registered(self):
        # A module with a CHECKS list that the registry does not concatenate would never run.
        hooks = os.path.dirname(
            check_test_support.repository_file("tools/hooks/checks.py")
        )
        registered = {
            check.check_source.__module__.rsplit(".", 1)[-1]
            for check in checks.REGISTRY
        }
        for name in os.listdir(hooks):
            if not name.endswith(".py") or name.startswith(("test_", "_")):
                continue
            with open(os.path.join(hooks, name), encoding="utf-8") as f:
                if re.search(r"^CHECKS = \[", f.read(), re.MULTILINE):
                    self.assertIn(
                        name[: -len(".py")],
                        registered,
                        f"{name} defines CHECKS the registry misses",
                    )

    def test_by_name(self):
        self.assertEqual(checks.by_name("boost").name, "boost")
        with self.assertRaises(KeyError):
            checks.by_name("no-such-check")

    def test_format_fixes_are_enforced_and_behavior_preserving(self):
        for name in checks.format_fixes():
            check = checks.by_name(name)
            self.assertTrue(check.fixed_by_format)
            self.assertNotIn(name, checks.PENDING)


class EngineTest(unittest.TestCase):
    def test_only_selected_checks_are_reported(self):
        source = "#include <boost/variant.hpp>\nvoid g() { f(a, 0); }\n"
        self.assertEqual(_run(source, "src/a.cpp", "boost"), [("boost", 1)])
        self.assertEqual(
            sorted(_run(source, "src/a.cpp", "boost", "argument-comment")),
            [("argument-comment", 2), ("boost", 1)],
        )

    def test_a_check_reads_only_its_languages_and_scope(self):
        self.assertEqual(
            _run("void g() { f(a, 0); }\n", "src/a.py", "argument-comment"), []
        )
        self.assertEqual(
            _run("void g() { f(a, 0); }\n", "lib/x/a.cpp", "argument-comment"), []
        )

    def test_a_marker_with_a_reason_suppresses(self):
        source = "void g() { f(a, 0); }  // NOLINT(argument-comment): a test\n"
        self.assertEqual(
            _run(
                source,
                "src/a.cpp",
                "argument-comment",
                "nolint-reason",
                "nolint-unused",
            ),
            [],
        )

    def test_a_marker_without_a_reason_suppresses_nothing_and_is_reported(self):
        source = "void g() { f(a, 0); }  // NOLINT(argument-comment)\n"
        self.assertEqual(
            sorted(_run(source, "src/a.cpp", "argument-comment", "nolint-reason")),
            [("argument-comment", 1), ("nolint-reason", 1)],
        )

    def test_a_bare_marker(self):
        self.assertEqual(
            _run("int x;  // NOLINT\n", "src/a.cpp", "nolint-category"),
            [("nolint-category", 1)],
        )

    def test_unused_markers(self):
        source = "int x;  // NOLINT(boost): nothing to suppress\n"
        self.assertEqual(
            _run(source, "src/a.cpp", "nolint-unused"), [("nolint-unused", 1)]
        )
        # A marker for a check that does not read the file is unused too.
        self.assertEqual(
            _run(
                "x = 1  # NOLINT(argument-comment): not C++\n",
                "src/a.py",
                "nolint-unused",
            ),
            [("nolint-unused", 1)],
        )
        # A marker for a cpplint or clang-tidy check is not the registry's to judge.
        self.assertEqual(
            _run(
                "long x;  // NOLINT(runtime/int): POSIX\n", "src/a.cpp", "nolint-unused"
            ),
            [],
        )

    def test_a_marker_for_an_unselected_check_is_used_when_it_suppresses(self):
        # During a sweep the check named by the marker may be PENDING, and so not reported; its marker is still used.
        source = "#include <boost/variant.hpp>  // NOLINT(boost): a test\n"
        self.assertEqual(_run(source, "src/a.cpp", "nolint-unused"), [])
        self.assertEqual(_run(source, "src/a.cpp", "nolint-unused", "boost"), [])

    def test_unbalanced_blocks(self):
        source = "// NOLINTBEGIN(boost): a block\nint x;\n"
        self.assertEqual(
            _run(source, "src/a.cpp", "nolint-unbalanced"), [("nolint-unbalanced", 1)]
        )

    def test_markers_of_text_files_are_whole_lines_and_not_judged(self):
        source = _MARKED
        self.assertEqual(
            _run(
                source,
                "docs/README.md",
                "american-spelling",
                "nolint-unused",
                "nolint-unknown",
            ),
            [],
        )
        self.assertEqual(
            _run(_BRITISH, "docs/README.md", "american-spelling"),
            [("american-spelling", 1)],
        )

    def test_unknown_categories_and_the_clang_tidy_configurations(self):
        source = "int x;  // NOLINT(google-explicit-constructor): a test\n"
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(
                _run(source, "src/a.cpp", "nolint-unknown", root=root),
                [("nolint-unknown", 1)],
            )
            os.makedirs(os.path.join(root, "tools/clang_tidy"))
            sweep = os.path.join(root, "tools/clang_tidy/sweep.clang-tidy")
            with open(sweep, "w", encoding="utf-8") as f:
                f.write(
                    'Checks:\n  - "-*"\n  - google-explicit-constructor  # G: Implicit conversions\n'
                )
            checks._clang_tidy_checks_cached.cache_clear()
            # Accepted while the sweep configuration lists it ...
            self.assertEqual(_run(source, "src/a.cpp", "nolint-unknown", root=root), [])
            # ... and rejected once that file is gone and .clang-tidy does not list it.
            os.remove(sweep)
            with open(os.path.join(root, ".clang-tidy"), "w", encoding="utf-8") as f:
                f.write("Checks: '-*,modernize-*'\n")
            checks._clang_tidy_checks_cached.cache_clear()
            self.assertEqual(
                _run(source, "src/a.cpp", "nolint-unknown", root=root),
                [("nolint-unknown", 1)],
            )
            self.assertEqual(
                _run(
                    "int x;  // NOLINT(modernize-use-nullptr): a test\n",
                    "src/a.cpp",
                    "nolint-unknown",
                    root=root,
                ),
                [],
            )
            checks._clang_tidy_checks_cached.cache_clear()

    def test_a_pending_check_is_known_to_nolint(self):
        for name in checks.names():
            self.assertTrue(checks.is_known_category(name, "/nonexistent"))

    def test_fix_file(self):
        source = _BRITISH
        self.assertEqual(
            checks.fix_file(source, "a.md", frozenset({"american-spelling"})),
            _AMERICAN,
        )
        self.assertEqual(checks.fix_file(source, "a.md", frozenset({"boost"})), source)

    def test_clang_tidy_checks(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, ".clang-tidy"), "w", encoding="utf-8") as f:
                f.write(
                    '---\nChecks:\n  - "-*"\n  # Abseil\n  - abseil-str-cat-append   # ToTW #3\n  - bugprone-*\n'
                    'WarningsAsErrors: "*"\n'
                )
            self.assertEqual(
                checks.clang_tidy_checks(root),
                frozenset({"abseil-str-cat-append", "bugprone-*"}),
            )

    def test_finding_format(self):
        finding = check_types.Finding("a.cpp", 3, 7, "boost", "message")
        self.assertEqual(str(finding), "a.cpp:3:7: message [boost]")


if __name__ == "__main__":
    unittest.main()
