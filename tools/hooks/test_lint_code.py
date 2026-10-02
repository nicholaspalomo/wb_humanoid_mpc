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

"""Tests for lint_code.py: the step runner, --only / --paths / --fix / --summary, the tool versions and the lint lock."""

import contextlib
import io
import multiprocessing
import os
import tempfile
import time
import unittest
from unittest import mock

from tools.hooks import check_test_support
from tools.hooks import checks
from tools.hooks import lint_code

_BRITISH = "the centre\n"  # NOLINT(american-spelling): the British spelling is the test's input


def _context(root: str, files: list[str], **overrides: object) -> lint_code.Context:
    values: dict[str, object] = {
        "root": root,
        "files": files,
        "staged": False,
        "selected": checks.enforced(),
        "steps": frozenset({"token-checks"}),
        "summary": False,
        "ci": False,
    }
    values.update(overrides)
    return lint_code.Context(**values)  # type: ignore[arg-type]


class StepsTest(unittest.TestCase):
    def test_the_steps(self):
        names = [step.name for step in lint_code.STEPS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[0], "ifttt")
        self.assertIn("token-checks", names)
        for heavy in ("cpplint", "pylint", "pylint-tests", "mypy", "mypy-tests"):
            self.assertTrue(
                next(s for s in lint_code.STEPS if s.name == heavy).heavy, heavy
            )
        self.assertFalse(
            next(s for s in lint_code.STEPS if s.name == "token-checks").heavy
        )

    def test_every_step_runs_and_the_run_fails_at_the_end(self):
        ran = []

        def step(name: str, status: str) -> lint_code.Step:
            def run(context: lint_code.Context) -> lint_code.Result:
                del context  # Unused.
                ran.append(name)
                return lint_code.Result(
                    status, [f"{name} said something"], hint=f"fix {name}"
                )

            return lint_code.Step(name, f"Running {name}", run)

        fake = (step("a", "failed"), step("b", "passed"), step("c", "skipped"))
        output = io.StringIO()
        with mock.patch.object(lint_code, "STEPS", fake), contextlib.redirect_stdout(
            output
        ):
            status = lint_code.run(_context("/x", [], steps=frozenset({"a", "b", "c"})))
        self.assertEqual(status, 1)
        self.assertEqual(ran, ["a", "b", "c"], "a failing step stopped the run")
        text = output.getvalue()
        self.assertIn("🔍 1/3 Running a...", text)
        self.assertIn("🔍 3/3 Running c...", text)
        self.assertIn("💡 fix a", text)
        self.assertNotIn("💡 fix b", text)
        self.assertIn("❌ Lint failed in: a (skipped: c)", text)

    def test_a_clean_run_passes(self):
        fake = (lint_code.Step("a", "A", lambda context: lint_code.Result("passed")),)
        with mock.patch.object(lint_code, "STEPS", fake), contextlib.redirect_stdout(
            io.StringIO()
        ):
            self.assertEqual(
                lint_code.run(_context("/x", [], steps=frozenset({"a"}))), 0
            )


class OnlyTest(unittest.TestCase):
    def test_by_default_every_enforced_check_and_step(self):
        selected, steps = lint_code._parse_only(None)
        self.assertEqual(selected, checks.enforced())
        self.assertEqual(steps & checks.PENDING, frozenset())
        self.assertIn("token-checks", steps)
        self.assertIn("cpplint", steps)

    def test_checks_and_steps_by_name(self):
        selected, steps = lint_code._parse_only("boost, cpplint")
        self.assertEqual(selected, frozenset({"boost"}))
        self.assertEqual(steps, frozenset({"cpplint", "token-checks"}))
        selected, steps = lint_code._parse_only("cpplint")
        self.assertEqual((selected, steps), (frozenset(), frozenset({"cpplint"})))

    def test_a_pending_check_runs_when_named(self):
        for name in checks.PENDING & checks.names():
            selected, _ = lint_code._parse_only(name)
            self.assertEqual(selected, frozenset({name}))

    def test_an_unknown_name(self):
        with self.assertRaisesRegex(ValueError, "unknown check or step no-such"):
            lint_code._parse_only("boost,no-such")


class FilesTest(unittest.TestCase):
    def setUp(self):
        self.repository = check_test_support.StagedRepository()
        for path in ["a/x.cpp", "a/b/y.py", "c/z.md"]:
            self.repository.write(path, "x\n")

    def tearDown(self):
        self.repository.cleanup()

    def test_paths_narrow_the_files(self):
        root = self.repository.root
        select = lint_code._select_files
        self.assertEqual(select(root, False, []), ["a/b/y.py", "a/x.cpp", "c/z.md"])
        self.assertEqual(select(root, False, ["a"]), ["a/b/y.py", "a/x.cpp"])
        self.assertEqual(
            select(root, False, ["a/b/", "c/z.md"]), ["a/b/y.py", "c/z.md"]
        )

    def test_staged_files(self):
        self.repository.git("add", "a/x.cpp")
        self.assertEqual(
            lint_code._select_files(self.repository.root, True, []), ["a/x.cpp"]
        )

    def test_fix_rewrites_only_what_the_selected_checks_fix(self):
        root = self.repository.root
        self.repository.write("c/z.md", _BRITISH)
        fixed = lint_code._apply_fixes(
            root, ["c/z.md", "a/x.cpp"], frozenset({"american-spelling"})
        )
        self.assertEqual(fixed, ["c/z.md"])
        with open(os.path.join(root, "c/z.md"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "the center\n")
        self.assertEqual(
            lint_code._apply_fixes(root, ["c/z.md"], frozenset({"boost"})), []
        )


class TokenStepTest(unittest.TestCase):
    def setUp(self):
        self.repository = check_test_support.StagedRepository()
        self.repository.write("src/a.cpp", "#include <boost/variant.hpp>\n")
        self.repository.write("src/b.cpp", "boost::optional<int> x;\nboost::any y;\n")

    def tearDown(self):
        self.repository.cleanup()

    def test_findings_and_hints(self):
        context = _context(
            self.repository.root,
            ["src/a.cpp", "src/b.cpp"],
            selected=frozenset({"boost"}),
        )
        result = lint_code._run_token_checks(context)
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(result.lines), 3)
        self.assertTrue(result.lines[0].startswith("src/a.cpp:1:10: "))
        self.assertTrue(result.lines[0].endswith("[boost]"))
        self.assertIn("[boost] 3 finding(s)", result.hint)

    def test_summary(self):
        with open(
            os.path.join(self.repository.root, "src/BUILD.bazel"), "w", encoding="utf-8"
        ) as f:
            f.write("")
        context = _context(
            self.repository.root,
            ["src/a.cpp", "src/b.cpp"],
            selected=frozenset({"boost"}),
            summary=True,
        )
        result = lint_code._run_token_checks(context)
        self.assertEqual(result.status, "failed")
        self.assertEqual(
            [line.split() for line in result.lines], [["3", "boost"], ["3", "//src"]]
        )


class WhitespaceTest(unittest.TestCase):
    def test_whitespace_problem(self):
        self.assertIsNone(lint_code.whitespace_problem(""))
        self.assertIsNone(lint_code.whitespace_problem("a\nb\n"))
        for bad in ["a \n", "a", "a\t\nb\n"]:
            with self.subTest(content=bad):
                self.assertIsNotNone(lint_code.whitespace_problem(bad))


class CpplintOutputTest(unittest.TestCase):
    def test_markers_for_the_repositorys_own_checks_are_not_cpplints_business(self):
        output = (
            "a.cpp:3:  Unknown NOLINT error category: argument-comment  [readability/nolint] [5]\n"
            "a.cpp:4:  Unknown NOLINT error category: argumnet-comment  [readability/nolint] [5]\n"
            "a.cpp:9:  Use int16_t/int64_t/etc, rather than the C type long  [runtime/int] [4]\n"
            "Done processing a.cpp\n"
        )
        self.assertEqual(
            lint_code.filter_cpplint_output(output),
            [
                "a.cpp:4: Unknown NOLINT error category: argumnet-comment [readability/nolint]",
                "a.cpp:9: Use int16_t/int64_t/etc, rather than the C type long [runtime/int]",
            ],
        )


class ToolVersionTest(unittest.TestCase):
    def setUp(self):
        # pylint: disable-next=consider-using-with  # tearDown() deletes it.
        self.directory = tempfile.TemporaryDirectory()
        self.root = self.directory.name
        os.makedirs(os.path.join(self.root, "tools/hooks"))
        with open(
            os.path.join(self.root, lint_code.LOCK_FILE), "w", encoding="utf-8"
        ) as f:
            f.write(
                "cpplint==2.0.2 \\\n    --hash=sha256:abc\npylint==4.1.1 \\\n    --hash=sha256:def\n"
            )

    def tearDown(self):
        self.directory.cleanup()

    def test_pinned_versions(self):
        self.assertEqual(
            lint_code.pinned_versions(self.root),
            {"cpplint": "2.0.2", "pylint": "4.1.1"},
        )

    def test_a_matching_tool_runs(self):
        with mock.patch.object(lint_code, "installed_version", return_value="2.0.2"):
            self.assertIsNone(
                lint_code._tool_problem(_context(self.root, []), "cpplint")
            )

    def test_a_missing_or_other_tool_is_an_error_in_ci_and_a_warning_otherwise(self):
        for installed in (None, "1.6.1"):
            with self.subTest(installed=installed), mock.patch.object(
                lint_code, "installed_version", return_value=installed
            ):
                local = lint_code._tool_problem(_context(self.root, []), "cpplint")
                assert local is not None
                self.assertEqual(local.status, "skipped")
                self.assertIn("rebuild the dev container", local.lines[0])
                ci = lint_code._tool_problem(
                    _context(self.root, [], ci=True), "cpplint"
                )
                assert ci is not None
                self.assertEqual(ci.status, "failed")

    def test_the_version_is_read_from_the_tool(self):
        with mock.patch.object(
            lint_code.shutil, "which", return_value="/bin/tool"
        ), mock.patch.object(
            lint_code.subprocess,
            "run",
            return_value=mock.Mock(
                stdout="pylint 4.1.1\nastroid 4.3.3\nPython 3.12.3\n", stderr=""
            ),
        ):
            self.assertEqual(lint_code.installed_version("pylint"), "4.1.1")
        with mock.patch.object(lint_code.shutil, "which", return_value=None):
            self.assertIsNone(lint_code.installed_version("pylint"))


def _hold_lock(path: str, held: object, release: object) -> None:
    with mock.patch.dict(os.environ, {lint_code.LINT_LOCK_ENVIRONMENT: path}):
        with lint_code.lint_lock("/unused"):
            held.set()  # type: ignore[attr-defined]
            release.wait(30)  # type: ignore[attr-defined]


class MypyConfigTest(unittest.TestCase):
    def test_module_sections_name_modules_and_only_relax(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "pkg"))
            with open(
                os.path.join(root, "pkg", "BUILD.bazel"), "w", encoding="utf-8"
            ) as f:
                f.write('py_library(name = "a", imports = ["."])\n')
            for path in ("pkg/known.py", "tools/hooks/checks.py"):
                os.makedirs(os.path.dirname(os.path.join(root, path)), exist_ok=True)
                with open(os.path.join(root, path), "w", encoding="utf-8") as f:
                    f.write("")
            with open(
                os.path.join(root, lint_code.MYPY_CONFIG), "w", encoding="utf-8"
            ) as f:
                f.write(
                    "[mypy]\nstrict_equality = True\n"
                    "[mypy-known]\nignore_errors = True\n"
                    "[mypy-tools.hooks.checks]\ndisallow_untyped_defs = False\n"
                    "[mypy-gone]\nignore_errors = True\n"
                )
            with mock.patch.object(
                lint_code.lint_files.subprocess, "run", side_effect=OSError("no git")
            ):
                problems = lint_code.mypy_config_problems(root)
            self.assertEqual(len(problems), 1, problems)
            self.assertIn("[mypy-gone]", problems[0])
            with open(
                os.path.join(root, lint_code.MYPY_CONFIG), "w", encoding="utf-8"
            ) as f:
                f.write("[mypy-known]\nstrict_equality = False\n")
            with mock.patch.object(
                lint_code.lint_files.subprocess, "run", side_effect=OSError("no git")
            ):
                problems = lint_code.mypy_config_problems(root)
            self.assertEqual(len(problems), 1)
            self.assertIn("only relaxes", problems[0])


class LintLockTest(unittest.TestCase):
    def test_the_lock_file(self):
        with mock.patch.dict(os.environ, {lint_code.LINT_LOCK_ENVIRONMENT: ""}):
            self.assertEqual(
                lint_code.lint_lock_path("/repo"), "/repo/.lint_machine.lock"
            )
        with mock.patch.dict(
            os.environ, {lint_code.LINT_LOCK_ENVIRONMENT: "/tmp/x.lock"}
        ):
            self.assertEqual(lint_code.lint_lock_path("/repo"), "/tmp/x.lock")

    def test_a_second_run_waits_and_says_why(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "lint.lock")
            context = multiprocessing.get_context("fork")
            held = context.Event()
            release = context.Event()
            holder = context.Process(target=_hold_lock, args=(path, held, release))
            holder.start()
            try:
                self.assertTrue(held.wait(30), "the child never took the lock")
                output = io.StringIO()
                start = time.monotonic()
                with mock.patch.dict(
                    os.environ, {lint_code.LINT_LOCK_ENVIRONMENT: path}
                ):
                    with contextlib.redirect_stdout(output):
                        releaser = context.Process(
                            target=_release_later, args=(release,)
                        )
                        releaser.start()
                        with lint_code.lint_lock("/unused"):
                            waited = time.monotonic() - start
                        releaser.join(30)
                self.assertIn(
                    "waiting for the lint lock (another make lint is running)",
                    output.getvalue(),
                )
                self.assertGreaterEqual(waited, 0.4)
            finally:
                release.set()
                holder.join(30)

    def test_an_uncontended_lock_does_not_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {lint_code.LINT_LOCK_ENVIRONMENT: os.path.join(directory, "lint.lock")},
            ), contextlib.redirect_stdout(output):
                with lint_code.lint_lock("/unused"):
                    pass
            self.assertEqual(output.getvalue(), "")


def _release_later(release: object) -> None:
    time.sleep(0.5)
    release.set()  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
