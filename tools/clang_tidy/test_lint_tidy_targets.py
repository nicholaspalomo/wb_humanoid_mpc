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

"""Tests for the Makefile's clang-tidy targets (`make lint-tidy`, `make lint-tidy-fix`).

Each target builds with Bazel, which writes a build event file, and then runs a collector that reads it. The collector
runs after the build has released the machine lock. When every run used one fixed file, a run that started meanwhile
overwrote it, and the first run printed the second one's findings. These tests run the real recipes with stand-ins for
`bazel` and `python3` that record what they were given, and check that each run reads the file its own build wrote.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

# The targets, each with the variables it needs and the collector module it runs.
_TARGETS = (
    ("lint-tidy", (), "tools.clang_tidy.clang_tidy_report"),
    (
        "lint-tidy-fix",
        ("CHECKS=modernize-use-override",),
        "tools.clang_tidy.clang_tidy_apply",
    ),
)

# Records the build event file and writes into it a token the collector must see; exits with $FAKE_BAZEL_STATUS.
_FAKE_BAZEL = """#!/bin/sh
for argument in "$@"; do
  case "$argument" in
    --build_event_json_file=*)
      file="${argument#--build_event_json_file=}"
      token="build-$$"
      printf '%s\\n' "$token" > "$file"
      printf '%s %s\\n' "$file" "$token" >> "$FAKE_LOG/bazel"
      ;;
  esac
done
exit "${FAKE_BAZEL_STATUS:-0}"
"""

# Records `-m <module> <file>` and the content of the file it was given; exits with $FAKE_REPORT_STATUS.
_FAKE_PYTHON = """#!/bin/sh
module="$2"
file="$3"
content="$(cat "$file" 2>/dev/null || echo missing)"
printf '%s %s %s\\n' "$module" "$file" "$content" >> "$FAKE_LOG/python3"
exit "${FAKE_REPORT_STATUS:-0}"
"""


def _runfile(relative_path: str) -> str:
    """A data file of this test, found in the runfiles when run by Bazel and in the source tree otherwise."""
    roots = []
    if "TEST_SRCDIR" in os.environ:
        roots += [
            os.path.join(os.environ["TEST_SRCDIR"], "_main"),
            os.environ["TEST_SRCDIR"],
        ]
    roots.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    for root in roots:
        candidate = os.path.join(root, relative_path)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(relative_path)


def _write_executable(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(path, 0o755)


def _read_lines(path: str) -> list[list[str]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.split() for line in f if line.strip()]


@unittest.skipIf(shutil.which("make") is None, "no make")
class LintTidyTargetsTest(unittest.TestCase):
    """Runs the recipes in a scratch checkout: the Makefile, an empty setup_env.sh and the stand-ins on PATH."""

    def setUp(self) -> None:
        # pylint: disable-next=consider-using-with  # tearDown() deletes it.
        self._directory = tempfile.TemporaryDirectory()
        self.checkout = self._directory.name
        shutil.copy(_runfile("Makefile"), os.path.join(self.checkout, "Makefile"))
        _write_executable(os.path.join(self.checkout, "setup_env.sh"), "true\n")
        stand_ins = os.path.join(self.checkout, "bin")
        os.mkdir(stand_ins)
        _write_executable(os.path.join(stand_ins, "bazel"), _FAKE_BAZEL)
        _write_executable(os.path.join(stand_ins, "python3"), _FAKE_PYTHON)
        self.log = os.path.join(self.checkout, "log")
        os.mkdir(self.log)
        self.environment = {
            "PATH": stand_ins + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
            "FAKE_LOG": self.log,
            "HOME": self.checkout,
        }

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _clear_log(self) -> None:
        shutil.rmtree(self.log)
        os.mkdir(self.log)

    def _make(self, target: str, variables: tuple[str, ...], **environment: str) -> int:
        """Runs `make <target>` in the scratch checkout and returns its exit status."""
        completed = subprocess.run(
            ["make", "--no-print-directory", target, "PKG=//humanoid_nmpc/..."]
            + list(variables),
            cwd=self.checkout,
            env={**self.environment, **environment},
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.returncode

    def test_each_run_reads_the_file_its_own_build_wrote(self) -> None:
        for target, variables, module in _TARGETS:
            with self.subTest(target=target):
                self._clear_log()
                for _ in range(2):
                    self.assertEqual(self._make(target, variables), 0)
                builds = _read_lines(os.path.join(self.log, "bazel"))
                reports = _read_lines(os.path.join(self.log, "python3"))
                self.assertEqual(len(builds), 2)
                self.assertEqual(len(reports), 2)
                for (built, token), (reported_module, read, content) in zip(
                    builds, reports
                ):
                    self.assertEqual(reported_module, module)
                    self.assertEqual(read, built)
                    self.assertEqual(content, token)
                    self.assertTrue(read.startswith(".bazel/"), read)
                    # The file of a run is deleted afterwards, so that runs leave nothing behind.
                    self.assertFalse(os.path.exists(os.path.join(self.checkout, read)))
                self.assertNotEqual(
                    builds[0][0], builds[1][0], "two runs shared a build event file"
                )

    def test_a_named_file_is_used_and_kept(self) -> None:
        for target, variables, _ in _TARGETS:
            with self.subTest(target=target):
                kept = f".bazel/kept_{target}.json"
                self.assertEqual(
                    self._make(target, variables + (f"CLANG_TIDY_BEP={kept}",)), 0
                )
                builds = _read_lines(os.path.join(self.log, "bazel"))
                reports = _read_lines(os.path.join(self.log, "python3"))
                self.assertEqual(builds[-1][0], kept)
                self.assertEqual(reports[-1][1], kept)
                self.assertTrue(os.path.exists(os.path.join(self.checkout, kept)))

    def test_a_failure_of_either_step_fails_the_target(self) -> None:
        for target, variables, _ in _TARGETS:
            with self.subTest(target=target, failing="bazel"):
                reports_before = len(_read_lines(os.path.join(self.log, "python3")))
                self.assertNotEqual(
                    self._make(target, variables, FAKE_BAZEL_STATUS="1"), 0
                )
                # The collector still prints the findings of a build that failed.
                self.assertEqual(
                    len(_read_lines(os.path.join(self.log, "python3"))),
                    reports_before + 1,
                )
            with self.subTest(target=target, failing="collector"):
                self.assertNotEqual(
                    self._make(target, variables, FAKE_REPORT_STATUS="2"), 0
                )


if __name__ == "__main__":
    unittest.main()
