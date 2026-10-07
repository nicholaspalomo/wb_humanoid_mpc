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

"""The lint tools' configurations work with the real tools: cpplint, pylint, mypy, isort and black from the lock.

Bazel runs this with the @lint_deps hub (tools/hooks/lint_requirements_lock.txt), so it needs neither the dev image nor
CI's installs: each tool runs as `python -m <tool>` on a fixture in a temporary directory, with the repository's
configuration file.
"""

import configparser
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from tools.hooks import check_test_support
from tools.hooks import lint_code

COLUMN_LIMIT = 140


def _run(module: str, *args: str, cwd: str) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def _clang_format_column_limit() -> int:
    match = re.search(
        r"^\s*ColumnLimit:\s*(\d+)",
        check_test_support.read_repository_file(".clang-format"),
        re.MULTILINE,
    )
    assert match is not None
    return int(match.group(1))


class FixtureDirectoryTest(unittest.TestCase):
    """A temporary directory with copies of the repository's configuration files."""

    def setUp(self):
        # pylint: disable-next=consider-using-with  # tearDown() deletes it.
        self.directory = tempfile.TemporaryDirectory()
        self.root = self.directory.name
        for config in ("CPPLINT.cfg", ".pylintrc", "mypy.ini", ".isort.cfg"):
            shutil.copy(
                check_test_support.repository_file(config),
                os.path.join(self.root, config),
            )

    def tearDown(self):
        self.directory.cleanup()

    def write(self, name: str, text: str) -> str:
        """Writes a fixture file and returns its name."""
        with open(os.path.join(self.root, name), "w", encoding="utf-8") as f:
            f.write(text)
        return name


class CpplintTest(FixtureDirectoryTest):
    def test_the_line_length_is_the_column_limit(self):
        config = check_test_support.read_repository_file("CPPLINT.cfg")
        match = re.search(r"^linelength=(\d+)$", config, re.MULTILINE)
        assert match is not None
        self.assertEqual(int(match.group(1)), _clang_format_column_limit())
        self.assertEqual(_clang_format_column_limit(), COLUMN_LIMIT)
        self.assertIn("set noparent", config)

    def test_the_configuration_applies(self):
        name = self.write(
            "a.cpp",
            "// Copyright (c) 2026, Nicholas Palomo. All rights reserved.\nlong x;\nvoid f() {\n  if (x) {\n  }\n"
            "  while (y);\n}\n" + "int z = 1;  // " + "x" * 120 + "\n",
        )
        result = _run("cpplint", "--quiet", name, cwd=self.root)
        output = result.stdout + result.stderr
        # The empty-body checks are the only whitespace checks left on (the filter order of CPPLINT.cfg).
        self.assertIn("[whitespace/empty_if_body]", output)
        self.assertIn("[whitespace/empty_loop_body]", output)
        self.assertIn("[runtime/int]", output)
        self.assertNotIn("[whitespace/line_length]", output)

    def test_markers_for_the_repositorys_checks_are_filtered(self):
        name = self.write(
            "b.cpp",
            "// Copyright (c) 2026, Nicholas Palomo. All rights reserved.\n"
            "int x = f(a, 0);  // NOLINT(argument-comment): a test\n"
            "int y = g(a, 0);  // NOLINT(argumnet-comment): a typo\n",
        )
        result = _run("cpplint", "--quiet", name, cwd=self.root)
        lines = lint_code.filter_cpplint_output(result.stdout + result.stderr)
        self.assertEqual(
            len([line for line in lines if "readability/nolint" in line]), 1, lines
        )
        self.assertIn("argumnet-comment", "\n".join(lines))


class PylintTest(FixtureDirectoryTest):
    def test_the_line_length_is_the_column_limit(self):
        parser = configparser.ConfigParser()
        parser.read(check_test_support.repository_file(".pylintrc"), encoding="utf-8")
        self.assertEqual(
            parser.getint("FORMAT", "max-line-length"), _clang_format_column_limit()
        )
        self.assertEqual(
            parser.get("MAIN", "jobs"),
            "1",
            "pylint takes one process (AGENTS.md, memory)",
        )

    def test_the_configuration_loads_and_applies(self):
        name = self.write(
            "module.py",
            '"""A module."""\n\n\ndef risky() -> None:\n    """Fails."""\n    try:\n        pass\n    except:\n        pass\n',
        )
        result = _run("pylint", "--rcfile=.pylintrc", name, cwd=self.root)
        output = result.stdout + result.stderr
        for problem in (
            "unknown-option-value",
            "useless-option-value",
            "bad-option-value",
            "unrecognized-option",
        ):
            self.assertNotIn(problem, output)
        self.assertIn("[bare-except]", output)
        self.assertRegex(
            output, r"module\.py:\d+:\d+: .* \[bare-except\]", "msg-template"
        )

    def test_negated_comparisons_are_kept(self):
        # `not value >= 0.0` rejects NaN; pylint's suggested `value < 0.0` would not.
        name = self.write(
            "nan.py",
            '"""A module."""\n\n\ndef valid(value: float) -> bool:\n    """Validates."""\n    return not value >= 0.0\n',
        )
        result = _run("pylint", "--rcfile=.pylintrc", name, cwd=self.root)
        self.assertNotIn("unnecessary-negation", result.stdout + result.stderr)
        self.assertTrue(valid_rejects_nan())


def valid_rejects_nan() -> bool:
    """The reason for the permanent disable: the not-form is False for NaN, the rewritten form would be True."""
    nan = float("nan")
    return (not nan >= 0.0) and not nan < 0.0


class MypyTest(FixtureDirectoryTest):
    def test_the_configuration_loads_and_applies(self):
        self.write("source.py", "def untyped(x):\n    return x\n")
        result = _run(
            "mypy",
            "--config-file=mypy.ini",
            "--cache-dir=.cache",
            "source.py",
            cwd=self.root,
        )
        self.assertIn("[no-untyped-def]", result.stdout, result.stdout + result.stderr)
        self.assertNotIn("error: Unrecognized", result.stdout + result.stderr)
        self.write("typed.py", "def typed(x: int) -> int:\n    return x\n")
        result = _run(
            "mypy",
            "--config-file=mypy.ini",
            "--cache-dir=.cache",
            "typed.py",
            cwd=self.root,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_tests_may_be_untyped(self):
        self.write("test_source.py", "def test_it(x):\n    return x\n")
        result = _run(
            "mypy",
            "--config-file=mypy.ini",
            "--cache-dir=.cache",
            "--allow-untyped-defs",
            "--allow-incomplete-defs",
            "test_source.py",
            cwd=self.root,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_the_global_options_hold_every_module(self):
        parser = configparser.ConfigParser()
        parser.read(check_test_support.repository_file("mypy.ini"), encoding="utf-8")
        self.assertEqual(parser.get("mypy", "python_version"), "3.11")
        self.assertTrue(parser.getboolean("mypy", "disallow_untyped_defs"))
        self.assertTrue(parser.getboolean("mypy", "disallow_incomplete_defs"))
        self.assertEqual(
            [section for section in parser.sections() if section.startswith("mypy-")],
            [],
            "a per-module section relaxes mypy for one module (lint_code.mypy_config_problems)",
        )


class IsortAndBlackTest(FixtureDirectoryTest):
    def test_they_reach_a_fixed_point_together(self):
        source = (
            '"""A module."""\n\nimport sys\nfrom tools.hooks import lint_files\nimport os\n'
            "from collections.abc import Callable, Iterator, Sequence, Mapping, MutableMapping, Iterable, Generator\n"
            "import numpy as np\n\n"
            "print(os, sys, np, lint_files, Callable, Iterator, Sequence, Mapping, MutableMapping, Iterable, Generator)\n"
        )
        name = self.write("imports.py", source)
        isort = ["isort", "--settings-path", ".isort.cfg", "-p", "tools"]
        for _ in range(2):
            self.assertEqual(_run(*isort, name, cwd=self.root).returncode, 0)
            self.assertEqual(
                _run("black", "--quiet", name, cwd=self.root).returncode, 0
            )
        with open(os.path.join(self.root, name), encoding="utf-8") as f:
            formatted = f.read()
        self.assertEqual(
            _run(*isort, "--check-only", name, cwd=self.root).returncode, 0, formatted
        )
        self.assertEqual(
            _run("black", "--check", "--quiet", name, cwd=self.root).returncode,
            0,
            formatted,
        )
        # Google order: the standard library, then third-party, then first-party, each its own block.
        self.assertLess(formatted.index("import os"), formatted.index("import numpy"))
        self.assertLess(
            formatted.index("import numpy"),
            formatted.index("from tools.hooks import lint_files"),
        )


if __name__ == "__main__":
    unittest.main()
