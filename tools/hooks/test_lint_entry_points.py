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

"""The hooks run as modules from the repository root: `python3 -m tools.hooks.<module>`, with no sys.path edits.

The Makefile, the pre-commit hook and CI call them that way, so every module imports its siblings as
`from tools.hooks import ...` (AGENTS.md, Python: imports).
"""

import os
import subprocess
import sys
import unittest

from tools.hooks import check_test_support
from tools.hooks import python_imports


def _repository_root() -> str:
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                check_test_support.repository_file("tools/hooks/lint_code.py")
            )
        )
    )


def _run_module(*args: str) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=_repository_root(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


class EntryPointsTest(unittest.TestCase):
    def test_the_linter(self):
        result = _run_module("tools.hooks.lint_code", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--git-staged", "--only", "--paths", "--fix", "--summary"):
            self.assertIn(flag, result.stdout)

    def test_the_formatter(self):
        result = _run_module("tools.hooks.format_code", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_ifttt_check(self):
        result = _run_module("tools.hooks.check_ifttt", "tools/hooks/no_such_file.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_no_hook_edits_sys_path(self):
        hooks = os.path.join(_repository_root(), "tools", "hooks")
        for name in sorted(os.listdir(hooks)):
            # Bazel's runfiles also hold the generated `_<test>_stage2_bootstrap.py` files.
            if not name.endswith(".py") or name.startswith(("test_", "_")):
                continue
            with open(os.path.join(hooks, name), encoding="utf-8") as f:
                source = f.read()
            with self.subTest(module=name):
                self.assertEqual(python_imports.check_sys_path(source, name), [])

    def test_the_callers_use_the_module_form(self):
        makefile = check_test_support.read_repository_file("Makefile")
        self.assertIn("python3 -m tools.hooks.format_code", makefile)
        self.assertIn("python3 -m tools.hooks.lint_code", makefile)
        workflow = check_test_support.read_repository_file(
            ".github/workflows/format_test.yml"
        )
        self.assertIn("python3 -m tools.hooks.lint_code", workflow)
        for text in (
            makefile,
            workflow,
            check_test_support.read_repository_file("tools/hooks/pre-commit"),
        ):
            self.assertNotIn("python3 tools/hooks/", text)


if __name__ == "__main__":
    unittest.main()
