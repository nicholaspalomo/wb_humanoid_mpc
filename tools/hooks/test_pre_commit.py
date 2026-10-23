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

"""Tests for the pre-commit hook in a temporary repository: formatting is staged, but never a partial file's hunks.

`make` and `python3` are stand-ins on PATH: the formatter rewrites UNFORMATTED to FORMATTED in every tracked file, and
the linter passes, so what is under test is the hook's own staging.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

from tools.hooks import check_test_support

_FAKE_MAKE = """#!/bin/sh
[ "$1" = "format" ] || exit 2
for file in $(git ls-files); do
    sed -i 's/UNFORMATTED/FORMATTED/' "$file"
done
"""
_FAKE_PYTHON = "#!/bin/sh\nexit 0\n"


class PreCommitHookTest(unittest.TestCase):
    def setUp(self):
        # pylint: disable-next=consider-using-with  # tearDown() deletes it.
        self.directory = tempfile.TemporaryDirectory()
        self.repository = os.path.join(self.directory.name, "repo")
        self.bin = os.path.join(self.directory.name, "bin")
        os.makedirs(self.repository)
        os.makedirs(self.bin)
        for name, content in (("make", _FAKE_MAKE), ("python3", _FAKE_PYTHON)):
            path = os.path.join(self.bin, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            os.chmod(path, 0o755)
        self.hook = os.path.join(self.directory.name, "pre-commit")
        shutil.copy(
            check_test_support.repository_file("tools/hooks/pre-commit"), self.hook
        )
        os.chmod(self.hook, 0o755)
        self.environment = dict(os.environ)
        self.environment["PATH"] = self.bin + os.pathsep + os.environ.get("PATH", "")
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "Test")
        self.write("a.txt", "one\ntwo\nthree\n")
        self.git("add", "a.txt")
        self.git("commit", "-q", "-m", "base")

    def tearDown(self):
        self.directory.cleanup()

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=self.repository,
            env=self.environment,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    def write(self, name: str, content: str) -> None:
        with open(os.path.join(self.repository, name), "w", encoding="utf-8") as f:
            f.write(content)

    def run_hook(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", self.hook],
            cwd=self.repository,
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def staged(self, name: str) -> str:
        return self.git("show", f":{name}")

    def test_a_fully_staged_file_is_formatted_and_restaged(self):
        self.write("a.txt", "one UNFORMATTED\ntwo\nthree\n")
        self.git("add", "a.txt")
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.staged("a.txt"), "one FORMATTED\ntwo\nthree\n")

    def test_a_partially_staged_file_the_formatter_changes_blocks_the_commit(self):
        # The first line is staged; the change of the last line, which the formatter rewrites, is not.
        self.write("a.txt", "one staged\ntwo\nthree\n")
        self.git("add", "a.txt")
        self.write("a.txt", "one staged\ntwo\nthree UNFORMATTED\n")
        result = self.run_hook()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("partially staged", result.stdout)
        self.assertEqual(
            self.staged("a.txt"),
            "one staged\ntwo\nthree\n",
            "the unstaged hunk was staged",
        )

    def test_a_partially_staged_file_the_formatter_leaves_alone_keeps_its_index(self):
        self.write("a.txt", "one staged\ntwo\nthree\n")
        self.git("add", "a.txt")
        self.write("a.txt", "one staged\ntwo\nthree unstaged\n")
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            self.staged("a.txt"),
            "one staged\ntwo\nthree\n",
            "the unstaged hunk was staged",
        )


if __name__ == "__main__":
    unittest.main()
