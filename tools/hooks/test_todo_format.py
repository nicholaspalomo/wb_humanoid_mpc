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

"""Tests for todo_format.py: a TODO names a bug, a link or an owner."""

import unittest

from tools.hooks import check_test_support
from tools.hooks import todo_format


def _count(source: str, path: str = "src/a.cpp") -> int:
    return len(todo_format.check_source(source, path))


class TodoFormatTest(unittest.TestCase):
    def test_accepted_forms(self):
        for source in [
            "// TODO: bug 123 - Remove this.\n",
            "// TODO: #42 - Remove this.\n",
            "// TODO: b/42 - Remove this.\n",
            "// TODO: https://github.com/org/repo/issues/42 - Remove this.\n",
            "// TODO(npalomo): Remove this.\n",
            "// TODO(bug 123): Remove this.\n",
            "// Mentions TODOs in prose.\n",
            "// A `TODO` in backticks is prose.\n",
            'const char kText[] = "TODO fix";\n',
        ]:
            with self.subTest(source=source):
                self.assertEqual(_count(source), 0)

    def test_flagged_forms(self):
        for source in [
            "// TODO Update this reference.\n",
            "// TODO: Update this reference.\n",
            "// TODO(npalomo) no colon\n",
            "/* TODO: later */\n",
        ]:
            with self.subTest(source=source):
                self.assertEqual(_count(source), 1)

    def test_the_languages(self):
        self.assertEqual(_count("x = 1  # TODO later\n", "src/a.py"), 1)
        self.assertEqual(_count('x = "TODO later"\n', "src/a.py"), 0)
        self.assertEqual(_count("# TODO later\n", "src/BUILD.bazel"), 1)
        self.assertEqual(_count("# TODO later\n", "src/run.sh"), 1)
        self.assertEqual(_count("TODO later\n", "src/README.md"), 0)

    def test_the_registry_runs_it(self):
        check_test_support.assert_check_behaves(
            self,
            "todo-format",
            "int x;  // TODO later\n",
            "src/a.cpp",
            clean="int x;  // TODO(npalomo): later\n",
        )


if __name__ == "__main__":
    unittest.main()
