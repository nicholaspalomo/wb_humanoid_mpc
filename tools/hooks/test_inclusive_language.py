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

"""Tests for inclusive_language.py: no master/slave, blacklist/whitelist or redline."""

import unittest

from tools.hooks import check_test_support
from tools.hooks import inclusive_language


def _words(source: str, path: str = "src/a.cpp") -> list[str]:
    return [
        f.message.split("`")[1] for f in inclusive_language.check_source(source, path)
    ]


class InclusiveLanguageTest(unittest.TestCase):
    def test_words_and_identifier_parts(self):
        self.assertEqual(
            _words(
                "// 4. Test master switch\nint kWhitelist = blacklisted_ids;\nvoid redlineCheck();\n"
            ),
            ["master", "Whitelist", "blacklisted", "redline"],
        )
        self.assertEqual(
            _words("def build(master, slave):\n", "a.py"), ["master", "slave"]
        )

    def test_other_words_and_urls(self):
        self.assertEqual(_words("mastery masterpiece remaster\n"), [])
        self.assertEqual(
            _words("# see https://github.com/git/git/blob/master/README.md\n", "a.sh"),
            [],
        )

    def test_model_files_and_the_checker_itself(self):
        self.assertEqual(_words("<joint name='master'/>\n", "robot_models/x.urdf"), [])
        self.assertEqual(_words("master\n", "tools/hooks/inclusive_language.py"), [])

    def test_the_registry_runs_it(self):
        check_test_support.assert_check_behaves(
            self,
            "inclusive-language",
            "int x;\n// the master switch\n",
            "src/a.cpp",
        )
        check_test_support.assert_check_behaves(
            self,
            "inclusive-language",
            "Some text.\n\nThe master scale.\n",
            "docs/README.md",
        )


if __name__ == "__main__":
    unittest.main()
