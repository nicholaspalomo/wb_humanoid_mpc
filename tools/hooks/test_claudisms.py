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

"""Tests for claudisms.py: no em dashes, emojis, `---` dividers, `**Term**:` bullets or banned phrases."""

import unittest

from tools.hooks import check_test_support
from tools.hooks import claudisms


def _messages(source: str, path: str = "src/a.cpp") -> list[str]:
    return [f.message for f in claudisms.check_source(source, path)]


class ClaudismsTest(unittest.TestCase):
    def test_em_dashes_and_emojis(self):
        findings = _messages("// First — second ✅\n")
        self.assertEqual(len(findings), 2)
        self.assertIn("em dash `\u2014`", findings[0])
        self.assertIn("emoji `✅`", findings[1])
        # Plain ASCII-like symbols and spaced hyphens are allowed.
        self.assertEqual(_messages("// First - second ✓ ▲ ▼\n"), [])

    def test_markdown_horizontal_rules_and_bold_colon_lists(self):
        md = (
            "---\n"
            "name: template\n"
            "---\n"
            "\n"
            "# Heading\n"
            "\n"
            "- **First**: bad bullet.\n"
            "- **Second:** also bad.\n"
            "- **Third.** Period inside bold is also a bold-lead bullet.\n"
            "- **Fourth** - spaced-hyphen lead is also a bold-lead bullet.\n"
            "- **Fifth** is the subject of the sentence.\n"
            "\n"
            "---\n"
            "\n"
            "```yaml\n"
            "---\n"
            "- **InCodeBlock**: ignored.\n"
            "```\n"
        )
        findings = claudisms.check_source(md, "docs/README.md")
        self.assertEqual([f.line for f in findings], [7, 8, 9, 10, 13])

    def test_banned_phrases_and_allowed_idioms(self):
        bad = (
            "/* This is a load-bearing assumption worth\n"
            "   noting. */\n"
            "// That's the whole point, and it quietly fails.\n"
            "// Leveraging a robust solver to dive into the realm.\n"
            "// Settled by the owner; there is a real distinction here.\n"
        )
        self.assertEqual(len(claudisms.check_source(bad, "src/a.cpp")), 10)
        good = (
            "// Drains a few polls' worth of samples (a queue's worth per"
            " call).\n"
            "// The stance foot carries the weight while the filter settles.\n"
            "// Hyphenated compounds and whole-body kinematics have no"
            " underscore.\n"
        )
        self.assertEqual(_messages(good), [])

    def test_model_files_and_exempt_files(self):
        self.assertEqual(
            _messages("<link name='a—b'/>\n", "robot_models/x.urdf"), []
        )
        self.assertEqual(_messages("— ✅\n", "AGENTS.md"), [])
        self.assertEqual(_messages("— ✅\n", "tools/hooks/claudisms.py"), [])

    def test_the_registry_runs_it(self):
        check_test_support.assert_check_behaves(
            self,
            claudisms.NAME,
            "int x;\n// a load-bearing invariant\n",
            "src/a.cpp",
        )
        check_test_support.assert_check_behaves(
            self,
            claudisms.NAME,
            "Some text.\n\nA load-bearing sentence.\n",
            "docs/README.md",
        )


if __name__ == "__main__":
    unittest.main()
