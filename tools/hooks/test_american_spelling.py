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

"""Tests for american_spelling.py: the repository is written in American English."""

import unittest

from tools.hooks import american_spelling
from tools.hooks import check_test_support


def fixed(text):
    return american_spelling.fix_source(text)[0]


class AmericanSpellingTest(unittest.TestCase):
    def test_the_families_of_british_spelling(self):
        cases = {
            "colour": "color",
            "behavioural": "behavioral",
            "favourite": "favorite",
            "initialise": "initialize",
            "normalisation": "normalization",
            "rationalise": "rationalize",
            "analysed": "analyzed",
            "centre": "center",
            "millimetres": "millimeters",
            "modelling": "modeling",
            "labelled": "labeled",
            "cancelled": "canceled",
            "defence": "defense",
            "grey": "gray",
            "whilst": "while",
            "judgement": "judgment",
            "uninitialised": "uninitialized",
        }
        for british, american in cases.items():
            with self.subTest(british=british):
                self.assertEqual(fixed(british), american)

    def test_identifiers_are_fixed_part_by_part_with_their_case(self):
        self.assertEqual(fixed("normalisedTime"), "normalizedTime")
        self.assertEqual(fixed("kCentreOfMass"), "kCenterOfMass")
        self.assertEqual(
            fixed("HeadingRelinearisationStage"), "HeadingRelinearizationStage"
        )
        self.assertEqual(fixed("heading_relinearisation"), "heading_relinearization")
        self.assertEqual(fixed("MPCCentre COLOUR Colour"), "MPCCenter COLOR Color")

    def test_american_words_that_look_british_are_left_alone(self):
        # -ise and -our words that are American too, and "analyses", the plural of "analysis".
        text = "advise exercise precise promise noise otherwise four hour your tour source course analyses"
        self.assertEqual(fixed(text), text)
        self.assertEqual(american_spelling.check_source(text), [])

    def test_urls_and_marked_lines_are_left_alone(self):
        self.assertEqual(
            fixed("see https://example.org/colour/centre"),
            "see https://example.org/colour/centre",
        )
        line = "Title: Optimisation of Colour  # NOLINT(american-spelling)"
        self.assertEqual(fixed(line), line)
        self.assertEqual(american_spelling.check_source(line), [])

    def test_findings_name_the_line_and_the_fix(self):
        findings = american_spelling.check_source("ok\nthe centre of mass\n", "a.cpp")
        self.assertEqual(
            [(f.line, f.british, f.american) for f in findings],
            [(2, "centre", "center")],
        )
        self.assertIn("write `center`", str(findings[0]))

    def test_the_linter_runs_it(self):
        check_test_support.assert_check_behaves(
            self, "american-spelling", "x = 1\n# the centre of mass\n", "src/a.py"
        )
        check_test_support.assert_check_behaves(
            self, "american-spelling", "int x;\n// the centre of mass\n", "src/a.cpp"
        )
        check_test_support.assert_registered(self, "american-spelling")
        # Robot model files are upstream data, and the checker's own word list is British on purpose.
        self.assertEqual(
            check_test_support.findings(
                "american-spelling", "<centre/>\n", "robot_models/x.urdf"
            ),
            [],
        )
        self.assertEqual(
            check_test_support.findings(
                "american-spelling", "centre\n", "tools/hooks/american_spelling.py"
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
