"""Tests for american_spelling.py: the repository is written in American English."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import american_spelling  # noqa: E402

from tools.hooks import check_test_support  # noqa: E402


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
        check = check_test_support.assert_registered(self, "american-spelling")
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
