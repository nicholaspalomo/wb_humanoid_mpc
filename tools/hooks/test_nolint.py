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

"""Tests for nolint.py: NOLINT markers are read from comments, need a reason, and are reported when wrong."""

import unittest

from tools.hooks import nolint

_MARKED = "Title: Colour  NOLINT(american-spelling): a title\n"  # NOLINT(american-spelling): the test's input


def _parse(source: str, style: str = nolint.CPP_COMMENTS) -> list[tuple]:
    return [
        (m.line, m.variant, m.categories, m.reason, m.bare)
        for m in nolint.parse(source, style)
    ]


class ParseTest(unittest.TestCase):
    def test_the_forms_of_a_marker(self):
        source = (
            "f(0);  // NOLINT(argument-comment): libstdc++ names it __i\n"
            "// NOLINTNEXTLINE(boost, no-auto): two checks\n"
            "/* NOLINTBEGIN(boost): a block */\n"
            "// NOLINTEND(boost)\n"
            "int x;  // NOLINT\n"
        )
        self.assertEqual(
            _parse(source),
            [
                (1, "", ("argument-comment",), "libstdc++ names it __i", False),
                (2, "NEXTLINE", ("boost", "no-auto"), "two checks", False),
                (3, "BEGIN", ("boost",), "a block", False),
                (4, "END", ("boost",), "", False),
                (5, "", (), "", True),
            ],
        )

    def test_markers_in_string_literals_are_text(self):
        source = 'const char* s = "NOLINT(boost): no";\nauto r = R"(// NOLINT(boost): no)";\n'
        self.assertEqual(_parse(source), [])
        python = 'x = "# NOLINT(boost): no"\ny = 1  # NOLINT(boost): yes\n'
        self.assertEqual([m[0] for m in _parse(python, nolint.PYTHON_COMMENTS)], [2])
        starlark = 'deps = ["# NOLINT(boost): no"],  # NOLINT(boost): yes\n'
        self.assertEqual(len(_parse(starlark, nolint.HASH_COMMENTS)), 1)

    def test_a_python_docstring_is_not_a_comment(self):
        source = '"""Write `# NOLINT(boost): reason` on the line.\n\nNOLINT(boost): also prose\n"""\n'
        self.assertEqual(_parse(source, nolint.PYTHON_COMMENTS), [])

    def test_a_bare_word_is_a_marker_in_cpp_only(self):
        self.assertEqual(len(_parse("int x;  // NOLINT\n")), 1)
        self.assertEqual(
            _parse("x = 1  # the NOLINT markers\n", nolint.PYTHON_COMMENTS), []
        )
        self.assertEqual(_parse("# NOLINTNEXTLINE markers\n", nolint.HASH_COMMENTS), [])

    def test_a_quoted_marker_is_prose(self):
        self.assertEqual(_parse("// write `NOLINT(boost): <reason>` on the line\n"), [])
        self.assertEqual(_parse("// write 'NOLINT(boost)' on the line\n"), [])

    def test_whole_line_markers(self):
        self.assertEqual(
            _parse(_MARKED, nolint.WHOLE_LINE)[0][2],
            ("american-spelling",),
        )

    def test_a_reason_needs_a_colon_and_text(self):
        self.assertEqual(_parse("x;  // NOLINT(boost) because\n")[0][3], "")
        self.assertEqual(_parse("x;  // NOLINT(boost):\n")[0][3], "")
        self.assertEqual(_parse("x;  /* NOLINT(boost): why */ y;\n")[0][3], "why")

    def test_unparseable_python_falls_back_to_hash_comments(self):
        self.assertEqual(
            len(
                _parse(
                    "def f(:\n  x = 1  # NOLINT(boost): why\n", nolint.PYTHON_COMMENTS
                )
            ),
            1,
        )


class SuppressionsTest(unittest.TestCase):
    def suppressions(self, source: str) -> nolint.Suppressions:
        return nolint.Suppressions(nolint.parse(source, nolint.CPP_COMMENTS))

    def test_nolint_and_nextline(self):
        s = self.suppressions(
            "a;  // NOLINT(boost): why\n// NOLINTNEXTLINE(no-auto): why\nb;\n"
        )
        self.assertTrue(s.suppresses("boost", 1))
        self.assertFalse(s.suppresses("boost", 2))
        self.assertTrue(s.suppresses("no-auto", 3))
        self.assertFalse(s.suppresses("no-auto", 2))
        self.assertFalse(s.suppresses("other", 1), "only the named check is suppressed")

    def test_a_block(self):
        s = self.suppressions(
            "// NOLINTBEGIN(boost): why\na;\nb;\n// NOLINTEND(boost)\nc;\n"
        )
        self.assertEqual(
            [line for line in range(1, 6) if s.suppresses("boost", line)], [1, 2, 3, 4]
        )
        self.assertEqual(s.unbalanced, [])

    def test_a_marker_without_a_reason_suppresses_nothing(self):
        s = self.suppressions(
            "a;  // NOLINT(boost)\n// NOLINTBEGIN(no-auto)\nb;\n// NOLINTEND(no-auto)\n"
        )
        self.assertFalse(s.suppresses("boost", 1))
        self.assertFalse(s.suppresses("no-auto", 3))

    def test_unbalanced_blocks(self):
        s = self.suppressions("// NOLINTBEGIN(boost): why\na;\n// NOLINTEND(no-auto)\n")
        self.assertEqual(
            sorted((p.check, p.line) for p in s.unbalanced),
            [("nolint-unbalanced", 1), ("nolint-unbalanced", 3)],
        )
        self.assertFalse(s.suppresses("boost", 2), "an unclosed block exempts nothing")

    def test_unused_markers(self):
        s = self.suppressions(
            "a;  // NOLINT(boost): why\nb;  // NOLINT(no-auto): why\n"
        )
        self.assertTrue(s.suppresses("boost", 1))
        self.assertEqual(
            [m.line for m in s.unused(frozenset({"boost", "no-auto"}))], [2]
        )
        self.assertEqual(s.unused(frozenset({"other"})), [])


class MarkerProblemsTest(unittest.TestCase):
    def problems(self, source: str) -> list[tuple[str, int]]:
        markers = nolint.parse(source, nolint.CPP_COMMENTS)
        return [
            (p.check, p.line)
            for p in nolint.marker_problems(
                markers, lambda c: c in {"boost", "runtime/int"}
            )
        ]

    def test_a_bare_marker(self):
        self.assertEqual(
            self.problems("a;  // NOLINT\nb;  // NOLINTNEXTLINE\n"),
            [("nolint-category", 1), ("nolint-category", 2)],
        )
        self.assertEqual(
            self.problems("a;  // NOLINT(): why\n"), [("nolint-category", 1)]
        )

    def test_a_missing_reason(self):
        self.assertEqual(
            self.problems("a;  // NOLINT(boost)\n"), [("nolint-reason", 1)]
        )
        # A NOLINTEND carries no reason of its own: its NOLINTBEGIN does.
        self.assertEqual(
            self.problems("// NOLINTBEGIN(boost): why\n// NOLINTEND(boost)\n"), []
        )

    def test_an_unknown_category(self):
        self.assertEqual(
            self.problems("a;  // NOLINT(runtime/int, bost): why\n"),
            [("nolint-unknown", 1)],
        )

    def test_a_good_marker(self):
        self.assertEqual(
            self.problems("long a;  // NOLINT(runtime/int): POSIX API\n"), []
        )


if __name__ == "__main__":
    unittest.main()
