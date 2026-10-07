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

"""Tests for python_docstrings.py: license comments, docstring summaries and sections."""

import unittest

from tools.hooks import check_test_support
from tools.hooks import python_docstrings as docstrings

# This file's own license block: the complete one, as every Python file carries it.
with open(__file__, encoding="utf-8") as _self:
    LICENSE = "".join(line for line in _self.readlines()[:26] if line.startswith("#"))


def _count(check, source: str) -> int:
    return len(check(source, "src/a.py"))


class LicenseTest(unittest.TestCase):
    def test_the_license_is_a_comment(self):
        self.assertEqual(
            _count(
                docstrings.check_license_docstring,
                '"""Copyright (c) 2026, x. All rights."""\n',
            ),
            1,
        )
        self.assertEqual(
            _count(docstrings.check_license_docstring, LICENSE + '"""The module."""\n'),
            0,
        )

    def test_every_file_has_the_header(self):
        self.assertEqual(
            _count(docstrings.check_license_header, '"""The module."""\n'), 1
        )
        self.assertEqual(
            _count(docstrings.check_license_header, LICENSE + '"""The module."""\n'), 0
        )
        self.assertEqual(
            _count(
                docstrings.check_license_header, "#!/usr/bin/env python3\n" + LICENSE
            ),
            0,
        )
        self.assertEqual(
            _count(docstrings.check_license_header, ""),
            0,
            "an empty __init__.py needs none",
        )

    def test_the_whole_block_is_read(self):
        holder_only = (
            "# Copyright (c) 2026, Nicholas Palomo. All rights reserved.\n\n"
            + '"""Doc."""\n'
        )
        findings = docstrings.check_license_header(holder_only, "src/a.py")
        self.assertEqual(len(findings), 1, findings)
        self.assertIn("lacks the preamble", findings[0].message)
        holderless = LICENSE.replace(", Nicholas Palomo.", ".", 1) + '"""Doc."""\n'
        findings = docstrings.check_license_header(holderless, "src/a.py")
        self.assertEqual(len(findings), 1, findings)
        self.assertIn("name the holder", findings[0].message)
        self.assertEqual(findings[0].line, 1)
        two_clause = (
            LICENSE.replace("# * Neither the name of the copyright holder", "# * Nor")
            + '"""Doc."""\n'
        )
        self.assertIn(
            "the endorsement condition",
            docstrings.check_license_header(two_clause, "src/a.py")[0].message,
        )


class SummaryTest(unittest.TestCase):
    def test_flagged(self):
        for source in [
            '"""No period"""\n',
            '"""\nOn the next line.\n"""\n',
            '"""A summary\ncontinued on a second line.\n"""\n',
            'def f():\n    """Two lines.\n    No blank line."""\n',
            '"""' + "x" * 140 + '."""\n',
        ]:
            with self.subTest(source=source):
                self.assertEqual(_count(docstrings.check_docstring_summary, source), 1)

    def test_accepted(self):
        for source in [
            '"""One line."""\n',
            '"""Is it a question?"""\n',
            '"""A summary.\n\nMore text.\n"""\n',
            "def f():\n    pass\n",
        ]:
            with self.subTest(source=source):
                self.assertEqual(_count(docstrings.check_docstring_summary, source), 0)


def _function(docstring: str, body: str, parameters: str = "x") -> str:
    padding = "".join("    y = 1\n" for _ in range(docstrings.DOCSTRING_MIN_LENGTH))
    return f'def f({parameters}):\n    """{docstring}"""\n{padding}{body}'


class SectionsTest(unittest.TestCase):
    def test_flagged(self):
        self.assertEqual(
            _count(
                docstrings.check_docstring_sections,
                _function("Does it.\n\n    Details.\n    ", "    return 1\n"),
            ),
            1,
        )
        self.assertIn(
            "Args: or Returns:",
            docstrings.check_docstring_sections(
                _function("Does it.\n\n    Details.\n    ", "    return 1\n"), "a.py"
            )[0].message,
        )
        self.assertIn(
            "Yields:",
            docstrings.check_docstring_sections(
                _function("Does it.\n\n    Details.\n    ", "    yield 1\n", ""), "a.py"
            )[0].message,
        )

    def test_accepted(self):
        for source in [
            _function("Does it.", "    return 1\n"),
            _function(
                "Does it.\n\n    Args:\n      x: The x.\n\n    Returns:\n      One.\n    ",
                "    return 1\n",
            ),
            _function(
                "Returns one.\n\n    Args:\n      x: The x.\n    ", "    return 1\n"
            ),
            _function("Does it.\n\n    Details.\n    ", "    pass\n", ""),
            _function(
                "Does it.\n\n    Details.\n    ", "    def g():\n        return 1\n", ""
            ),
            'def f(x):\n    """Does it.\n\n    Details.\n    """\n    return 1\n',
        ]:
            with self.subTest(source=source):
                self.assertEqual(_count(docstrings.check_docstring_sections, source), 0)


class PropertyTest(unittest.TestCase):
    def test_a_property_describes_the_attribute(self):
        source = 'class A:\n    @property\n    def name(self):\n        """Returns the name."""\n        return 1\n'
        self.assertEqual(_count(docstrings.check_property_docstring, source), 1)
        self.assertEqual(
            _count(
                docstrings.check_property_docstring,
                source.replace("Returns the name.", "The name."),
            ),
            0,
        )


class RegistryTest(unittest.TestCase):
    def test_every_check_behaves(self):
        check_test_support.assert_check_behaves(
            self,
            "py-license-docstring",
            '"""Copyright (c) 2026, x."""\n',
            "src/a.py",
        )
        # The header finding is always on line 1, so a NOLINTNEXTLINE cannot reach it: NOLINT is checked by hand.
        check_test_support.assert_registered(self, "py-license-header")
        self.assertEqual(
            len(
                check_test_support.findings("py-license-header", "x = 1\n", "src/a.py")
            ),
            1,
        )
        self.assertEqual(
            check_test_support.findings(
                "py-license-header",
                "x = 1  # NOLINT(py-license-header): a test\n",
                "src/a.py",
            ),
            [],
        )
        check_test_support.assert_check_behaves(
            self,
            "py-docstring-summary",
            LICENSE + '"""No period"""\n',
            "src/a.py",
        )
        check_test_support.assert_check_behaves(
            self,
            "py-docstring-sections",
            _function("Does it.\n\n    Details.\n    ", "    pass\n"),
            "src/a.py",
        )
        check_test_support.assert_check_behaves(
            self,
            "py-property-docstring",
            'class A:\n    @property\n    def name(self):\n        """Returns the name."""\n        return 1\n',
            "src/a.py",
        )


if __name__ == "__main__":
    unittest.main()
