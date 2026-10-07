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

"""Tests for license_header.py: the BSD-3 license block that starts every first-party C++ file."""

import unittest

from tools.hooks import check_test_support
from tools.hooks import license_header

_CONDITIONS = """
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
_HOLDER = "Copyright (c) 2026, Nicholas Palomo. All rights reserved."


def _block(holder: str = _HOLDER, conditions: str = _CONDITIONS) -> str:
    rule = "*" * 78
    return f"/{rule}\n{holder}\n{conditions}{rule}/\n\n#pragma once\n"


def _messages(source: str) -> list[str]:
    return [f.message for f in license_header.check_source(source, "src/a.h")]


class LicenseHeaderTest(unittest.TestCase):
    def test_the_complete_block_is_clean(self):
        self.assertEqual(_messages(_block()), [])
        two_holders = (
            "Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.\n"
            "Copyright (c) 2024, 1X Technologies. All rights reserved."
        )
        self.assertEqual(_messages(_block(holder=two_holders)), [])
        self.assertEqual(
            _messages(
                _block(
                    holder="Copyright (c) 2024-2026, Nicholas Palomo & Manuel Yves Galliker. All rights reserved."
                )
            ),
            [],
        )

    def test_line_comments_wrap_the_same_text(self):
        lines = [
            f"// {line}".rstrip()
            for line in (_HOLDER + "\n" + _CONDITIONS).splitlines()
        ]
        self.assertEqual(_messages("\n".join(lines) + "\n\n#pragma once\n"), [])

    def test_a_line_only_header(self):
        messages = _messages(
            f"/{'*' * 78}\n{_HOLDER}\n{'*' * 78}/\n\n#include <cstdlib>\n"
        )
        self.assertEqual(len(messages), 1, messages)
        self.assertIn("lacks the preamble, the source-code condition", messages[0])
        self.assertIn("the disclaimer", messages[0])

    def test_a_holderless_line(self):
        for holder in [
            "Copyright (c) 2025. All rights reserved.",
            "Copyright (c) 2024, Unitree R1 Centroidal MPC",
        ]:
            with self.subTest(holder=holder):
                messages = _messages(_block(holder=holder))
                self.assertEqual(len(messages), 1, messages)
                self.assertIn("name the holder", messages[0])

    def test_a_missing_condition(self):
        two_clause = _CONDITIONS.replace(
            "* Neither the name of the copyright holder nor the names of its\n"
            "  contributors may be used to endorse or promote products derived from\n"
            "  this software without specific prior written permission.\n",
            "",
        )
        messages = _messages(_block(conditions=two_clause))
        self.assertEqual(len(messages), 1, messages)
        self.assertIn("lacks the endorsement condition", messages[0])
        no_disclaimer = _CONDITIONS.split("THIS SOFTWARE", maxsplit=1)[0]
        self.assertIn(
            "lacks the disclaimer", _messages(_block(conditions=no_disclaimer))[0]
        )

    def test_no_header(self):
        self.assertIn("no license header", _messages("#pragma once\nint x;\n")[0])
        self.assertIn(
            "no license header", _messages("// Just a comment.\n#pragma once\n")[0]
        )
        self.assertEqual(_messages(""), [])

    def test_a_block_that_follows_code_is_not_the_header(self):
        self.assertIn("no license header", _messages("#pragma once\n" + _block())[0])

    def test_generated_files_are_exempt(self):
        self.assertEqual(
            _messages(
                "// Generated by nproto (tools/nproto/README.md) from a.proto. Do not edit.\n\n#pragma once\n"
            ),
            [],
        )

    def test_the_text_is_that_of_the_license_file(self):
        text = " ".join(
            check_test_support.read_repository_file(license_header.LICENSE_FILE).split()
        )
        self.assertIn("BSD 3-Clause License", text)
        self.assertIn(license_header.PREAMBLE, text)
        for _, condition in license_header.CONDITIONS:
            self.assertIn(condition, text)
        self.assertIn(license_header.DISCLAIMER, text)


class WiringTest(unittest.TestCase):
    def test_the_registry_runs_it_on_cpp_files(self):
        check_test_support.assert_check_behaves(
            self,
            "license-header",
            "#pragma once\n",
            "src/a.h",
            clean=_block(),
        )
        check = check_test_support.assert_registered(self, "license-header")
        self.assertFalse(check.applies_to("lib/ocs2/core/include/ocs2_core/Types.h"))
        self.assertFalse(check.applies_to("tools/hooks/checks.py"))


if __name__ == "__main__":
    unittest.main()
