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


"""Tests for textproto_headers.py: every .textproto names its schema, and the schema exists."""

import os
import tempfile
import unittest

from tools.hooks import check_test_support
from tools.hooks import textproto_headers

PROTO = """edition = "2023";

package my_config;

import "nproto/options.proto";

// Next ID: 1
message MyFile {
  option (nproto.generate_struct) = "my::config::MyFile";

  reserved 1 to max;
}
"""
HEADER = "# proto-file: config/my_file.proto\n# proto-message: my_config.MyFile\n"


class TextprotoHeaderTest(unittest.TestCase):
    def setUp(self):
        # pylint: disable-next=consider-using-with  # enterContext() deletes it after the test.
        self.root = self.enterContext(tempfile.TemporaryDirectory())
        os.makedirs(os.path.join(self.root, "config"))
        with open(
            os.path.join(self.root, "config", "my_file.proto"), "w", encoding="utf-8"
        ) as f:
            f.write(PROTO)

    def texts(self, source, path="config/my.textproto"):
        return [
            (v.line, v.text)
            for v in textproto_headers.check_header(source, path, self.root)
        ]

    def test_a_file_with_its_header_passes(self):
        self.assertEqual(self.texts(HEADER + "#\n# What it is.\n\nvalue: 1\n"), [])
        # Header lines further down the leading comment block count too.
        self.assertEqual(self.texts("# A title.\n" + HEADER), [])

    def test_a_missing_header(self):
        found = self.texts("value: 1\n")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0][0], 1)
        self.assertIn("no # proto-file: and no # proto-message: line", found[0][1])
        self.assertIn(
            "e.g. '# proto-file: humanoid_nmpc/humanoid_mpc_config/task_file.proto'",
            found[0][1],
        )

    def test_a_missing_line_is_named(self):
        found = self.texts("# proto-file: config/my_file.proto\nvalue: 1\n")
        self.assertEqual(len(found), 1)
        self.assertIn("no # proto-message: line", found[0][1])
        self.assertNotIn("proto-file: line", found[0][1])

    def test_header_lines_after_the_first_field_do_not_count(self):
        found = self.texts("value: 1\n" + HEADER)
        self.assertEqual([line for line, _ in found], [1])

    def test_a_proto_file_that_does_not_exist(self):
        found = self.texts(
            "# proto-file: config/gone.proto\n# proto-message: my_config.MyFile\n"
        )
        self.assertEqual(
            found,
            [
                (
                    1,
                    "'# proto-file: config/gone.proto' names no file of the repository (the path is from its root)",
                )
            ],
        )

    def test_a_message_the_file_does_not_define(self):
        found = self.texts(
            "# proto-file: config/my_file.proto\n# proto-message: my_config.Other\n"
        )
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0][0], 2)
        self.assertIn(
            "not a top-level message of config/my_file.proto, which defines my_config.MyFile",
            found[0][1],
        )
        # The package is part of the name.
        self.assertEqual(
            len(
                self.texts(
                    "# proto-file: config/my_file.proto\n# proto-message: MyFile\n"
                )
            ),
            1,
        )

    def test_other_files_are_not_read(self):
        self.assertEqual(self.texts("value: 1\n", path="config/my.txt"), [])

    def test_schema_messages_reads_the_package(self):
        self.assertEqual(textproto_headers.schema_messages(PROTO), ["my_config.MyFile"])
        self.assertEqual(textproto_headers.schema_messages("message A {}\n"), ["A"])


class ShippedFilesTest(unittest.TestCase):
    def test_the_repository_proto_files_resolve(self):
        # A header of the repository, checked against the repository root.
        source = check_test_support.read_repository_file(
            "tools/nproto/test/testdata/scalars.textproto"
        )
        self.assertEqual(
            textproto_headers.check_header(
                source,
                "tools/nproto/test/testdata/scalars.textproto",
                check_test_support.source_tree_root(),
            ),
            [],
        )


class WiringTest(unittest.TestCase):
    def test_the_pre_commit_hook_and_the_linter_run_the_check(self):
        check_test_support.assert_hook_runs_the_linter(self)
        check_test_support.assert_check_behaves(
            self,
            textproto_headers.NAME,
            "# proto-file: config/no_such_schema.proto\n# proto-message: my_config.MyFile\n\nvalue: 1\n",
            "config/my.textproto",
        )


if __name__ == "__main__":
    unittest.main()
