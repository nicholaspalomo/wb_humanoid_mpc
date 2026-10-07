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


"""The Python textproto parser: the inputs of //tools/nproto/test:test_textproto, answered as the C++ parser answers them.

Positions and the retired-field and header messages are the C++ parser's to the character; the wording of the other
problems is text_format's, so only their position is compared.
"""

import importlib
import os
import unittest

from google.protobuf import descriptor_pb2
from google.protobuf import descriptor_pool
from google.protobuf import message as message_module
from google.protobuf import message_factory
from google.protobuf import text_format

import nproto_textproto

# The test protos' modules exist in Bazel's output only (:test_py_protos), under the source package tools.nproto.test,
# where the linters cannot see them: imported by name.
defaults_pb2 = importlib.import_module("tools.nproto.test.defaults_pb2")
oneofs_pb2 = importlib.import_module("tools.nproto.test.oneofs_pb2")
retired_pb2 = importlib.import_module("tools.nproto.test.retired_pb2")
scalars_pb2 = importlib.import_module("tools.nproto.test.scalars_pb2")

SCALARS_FILE = "tools/nproto/test/testdata/scalars.textproto"


class ParserTestCase(unittest.TestCase):
    """The assertion the tests share."""

    def assert_invalid(
        self, text: str, message: message_module.Message, source: str, expected: str
    ) -> None:
        """Parsing `text` into `message` fails with an error containing `expected`."""
        with self.assertRaises(nproto_textproto.TextprotoError) as caught:
            nproto_textproto.parse_textproto(text, message, source)
        self.assertIn(expected, str(caught.exception))

    def error(self, text: str, message: message_module.Message, source: str) -> str:
        """The error of parsing `text` into `message`."""
        with self.assertRaises(nproto_textproto.TextprotoError) as caught:
            nproto_textproto.parse_textproto(text, message, source)
        return str(caught.exception)


class TextprotoTest(ParserTestCase):
    def test_a_file_that_matches_the_schema_parses(self):
        scalars = nproto_textproto.load_textproto(SCALARS_FILE, scalars_pb2.Scalars)
        self.assertEqual(scalars.double_value, 1.5)
        self.assertEqual(scalars.uint64_value, 18000000000)
        self.assertEqual(scalars.string_value, "a configuration value")
        self.assertEqual(scalars.bytes_value, b"\001\002")

    def test_an_unknown_field_is_an_error_with_its_line_and_column(self):
        self.assert_invalid(
            "double_value: 1.5\n  doubel_value: 2\n",
            scalars_pb2.Scalars(),
            "config.textproto",
            "config.textproto:2:3: ",
        )
        self.assert_invalid(
            "# A comment.\nscalars { int32_value: 1 bogus: 2 }\n",
            oneofs_pb2.Oneofs(),
            "nested.textproto",
            'nested.textproto:2:26: Message type "nproto_test.Scalars" has no field named "bogus".',
        )
        self.assert_invalid(
            "int32_value: 1\nbogus",
            scalars_pb2.Scalars(),
            "end.textproto",
            "end.textproto:2:1: ",
        )
        self.assert_invalid(
            "scalars {\n  bogus {}\n}\n",
            oneofs_pb2.Oneofs(),
            "message.textproto",
            "message.textproto:2:3: ",
        )

    def test_a_value_of_the_wrong_type_is_an_error_with_its_line_and_column(self):
        self.assert_invalid(
            'int32_value: "seven"\n',
            scalars_pb2.Scalars(),
            "config.textproto",
            "config.textproto:1:14: ",
        )
        self.assert_invalid(
            "\nint32_value: 3000000000\n",
            scalars_pb2.Scalars(),
            "config.textproto",
            "config.textproto:2:14: ",
        )
        self.assert_invalid(
            "bool_value: maybe\n",
            scalars_pb2.Scalars(),
            "config.textproto",
            "config.textproto:1:13: ",
        )
        self.assert_invalid(
            "severity: SEVERITY_NOPE\n",
            oneofs_pb2.Oneofs(),
            "config.textproto",
            "config.textproto:1:11: ",
        )

    def test_other_mistakes_are_errors_too(self):
        # A non-repeated field given twice, at its name as the C++ parser reports it.
        self.assert_invalid(
            "int32_value: 1\nint32_value: 2\n",
            scalars_pb2.Scalars(),
            "twice.textproto",
            "twice.textproto:2:1: ",
        )
        # Two alternatives of one oneof.
        self.assert_invalid(
            "radius: 1\n  side: 2\n",
            oneofs_pb2.Oneofs(),
            "oneof.textproto",
            "oneof.textproto:2:3: ",
        )
        # A syntax error.
        self.assert_invalid(
            "int32_value 1\n",
            scalars_pb2.Scalars(),
            "syntax.textproto",
            "syntax.textproto:1:",
        )
        # A missing proto2 required field has no position.
        error = self.error("count: 1\n", defaults_pb2.Defaults(), "required.textproto")
        self.assertTrue(error.startswith("required.textproto: "), error)
        self.assertIn("required_count", error)
        # A field number instead of a name.
        self.assert_invalid(
            "3: 1\n", scalars_pb2.Scalars(), "number.textproto", "number.textproto:1:"
        )

    def test_a_deprecated_field_is_an_error_at_its_name(self):
        # The C++ parser treats text_format's deprecation warning as an error; text_format has no such warning, so the
        # Python parser looks for the field itself.
        file_proto = text_format.Parse(
            'name: "deprecated_test.proto" package: "deprecated_test" syntax: "proto3" '
            'message_type { name: "Gains" field { name: "kp" number: 1 type: TYPE_DOUBLE label: LABEL_OPTIONAL } '
            'field { name: "old_kp" number: 2 type: TYPE_DOUBLE label: LABEL_OPTIONAL options { deprecated: true } } '
            'field { name: "inner" number: 3 type: TYPE_MESSAGE label: LABEL_OPTIONAL type_name: ".deprecated_test.Gains" } }',
            descriptor_pb2.FileDescriptorProto(),
        )
        pool = descriptor_pool.DescriptorPool()
        pool.Add(file_proto)
        gains_class = message_factory.GetMessageClass(
            pool.FindMessageTypeByName("deprecated_test.Gains")
        )
        self.assertEqual(
            self.error(
                "kp: 1\ninner {\n  kp: 2 old_kp: 3\n}\n",
                gains_class(),
                "gains.textproto",
            ),
            'gains.textproto:3:9: text format contains deprecated field "old_kp"',
        )
        nproto_textproto.parse_textproto(
            "kp: 1 inner { kp: 2 }", gains_class(), "gains.textproto"
        )

    def test_a_missing_file_names_its_path(self):
        path = "tools/nproto/test/testdata/no_such_file.textproto"
        with self.assertRaises(FileNotFoundError) as caught:
            nproto_textproto.load_textproto(path, scalars_pb2.Scalars)
        self.assertIn(path, str(caught.exception))

    def test_parsing_replaces_the_message(self):
        message = scalars_pb2.Scalars(int64_value=5)
        nproto_textproto.parse_textproto("int32_value: 1", message, "inline")
        self.assertEqual(message.int32_value, 1)
        self.assertEqual(message.int64_value, 0)


class RetiredFieldTest(ParserTestCase):
    def test_a_retired_name_is_answered_with_its_replacement(self):
        self.assertEqual(
            self.error(
                'switches: "a"\nuse_old_switch: true\n',
                retired_pb2.Retired(),
                "retired.textproto",
            ),
            "retired.textproto:2:1: 'use_old_switch' is retired: list old_switch under switches",
        )

    def test_every_spelling_of_the_name_hits_and_the_position_is_kept(self):
        self.assertEqual(
            self.error("  useOldSwitch: 1\n", retired_pb2.Retired(), "camel.textproto"),
            "camel.textproto:1:3: 'useOldSwitch' is retired: list old_switch under switches",
        )
        self.assertEqual(
            self.error(
                "step_width: 0.1 legacy_gain: 2\n",
                retired_pb2.Retired(),
                "snake.textproto",
            ),
            "snake.textproto:1:17: 'legacy_gain' is retired: set block.gain instead",
        )

    def test_an_unknown_name_keeps_the_plain_error(self):
        self.assertEqual(
            self.error("bogus: 1\n", retired_pb2.Retired(), "plain.textproto"),
            'plain.textproto:1:1: Message type "nproto_test.Retired" has no field named "bogus".',
        )

    def test_a_snake_cased_field_is_suggested(self):
        self.assertEqual(
            self.error("stepWidth: 0.2\n", retired_pb2.Retired(), "suggest.textproto"),
            'suggest.textproto:1:1: Message type "nproto_test.Retired" has no field named "stepWidth". Did you mean "step_width"?',
        )

    def test_the_layout_hint_ends_every_unknown_field_of_its_message(self):
        self.assertEqual(
            self.error(
                "block {\n  flat_gain: 1\n}\n",
                retired_pb2.Retired(),
                "layout.textproto",
            ),
            'layout.textproto:2:3: Message type "nproto_test.Retired.Block" has no field named "flat_gain". '
            "(the flat keys moved into block, tools/nproto/README.md)",
        )

    def test_the_options_are_readable(self):
        self.assertEqual(
            nproto_textproto.retired_fields(retired_pb2.Retired.DESCRIPTOR),
            [
                ("use_old_switch", "list old_switch under switches"),
                ("legacyGain", "set block.gain instead"),
            ],
        )
        self.assertEqual(
            nproto_textproto.layout_hint(retired_pb2.Retired.DESCRIPTOR), ""
        )
        self.assertEqual(
            nproto_textproto.snake_case("useDcmTerminalCost"), "use_dcm_terminal_cost"
        )
        self.assertEqual(nproto_textproto.snake_case("HTTPRequest"), "http_request")


class HeaderTest(ParserTestCase):
    TEXT = "# proto-file: tools/nproto/test/scalars.proto\n# proto-message: nproto_test.Scalars\n\nint32_value: 1\n"

    def test_a_file_of_another_message_is_refused(self):
        self.assertEqual(
            self.error(self.TEXT, oneofs_pb2.Oneofs(), "scalars.textproto"),
            "scalars.textproto:2:1: scalars.textproto is a nproto_test.Scalars (its '# proto-message:' header), not a "
            "nproto_test.Oneofs",
        )
        # The message the header names parses, and so does a file without a header.
        nproto_textproto.parse_textproto(
            self.TEXT, scalars_pb2.Scalars(), "scalars.textproto"
        )
        nproto_textproto.parse_textproto(
            "radius: 1\n", oneofs_pb2.Oneofs(), "no_header.textproto"
        )

    def test_only_the_leading_comment_block_is_the_header(self):
        nproto_textproto.parse_textproto(
            "# A comment.\n\nradius: 1\n# proto-message: nproto_test.Scalars\n",
            oneofs_pb2.Oneofs(),
            "later.textproto",
        )
        self.assertEqual(
            nproto_textproto.header_message(self.TEXT), (2, "nproto_test.Scalars")
        )
        self.assertIsNone(
            nproto_textproto.header_message("x: 1\n# proto-message: a.B\n")
        )

    def test_the_shipped_test_file_names_its_message(self):
        with open(SCALARS_FILE, encoding="utf-8") as file:
            self.assertEqual(
                nproto_textproto.header_message(file.read()), (2, "nproto_test.Scalars")
            )
        self.assertTrue(os.path.isfile(SCALARS_FILE))


if __name__ == "__main__":
    unittest.main()
