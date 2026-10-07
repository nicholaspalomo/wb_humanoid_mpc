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


"""textproto_document: the tokens, the tree, paths, comments and edits, on texts of the test and on the repository's files.

The properties hold on every textproto the test has as data (the configuration files of every robot as the build has
them, the gait file, the dodgeball golden, the launch and network files) and on random documents: an unedited document
is its text; a value edit changes the characters of that value only; an insertion only adds characters, and a removal
only deletes them; every comment outside an edit is kept.
"""

import math
import os
import random
import subprocess
import sys
import unittest

from config_textproto import textproto_document

ROOT = os.path.join(os.environ.get("TEST_SRCDIR", ""), "_main")
# The texts hold LINT directives as the configuration files do; spelled apart here, so that the IFTTT check does not
# read them as directives of this file.
LINT = "LINT"

ATLAS_STYLE = """# proto-file: humanoid_nmpc/humanoid_mpc_config/task_file.proto
# proto-message: humanoid_mpc_config.TaskFile
#
# Simulation & Telemetry Configuration
# @LINT@.IfChange(telemetry_sinks)
telemetry_sinks: "bus"
# @LINT@.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/include/humanoid_common_mpc_app/robot/TelemetrySinkRegistry.h:telemetry_sink_names)
telemetry_frequency: 100  # [Hz] robot/state, decimated from the control loop
acom_weights {
  scaling: 100
  yaw: 25  # theta_acom_z / yaw (matches theta_base_z)
  pitch: 150  # theta_acom_y / pitch
  roll: 50
}
state_weights {
  scaling: 85
  # Raised from 2.
  normalized_linear_momentum {
    x: 10  # h_com_x / robotMass
    y: 10  # h_com_y / robotMass
    z: 15  # h_com_z / robotMass
  }
  joint_positions { joint: "back_bkz" value: 5 }
  # Lowered from 250.
  joint_positions { joint: "l_leg_hpx" value: 25 }  # hip roll
}
""".replace(
    "@LINT@", LINT
)


def _textproto_files() -> list[str]:
    """Every .textproto in the test's data."""
    found = []
    for directory, _, names in os.walk(ROOT):
        found += [
            os.path.join(directory, name)
            for name in names
            if name.endswith(".textproto")
        ]
    return sorted(found)


def _comments(text: str) -> list[str]:
    return [
        token.text
        for token in textproto_document.tokenize(text)
        if token.kind == textproto_document.COMMENT
    ]


def _common_affixes(old: str, new: str) -> tuple[int, int]:
    """(the length of the longest common prefix, of the longest common suffix that does not overlap it)."""
    prefix = 0
    limit = min(len(old), len(new))
    while prefix < limit and old[prefix] == new[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < limit - prefix
        and old[len(old) - 1 - suffix] == new[len(new) - 1 - suffix]
    ):
        suffix += 1
    return prefix, suffix


class StandardLibraryTest(unittest.TestCase):
    def test_the_module_needs_the_standard_library_only(self):
        # A fresh interpreter: this one may have loaded other modules. The system Python has no protobuf.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from config_textproto import textproto_document; "
                "print(sorted(m for m in sys.modules if m.split('.')[0] in ('google', 'config_textproto')))",
            ],
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            result.stdout.strip(),
            "['config_textproto', 'config_textproto.textproto_document']",
            result.stderr,
        )


class TokenizerTest(unittest.TestCase):
    def kinds(self, text: str) -> list[tuple[str, str]]:
        return [
            (token.kind, token.text)
            for token in textproto_document.tokenize(text)
            if token.kind != textproto_document.SPACE
        ]

    def test_the_tokens_are_the_text(self):
        for text in (
            ATLAS_STYLE,
            "",
            " \n\t",
            "a:1",
            "a: -1.5e-3f # c\n",
            "a: 'x' \"y\"",
            "[a.b/c]: <>",
        ):
            with self.subTest(text=text):
                tokens = textproto_document.tokenize(text)
                self.assertEqual("".join(token.text for token in tokens), text)
                for token in tokens:
                    self.assertEqual(text[token.start : token.end], token.text)

    def test_numbers(self):
        for text in (
            "1",
            "1.5",
            ".5",
            "1.",
            "1e5",
            "1E+5",
            "1.5e-3",
            "1.5f",
            "2F",
            "0x1F",
            "017",
            "18000000000",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.kinds(text), [(textproto_document.NUMBER, text)])

    def test_names_and_signs(self):
        self.assertEqual(
            self.kinds("-inf nan true KIND_STANCE"),
            [
                (textproto_document.SYMBOL, "-"),
                (textproto_document.IDENTIFIER, "inf"),
                (textproto_document.IDENTIFIER, "nan"),
                (textproto_document.IDENTIFIER, "true"),
                (textproto_document.IDENTIFIER, "KIND_STANCE"),
            ],
        )

    def test_strings_of_both_quotes_with_escapes(self):
        text = r""""a\"b" 'c\'d' "e\\" '\n'"""
        self.assertEqual(
            [kind for kind, _ in self.kinds(text)], [textproto_document.STRING] * 4
        )

    def test_symbols_and_comments(self):
        self.assertEqual(
            self.kinds("{}<>[]:,;./ # rest of line\n"),
            [(textproto_document.SYMBOL, symbol) for symbol in "{}<>[]:,;./"]
            + [(textproto_document.COMMENT, "# rest of line")],
        )

    def test_an_unterminated_string_is_an_error_at_its_quote(self):
        with self.assertRaises(textproto_document.TextprotoSyntaxError) as caught:
            textproto_document.tokenize('a: 1\nb: "open\n', "f.textproto")
        self.assertEqual(str(caught.exception), "f.textproto:2:4: unterminated string")

    def test_a_character_no_token_starts_with_is_an_error(self):
        with self.assertRaises(textproto_document.TextprotoSyntaxError) as caught:
            textproto_document.tokenize("a: 1 @", "f.textproto")
        self.assertEqual(
            str(caught.exception), "f.textproto:1:6: unexpected character '@'"
        )


class ValuesTest(unittest.TestCase):
    def value(self, text: str) -> object:
        return textproto_document.parse(f"x: {text}").value("x")

    def test_numbers_read_as_text_format_reads_them(self):
        self.assertEqual(self.value("5"), 5)
        self.assertIsInstance(self.value("5"), int)
        self.assertEqual(self.value("-5"), -5)
        self.assertEqual(self.value("1.5"), 1.5)
        self.assertEqual(self.value("1.5f"), 1.5)
        self.assertEqual(self.value("5e-5"), 5e-5)
        self.assertEqual(self.value("0x10"), 16)
        self.assertEqual(self.value("010"), 8)
        self.assertEqual(self.value("- 2"), -2)
        self.assertEqual(self.value("inf"), math.inf)
        self.assertEqual(self.value("-Infinity"), -math.inf)
        self.assertTrue(math.isnan(self.value("nan")))  # type: ignore[arg-type]  # a float here

    def test_names(self):
        self.assertIs(self.value("true"), True)
        self.assertIs(self.value("False"), False)
        self.assertEqual(self.value("KIND_STANCE"), "KIND_STANCE")

    def test_strings_are_concatenated_and_unescaped(self):
        self.assertEqual(self.value(r'"a" "b" ' + "'c'"), "abc")
        self.assertEqual(self.value(r'"\n\t\\\"\'\?"'), "\n\t\\\"'?")
        self.assertEqual(self.value(r'"\101\x42é\U0001F600"'), "ABé\U0001f600")
        self.assertEqual(self.value('"hé"'), "hé")
        node = textproto_document.parse(r'x: "\001\377"').scalar("x")
        self.assertEqual(node.bytes_value(), b"\x01\xff")

    def test_an_unknown_escape_is_an_error(self):
        node = textproto_document.parse(r'x: "\q"').scalar("x")
        with self.assertRaises(textproto_document.DocumentError):
            node.bytes_value()

    def test_parse_number_takes_a_sign(self):
        self.assertEqual(textproto_document.parse_number("-0x10"), -16)
        self.assertEqual(textproto_document.parse_number("+2.5"), 2.5)
        with self.assertRaises(ValueError):
            textproto_document.parse_number("1_000")
        with self.assertRaises(ValueError):
            textproto_document.parse_number("abc")

    def test_formatting_reads_back(self):
        for value in (0.1, -0.0, 1e-05, 1e16, 123456789.123, 5.0, math.inf, -math.inf):
            with self.subTest(value=value):
                text = textproto_document.format_scalar(value)
                read = textproto_document.parse(f"x: {text}").scalar("x").number_value()
                self.assertTrue(
                    textproto_document.same_value(
                        textproto_document.parse(f"x: {text}").scalar("x"), value
                    )
                )
                self.assertEqual(float(read), value)
        self.assertEqual(textproto_document.format_scalar(math.nan), "nan")
        for string in (
            "",
            "plain",
            'quote " and \\ backslash',
            "tab\tnewline\n",
            "\x01\x7f",
            "hé ✓",
        ):
            with self.subTest(string=string):
                text = textproto_document.format_scalar(string)
                self.assertEqual(
                    textproto_document.parse(f"x: {text}").value("x"), string
                )
        self.assertEqual(textproto_document.format_scalar(b"\x00a\xff"), r'"\000a\377"')
        self.assertEqual(textproto_document.format_scalar(True), "true")
        self.assertEqual(textproto_document.format_scalar(-7), "-7")
        self.assertEqual(
            textproto_document.format_scalar(
                textproto_document.Identifier("KIND_SWING")
            ),
            "KIND_SWING",
        )

    def test_a_literal_must_be_one_value(self):
        self.assertEqual(
            textproto_document.format_scalar(textproto_document.Literal("0.1")), "0.1"
        )
        for text in ("1 2", "", "{", "a: 1"):
            with self.subTest(text=text), self.assertRaises(
                textproto_document.DocumentError
            ):
                textproto_document.format_scalar(textproto_document.Literal(text))
        with self.assertRaises(textproto_document.DocumentError):
            textproto_document.Identifier("not a name")

    def test_same_value_compares_by_value_and_bit_for_bit(self):
        def node(text: str) -> textproto_document.ScalarNode:
            return textproto_document.parse(f"x: {text}").scalar("x")

        self.assertTrue(textproto_document.same_value(node("5"), 5.0))
        self.assertTrue(textproto_document.same_value(node("5e-5"), 0.00005))
        self.assertTrue(textproto_document.same_value(node("0x10"), 16))
        self.assertFalse(textproto_document.same_value(node("0"), -0.0))
        self.assertTrue(textproto_document.same_value(node("-0.0"), -0.0))
        self.assertTrue(textproto_document.same_value(node("nan"), math.nan))
        self.assertFalse(textproto_document.same_value(node("5"), 5.5))
        self.assertFalse(textproto_document.same_value(node("5.5"), 5))
        self.assertTrue(textproto_document.same_value(node("t"), True))
        self.assertFalse(textproto_document.same_value(node("true"), False))
        self.assertTrue(textproto_document.same_value(node('"a" "b"'), "ab"))
        self.assertFalse(textproto_document.same_value(node("ab"), "ab"))
        self.assertTrue(
            textproto_document.same_value(
                node("KIND_A"), textproto_document.Identifier("KIND_A")
            )
        )
        self.assertTrue(
            textproto_document.same_value(
                node("- inf"), textproto_document.Literal("- inf")
            )
        )


class TreeTest(unittest.TestCase):
    def test_fields_lists_blocks_and_separators(self):
        document = textproto_document.parse(
            'a: 1, b { c: "x"; } d <e: 2> f: [1, 2] g [{h: 1}, <h: 2>] [ext.name]: 3'
        )
        self.assertEqual(
            document.field_names(), ["a", "b", "d", "f", "g", "[ext.name]"]
        )
        self.assertEqual(document.value("b.c"), "x")
        self.assertEqual(document.value("d.e"), 2)
        self.assertEqual(document.value("f[1]"), 2)
        self.assertEqual(document.value("g[1].h"), 2)
        self.assertEqual(document.count("g"), 2)
        self.assertEqual(document.count("missing"), 0)
        self.assertEqual(document.count("absent_block.x"), 0)

    def test_syntax_errors_name_the_position(self):
        cases = {
            "a 5\n": "1:3: expected \":\" between a and its value, found '5'",
            "a: 1\nb {\n  c: 2\n": '4:1: expected "}" before the end of the text',
            "a { b: 1 >": "1:10: expected a field name, found '>'",
            "a: [1 2]": '1:7: expected "," or "]", found \'2\'',
            'a: -"x"': "1:5: expected a number after '-', found '\"x\"'",
            "}": "1:1: expected a field name, found '}'",
            "a:": "1:3: expected a value of a before the end of the text",
            "a: :": "1:4: expected a value, found ':'",
        }
        for text, expected in cases.items():
            with self.subTest(text=text), self.assertRaises(
                textproto_document.TextprotoSyntaxError
            ) as caught:
                textproto_document.parse(text, "f.textproto")
            self.assertEqual(str(caught.exception), f"f.textproto:{expected}")

    def test_positions_comments_and_spans(self):
        document = textproto_document.parse(ATLAS_STYLE)
        self.assertEqual(document.position("telemetry_frequency"), (8, 1))
        self.assertEqual(
            document.position("state_weights.normalized_linear_momentum.y"), (20, 5)
        )
        self.assertEqual(
            document.trailing_comment("telemetry_frequency"),
            "[Hz] robot/state, decimated from the control loop",
        )
        self.assertEqual(document.trailing_comment("acom_weights.roll"), None)
        self.assertEqual(
            document.trailing_comment(
                "state_weights.joint_positions[joint=l_leg_hpx].value"
            ),
            "hip roll",
        )
        self.assertEqual(
            document.trailing_comment("state_weights.joint_positions[joint=l_leg_hpx]"),
            "hip roll",
        )
        self.assertEqual(
            document.trailing_comment(
                "state_weights.joint_positions[joint=l_leg_hpx].joint"
            ),
            None,
        )
        self.assertEqual(
            document.leading_comments("state_weights.normalized_linear_momentum"),
            ["Raised from 2."],
        )
        self.assertEqual(
            document.leading_comments("state_weights.joint_positions[1]"),
            ["Lowered from 250."],
        )
        self.assertEqual(document.leading_comments("state_weights.scaling"), [])
        start, end = document.span("acom_weights.yaw")
        self.assertEqual(ATLAS_STYLE[start:end], "yaw: 25")
        block = textproto_document.parse("a {  # about a\n  b: 1\n}\n")
        self.assertEqual(block.trailing_comment("a"), "about a")

    def test_the_file_and_its_root(self):
        document = textproto_document.parse(ATLAS_STYLE)
        self.assertIs(document.find(""), document.root)
        self.assertEqual(document.source, "<text>")


class PathTest(unittest.TestCase):
    def test_paths_read_back(self):
        for path in (
            "a",
            "a.b.c",
            "a[0]",
            "a[12].b",
            "a[joint=l_leg_aky].value",
            'a[name="with space"].b',
            'a[name="q\\"uote]"]',
            "a[key=-1.5]",
        ):
            with self.subTest(path=path):
                segments = textproto_document.parse_path(path)
                self.assertEqual(
                    textproto_document.parse_path(
                        textproto_document.format_path(segments)
                    ),
                    segments,
                )
        self.assertEqual(textproto_document.parse_path(""), ())
        self.assertEqual(
            textproto_document.parse_path('a[name="x]y"]')[0].key,
            ("name", "x]y"),
        )

    def test_malformed_paths(self):
        for path in (
            "a.",
            ".a",
            "a[",
            "a[x]",
            "a[x=]",
            "a[0",
            "a b",
            'a[name="x]',
            "1a",
            "a..b",
        ):
            with self.subTest(path=path), self.assertRaises(
                textproto_document.PathError
            ):
                textproto_document.parse_path(path)

    def test_selection(self):
        document = textproto_document.parse(ATLAS_STYLE)
        self.assertEqual(
            document.value("state_weights.joint_positions[joint=back_bkz].value"), 5
        )
        self.assertEqual(
            document.value('state_weights.joint_positions[joint="back_bkz"].value'), 5
        )
        self.assertEqual(document.value("state_weights.joint_positions[1].value"), 25)
        self.assertIsNone(
            document.find("state_weights.joint_positions[joint=nope].value")
        )
        self.assertIsNone(document.find("state_weights.joint_positions[7]"))
        self.assertIsNone(document.find("missing.deeper"))
        self.assertFalse(document.has("state_weights.missing"))
        with self.assertRaises(textproto_document.PathError):
            document.find(
                "state_weights.joint_positions.value"
            )  # two elements, no selector
        with self.assertRaises(textproto_document.PathError):
            document.find("state_weights.scaling.x")  # crosses a value
        with self.assertRaises(textproto_document.PathError):
            document.get("state_weights.missing")
        with self.assertRaises(textproto_document.PathError):
            document.scalar("state_weights")
        duplicate = textproto_document.parse('e { k: "a" } e { k: "a" }')
        with self.assertRaises(textproto_document.PathError):
            duplicate.find("e[k=a]")

    def test_keys_compare_by_value(self):
        document = textproto_document.parse(
            "e { id: 0x10 v: 1 } e { id: -2 v: 2 } e { id: K v: 3 }"
        )
        self.assertEqual(document.value("e[id=16].v"), 1)
        self.assertEqual(document.value("e[id=-2].v"), 2)
        self.assertEqual(document.value("e[id=K].v"), 3)


class EditTest(unittest.TestCase):
    def assert_one_splice(
        self, old: str, new: str, start: int, end: int, replacement: str
    ) -> None:
        self.assertEqual(new, old[:start] + replacement + old[end:])

    def test_set_scalar_replaces_the_value_token_only(self):
        document = textproto_document.parse(ATLAS_STYLE)
        node = document.scalar("state_weights.normalized_linear_momentum.y")
        self.assertTrue(
            document.set_scalar("state_weights.normalized_linear_momentum.y", 12.5)
        )
        self.assert_one_splice(
            ATLAS_STYLE, document.text(), node.start, node.end, "12.5"
        )
        self.assertEqual(_comments(document.text()), _comments(ATLAS_STYLE))

    def test_an_unchanged_value_keeps_its_spelling(self):
        document = textproto_document.parse("a: 5\nb: 5e-5\nc: 0x10\nd: t\ne: 'x'\n")
        for path, value in (
            ("a", 5.0),
            ("b", 0.00005),
            ("c", 16),
            ("d", True),
            ("e", "x"),
        ):
            with self.subTest(path=path):
                self.assertFalse(document.set_scalar(path, value))
        self.assertEqual(document.text(), "a: 5\nb: 5e-5\nc: 0x10\nd: t\ne: 'x'\n")

    def test_values_of_every_kind_are_written(self):
        document = textproto_document.parse(
            'a: 1\nb: - 2\nc: "x" "y"\nd: true\ne: K1\nf: [1, 2]\n'
        )
        document.set_scalar("a", 2.5)
        document.set_scalar("b", -3)
        document.set_scalar("c", 'q"')
        document.set_scalar("d", False)
        document.set_scalar("e", textproto_document.Identifier("K2"))
        document.set_scalar("f[1]", 7)
        self.assertEqual(
            document.text(), 'a: 2.5\nb: -3\nc: "q\\""\nd: false\ne: K2\nf: [1, 7]\n'
        )

    def test_insert_at_the_end_of_a_block_at_the_siblings_indentation(self):
        document = textproto_document.parse(ATLAS_STYLE)
        document.set_or_insert("acom_weights.extra", 1.0)
        self.assertIn("  roll: 50\n  extra: 1.0\n}\n", document.text())
        document.set_or_insert("state_weights.normalized_linear_momentum.w", 2.0)
        self.assertIn(
            "    z: 15  # h_com_z / robotMass\n    w: 2.0\n  }\n", document.text()
        )

    def test_insert_after_the_last_field_of_the_same_name(self):
        document = textproto_document.parse(ATLAS_STYLE)
        document.set_or_insert(
            "state_weights.joint_positions[joint=r_leg_hpx].value", 30.0
        )
        self.assertIn(
            '  joint_positions { joint: "l_leg_hpx" value: 25 }  # hip roll\n'
            '  joint_positions { joint: "r_leg_hpx" value: 30.0 }\n}\n',
            document.text(),
        )
        document.append("telemetry_sinks", "rerun")
        self.assertIn(
            f'telemetry_sinks: "bus"\ntelemetry_sinks: "rerun"\n# {LINT}.ThenChange(',
            document.text(),
        )

    def test_insert_creates_the_blocks_it_needs(self):
        document = textproto_document.parse(ATLAS_STYLE)
        document.set_or_insert("contacts.contact_rectangle.length", 0.2)
        self.assertTrue(
            document.text().endswith(
                "contacts {\n  contact_rectangle {\n    length: 0.2\n  }\n}\n"
            )
        )
        document.set_or_insert("state_weights.base_orientation.yaw", 1.0)
        self.assertIn("  base_orientation {\n    yaw: 1.0\n  }\n}\n", document.text())
        document.set_or_insert(
            "input_weights.contact_wrenches[contact=left].force.x", 3.0
        )
        self.assertIn(
            'input_weights {\n  contact_wrenches {\n    contact: "left"\n    force {\n      x: 3.0\n    }\n  }\n}\n',
            document.text(),
        )

    def test_insert_in_one_line_blocks_lists_and_files_without_a_final_newline(self):
        document = textproto_document.parse("a { b: 1 }\nc {}\nd: [1]\ne: []\nf {b:1}")
        document.set_or_insert("a.x", 2)
        document.set_or_insert("c.x", 3)
        document.append("d", 4)
        document.append("e", 5)
        document.set_or_insert("f.x", 6)
        document.set_or_insert("g", 7)
        self.assertEqual(
            document.text(),
            "a { b: 1 x: 2 }\nc { x: 3 }\nd: [1, 4]\ne: [5]\nf {b:1 x: 6}\ng: 7\n",
        )
        last = textproto_document.parse("a: 1")
        last.append("a", 2)
        self.assertEqual(last.text(), "a: 1\na: 2")
        empty = textproto_document.parse("")
        empty.set_or_insert("x.y", 1)
        self.assertEqual(empty.text(), "x {\n  y: 1\n}\n")

    def test_insert_follows_the_file_indentation_and_line_endings(self):
        document = textproto_document.parse("a {\r\n    b: 1\r\n}\r\n")
        document.set_or_insert("a.c.d", 2)
        self.assertEqual(
            document.text(),
            "a {\r\n    b: 1\r\n    c {\r\n        d: 2\r\n    }\r\n}\r\n",
        )

    def test_insert_field_takes_blocks(self):
        document = textproto_document.parse("x {\n  y: 1\n}\n")
        document.insert_field(
            "x",
            "z",
            textproto_document.NewBlock(
                (("a", 1), ("a", 2), ("b", textproto_document.NewBlock()))
            ),
        )
        self.assertEqual(
            document.text(),
            "x {\n  y: 1\n  z {\n    a: 1\n    a: 2\n    b {}\n  }\n}\n",
        )
        document.append("x.z.a", 3)
        self.assertIn("    a: 2\n    a: 3\n", document.text())

    def test_impossible_inserts_are_refused(self):
        document = textproto_document.parse("a: 1\nb { c: 1 }\nl: [1, 2]\n")
        for path, problem in (
            ("a.x", "is a value, not a block"),
            ("l[5]", "only [2] can be added"),
            ("b.e[k=v]", "names a block, not a value"),
            ("b.n.m[3].x", "no element [3]"),
            ("", "names the file"),
        ):
            with self.subTest(path=path), self.assertRaises(
                textproto_document.PathError
            ) as caught:
                document.set_or_insert(path, 1)
            self.assertIn(problem, str(caught.exception))
        self.assertEqual(document.text(), "a: 1\nb { c: 1 }\nl: [1, 2]\n")
        with self.assertRaises(textproto_document.PathError):
            document.set_scalar("b", 1)
        with self.assertRaises(textproto_document.PathError):
            document.insert_field("a", "x", 1)
        with self.assertRaises(textproto_document.PathError):
            document.append("l[0]", 1)

    def test_remove_a_field_with_its_line_and_comment(self):
        document = textproto_document.parse(ATLAS_STYLE)
        self.assertEqual(document.remove("telemetry_frequency"), 1)
        self.assertNotIn("telemetry_frequency", document.text())
        self.assertNotIn("decimated", document.text())
        self.assertIn(f"# {LINT}.ThenChange(", document.text())
        self.assertEqual(
            document.remove("state_weights.joint_positions[joint=l_leg_hpx]"), 1
        )
        self.assertIn(
            '  joint_positions { joint: "back_bkz" value: 5 }\n  # Lowered from 250.\n}\n',
            document.text(),
        )
        self.assertEqual(document.remove("state_weights.normalized_linear_momentum"), 1)
        self.assertEqual(document.remove("nothing.here"), 0)
        self.assertEqual(document.remove("state_weights.nothing"), 0)

    def test_remove_on_shared_lines_lists_and_repeated_fields(self):
        document = textproto_document.parse(
            "a { b: 1 c: 2 }\nl: [1, 2, 3]\nr: 1\nr: [2, 3]\nm: [4]\nx: 1 y: 2\n"
        )
        document.remove("a.b")
        document.remove("l[1]")
        document.remove("l[1]")
        self.assertEqual(document.remove("r"), 3)
        document.remove("m[0]")
        document.remove("y")
        self.assertEqual(document.text(), "a { c: 2 }\nl: [1]\nm: []\nx: 1\n")
        same_line = textproto_document.parse("a: 1 a: 2\n")
        self.assertEqual(same_line.remove("a"), 2)
        self.assertEqual(same_line.text(), "\n")

    def test_removing_a_list_element_keeps_the_comment_lines_between_the_elements(self):
        # A LINT directive, written so that the repository's IFTTT check does not take this source line for one.
        if_change = f"  # {LINT}.IfChange(log_sink)\n"
        then_change = f"  # {LINT}.ThenChange(//x/y.cpp:log_sink)\n"
        text = (
            'telemetry_sinks: [\n  "rerun",\n'
            + if_change
            + '  "log",\n'
            + then_change
            + '  "csv"\n]\n'
        )
        expected = {
            0: "telemetry_sinks: [\n"
            + if_change
            + '  "log",\n'
            + then_change
            + '  "csv"\n]\n',
            1: 'telemetry_sinks: [\n  "rerun",\n'
            + if_change
            + then_change
            + '  "csv"\n]\n',
            2: 'telemetry_sinks: [\n  "rerun",\n'
            + if_change
            + '  "log"\n'
            + then_change
            + "]\n",
        }
        for index, kept in expected.items():
            with self.subTest(index=index):
                document = textproto_document.parse(text)
                self.assertEqual(document.remove(f"telemetry_sinks[{index}]"), 1)
                self.assertEqual(document.text(), kept)
                # Still a list of the other two, the directives in place.
                textproto_document.parse(document.text())

    def test_a_failed_edit_leaves_the_document_as_it_was(self):
        document = textproto_document.parse(ATLAS_STYLE)
        with self.assertRaises(textproto_document.DocumentError):
            document.set_scalar("acom_weights.yaw", textproto_document.Literal("1 2"))
        self.assertEqual(document.text(), ATLAS_STYLE)


def _random_document(rng: random.Random, depth: int = 0) -> str:
    """A random textproto: fields, blocks, lists, comments and every spelling of a value, at random indentation."""
    lines = []
    for _ in range(rng.randint(0, 6)):
        indent = "  " * depth
        name = rng.choice(["a", "b", "joint_positions", "x", "value"])
        roll = rng.random()
        if roll < 0.15:
            lines.append(
                f"{indent}# {rng.choice([f'{LINT}.IfChange(x)', 'a comment', f'{LINT}.ThenChange(//y:z)', ''])}"
            )
        elif roll < 0.25 and depth < 3:
            inner = _random_document(rng, depth + 1).rstrip("\n")
            lines.append(
                f"{indent}{name} {{"
                + ("\n" + inner + "\n" + indent if inner else "")
                + "}"
            )
        elif roll < 0.35:
            lines.append(
                f'{indent}{name} {{ k: "{rng.randint(0, 9)}" v: {rng.randint(-5, 5)} }}'
            )
        elif roll < 0.45:
            values = ", ".join(
                rng.choice(["1", "-2.5", '"s"', "inf", "E"])
                for _ in range(rng.randint(0, 3))
            )
            lines.append(f"{indent}{name}: [{values}]")
        else:
            value = rng.choice(
                [
                    "1",
                    "-1.5e-3",
                    "0x1F",
                    ".5f",
                    "- 2",
                    "true",
                    "KIND",
                    '"q\\"x"',
                    "'s' \"t\"",
                    "nan",
                    "-inf",
                ]
            )
            separator = rng.choice(["", "", ",", ";"])
            comment = rng.choice(["", "", "  # trailing"])
            lines.append(f"{indent}{name}: {value}{separator}{comment}")
    return (
        "".join(f"{line}\n" for line in lines)
        if rng.random() < 0.9
        else "\n".join(lines)
    )


class PropertyTest(unittest.TestCase):
    def documents(self) -> list[tuple[str, str]]:
        """(source, text) of every textproto of the data, the test's text and random documents."""
        found = [("atlas_style", ATLAS_STYLE)]
        for path in _textproto_files():
            with open(path, encoding="utf-8", newline="") as file:
                found.append((path, file.read()))
        rng = random.Random(20261003)
        found += [(f"random {index}", _random_document(rng)) for index in range(300)]
        return found

    def test_the_data_holds_the_configuration_files(self):
        names = {os.path.basename(path) for path in _textproto_files()}
        for expected in (
            "task.textproto",
            "reference.textproto",
            "joint_pd_gains.textproto",
            "gait.textproto",
        ):
            self.assertIn(expected, names)

    def test_an_unedited_document_is_its_text(self):
        for source, text in self.documents():
            with self.subTest(source=source):
                self.assertEqual(textproto_document.parse(text, source).text(), text)

    def test_scalar_edits_change_only_their_value(self):
        for source, text in self.documents():
            document = textproto_document.parse(text, source)
            scalars = _scalars(document)
            with self.subTest(source=source, edit="every value as it is"):
                unchanged = document.copy()
                same = [
                    (path, textproto_document.Literal(node.text))
                    for path, node in scalars
                ]
                self.assertEqual(unchanged.set_scalars(same), 0)
                self.assertEqual(unchanged.text(), text)
            with self.subTest(source=source, edit="every value at once"):
                edited = document.copy()
                new = [
                    (path, textproto_document.Literal("12345")) for path, _ in scalars
                ]
                self.assertEqual(
                    edited.set_scalars(new),
                    sum(1 for _, node in scalars if node.text != "12345"),
                )
                self.assertEqual(_gaps(edited), _gaps(document))
                self.assertEqual(_comments(edited.text()), _comments(text))
            for path, node in _sample(scalars):
                with self.subTest(source=source, path=path):
                    edited = document.copy()
                    edited.set_scalar(path, textproto_document.Literal("12345"))
                    self.assertEqual(
                        edited.text(), text[: node.start] + "12345" + text[node.end :]
                    )
                    self.assertEqual(document.text(), text)

    def test_inserts_only_add_and_removals_only_delete(self):
        rng = random.Random(7)
        values: list[textproto_document.ScalarValue] = [1, 2.5, "s", True]
        for source, text in self.documents():
            document = textproto_document.parse(text, source)
            blocks = [""] + [path for path, node in _blocks(document)]
            for parent in _sample(blocks):
                with self.subTest(source=source, parent=parent):
                    edited = document.copy()
                    edited.insert_field(parent, "inserted_field", rng.choice(values))
                    prefix, suffix = _common_affixes(text, edited.text())
                    self.assertEqual(prefix + suffix, len(text))
                    self.assertEqual(_comments(edited.text()), _comments(text))
                    path = f"{parent}.inserted_field" if parent else "inserted_field"
                    self.assertTrue(edited.has(path))
            for path, _ in _sample(_scalars(document) + _blocks(document)):
                with self.subTest(source=source, removed=path):
                    edited = document.copy()
                    edited.remove(path)
                    prefix, suffix = _common_affixes(text, edited.text())
                    self.assertEqual(prefix + suffix, len(edited.text()))


# How many values of a file the single-edit properties take, spread over the file (a file of the robots has about a
# thousand; each edit parses the file again).
SAMPLE_SIZE = 40


def _sample(items: list) -> list:
    """At most SAMPLE_SIZE of `items`, evenly spread, in order."""
    if len(items) <= SAMPLE_SIZE:
        return items
    return [items[index * len(items) // SAMPLE_SIZE] for index in range(SAMPLE_SIZE)]


def _gaps(document: textproto_document.TextprotoDocument) -> list[str]:
    """The text between the scalars of `document`: what an edit of values only must leave as it is."""
    text = document.text()
    gaps = []
    position = 0
    for _, node in _scalars(document):
        gaps.append(text[position : node.start])
        position = node.end
    gaps.append(text[position:])
    return gaps


def _scalars(
    document: textproto_document.TextprotoDocument,
) -> list[tuple[str, textproto_document.ScalarNode]]:
    """(an index path, the node) of every scalar of the document."""
    return [
        (path, node)
        for path, node in _values(document, document.root, ())
        if isinstance(node, textproto_document.ScalarNode)
    ]


def _blocks(
    document: textproto_document.TextprotoDocument,
) -> list[tuple[str, textproto_document.BlockNode]]:
    return [
        (path, node)
        for path, node in _values(document, document.root, ())
        if isinstance(node, textproto_document.BlockNode)
    ]


def _values(
    document: textproto_document.TextprotoDocument,
    block: textproto_document.BlockNode,
    prefix: tuple[textproto_document.PathSegment, ...],
) -> list[tuple[str, textproto_document.ScalarNode | textproto_document.BlockNode]]:
    """Every value below `block`, each by its path with an index on every segment."""
    found: list[
        tuple[str, textproto_document.ScalarNode | textproto_document.BlockNode]
    ] = []
    counts: dict[str, int] = {}
    for field in block.fields:
        if field.name.startswith("["):
            continue
        nodes = (
            field.value.elements
            if isinstance(field.value, textproto_document.ListNode)
            else (field.value,)
        )
        for node in nodes:
            index = counts.get(field.name, 0)
            counts[field.name] = index + 1
            segments = (
                *prefix,
                textproto_document.PathSegment(field.name, index=index),
            )
            found.append((textproto_document.format_path(segments), node))
            if isinstance(node, textproto_document.BlockNode):
                found += _values(document, node, segments)
    return found


if __name__ == "__main__":
    unittest.main()
