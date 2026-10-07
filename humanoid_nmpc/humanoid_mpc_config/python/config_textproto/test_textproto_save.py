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


"""textproto_save: edits checked against the schema, written atomically, on nproto's test protos and the file schemas.

The properties hold on every configuration file the test has as data and on random messages of every file schema and of
nproto's test protos, written in random spellings and layouts with comments:

- an edit of a value gives exactly the edited message, and changes the characters of that value only;
- setting a value to what the file already holds changes nothing, byte for byte;
- an insertion only adds characters and a removal only deletes them, and every other comment stays.
"""

import dataclasses
import importlib
import importlib.util
import math
import os
import random
import shutil
import stat
import struct
import tempfile
from typing import Any, TypeAlias
import unittest
from unittest import mock

from google.protobuf import descriptor as descriptor_module
from google.protobuf import descriptor_pool
from google.protobuf import message_factory
from humanoid_mpc_config import contact_planning_file_pb2
from humanoid_mpc_config import gait_file_pb2
from humanoid_mpc_config import joint_pd_gains_file_pb2
from humanoid_mpc_config import mpc_parameter_update_pb2
from humanoid_mpc_config import reference_file_pb2
from humanoid_mpc_config import task_file_pb2

from config_textproto import textproto_document
from config_textproto import textproto_save
import nproto_textproto

# The test protos' modules exist in Bazel's output only (//tools/nproto/test:test_py_protos), under the source package
# tools.nproto.test, where the linters cannot see them: imported by name.
defaults_pb2 = importlib.import_module("tools.nproto.test.defaults_pb2")
editions_pb2 = importlib.import_module("tools.nproto.test.editions_pb2")
oneofs_pb2 = importlib.import_module("tools.nproto.test.oneofs_pb2")
optionals_pb2 = importlib.import_module("tools.nproto.test.optionals_pb2")
outer_pb2 = importlib.import_module("tools.nproto.test.outer_pb2")
repeated_fields_pb2 = importlib.import_module("tools.nproto.test.repeated_fields_pb2")
scalars_pb2 = importlib.import_module("tools.nproto.test.scalars_pb2")

ROOT = os.path.join(os.environ.get("TEST_SRCDIR", ""), "_main")
FieldDescriptor: TypeAlias = descriptor_module.FieldDescriptor
# The texts hold LINT directives as the configuration files do; spelled apart here, so that the IFTTT check does not
# read them as directives of this file.
LINT = "LINT"

FILE_MESSAGES = (
    task_file_pb2.TaskFile,
    reference_file_pb2.ReferenceFile,
    joint_pd_gains_file_pb2.JointPdGainsFile,
    contact_planning_file_pb2.ContactPlanningFile,
    gait_file_pb2.GaitFile,
    mpc_parameter_update_pb2.MpcParameterUpdate,
)
TEST_MESSAGES = (
    scalars_pb2.Scalars,
    repeated_fields_pb2.RepeatedFields,
    outer_pb2.Outer,
    editions_pb2.Editions,
    optionals_pb2.Optionals,
    oneofs_pb2.Oneofs,
    defaults_pb2.Defaults,
)
_INTEGER_RANGES = {
    FieldDescriptor.TYPE_INT32: (-(2**31), 2**31 - 1),
    FieldDescriptor.TYPE_SINT32: (-(2**31), 2**31 - 1),
    FieldDescriptor.TYPE_SFIXED32: (-(2**31), 2**31 - 1),
    FieldDescriptor.TYPE_UINT32: (0, 2**32 - 1),
    FieldDescriptor.TYPE_FIXED32: (0, 2**32 - 1),
    FieldDescriptor.TYPE_INT64: (-(2**63), 2**63 - 1),
    FieldDescriptor.TYPE_SINT64: (-(2**63), 2**63 - 1),
    FieldDescriptor.TYPE_SFIXED64: (-(2**63), 2**63 - 1),
    FieldDescriptor.TYPE_UINT64: (0, 2**64 - 1),
    FieldDescriptor.TYPE_FIXED64: (0, 2**64 - 1),
}

SCALARS_TEXT = """# proto-message: nproto_test.Scalars
# A comment that stays.
double_value: 1.5  # @LINT@.IfChange(x)
float_value: 0.1
int32_value: 5
uint64_value: 0x10
bool_value: t
string_value: 'a' "b"
bytes_value: "\\001"
# @LINT@.ThenChange(//y:z)
""".replace(
    "@LINT@", LINT
)


def _serialized(message: Any) -> bytes:
    return bytes(message.SerializeToString(deterministic=True))


def _float32(value: float) -> float:
    return float(struct.unpack("<f", struct.pack("<f", value))[0])


def _common_affixes(old: str, new: str) -> tuple[int, int]:
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


def _comments(text: str) -> list[str]:
    return [
        token.text
        for token in textproto_document.tokenize(text)
        if token.kind == textproto_document.COMMENT
    ]


class ValueEditTest(unittest.TestCase):
    def edit(
        self, *edits: textproto_save.Edit, text: str = SCALARS_TEXT
    ) -> textproto_save.EditResult:
        return textproto_save.apply_edits(
            text, scalars_pb2.Scalars, edits, "scalars.textproto"
        )

    def test_every_scalar_type(self):
        result = self.edit(
            textproto_save.SetValue("double_value", 2.25),
            textproto_save.SetValue("float_value", 0.2),
            textproto_save.SetValue("int32_value", -7),
            textproto_save.SetValue("uint64_value", 2**64 - 1),
            textproto_save.SetValue("bool_value", False),
            textproto_save.SetValue("string_value", 'quo"te\n'),
            textproto_save.SetValue("bytes_value", b"\x00\xff"),
            textproto_save.SetValue("sint64_value", -(2**63)),
            textproto_save.SetValue("fixed32_value", 7.0),
        )
        message = result.message
        self.assertEqual(message.double_value, 2.25)
        self.assertEqual(message.float_value, _float32(0.2))
        self.assertEqual(message.int32_value, -7)
        self.assertEqual(message.uint64_value, 2**64 - 1)
        self.assertFalse(message.bool_value)
        self.assertEqual(message.string_value, 'quo"te\n')
        self.assertEqual(message.bytes_value, b"\x00\xff")
        self.assertEqual(message.sint64_value, -(2**63))
        self.assertEqual(message.fixed32_value, 7)
        self.assertIn("float_value: 0.2\n", result.text)
        self.assertIn('string_value: "quo\\"te\\n"\n', result.text)
        self.assertIn(f"double_value: 2.25  # {LINT}.IfChange(x)\n", result.text)
        self.assertEqual(_comments(result.text), _comments(SCALARS_TEXT))

    def test_a_value_the_file_holds_keeps_its_spelling(self):
        result = self.edit(
            textproto_save.SetValue("double_value", 1.5),
            textproto_save.SetValue("float_value", _float32(0.1)),
            textproto_save.SetValue("float_value", 0.1),
            textproto_save.SetValue("int32_value", 5.0),
            textproto_save.SetValue("uint64_value", 16),
            textproto_save.SetValue("bool_value", True),
            textproto_save.SetValue("string_value", "ab"),
            textproto_save.SetValue("bytes_value", b"\x01"),
        )
        self.assertFalse(result.changed)
        self.assertEqual(result.text, SCALARS_TEXT)

    def test_doubles_take_the_shortest_digits_and_floats_the_shortest_float_digits(
        self,
    ):
        result = self.edit(
            textproto_save.SetValue("double_value", 0.1 + 0.2),
            textproto_save.SetValue("float_value", 1.0 / 3.0),
        )
        self.assertIn("double_value: 0.30000000000000004 ", result.text)
        self.assertIn("float_value: 0.33333334\n", result.text)
        self.assertEqual(result.message.float_value, _float32(1.0 / 3.0))
        special = self.edit(
            textproto_save.SetValue("double_value", -0.0),
            textproto_save.SetValue("float_value", math.inf),
        )
        self.assertIn("double_value: -0.0 ", special.text)
        self.assertIn("float_value: inf\n", special.text)
        self.assertEqual(math.copysign(1.0, special.message.double_value), -1.0)
        nan = self.edit(textproto_save.SetValue("double_value", math.nan))
        self.assertTrue(math.isnan(nan.message.double_value))

    def test_values_of_the_wrong_type_or_range_are_refused(self):
        for path, value, problem in (
            ("int32_value", 2**31, "out of the range"),
            ("uint32_value", -1, "out of the range"),
            ("int32_value", True, "is an integer"),
            ("int32_value", 1.5, "is an integer"),
            ("double_value", "1", "is a number"),
            ("double_value", False, "is a number"),
            ("float_value", 1e39, "out of the range of the float"),
            ("bool_value", 1, "is a bool"),
            ("string_value", b"x", "is a string"),
            ("bytes_value", "x", "is bytes"),
            ("no_such_field", 1, "has no field 'no_such_field'"),
            ("double_value[0]", 1.0, "is not repeated"),
            ("double_value.x", 1.0, "is a value, not a block"),
        ):
            with self.subTest(path=path), self.assertRaises(
                textproto_save.SaveError
            ) as caught:
                self.edit(textproto_save.SetValue(path, value))
            self.assertIn(problem, str(caught.exception))
            self.assertTrue(
                str(caught.exception).startswith(f"scalars.textproto: {path}: ")
            )

    def test_absent_fields_are_inserted(self):
        result = self.edit(textproto_save.SetValue("sfixed32_value", -3))
        self.assertEqual(result.text, SCALARS_TEXT + "sfixed32_value: -3\n")
        self.assertEqual(result.message.sfixed32_value, -3)

    def test_edit_leaves_the_parsed_file_as_it_was(self):
        message = scalars_pb2.Scalars()
        nproto_textproto.parse_textproto(SCALARS_TEXT, message, "scalars.textproto")
        document = textproto_document.parse(SCALARS_TEXT, "scalars.textproto")
        before = _serialized(message)
        first = textproto_save.edit(
            document, message, [textproto_save.SetValue("int32_value", 6)]
        )
        second = textproto_save.edit(
            document, message, [textproto_save.SetValue("int32_value", 7)]
        )
        self.assertEqual(
            (first.message.int32_value, second.message.int32_value), (6, 7)
        )
        self.assertEqual(document.text(), SCALARS_TEXT)
        self.assertEqual(_serialized(message), before)

    def test_a_file_that_does_not_parse_strictly_is_refused(self):
        for text in (
            "doubel_value: 1\n",
            "double_value: 1\ndouble_value: 2\n",
            "double_value: 'x'\n",
            "a {",
        ):
            with self.subTest(text=text), self.assertRaises(textproto_save.SaveError):
                textproto_save.apply_edits(
                    text, scalars_pb2.Scalars, [], "bad.textproto"
                )

    def test_an_edited_text_that_is_not_the_edited_message_is_refused(self):
        # A writer that spells every value as 42 stands for any bug that would make the text mean something else.
        with mock.patch.object(textproto_save, "_literal", return_value=42):
            with self.assertRaises(textproto_save.SaveError) as caught:
                self.edit(textproto_save.SetValue("int32_value", 6))
        self.assertIn(
            "the edited text is not the edited message, so nothing is written",
            str(caught.exception),
        )
        self.assertIn("int32_value", str(caught.exception))
        with mock.patch.object(textproto_save, "_literal", return_value="text"):
            with self.assertRaises(textproto_save.SaveError) as caught:
                self.edit(textproto_save.SetValue("int32_value", 6))
        self.assertIn("the edited text does not parse", str(caught.exception))


class StructureEditTest(unittest.TestCase):
    OUTER = """inner {
  kind: KIND_STANCE
  leaf { label: "x" }
}
inners { kind: KIND_SWING leaf { label: "a" } }
inners {
  leaf { label: "b" }
  weights: [1, 2]
}
kind: 1
"""

    def edit(
        self, *edits: textproto_save.Edit, text: str = OUTER
    ) -> textproto_save.EditResult:
        return textproto_save.apply_edits(
            text, outer_pb2.Outer, edits, "outer.textproto"
        )

    def test_enums_by_name_or_number(self):
        result = self.edit(
            textproto_save.SetValue("inner.kind", "KIND_SWING"),
            textproto_save.SetValue("kind", 2),
        )
        self.assertIn("  kind: KIND_SWING\n", result.text)
        self.assertIn("kind: KIND_SWING\n", result.text.split("\n", 9)[-1])
        self.assertFalse(
            self.edit(textproto_save.SetValue("kind", "KIND_STANCE")).changed
        )
        with self.assertRaises(textproto_save.SaveError) as caught:
            self.edit(textproto_save.SetValue("kind", "KIND_RUN"))
        self.assertIn(
            "KIND_UNSPECIFIED, KIND_STANCE, KIND_SWING", str(caught.exception)
        )
        with self.assertRaises(textproto_save.SaveError):
            self.edit(textproto_save.SetValue("kind", 9))

    def test_elements_by_index_and_by_key(self):
        result = self.edit(
            textproto_save.SetValue("inners[1].weights[0]", 1.5),
            textproto_save.SetValue("inners[1].weights[2]", 3.0),
            textproto_save.SetValue("inners[0].leaf.label", "c"),
        )
        self.assertEqual(list(result.message.inners[1].weights), [1.5, 2.0, 3.0])
        self.assertIn("  weights: [1.5, 2, 3.0]\n", result.text)
        self.assertEqual(result.message.inners[0].leaf.label, "c")
        with self.assertRaises(textproto_save.SaveError):
            self.edit(textproto_save.SetValue("inners[1].weights[5]", 1.0))
        with self.assertRaises(textproto_save.SaveError):
            self.edit(textproto_save.SetValue("inners[1].weights", 1.0))
        with self.assertRaises(textproto_save.SaveError):
            self.edit(textproto_save.SetValue("inners.kind", 1))

    def test_append_values_and_blocks(self):
        result = self.edit(
            textproto_save.AppendValue("inners[1].weights", 3),
            textproto_save.AppendValue(
                "inners",
                {"kind": "KIND_SWING", "weights": [4.0], "leaf": {"label": "d"}},
            ),
            textproto_save.AppendValue("inners", {}),
        )
        self.assertEqual(len(result.message.inners), 4)
        self.assertEqual(list(result.message.inners[1].weights), [1.0, 2.0, 3.0])
        self.assertEqual(result.message.inners[2].leaf.label, "d")
        # After a block over several lines, a new block is written over several lines too.
        self.assertTrue(
            result.text.endswith(
                'inners {\n  kind: KIND_SWING\n  weights: 4.0\n  leaf {\n    label: "d"\n  }\n}\ninners {}\nkind: 1\n'
            )
        )
        one_line = textproto_save.apply_edits(
            "inners { kind: KIND_STANCE }\n",
            outer_pb2.Outer,
            [textproto_save.AppendValue("inners", {"weights": [1.0]})],
            "o",
        )
        self.assertEqual(
            one_line.text, "inners { kind: KIND_STANCE }\ninners { weights: 1.0 }\n"
        )
        for edit in (
            textproto_save.AppendValue("kind", 1),
            textproto_save.AppendValue("inners", 1),
            textproto_save.AppendValue("inners[0].weights", {"x": 1}),
            textproto_save.AppendValue("inners", {"weights": 1.0}),
            textproto_save.AppendValue("inners[0]", {}),
        ):
            with self.subTest(edit=edit), self.assertRaises(textproto_save.SaveError):
                self.edit(edit)

    def test_remove_fields_and_elements(self):
        result = self.edit(
            textproto_save.RemoveField("inners[0]"),
            textproto_save.RemoveField("inner.leaf"),
            textproto_save.RemoveField("inners[0].weights[1]"),
            textproto_save.RemoveField("absent.field"),
            textproto_save.RemoveField("leaf"),
        )
        self.assertEqual(len(result.message.inners), 1)
        self.assertFalse(result.message.inner.HasField("leaf"))
        self.assertEqual(list(result.message.inners[0].weights), [1.0])
        cleared = self.edit(textproto_save.RemoveField("inners"))
        self.assertEqual(len(cleared.message.inners), 0)
        self.assertNotIn("inners", cleared.text)

    def test_maps_are_refused(self):
        with self.assertRaises(textproto_save.SaveError) as caught:
            self.edit(textproto_save.SetValue("inner_by_name[key=a].kind", 1))
        self.assertIn("is a map", str(caught.exception))

    def test_a_second_alternative_of_a_oneof_is_refused(self):
        with self.assertRaises(textproto_save.SaveError) as caught:
            textproto_save.apply_edits(
                "radius: 1\n",
                oneofs_pb2.Oneofs,
                [textproto_save.SetValue("side", 2.0)],
                "o",
            )
        self.assertIn("does not parse", str(caught.exception))
        swapped = textproto_save.apply_edits(
            "radius: 1\n",
            oneofs_pb2.Oneofs,
            [
                textproto_save.RemoveField("radius"),
                textproto_save.SetValue("side", 2.0),
            ],
            "o",
        )
        self.assertEqual(swapped.message.WhichOneof("shape"), "side")


class KeyedElementTest(unittest.TestCase):
    TASK = """# proto-message: nproto_test.RepeatedFields
scalars {
  string_value: "back_bkz"
  double_value: 5
}
scalars { string_value: "l_leg_hpx" double_value: 25 }  # hip roll
"""

    def edit(self, *edits: textproto_save.Edit) -> textproto_save.EditResult:
        return textproto_save.apply_edits(
            self.TASK, repeated_fields_pb2.RepeatedFields, edits, "keyed.textproto"
        )

    def test_keyed_elements_are_edited_and_created(self):
        result = self.edit(
            textproto_save.SetValue("scalars[string_value=back_bkz].double_value", 6.0),
            textproto_save.SetValue(
                "scalars[string_value=r_leg_hpx].double_value", 30.0
            ),
        )
        self.assertEqual(
            [s.string_value for s in result.message.scalars],
            ["back_bkz", "l_leg_hpx", "r_leg_hpx"],
        )
        self.assertEqual(
            [s.double_value for s in result.message.scalars], [6.0, 25.0, 30.0]
        )
        self.assertTrue(
            result.text.endswith(
                'scalars { string_value: "l_leg_hpx" double_value: 25 }  # hip roll\n'
                'scalars { string_value: "r_leg_hpx" double_value: 30.0 }\n'
            )
        )
        removed = self.edit(
            textproto_save.RemoveField("scalars[string_value=l_leg_hpx]")
        )
        self.assertEqual(len(removed.message.scalars), 1)
        self.assertNotIn("hip roll", removed.text)

    def test_edits_after_a_key_change_mean_what_they_mean_one_after_another(self):
        rename = textproto_save.SetValue(
            "scalars[string_value=back_bkz].string_value", "back_bky"
        )
        renamed = self.edit(
            rename,
            textproto_save.SetValue("scalars[string_value=back_bky].double_value", 7.0),
        )
        self.assertEqual(
            [(s.string_value, s.double_value) for s in renamed.message.scalars],
            [("back_bky", 7.0), ("l_leg_hpx", 25.0)],
        )
        created = self.edit(
            rename,
            textproto_save.SetValue("scalars[string_value=back_bkz].double_value", 1.0),
        )
        self.assertEqual(
            [(s.string_value, s.double_value) for s in created.message.scalars],
            [("back_bky", 5.0), ("l_leg_hpx", 25.0), ("back_bkz", 1.0)],
        )

    def test_many_values_are_written_together(self):
        edits = (
            textproto_save.SetValue("scalars[0].double_value", 6.0),
            textproto_save.SetValue("scalars[1].double_value", 26.0),
            textproto_save.SetValue(
                "scalars[1].int32_value", 3
            ),  # Absent: inserted, after the two are written.
        )
        original = textproto_document.TextprotoDocument.set_scalars
        with mock.patch.object(
            textproto_document.TextprotoDocument,
            "set_scalars",
            autospec=True,
            side_effect=original,
        ) as set_scalars:
            result = self.edit(*edits)
        self.assertEqual([s.double_value for s in result.message.scalars], [6.0, 26.0])
        self.assertEqual(result.message.scalars[1].int32_value, 3)
        written = [call.args[1] for call in set_scalars.call_args_list if call.args[1]]
        self.assertEqual([len(values) for values in written], [2])

    def test_a_key_must_be_a_value_and_select_one_element(self):
        with self.assertRaises(textproto_save.SaveError):
            self.edit(textproto_save.SetValue("scalars[no_field=x].double_value", 1.0))
        with self.assertRaises(textproto_save.SaveError):
            self.edit(
                textproto_save.SetValue("scalars[int32_value=3].double_value", 1.0)
            )  # not a string: not created
        twice = self.TASK + 'scalars { string_value: "back_bkz" }\n'
        with self.assertRaises(textproto_save.SaveError):
            textproto_save.apply_edits(
                twice,
                repeated_fields_pb2.RepeatedFields,
                [
                    textproto_save.SetValue(
                        "scalars[string_value=back_bkz].double_value", 1.0
                    )
                ],
                "keyed.textproto",
            )


class FileTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(dir=os.environ.get("TEST_TMPDIR"))
        self.addCleanup(shutil.rmtree, self.directory)
        self.path = os.path.join(self.directory, "scalars.textproto")
        with open(self.path, "w", encoding="utf-8") as file:
            file.write(SCALARS_TEXT)
        os.chmod(self.path, 0o640)

    def contents(self, path: str | None = None) -> str:
        with open(path or self.path, encoding="utf-8", newline="") as file:
            return file.read()

    def test_save_replaces_the_file_in_one_rename(self):
        inode = os.stat(self.path).st_ino
        result = textproto_save.save(
            self.path, scalars_pb2.Scalars, [textproto_save.SetValue("int32_value", 6)]
        )
        self.assertTrue(result.changed)
        self.assertEqual(
            self.contents(), SCALARS_TEXT.replace("int32_value: 5", "int32_value: 6")
        )
        self.assertNotEqual(os.stat(self.path).st_ino, inode)
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o640)
        self.assertEqual(sorted(os.listdir(self.directory)), ["scalars.textproto"])

    def test_an_unchanged_file_is_not_written(self):
        inode = os.stat(self.path).st_ino
        result = textproto_save.save(
            self.path, scalars_pb2.Scalars, [textproto_save.SetValue("int32_value", 5)]
        )
        self.assertFalse(result.changed)
        self.assertEqual(os.stat(self.path).st_ino, inode)

    def test_a_refused_edit_leaves_the_file_untouched(self):
        with self.assertRaises(textproto_save.SaveError):
            textproto_save.save(
                self.path,
                scalars_pb2.Scalars,
                [textproto_save.SetValue("int32_value", "x")],
            )
        self.assertEqual(self.contents(), SCALARS_TEXT)

    def test_a_failed_rename_leaves_the_file_and_no_temporary(self):
        with mock.patch.object(
            textproto_save.os, "replace", side_effect=OSError("disk full")
        ):
            with self.assertRaises(OSError):
                textproto_save.write_atomically(self.path, "int32_value: 7\n")
        self.assertEqual(self.contents(), SCALARS_TEXT)
        self.assertEqual(sorted(os.listdir(self.directory)), ["scalars.textproto"])

    def test_a_link_is_written_through_and_a_new_file_created(self):
        link = os.path.join(self.directory, "link.textproto")
        os.symlink(self.path, link)
        textproto_save.write_atomically(link, "int32_value: 8\n")
        self.assertTrue(os.path.islink(link))
        self.assertEqual(self.contents(), "int32_value: 8\n")
        new = os.path.join(self.directory, "new.textproto")
        textproto_save.write_atomically(new, "a: 1\r\n")
        self.assertEqual(self.contents(new), "a: 1\r\n")
        self.assertEqual(stat.S_IMODE(os.stat(new).st_mode), 0o644)

    def test_the_backup_is_made_once(self):
        backup = textproto_save.ensure_backup(self.path)
        self.assertEqual(backup, f"{self.path}.bak")
        textproto_save.save(
            self.path, scalars_pb2.Scalars, [textproto_save.SetValue("int32_value", 6)]
        )
        self.assertEqual(textproto_save.ensure_backup(self.path), backup)
        self.assertEqual(self.contents(backup), SCALARS_TEXT)


# ---- Properties ----


@dataclasses.dataclass(frozen=True)
class _Leaf:
    """A value of a message: its path, how to reach it by index (field, index or None), its field, its value."""

    path: str
    steps: tuple[tuple[str, int | None], ...]
    field: FieldDescriptor
    value: Any


def _is_map(field: FieldDescriptor) -> bool:
    return field.message_type is not None and bool(
        field.message_type.GetOptions().map_entry
    )


def _key_field(
    descriptor: descriptor_module.Descriptor, elements: list[Any]
) -> FieldDescriptor | None:
    """The first singular string field whose values are set and unique across `elements`, to select them by."""
    for field in descriptor.fields:
        if field.type != FieldDescriptor.TYPE_STRING or field.is_repeated:
            continue
        values = [getattr(element, field.name) for element in elements]
        present = (
            all(element.HasField(field.name) for element in elements)
            if field.has_presence
            else all(values)
        )
        if present and len(set(values)) == len(values):
            return field
    return None


def _leaves(message: Any, segments=(), steps=()) -> tuple[list[_Leaf], list[_Leaf]]:
    """(the scalars of `message`, its blocks and elements), each with a path that selects keyed elements by key."""
    scalars: list[_Leaf] = []
    blocks: list[_Leaf] = []
    for field, value in message.ListFields():
        if field.is_extension or _is_map(field):
            continue
        if field.message_type is None:
            if field.is_repeated:
                for index, item in enumerate(value):
                    segment = textproto_document.PathSegment(field.name, index=index)
                    path = textproto_document.format_path((*segments, segment))
                    scalars.append(
                        _Leaf(path, (*steps, (field.name, index)), field, item)
                    )
            else:
                path = textproto_document.format_path(
                    (*segments, textproto_document.PathSegment(field.name))
                )
                scalars.append(_Leaf(path, (*steps, (field.name, None)), field, value))
            continue
        elements = list(value) if field.is_repeated else [value]
        key_field = (
            _key_field(field.message_type, elements) if field.is_repeated else None
        )
        for index, element in enumerate(elements):
            if not field.is_repeated:
                segment = textproto_document.PathSegment(field.name)
            elif key_field is not None:
                segment = textproto_document.PathSegment(
                    field.name, key=(key_field.name, getattr(element, key_field.name))
                )
            else:
                segment = textproto_document.PathSegment(field.name, index=index)
            child_steps = (*steps, (field.name, index if field.is_repeated else None))
            path = textproto_document.format_path((*segments, segment))
            blocks.append(_Leaf(path, child_steps, field, element))
            child_scalars, child_blocks = _leaves(
                element, (*segments, segment), child_steps
            )
            scalars += child_scalars
            blocks += child_blocks
    return scalars, blocks


def _parent(message: Any, steps: tuple[tuple[str, int | None], ...]) -> Any:
    current = message
    for name, index in steps[:-1]:
        current = (
            getattr(current, name) if index is None else getattr(current, name)[index]
        )
    return current


def _set(message: Any, steps: tuple[tuple[str, int | None], ...], value: Any) -> None:
    parent = _parent(message, steps)
    name, index = steps[-1]
    if index is None:
        setattr(parent, name, value)
    else:
        getattr(parent, name)[index] = value


def _clear(message: Any, steps: tuple[tuple[str, int | None], ...]) -> None:
    parent = _parent(message, steps)
    name, index = steps[-1]
    if index is None:
        parent.ClearField(name)
    else:
        del getattr(parent, name)[index]


def _mutated(field: FieldDescriptor, value: Any) -> Any:
    """A value of `field` other than `value`, in the type reflection holds."""
    if field.type == FieldDescriptor.TYPE_BOOL:
        return not value
    if field.type in _INTEGER_RANGES:
        assert isinstance(value, int)
        return value + 1 if value < _INTEGER_RANGES[field.type][1] else value - 1
    if field.type in (FieldDescriptor.TYPE_DOUBLE, FieldDescriptor.TYPE_FLOAT):
        assert isinstance(value, float)
        changed = value * 1.5 + 0.25 if math.isfinite(value) else 1.0
        if field.type == FieldDescriptor.TYPE_FLOAT:
            changed = _float32(changed)
        return (
            changed
            if struct.pack("<d", changed) != struct.pack("<d", value)
            else changed + 1.0
        )
    if field.type == FieldDescriptor.TYPE_STRING:
        return f"{value}x"
    if field.type == FieldDescriptor.TYPE_BYTES:
        assert isinstance(value, bytes)
        return value + b"x"
    numbers = [enum_value.number for enum_value in field.enum_type.values]
    return (
        numbers[(numbers.index(value) + 1) % len(numbers)]
        if value in numbers
        else numbers[0]
    )


def _random_scalar(field: FieldDescriptor, rng: random.Random) -> Any:
    """A random value of `field`'s type, extremes and special floats included."""
    if field.type == FieldDescriptor.TYPE_BOOL:
        return rng.random() < 0.5
    if field.type in _INTEGER_RANGES:
        low, high = _INTEGER_RANGES[field.type]
        return rng.choice([0, 1, low, high, rng.randint(max(low, -1000), 1000)])
    if field.type in (FieldDescriptor.TYPE_DOUBLE, FieldDescriptor.TYPE_FLOAT):
        value = rng.choice(
            [
                0.0,
                -0.0,
                1.0,
                1.5,
                -2.25,
                1e-5,
                1e20,
                math.inf,
                -math.inf,
                math.nan,
                rng.uniform(-1e3, 1e3),
            ]
        )
        return _float32(value) if field.type == FieldDescriptor.TYPE_FLOAT else value
    if field.type == FieldDescriptor.TYPE_STRING:
        return rng.choice(
            [
                "",
                "bus",
                "l_leg_aky",
                'q"uo\\te',
                "tab\there",
                "hé",
                f"name_{rng.randint(0, 99)}",
            ]
        )
    if field.type == FieldDescriptor.TYPE_BYTES:
        return bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 4)))
    return rng.choice(field.enum_type.values).number


def _random_message(message: Any, rng: random.Random, depth: int = 0) -> None:
    """Fills `message` with random fields: a few of each kind, every required one, one alternative of a oneof."""
    descriptor = message.DESCRIPTOR
    taken_oneofs: set[str] = set()
    for field in descriptor.fields:
        if field.GetOptions().deprecated or _is_map(field):
            continue
        if field.containing_oneof is not None:
            if field.containing_oneof.name in taken_oneofs or rng.random() < 0.5:
                continue
            taken_oneofs.add(field.containing_oneof.name)
        elif not field.is_required and rng.random() < 0.3:
            continue
        if field.message_type is not None:
            if depth >= 3:
                continue
            if field.is_repeated:
                for _ in range(rng.randint(0, 3)):
                    _random_message(getattr(message, field.name).add(), rng, depth + 1)
            else:
                getattr(message, field.name).SetInParent()
                _random_message(getattr(message, field.name), rng, depth + 1)
        elif field.is_repeated:
            for _ in range(rng.randint(0, 3)):
                getattr(message, field.name).append(_random_scalar(field, rng))
        else:
            setattr(message, field.name, _random_scalar(field, rng))


def _spelling(field: FieldDescriptor, value: Any, rng: random.Random) -> str:
    """One of the textproto spellings of `value`."""
    if field.type == FieldDescriptor.TYPE_BOOL:
        return rng.choice(["true", "True", "t"] if value else ["false", "False", "f"])
    if field.type == FieldDescriptor.TYPE_ENUM:
        name = field.enum_type.values_by_number[value].name
        return rng.choice([name, name, str(value)])
    if field.type in _INTEGER_RANGES:
        assert isinstance(value, int)
        return rng.choice([str(value), hex(value) if value >= 0 else str(value)])
    if field.type in (FieldDescriptor.TYPE_DOUBLE, FieldDescriptor.TYPE_FLOAT):
        assert isinstance(value, float)
        if not math.isfinite(value):
            return rng.choice(
                [
                    textproto_document.format_double(value),
                    textproto_document.format_double(value).upper(),
                ]
            )
        if field.type == FieldDescriptor.TYPE_FLOAT:
            text = textproto_save.format_float32(value)
            return rng.choice(
                [
                    text,
                    f"{value:.9g}",
                    f"{text}f" if "e" not in text and "." in text else text,
                ]
            )
        options = [
            repr(value),
            f"{value:.17g}",
            f"{value:e}" if float(f"{value:e}") == value else repr(value),
        ]
        if value.is_integer() and abs(value) < 1e15 and math.copysign(1.0, value) > 0:
            options.append(str(int(value)))
        return rng.choice(options)
    if field.type == FieldDescriptor.TYPE_BYTES:
        assert isinstance(value, bytes)
        return textproto_document.quote_bytes(value)
    assert isinstance(value, str)
    quoted = textproto_document.quote_string(value)
    if len(value) > 1 and rng.random() < 0.3:
        middle = len(value) // 2
        return f"{textproto_document.quote_string(value[:middle])} {textproto_document.quote_string(value[middle:])}"
    return quoted


def _render(message: Any, rng: random.Random, indent: str = "") -> list[str]:
    """The lines of `message` in random layouts: comments, one-line and multi-line blocks, lists, separators."""
    lines: list[str] = []
    for field, value in message.ListFields():
        if rng.random() < 0.2:
            lines.append(
                f"{indent}# {rng.choice(['a comment', f'{LINT}.IfChange(label)', f'{LINT}.ThenChange(//a:b)'])}"
            )
        trailing = rng.choice(["", "", "  # trailing comment"])
        if field.message_type is None:
            values = list(value) if field.is_repeated else [value]
            if field.is_repeated and rng.random() < 0.3:
                spelled = ", ".join(_spelling(field, item, rng) for item in values)
                lines.append(f"{indent}{field.name}: [{spelled}]{trailing}")
                continue
            for item in values:
                separator = rng.choice(["", "", ",", ";"])
                lines.append(
                    f"{indent}{field.name}: {_spelling(field, item, rng)}{separator}{trailing}"
                )
            continue
        for element in list(value) if field.is_repeated else [value]:
            inner = _render(element, rng, f"{indent}  ")
            one_line = all("#" not in line for line in inner) and rng.random() < 0.4
            if one_line:
                body = " ".join(line.strip() for line in inner)
                padded = f" {body} " if body else ""
                lines.append(f"{indent}{field.name} {{{padded}}}{trailing}")
            elif rng.random() < 0.15:
                lines += [f"{indent}{field.name}: <", *inner, f"{indent}>"]
            else:
                lines += [f"{indent}{field.name} {{{trailing}", *inner, f"{indent}}}"]
    return lines


def _random_file(message_class: Any, rng: random.Random) -> tuple[str, Any]:
    """A random message of `message_class` and a text of it, which parses back to it."""
    message = message_class()
    _random_message(message, rng)
    text = "".join(f"{line}\n" for line in _render(message, rng))
    return text, message


def _absent_scalars(
    message: Any, prefix: str = "", steps=(), depth: int = 0
) -> list[_Leaf]:
    """Singular scalars a file can be given: absent ones of present blocks, and those of absent blocks one level down."""
    found: list[_Leaf] = []
    descriptor = message.DESCRIPTOR
    present = {field.name for field, _ in message.ListFields()}
    for field in descriptor.fields:
        if (
            field.is_repeated
            or field.GetOptions().deprecated
            or field.containing_oneof is not None
        ):
            continue
        path = f"{prefix}.{field.name}" if prefix else field.name
        if field.message_type is None:
            if field.name not in present:
                found.append(_Leaf(path, (*steps, (field.name, None)), field, None))
        elif depth < 2 and not message.HasField(field.name):
            found += _absent_scalars(
                getattr(message, field.name),
                path,
                (*steps, (field.name, None)),
                depth + 1,
            )
    return found


# How many values of a file the single-edit properties take, spread over the file (a robot's task file has about a
# thousand; each edit parses the file twice).
SAMPLE_SIZE = 30


def _sample(items: list) -> list:
    """At most SAMPLE_SIZE of `items`, evenly spread, in order."""
    if len(items) <= SAMPLE_SIZE:
        return items
    return [items[index * len(items) // SAMPLE_SIZE] for index in range(SAMPLE_SIZE)]


def _is_key(leaf: _Leaf) -> bool:
    """Whether `leaf` is the key field its element is selected by (`gains[joint=x].joint`)."""
    segments = textproto_document.parse_path(leaf.path)
    return (
        len(segments) > 1
        and segments[-2].key is not None
        and segments[-2].key[0] == segments[-1].name
    )


def _gaps(document: textproto_document.TextprotoDocument) -> list[str]:
    """The text between the values of `document`: what an edit of values only leaves as it is."""
    text = document.text()
    gaps = []
    position = 0
    pending = [document.root]
    nodes = []
    while pending:
        block = pending.pop()
        for field in block.fields:
            values = (
                field.value.elements
                if isinstance(field.value, textproto_document.ListNode)
                else (field.value,)
            )
            for value in values:
                if isinstance(value, textproto_document.ScalarNode):
                    nodes.append(value)
                else:
                    pending.append(value)
    for node in sorted(nodes, key=lambda node: node.start):
        gaps.append(text[position : node.start])
        position = node.end
    gaps.append(text[position:])
    return gaps


class PropertyTest(unittest.TestCase):
    def check_file(
        self, source: str, text: str, message_class: Any, rng: random.Random
    ) -> None:
        """Every property of the module docstring on one file: every value at once, then each of a sample."""
        original = message_class()
        nproto_textproto.parse_textproto(text, original, source)
        document = textproto_document.parse(text, source)
        scalars, blocks = _leaves(original)
        for leaf in scalars:
            with self.subTest(source=source, unchanged=leaf.path):
                same = textproto_save.edit(
                    document, original, [textproto_save.SetValue(leaf.path, leaf.value)]
                )
                self.assertFalse(same.changed)
                self.assertEqual(same.text, text)
        with self.subTest(source=source, edit="every value at once"):
            expected = message_class()
            expected.CopyFrom(original)
            edits = []
            # A key goes last: the paths of the other values of its element select the element by it.
            for leaf in sorted(scalars, key=_is_key):
                value = _mutated(leaf.field, leaf.value)
                _set(expected, leaf.steps, value)
                edits.append(textproto_save.SetValue(leaf.path, value))
            result = textproto_save.edit(document, original, edits)
            self.assertEqual(_serialized(result.message), _serialized(expected))
            self.assertEqual(
                _gaps(textproto_document.parse(result.text)), _gaps(document)
            )
        for leaf in _sample(scalars):
            with self.subTest(source=source, path=leaf.path):
                node = document.scalar(leaf.path)
                value = _mutated(leaf.field, leaf.value)
                result = textproto_save.edit(
                    document, original, [textproto_save.SetValue(leaf.path, value)]
                )
                expected = message_class()
                expected.CopyFrom(original)
                _set(expected, leaf.steps, value)
                self.assertEqual(_serialized(result.message), _serialized(expected))
                self.assertEqual(result.text[: node.start], text[: node.start])
                self.assertEqual(
                    result.text[len(result.text) - (len(text) - node.end) :],
                    text[node.end :],
                )
                self.assertEqual(_comments(result.text), _comments(text))
        for leaf in _sample(_absent_scalars(original)):
            with self.subTest(source=source, inserted=leaf.path):
                value = _random_scalar(leaf.field, rng)
                result = textproto_save.edit(
                    document, original, [textproto_save.SetValue(leaf.path, value)]
                )
                expected = message_class()
                expected.CopyFrom(original)
                _set(expected, leaf.steps, value)
                self.assertEqual(_serialized(result.message), _serialized(expected))
                prefix, suffix = _common_affixes(text, result.text)
                self.assertEqual(prefix + suffix, len(text))
                self.assertEqual(_comments(result.text), _comments(text))
        for leaf in _sample(
            [leaf for leaf in scalars + blocks if not leaf.field.is_required]
        ):
            with self.subTest(source=source, removed=leaf.path):
                result = textproto_save.edit(
                    document, original, [textproto_save.RemoveField(leaf.path)]
                )
                expected = message_class()
                expected.CopyFrom(original)
                _clear(expected, leaf.steps)
                self.assertEqual(_serialized(result.message), _serialized(expected))
                prefix, suffix = _common_affixes(text, result.text)
                self.assertEqual(prefix + suffix, len(result.text))

    def test_random_files_of_every_schema(self):
        rng = random.Random(20261003)
        for message_class in FILE_MESSAGES + TEST_MESSAGES:
            for index in range(4):
                text, message = _random_file(message_class, rng)
                source = f"{message_class.DESCRIPTOR.full_name} {index}"
                with self.subTest(source=source):
                    parsed = message_class()
                    nproto_textproto.parse_textproto(text, parsed, source)
                    self.assertEqual(_serialized(parsed), _serialized(message), text)
                self.check_file(source, text, message_class, rng)

    def test_the_shipped_configuration_files(self):
        checked = []
        rng = random.Random(5)
        for path, message_class in _shipped_files():
            with open(path, encoding="utf-8", newline="") as file:
                text = file.read()
            self.check_file(path, text, message_class, rng)
            checked.append(message_class.DESCRIPTOR.full_name)
        for message_class in FILE_MESSAGES[:5]:
            self.assertIn(message_class.DESCRIPTOR.full_name, checked)


def _shipped_files() -> list[tuple[str, Any]]:
    """(path, message class) of every configuration textproto of the data whose schema the build has.

    A file's schema is its `# proto-message:` header, in the module its `# proto-file:` header names. A schema of
    humanoid_mpc_msgs the build does not have leaves its file out; every file of a humanoid_mpc_config schema is in.

    Returns:
      The files, in the order of the walk.
    """
    found = []
    for directory, _, names in os.walk(ROOT):
        for name in sorted(names):
            if not name.endswith(".textproto"):
                continue
            path = os.path.join(directory, name)
            with open(path, encoding="utf-8") as file:
                text = file.read()
            header = nproto_textproto.header_message(text)
            if header is None or not header[1].startswith(
                ("humanoid_mpc_config.", "humanoid_mpc_msgs.")
            ):
                continue
            proto_file = _header_proto_file(text)
            module = (
                proto_file.removeprefix("humanoid_nmpc/")
                .removesuffix(".proto")
                .replace("/", ".")
                + "_pb2"
            )
            if importlib.util.find_spec(module) is None:
                assert not header[1].startswith(
                    "humanoid_mpc_config."
                ), f"{path}: no module {module}"
                continue
            importlib.import_module(module)
            descriptor = descriptor_pool.Default().FindMessageTypeByName(header[1])
            found.append((path, message_factory.GetMessageClass(descriptor)))
    return found


def _header_proto_file(text: str) -> str:
    for line in text.splitlines():
        stripped = line.lstrip("# ").strip()
        if stripped.startswith(nproto_textproto.PROTO_FILE_HEADER):
            return stripped[len(nproto_textproto.PROTO_FILE_HEADER) :].strip()
    return ""


if __name__ == "__main__":
    unittest.main()
