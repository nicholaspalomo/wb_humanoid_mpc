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


"""config_schema: the parameters of a file from its schema, headless, on a test schema and on the shipped files.

The test schema (tuning_test_file.proto) holds a field of every kind with the tuning options on it; the shipped files
(as the build has them) are walked with the real schemas, and every parameter's path edits exactly its value.
"""

import math
import os
from typing import Any
import unittest

from humanoid_mpc_config import contact_planning_file_pb2
from humanoid_mpc_config import gait_file_pb2
from humanoid_mpc_config import joint_pd_gains_file_pb2
from humanoid_mpc_config import mpc_parameter_update_pb2
from humanoid_mpc_config import reference_file_pb2
from humanoid_mpc_config import task_file_pb2
from remote_control_test_proto import tuning_test_file_pb2

from config_textproto import textproto_document
from config_textproto import textproto_save
import nproto_textproto
from remote_control import config_schema

ROOT = os.path.join(os.environ.get("TEST_SRCDIR", ""), "_main")
FILE_MESSAGES = (
    task_file_pb2.TaskFile,
    reference_file_pb2.ReferenceFile,
    joint_pd_gains_file_pb2.JointPdGainsFile,
    contact_planning_file_pb2.ContactPlanningFile,
    gait_file_pb2.GaitFile,
    mpc_parameter_update_pb2.MpcParameterUpdate,
)
REGISTRIES = {
    "contact_estimator": ("cheater_sim", "always_in_contact"),
    "mpc_costs": ("terminal_cost", "dcm_terminal_cost", "com_and_acom_tracking_cost"),
}

TEXT = """# proto-file: humanoid_nmpc/remote_control/test/tuning_test_file.proto
# proto-message: remote_control_test.TuningTestFile
gain: 3.5  # [N] the gain
count: 2
mode: MODE_STAND
estimator: "always_in_contact"
robot_name: "atlas"
costs: "terminal_cost"
costs: "dcm_terminal_cost"
times: [0.1, 0.2]
weights {
  scaling: 10
  position {
    x: 1  # forward
  }
  joints { joint: "back_bkz" value: 5 }  # the back
  joints { joint: "l_leg_aky" value: 25 }
}
gains { joint: "l_leg_kny" kp: 100 kd: 2 }
points { time: 0.5 height: 0.1 }
points { time: 1.0 }
conditional_block { weight: 4 }
"""


def _parse(text: str = TEXT) -> tuple[Any, textproto_document.TextprotoDocument]:
    message = tuning_test_file_pb2.TuningTestFile()
    nproto_textproto.parse_textproto(text, message, "tuning.textproto")
    return message, textproto_document.parse(text, "tuning.textproto")


def _task(centroidal: bool) -> task_file_pb2.TaskFile:
    """A task file of the centroidal formulation (it names a centroidal_model) or of the whole-body one."""
    task = task_file_pb2.TaskFile()
    if centroidal:
        task.centroidal_model = "full_centroidal_dynamics"
    return task


def _spec(tuning: config_schema.Tuning) -> config_schema.ParameterSpec:
    """A parameter `weight` of a task file that sets it, with the tuning `tuning`."""
    field = task_file_pb2.TaskFile.DESCRIPTOR.fields_by_name["terrain_height"]
    return config_schema.ParameterSpec(
        path="weight",
        field=field,
        kind=config_schema.Kind.NUMBER,
        tuning=tuning,
        value=1.0,
        source=config_schema.ValueSource.FILE,
        block="",
        tab="",
        order=0,
        label="weight",
        row=None,
        choices=None,
        line=None,
    )


def _specs(text: str = TEXT) -> dict[str, config_schema.ParameterSpec]:
    message, document = _parse(text)
    return {
        spec.path: spec
        for spec in config_schema.parameters(message, document, REGISTRIES)
    }


class WalkTest(unittest.TestCase):
    def setUp(self):
        self.message, self.document = _parse()
        self.specs = _specs()

    def test_every_kind(self):
        expected = {
            "gain": config_schema.Kind.NUMBER,
            "ratio": config_schema.Kind.NUMBER,
            "count": config_schema.Kind.INTEGER,
            "enabled": config_schema.Kind.BOOL,
            "mode": config_schema.Kind.CHOICE,
            "estimator": config_schema.Kind.CHOICE,
            "robot_name": config_schema.Kind.TEXT,
            "costs": config_schema.Kind.NAME_LIST,
            "times[1]": config_schema.Kind.NUMBER,
            "weights.joints[joint=back_bkz].joint": config_schema.Kind.TEXT,
        }
        for path, kind in expected.items():
            with self.subTest(path=path):
                self.assertEqual(self.specs[path].kind, kind)
        self.assertTrue(self.specs["gain"].renders)
        self.assertFalse(self.specs["robot_name"].renders)
        self.assertTrue(self.specs["costs"].read_only)
        self.assertFalse(self.specs["mode"].read_only)

    def test_values_come_from_the_file_the_schema_default_or_nowhere(self):
        file_values = {
            "gain": 3.5,
            "count": 2,
            "mode": "MODE_STAND",
            "estimator": "always_in_contact",
        }
        for path, value in file_values.items():
            with self.subTest(path=path):
                self.assertEqual(self.specs[path].value, value)
                self.assertEqual(
                    self.specs[path].source, config_schema.ValueSource.FILE
                )
        self.assertEqual(
            self.specs["costs"].value, ("terminal_cost", "dcm_terminal_cost")
        )
        self.assertEqual(self.specs["enabled"].value, False)
        self.assertEqual(
            self.specs["enabled"].source, config_schema.ValueSource.DEFAULT
        )
        self.assertEqual(self.specs["ratio"].value, 0.5)
        self.assertIsNone(self.specs["offset"].value)
        self.assertEqual(self.specs["offset"].source, config_schema.ValueSource.UNSET)
        # The fields of a block the file leaves out, with their defaults.
        self.assertEqual(self.specs["conditional.rate"].value, 2.0)
        self.assertEqual(
            self.specs["conditional.rate"].source, config_schema.ValueSource.DEFAULT
        )
        self.assertEqual(
            self.specs["weights.position.y"].source, config_schema.ValueSource.DEFAULT
        )
        self.assertEqual(
            self.specs["points[1].height"].source, config_schema.ValueSource.DEFAULT
        )

    def test_an_absent_optional_block_and_a_deprecated_field_are_left_out(self):
        self.assertFalse(any(path.startswith("gate") for path in self.specs))
        self.assertNotIn("old_gain", self.specs)
        with_gate = _specs(TEXT + "gate { }\n")
        self.assertEqual(
            with_gate["gate.rate"].source, config_schema.ValueSource.DEFAULT
        )

    def test_elements_by_key_and_by_index(self):
        joint = self.specs["weights.joints[joint=l_leg_aky].value"]
        self.assertEqual(joint.value, 25)
        self.assertEqual(joint.label, "joints[l_leg_aky]")
        self.assertEqual(joint.row, "l_leg_aky")
        self.assertEqual(
            self.specs["weights.joints[joint=l_leg_aky].joint"].label,
            "joints[l_leg_aky].joint",
        )
        gain = self.specs["gains[joint=l_leg_kny].kp"]
        self.assertEqual(
            (gain.label, gain.block, gain.value), ("l_leg_kny.kp", "gains", 100)
        )
        self.assertEqual(self.specs["points[1].time"].label, "points[1].time")
        self.assertEqual(self.specs["points[1].time"].row, "points[1]")
        self.assertEqual(self.specs["times[0]"].label, "times[0]")
        self.assertEqual(self.specs["weights.position.x"].label, "position.x")
        self.assertEqual(self.specs["weights.scaling"].label, "scaling")
        duplicate = _specs(TEXT + 'gains { joint: "l_leg_kny" kp: 1 }\n')
        self.assertIn(
            "gains[1].kp", duplicate
        )  # keys that are not unique select nothing: by index

    def test_options_on_the_field_and_inherited_from_its_blocks(self):
        gain = self.specs["gain"].tuning
        self.assertEqual(
            (gain.unit, gain.reload, gain.consumer),
            ("N", config_schema.RELOAD_HOT, "mpc"),
        )
        position = self.specs["weights.position.z"].tuning
        self.assertEqual(
            (position.unit, position.slider_min, position.slider_max), ("m", 0.0, 10.0)
        )
        orientation = self.specs["weights.orientation.roll"].tuning
        self.assertEqual(
            (orientation.unit, orientation.reload),
            ("rad", config_schema.RELOAD_START_UP),
        )
        self.assertTrue(self.specs["weights.scaling"].tuning.log_scale)
        self.assertEqual(self.specs["estimator"].tuning.registry, "contact_estimator")
        self.assertEqual(
            self.specs["weights.joints[joint=back_bkz].value"].tuning,
            config_schema.Tuning(),
        )
        for path in (
            "initial.scaling",
            "initial.position.x",
            "initial.orientation.yaw",
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    self.specs[path].tuning.exclude_reason, "the initial pose"
                )
                self.assertFalse(self.specs[path].renders)
        weight = self.specs["conditional_block.weight"].tuning
        self.assertEqual((weight.unit, weight.reload), ("kg", config_schema.RELOAD_HOT))
        # The formulations are inherited like the reload class, and a field that names its own replaces the block's.
        inherited = self.specs["formulation_block.weight"].tuning
        self.assertEqual(
            (inherited.formulations, inherited.reload),
            ((config_schema.WHOLE_BODY,), config_schema.RELOAD_HOT),
        )
        self.assertEqual(
            self.specs["formulation_block.centroidal_weight"].tuning.formulations,
            (config_schema.CENTROIDAL,),
        )
        self.assertEqual(self.specs["gain"].tuning.formulations, ())
        self.assertEqual(
            weight.active_when,
            (
                config_schema.Condition("enabled", ("true",)),
                config_schema.Condition("mode", ("MODE_WALK", "MODE_STAND")),
            ),
        )

    def test_tabs_of_top_level_blocks(self):
        self.assertEqual(
            (self.specs["weights.scaling"].tab, self.specs["weights.scaling"].order),
            ("Weights", 2),
        )
        self.assertEqual(self.specs["weights.scaling"].block, "weights")
        self.assertEqual((self.specs["gain"].tab, self.specs["gain"].block), ("", ""))
        self.assertEqual(self.specs["initial.scaling"].tab, "Weights")
        self.assertEqual(self.specs["points[0].time"].tab, "")

    def test_lines_of_the_values_the_file_sets(self):
        self.assertEqual(self.specs["gain"].line, 3)
        self.assertEqual(self.specs["weights.position.x"].line, 14)
        self.assertEqual(self.specs["costs"].line, 8)
        self.assertIsNone(self.specs["enabled"].line)
        message, _ = _parse()
        without_document = {
            spec.path: spec for spec in config_schema.parameters(message)
        }
        self.assertIsNone(without_document["gain"].line)
        self.assertEqual(without_document["gain"].value, 3.5)

    def test_choices(self):
        self.assertEqual(
            self.specs["mode"].choices, ("MODE_UNSPECIFIED", "MODE_WALK", "MODE_STAND")
        )
        self.assertEqual(
            self.specs["estimator"].choices, REGISTRIES["contact_estimator"]
        )
        self.assertEqual(self.specs["costs"].choices, REGISTRIES["mpc_costs"])
        message, document = _parse()
        unknown = {
            spec.path: spec for spec in config_schema.parameters(message, document)
        }
        self.assertIsNone(unknown["estimator"].choices)
        self.assertIsNone(self.specs["gain"].choices)


class RangeAndLabelTest(unittest.TestCase):
    def test_slider_range_from_the_value(self):
        plain = config_schema.Tuning()
        self.assertEqual(config_schema.slider_range(2.0, plain), (0.0, 8.0))
        self.assertEqual(config_schema.slider_range(0.1, plain), (0.0, 1.0))
        self.assertEqual(config_schema.slider_range(-3.0, plain), (-12.0, 12.0))
        self.assertEqual(config_schema.slider_range(math.nan, plain), (0.0, 1.0))
        logarithmic = config_schema.Tuning(log_scale=True)
        self.assertEqual(config_schema.slider_range(10.0, logarithmic), (0.1, 1000.0))
        self.assertEqual(config_schema.slider_range(0.0, logarithmic), (0.01, 100.0))
        bounded = config_schema.Tuning(slider_min=-1.0)
        self.assertEqual(config_schema.slider_range(2.0, bounded), (-1.0, 8.0))
        specs = _specs()
        self.assertEqual(specs["weights.position.x"].slider_range(), (0.0, 10.0))
        self.assertEqual(specs["count"].slider_range(), (0.0, 8.0))
        self.assertIsNone(specs["mode"].slider_range())
        self.assertEqual(specs["offset"].slider_range(), (0.0, 1.0))

    def test_conditions(self):
        message, _ = _parse()
        specs = _specs()
        self.assertTrue(config_schema.is_active(specs["stand_height"].tuning, message))
        self.assertFalse(
            config_schema.is_active(specs["conditional.rate"].tuning, message)
        )
        self.assertEqual(
            config_schema.unmet_conditions(
                specs["conditional_block.weight"].tuning, message
            ),
            [config_schema.Condition("enabled", ("true",))],
        )
        walking, _ = _parse(
            TEXT.replace("mode: MODE_STAND", "mode: MODE_WALK\nenabled: true")
        )
        self.assertFalse(config_schema.is_active(specs["stand_height"].tuning, walking))
        self.assertTrue(
            config_schema.is_active(specs["conditional_block.weight"].tuning, walking)
        )
        self.assertEqual(config_schema.condition_value(message, "count"), "2")
        self.assertEqual(config_schema.condition_value(message, "gain"), "3.5")
        self.assertEqual(config_schema.condition_value(message, "enabled"), "false")
        self.assertEqual(
            config_schema.condition_value(message, "weights.position.x"), "1.0"
        )
        numeric = config_schema.Tuning(
            active_when=(config_schema.Condition("count", ("2.0", "x")),)
        )
        self.assertTrue(config_schema.is_active(numeric, message))
        for path in ("nope", "weights", "costs", "weights.joints"):
            with self.subTest(path=path), self.assertRaises(config_schema.SchemaError):
                config_schema.condition_value(message, path)

    def test_annotations_and_display_labels(self):
        message, _ = _parse()
        specs = _specs()
        # The line of gain, weights.position.x and the back joint ends in a comment, which no label shows.
        self.assertEqual(
            config_schema.display_label(specs["gain"], message), "gain (N)"
        )
        self.assertEqual(
            config_schema.display_label(specs["weights.position.x"], message),
            "position.x (m)",
        )
        self.assertEqual(
            config_schema.display_label(
                specs["weights.joints[joint=back_bkz].value"], message
            ),
            "joints[back_bkz]",
        )
        self.assertEqual(
            config_schema.display_label(specs["costs"], message), "costs (restart)"
        )
        self.assertEqual(
            config_schema.display_label(specs["enabled"], message), "enabled (default)"
        )
        self.assertEqual(
            config_schema.display_label(specs["offset"], message), "offset (unset)"
        )
        self.assertEqual(
            config_schema.display_label(specs["weights.orientation.yaw"], message),
            "orientation.yaw (rad) (restart; default)",
        )
        self.assertEqual(
            config_schema.annotations(specs["conditional_block.weight"], message),
            ["not applicable: enabled is false"],
        )
        self.assertEqual(config_schema.annotations(specs["count"], message), [])
        # A formulation names who reads a field of a task file; another file's field is never "not applicable" for it.
        self.assertEqual(
            config_schema.annotations(
                specs["formulation_block.centroidal_weight"], message
            ),
            ["default"],
        )

    def test_a_hot_field_is_live_on_every_formulation(self):
        # There is no "restart" for want of a parameter updater: every formulation's MPC applies its hot fields.
        hot = config_schema.Tuning(reload=config_schema.RELOAD_HOT, consumer="mpc")
        start_up = config_schema.Tuning(reload=config_schema.RELOAD_START_UP)
        for formulation_file in (_task(centroidal=True), _task(centroidal=False)):
            with self.subTest(
                formulation=config_schema.formulation_of(formulation_file)
            ):
                self.assertNotIn(
                    "restart", config_schema.annotations(_spec(hot), formulation_file)
                )
                self.assertIn(
                    "restart",
                    config_schema.annotations(_spec(start_up), formulation_file),
                )

    def test_a_task_file_field_of_another_formulation_is_not_applicable(self):
        centroidal_only = _spec(
            config_schema.Tuning(
                reload=config_schema.RELOAD_HOT,
                formulations=(config_schema.CENTROIDAL,),
            )
        )
        whole_body_only = _spec(
            config_schema.Tuning(
                reload=config_schema.RELOAD_HOT,
                formulations=(config_schema.WHOLE_BODY,),
            )
        )
        both = _spec(
            config_schema.Tuning(
                reload=config_schema.RELOAD_HOT, formulations=config_schema.FORMULATIONS
            )
        )
        centroidal, whole_body = _task(centroidal=True), _task(centroidal=False)
        self.assertEqual(
            config_schema.formulation_of(centroidal), config_schema.CENTROIDAL
        )
        self.assertEqual(
            config_schema.formulation_of(whole_body), config_schema.WHOLE_BODY
        )
        self.assertEqual(
            config_schema.annotations(centroidal_only, whole_body),
            ["not applicable: the whole-body MPC does not read it"],
        )
        self.assertEqual(
            config_schema.annotations(whole_body_only, centroidal),
            ["not applicable: the centroidal MPC does not read it"],
        )
        for spec, task in (
            (centroidal_only, centroidal),
            (whole_body_only, whole_body),
            (both, centroidal),
            (both, whole_body),
        ):
            with self.subTest(formulations=spec.tuning.formulations):
                self.assertEqual(config_schema.annotations(spec, task), [])
        # The note is a suffix, as "restart" is: the label is still the field's name.
        self.assertEqual(
            config_schema.display_label(centroidal_only, whole_body),
            "weight (not applicable: the whole-body MPC does not read it)",
        )


class SchemaFieldsTest(unittest.TestCase):
    def test_every_scalar_of_the_test_schema(self):
        fields = {
            field.path: field
            for field in config_schema.schema_fields(
                tuning_test_file_pb2.TuningTestFile.DESCRIPTOR
            )
        }
        for path in (
            "gain",
            "costs",
            "weights.position.x",
            "weights.joints[*].value",
            "gains[*].kp",
            "points[*].height",
            "gate.rate",
            "initial.orientation.roll",
            "conditional_block.weight",
        ):
            self.assertIn(path, fields)
        self.assertNotIn("old_gain", fields)
        self.assertFalse(fields["initial.scaling"].renders)
        self.assertTrue(fields["gate.rate"].renders)
        self.assertFalse(fields["gains[*].joint"].renders)
        self.assertEqual(fields["weights.position.x"].tuning.unit, "m")

    def test_the_file_schemas_walk(self):
        for message_class in FILE_MESSAGES:
            with self.subTest(message=message_class.DESCRIPTOR.full_name):
                fields = config_schema.schema_fields(message_class.DESCRIPTOR)
                self.assertTrue(fields)
                for field in fields:
                    self.assertIsNotNone(field.field.containing_type)
                    self.assertNotIn("[*][*]", field.path)
        top_level = {
            field.path
            for field in config_schema.schema_fields(task_file_pb2.TaskFile.DESCRIPTOR)
        }
        self.assertIn("terrain_height", top_level)
        self.assertIn("contact_estimator", top_level)


class EditTest(unittest.TestCase):
    """Every parameter's path is where its value is: setting it edits that value and nothing else."""

    def check(self, message_class: Any, text: str, source: str) -> None:
        """Every parameter of the file `text` is where its path points; a sample of them edits its value, and only it."""
        message = message_class()
        nproto_textproto.parse_textproto(text, message, source)
        document = textproto_document.parse(text, source)
        specs = config_schema.parameters(message, document)
        editable = []
        for spec in specs:
            if spec.kind == config_schema.Kind.NAME_LIST:
                continue
            if spec.source == config_schema.ValueSource.FILE:
                with self.subTest(source=source, unchanged=spec.path):
                    same = textproto_save.edit(
                        document,
                        message,
                        [textproto_save.SetValue(spec.path, spec.value)],
                    )
                    self.assertFalse(same.changed)
            if (
                spec.kind != config_schema.Kind.TEXT
            ):  # Text includes the keys, whose edit moves the element's path.
                editable.append(spec)
        before = {
            spec.path: spec
            for spec in specs
            if spec.source == config_schema.ValueSource.FILE
        }
        for spec in _sample(editable):
            value = _edited_value(spec)
            with self.subTest(source=source, path=spec.path):
                result = textproto_save.edit(
                    document, message, [textproto_save.SetValue(spec.path, value)]
                )
                after = {
                    other.path: other
                    for other in config_schema.parameters(
                        result.message, textproto_document.parse(result.text)
                    )
                }
                self.assertEqual(after[spec.path].value, value)
                self.assertEqual(
                    after[spec.path].source, config_schema.ValueSource.FILE
                )
                for path, other in before.items():
                    if path != spec.path and path in after:
                        self.assertEqual(after[path].value, other.value, path)

    def test_the_test_schema(self):
        self.check(tuning_test_file_pb2.TuningTestFile, TEXT, "tuning.textproto")

    def test_the_shipped_files(self):
        found = []
        for path, message_class in _shipped_files():
            with open(path, encoding="utf-8", newline="") as file:
                self.check(message_class, file.read(), path)
            found.append(message_class.DESCRIPTOR.full_name)
        for message_class in FILE_MESSAGES[:5]:
            self.assertIn(message_class.DESCRIPTOR.full_name, found)


# How many parameters of a file the edit property takes, spread over the file.
SAMPLE_SIZE = 25


def _sample(items: list) -> list:
    """At most SAMPLE_SIZE of `items`, evenly spread, in order."""
    if len(items) <= SAMPLE_SIZE:
        return items
    return [items[index * len(items) // SAMPLE_SIZE] for index in range(SAMPLE_SIZE)]


def _edited_value(spec: config_schema.ParameterSpec) -> Any:
    """A value of the parameter other than its own."""
    if spec.kind == config_schema.Kind.BOOL:
        return not spec.value
    if spec.kind == config_schema.Kind.INTEGER:
        return (spec.value or 0) + 1
    if spec.kind == config_schema.Kind.NUMBER:
        return float(spec.value or 0.0) * 2.0 + 0.5
    if spec.kind == config_schema.Kind.CHOICE and spec.choices:
        others = [choice for choice in spec.choices if choice != spec.value]
        return others[-1]
    return f"{spec.value or ''}_edited"


def _shipped_files() -> list[tuple[str, Any]]:
    """(path, message class) of every configuration textproto of the data whose schema is a humanoid_mpc_config one."""
    found = []
    for directory, _, names in os.walk(ROOT):
        for name in sorted(names):
            if not name.endswith(".textproto"):
                continue
            path = os.path.join(directory, name)
            with open(path, encoding="utf-8") as file:
                header = nproto_textproto.header_message(file.read())
            if header is None or not header[1].startswith("humanoid_mpc_config."):
                continue
            message_name = header[1].split(".", 1)[1]
            for message_class in FILE_MESSAGES:
                if message_class.DESCRIPTOR.name == message_name:
                    found.append((path, message_class))
    return found


if __name__ == "__main__":
    unittest.main()
