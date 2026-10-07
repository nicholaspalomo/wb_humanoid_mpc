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

"""Every tunable parameter of every file schema reaches the GUI, because the GUI is built from the schemas (test T8).

There are no lists of parameter names in the tabs: config_schema walks a file's schema, and every number, bool and
registry name the schema does not exclude, with its stated reason, gets a row. This test holds that from the outside,
in two halves:

- headless, on the descriptors of the five file schemas: every scalar renders or carries its exclusion reason, every
  rendered one says whether the running stack applies it (RELOAD_HOT) or the next start (RELOAD_START_UP), every
  registry name is a choice, and every condition names a field the file has;
- with a display, on every robot's files: each tab shows a row for exactly the parameters the walk renders.

A row is labeled by its field names (the user's decision for the tuning GUI): no option of a schema renames a field, and
no comment of a file is part of a label, which the shipped files are walked headless to show.

The RELOAD_HOT set is the half of test T8 (humanoid_mpc_config/README.md) that the C++ side compares with what each
formulation's parameter updater declares it applies (humanoid_mpc_validation, testHotFieldCoverage); here it is held to
its own rules, and every formulation a field names is one the GUI knows.
"""

from collections.abc import Iterator
import glob
import os
import re
import tkinter as tk
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
import operator_test_support
from remote_control import config_schema
from remote_control import tuned_file
from remote_control.tk_app import command_limits_tab
from remote_control.tk_app import joint_pd_tab
from remote_control.tk_app import mpc_params_tab
from remote_control.tk_app import parameter_rows

FILE_SCHEMAS = (
    task_file_pb2.TaskFile,
    reference_file_pb2.ReferenceFile,
    joint_pd_gains_file_pb2.JointPdGainsFile,
    contact_planning_file_pb2.ContactPlanningFile,
    gait_file_pb2.GaitFile,
)
# The files of a robot's config/ directory that the tabs tune, with their schemas.
ROBOT_FILE_SCHEMAS = (
    (os.path.join("mpc", "task.textproto"), task_file_pb2.TaskFile),
    (
        os.path.join("mpc", "contact_planning.textproto"),
        contact_planning_file_pb2.ContactPlanningFile,
    ),
    (os.path.join("command", "reference.textproto"), reference_file_pb2.ReferenceFile),
    (
        os.path.join("controller", "joint_pd_gains.textproto"),
        joint_pd_gains_file_pb2.JointPdGainsFile,
    ),
)
RELOAD_CLASSES = (config_schema.RELOAD_HOT, config_schema.RELOAD_START_UP)
# The text fields that name an entity of the robot model (a robot, a joint, a link, a frame) or a term: there is
# nothing to choose them from, so they get no widget and need no stated reason. Any other text field without a
# registry must say why it gets no widget, so that a new string hyperparameter cannot pass unrendered unnoticed.
ENTITY_NAME_FIELDS = frozenset(
    {
        "humanoid_mpc_config.ModelSettingsConfig.robot_name",
        "humanoid_mpc_config.ModelSettingsConfig.ArmJointNames.left_shoulder_y",
        "humanoid_mpc_config.ModelSettingsConfig.ArmJointNames.right_shoulder_y",
        "humanoid_mpc_config.ModelSettingsConfig.ArmJointNames.left_elbow_y",
        "humanoid_mpc_config.ModelSettingsConfig.ArmJointNames.right_elbow_y",
        "humanoid_mpc_config.TaskSpaceCostConfig.name",
        "humanoid_mpc_config.TaskSpaceCostConfig.link_name",
        "humanoid_mpc_config.CollisionConstraintConfig.Foot.left_ankle_frame",
        "humanoid_mpc_config.CollisionConstraintConfig.Foot.right_ankle_frame",
        "humanoid_mpc_config.CollisionConstraintConfig.Knee.left_knee_frame",
        "humanoid_mpc_config.CollisionConstraintConfig.Knee.right_knee_frame",
        "humanoid_mpc_config.MimicJointsConfig.MimicJoint.parent_joint_name",
        "humanoid_mpc_config.MimicJointsConfig.MimicJoint.child_joint_name",
    }
)
CONSUMERS = ("", "gui", "robot", "mpc")


def _fields(message_class: Any) -> list[config_schema.SchemaField]:
    return config_schema.schema_fields(message_class.DESCRIPTOR)


def _condition_paths(message_class: Any) -> set[str]:
    """The fields the conditions of every field of a schema name, by their paths from its message."""
    paths: set[str] = set()
    for field in _fields(message_class):
        paths.update(condition.field_path for condition in field.tuning.active_when)
    return paths


def _is_element_key(field: config_schema.SchemaField) -> bool:
    """Whether a text field is the key of its repeated element: the element's first string field (config_schema)."""
    if not field.path.endswith(f"{config_schema.EVERY_ELEMENT}.{field.field.name}"):
        return False
    strings = [
        candidate
        for candidate in field.field.containing_type.fields
        if candidate.type == candidate.TYPE_STRING and not candidate.is_repeated
    ]
    return bool(strings) and strings[0].name == field.field.name


class TestTheSchemasSayHowEveryFieldIsTuned(unittest.TestCase):
    def test_every_field_renders_or_says_why_not(self):
        for message_class in FILE_SCHEMAS:
            for field in _fields(message_class):
                with self.subTest(
                    schema=message_class.DESCRIPTOR.name, path=field.path
                ):
                    if field.kind == config_schema.Kind.TEXT and (
                        _is_element_key(field)
                        or field.field.full_name in ENTITY_NAME_FIELDS
                    ):
                        continue  # a key, or the name of a joint, a link or a frame: nothing to choose it from
                    self.assertTrue(
                        field.renders or field.tuning.exclude_reason.strip(),
                        "neither rendered nor excluded with a reason (a string that picks from a fixed set "
                        "names its registry)",
                    )

    def test_every_entity_name_field_is_a_text_field_of_a_schema(self):
        # The exemptions above name fields that exist, and stay text: one that gains a registry leaves the list.
        text: set[str] = set()
        for message_class in FILE_SCHEMAS:
            text.update(
                field.field.full_name
                for field in _fields(message_class)
                if field.kind == config_schema.Kind.TEXT
            )
        self.assertLessEqual(ENTITY_NAME_FIELDS, text)

    def test_the_solver_selectors_are_choices_of_their_registries(self):
        fields = {field.path: field for field in _fields(task_file_pb2.TaskFile)}
        for path, registry in (
            ("multiple_shooting.integrator_type", "sensitivity_integrator"),
            ("rollout.integrator_type", "rollout_integrator"),
            ("rollout.root_finder_type", "root_finder"),
        ):
            with self.subTest(path=path):
                self.assertEqual(fields[path].kind, config_schema.Kind.CHOICE)
                self.assertEqual(fields[path].tuning.registry, registry)
                self.assertTrue(fields[path].renders)

    def test_every_rendered_field_says_who_applies_it(self):
        for message_class in FILE_SCHEMAS:
            for field in _fields(message_class):
                if not field.renders:
                    continue
                with self.subTest(
                    schema=message_class.DESCRIPTOR.name, path=field.path
                ):
                    self.assertIn(field.tuning.reload, RELOAD_CLASSES)
                    self.assertIn(field.tuning.consumer, CONSUMERS)

    def test_every_formulation_a_field_names_is_known(self):
        named: set[str] = set()
        for message_class in FILE_SCHEMAS:
            for field in _fields(message_class):
                named.update(field.tuning.formulations)
                with self.subTest(
                    schema=message_class.DESCRIPTOR.name, path=field.path
                ):
                    self.assertLessEqual(
                        set(field.tuning.formulations), set(config_schema.FORMULATIONS)
                    )
        # The task file names both, so the rule above was exercised.
        self.assertEqual(named, set(config_schema.FORMULATIONS))

    def test_the_hot_fields_are_a_proper_part_of_each_file_the_mpc_reloads(self):
        # A positive control for the rule above: the task file has both kinds, the gait file (never reloaded) none hot.
        for message_class, has_hot in (
            (task_file_pb2.TaskFile, True),
            (contact_planning_file_pb2.ContactPlanningFile, True),
            (reference_file_pb2.ReferenceFile, True),
            (joint_pd_gains_file_pb2.JointPdGainsFile, True),
            (gait_file_pb2.GaitFile, False),
        ):
            with self.subTest(schema=message_class.DESCRIPTOR.name):
                classes = {
                    field.tuning.reload
                    for field in _fields(message_class)
                    if field.kind != config_schema.Kind.TEXT
                }
                self.assertEqual(config_schema.RELOAD_HOT in classes, has_hot)

    def test_every_registry_name_is_a_choice_or_a_name_list(self):
        for message_class in FILE_SCHEMAS:
            for field in _fields(message_class):
                if not field.tuning.registry:
                    continue
                with self.subTest(
                    schema=message_class.DESCRIPTOR.name, path=field.path
                ):
                    self.assertIn(
                        field.kind,
                        (config_schema.Kind.CHOICE, config_schema.Kind.NAME_LIST),
                    )

    def test_the_parameter_update_resolves_the_task_files_conditions_inside_it(self):
        # A condition names a field of the file; inside the bus message the file is `task`.
        self.assertIn(
            "task.contact_input_parameterization",
            _condition_paths(mpc_parameter_update_pb2.MpcParameterUpdate),
        )
        self.assertIn(
            "contact_input_parameterization", _condition_paths(task_file_pb2.TaskFile)
        )

    def test_a_field_added_to_a_schema_has_a_row_with_no_gui_code(self):
        # The test schema has a field of every kind; the rows are what the tabs build for any schema.
        kinds = {
            field.kind
            for field in _fields(tuning_test_file_pb2.TuningTestFile)
            if field.renders
        }
        self.assertEqual(
            kinds,
            {
                config_schema.Kind.NUMBER,
                config_schema.Kind.INTEGER,
                config_schema.Kind.BOOL,
                config_schema.Kind.CHOICE,
                config_schema.Kind.NAME_LIST,
            },
        )


def _robot_files() -> list[tuple[str, str]]:
    """(robot package, config/ directory) of every robot."""
    return sorted(operator_test_support.ROBOT_CONFIGS.items())


def _shipped_tuned_files() -> list[tuned_file.TunedFile]:
    """Every file the tabs tune, of every robot (contact_planning only where the robot has one)."""
    files = []
    for _, config in _robot_files():
        for relative, message_class in ROBOT_FILE_SCHEMAS:
            path = os.path.join(config, relative)
            if os.path.exists(path):
                files.append(tuned_file.TunedFile(path, message_class))
    return files


def _every_field(descriptor: Any, seen: set[str]) -> Iterator[Any]:
    """Every field of `descriptor` and of the messages below it, each message once."""
    if descriptor.full_name in seen:
        return
    seen.add(descriptor.full_name)
    for field in descriptor.fields:
        yield field
        if field.message_type is not None:
            yield from _every_field(field.message_type, seen)


def _label_names(label: str) -> set[str]:
    """The names a label is made of: its dotted parts, and the key or index of each element in them."""
    names: set[str] = set()
    for part in label.split("."):
        names.update(name for name in re.split(r"[\[\]]", part) if name)
    return names


def _path_names(path: str) -> set[str]:
    """The field names of a textproto_document path, and the key or index of each element it selects."""
    names: set[str] = set()
    for segment in textproto_document.parse_path(path):
        names.add(segment.name)
        if segment.key is not None:
            names.add(segment.key[1])
        if segment.index is not None:
            names.add(str(segment.index))
    return names


_FIELD_LINE = re.compile(r"^\s*(?:repeated\s+)?[\w.]+\s+(\w+)\s*=\s*\d+")
_BLOCK_LINE = re.compile(r"^\s*(message|enum)\s+(\w+)\s*\{")


def _stated_units(proto_path: str) -> dict[str, str]:
    """The unit each field's leading comment states, "[s] ...", by the field's full name, from a .proto source.

    LINT directives are not comments of a field; a dimensionless "[-]" states no unit; "[s, robot clock]" states "s".

    Args:
      proto_path: The .proto file.

    Returns:
      Field full name -> the unit its leading comment states, for the fields whose comment states one.
    """
    units: dict[str, str] = {}
    package = ""
    blocks: list[tuple[str, int]] = []
    depth = 0
    comments: list[str] = []
    with open(proto_path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("package "):
                package = stripped[len("package ") :].rstrip(";")
            if stripped.startswith("//"):
                comment = stripped[2:].strip()
                if not comment.startswith("LINT."):
                    comments.append(comment)
                continue
            block = _BLOCK_LINE.match(line)
            if block:
                blocks.append((block.group(2), depth))
            field = _FIELD_LINE.match(line)
            if field and blocks and not stripped.startswith(("reserved", "option")):
                stated = re.match(r"\[([^\]]+)\]", comments[0]) if comments else None
                unit = stated.group(1).split(",")[0].strip() if stated else "-"
                if unit != "-":
                    name = ".".join([package] + [name for name, _ in blocks])
                    units[f"{name}.{field.group(1)}"] = unit
            comments = []
            depth += line.count("{") - line.count("}")
            while blocks and depth <= blocks[-1][1]:
                blocks.pop()
    return units


class TestTheUnitsOfTheFieldComments(unittest.TestCase):
    def test_a_field_whose_comment_states_a_unit_carries_it_in_its_tuning(self):
        sources = sorted(
            glob.glob(
                operator_test_support.repo_path(
                    "humanoid_nmpc", "humanoid_mpc_config", "*.proto"
                )
            )
        )
        self.assertTrue(sources)
        pool = task_file_pb2.DESCRIPTOR.pool
        stated = 0
        for source in sources:
            for full_name, unit in _stated_units(source).items():
                stated += 1
                with self.subTest(field=full_name):
                    own = config_schema.own_tuning(pool.FindFieldByName(full_name))
                    self.assertIsNotNone(own, f"[{unit}] in the comment, no unit")
                    self.assertEqual(own.unit, unit)
        # The schemas do state units, so the rule above was exercised.
        self.assertGreater(stated, 50)


class TestTheLabelsAreTheFieldNames(unittest.TestCase):
    def test_no_schema_sets_a_label_option(self):
        # The GUI reads no label option: a field is shown by its name, so a schema that set one would mislead.
        for message_class in FILE_SCHEMAS + (tuning_test_file_pb2.TuningTestFile,):
            for field in _every_field(message_class.DESCRIPTOR, set()):
                own = config_schema.own_tuning(field)
                if own is None:
                    continue
                with self.subTest(field=field.full_name):
                    self.assertNotIn(
                        "label", {option.name for option, _ in own.ListFields()}
                    )

    def test_every_label_is_the_field_names_and_keys_of_its_path(self):
        files = _shipped_tuned_files()
        self.assertTrue(files)
        for file in files:
            for spec in file.rendered():
                with self.subTest(file=file.path, path=spec.path):
                    self.assertTrue(spec.label)
                    self.assertLessEqual(
                        _label_names(spec.label), _path_names(spec.path)
                    )
                    self.assertTrue(
                        config_schema.display_label(spec, file.message).startswith(
                            spec.label
                        )
                    )

    def test_no_label_shows_a_comment_of_the_file(self):
        commented = 0
        for file in _shipped_tuned_files():
            for spec in file.rendered():
                try:
                    comment = file.document.trailing_comment(spec.path)
                except textproto_document.DocumentError:
                    continue  # a value the file leaves out, or a name list
                if not comment:
                    continue
                commented += 1
                with self.subTest(file=file.path, path=spec.path):
                    self.assertNotIn(
                        comment, config_schema.display_label(spec, file.message)
                    )
        # The shipped files do end lines in comments, so the rule above was exercised.
        self.assertGreater(commented, 0)


@operator_test_support.requires_display
class TestEveryTabShowsEveryRenderedParameter(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)

    def test_the_mpc_parameters_tab(self):
        for package, config in _robot_files():
            with self.subTest(robot=package):
                tab = mpc_params_tab.MpcParamsTab(
                    self.root,
                    task_file=os.path.join(config, "mpc", "task.textproto"),
                    enable_online_tuning=False,
                )
                expected: set[str] = set()
                for prefix, file in tab._files():
                    expected.update(prefix + spec.path for spec in file.rendered())
                shown: set[str] = set()
                for category in tab.categories:
                    tab.active_category.set(category)
                    tab._render_active_category()
                    self.assertFalse(shown & set(tab.slider_rows), category)
                    shown |= set(tab.slider_rows)
                self.assertEqual(shown, expected)
                self.assertTrue(expected)
                tab.destroy()

    def test_the_pd_gains_and_command_limits_tabs(self):
        for package, config in _robot_files():
            with self.subTest(robot=package):
                gains = joint_pd_tab.JointPdGainsTab(
                    self.root,
                    pd_gains_file=os.path.join(
                        config, "controller", "joint_pd_gains.textproto"
                    ),
                    enable_online_tuning=False,
                )
                assert gains.gains is not None
                self.assertEqual(
                    set(gains.slider_rows),
                    {spec.path for spec in gains.gains.rendered()},
                )
                limits = command_limits_tab.CommandLimitsTab(
                    self.root,
                    reference_file=os.path.join(
                        config, "command", "reference.textproto"
                    ),
                )
                assert limits.reference is not None
                self.assertEqual(
                    set(limits.slider_rows),
                    {spec.path for spec in limits.reference.rendered()},
                )
                gains.destroy()
                limits.destroy()

    def test_every_kind_of_the_test_schema_gets_a_widget(self):
        message = tuning_test_file_pb2.TuningTestFile()
        for spec in config_schema.parameters(message):
            row = parameter_rows.make_row(self.root, spec, spec.label, spec.value)
            with self.subTest(path=spec.path):
                self.assertEqual(row is not None, spec.renders)

    def test_tuned_file_and_the_walk_agree_on_the_shipped_files(self):
        for package, config in _robot_files():
            with self.subTest(robot=package):
                task = tuned_file.TunedFile(
                    os.path.join(config, "mpc", "task.textproto"),
                    task_file_pb2.TaskFile,
                )
                walked = config_schema.parameters(task.message, task.document)
                self.assertEqual(
                    [spec.path for spec in task.parameters()],
                    [spec.path for spec in walked],
                )


if __name__ == "__main__":
    unittest.main()
