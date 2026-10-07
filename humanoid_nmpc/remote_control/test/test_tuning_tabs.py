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

"""The tuning tabs' model of a file (tuned_file.TunedFile), headless, and the rows and tabs built on it.

TunedFile is what a slider changes: a change is held, published from the edited file (parsed strictly from its edited
text), and written only by save(), which keeps every byte it does not touch. A tab's Save writes the laptop's copy and
sends exactly its text to the robot's store, and its status line follows the robot's answer. The labels say which
fields the running MPC of the file's formulation applies live. The widget tests need a display.
"""

import functools
import os
import shutil
import tempfile
import tkinter as tk
import unittest
from unittest import mock

from humanoid_mpc_config import contact_planning_file_pb2
from humanoid_mpc_config import joint_pd_gains_file_pb2
from humanoid_mpc_config import reference_file_pb2
from humanoid_mpc_config import task_file_pb2
from humanoid_mpc_msgs import config_file_save_pb2
from humanoid_mpc_msgs import config_file_save_status_pb2

from humanoid_mpc_ipc import topics
import nproto_schema
import operator_test_support
from remote_control import config_schema
from remote_control import operator_bus
from remote_control import robot_config_save
from remote_control import tuned_file
from remote_control.tk_app import command_limits_tab
from remote_control.tk_app import joint_pd_tab
from remote_control.tk_app import mpc_params_tab
from remote_control.tk_app import parameter_rows
from remote_control.tk_app import slider_row

_TASK_HEADER = (
    "# proto-file: humanoid_nmpc/humanoid_mpc_config/task_file.proto\n"
    "# proto-message: humanoid_mpc_config.TaskFile\n"
)
TASK = (
    _TASK_HEADER
    + """# The ground.
terrain_height: 0.0  # [m] flat
state_weights {
  scaling: 85  # the overall scale
  joint_positions { joint: "back_bkz" value: 5 }  # the back
  joint_positions { joint: "l_leg_kny" value: 1e-3 }
}
"""
    # A LINT directive, written so that the repository's IFTTT check does not take this source line for one.
    + "# LINT."
    + "IfChange(example)\n"
    + 'contact_estimator: "cheater_sim"\n'
    + "# LINT."
    + "ThenChange(//nowhere.txt:example)\n"
)


class TempFileTestCase(unittest.TestCase):
    """A test case with a temporary directory, and files written into it and read back byte for byte."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def write(self, name: str, text: str) -> str:
        path = os.path.join(self.tmpdir, name)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        return path

    def read(self, path: str) -> str:
        with open(path, encoding="utf-8", newline="") as handle:
            return handle.read()


class TestTunedFile(TempFileTestCase):
    def setUp(self):
        super().setUp()
        self.path = self.write("task.textproto", TASK)
        self.task = tuned_file.TunedFile(self.path, task_file_pb2.TaskFile)

    def test_the_parameters_are_the_schemas(self):
        paths = {spec.path for spec in self.task.parameters()}
        self.assertIn("state_weights.joint_positions[joint=back_bkz].value", paths)
        self.assertIn("terrain_height", paths)
        # A block the file leaves out is there with its defaults, so that a slider can insert it.
        self.assertIn("swing_trajectory_config.swing_height", paths)
        rendered = {spec.path for spec in self.task.rendered()}
        self.assertNotIn("enable_online_tuning", rendered)  # excluded, with a reason
        self.assertTrue(self.task.spec("enable_online_tuning").tuning.exclude_reason)

    def test_a_change_is_held_and_not_written(self):
        self.task.set("state_weights.scaling", 90.0)
        self.assertEqual(self.task.value("state_weights.scaling"), 90.0)
        self.assertEqual(self.task.saved_value("state_weights.scaling"), 85)
        self.assertEqual(self.task.changes(), {"state_weights.scaling": 90.0})
        self.assertEqual(self.read(self.path), TASK)
        edited = self.task.edited()
        self.assertEqual(edited.message.state_weights.scaling, 90.0)
        # A double the file did not spell is written with the shortest digits that read back to it.
        self.assertIn("scaling: 90.0  # the overall scale", edited.text)

    def test_setting_the_files_value_back_drops_the_change(self):
        self.task.set("state_weights.scaling", 90.0)
        self.task.set("state_weights.scaling", 85.0)
        self.assertEqual(self.task.changes(), {})
        # A default left at the default is not written into the file either.
        self.task.set("swing_trajectory_config.swing_height", 0.1)
        self.assertEqual(self.task.changes(), {})
        self.assertFalse(self.task.edited().changed)

    def test_an_integer_is_rounded_and_a_bool_is_a_bool(self):
        self.task.set("multiple_shooting.sqp_iteration", 3.6)
        self.assertEqual(self.task.value("multiple_shooting.sqp_iteration"), 4)
        self.task.set("mpc.cold_start", 1)
        self.assertIs(self.task.value("mpc.cold_start"), True)

    def test_only_what_a_widget_edits_can_be_set(self):
        with self.assertRaises(KeyError):
            self.task.set("no_such_field", 1.0)
        with self.assertRaises(ValueError):
            self.task.set("costs", "terminal_cost")  # a name list

    def test_save_writes_the_changes_and_keeps_every_other_byte(self):
        self.task.set("state_weights.joint_positions[joint=l_leg_kny].value", 2.5)
        self.task.set("terrain_height", 0.02)
        self.task.set(
            "swing_trajectory_config.swing_height", 0.12
        )  # an absent block: inserted
        result = self.task.save()
        text = self.read(self.path)
        self.assertEqual(text, result.text)
        self.assertIn('joint_positions { joint: "l_leg_kny" value: 2.5 }', text)
        self.assertIn("terrain_height: 0.02  # [m] flat", text)
        for kept in (
            "# The ground.",
            '  joint_positions { joint: "back_bkz" value: 5 }  # the back',
            "# LINT." + "IfChange(example)",
            "# LINT." + "ThenChange(//nowhere.txt:example)",
        ):
            self.assertIn(kept, text)
        # The saved file is the checkpoint now: nothing is changed, and its values are the saved ones.
        self.assertEqual(self.task.changes(), {})
        self.assertEqual(self.task.saved_value("terrain_height"), 0.02)
        self.assertEqual(
            self.task.saved_value("swing_trajectory_config.swing_height"), 0.12
        )
        # The first save keeps the file as it was.
        self.assertEqual(self.read(self.path + ".bak"), TASK)

    def test_save_without_changes_writes_nothing(self):
        self.task.save()
        self.assertEqual(self.read(self.path), TASK)
        self.assertFalse(os.path.exists(self.path + ".bak"))

    def test_reset_returns_to_the_checkpoint(self):
        self.task.set("terrain_height", 0.5)
        self.task.reset()
        self.assertEqual(self.task.value("terrain_height"), 0.0)
        self.assertFalse(self.task.edited().changed)

    def test_a_file_that_does_not_parse_is_refused_with_its_position(self):
        path = self.write("bad.textproto", _TASK_HEADER + "terrainHeight: 0.1\n")
        with self.assertRaisesRegex(tuned_file.TunedFileError, r"bad.textproto:3:"):
            tuned_file.TunedFile(path, task_file_pb2.TaskFile)

    def test_the_groups_of_rows(self):
        joint = self.task.spec("state_weights.joint_positions[joint=back_bkz].value")
        self.assertEqual(tuned_file.group_of(joint), "state_weights.joint_positions")
        self.assertEqual(
            tuned_file.group_of(self.task.spec("state_weights.scaling")),
            "state_weights",
        )
        self.assertEqual(tuned_file.group_of(self.task.spec("terrain_height")), "")


def _shipped_task_files() -> list[tuned_file.TunedFile]:
    """Every robot's task file, of both formulations."""
    return [
        tuned_file.TunedFile(
            os.path.join(config, "mpc", "task.textproto"), task_file_pb2.TaskFile
        )
        for config in operator_test_support.ROBOT_CONFIGS.values()
    ]


class TestEveryFormulationAppliesItsHotFieldsLive(unittest.TestCase):
    """The labels of the task files (test L9): live where the file's formulation applies a field, and only there."""

    def test_both_formulations_are_shipped(self):
        formulations = {
            config_schema.formulation_of(file.message) for file in _shipped_task_files()
        }
        self.assertEqual(formulations, set(config_schema.FORMULATIONS))

    def test_a_field_of_another_formulation_is_not_applicable(self):
        shown = {formulation: 0 for formulation in config_schema.FORMULATIONS}
        for file in _shipped_task_files():
            formulation = config_schema.formulation_of(file.message)
            for spec in file.rendered():
                if (
                    not spec.tuning.formulations
                    or formulation in spec.tuning.formulations
                ):
                    continue
                shown[formulation] += 1
                with self.subTest(file=file.path, path=spec.path):
                    self.assertIn(
                        "not applicable: ",
                        config_schema.display_label(spec, file.message),
                    )
        # Each formulation's files show fields only the other one reads (the centroidal costs on the whole-body G1,
        # joint_torque_weights on the centroidal robots), so the rule above was exercised both ways.
        self.assertGreater(shown[config_schema.WHOLE_BODY], 0)
        self.assertGreater(shown[config_schema.CENTROIDAL], 0)

    def test_every_hot_field_the_formulation_reads_is_live(self):
        live = {formulation: 0 for formulation in config_schema.FORMULATIONS}
        for file in _shipped_task_files():
            formulation = config_schema.formulation_of(file.message)
            for spec in file.rendered():
                if spec.tuning.reload != config_schema.RELOAD_HOT:
                    continue
                if (
                    spec.tuning.formulations
                    and formulation not in spec.tuning.formulations
                ):
                    continue
                if not config_schema.is_active(spec.tuning, file.message):
                    continue
                live[formulation] += 1
                with self.subTest(file=file.path, path=spec.path):
                    label = config_schema.display_label(spec, file.message)
                    self.assertNotIn("restart", label)
                    self.assertNotIn("not applicable", label)
        self.assertGreater(live[config_schema.WHOLE_BODY], 0)
        self.assertGreater(live[config_schema.CENTROIDAL], 0)

    def test_restart_is_the_start_up_fields_alone(self):
        for file in _shipped_task_files():
            for spec in file.rendered():
                with self.subTest(file=file.path, path=spec.path):
                    restart = "restart" in config_schema.annotations(spec, file.message)
                    self.assertEqual(
                        restart, spec.tuning.reload == config_schema.RELOAD_START_UP
                    )

    def test_the_whole_body_g1s_weights_and_torque_weights_are_live(self):
        whole_body = tuned_file.TunedFile(
            os.path.join(
                operator_test_support.ROBOT_CONFIGS["g1_wb_mpc"],
                "mpc",
                "task.textproto",
            ),
            task_file_pb2.TaskFile,
        )
        self.assertEqual(
            config_schema.formulation_of(whole_body.message), config_schema.WHOLE_BODY
        )
        for path in ("state_weights.scaling", "contact_estimator"):
            with self.subTest(path=path):
                self.assertEqual(
                    config_schema.annotations(
                        whole_body.spec(path), whole_body.message
                    ),
                    [],
                )
        torque = [
            spec
            for spec in whole_body.rendered()
            if spec.path.startswith("joint_torque_weights")
        ]
        self.assertTrue(torque)
        for spec in torque:
            with self.subTest(path=spec.path):
                self.assertNotIn(
                    "not applicable",
                    config_schema.display_label(spec, whole_body.message),
                )


class TestTheShippedFilesLoad(unittest.TestCase):
    def test_every_robots_files_are_tunable(self):
        for package, config in operator_test_support.ROBOT_CONFIGS.items():
            for relative, message_class in (
                ("mpc/task.textproto", task_file_pb2.TaskFile),
                ("command/reference.textproto", reference_file_pb2.ReferenceFile),
                (
                    "controller/joint_pd_gains.textproto",
                    joint_pd_gains_file_pb2.JointPdGainsFile,
                ),
            ):
                with self.subTest(robot=package, file=relative):
                    loaded = tuned_file.TunedFile(
                        os.path.join(config, relative), message_class
                    )
                    self.assertTrue(loaded.rendered())
                    self.assertFalse(loaded.edited().changed)


@operator_test_support.requires_display
class TestParameterRows(TempFileTestCase):
    def setUp(self):
        super().setUp()
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.task = tuned_file.TunedFile(
            self.write("task.textproto", TASK), task_file_pb2.TaskFile
        )
        self.changes: list[tuple[str, object]] = []

    def _row(self, path: str) -> parameter_rows.ParameterRow:
        spec = self.task.spec(path)
        row = parameter_rows.make_row(
            self.root,
            spec,
            config_schema.display_label(spec, self.task.message),
            self.task.value(path),
            on_change=lambda label, value: self.changes.append((label, value)),
        )
        assert row is not None, path
        return row

    def test_each_kind_gets_its_widget(self):
        self.assertIsInstance(self._row("terrain_height"), slider_row.SliderRow)
        self.assertIsInstance(self._row("mpc.cold_start"), parameter_rows.CheckRow)
        self.assertIsInstance(self._row("contact_estimator"), parameter_rows.ChoiceRow)
        # A selection by name, here one the file leaves at its default, is a choice too.
        self.assertIsInstance(
            self._row("model_settings.foot_constraint.stance_constraint"),
            parameter_rows.ChoiceRow,
        )
        self.assertIsInstance(self._row("costs"), parameter_rows.NameListRow)
        excluded = self.task.spec("enable_online_tuning")
        self.assertIsNone(parameter_rows.make_row(self.root, excluded, "x", True))

    def test_a_row_reports_its_changes_and_resets_to_the_files_value(self):
        row = self._row("mpc.cold_start")
        row.set_value(True)
        self.assertEqual(self.changes[-1][1], True)
        self.assertTrue(row.is_modified())
        row.reset_to_default()
        self.assertEqual(row.get_value(), False)
        self.assertFalse(row.is_modified())
        choice = self._row("contact_estimator")
        assert isinstance(choice, parameter_rows.ChoiceRow)
        choice.set_value("always_in_contact")
        self.assertEqual(self.changes[-1][1], "always_in_contact")

    def test_an_integer_slider_holds_whole_numbers(self):
        row = self._row("multiple_shooting.sqp_iteration")
        assert isinstance(row, slider_row.SliderRow)
        self.assertTrue(row.integer)
        row.set_value(3.4)
        self.assertEqual(row.get_value(), 3.0)

    def test_a_logarithmic_slider_spans_decades(self):
        row = self._row("multiple_shooting.delta_tol")
        assert isinstance(row, slider_row.SliderRow)
        self.assertTrue(row.log_scale)
        # Its travel is the decades of its range, and a value placed on it reads back.
        self.assertAlmostEqual(
            float(row.scale.cget("to")) - float(row.scale.cget("from")), 4.0
        )
        row.set_value(1e-5)
        self.assertAlmostEqual(row.get_value(), 1e-5)
        self.assertAlmostEqual(float(row.scale_var.get()), -5.0)
        row._on_scale_change("-3.0")
        self.assertAlmostEqual(row.get_value(), 1e-3)

    def test_the_label_says_restart_and_the_unit(self):
        self.assertIn("(restart", self._row("costs").name)
        self.assertIn("(m)", self._row("terrain_height").name)
        # The field's name, without the comment its line ends in ("# the overall scale").
        self.assertEqual(self._row("state_weights.scaling").name, "scaling")


@operator_test_support.requires_display
class TestSliderRowEntry(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.changes: list[tuple[str, float | None]] = []

    def _slider(self, value: float, unset: bool = False) -> slider_row.SliderRow:
        return slider_row.SliderRow(
            self.root,
            name="gain",
            initial_value=value,
            max_val=4000.0,
            on_change=lambda name, changed: self.changes.append((name, changed)),
            unset=unset,
        )

    def test_leaving_the_entry_untouched_keeps_the_value_the_entry_rounds(self):
        row = self._slider(1406.25)
        self.assertEqual(row.entry_var.get(), "1.41e+03")  # shown rounded
        row._on_entry_submit()  # <FocusOut>, or Return, without typing
        self.assertEqual(row.get_value(), 1406.25)
        self.assertEqual(self.changes, [])
        self.assertFalse(row.is_modified())

    def test_a_typed_value_is_taken(self):
        row = self._slider(1406.25)
        row.entry_var.set("1500")
        row._on_entry_submit()
        self.assertEqual(row.get_value(), 1500.0)
        self.assertEqual(self.changes, [("gain", 1500.0)])
        # The text the row shows now is its own: leaving it again is no change.
        row._on_entry_submit()
        self.assertEqual(len(self.changes), 1)

    def test_an_unset_row_shows_no_value_until_one_is_given_and_resets_to_none(self):
        row = self._slider(0.1, unset=True)
        self.assertFalse(row.has_value())
        self.assertEqual(row.entry_var.get(), "")
        row._on_entry_submit()
        self.assertEqual(self.changes, [])
        self.assertFalse(row.is_modified())
        row.entry_var.set("0.9")
        row._on_entry_submit()
        self.assertTrue(row.has_value())
        self.assertTrue(row.is_modified())
        self.assertEqual(self.changes, [("gain", 0.9)])
        row.reset_to_default()
        self.assertFalse(row.has_value())
        self.assertEqual(row.entry_var.get(), "")
        self.assertEqual(self.changes[-1], ("gain", None))
        self.assertFalse(row.is_modified())


@operator_test_support.requires_display
class TestUnsetParameters(TempFileTestCase):
    """A parameter the file leaves unset (no value, no schema default) stays unset unless the operator gives it one."""

    UNSET = "dcm_terminal_cost.com_height"

    def setUp(self):
        super().setUp()
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.path = os.path.join(self.tmpdir, "task.textproto")
        shutil.copyfile(operator_test_support.ATLAS_TASK_FILE, self.path)
        self.original = self.read(self.path)
        self.task = tuned_file.TunedFile(self.path, task_file_pb2.TaskFile)

    def _row(self, path: str) -> parameter_rows.ParameterRow:
        spec = self.task.spec(path)
        row = parameter_rows.make_row(
            self.root,
            spec,
            config_schema.display_label(spec, self.task.message),
            self.task.value(path),
            on_change=lambda label, value: self.task.set(path, value),
        )
        assert row is not None, path
        return row

    def test_the_shipped_atlas_file_leaves_the_dcm_height_unset(self):
        spec = self.task.spec(self.UNSET)
        self.assertEqual(spec.source, config_schema.ValueSource.UNSET)
        self.assertIsNone(spec.value)
        self.assertTrue(spec.renders)

    def test_focus_out_and_reset_leave_it_unset_and_save_writes_nothing(self):
        row = self._row(self.UNSET)
        assert isinstance(row, slider_row.SliderRow)
        self.assertFalse(row.has_value())
        row._on_entry_submit()
        row.reset_to_default()
        self.assertEqual(self.task.changes(), {})
        self.task.save()
        self.assertEqual(self.read(self.path), self.original)

    def test_a_value_given_and_reset_is_unset_again(self):
        row = self._row(self.UNSET)
        row.set_value(0.9)
        self.assertEqual(self.task.changes(), {self.UNSET: 0.9})
        self.assertEqual(self.task.edited().message.dcm_terminal_cost.com_height, 0.9)
        row.reset_to_default()
        self.assertEqual(self.task.changes(), {})
        self.assertFalse(
            self.task.edited().message.dcm_terminal_cost.HasField("com_height")
        )

    def test_a_row_built_for_a_change_shows_it(self):
        # The tabs rebuild a block's rows from the file's values, the operator's changes included.
        self.task.set(self.UNSET, 0.0)
        row = self._row(self.UNSET)
        self.assertTrue(row.has_value())
        self.assertEqual(row.get_value(), 0.0)


def _set_by_path(
    file: tuned_file.TunedFile, path: str, label: str, value: object
) -> None:
    """A row's on_change: sets the parameter `path` of `file`; the label is for display only."""
    del label  # Unused: for display only.
    file.set(path, value)


def _shipped_files() -> list[tuned_file.TunedFile]:
    """Copies of every file the tabs tune, of every robot."""
    files = []
    for config in operator_test_support.ROBOT_CONFIGS.values():
        for relative, message_class in (
            ("mpc/task.textproto", task_file_pb2.TaskFile),
            (
                "mpc/contact_planning.textproto",
                contact_planning_file_pb2.ContactPlanningFile,
            ),
            ("command/reference.textproto", reference_file_pb2.ReferenceFile),
            (
                "controller/joint_pd_gains.textproto",
                joint_pd_gains_file_pb2.JointPdGainsFile,
            ),
        ):
            path = os.path.join(config, relative)
            if os.path.exists(path):
                files.append(tuned_file.TunedFile(path, message_class))
    return files


@operator_test_support.requires_display
class TestUntouchedRowsChangeNothing(unittest.TestCase):
    """GUI saving is lossless: a widget the operator does not change never changes the file (test T8's other half)."""

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)

    def _rows(
        self, file: tuned_file.TunedFile
    ) -> list[tuple[str, parameter_rows.ParameterRow]]:
        """A row per rendered parameter of `file`, each setting its parameter of `file` on a change, by path."""
        rows = []
        for spec in file.rendered():
            row = parameter_rows.make_row(
                self.root,
                spec,
                spec.label,
                file.value(spec.path),
                on_change=functools.partial(_set_by_path, file, spec.path),
            )
            if row is not None:
                rows.append((spec.path, row))
        return rows

    def test_leaving_every_entry_of_every_shipped_file_changes_nothing(self):
        files = _shipped_files()
        self.assertTrue(files)
        for file in files:
            with self.subTest(file=file.path):
                rows = self._rows(file)
                for _, row in rows:
                    if isinstance(row, slider_row.SliderRow):
                        row._on_entry_submit()  # <FocusOut>
                    elif isinstance(row, parameter_rows.ChoiceRow):
                        row._on_selected()  # <FocusOut>
                self.assertEqual(file.changes(), {})
                for _, row in rows:
                    row.destroy()

    def test_resetting_every_unset_row_changes_nothing(self):
        unset = 0
        for file in _shipped_files():
            with self.subTest(file=file.path):
                rows = self._rows(file)
                for path, row in rows:
                    if file.spec(path).value is None:
                        unset += 1
                        self.assertFalse(row.has_value(), path)
                        row.reset_to_default()
                self.assertEqual(file.changes(), {})
                for _, row in rows:
                    row.destroy()
        # The shipped files do leave parameters unset, so the rule above was exercised.
        self.assertGreater(unset, 0)


@operator_test_support.requires_display
class TestTabsOnlineTuning(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)

    def test_joint_pd_tab_online_tuning_toggle(self):
        tab = joint_pd_tab.JointPdGainsTab(
            self.root,
            pd_gains_file=operator_test_support.ATLAS_PD_GAINS_FILE,
            enable_online_tuning=False,
        )
        self.assertTrue(tab.slider_rows)
        self.assertEqual(str(tab.save_btn.cget("state")), "disabled")
        tab.set_online_tuning_enabled(True)
        self.assertEqual(str(tab.save_btn.cget("state")), "normal")
        self.assertTrue(
            all(str(b.cget("state")) == "normal" for b in tab.scale_buttons)
        )

    def test_mpc_params_tab_online_tuning_follows_the_file(self):
        tab = mpc_params_tab.MpcParamsTab(
            self.root, task_file=operator_test_support.ATLAS_TASK_FILE
        )
        assert tab.task is not None
        self.assertEqual(
            tab.enable_online_tuning, tab.task.message.enable_online_tuning
        )
        tab.set_online_tuning_enabled(False)
        self.assertEqual(str(tab.save_btn.cget("state")), "disabled")

    def test_a_given_file_that_does_not_exist_is_reported_not_replaced_by_a_preset(
        self,
    ):
        missing = os.path.join(tempfile.gettempdir(), "no_such_robot", "task.textproto")
        mpc = mpc_params_tab.MpcParamsTab(self.root, task_file=missing)
        self.assertIsNone(mpc.task)
        self.assertIn("Cannot load", str(mpc.status_label.cget("text")))
        gains = joint_pd_tab.JointPdGainsTab(self.root, pd_gains_file=missing)
        self.assertIsNone(gains.gains)
        self.assertFalse(gains.slider_rows)

    def test_the_banner_says_why_tuning_is_off(self):
        tab = joint_pd_tab.JointPdGainsTab(
            self.root, pd_gains_file=operator_test_support.ATLAS_PD_GAINS_FILE
        )
        tab.set_online_tuning_enabled(False, "the task file does not parse: x:3:1")
        self.assertIn("does not parse", str(tab.warning_banner.cget("text")))
        tab.set_online_tuning_enabled(False)
        self.assertIn("enable_online_tuning", str(tab.warning_banner.cget("text")))

    def test_every_category_renders_and_is_reachable(self):
        tab = mpc_params_tab.MpcParamsTab(
            self.root,
            task_file=operator_test_support.ATLAS_TASK_FILE,
            enable_online_tuning=False,
        )
        self.assertIn(mpc_params_tab.CONTACT_PLANNING_BLOCK, tab.categories)
        assert tab.category_selector is not None
        self.assertEqual(
            list(tab.category_selector.cget("values")), list(tab.categories)
        )
        for category in tab.categories:
            with self.subTest(category=category):
                tab.active_category.set(category)
                tab._render_active_category()
                self.assertTrue(tab.slider_rows, category)


@operator_test_support.requires_display
class TestCommandLimitsTab(TempFileTestCase):
    def test_saving_writes_the_changed_limit_and_keeps_the_file(self):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        path = os.path.join(self.tmpdir, "reference.textproto")
        shutil.copyfile(operator_test_support.ATLAS_REFERENCE_FILE, path)
        original = self.read(path)
        tab = command_limits_tab.CommandLimitsTab(root, reference_file=path)
        self.assertIn("max_rotation_velocity", tab.slider_rows)
        self.assertTrue(
            tab.save()
        )  # nothing changed: nothing written, and the file is sent all the same
        self.assertEqual(self.read(path), original)
        self.assertFalse(os.path.exists(path + ".bak"))
        row = tab.slider_rows["max_rotation_velocity"]
        row.set_value(float(row.get_value()) + 0.25)
        self.assertEqual(tab.status_var.get(), "unsaved changes")
        self.assertTrue(tab.save())
        reloaded = tuned_file.TunedFile(path, reference_file_pb2.ReferenceFile)
        self.assertEqual(reloaded.saved_value("max_rotation_velocity"), row.get_value())
        changed = [
            (before, after)
            for before, after in zip(
                original.splitlines(), self.read(path).splitlines()
            )
            if before != after
        ]
        self.assertEqual(len(changed), 1, changed)
        self.assertIn("max_rotation_velocity", changed[0][1])


_Status = config_file_save_status_pb2.ConfigFileSaveStatus


class _FakeClock:
    """Monotonic seconds that move only when a test advances them."""

    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now


class _QueueFullBus(operator_test_support.RecordingBus):
    """A bus whose send queue is always full: publish() drops every message."""

    def publish(self, topic: str, message: object) -> bool:
        del topic, message  # Unused: dropped.
        return False


@operator_test_support.requires_display
class TestTwoCopySave(TempFileTestCase):
    """Save writes the laptop's copy and sends exactly its bytes to the robot's store, then shows the robot's answer."""

    def setUp(self):
        super().setUp()
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.config = operator_test_support.copy_config(
            operator_test_support.ATLAS_CONFIG, self.tmpdir
        )
        self.bus = operator_test_support.RecordingBus()
        self.statuses = operator_bus.ConfigSaveStatusMailbox()
        self.clock = _FakeClock()
        self.saver = self._saver(self.bus)

    def _saver(
        self, bus: operator_test_support.RecordingBus
    ) -> robot_config_save.RobotConfigSaver:
        return robot_config_save.RobotConfigSaver(
            operator_bus.TopicPublisher.for_topic(bus, topics.OPERATOR_CONFIG_SAVE),
            self.statuses,
            clock=self.clock,
        )

    def _path(self, *parts: str) -> str:
        return os.path.join(self.config, *parts)

    def _saves(self) -> list[config_file_save_pb2.ConfigFileSave]:
        saves: list[config_file_save_pb2.ConfigFileSave] = []
        for message in self.bus.messages_on(topics.OPERATOR_CONFIG_SAVE):
            assert isinstance(message, config_file_save_pb2.ConfigFileSave)
            saves.append(message)
        return saves

    def _one_save(self) -> config_file_save_pb2.ConfigFileSave:
        """The one save published so far."""
        saves = self._saves()
        self.assertEqual(len(saves), 1)
        return saves[0]

    def _answer(self, result: int, message: str = "", stored: str = "") -> None:
        """The robot's answer to the last save, delivered as the bus's receive thread would."""
        last = self._saves()[-1]
        self.statuses.put(
            _Status(
                sequence=last.sequence,
                kind=last.kind,
                config_path=last.config_path,
                result=result,
                message=message,
                stored_path=stored,
            )
        )

    def _mpc_tab(self, saver=None) -> mpc_params_tab.MpcParamsTab:
        return mpc_params_tab.MpcParamsTab(
            self.root,
            task_file=self._path("mpc", "task.textproto"),
            enable_online_tuning=True,
            robot_saver=self.saver if saver is None else saver,
        )

    def _status(self, tab: mpc_params_tab.MpcParamsTab) -> str:
        return str(tab.status_label.cget("text"))

    def _check_sent(
        self,
        save: config_file_save_pb2.ConfigFileSave,
        kind: int,
        path: str,
        message_class,
    ) -> None:
        """The save is the laptop's file: its kind, robot, identity, schema and exactly its bytes."""
        self.assertEqual(save.kind, kind)
        self.assertEqual(save.robot_name, "atlas")
        self.assertEqual(save.config_path, robot_config_save.config_path_of(path))
        self.assertEqual(
            save.schema_fingerprint,
            nproto_schema.schema_fingerprint(message_class.DESCRIPTOR),
        )
        with open(path, "rb") as handle:
            self.assertEqual(save.text.encode("utf-8"), handle.read())

    def test_the_task_files_save_sends_the_laptops_bytes(self):
        tab = self._mpc_tab()
        assert tab.task is not None
        tab._on_any_change("terrain_height", 0.03)
        self.assertTrue(tab.save())
        save = self._one_save()
        self._check_sent(
            save,
            robot_config_save.KIND_TASK,
            self._path("mpc", "task.textproto"),
            task_file_pb2.TaskFile,
        )
        self.assertIn("terrain_height: 0.03", save.text)
        self.assertEqual(
            self._status(tab), "Saved on the laptop · saving on the robot…"
        )
        # The contact planner's file is the MPC's: saved on the laptop, never sent.
        self.assertEqual(len(self._saves()), 1)

    def test_a_planner_file_that_does_not_save_leaves_both_task_copies_as_they_were(
        self,
    ):
        tab = self._mpc_tab()
        assert tab.task is not None and tab.contact_planning is not None
        task_path = self._path("mpc", "task.textproto")
        with open(task_path, "rb") as handle:
            before = handle.read()
        tab._on_any_change("terrain_height", 0.03)
        with mock.patch.object(
            tab.contact_planning, "save", side_effect=OSError("disk full")
        ):
            self.assertFalse(tab.save())
        with open(task_path, "rb") as handle:
            self.assertEqual(handle.read(), before, "the laptop's task file changed")
        self.assertEqual(self._saves(), [], "the robot's task file was sent")
        self.assertIn("contact planner's file: disk full", self._status(tab))
        # Once the planner saves again, both copies follow.
        self.assertTrue(tab.save())
        self._check_sent(
            self._one_save(),
            robot_config_save.KIND_TASK,
            task_path,
            task_file_pb2.TaskFile,
        )

    def test_every_answer_of_the_robot_is_shown(self):
        tab = self._mpc_tab()
        stored = "/var/lib/wb-humanoid-robot/config/drc_atlas/mpc/task.textproto"
        for result, expected in (
            (_Status.RESULT_SAVED, f"Saved on the laptop and on the robot ({stored})"),
            (_Status.RESULT_UNCHANGED, "robot unchanged"),
            (_Status.RESULT_REFUSED, "the robot refused it: another robot"),
            (_Status.RESULT_FAILED, "the robot could not store it: another robot"),
            (
                _Status.RESULT_NOT_STORED,
                f"the robot has no store: it reads {stored} in place; this Save did not change it",
            ),
        ):
            with self.subTest(result=result):
                self.assertTrue(tab.save())
                self._answer(result, "another robot", stored)
                tab.robot_save.poll()
                self.assertIn(expected, self._status(tab))
                self.assertTrue(self._status(tab).startswith("Saved on the laptop"))

    def test_an_unanswered_save_is_unknown_and_a_late_answer_replaces_it(self):
        tab = self._mpc_tab()
        self.assertTrue(tab.save())
        self.clock.now += robot_config_save.SAVE_TIMEOUT_SECONDS + 0.1
        tab.robot_save.poll()
        status = self._status(tab)
        self.assertIn("did not answer in 5 s", status)
        self.assertIn("unknown", status)
        self.assertIn("Save again", status)
        self.clock.now += 60.0
        self._answer(_Status.RESULT_SAVED, stored="/stored/task.textproto")
        tab.robot_save.poll()
        self.assertEqual(
            self._status(tab),
            "Saved on the laptop and on the robot (/stored/task.textproto)",
        )

    def test_a_save_the_bus_drops_says_not_sent(self):
        tab = self._mpc_tab(saver=self._saver(_QueueFullBus()))
        self.assertTrue(tab.save())
        self.assertIn("not sent to the robot: the bus dropped it", self._status(tab))

    def test_without_a_bus_the_save_says_it_was_not_sent(self):
        tab = mpc_params_tab.MpcParamsTab(
            self.root,
            task_file=self._path("mpc", "task.textproto"),
            enable_online_tuning=True,
        )
        self.assertTrue(tab.save())
        self.assertIn("not sent to the robot", self._status(tab))

    def test_a_failed_laptop_save_sends_nothing(self):
        tab = self._mpc_tab()
        tab._on_any_change("terrain_height", 0.03)
        task = self._path("mpc", "task.textproto")
        with open(task, "a", encoding="utf-8") as handle:
            handle.write(
                "terrainHeight: 0.1\n"
            )  # an editor's mistake since the tab loaded the file
        self.assertFalse(tab.save())
        self.assertEqual(self._saves(), [])
        self.assertIn("Error saving", self._status(tab))

    def test_a_save_without_changes_still_sends_the_file(self):
        tab = self._mpc_tab()
        task = self._path("mpc", "task.textproto")
        before = self.read(task)
        self.assertTrue(tab.save())
        self.assertEqual(self.read(task), before)
        save = self._one_save()
        self.assertEqual(save.text, before)

    def test_the_pd_gains_and_the_reference_file_are_sent_too(self):
        gains_path = self._path("controller", "joint_pd_gains.textproto")
        gains = joint_pd_gains_tab_for(self.root, gains_path, self.saver)
        assert gains.gains is not None
        spec = gains.gains.rendered()[0]
        gains._on_row_change(spec.path, spec.label, float(spec.value or 0.0) + 1.0)
        self.assertTrue(gains.save())
        reference_path = self._path("command", "reference.textproto")
        limits = command_limits_tab.CommandLimitsTab(
            self.root, reference_file=reference_path, robot_saver=self.saver
        )
        self.assertTrue(limits.save())
        saves = self._saves()
        self.assertEqual(len(saves), 2)
        gains_save, reference_save = saves[0], saves[1]
        self._check_sent(
            gains_save,
            robot_config_save.KIND_JOINT_PD_GAINS,
            gains_path,
            joint_pd_gains_file_pb2.JointPdGainsFile,
        )
        self._check_sent(
            reference_save,
            robot_config_save.KIND_REFERENCE,
            reference_path,
            reference_file_pb2.ReferenceFile,
        )
        self._answer(_Status.RESULT_REFUSED, "the reference does not parse")
        limits.robot_save.poll()
        self.assertIn(
            "the robot refused it: the reference does not parse",
            limits.status_var.get(),
        )
        # Each tab follows its own save: the PD gains tab still waits for its answer.
        gains.robot_save.poll()
        self.assertIn("saving on the robot", str(gains.status_label.cget("text")))


def joint_pd_gains_tab_for(
    root: tk.Misc, path: str, saver: robot_config_save.RobotConfigSaver
) -> joint_pd_tab.JointPdGainsTab:
    """A Joint PD Gains tab on `path`, tuning enabled, saving to the robot through `saver`."""
    return joint_pd_tab.JointPdGainsTab(
        root, pd_gains_file=path, enable_online_tuning=True, robot_saver=saver
    )


if __name__ == "__main__":
    unittest.main()
