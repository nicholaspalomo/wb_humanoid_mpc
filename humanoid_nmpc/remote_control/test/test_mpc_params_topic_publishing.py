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

"""The MPC Parameters tab on the bus: a change is published as the whole edited files, and only Save writes them.

The tab is given the GUI's publisher of operator/mpc_parameters over a bus that records, so the tests assert on the
humanoid_mpc_config.MpcParameterUpdate that would have gone out. The tab works on copies of the DRC Atlas files (which
has a contact-planning file) and of the Unitree G1's (which has none). Needs a display.
"""

import os
import shutil
import tempfile
import tkinter as tk
from typing import Any
import unittest

from humanoid_mpc_config import contact_planning_file_pb2
from humanoid_mpc_config import mpc_parameter_update_pb2
from humanoid_mpc_config import task_file_pb2

from humanoid_mpc_ipc import topics
import nproto_textproto
import operator_test_support
from remote_control import robot_config_save
from remote_control.tk_app import mpc_params_tab
from remote_control.tk_app import slider_row

# A parameter of the task file the tests drag, and one of the contact-planning file.
STATE_SCALING = "state_weights.scaling"
PLANNER_DT = mpc_params_tab.CONTACT_PLANNING_BLOCK + ".planner.dt"


@operator_test_support.requires_display
class TestMpcParamsTopicPublishing(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        config = operator_test_support.copy_config(
            operator_test_support.ATLAS_CONFIG, self.tmpdir
        )
        self.task_file = os.path.join(config, "mpc", "task.textproto")
        self.planner_file = os.path.join(config, "mpc", "contact_planning.textproto")
        self.publisher = operator_test_support.RecordingPublisher(
            topics.OPERATOR_MPC_PARAMETERS
        )
        self.original = {
            path: self._read(path) for path in (self.task_file, self.planner_file)
        }

    def _read(self, path: str) -> str:
        with open(path, encoding="utf-8", newline="") as handle:
            return handle.read()

    def _tab(
        self, task_file: str | None = None, **kwargs: Any
    ) -> mpc_params_tab.MpcParamsTab:
        options: dict[str, Any] = {
            "enable_online_tuning": True,
            "param_publisher": self.publisher,
        }
        options.update(kwargs)
        return mpc_params_tab.MpcParamsTab(
            self.root, task_file=task_file or self.task_file, **options
        )

    def _row(self, tab: mpc_params_tab.MpcParamsTab, key: str) -> slider_row.SliderRow:
        self.assertTrue(tab.render_category_containing(key), key)
        row = tab.slider_rows[key]
        assert isinstance(row, slider_row.SliderRow), key
        return row

    def _published(self) -> mpc_parameter_update_pb2.MpcParameterUpdate:
        """The last update published; every update published so far names the task file it is of."""
        for update in self.publisher.messages:
            assert isinstance(update, mpc_parameter_update_pb2.MpcParameterUpdate)
            self.assertTrue(update.config_path)
        message = self.publisher.last_message
        assert isinstance(message, mpc_parameter_update_pb2.MpcParameterUpdate), message
        return message

    def _files_unchanged(self) -> bool:
        return all(self._read(path) == text for path, text in self.original.items())

    def _flush(self, tab: mpc_params_tab.MpcParamsTab) -> None:
        """Runs the pending debounced publish now."""
        if tab._debounce_publish_id is not None:
            tab.after_cancel(tab._debounce_publish_id)
        tab._publish_to_topic()

    def test_a_change_publishes_both_files_and_writes_nothing(self):
        tab = self._tab()
        row = self._row(tab, STATE_SCALING)
        row.set_value(row.get_value() * 1.5)
        self._flush(tab)
        published = self._published()
        self.assertEqual(published.task.state_weights.scaling, row.get_value())
        self.assertTrue(published.HasField("contact_planning"))
        self.assertTrue(self._files_unchanged())
        # The task file's identity, which a receiver running another configuration refuses the update by.
        self.assertEqual(
            published.config_path, robot_config_save.config_path_of(self.task_file)
        )

    def test_the_update_is_the_files_parsed_strictly_with_the_change(self):
        tab = self._tab()
        planner_row = self._row(tab, PLANNER_DT)
        planner_row.set_value(0.05)
        self._flush(tab)
        published = self._published()
        task = nproto_textproto.load_textproto(self.task_file, task_file_pb2.TaskFile)
        planner = nproto_textproto.load_textproto(
            self.planner_file, contact_planning_file_pb2.ContactPlanningFile
        )
        # The task file as it is; the planner's with its one change.
        self.assertEqual(published.task, task)
        planner.planner.dt = 0.05
        self.assertEqual(published.contact_planning, planner)

    def test_without_a_change_the_update_is_the_files(self):
        tab = self._tab()
        tab.reload_file()
        published = self._published()
        self.assertEqual(
            published.task,
            nproto_textproto.load_textproto(self.task_file, task_file_pb2.TaskFile),
        )

    def test_a_robot_without_a_contact_planner_sends_none(self):
        config = operator_test_support.copy_config(
            operator_test_support.ROBOT_CONFIGS["g1_centroidal_mpc"],
            os.path.join(self.tmpdir, "g1"),
        )
        tab = self._tab(os.path.join(config, "mpc", "task.textproto"))
        self.assertIsNone(tab.contact_planning)
        self.assertNotIn(mpc_params_tab.CONTACT_PLANNING_BLOCK, tab.categories)
        row = self._row(tab, STATE_SCALING)
        row.set_value(row.get_value() + 1.0)
        self._flush(tab)
        self.assertFalse(self._published().HasField("contact_planning"))

    def test_save_writes_the_changes_into_their_files(self):
        tab = self._tab()
        row = self._row(tab, STATE_SCALING)
        row.set_value(row.get_value() + 1.0)
        self._row(tab, PLANNER_DT).set_value(0.05)
        self.assertTrue(tab.save())
        task = nproto_textproto.load_textproto(self.task_file, task_file_pb2.TaskFile)
        planner = nproto_textproto.load_textproto(
            self.planner_file, contact_planning_file_pb2.ContactPlanningFile
        )
        saved = row.get_value()
        self.assertEqual(task.state_weights.scaling, saved)
        self.assertEqual(planner.planner.dt, 0.05)
        # The saved values are the reset checkpoint now: a row shows them unmodified, and Reset All publishes them.
        shown = self._row(tab, STATE_SCALING)
        self.assertEqual(shown.get_value(), saved)
        self.assertFalse(shown.is_modified())
        tab.reset_all_defaults()
        self.assertEqual(self._published().task.state_weights.scaling, saved)
        self.assertEqual(self._published().contact_planning.planner.dt, 0.05)

    def test_reset_all_returns_to_the_files_and_publishes_them(self):
        tab = self._tab()
        row = self._row(tab, STATE_SCALING)
        original = row.get_value()
        row.set_value(original * 3.0)
        tab.reset_all_defaults()
        self.assertEqual(row.get_value(), original)
        self.assertEqual(self._published().task.state_weights.scaling, original)
        self.assertTrue(self._files_unchanged())

    def test_without_a_publisher_nothing_is_sent_and_nothing_fails(self):
        tab = self._tab(param_publisher=None)
        row = self._row(tab, STATE_SCALING)
        row.set_value(row.get_value() + 1.0)
        self._flush(tab)
        self.assertEqual(self.publisher.publish_count, 0)

    def test_without_online_tuning_nothing_is_published_or_saved(self):
        tab = self._tab(enable_online_tuning=False)
        tab._publish_to_topic()
        self.assertEqual(self.publisher.publish_count, 0)
        self.assertFalse(tab.save())
        self.assertTrue(self._files_unchanged())

    def test_rapid_changes_publish_once(self):
        tab = self._tab()
        row = self._row(tab, STATE_SCALING)
        for step in range(5):
            row.set_value(row.get_value() + step)
        self.assertIsNotNone(tab._debounce_publish_id)
        self._flush(tab)
        self.assertEqual(self.publisher.publish_count, 1)

    def test_rows_are_keyed_by_their_path_and_labeled_by_their_field_names(self):
        tab = self._tab()
        joint = "state_weights.joint_positions[joint=back_bkz].value"
        self.assertTrue(tab.render_category_containing(joint))
        # The field and the joint's key, from below the block; the comment of the line is not part of it.
        self.assertEqual(tab.slider_rows[joint].name, "joint_positions[back_bkz]")
        # Every row of the block on screen is a parameter of a file, by its path.
        for key in tab.slider_rows:
            with self.subTest(key=key):
                file, path = tab._locate(key)
                self.assertTrue(file.has(path))

    def test_the_contact_estimator_selection_publishes_its_name_and_saves_it(self):
        tab = self._tab()
        notified: list[str | None] = []

        def record() -> None:
            notified.append(tab.selected_contact_estimator())

        tab.on_contact_estimator_changed = record
        self.assertTrue(tab.has_contact_estimator_selection())
        self.assertEqual(tab.selected_contact_estimator(), "cheater_sim")
        self.assertIn("always_in_contact", tab.contact_estimator_names())
        tab.set_contact_estimator("always_in_contact")
        self.assertEqual(tab.selected_contact_estimator(), "always_in_contact")
        self.assertIn("always_in_contact", notified)
        self._flush(tab)
        self.assertEqual(self._published().task.contact_estimator, "always_in_contact")
        self.assertTrue(self._files_unchanged())
        self.assertTrue(tab.save())
        self.assertIn(
            'contact_estimator: "always_in_contact"', self._read(self.task_file)
        )
        tab.set_contact_estimator("cheater_sim")
        tab.reset_all_defaults()
        self.assertEqual(tab.selected_contact_estimator(), "always_in_contact")

    def test_the_contact_estimator_selection_is_ignored_without_online_tuning(self):
        tab = self._tab(enable_online_tuning=False)
        tab.set_contact_estimator("always_in_contact")
        self.assertEqual(tab.selected_contact_estimator(), "cheater_sim")
        self.assertEqual(self.publisher.publish_count, 0)

    def test_a_file_that_does_not_parse_is_reported_and_shows_nothing(self):
        with open(self.task_file, "a", encoding="utf-8") as handle:
            handle.write("terrainHeight: 0.1\n")
        tab = self._tab()
        self.assertIsNone(tab.task)
        self.assertEqual(tab.categories, [])
        self.assertIn("task.textproto", tab.status_label.cget("text"))
        self.assertFalse(tab.save())


if __name__ == "__main__":
    unittest.main()
