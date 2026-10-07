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

"""Every tunable parameter of the DRC Atlas task and contact-planning files, driven through the tab's own widgets.

The parameters are what the schemas render (config_schema); this test drives each of them through the row the MPC
Parameters tab builds for it - every state and input weight by joint and contact name, the task-space and torque costs,
the barriers, the foot constraint, the swing trajectory, the solver, the planner - and checks that:

- the published MpcParameterUpdate carries every value the rows were set to, and nothing else changed;
- Save writes them into the two files (a strict parse of each reads them back), keeping every other line;
- Reset All after the save returns nothing: the saved files are the checkpoint.

Needs a display.
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
import operator_test_support
from remote_control import config_schema
from remote_control import tuned_file
from remote_control.tk_app import mpc_params_tab
from remote_control.tk_app import parameter_rows


def _new_value(
    row: parameter_rows.ParameterRow, spec: config_schema.ParameterSpec
) -> Any:
    """A value of the row other than its own, of its kind."""
    current = row.get_value()
    if spec.kind == config_schema.Kind.BOOL:
        return not current
    if spec.kind == config_schema.Kind.INTEGER:
        return float(int(current) + 1)
    if spec.kind == config_schema.Kind.NUMBER:
        return float(current) * 1.5 + 0.25
    return f"{current or 'name'}_edited"


@operator_test_support.requires_display
class TestDrcAtlasParameterCoverage(unittest.TestCase):
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
        self.tab = mpc_params_tab.MpcParamsTab(
            self.root,
            task_file=self.task_file,
            enable_online_tuning=True,
            param_publisher=self.publisher,
        )

    def _drive_every_row(self) -> dict[str, Any]:
        """Sets every editable row of every block through the widget; returns the values set, by row key."""
        values: dict[str, Any] = {}
        for category in self.tab.categories:
            self.tab.active_category.set(category)
            self.tab._render_active_category()
            for key, row in self.tab.slider_rows.items():
                file, path = self.tab._locate(key)
                spec = file.spec(path)
                if spec.kind not in tuned_file.EDITABLE_KINDS:
                    continue
                value = _new_value(row, spec)
                row.set_value(value)
                values[key] = file.value(path)
        return values

    def test_the_tab_covers_the_state_and_input_weights_by_name(self):
        assert self.tab.task is not None
        rendered = {spec.path for spec in self.tab.task.rendered()}
        task = self.tab.task.message
        for joint in task.state_weights.joint_positions:
            self.assertIn(
                f"state_weights.joint_positions[joint={joint.joint}].value", rendered
            )
        for wrench in task.input_weights.contact_wrenches:
            for axis in "xyz":
                self.assertIn(
                    f"input_weights.contact_wrenches[contact={wrench.contact}].force.{axis}",
                    rendered,
                )

    def test_every_row_is_published_and_saved(self):
        values = self._drive_every_row()
        self.assertGreater(len(values), 100)
        assert self.tab.task is not None and self.tab.contact_planning is not None
        expected_task = self.tab.task.edited().message
        expected_planner = self.tab.contact_planning.edited().message

        self.tab._publish_to_topic()
        published = self.publisher.last_message
        assert isinstance(published, mpc_parameter_update_pb2.MpcParameterUpdate)
        self.assertEqual(published.task, expected_task)
        self.assertEqual(published.contact_planning, expected_planner)
        for key, value in values.items():
            file, path = self.tab._locate(key)
            message = (
                published.contact_planning
                if file is self.tab.contact_planning
                else published.task
            )
            walked = {spec.path: spec for spec in config_schema.parameters(message)}
            self.assertEqual(walked[path].value, value, key)

        originals = {}
        for path in (self.task_file, self.planner_file):
            with open(
                path + ".bak" if os.path.exists(path + ".bak") else path,
                encoding="utf-8",
            ) as handle:
                originals[path] = handle.read()
        self.assertTrue(self.tab.save())
        task = tuned_file.TunedFile(self.task_file, task_file_pb2.TaskFile)
        planner = tuned_file.TunedFile(
            self.planner_file, contact_planning_file_pb2.ContactPlanningFile
        )
        self.assertEqual(task.message, expected_task)
        self.assertEqual(planner.message, expected_planner)
        # The comments and LINT directives of both files are kept.
        for path, original in originals.items():
            with open(path, encoding="utf-8") as handle:
                saved = handle.read()
            original_comments = [
                line for line in original.splitlines() if line.lstrip().startswith("#")
            ]
            saved_comments = [
                line for line in saved.splitlines() if line.lstrip().startswith("#")
            ]
            self.assertEqual(saved_comments, original_comments, path)

        # The saved files are the checkpoint: Reset All publishes them as they are.
        self.tab.reset_all_defaults()
        reset = self.publisher.last_message
        assert isinstance(reset, mpc_parameter_update_pb2.MpcParameterUpdate)
        self.assertEqual(reset.task, expected_task)
        self.assertEqual(reset.contact_planning, expected_planner)


if __name__ == "__main__":
    unittest.main()
