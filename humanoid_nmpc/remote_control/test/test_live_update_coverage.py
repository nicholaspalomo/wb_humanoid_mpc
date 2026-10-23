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

"""Every parameter the GUI renders reaches the wire and the file, on every robot, headless (no Tk, no bus, no C++).

For each robot's task, contact-planning, reference and PD gains files, as the tabs hold them (tuned_file.TunedFile):

1. every rendered value the file sets is changed at once, and the edited file - parsed strictly from its edited text,
   which is what the tabs publish - holds every new value and every other parameter as it was;
2. saving writes exactly those values: every other line of the file is kept byte for byte, comments and LINT
   directives included;
3. values the file leaves out (defaults) are inserted where the schema puts them, and read back as the file's;
4. the MPC parameter update built from the edited files round-trips through the wire format and is what the saved
   files parse into.

Nothing here names a parameter: the parameters are the schemas' (config_schema), so a field added to a schema is
covered the day it is added.
"""

import os
import shutil
import tempfile
from typing import Any
import unittest

from humanoid_mpc_config import contact_planning_file_pb2
from humanoid_mpc_config import joint_pd_gains_file_pb2
from humanoid_mpc_config import mpc_parameter_update_pb2
from humanoid_mpc_config import reference_file_pb2
from humanoid_mpc_config import task_file_pb2

import nproto_textproto
import operator_test_support
from remote_control import config_schema
from remote_control import operator_bus
from remote_control import robot_config_save
from remote_control import tuned_file

FILES = (
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
# How many absent parameters of a file the insertion property inserts, spread over the file.
INSERTED_SAMPLE = 30


def new_value(spec: config_schema.ParameterSpec) -> Any:
    """A value of the parameter other than its own, of its kind."""
    if spec.kind == config_schema.Kind.BOOL:
        return not spec.value
    if spec.kind == config_schema.Kind.INTEGER:
        return int(spec.value or 0) + 1
    if spec.kind == config_schema.Kind.NUMBER:
        return float(spec.value or 0.0) * 1.5 + 0.25
    if spec.choices:
        others = [choice for choice in spec.choices if choice != spec.value]
        return others[-1]
    return f"{spec.value or 'name'}_edited"


def editable(file: tuned_file.TunedFile) -> list[config_schema.ParameterSpec]:
    return [spec for spec in file.rendered() if spec.kind in tuned_file.EDITABLE_KINDS]


class LiveUpdateCoverageTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def _files(self) -> list[tuple[str, str, Any]]:
        """(robot, path of a copy, message class) of every configuration file of every robot."""
        found = []
        for package, config in sorted(operator_test_support.ROBOT_CONFIGS.items()):
            copied = operator_test_support.copy_config(
                config, os.path.join(self.tmpdir, package)
            )
            for relative, message_class in FILES:
                path = os.path.join(copied, relative)
                if os.path.exists(path):
                    found.append((package, path, message_class))
        return found

    def test_every_value_the_file_sets_is_published_and_saved_and_nothing_else(self):
        for robot, path, message_class in self._files():
            with self.subTest(robot=robot, file=os.path.basename(path)):
                with open(path, encoding="utf-8", newline="") as handle:
                    original = handle.read()
                file = tuned_file.TunedFile(path, message_class)
                changed = {
                    spec.path: new_value(spec)
                    for spec in editable(file)
                    if spec.source == config_schema.ValueSource.FILE
                }
                self.assertTrue(changed)
                lines = {file.spec(key).line for key in changed}
                for key, value in changed.items():
                    file.set(key, value)
                edited = file.edited()
                after = {
                    spec.path: spec for spec in config_schema.parameters(edited.message)
                }
                for spec in file.parameters():
                    expected = changed.get(spec.path, spec.value)
                    self.assertEqual(after[spec.path].value, expected, spec.path)

                saved = file.save()
                self.assertEqual(saved.message, edited.message)
                with open(path, encoding="utf-8", newline="") as handle:
                    text = handle.read()
                self.assertEqual(text, edited.text)
                before_lines = original.splitlines()
                after_lines = text.splitlines()
                self.assertEqual(len(before_lines), len(after_lines))
                for number, (before, now) in enumerate(
                    zip(before_lines, after_lines), start=1
                ):
                    if number not in lines:
                        self.assertEqual(now, before, f"line {number}")

    def test_absent_values_are_inserted_and_read_back(self):
        for robot, path, message_class in self._files():
            with self.subTest(robot=robot, file=os.path.basename(path)):
                file = tuned_file.TunedFile(path, message_class)
                absent = [
                    spec
                    for spec in editable(file)
                    if spec.source != config_schema.ValueSource.FILE
                ]
                if not absent:
                    continue
                step = max(1, len(absent) // INSERTED_SAMPLE)
                sample = {spec.path: new_value(spec) for spec in absent[::step]}
                for key, value in sample.items():
                    file.set(key, value)
                file.save()
                reloaded = tuned_file.TunedFile(path, message_class)
                for key, value in sample.items():
                    spec = reloaded.spec(key)
                    self.assertEqual(spec.value, value, key)
                    self.assertEqual(spec.source, config_schema.ValueSource.FILE, key)

    def test_the_parameter_update_is_what_the_saved_files_parse_into(self):
        for package, config in sorted(operator_test_support.ROBOT_CONFIGS.items()):
            with self.subTest(robot=package):
                copied = operator_test_support.copy_config(
                    config, os.path.join(self.tmpdir, "update", package)
                )
                task = tuned_file.TunedFile(
                    os.path.join(copied, "mpc", "task.textproto"),
                    task_file_pb2.TaskFile,
                )
                planner_path = os.path.join(copied, "mpc", "contact_planning.textproto")
                planner = None
                if os.path.exists(planner_path):
                    planner = tuned_file.TunedFile(
                        planner_path, contact_planning_file_pb2.ContactPlanningFile
                    )
                for file in [task] + ([planner] if planner is not None else []):
                    first = editable(file)[0]
                    file.set(first.path, new_value(first))
                update = operator_bus.mpc_parameter_update(
                    task.edited().message,
                    planner.edited().message if planner is not None else None,
                    config_path=robot_config_save.config_path_of(task.path),
                )
                wire = mpc_parameter_update_pb2.MpcParameterUpdate.FromString(
                    update.SerializeToString()
                )
                self.assertEqual(wire, update)
                task.save()
                self.assertEqual(
                    wire.task,
                    nproto_textproto.load_textproto(task.path, task_file_pb2.TaskFile),
                )
                if planner is not None:
                    planner.save()
                    self.assertEqual(
                        wire.contact_planning,
                        nproto_textproto.load_textproto(
                            planner.path, contact_planning_file_pb2.ContactPlanningFile
                        ),
                    )
                else:
                    self.assertFalse(wire.HasField("contact_planning"))


if __name__ == "__main__":
    unittest.main()
