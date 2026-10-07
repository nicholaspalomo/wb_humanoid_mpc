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

"""The error paths of the remote control's file readers: an unreadable file falls back, and never raises.

The readers used to catch every exception. They now catch the ones their files can raise (an unreadable file, a
textproto that does not parse strictly into its schema), so each case here is one the narrower handlers must still
cover. Also the helpers that moved out of the tab classes.
"""

import os
import shutil
import tempfile
import types
from typing import Any
import unittest

from remote_control import config_files
from remote_control.tk_app import combobox
from remote_control.tk_app import joint_targets_tab

_REFERENCE_HEADER = (
    "# proto-file: humanoid_nmpc/humanoid_mpc_config/reference_file.proto\n"
    "# proto-message: humanoid_mpc_config.ReferenceFile\n"
)
_GAINS_HEADER = (
    "# proto-file: humanoid_nmpc/humanoid_mpc_config/joint_pd_gains_file.proto\n"
    "# proto-message: humanoid_mpc_config.JointPdGainsFile\n"
)


class TestReferenceDefaults(unittest.TestCase):
    """config_files.read_default_joint_state(): the default joint positions of the reference file, by joint name."""

    def _read(self, text: str) -> dict[str, float]:
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        path = os.path.join(directory, "reference.textproto")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return config_files.read_default_joint_state(path)

    def test_positive_control_the_named_entries_are_read(self):
        text = (
            _REFERENCE_HEADER
            + "default_base_height: 0.8\n"
            + 'default_joint_state { joint: "left_hip_pitch_joint" value: -0.15 }\n'
            + 'default_joint_state { joint: "left_knee_joint" value: 0.3 }\n'
        )
        self.assertEqual(
            self._read(text), {"left_hip_pitch_joint": -0.15, "left_knee_joint": 0.3}
        )

    def test_no_file_is_no_defaults(self):
        self.assertEqual(config_files.read_default_joint_state(None), {})
        self.assertEqual(config_files.read_default_joint_state(""), {})
        self.assertEqual(
            config_files.read_default_joint_state("/nonexistent/reference.textproto"),
            {},
        )

    def test_an_empty_file_is_no_defaults(self):
        self.assertEqual(self._read(""), {})

    def test_a_file_that_does_not_parse_is_no_defaults(self):
        self.assertEqual(self._read("default_joint_state [unclosed\n"), {})
        # The retired YAML key is refused by the strict parser, not read.
        self.assertEqual(self._read(_REFERENCE_HEADER + "defaultJointState {}\n"), {})

    def test_the_joints_of_the_gains_file_in_its_order(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        path = os.path.join(directory, "joint_pd_gains.textproto")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                _GAINS_HEADER
                + 'joint_gains { joint: "b_joint" kp: 1 }\njoint_gains { joint: "a_joint" kp: 2 }\n'
            )
        self.assertEqual(
            config_files.read_pd_gains_joint_names(path), ["b_joint", "a_joint"]
        )
        self.assertEqual(config_files.read_pd_gains_joint_names(None), [])
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(_GAINS_HEADER + "joint_gains { unknown: 1 }\n")
        self.assertEqual(config_files.read_pd_gains_joint_names(path), [])

    def test_joints_are_grouped_by_limb(self):
        self.assertEqual(
            joint_targets_tab._classify_joint_section("left_knee_joint"), "Left Leg"
        )
        self.assertEqual(
            joint_targets_tab._classify_joint_section("waist_yaw_joint"),
            "Torso & Spine",
        )
        self.assertEqual(
            joint_targets_tab._classify_joint_section("gripper"), "Other Joints"
        )


class _FakeCombobox:
    def __init__(self, part: str) -> None:
        self.part = part
        self.generated: list[tuple[str, str]] = []

    def identify(self, x: int, y: int) -> str:
        del x, y
        return self.part

    def event_generate(self, sequence: str, when: str) -> None:
        self.generated.append((sequence, when))


class TestComboboxClick(unittest.TestCase):
    def _click(self, part: str) -> list[tuple[str, str]]:
        widget = _FakeCombobox(part)
        event: Any = types.SimpleNamespace(widget=widget, x=3, y=4)
        combobox.open_dropdown_on_click(event)
        return widget.generated

    def test_a_click_beside_the_arrow_opens_the_list(self):
        self.assertEqual(self._click("textarea"), [("<Down>", "head")])

    def test_a_click_on_the_arrow_is_left_to_the_combobox(self):
        self.assertEqual(self._click("downarrow"), [])


if __name__ == "__main__":
    unittest.main()
