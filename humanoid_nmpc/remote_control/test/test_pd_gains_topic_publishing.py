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

"""The Joint PD Gains tab on the bus: a change is published as the whole edited file, and only Save writes it.

The tab is given the GUI's publisher of operator/pd_gains over a bus that records, so the tests assert on the
humanoid_mpc_config.JointPdGainsFile that would have gone out. The tab works on a copy of the DRC Atlas file. The
widget tests need a display; the schema checks do not.
"""

import os
import shutil
import tempfile
import tkinter as tk
from typing import Any
import unittest

from humanoid_mpc_config import joint_pd_gains_file_pb2

from humanoid_mpc_ipc import topics
import nproto_textproto
import operator_test_support
from remote_control.tk_app import joint_pd_tab
from remote_control.tk_app import slider_row

KNEE_KP = "joint_gains[joint=l_leg_kny].kp"
DEFAULT_KD = "default_gains.kd"


class TestTheSchemaTheTabReliesOn(unittest.TestCase):
    def test_the_scale_buttons_name_fields_of_both_gain_messages(self):
        for message in (
            joint_pd_gains_file_pb2.JointPdGainsFile.Gains,
            joint_pd_gains_file_pb2.JointPdGainsFile.JointGains,
        ):
            with self.subTest(message=message.DESCRIPTOR.full_name):
                self.assertIn(joint_pd_tab.KP_FIELD, message.DESCRIPTOR.fields_by_name)
                self.assertIn(joint_pd_tab.KD_FIELD, message.DESCRIPTOR.fields_by_name)

    def test_joints_are_grouped_by_limb(self):
        self.assertEqual(joint_pd_tab.classify_joint_section("l_leg_kny"), "Left Leg")
        self.assertEqual(
            joint_pd_tab.classify_joint_section("back_bkz"), "Torso & Spine"
        )
        self.assertEqual(joint_pd_tab.classify_joint_section("gripper"), "Other Joints")


@operator_test_support.requires_display
class TestPdGainsTopicPublishing(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.gains_file = os.path.join(
            operator_test_support.copy_config(
                operator_test_support.ATLAS_CONFIG, self.tmpdir
            ),
            "controller",
            "joint_pd_gains.textproto",
        )
        self.original = self._read()
        self.publisher = operator_test_support.RecordingPublisher(
            topics.OPERATOR_PD_GAINS
        )

    def _read(self) -> str:
        with open(self.gains_file, encoding="utf-8", newline="") as handle:
            return handle.read()

    def _tab(self, **kwargs: Any) -> joint_pd_tab.JointPdGainsTab:
        options: dict[str, Any] = {
            "enable_online_tuning": True,
            "param_publisher": self.publisher,
        }
        options.update(kwargs)
        return joint_pd_tab.JointPdGainsTab(
            self.root, pd_gains_file=self.gains_file, **options
        )

    def _slider(
        self, tab: joint_pd_tab.JointPdGainsTab, path: str
    ) -> slider_row.SliderRow:
        row = tab.slider_rows[path]
        assert isinstance(row, slider_row.SliderRow), path
        return row

    def _flush(self, tab: joint_pd_tab.JointPdGainsTab) -> None:
        if tab._debounce_publish_id is not None:
            tab.after_cancel(tab._debounce_publish_id)
        tab._publish_to_topic()

    def _published(self) -> joint_pd_gains_file_pb2.JointPdGainsFile:
        message = self.publisher.last_message
        assert isinstance(message, joint_pd_gains_file_pb2.JointPdGainsFile), message
        return message

    def test_every_gain_has_a_row_and_no_torque_limit_does(self):
        tab = self._tab()
        assert tab.gains is not None
        self.assertEqual(
            set(tab.slider_rows), {spec.path for spec in tab.gains.rendered()}
        )
        self.assertIn(KNEE_KP, tab.slider_rows)
        self.assertIn(DEFAULT_KD, tab.slider_rows)
        self.assertFalse(any(path.endswith("torque_limit") for path in tab.slider_rows))

    def test_a_change_publishes_the_file_and_writes_nothing(self):
        tab = self._tab()
        row = self._slider(tab, KNEE_KP)
        row.set_value(row.get_value() * 1.5)
        self._flush(tab)
        published = self._published()
        expected = nproto_textproto.load_textproto(
            self.gains_file, joint_pd_gains_file_pb2.JointPdGainsFile
        )
        next(
            entry for entry in expected.joint_gains if entry.joint == "l_leg_kny"
        ).kp = row.get_value()
        self.assertEqual(published, expected)
        self.assertEqual(self._read(), self.original)

    def test_save_writes_the_change_and_keeps_every_other_line(self):
        tab = self._tab()
        row = self._slider(tab, KNEE_KP)
        row.set_value(row.get_value() + 10.0)
        self.assertTrue(tab.save())
        changed = [
            after
            for before, after in zip(
                self.original.splitlines(), self._read().splitlines()
            )
            if before != after
        ]
        self.assertEqual(len(changed), 1, changed)
        self.assertIn('joint: "l_leg_kny"', changed[0])
        self.assertTrue(os.path.exists(self.gains_file + ".bak"))
        self.assertFalse(row.is_modified())

    def test_the_scale_buttons_scale_every_kp_from_its_saved_value(self):
        tab = self._tab()
        assert tab.gains is not None
        saved = {
            path: tab.gains.saved_value(path)
            for path in tab.slider_rows
            if path.endswith("." + joint_pd_tab.KP_FIELD)
        }
        tab._scale_all(joint_pd_tab.KP_FIELD, 0.5)
        for path, value in saved.items():
            self.assertAlmostEqual(tab.slider_rows[path].get_value(), value * 0.5)
        kd = self._slider(tab, DEFAULT_KD)
        self.assertFalse(kd.is_modified())
        self._flush(tab)
        published = self._published()
        self.assertAlmostEqual(
            published.default_gains.kp, saved["default_gains.kp"] * 0.5
        )

    def test_a_joint_that_inherits_a_gain_keeps_inheriting_it_through_the_scale_buttons(
        self,
    ):
        with open(self.gains_file, "a", encoding="utf-8") as handle:
            # A joint the file gives only its kd: its kp is default_gains', which the schema allows.
            handle.write('joint_gains { joint: "inheriting_joint" kd: 12.0 }\n')
        tab = self._tab()
        assert tab.gains is not None
        inherited = "joint_gains[joint=inheriting_joint].kp"
        self.assertIsNone(tab.gains.saved_value(inherited))
        default_kp = tab.gains.saved_value("default_gains.kp")
        tab._scale_all(joint_pd_tab.KP_FIELD, 1.0)
        tab._scale_all(joint_pd_tab.KP_FIELD, 2.0)
        self.assertNotIn(inherited, tab.gains.changes())
        self.assertFalse(tab.slider_rows[inherited].has_value())
        self._flush(tab)
        published = self._published()
        entry = next(
            entry
            for entry in published.joint_gains
            if entry.joint == "inheriting_joint"
        )
        self.assertFalse(entry.HasField("kp"))
        self.assertEqual(entry.kd, 12.0)
        self.assertAlmostEqual(published.default_gains.kp, default_kp * 2.0)

    def test_reload_drops_the_changes_and_publishes_the_file(self):
        tab = self._tab()
        row = self._slider(tab, KNEE_KP)
        row.set_value(row.get_value() * 2.0)
        self._flush(tab)
        self.assertNotEqual(
            self._published(),
            nproto_textproto.load_textproto(
                self.gains_file, joint_pd_gains_file_pb2.JointPdGainsFile
            ),
        )
        tab.reload_file()
        # The robot runs what it was last sent; Reload sends the file, so that it runs what the rows show.
        self.assertEqual(
            self._published(),
            nproto_textproto.load_textproto(
                self.gains_file, joint_pd_gains_file_pb2.JointPdGainsFile
            ),
        )
        self.assertEqual(self._read(), self.original)

    def test_leaving_an_untouched_gain_box_changes_nothing(self):
        tab = self._tab()
        assert tab.gains is not None
        for row in tab.slider_rows.values():
            if isinstance(row, slider_row.SliderRow):
                row._on_entry_submit()  # what <FocusOut> calls
                row.reset_to_default()
        self.assertEqual(tab.gains.changes(), {})
        self._flush(tab)
        self.assertEqual(
            self._published(),
            nproto_textproto.load_textproto(
                self.gains_file, joint_pd_gains_file_pb2.JointPdGainsFile
            ),
        )

    def test_reset_all_returns_to_the_file(self):
        tab = self._tab()
        row = self._slider(tab, KNEE_KP)
        original = row.get_value()
        row.set_value(original * 2.0)
        tab.reset_all_defaults()
        self.assertEqual(row.get_value(), original)
        self._flush(tab)
        self.assertEqual(
            self._published(),
            nproto_textproto.load_textproto(
                self.gains_file, joint_pd_gains_file_pb2.JointPdGainsFile
            ),
        )
        self.assertEqual(self._read(), self.original)

    def test_without_a_publisher_or_online_tuning_nothing_is_sent(self):
        tab = self._tab(param_publisher=None)
        self._slider(tab, KNEE_KP).set_value(1.0)
        self._flush(tab)
        locked = self._tab(enable_online_tuning=False)
        locked._publish_to_topic()
        self.assertEqual(self.publisher.publish_count, 0)
        self.assertFalse(locked.save())

    def test_rapid_changes_publish_once(self):
        tab = self._tab()
        row = self._slider(tab, KNEE_KP)
        for step in range(5):
            row.set_value(row.get_value() + step)
        self.assertIsNotNone(tab._debounce_publish_id)
        self._flush(tab)
        self.assertEqual(self.publisher.publish_count, 1)

    def test_the_rows_are_grouped_by_limb_with_the_default_gains_first(self):
        tab = self._tab()
        shown = [
            section
            for section, frame in tab.section_frames.items()
            if frame.winfo_children()
        ]
        self.assertEqual(shown[0], joint_pd_tab.DEFAULT_GAINS_SECTION)
        self.assertIn("Left Leg", shown)


if __name__ == "__main__":
    unittest.main()
