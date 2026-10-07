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

"""The Tk widgets' constructors and the small widget behaviors the style sweep changed, against withdrawn windows.

Every frame of the GUI took `*args` after keyword defaults, so no extra positional argument could ever reach the frame
without first filling every keyword parameter by position. They now take the frame's options as keywords only, and the
tests check that those reach the ttk.Frame. Also: a joystick built without an auto-center variable can be released
(its default used to be False, whose .get() raised), SliderRow.set_default(), and the GUI's controller indicator.
"""

import inspect
import os
import tkinter as tk
from typing import Any, cast
import unittest

import operator_test_support
from remote_control import base_velocity_controller_gui
from remote_control.tk_app import command_limits_tab
from remote_control.tk_app import dodgeball_tab
from remote_control.tk_app import joint_pd_tab
from remote_control.tk_app import joint_targets_tab
from remote_control.tk_app import joystick_gui
from remote_control.tk_app import mpc_params_tab
from remote_control.tk_app import scrollable_frame
from remote_control.tk_app import slider_row

_FRAMES = (
    scrollable_frame.ScrollableFrame,
    slider_row.SliderRow,
    command_limits_tab.CommandLimitsTab,
    dodgeball_tab.DodgeballTab,
    joint_pd_tab.JointPdGainsTab,
    joint_targets_tab.JointTargetsTab,
    mpc_params_tab.MpcParamsTab,
)


class TestConstructorSignatures(unittest.TestCase):
    def test_no_frame_takes_extra_positional_arguments(self):
        for frame_class in _FRAMES:
            with self.subTest(frame=frame_class.__name__):
                kinds = [
                    parameter.kind
                    for parameter in inspect.signature(
                        frame_class.__init__
                    ).parameters.values()
                ]
                self.assertNotIn(inspect.Parameter.VAR_POSITIONAL, kinds)
                # Positive control: the frame's options are still accepted, as keywords.
                self.assertIn(inspect.Parameter.VAR_KEYWORD, kinds)


@operator_test_support.requires_display
class TestFrameOptions(unittest.TestCase):
    """A ttk.Frame option given to a widget reaches its frame."""

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)

    def _assert_cursor(self, frame: tk.Misc) -> None:
        self.assertEqual(str(frame.cget("cursor")), "hand2")

    def test_scrollable_frame(self):
        self._assert_cursor(scrollable_frame.ScrollableFrame(self.root, cursor="hand2"))

    def test_slider_row(self):
        self._assert_cursor(slider_row.SliderRow(self.root, "kp", 1.0, cursor="hand2"))

    def test_slider_row_refuses_an_extra_positional_argument(self):
        # Every parameter by position, and one more: it used to reach the frame as its `cnf`.
        arguments: list[Any] = ["kp", 1.0, 0.0, 10.0, "", None, 24, False, "extra"]
        with self.assertRaises(TypeError):
            # pylint: disable-next=too-many-function-args  # The refusal is under test.
            slider_row.SliderRow(self.root, *arguments)

    def test_command_limits_tab(self):
        self._assert_cursor(
            command_limits_tab.CommandLimitsTab(
                self.root, reference_file=None, cursor="hand2"
            )
        )

    def test_dodgeball_tab(self):
        self._assert_cursor(
            dodgeball_tab.DodgeballTab(self.root, throw_publisher=None, cursor="hand2")
        )

    def test_joint_targets_tab(self):
        self._assert_cursor(
            joint_targets_tab.JointTargetsTab(self.root, cursor="hand2")
        )

    def test_joint_pd_gains_tab(self):
        self._assert_cursor(
            joint_pd_tab.JointPdGainsTab(
                self.root,
                pd_gains_file=os.path.join(
                    operator_test_support.ATLAS_CONFIG,
                    "controller/joint_pd_gains.textproto",
                ),
                enable_online_tuning=False,
                cursor="hand2",
            )
        )

    def test_mpc_params_tab(self):
        self._assert_cursor(
            mpc_params_tab.MpcParamsTab(
                self.root,
                task_file=os.path.join(
                    operator_test_support.ATLAS_CONFIG, "mpc/task.textproto"
                ),
                enable_online_tuning=False,
                cursor="hand2",
            )
        )


@operator_test_support.requires_display
class TestWidgets(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)

    def test_a_joystick_without_auto_center_stays_where_it_was_released(self):
        joystick = joystick_gui.JoystickGui(self.root)
        joystick.set_position(0.5, 0.0)
        joystick.stop_drag(cast(tk.Event, None))
        self.assertEqual(joystick.x_norm, 0.5)

    def test_positive_control_a_joystick_with_auto_center_springs_back(self):
        joystick = joystick_gui.JoystickGui(
            self.root, auto_center_var=tk.BooleanVar(self.root, value=True)
        )
        joystick.set_position(0.5, 0.0)
        joystick.stop_drag(cast(tk.Event, None))
        self.assertEqual(joystick.x_norm, 0.0)

    def test_set_default_moves_the_reset_point_and_clears_the_highlight(self):
        row = slider_row.SliderRow(self.root, "kp", 1.0)
        row.set_value(2.0)
        self.assertTrue(row.is_modified())
        highlighted = str(row.label.cget("foreground"))
        row.set_default(2.0)
        self.assertFalse(row.is_modified())
        self.assertNotEqual(str(row.label.cget("foreground")), highlighted)
        row.reset_to_default()
        self.assertEqual(row.get_value(), 2.0)
        # Positive control: moving away from the new default highlights the row again.
        row.set_value(3.0)
        self.assertEqual(str(row.label.cget("foreground")), highlighted)


@operator_test_support.requires_display
class TestControllerIndicator(unittest.TestCase):
    """App.set_joystick_connected(), called every tick: the first call and every change reconfigure the window."""

    def setUp(self):
        self.app = base_velocity_controller_gui.App(
            pd_gains_file=os.path.join(
                operator_test_support.ATLAS_CONFIG,
                "controller/joint_pd_gains.textproto",
            ),
            task_file=os.path.join(
                operator_test_support.ATLAS_CONFIG, "mpc/task.textproto"
            ),
            reference_file=os.path.join(
                operator_test_support.ATLAS_CONFIG, "command/reference.textproto"
            ),
            enable_online_tuning=False,
        )
        self.app.withdraw()
        self.addCleanup(self.app.destroy)

    def _center_button_state(self) -> str:
        return str(self.app.center_button.cget("state"))

    def test_a_connected_controller_disables_the_on_screen_centering(self):
        self.app.set_joystick_connected(True)
        self.assertEqual(self._center_button_state(), "disabled")
        self.app.set_joystick_connected(False)
        self.assertEqual(self._center_button_state(), "normal")

    def test_the_first_call_applies_even_when_nothing_is_connected(self):
        self.app.center_button.configure(state="disabled")
        self.app.set_joystick_connected(False)
        self.assertEqual(self._center_button_state(), "normal")

    def test_a_repeated_state_is_not_applied_again(self):
        self.app.set_joystick_connected(False)
        self.app.center_button.configure(state="disabled")
        self.app.set_joystick_connected(False)
        self.assertEqual(self._center_button_state(), "disabled")


if __name__ == "__main__":
    unittest.main()
