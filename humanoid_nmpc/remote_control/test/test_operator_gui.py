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

"""OperatorGui, the loop that runs the window against the bus: what one tick publishes, and what it hands the window.

The headless tests drive OperatorGui with a stand-in window that records what it is told; the display tests build the
real App over an operator bus that records, and check the wiring end to end: the walking command of the on-screen
sticks, the FSM commands of the mode selector and the gantry checkbox, the robot's FSM state, and the tabs' publishers.
"""

from collections.abc import Callable
import contextlib
import io
import math
import os
import shutil
import signal
import tempfile
import time
from typing import cast
import unittest

from humanoid_mpc_msgs import fsm_command_pb2
from humanoid_mpc_msgs import fsm_state_pb2
from humanoid_mpc_msgs import walking_velocity_command_pb2

from humanoid_mpc_ipc import topics
import operator_test_support
from remote_control import base_velocity_controller_gui
from remote_control import config_files
from remote_control import operator_bus
from remote_control import teleop


class FakeApp:
    """Records what OperatorGui tells the window, and holds the command of its sticks."""

    def __init__(self) -> None:
        self.fsm_command_callback: Callable[[str], None] | None = None
        self.states: list[fsm_state_pb2.FsmState] = []
        self.knobs: list[walking_velocity_command_pb2.WalkingVelocityCommand] = []
        self.joystick_connected: list[bool] = []
        self.sticks = operator_bus.walking_velocity_command(0.25, 0.0, -0.5, 0.75)
        self.reference_file_polls = 0

    def set_fsm_command_callback(self, callback) -> None:
        self.fsm_command_callback = callback

    def update_fsm_state(self, message) -> None:
        self.states.append(message)

    def set_knob_positions(self, message) -> None:
        self.knobs.append(message)

    def set_joystick_connected(self, connected: bool) -> None:
        self.joystick_connected.append(connected)

    def get_walking_command_msg(self):
        return self.sticks

    def poll_reference_file(self) -> None:
        self.reference_file_polls += 1


def _as_app(app: FakeApp) -> base_velocity_controller_gui.App:
    """A stand-in as the window OperatorGui takes: OperatorGui calls only the methods FakeApp has."""
    return cast(base_velocity_controller_gui.App, app)


class FakeGamepad:
    """A GamepadPoller stand-in: connected or not, with the command it reads."""

    def __init__(self, connected: bool, command=None) -> None:
        self.connected = connected
        self.command = command
        self.ticks = 0

    def tick(self):
        self.ticks += 1
        return self.command if self.connected else None


class TestOperatorGuiTick(unittest.TestCase):
    def _gui(self, gamepad=None):
        self.bus = operator_test_support.RecordingBus()
        self.gui_bus = operator_bus.OperatorBus(self.bus)
        self.app = FakeApp()
        return base_velocity_controller_gui.OperatorGui(
            _as_app(self.app), self.gui_bus, gamepad=gamepad
        )

    def _walking_commands(self):
        return self.bus.messages_on(topics.OPERATOR_WALKING_VELOCITY_COMMAND)

    def test_without_a_controller_every_tick_publishes_the_sticks(self):
        gui = self._gui()
        for _ in range(3):
            gui.tick()
        self.assertEqual(self._walking_commands(), [self.app.sticks] * 3)
        self.assertEqual(self.app.joystick_connected, [False] * 3)

    def test_an_all_zero_command_is_still_published(self):
        # The height slider reaches the gantry in simulation only through this topic.
        gui = self._gui()
        self.app.sticks = operator_bus.walking_velocity_command(0.0, 0.0, 0.0, 0.6)
        gui.tick()
        self.assertEqual(self._walking_commands()[-1].desired_pelvis_height, 0.6)

    def test_a_connected_controller_drives_the_command_and_the_sticks(self):
        controller_command = operator_bus.walking_velocity_command(1.0, 0.0, 0.0, 0.9)
        gui = self._gui(FakeGamepad(connected=True, command=controller_command))
        gui.tick()
        self.assertEqual(self._walking_commands(), [controller_command])
        self.assertEqual(self.app.knobs, [controller_command])
        self.assertEqual(self.app.joystick_connected, [True])

    def test_a_controller_that_could_not_be_read_publishes_nothing_that_tick(self):
        gui = self._gui(FakeGamepad(connected=True, command=None))
        gui.tick()
        self.assertEqual(self._walking_commands(), [])

    def test_a_disconnected_controller_is_still_polled_for_its_rescans(self):
        gamepad = FakeGamepad(connected=False)
        gui = self._gui(gamepad)
        gui.tick()
        self.assertEqual(gamepad.ticks, 1)
        self.assertEqual(self._walking_commands(), [self.app.sticks])

    def test_the_robots_states_reach_the_window_in_order_before_the_command(self):
        gui = self._gui()
        first = fsm_state_pb2.FsmState(mode="JOINT_PD", controller_resets=1)
        second = fsm_state_pb2.FsmState(mode="WB_MPC", controller_resets=1)
        self.bus.deliver(topics.ROBOT_FSM_STATE, first)
        self.bus.deliver(topics.ROBOT_FSM_STATE, second)
        gui.tick()
        self.assertEqual(self.app.states, [first, second])
        gui.tick()
        self.assertEqual(self.app.states, [first, second], "a state is applied once")

    def test_the_windows_fsm_commands_go_out_numbered(self):
        self._gui()
        callback = self.app.fsm_command_callback
        assert callback is not None
        callback("JOINT_PD")
        callback("LOCK_GANTRY")
        commands = self.bus.messages_on(topics.OPERATOR_FSM_COMMAND)
        self.assertEqual(
            [command.command for command in commands], ["JOINT_PD", "LOCK_GANTRY"]
        )
        self.assertLess(commands[0].sequence, commands[1].sequence)

    def test_the_reference_file_is_polled_about_once_a_second(self):
        gui = self._gui()
        for _ in range(round(teleop.WALKING_COMMAND_RATE_HZ) * 3):
            gui.tick()
        self.assertEqual(self.app.reference_file_polls, 3)

    def test_a_moved_default_keeps_the_operators_offset(self):
        self.assertEqual(
            base_velocity_controller_gui.moved_pelvis_height(0.8, 0.8, 0.9), 0.9
        )
        self.assertAlmostEqual(
            base_velocity_controller_gui.moved_pelvis_height(0.75, 0.8, 0.9), 0.85
        )

    def test_the_rate_must_be_positive(self):
        for rate_hz in (0.0, -1.0, math.nan):
            with self.subTest(rate_hz=rate_hz):
                with self.assertRaisesRegex(ValueError, "the rate must be positive"):
                    base_velocity_controller_gui.OperatorGui(
                        _as_app(FakeApp()),
                        operator_bus.OperatorBus(operator_test_support.RecordingBus()),
                        rate_hz=rate_hz,
                    )


class SignaledApp(FakeApp):
    """A window whose main loop is ended by a signal the process sends itself, as tools/launch sends it."""

    def __init__(self, signum: int) -> None:
        super().__init__()
        self.signum = signum
        self.loop_ended_by_itself = False

    def after(self, delay_ms, callback) -> None:
        # The ticks are not run: only the shutdown is under test.
        del delay_ms, callback

    def mainloop(self) -> None:
        os.kill(os.getpid(), self.signum)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            time.sleep(0.01)
        self.loop_ended_by_itself = True


class TestOperatorGuiShutdown(unittest.TestCase):
    """SIGTERM and SIGHUP end run() as closing the window does: the bus is closed and on_close stops the Rerun bridge."""

    def _run_until_signaled(self, signum: int) -> None:
        """Runs OperatorGui until `signum` ends its main loop; checks the cleanup and the restored handler."""
        bus = operator_test_support.RecordingBus()
        app = SignaledApp(signum)
        closed: list[bool] = []
        gui = base_velocity_controller_gui.OperatorGui(
            _as_app(app),
            operator_bus.OperatorBus(bus),
            on_close=lambda: closed.append(True),
        )
        handler_before = signal.getsignal(signum)
        gui.run()
        self.assertFalse(
            app.loop_ended_by_itself, "the signal did not end the main loop"
        )
        self.assertTrue(bus.closed)
        self.assertEqual(closed, [True])
        self.assertIs(
            signal.getsignal(signum), handler_before, "run() left its handler installed"
        )

    def test_sigterm_ends_the_gui_with_its_cleanup(self):
        self._run_until_signaled(signal.SIGTERM)

    def test_sighup_ends_the_gui_with_its_cleanup(self):
        self._run_until_signaled(signal.SIGHUP)


class TestCommandLine(unittest.TestCase):
    def test_the_flags_and_their_defaults(self):
        args = base_velocity_controller_gui.build_parser().parse_args([])
        self.assertEqual(args.network_config, "config/ipc/network.textproto")
        self.assertEqual(args.ipc_node, "operator")
        self.assertEqual(args.task_file, "")
        self.assertIsNone(args.default_pelvis_height)
        self.assertEqual(args.urdf_file, "")
        args = base_velocity_controller_gui.build_parser().parse_args(
            [
                "--task_file=t.textproto",
                "--reference_file=r.textproto",
                "--pd_gains_file=p.textproto",
                "--default_pelvis_height=0.7",
            ]
        )
        self.assertEqual(
            (args.task_file, args.reference_file, args.pd_gains_file),
            ("t.textproto", "r.textproto", "p.textproto"),
        )
        self.assertAlmostEqual(args.default_pelvis_height, 0.7)

    def test_the_ros_arguments_of_the_old_launch_files_are_refused(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit):
            base_velocity_controller_gui.build_parser().parse_args(
                ["--ros-args", "-p", "task_file:=t.textproto"]
            )
        self.assertIn("--ros-args", errors.getvalue())


@operator_test_support.requires_display
class TestTheWindowOnTheBus(unittest.TestCase):
    """The real App over an operator bus that records."""

    def setUp(self):
        self.bus = operator_test_support.RecordingBus()
        self.gui_bus = operator_bus.OperatorBus(self.bus)
        # Online tuning off: the tabs only read the shipped files.
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
            param_publisher=self.gui_bus.mpc_parameters,
            pd_gains_publisher=self.gui_bus.pd_gains,
            joint_targets_publisher=self.gui_bus.joint_targets,
            dodgeball_publisher=self.gui_bus.dodgeball_throw,
        )
        self.app.withdraw()
        self.addCleanup(self.app.destroy)
        self.gui = base_velocity_controller_gui.OperatorGui(self.app, self.gui_bus)

    def test_the_sticks_are_the_walking_command(self):
        self.app.joystick_left.set_position(0.5, 0.0)
        self.gui.tick()
        command = self.bus.messages_on(topics.OPERATOR_WALKING_VELOCITY_COMMAND)[-1]
        self.assertAlmostEqual(command.linear_velocity_x, self.app.joystick_left.x_norm)
        self.assertGreater(command.linear_velocity_x, 0.0)
        self.assertGreater(command.desired_pelvis_height, 0.0)

    def test_the_mode_selector_and_the_gantry_checkbox_send_fsm_commands(self):
        self.app.fsm_mode_var.set("JOINT_PD")
        self.app._on_fsm_change(None)
        self.app.gantry_var.set(False)
        self.app._on_gantry_toggle()
        commands = self.bus.messages_on(topics.OPERATOR_FSM_COMMAND)
        self.assertEqual(
            [command.command for command in commands], ["JOINT_PD", "UNLOCK_GANTRY"]
        )
        for command in commands:
            self.assertIsInstance(command, fsm_command_pb2.FsmCommand)

    def test_a_state_from_the_robot_moves_the_mode_selector(self):
        self.bus.deliver(
            topics.ROBOT_FSM_STATE,
            fsm_state_pb2.FsmState(mode="GRAVITY_COMP", gantry_locked=False),
        )
        self.gui.tick()
        self.assertEqual(self.app.fsm_mode_var.get(), "GRAVITY_COMP")
        self.assertFalse(self.app.gantry_var.get())

    def test_each_tab_publishes_through_its_topics_publisher(self):
        self.assertIs(
            self.app.mpc_params_tab.param_publisher, self.gui_bus.mpc_parameters
        )
        self.assertIs(self.app.joint_pd_tab.param_publisher, self.gui_bus.pd_gains)
        self.assertIs(
            self.app.joint_targets_tab.param_publisher, self.gui_bus.joint_targets
        )
        self.assertIs(
            self.app.dodgeball_tab.throw_publisher, self.gui_bus.dodgeball_throw
        )
        self.app.dodgeball_tab.throw()
        self.assertEqual(len(self.bus.messages_on(topics.OPERATOR_DODGEBALL_THROW)), 1)

    def test_without_a_launcher_the_rerun_button_is_disabled(self):
        self.assertIn("disabled", self.app.rerun_viewer_btn.state())

    def test_joint_targets_are_published_in_joint_pd_only(self):
        tab = self.app.joint_targets_tab
        if not tab.slider_rows:
            self.skipTest("the Atlas reference file names no joints")
        tab._publish_to_topic()
        self.assertEqual(self.bus.messages_on(topics.OPERATOR_JOINT_TARGETS), [])
        self.app.fsm_mode_var.set("JOINT_PD")
        tab._publish_to_topic()
        targets = self.bus.messages_on(topics.OPERATOR_JOINT_TARGETS)
        self.assertEqual(len(targets), 1)
        self.assertEqual(set(targets[0].positions), set(tab.slider_rows))
        for name, row in tab.slider_rows.items():
            self.assertEqual(targets[0].positions[name], row.get_value())


@operator_test_support.requires_display
class TestTheDefaultPelvisHeightFollowsTheReferenceFile(unittest.TestCase):
    """The height slider centers on the reference file's default_base_height, also after the file changed."""

    def setUp(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        config = operator_test_support.copy_config(
            operator_test_support.ATLAS_CONFIG, directory
        )
        self.reference_file = os.path.join(config, "command", "reference.textproto")
        self.app = base_velocity_controller_gui.App(
            task_file=os.path.join(config, "mpc", "task.textproto"),
            reference_file=self.reference_file,
            enable_online_tuning=False,
        )
        self.app.withdraw()
        self.addCleanup(self.app.destroy)
        self.shipped = config_files.read_default_pelvis_height(self.reference_file)
        self.app.set_default_pelvis_height(self.shipped)

    def _commanded_height(self) -> float:
        return self.app.get_walking_command_msg().desired_pelvis_height

    def _save_default(self, height: float) -> None:
        tab = self.app.command_limits_tab
        assert tab.reference is not None
        tab.reference.set("default_base_height", height)
        self.assertTrue(tab.save())

    def test_a_saved_default_moves_the_next_command(self):
        self.assertAlmostEqual(self._commanded_height(), self.shipped, places=6)
        self._save_default(self.shipped - 0.1)
        self.assertAlmostEqual(self._commanded_height(), self.shipped - 0.1, places=6)
        self.app.center_all()
        self.assertAlmostEqual(self._commanded_height(), self.shipped - 0.1, places=6)

    def test_a_height_the_operator_set_keeps_its_offset(self):
        self.app.slider.set(self.app.slider.get() - 5.0)
        offset = self._commanded_height() - self.shipped
        self.assertLess(offset, 0.0)
        self._save_default(self.shipped - 0.1)
        self.assertAlmostEqual(
            self._commanded_height(), self.shipped - 0.1 + offset, places=6
        )

    def test_an_edit_of_the_file_is_followed_at_the_next_poll(self):
        with open(self.reference_file, encoding="utf-8") as file:
            text = file.read()
        edited = text.replace(
            f"default_base_height: {self.shipped:g}",
            f"default_base_height: {self.shipped - 0.05:g}",
        )
        self.assertNotEqual(edited, text, "the edit changed nothing")
        with open(self.reference_file, "w", encoding="utf-8") as file:
            file.write(edited)
        os.utime(self.reference_file, (time.time() + 5.0, time.time() + 5.0))
        self.app.poll_reference_file()
        self.assertAlmostEqual(self._commanded_height(), self.shipped - 0.05, places=6)

    def test_a_default_the_command_line_set_stays(self):
        self.app.set_default_pelvis_height(0.7)
        self.app.follows_reference_file_default = False
        self._save_default(self.shipped - 0.1)
        self.assertAlmostEqual(self._commanded_height(), 0.7, places=6)


if __name__ == "__main__":
    unittest.main()
