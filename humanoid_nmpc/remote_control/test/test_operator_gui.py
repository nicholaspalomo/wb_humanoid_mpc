"""OperatorGui, the loop that runs the window against the bus: what one tick publishes, and what it hands the window.

The headless tests drive OperatorGui with a stand-in window that records what it is told; the display tests build the
real App over an operator bus that records, and check the wiring end to end: the walking command of the on-screen
sticks, the FSM commands of the mode selector and the gantry checkbox, the robot's FSM state, and the tabs' publishers.
"""

import contextlib
import io
import os
import signal
import time
import unittest
from typing import List, Optional

from humanoid_mpc_ipc import topics
from humanoid_mpc_msgs import fsm_command_pb2
from humanoid_mpc_msgs import fsm_state_pb2
from humanoid_mpc_msgs import walking_velocity_command_pb2
from operator_test_support import ATLAS_CONFIG, RecordingBus, requires_display
from remote_control.operator_bus import OperatorBus, walking_velocity_command


class FakeApp:
    """Records what OperatorGui tells the window, and holds the command of its sticks."""

    def __init__(self) -> None:
        self.fsm_command_callback = None
        self.states: List[fsm_state_pb2.FsmState] = []
        self.knobs: List[walking_velocity_command_pb2.WalkingVelocityCommand] = []
        self.joystick_connected: List[bool] = []
        self.sticks = walking_velocity_command(0.25, 0.0, -0.5, 0.75)

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
        from remote_control.base_velocity_controller_gui import OperatorGui

        self.bus = RecordingBus()
        self.operator_bus = OperatorBus(self.bus)
        self.app = FakeApp()
        return OperatorGui(self.app, self.operator_bus, gamepad=gamepad)

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
        self.app.sticks = walking_velocity_command(0.0, 0.0, 0.0, 0.6)
        gui.tick()
        self.assertEqual(self._walking_commands()[-1].desired_pelvis_height, 0.6)

    def test_a_connected_controller_drives_the_command_and_the_sticks(self):
        controller_command = walking_velocity_command(1.0, 0.0, 0.0, 0.9)
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
        self.app.fsm_command_callback("JOINT_PD")
        self.app.fsm_command_callback("LOCK_GANTRY")
        commands = self.bus.messages_on(topics.OPERATOR_FSM_COMMAND)
        self.assertEqual(
            [command.command for command in commands], ["JOINT_PD", "LOCK_GANTRY"]
        )
        self.assertLess(commands[0].sequence, commands[1].sequence)

    def test_the_rate_must_be_positive(self):
        from remote_control.base_velocity_controller_gui import OperatorGui

        with self.assertRaises(ValueError):
            OperatorGui(FakeApp(), OperatorBus(RecordingBus()), rate_hz=0.0)


class SignaledApp(FakeApp):
    """A window whose main loop is ended by a signal the process sends itself, as tools/launch sends it."""

    def __init__(self, signum: int) -> None:
        super().__init__()
        self.signum = signum
        self.loop_ended_by_itself = False

    def after(self, delay_ms, callback) -> None:
        del (
            delay_ms,
            callback,
        )  # The ticks are not run: only the shutdown is under test.

    def mainloop(self) -> None:
        os.kill(os.getpid(), self.signum)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            time.sleep(0.01)
        self.loop_ended_by_itself = True


class TestOperatorGuiShutdown(unittest.TestCase):
    """SIGTERM and SIGHUP end run() as closing the window does: the bus is closed and on_close stops the Rerun bridge."""

    def _run_until_signaled(self, signum: int) -> None:
        from remote_control.base_velocity_controller_gui import OperatorGui

        bus = RecordingBus()
        app = SignaledApp(signum)
        closed: List[bool] = []
        gui = OperatorGui(app, OperatorBus(bus), on_close=lambda: closed.append(True))
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
        from remote_control.base_velocity_controller_gui import build_parser

        args = build_parser().parse_args([])
        self.assertEqual(args.network_config, "config/ipc/network.textproto")
        self.assertEqual(args.ipc_node, "operator")
        self.assertEqual(args.task_file, "")
        self.assertIsNone(args.default_pelvis_height)
        self.assertEqual(args.urdf_file, "")
        args = build_parser().parse_args(
            [
                "--task_file=t.yaml",
                "--reference_file=r.yaml",
                "--pd_gains_file=p.yaml",
                "--default_pelvis_height=0.7",
            ]
        )
        self.assertEqual(
            (args.task_file, args.reference_file, args.pd_gains_file),
            ("t.yaml", "r.yaml", "p.yaml"),
        )
        self.assertAlmostEqual(args.default_pelvis_height, 0.7)

    def test_the_ros_arguments_of_the_old_launch_files_are_refused(self):
        from remote_control.base_velocity_controller_gui import build_parser

        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit):
            build_parser().parse_args(["--ros-args", "-p", "task_file:=t.yaml"])
        self.assertIn("--ros-args", errors.getvalue())


@requires_display
class TestTheWindowOnTheBus(unittest.TestCase):
    """The real App over an operator bus that records."""

    def setUp(self):
        from remote_control.base_velocity_controller_gui import App, OperatorGui

        self.bus = RecordingBus()
        self.operator_bus = OperatorBus(self.bus)
        # Online tuning off: the tabs only read the shipped files.
        self.app = App(
            pd_gains_file=os.path.join(ATLAS_CONFIG, "controller/joint_pd_gains.yaml"),
            task_file=os.path.join(ATLAS_CONFIG, "mpc/task.yaml"),
            reference_file=os.path.join(ATLAS_CONFIG, "command/reference.yaml"),
            enable_online_tuning=False,
            param_publisher=self.operator_bus.mpc_parameters,
            pd_gains_publisher=self.operator_bus.pd_gains,
            joint_targets_publisher=self.operator_bus.joint_targets,
            dodgeball_publisher=self.operator_bus.dodgeball_throw,
        )
        self.app.withdraw()
        self.addCleanup(self.app.destroy)
        self.gui = OperatorGui(self.app, self.operator_bus)

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
            self.app.mpc_params_tab.param_publisher, self.operator_bus.mpc_parameters
        )
        self.assertIs(self.app.joint_pd_tab.param_publisher, self.operator_bus.pd_gains)
        self.assertIs(
            self.app.joint_targets_tab.param_publisher, self.operator_bus.joint_targets
        )
        self.assertIs(
            self.app.dodgeball_tab.throw_publisher, self.operator_bus.dodgeball_throw
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


if __name__ == "__main__":
    unittest.main()
