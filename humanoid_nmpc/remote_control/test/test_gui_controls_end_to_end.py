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

"""The operator GUI's controls against the robot process of the simulation, end to end.

The laptop side is the GUI itself: base_velocity_controller_gui.App, ticked by OperatorGui, on an
operator_bus.OperatorBus, its widgets driven as the operator drives them (the FSM drop-down, the gantry checkbox, the
Touch Ground button, the height slider, the joint target sliders, the Kp scale buttons, the Base Controller tab's
contact estimator drop-down, the Save button, the Dodgeball tab's Throw button). The robot side is
humanoid_centroidal_mpc_robot on the DRC Atlas files with the MuJoCo backend and no viewer, as
docker-compose.robot.sim.yaml runs it. Both share the shipped network file's topics and QoS profiles
(config/ipc/network.textproto) on a DDS domain of this test's own (robot_ipc.isolated_test_network()).

Each control is held to its effect on the robot, as the robot reports it - robot/fsm_state, and robot/state, which
carries the control mode the controller runs, the joint action it applies and the base pose - never to the message the
GUI sent. No MPC node runs: WB_MPC is entered and held on the JOINT_PD action until a policy arrives, which none does;
what the MPC node does with the walking command and the parameters is the MPC node's tests'.

A Tk callback that raises fails the test (report_callback_exception), so that nothing the operator clicks can fail
silently.
"""

from collections.abc import Callable
import dataclasses
import math
import os
import shutil
import tempfile
import threading
import time
import unittest

from humanoid_mpc_msgs import robot_state_sample_pb2

from humanoid_mpc_ipc import topics
import operator_test_support
from remote_control import base_velocity_controller_gui
from remote_control import config_files
from remote_control import fsm_state
from remote_control import operator_bus
from remote_control import robot_config_save
from remote_control.tk_app import joint_pd_tab
import robot_ipc

ROBOT_BINARY = os.environ["ROBOT_BINARY"]
SHIPPED_NETWORK_FILE = os.path.join("config", "ipc", "network.textproto")
# The nodes of the shipped network file, which its topics name as publishers.
NETWORK_NODES = ("robot", "mpc", "operator", "teleop", "config_push")
STORED_TASK = os.path.join("mpc", "task.textproto")
# How long the robot may take to start, and how long one control may take to show its effect [s].
START_TIMEOUT_S = 60.0
EFFECT_TIMEOUT_S = 10.0
# The GUI's tick period [s] (OperatorGui at teleop.WALKING_COMMAND_RATE_HZ).
TICK_PERIOD_S = 0.04
# [m] How far the height slider moves the gantry, and how close the base must follow it.
HEIGHT_STEP_M = 0.1
HEIGHT_TOLERANCE_M = 0.03
# [s] How long Touch Ground takes to bring the slider down (App._on_gantry_touch()), with a margin.
TOUCH_GROUND_S = 2.5
# A joint of each kind on the Atlas: one the MPC controls, whose target the Joint Targets tab moves [rad].
TARGET_JOINT = "l_arm_shz"
TARGET_OFFSET_RAD = 0.3
TARGET_TOLERANCE_RAD = 1e-6
# The Kp scale button the test clicks, and the relative tolerance of the scaled gain the robot applies.
KP_FACTOR = 1.5
GAIN_RELATIVE_TOLERANCE = 1e-6
# [m/s] How much faster the base must move after the ball hits than it did standing before the throw.
DODGEBALL_SPEED_JUMP_MPS = 0.03
# [s] How long the base is watched before and after a throw.
DODGEBALL_WINDOW_S = 1.5
# The contact estimator the Base Controller tab's drop-down switches to, away from the Atlas's cheater_sim.
NEW_CONTACT_ESTIMATOR = "always_in_contact"


def write_network_file(path: str) -> None:
    """Writes the shipped network file moved to this test's own DDS domain: its topics, QoS and transport, kept.

    Args:
        path: The file to write.
    """
    shipped = robot_ipc.load_network_config(SHIPPED_NETWORK_FILE)
    isolated = robot_ipc.isolated_test_network(list(NETWORK_NODES))
    network = dataclasses.replace(
        shipped,
        nodes=isolated.nodes,
        domain_id=isolated.domain_id,
        domain_tag=isolated.domain_tag,
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(robot_ipc.format_network_config(network))


class _RobotStateRecorder:
    """The robot/state samples the bus's receive thread delivers, for the test's thread. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: robot_state_sample_pb2.RobotStateSample | None = None
        # (receive time [s], base speed [m/s]) of every sample received.
        self._speeds: list[tuple[float, float]] = []

    def put(self, sample: robot_state_sample_pb2.RobotStateSample) -> None:
        velocity = sample.base_linear_velocity_local
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        with self._lock:
            self._latest = sample
            self._speeds.append((time.monotonic(), speed))

    def latest(self) -> robot_state_sample_pb2.RobotStateSample | None:
        with self._lock:
            return self._latest

    def max_speed(self, since: float, until: float) -> float:
        """The fastest base speed received in [since, until] (monotonic times) [m/s]; 0 without a sample."""
        with self._lock:
            return max(
                (speed for at, speed in self._speeds if since <= at <= until),
                default=0.0,
            )


def joint_value(
    sample: robot_state_sample_pb2.RobotStateSample, values: list[float], joint: str
) -> float:
    """The entry of `values` (an array of `sample` aligned with joint_names) of `joint`; NaN without one."""
    names = list(sample.joint_names)
    if joint not in names or len(values) != len(names):
        return math.nan
    return float(values[names.index(joint)])


@operator_test_support.requires_display
class GuiControlsEndToEndTest(unittest.TestCase):
    """Each test starts the robot process and the GUI on the bus of a network file of its own, and ends both."""

    def setUp(self):
        self.scratch = tempfile.mkdtemp(dir=os.environ.get("TEST_TMPDIR"))
        self.addCleanup(shutil.rmtree, self.scratch, ignore_errors=True)
        logs = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", self.scratch)
        network_file = os.path.join(self.scratch, "network.textproto")
        write_network_file(network_file)
        self.store = os.path.join(self.scratch, "robot_config", "drc_atlas")
        # The laptop's copy, below a robot_models/ directory so that its configuration identity is the robot's.
        config = operator_test_support.copy_config(
            os.path.join(operator_test_support.ATLAS_PACKAGE, "config"),
            os.path.join(self.scratch, "laptop", operator_test_support.ATLAS_PACKAGE),
        )
        self.task_file = os.path.join(config, "mpc", "task.textproto")
        reference_file = os.path.join(config, "command", "reference.textproto")

        self.robot = operator_test_support.RobotProcess(
            ROBOT_BINARY, network_file, self.store, os.path.join(logs, "robot.log")
        )
        self.addCleanup(self.robot.stop)

        self.bus = operator_bus.OperatorBus.connect(network_file)
        self.states = _RobotStateRecorder()
        self.bus.bus.subscribe(
            topics.ROBOT_STATE,
            robot_state_sample_pb2.RobotStateSample,
            self.states.put,
            delivery=robot_ipc.Delivery.ALL,
        )
        self.tk_errors: list[str] = []
        self.app = base_velocity_controller_gui.App(
            pd_gains_file=os.path.join(
                config, "controller", "joint_pd_gains.textproto"
            ),
            task_file=self.task_file,
            reference_file=reference_file,
            enable_online_tuning=True,
            param_publisher=self.bus.mpc_parameters,
            pd_gains_publisher=self.bus.pd_gains,
            joint_targets_publisher=self.bus.joint_targets,
            dodgeball_publisher=self.bus.dodgeball_throw,
            robot_saver=robot_config_save.RobotConfigSaver(
                self.bus.config_save, self.bus.config_save_statuses
            ),
        )
        self.app.withdraw()
        self.app.report_callback_exception = self._record_tk_error
        self.addCleanup(self.app.destroy)
        # As the GUI's main() does.
        self.app.set_default_pelvis_height(
            config_files.read_default_pelvis_height(reference_file)
        )
        self.gui = base_velocity_controller_gui.OperatorGui(self.app, self.bus)
        self.bus.start()
        self.addCleanup(self.bus.close)
        self._next_tick = 0.0
        self._wait_for_robot()

    def tearDown(self):
        self.assertEqual(self.tk_errors, [], "a GUI callback raised")

    # ── Driving the GUI ─────────────────────────────────────────────────────────────────────────────────────────

    def _record_tk_error(self, exc: type, value: BaseException, tb: object) -> None:
        del exc, tb  # Unused: the message says what failed.
        self.tk_errors.append(repr(value))

    def _pump(self) -> None:
        """Runs Tk's pending events, and the GUI's tick when its period is up, as OperatorGui.run() does."""
        self.app.update()
        now = time.monotonic()
        if now >= self._next_tick:
            self._next_tick = now + TICK_PERIOD_S
            self.gui.tick()
        self.assertEqual(self.tk_errors, [], "a GUI callback raised")

    def _pump_until(
        self, condition: Callable[[], bool], timeout: float = EFFECT_TIMEOUT_S
    ) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._pump()
            if condition():
                return True
            time.sleep(0.01)
        return condition()

    def _pump_for(self, duration: float) -> None:
        self._pump_until(lambda: False, duration)

    def _wait_for_robot(self) -> None:
        """Waits until the robot's FSM state and telemetry reach the GUI."""

        def reported() -> bool:
            return (
                self._robot_fsm_state() is not None and self.states.latest() is not None
            ) or not self.robot.running()

        started = self._pump_until(reported, START_TIMEOUT_S)
        self.assertTrue(self.robot.running(), "the robot process exited on start-up")
        self.assertTrue(started, "the robot's FSM state or telemetry did not arrive")

    def _robot_fsm_state(self) -> fsm_state.FsmState | None:
        """The last robot/fsm_state the GUI followed, as App.update_fsm_state() kept it."""
        # pylint: disable-next=protected-access  # The state the App followed, which no widget shows whole.
        return self.app._last_fsm_state

    def _robot_state(self) -> fsm_state.FsmState:
        """_robot_fsm_state(), which the robot has published since setUp() waited for it."""
        state = self._robot_fsm_state()
        assert state is not None
        return state

    def _select_mode(self, mode: str) -> None:
        """Picks `mode` in the FSM drop-down, as a click on it does."""
        sent = self.bus.fsm_command.last_sequence
        self.app.fsm_mode_var.set(mode)
        self.app.fsm_dropdown.event_generate("<<ComboboxSelected>>")
        self.assertGreater(
            self.bus.fsm_command.last_sequence, sent, f"selecting {mode} sent nothing"
        )

    def _controller_mode(self) -> str:
        sample = self.states.latest()
        return sample.control_mode if sample is not None else ""

    def _assert_mode(self, mode: str) -> None:
        """The robot runs `mode`, its FSM state says so, and the GUI's drop-down shows it."""

        def entered() -> bool:
            return self._controller_mode() == mode and self._robot_state().mode == mode

        self.assertTrue(
            self._pump_until(entered),
            f"the robot did not enter {mode}: it runs '{self._controller_mode()}', reports "
            f"{self._robot_fsm_state()}",
        )
        self.assertEqual(self.app.fsm_mode_var.get(), mode)

    def _assert_gantry(self, locked: bool) -> None:
        self.assertTrue(
            self._pump_until(lambda: self._robot_state().gantry_locked == locked),
            f"the robot's gantry is not {'locked' if locked else 'released'}: {self._robot_fsm_state()}",
        )
        self.assertEqual(self.app.gantry_var.get(), locked)

    def _base_height(self) -> float:
        sample = self.states.latest()
        assert sample is not None
        return sample.base_position_world.z

    def _set_slider_height(self, height: float) -> None:
        """Moves the height slider to command the pelvis height `height` [m]."""
        self.app.slider.set((height - self.app.min_height) / self.app.height_scale)

    def _settled_base_height(self, duration: float = 1.5) -> float:
        self._pump_for(duration)
        return self._base_height()

    def _touch_ground(self) -> float:
        """Clicks Touch Ground and waits out its descent; the base height after it [m]."""
        self.app.gantry_touch_btn.invoke()
        # The descent moves the slider for TOUCH_GROUND_S, also when it starts at the default already.
        self._pump_for(TOUCH_GROUND_S)
        self.assertAlmostEqual(
            self.app.slider_height(), self.app.default_pelvis_height, places=6
        )
        return self._settled_base_height()

    def _joint_sample(self, field: str, joint: str) -> float:
        sample = self.states.latest()
        if sample is None:
            return math.nan
        return joint_value(sample, list(getattr(sample, field)), joint)

    # ── The tests ───────────────────────────────────────────────────────────────────────────────────────────────

    def test_the_fsm_cycle_and_the_gantry(self):
        # The robot spawns limp on the locked gantry, and the GUI shows it.
        self._assert_mode("ZERO_TORQUE")
        self._assert_gantry(locked=True)

        for mode in ("JOINT_PD", "GRAVITY_COMP", "SAFETY", "JOINT_PD"):
            with self.subTest(mode=mode):
                self._select_mode(mode)
                self._assert_mode(mode)

        # While locked, the gantry follows the height slider: the walking command's pelvis height reaches the robot.
        standing = self._touch_ground()
        self._set_slider_height(self.app.default_pelvis_height + HEIGHT_STEP_M)
        lifted = standing + HEIGHT_STEP_M
        self.assertTrue(
            self._pump_until(
                lambda: abs(self._base_height() - lifted) < HEIGHT_TOLERANCE_M
            ),
            f"the locked gantry did not lift the base from {standing} m: it is at {self._base_height()} m",
        )
        standing = self._touch_ground()

        # WB_MPC with the gantry locked, then the gantry released: the base no longer follows the slider.
        self._select_mode("WB_MPC")
        self._assert_mode("WB_MPC")
        # The robot's log says so, also for a command that switches no torque.
        logged = "FSM command received: WB_MPC."
        self.assertTrue(
            self._pump_until(lambda: logged in self.robot.log_text()),
            "the robot did not log the WB_MPC command",
        )
        self.app.gantry_toggle.invoke()
        self._assert_gantry(locked=False)
        resets = self._robot_state().controller_resets
        self._set_slider_height(self.app.default_pelvis_height + HEIGHT_STEP_M)
        released = self._settled_base_height()
        self.assertLess(
            released,
            standing + HEIGHT_STEP_M - HEIGHT_TOLERANCE_M,
            "the released gantry still lifts the base",
        )
        self._assert_mode("WB_MPC")

        # And back: a stick left deflected, the gantry locked again, which puts the robot in JOINT_PD, counts a
        # controller reset and re-centers the sticks; then zero torque.
        self.app.joystick_left.set_position(0.5, 0.0)
        self.app.gantry_toggle.invoke()
        self._assert_gantry(locked=True)
        self._assert_mode("JOINT_PD")
        self.assertGreater(self._robot_state().controller_resets, resets)
        self.assertEqual(self.app.joystick_left.x_norm, 0.0)
        self._select_mode("ZERO_TORQUE")
        self._assert_mode("ZERO_TORQUE")

    def test_joint_targets_and_pd_gains_reach_the_controller(self):
        self._select_mode("JOINT_PD")
        self._assert_mode("JOINT_PD")
        tab = self.app.joint_targets_tab
        target = tab.default_positions.get(TARGET_JOINT, 0.0) + TARGET_OFFSET_RAD
        tab.slider_rows[TARGET_JOINT].set_value(target)

        def targeted() -> bool:
            applied = self._joint_sample("joint_position_targets", TARGET_JOINT)
            return abs(applied - target) < TARGET_TOLERANCE_RAD

        self.assertTrue(
            self._pump_until(targeted),
            f"the JOINT_PD target of {TARGET_JOINT} is "
            f"{self._joint_sample('joint_position_targets', TARGET_JOINT)}, not {target}",
        )

        kp = self._joint_sample("joint_kp", TARGET_JOINT)
        self.assertGreater(kp, 0.0)
        pd_tab = self.app.joint_pd_tab
        button = pd_tab.scale_buttons[joint_pd_tab.SCALE_FACTORS.index(KP_FACTOR)]
        self.assertEqual(button.cget("text"), f"{KP_FACTOR}x")
        button.invoke()

        def scaled() -> bool:
            applied = self._joint_sample("joint_kp", TARGET_JOINT)
            return math.isclose(
                applied, KP_FACTOR * kp, rel_tol=GAIN_RELATIVE_TOLERANCE
            )

        self.assertTrue(
            self._pump_until(scaled),
            f"the Kp of {TARGET_JOINT} is {self._joint_sample('joint_kp', TARGET_JOINT)}, not "
            f"{KP_FACTOR} x {kp}",
        )

    def test_the_contact_estimator_choice_applies_and_save_stores_the_file(self):
        choice = self.app.schema_choices.get("contact_estimator")
        self.assertIsNotNone(
            choice, "the Base Controller tab offers no contact estimator"
        )
        assert choice is not None
        self.assertEqual(str(choice.combobox.cget("state")), "readonly")
        self.assertIn(NEW_CONTACT_ESTIMATOR, choice.combobox.cget("values"))
        choice.var.set(NEW_CONTACT_ESTIMATOR)
        choice.combobox.event_generate("<<ComboboxSelected>>")
        swapped = (
            "The measured contact state now comes from the contact estimator "
            f"'{NEW_CONTACT_ESTIMATOR}'"
        )
        self.assertTrue(
            self._pump_until(lambda: swapped in self.robot.log_text()),
            "the robot did not swap its contact estimator",
        )

        # Save writes the laptop's file and stores the same bytes on the robot, which says so.
        tab = self.app.mpc_params_tab
        tab.save_btn.invoke()

        def answered() -> bool:
            outcome = tab.robot_save.outcome()
            return outcome is not None and not outcome.waiting

        self.assertTrue(
            self._pump_until(answered),
            f"the robot did not answer the save: {tab.robot_save.outcome()}",
        )
        outcome = tab.robot_save.outcome()
        assert outcome is not None
        self.assertEqual(outcome.state, robot_config_save.SaveState.SAVED, outcome)
        stored = os.path.join(self.store, STORED_TASK)
        with open(stored, "rb") as robot_copy, open(self.task_file, "rb") as laptop:
            stored_bytes = robot_copy.read()
            self.assertEqual(stored_bytes, laptop.read())
        self.assertIn(
            f'contact_estimator: "{NEW_CONTACT_ESTIMATOR}"', stored_bytes.decode()
        )

    def test_a_dodgeball_hits_the_released_robot(self):
        self._select_mode("JOINT_PD")
        self._assert_mode("JOINT_PD")
        self._touch_ground()
        self.app.gantry_toggle.invoke()
        self._assert_gantry(locked=False)
        self._pump_for(DODGEBALL_WINDOW_S)
        before = time.monotonic()
        self._pump_for(DODGEBALL_WINDOW_S)
        thrown = time.monotonic()
        standing_speed = self.states.max_speed(before, thrown)
        self.app.dodgeball_tab.throw_button.invoke()
        self._pump_for(DODGEBALL_WINDOW_S)
        hit_speed = self.states.max_speed(thrown, time.monotonic())
        self.assertGreater(
            hit_speed,
            standing_speed + DODGEBALL_SPEED_JUMP_MPS,
            f"the base moved at most {hit_speed} m/s after the throw, {standing_speed} m/s before it",
        )


if __name__ == "__main__":
    unittest.main()
