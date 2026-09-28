"""
Tests of the `/humanoid/fsm_state` message and of when the Base Controller tab re-centers its joysticks.

The regression these pin: the sticks were released only when the gantry became locked. A controller reset the FSM
published while the gantry was ALREADY locked - the simulator putting the robot back in its initial state in JOINT_PD -
changed neither the mode nor the gantry, so a stick left forward stayed forward and walked the robot as soon as WB_MPC
was entered again. The simulator now counts its controller resets in the message (SimFsmBridge::publishControllerReset),
and every transition into a passive mode, every new gantry lock and every change of that count releases the sticks.

The logic tests are headless (remote_control/fsm_state.py has no Tk and no ROS). The App test at the end needs the ROS 2
Python packages the GUI module imports and a display (run under xvfb-run).
"""

import os
import unittest

from remote_control.fsm_state import FsmState, parse_fsm_state, should_recenter
from remote_control.humanoid_finite_state_machine import ControlMode, PASSIVE_MODES


class TestParseFsmState(unittest.TestCase):
    def test_the_three_fields_simfsmbridge_writes(self):
        # The exact strings formatFsmState() writes (testFsmStateMessage.cpp pins the C++ side of the same strings).
        self.assertEqual(
            parse_fsm_state("JOINT_PD,GANTRY_LOCKED,3"),
            FsmState(mode="JOINT_PD", gantry_locked=True, controller_resets=3),
        )
        self.assertEqual(
            parse_fsm_state("WB_MPC,GANTRY_UNLOCKED,0"),
            FsmState(mode="WB_MPC", gantry_locked=False, controller_resets=0),
        )

    def test_an_older_message_without_the_count_still_carries_mode_and_gantry(self):
        self.assertEqual(
            parse_fsm_state("SAFETY,GANTRY_LOCKED"),
            FsmState(mode="SAFETY", gantry_locked=True, controller_resets=None),
        )
        self.assertEqual(parse_fsm_state("GRAVITY_COMP"), FsmState("GRAVITY_COMP"))

    def test_a_malformed_field_is_absent_and_an_unknown_mode_is_no_state(self):
        self.assertEqual(
            parse_fsm_state("JOINT_PD,SOMEWHERE,-1"),
            FsmState(mode="JOINT_PD", gantry_locked=None, controller_resets=None),
        )
        self.assertIsNone(parse_fsm_state("LOCK_GANTRY,GANTRY_LOCKED,1"))
        self.assertIsNone(parse_fsm_state("wb_mpc,GANTRY_LOCKED,1"))
        self.assertIsNone(parse_fsm_state(""))

    def test_a_count_int_cannot_read_is_absent_rather_than_an_error(self):
        # "²" passes str.isdigit() but not int(): the message must still carry its mode and gantry, not raise.
        for count in ("²", "3.0", "+3", "0x3"):
            with self.subTest(count=count):
                self.assertEqual(
                    parse_fsm_state(f"JOINT_PD,GANTRY_LOCKED,{count}"),
                    FsmState(
                        mode="JOINT_PD", gantry_locked=True, controller_resets=None
                    ),
                )
        # Positive control: the digits formatFsmState() writes are read.
        self.assertEqual(
            parse_fsm_state("JOINT_PD,GANTRY_LOCKED, 12 ").controller_resets, 12
        )

    def test_every_mode_but_the_mpc_is_passive(self):
        # The family of control_mode::isPassive in ControlMode.h: every mode whose action is computed without the MPC.
        self.assertEqual(
            PASSIVE_MODES,
            {mode.value for mode in ControlMode if mode is not ControlMode.WB_MPC},
        )


class TestShouldRecenter(unittest.TestCase):
    def test_a_controller_reset_while_locked_in_joint_pd_releases_the_sticks(self):
        # The case the gantry and the mode cannot show: both are unchanged, only the count moved.
        before = parse_fsm_state("JOINT_PD,GANTRY_LOCKED,1")
        after = parse_fsm_state("JOINT_PD,GANTRY_LOCKED,2")
        self.assertTrue(should_recenter(before, after))
        # Positive control: the same state published again releases nothing.
        self.assertFalse(
            should_recenter(after, parse_fsm_state("JOINT_PD,GANTRY_LOCKED,2"))
        )

    def test_every_transition_into_a_passive_mode_releases_the_sticks(self):
        for passive in sorted(PASSIVE_MODES):
            with self.subTest(mode=passive):
                self.assertTrue(
                    should_recenter(
                        FsmState("WB_MPC", False, 0), FsmState(passive, False, 0)
                    )
                )
        # Between two passive modes too: the robot is still not walked by the MPC, and a new mode is a new start.
        self.assertTrue(
            should_recenter(FsmState("JOINT_PD", True, 0), FsmState("SAFETY", True, 0))
        )

    def test_entering_the_mpc_or_unlocking_keeps_the_sticks(self):
        self.assertFalse(
            should_recenter(FsmState("JOINT_PD", True, 4), FsmState("WB_MPC", True, 4))
        )
        self.assertFalse(
            should_recenter(FsmState("WB_MPC", True, 4), FsmState("WB_MPC", False, 4))
        )
        self.assertFalse(
            should_recenter(FsmState("WB_MPC", False, 4), FsmState("WB_MPC", False, 4))
        )

    def test_a_new_gantry_lock_releases_the_sticks(self):
        self.assertTrue(
            should_recenter(FsmState("WB_MPC", False, 0), FsmState("WB_MPC", True, 0))
        )

    def test_the_first_message_is_a_transition_from_an_unknown_state(self):
        self.assertTrue(should_recenter(None, FsmState("ZERO_TORQUE", True, 0)))
        self.assertTrue(should_recenter(None, FsmState("WB_MPC", True, 7)))
        # A walking robot's state is not a reason to release, and its count is only the baseline.
        self.assertFalse(should_recenter(None, FsmState("WB_MPC", False, 7)))
        self.assertTrue(
            should_recenter(FsmState("WB_MPC", False, 7), FsmState("WB_MPC", False, 8))
        )

    def test_a_publisher_without_the_count_never_releases_on_it(self):
        self.assertFalse(
            should_recenter(FsmState("WB_MPC", False), FsmState("WB_MPC", False, 3))
        )
        self.assertFalse(
            should_recenter(FsmState("WB_MPC", False, 3), FsmState("WB_MPC", False))
        )


def _gui_available():
    try:
        import tkinter as tk

        import rclpy  # noqa: F401 - the GUI module imports it
        from humanoid_mpc_msgs.msg import WalkingVelocityCommand  # noqa: F401

        root = tk.Tk()
        root.withdraw()
        root.destroy()
        return True
    except (
        Exception
    ):  # noqa: BLE001 - any failure here means no usable display or no ROS 2 Python packages
        return False


class MockPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


@unittest.skipUnless(
    _gui_available(), "no usable Tk display or no ROS 2 Python packages"
)
class TestAppFollowsTheFsmState(unittest.TestCase):
    """App.update_fsm_state() against a withdrawn window: the sticks the operator left forward are released."""

    def setUp(self):
        from remote_control.base_velocity_controller_gui import App

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        config = os.path.join(
            repo_root, "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config"
        )
        # The tuning tabs only read these files; nothing here saves them.
        self.app = App(
            pd_gains_file=os.path.join(config, "controller/joint_pd_gains.yaml"),
            task_file=os.path.join(config, "mpc/task.yaml"),
            reference_file=os.path.join(config, "command/reference.yaml"),
            enable_online_tuning=False,
            enable_telemetry=False,
            param_publisher=MockPublisher(),
            pd_gains_publisher=MockPublisher(),
            joint_targets_publisher=MockPublisher(),
        )
        self.app.withdraw()

    def tearDown(self):
        self.app.destroy()

    def _push_sticks_forward(self):
        self.app.joystick_left.set_position(0.8, 0.0)
        self.app.joystick_right.set_position(0.0, 0.5)

    def _sticks_are_forward(self):
        return (
            self.app.joystick_left.x_norm != 0.0
            and self.app.joystick_right.y_norm != 0.0
        )

    def _sticks_are_centered(self):
        return (
            self.app.joystick_left.x_norm == 0.0
            and self.app.joystick_left.y_norm == 0.0
            and self.app.joystick_right.x_norm == 0.0
            and self.app.joystick_right.y_norm == 0.0
        )

    def test_a_reset_while_locked_in_joint_pd_releases_the_sticks(self):
        self.app.update_fsm_state("JOINT_PD,GANTRY_LOCKED,1")
        self._push_sticks_forward()
        # Positive control: the same state again leaves the operator's sticks alone.
        self.app.update_fsm_state("JOINT_PD,GANTRY_LOCKED,1")
        self.assertTrue(self._sticks_are_forward())
        self.app.update_fsm_state("JOINT_PD,GANTRY_LOCKED,2")
        self.assertTrue(self._sticks_are_centered())
        self.assertEqual(self.app.fsm_mode_var.get(), "JOINT_PD")
        self.assertTrue(self.app.gantry_var.get())

    def test_the_mpc_handing_over_to_a_passive_mode_releases_the_sticks(self):
        self.app.update_fsm_state("WB_MPC,GANTRY_UNLOCKED,0")
        self._push_sticks_forward()
        self.app.update_fsm_state("WB_MPC,GANTRY_UNLOCKED,0")
        self.assertTrue(
            self._sticks_are_forward(), "walking: the sticks are the operator's"
        )
        self.app.update_fsm_state("SAFETY,GANTRY_UNLOCKED,0")
        self.assertTrue(self._sticks_are_centered())
        self.assertEqual(self.app.fsm_mode_var.get(), "SAFETY")
        self.assertFalse(self.app.gantry_var.get())

    def test_a_message_the_gui_cannot_read_changes_nothing(self):
        self.app.update_fsm_state("WB_MPC,GANTRY_UNLOCKED,0")
        self._push_sticks_forward()
        self.app.update_fsm_state("NOT_A_MODE,GANTRY_LOCKED,9")
        self.assertTrue(self._sticks_are_forward())
        self.assertEqual(self.app.fsm_mode_var.get(), "WB_MPC")
        self.assertFalse(self.app.gantry_var.get())


if __name__ == "__main__":
    unittest.main()
