"""
Tests of the robot/fsm_state message and of when the Base Controller tab re-centers its joysticks.

The regression these pin: the sticks were released only when the gantry became locked. A controller reset the FSM
published while the gantry was ALREADY locked - the simulator putting the robot back in its initial state in JOINT_PD -
changed neither the mode nor the gantry, so a stick left forward stayed forward and walked the robot as soon as WB_MPC
was entered again. The robot now counts its controller resets in the message (FsmState.controller_resets), and every
transition into a passive mode, every new gantry lock and every change of that count releases the sticks.

The logic tests are headless (remote_control/fsm_state.py has no Tk). The App test at the end needs a display.
"""

import os
import unittest

from humanoid_mpc_ipc import topics
from humanoid_mpc_msgs import fsm_state_pb2
from operator_test_support import ATLAS_CONFIG, RecordingPublisher, requires_display
from remote_control.fsm_state import FsmState, from_message, should_recenter
from remote_control.humanoid_finite_state_machine import ControlMode, PASSIVE_MODES


def message(mode, gantry_locked=False, controller_resets=0, mpc_healthy=True):
    return fsm_state_pb2.FsmState(
        mode=mode,
        gantry_locked=gantry_locked,
        controller_resets=controller_resets,
        mpc_healthy=mpc_healthy,
    )


class TestFromMessage(unittest.TestCase):
    def test_the_fields_of_the_message(self):
        self.assertEqual(
            from_message(message("JOINT_PD", gantry_locked=True, controller_resets=3)),
            FsmState(mode="JOINT_PD", gantry_locked=True, controller_resets=3),
        )
        self.assertEqual(
            from_message(message("WB_MPC", gantry_locked=False, controller_resets=0)),
            FsmState(mode="WB_MPC", gantry_locked=False, controller_resets=0),
        )

    def test_every_control_mode_is_read(self):
        for mode in ControlMode:
            with self.subTest(mode=mode.value):
                self.assertEqual(from_message(message(mode.value)).mode, mode.value)

    def test_an_unknown_mode_is_no_state(self):
        # A gantry command is not a mode, the names are case-sensitive, and an empty message carries no mode.
        self.assertIsNone(from_message(message("LOCK_GANTRY", gantry_locked=True)))
        self.assertIsNone(from_message(message("wb_mpc")))
        self.assertIsNone(from_message(fsm_state_pb2.FsmState()))

    def test_the_count_is_read_in_full(self):
        largest = 2**64 - 1
        self.assertEqual(
            from_message(
                message("SAFETY", controller_resets=largest)
            ).controller_resets,
            largest,
        )

    def test_the_message_survives_the_wire(self):
        # What the robot serializes is what the GUI reads.
        sent = message("GRAVITY_COMP", gantry_locked=True, controller_resets=7)
        received = fsm_state_pb2.FsmState.FromString(sent.SerializeToString())
        self.assertEqual(from_message(received), from_message(sent))

    def test_every_mode_but_the_mpc_is_passive(self):
        # The family of control_mode::isPassive in ControlMode.h: every mode whose action is computed without the MPC.
        self.assertEqual(
            PASSIVE_MODES,
            {mode.value for mode in ControlMode if mode is not ControlMode.WB_MPC},
        )


class TestShouldRecenter(unittest.TestCase):
    def test_a_controller_reset_while_locked_in_joint_pd_releases_the_sticks(self):
        # The case the gantry and the mode cannot show: both are unchanged, only the count moved.
        before = from_message(message("JOINT_PD", True, 1))
        after = from_message(message("JOINT_PD", True, 2))
        self.assertTrue(should_recenter(before, after))
        # Positive control: the same state published again releases nothing.
        self.assertFalse(
            should_recenter(after, from_message(message("JOINT_PD", True, 2)))
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


@requires_display
class TestAppFollowsTheFsmState(unittest.TestCase):
    """App.update_fsm_state() against a withdrawn window: the sticks the operator left forward are released."""

    def setUp(self):
        from remote_control.base_velocity_controller_gui import App

        # The tuning tabs only read these files; nothing here saves them.
        self.app = App(
            pd_gains_file=os.path.join(ATLAS_CONFIG, "controller/joint_pd_gains.yaml"),
            task_file=os.path.join(ATLAS_CONFIG, "mpc/task.yaml"),
            reference_file=os.path.join(ATLAS_CONFIG, "command/reference.yaml"),
            enable_online_tuning=False,
            param_publisher=RecordingPublisher(topics.OPERATOR_MPC_PARAMETERS),
            pd_gains_publisher=RecordingPublisher(topics.OPERATOR_PD_GAINS),
            joint_targets_publisher=RecordingPublisher(topics.OPERATOR_JOINT_TARGETS),
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
        self.app.update_fsm_state(message("JOINT_PD", True, 1))
        self._push_sticks_forward()
        # Positive control: the same state again leaves the operator's sticks alone.
        self.app.update_fsm_state(message("JOINT_PD", True, 1))
        self.assertTrue(self._sticks_are_forward())
        self.app.update_fsm_state(message("JOINT_PD", True, 2))
        self.assertTrue(self._sticks_are_centered())
        self.assertEqual(self.app.fsm_mode_var.get(), "JOINT_PD")
        self.assertTrue(self.app.gantry_var.get())

    def test_the_mpc_handing_over_to_a_passive_mode_releases_the_sticks(self):
        self.app.update_fsm_state(message("WB_MPC", False, 0))
        self._push_sticks_forward()
        self.app.update_fsm_state(message("WB_MPC", False, 0))
        self.assertTrue(
            self._sticks_are_forward(), "walking: the sticks are the operator's"
        )
        self.app.update_fsm_state(message("SAFETY", False, 0))
        self.assertTrue(self._sticks_are_centered())
        self.assertEqual(self.app.fsm_mode_var.get(), "SAFETY")
        self.assertFalse(self.app.gantry_var.get())

    def test_a_message_the_gui_cannot_read_changes_nothing(self):
        self.app.update_fsm_state(message("WB_MPC", False, 0))
        self._push_sticks_forward()
        self.app.update_fsm_state(message("NOT_A_MODE", True, 9))
        self.assertTrue(self._sticks_are_forward())
        self.assertEqual(self.app.fsm_mode_var.get(), "WB_MPC")
        self.assertFalse(self.app.gantry_var.get())


if __name__ == "__main__":
    unittest.main()
