"""****************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
****************************************************************************"""

import unittest
import numpy as np
from humanoid_nmpc.remote_control.remote_control.humanoid_finite_state_machine import (
    ControlMode,
    HumanoidFSM,
    VirtualGantry,
    CYCLE_MODES,
    load_robot_config,
    get_available_robots,
)


class TestHumanoidFSM(unittest.TestCase):
    def setUp(self):
        self.fsm_atlas = HumanoidFSM(robot_name="atlas")
        self.fsm_g1 = HumanoidFSM(robot_name="g1")

    def test_dynamic_config_loading_from_yaml_and_urdf(self):
        """Verifies that robot topology, active joint lists, and nominal posture are parsed directly from YAML and URDF/Pinocchio."""
        atlas_cfg = load_robot_config("atlas")
        self.assertIn("urdf_path", atlas_cfg)
        self.assertIsNotNone(atlas_cfg["urdf_path"])
        self.assertIn("task_path", atlas_cfg)
        self.assertIsNotNone(atlas_cfg["task_path"])
        self.assertEqual(len(atlas_cfg["nominal_q"]), len(atlas_cfg["joint_names"]))
        self.assertAlmostEqual(atlas_cfg["nominal_pelvis_height_bent"], 0.70)
        self.assertIn("r_leg_kny", atlas_cfg["joint_names"])
        self.assertIn("l_leg_kny", atlas_cfg["joint_names"])

        g1_cfg = load_robot_config("g1")
        self.assertIsNotNone(g1_cfg["urdf_path"])
        self.assertIsNotNone(g1_cfg["task_path"])
        self.assertEqual(len(g1_cfg["nominal_q"]), len(g1_cfg["joint_names"]))
        self.assertAlmostEqual(g1_cfg["nominal_pelvis_height_bent"], 0.7925)
        self.assertIn("left_knee_joint", g1_cfg["joint_names"])

        # Available robots detection
        available = get_available_robots()
        self.assertTrue(any("atlas" in r for r in available))
        self.assertTrue(any("g1" in r for r in available))

    def test_initial_state(self):
        """Verifies default initial mode and gantry suspension state."""
        self.assertEqual(self.fsm_atlas.current_mode, ControlMode.ZERO_TORQUE)
        self.assertTrue(self.fsm_atlas.gantry.is_locked)
        self.assertAlmostEqual(
            self.fsm_atlas.gantry.height,
            self.fsm_atlas.cfg["nominal_pelvis_height_bent"],
        )

    def test_mode_cycling(self):
        """Verifies cyclic progression through ZERO_TORQUE -> JOINT_PD -> GRAVITY_COMP -> WB_MPC -> ZERO_TORQUE."""
        fsm = self.fsm_atlas
        self.assertEqual(fsm.current_mode, ControlMode.ZERO_TORQUE)

        # Forward cycle
        self.assertEqual(fsm.cycle_next_mode(), ControlMode.JOINT_PD)
        self.assertEqual(fsm.cycle_next_mode(), ControlMode.GRAVITY_COMP)
        self.assertEqual(fsm.cycle_next_mode(), ControlMode.WB_MPC)
        self.assertEqual(fsm.cycle_next_mode(), ControlMode.ZERO_TORQUE)

        # Backward cycle
        self.assertEqual(fsm.cycle_prev_mode(), ControlMode.WB_MPC)
        self.assertEqual(fsm.cycle_prev_mode(), ControlMode.GRAVITY_COMP)
        self.assertEqual(fsm.cycle_prev_mode(), ControlMode.JOINT_PD)
        self.assertEqual(fsm.cycle_prev_mode(), ControlMode.ZERO_TORQUE)

    def test_virtual_gantry_stepping_and_ground_touch(self):
        """Verifies +1 cm / -1 cm stepping, clamping, and ground-touch auto-calibration."""
        gantry = VirtualGantry(robot_name="atlas", initial_height=0.70)

        # Step up +1 cm (+0.01 m)
        new_h = gantry.step_up()
        self.assertAlmostEqual(new_h, 0.71)
        self.assertAlmostEqual(gantry.height, 0.71)

        # Step down -1 cm (-0.01 m)
        new_h = gantry.step_down()
        self.assertAlmostEqual(new_h, 0.70)

        # Toggle lock
        self.assertTrue(gantry.is_locked)
        gantry.release()
        self.assertFalse(gantry.is_locked)
        gantry.lock()
        self.assertTrue(gantry.is_locked)

        # Auto-calibrate ground touch
        cal_h = gantry.auto_calibrate_ground_touch(foot_clearance=0.005)
        self.assertAlmostEqual(cal_h, 0.705)
        self.assertTrue(gantry.is_locked)

    def test_zero_torque_computation(self):
        """Verifies ZERO_TORQUE outputs all zeros."""
        fsm = self.fsm_atlas
        fsm.set_mode(ControlMode.ZERO_TORQUE)
        q = np.random.randn(fsm.num_actuators)
        v = np.random.randn(fsm.num_actuators)
        tau = fsm.compute_torques(q=q, v=v)
        self.assertEqual(len(tau), fsm.num_actuators)
        np.testing.assert_array_equal(tau, np.zeros(fsm.num_actuators))

    def test_joint_pd_torque_computation(self):
        """Verifies JOINT_PD outputs proportional-derivative restoring torques after snap completes."""
        fsm = self.fsm_atlas
        fsm.set_mode(ControlMode.JOINT_PD)
        # Advance time past snap duration for steady-state verification
        t_steady = fsm._joint_pd_start_time + fsm.joint_pd_snap_duration + 0.1

        # At nominal posture with zero velocity, torque is zero
        q_nom = fsm.nominal_q.copy()
        v_zero = np.zeros(fsm.num_actuators)
        tau_at_nom = fsm.compute_torques(q=q_nom, v=v_zero, now=t_steady)
        np.testing.assert_allclose(tau_at_nom, np.zeros(fsm.num_actuators), atol=1e-5)

        # Offset joint by -0.1 rad -> positive restoring torque
        q_offset = q_nom.copy()
        q_offset[0] -= 0.1
        tau = fsm.compute_torques(q=q_offset, v=v_zero, now=t_steady)
        expected_tau_0 = fsm.kp_vector[0] * 0.1
        self.assertAlmostEqual(tau[0], expected_tau_0)

    def test_joint_pd_gradual_snap_transition(self):
        """Verifies that transitioning to JOINT_PD smoothly interpolates from starting posture to nominal posture."""
        fsm = self.fsm_atlas
        fsm.joint_pd_snap_duration = 2.0
        q_start = np.zeros(fsm.num_actuators)
        fsm.set_mode(ControlMode.JOINT_PD, current_q=q_start)

        t0 = fsm._joint_pd_start_time

        # At t = 0 (0% progress), target is q_start, error is 0, velocity is 0 -> torque near 0
        tau_0 = fsm.compute_torques(q=q_start, v=np.zeros(fsm.num_actuators), now=t0)
        np.testing.assert_allclose(tau_0, np.zeros(fsm.num_actuators), atol=1e-5)

        # At t = 1.0s (50% progress), alpha = 0.5, target is halfway
        alpha_half, is_snapping_half, target_q_half = fsm.get_joint_pd_progress(now=t0 + 1.0)
        self.assertTrue(is_snapping_half)
        self.assertAlmostEqual(alpha_half, 0.5, places=4)
        np.testing.assert_allclose(target_q_half, 0.5 * (q_start + fsm.nominal_q), atol=1e-5)

        # At t = 2.0s (100% progress), alpha = 1.0, target is nominal_q
        alpha_end, is_snapping_end, target_q_end = fsm.get_joint_pd_progress(now=t0 + 2.0)
        self.assertFalse(is_snapping_end)
        self.assertAlmostEqual(alpha_end, 1.0)
        np.testing.assert_allclose(target_q_end, fsm.nominal_q, atol=1e-5)

    def test_controller_yaml_gains_loading(self):
        """Verifies that per-joint PD gains YAML files are correctly loaded into kp_vector and kd_vector."""
        atlas_cfg = load_robot_config("atlas")
        self.assertIn("controller_path", atlas_cfg)
        self.assertIsNotNone(atlas_cfg["controller_path"])
        self.assertEqual(len(atlas_cfg["kp_vector"]), len(atlas_cfg["joint_names"]))
        self.assertEqual(len(atlas_cfg["kd_vector"]), len(atlas_cfg["joint_names"]))
        # Atlas knee gain from joint_pd_gains.yaml is 500.0
        kny_idx = atlas_cfg["joint_names"].index("r_leg_kny")
        self.assertEqual(atlas_cfg["kp_vector"][kny_idx], 500.0)
        self.assertEqual(atlas_cfg["kd_vector"][kny_idx], 35.0)

    def test_safety_damped_pd_decay(self):
        """Verifies SAFETY mode decays PD gains smoothly and transitions to ZERO_TORQUE."""
        fsm = self.fsm_atlas
        fsm.safety_decay_duration = 2.0

        q_curr = np.zeros(fsm.num_actuators)
        fsm.trigger_safety(current_q=q_curr)
        self.assertEqual(fsm.current_mode, ControlMode.SAFETY)

        t0 = fsm._safety_start_time

        # Progress at t = 0 -> 100% gain
        frac_0, kp_0, kd_0 = fsm.get_safety_progress(now=t0)
        self.assertAlmostEqual(frac_0, 1.0)
        self.assertAlmostEqual(kp_0, fsm.default_kp)

        # Progress at t = 1.0 s (50%)
        frac_half, kp_half, kd_half = fsm.get_safety_progress(now=t0 + 1.0)
        self.assertAlmostEqual(frac_half, 0.5)
        self.assertAlmostEqual(kp_half, 0.5 * fsm.default_kp)

        # Progress at t = 2.0 s (100% elapsed -> 0% gain)
        frac_end, kp_end, kd_end = fsm.get_safety_progress(now=t0 + 2.0)
        self.assertAlmostEqual(frac_end, 0.0)
        self.assertAlmostEqual(kp_end, 0.0)

        # Compute torques at expiry -> auto-transitions to ZERO_TORQUE
        tau = fsm.compute_torques(
            q=q_curr, v=np.zeros(fsm.num_actuators), now=t0 + 2.05
        )
        self.assertEqual(fsm.current_mode, ControlMode.ZERO_TORQUE)
        np.testing.assert_allclose(tau, np.zeros(fsm.num_actuators))


if __name__ == "__main__":
    unittest.main()
