"""****************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

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

"""
Exhaustive parameter coverage test for the DRC Atlas MPC online tuning pipeline.

Verifies that every single tunable parameter and DOF in the DRC Atlas task.yaml
is exposed as a slider in the MpcParamsTab GUI and correctly round-trips through
the YAML auto-save mechanism.

Test matrix (DRC Atlas — 36 state DOFs, 36 input DOFs):
  - Q matrix: scaling + 36 diagonal entries (0,0)–(35,35)
  - R matrix: scaling + 36 diagonal entries (0,0)–(35,35)
  - Q_final matrix: scaling + terminalCostScaling + 12 entries (0,0)–(11,11)
  - Task-space foot costs: 18 weight parameters
  - Task-space torso costs: 18 weight parameters
  - ICP cost: icpErrorWeight
  - Foot constraint gains: 9 parameters
  - Swing trajectory config: 8 parameters
  - Friction cone barrier: frictionCoefficient, mu, delta
  - Joint limits barrier: mu, delta
  - Collision constraint barrier: mu, delta
"""

import os
import shutil
import tempfile
import unittest

from remote_control.tk_app.yaml_editor_utils import load_yaml_safe


class TestDrcAtlasParameterCoverage(unittest.TestCase):
    """
    Verify that every single parameter in the DRC Atlas task.yaml is exposed
    as a slider in the MpcParamsTab and can be round-tripped through YAML auto-save.
    """

    NUM_STATE_DOFS = 36  # 6 momentum + 6 base pose + 24 joints
    NUM_INPUT_DOFS = 36  # 12 contact wrenches + 24 joint velocities
    NUM_TERMINAL_DOFS = 12  # 6 momentum + 6 base pose (only first 12 exposed)

    FOOT_COST_WEIGHTS = [
        "pos_x",
        "pos_y",
        "pos_z",
        "orientation_x",
        "orientation_y",
        "orientation_z",
        "lin_velocity_x",
        "lin_velocity_y",
        "lin_velocity_z",
        "ang_velocity_x",
        "ang_velocity_y",
        "ang_velocity_z",
        "lin_acceleration_x",
        "lin_acceleration_y",
        "lin_acceleration_z",
        "ang_acceleration_x",
        "ang_acceleration_y",
        "ang_acceleration_z",
    ]

    TORSO_COST_WEIGHTS = [
        "pos_x",
        "pos_y",
        "pos_z",
        "orientation_x",
        "orientation_y",
        "orientation_z",
        "lin_velocity_x",
        "lin_velocity_y",
        "lin_velocity_z",
        "ang_velocity_x",
        "ang_velocity_y",
        "ang_velocity_z",
        "lin_acceleration_x",
        "lin_acceleration_y",
        "lin_acceleration_z",
        "ang_acceleration_x",
        "ang_acceleration_y",
        "ang_acceleration_z",
    ]

    FOOT_CONSTRAINT_GAINS = [
        "positionErrorGain_z",
        "orientationErrorGain",
        "linearVelocityErrorGain_z",
        "linearVelocityErrorGain_xy",
        "angularVelocityErrorGain",
        "linearAccelerationErrorGain_z",
        "linearAccelerationErrorGain_xy",
        "angularAccelerationErrorGain",
        "softConstraintWeight",
    ]

    SWING_TRAJECTORY_PARAMS = [
        "liftOffVelocity",
        "touchDownVelocity",
        "swingHeight",
        "touchDownHeightOffset",
        "swingTimeScale",
        "impactProximityFactorLiftOffVelocity",
        "impactProximityFactorTouchDownVelocity",
        "impactProximityFactorMidPointValue",
    ]

    FRICTION_CONE_PARAMS = ["frictionCoefficient", "mu", "delta"]
    CONTACT_MOMENT_XY_PARAMS = ["mu", "delta"]
    JOINT_LIMITS_PARAMS = ["mu", "delta"]
    COLLISION_CONSTRAINT_PARAMS = ["mu", "delta"]

    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )
        cls.atlas_task_file = os.path.join(
            cls.repo_root,
            "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml",
        )

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tmp_task_file = os.path.join(self.tmpdir, "task.yaml")
        shutil.copy2(self.atlas_task_file, self.tmp_task_file)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _all_slider_keys(self, root):
        """Every slider key the tab renders, across all of its categories.

        The categories are the configuration's own top-level blocks, so a check that spans blocks - the barriers, the
        two leg torque costs, the solver settings - has to render each of them and take the union.
        """
        tab = self._create_tab(root)
        keys = set()
        for category in tab.categories:
            tab.active_category.set(category)
            tab._render_active_category()
            keys |= set(tab.slider_rows)
        return keys

    def _create_tab(self, root, category=None):
        """Create MpcParamsTab and optionally switch to a category."""
        from remote_control.tk_app.mpc_params_tab import MpcParamsTab

        tab = MpcParamsTab(
            root, task_file=self.tmp_task_file, enable_online_tuning=True
        )
        if category:
            tab.active_category.set(category)
            tab._render_active_category()
        return tab

    def _set(self, tab, key, value):
        """Sets one slider, rendering whichever block holds it first.

        The categories are the configuration's own top-level blocks, so a check that spans blocks has to switch
        between them. The values survive the switch: the tab snapshots the rendered rows into `_live_values` before
        it destroys them, and `save_to_yaml` writes every live value, not only the visible ones.
        """
        if key not in tab.slider_rows:
            self.assertTrue(
                tab.render_category_containing(key),
                f"no slider renders for {key}",
            )
        tab.slider_rows[key].set_value(value)

    # ──────────────────────────────────────────────────────────
    #  Q matrix: scaling + 36 DOFs
    # ──────────────────────────────────────────────────────────
    def test_q_matrix_scaling_slider_exists(self):
        """Q.scaling slider must exist and be tunable."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "Q")
            self.assertIn("Q.scaling", tab.slider_rows, "Q.scaling slider missing")
        finally:
            root.destroy()

    def test_q_matrix_all_36_dofs_have_sliders(self):
        """Every diagonal entry Q(i,i) for i=0..35 must have a slider."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "Q")
            for i in range(self.NUM_STATE_DOFS):
                key = f'Q."({i},{i})"'
                self.assertIn(key, tab.slider_rows, f"Missing slider for {key}")
        finally:
            root.destroy()

    def test_q_matrix_slider_roundtrip_every_dof(self):
        """Modify each Q diagonal entry via slider and verify YAML round-trip."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "Q")

            # Set every Q DOF to a unique test value
            test_values = {}
            for i in range(self.NUM_STATE_DOFS):
                key = f'Q."({i},{i})"'
                if key in tab.slider_rows:
                    new_val = 100.0 + i
                    tab.slider_rows[key].set_value(new_val)
                    test_values[f"({i},{i})"] = new_val

            # Set Q.scaling
            tab.slider_rows["Q.scaling"].set_value(2.5)

            # Force an explicit save
            tab.save_to_yaml()

            # Verify every value in the saved YAML
            data = load_yaml_safe(self.tmp_task_file)
            self.assertAlmostEqual(float(data["Q"]["scaling"]), 2.5, places=2)
            for diag_key, expected in test_values.items():
                actual = float(data["Q"].get(diag_key, -1))
                self.assertAlmostEqual(
                    actual,
                    expected,
                    places=2,
                    msg=f"Q.{diag_key} expected {expected}, got {actual}",
                )
        finally:
            root.destroy()

    # ──────────────────────────────────────────────────────────
    #  R matrix: scaling + 36 DOFs
    # ──────────────────────────────────────────────────────────
    def test_r_matrix_scaling_slider_exists(self):
        """R.scaling slider must exist and be tunable."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "R")
            self.assertIn("R.scaling", tab.slider_rows, "R.scaling slider missing")
        finally:
            root.destroy()

    def test_r_matrix_all_36_dofs_have_sliders(self):
        """Every diagonal entry R(i,i) for i=0..35 must have a slider."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "R")
            for i in range(self.NUM_INPUT_DOFS):
                key = f'R."({i},{i})"'
                self.assertIn(key, tab.slider_rows, f"Missing slider for {key}")
        finally:
            root.destroy()

    def test_r_matrix_slider_roundtrip_every_dof(self):
        """Modify each R diagonal entry via slider and verify YAML round-trip."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "R")

            test_values = {}
            for i in range(self.NUM_INPUT_DOFS):
                key = f'R."({i},{i})"'
                if key in tab.slider_rows:
                    new_val = 0.5 + i * 0.1
                    tab.slider_rows[key].set_value(new_val)
                    test_values[f"({i},{i})"] = new_val

            tab.slider_rows["R.scaling"].set_value(0.05)
            tab.save_to_yaml()

            data = load_yaml_safe(self.tmp_task_file)
            self.assertAlmostEqual(float(data["R"]["scaling"]), 0.05, places=3)
            for diag_key, expected in test_values.items():
                actual = float(data["R"].get(diag_key, -1))
                self.assertAlmostEqual(
                    actual,
                    expected,
                    places=2,
                    msg=f"R.{diag_key} expected {expected}, got {actual}",
                )
        finally:
            root.destroy()

    # ──────────────────────────────────────────────────────────
    #  Q_final: scaling + terminalCostScaling + 12 DOFs
    # ──────────────────────────────────────────────────────────
    def test_terminal_cost_scaling_slider_exists(self):
        """terminalCostScaling slider must exist."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            keys = self._all_slider_keys(root)
            self.assertIn(
                "terminalCostScaling",
                keys,
                "terminalCostScaling slider missing",
            )
        finally:
            root.destroy()

    def test_q_final_scaling_slider_exists(self):
        """Q_final.scaling slider must exist."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "Q_final")
            self.assertIn(
                "Q_final.scaling",
                tab.slider_rows,
                "Q_final.scaling slider missing",
            )
        finally:
            root.destroy()

    def test_q_final_12_dofs_have_sliders(self):
        """Q_final entries (0,0)–(11,11) must have sliders."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "Q_final")
            for i in range(self.NUM_TERMINAL_DOFS):
                key = f'Q_final."({i},{i})"'
                self.assertIn(key, tab.slider_rows, f"Missing slider for {key}")
        finally:
            root.destroy()

    def test_q_final_roundtrip(self):
        """Modify Q_final entries and terminalCostScaling and verify YAML round-trip."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            # terminalCostScaling is a top-level scalar of the file, so it does not live in the Q_final block.
            tab = self._create_tab(root)
            self._set(tab, "terminalCostScaling", 8.0)
            self._set(tab, "Q_final.scaling", 3.0)
            for i in range(self.NUM_TERMINAL_DOFS):
                self._set(tab, f'Q_final."({i},{i})"', 77.0 + i)

            tab.save_to_yaml()

            data = load_yaml_safe(self.tmp_task_file)
            self.assertAlmostEqual(float(data["terminalCostScaling"]), 8.0, places=1)
            self.assertAlmostEqual(float(data["Q_final"]["scaling"]), 3.0, places=1)
            for i in range(self.NUM_TERMINAL_DOFS):
                actual = float(data["Q_final"].get(f"({i},{i})", -1))
                self.assertAlmostEqual(
                    actual,
                    77.0 + i,
                    places=1,
                    msg=f"Q_final.({i},{i}) expected {77.0+i}, got {actual}",
                )
        finally:
            root.destroy()

    # ──────────────────────────────────────────────────────────
    #  Task-space foot costs: 18 weight params
    # ──────────────────────────────────────────────────────────
    def test_foot_cost_all_18_weights_have_sliders(self):
        """All 18 task_space_foot_cost_weights must have sliders."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "task_space_foot_cost_weights")
            for w in self.FOOT_COST_WEIGHTS:
                key = f"task_space_foot_cost_weights.{w}"
                self.assertIn(key, tab.slider_rows, f"Missing slider for {key}")
        finally:
            root.destroy()

    def test_foot_cost_roundtrip(self):
        """Modify all 18 foot cost weights and verify YAML round-trip."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "task_space_foot_cost_weights")
            for idx, w in enumerate(self.FOOT_COST_WEIGHTS):
                key = f"task_space_foot_cost_weights.{w}"
                if key in tab.slider_rows:
                    tab.slider_rows[key].set_value(10.0 + idx)

            tab.save_to_yaml()

            data = load_yaml_safe(self.tmp_task_file)
            for idx, w in enumerate(self.FOOT_COST_WEIGHTS):
                actual = float(data.get("task_space_foot_cost_weights", {}).get(w, -1))
                self.assertAlmostEqual(
                    actual,
                    10.0 + idx,
                    places=2,
                    msg=f"task_space_foot_cost_weights.{w} expected {10.0+idx}, got {actual}",
                )
        finally:
            root.destroy()

    # ──────────────────────────────────────────────────────────
    #  Task-space torso costs: 18 weight params
    # ──────────────────────────────────────────────────────────
    def test_torso_cost_all_18_weights_have_sliders(self):
        """All 18 task_space_costs.torso.weights must have sliders."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "task_space_costs")
            for w in self.TORSO_COST_WEIGHTS:
                key = f"task_space_costs.torso.weights.{w}"
                self.assertIn(key, tab.slider_rows, f"Missing slider for {key}")
        finally:
            root.destroy()

    def test_torso_cost_roundtrip(self):
        """Modify all 18 torso cost weights and verify YAML round-trip."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "task_space_costs")
            for idx, w in enumerate(self.TORSO_COST_WEIGHTS):
                key = f"task_space_costs.torso.weights.{w}"
                if key in tab.slider_rows:
                    tab.slider_rows[key].set_value(20.0 + idx)

            tab.save_to_yaml()

            data = load_yaml_safe(self.tmp_task_file)
            for idx, w in enumerate(self.TORSO_COST_WEIGHTS):
                actual = float(
                    data.get("task_space_costs", {})
                    .get("torso", {})
                    .get("weights", {})
                    .get(w, -1)
                )
                self.assertAlmostEqual(
                    actual,
                    20.0 + idx,
                    places=2,
                    msg=f"task_space_costs.torso.weights.{w} expected {20.0+idx}, got {actual}",
                )
        finally:
            root.destroy()

    # ──────────────────────────────────────────────────────────
    #  ICP cost
    # ──────────────────────────────────────────────────────────
    def test_icp_cost_slider_exists(self):
        """icp_cost_weights.icpErrorWeight slider must exist."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "icp_cost_weights")
            self.assertIn(
                "icp_cost_weights.icpErrorWeight",
                tab.slider_rows,
                "icpErrorWeight slider missing",
            )
        finally:
            root.destroy()

    def test_icp_cost_roundtrip(self):
        """Modify icpErrorWeight and verify YAML round-trip."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "icp_cost_weights")
            key = "icp_cost_weights.icpErrorWeight"
            if key in tab.slider_rows:
                tab.slider_rows[key].set_value(42.0)
                tab.save_to_yaml()
                data = load_yaml_safe(self.tmp_task_file)
                actual = float(
                    data.get("icp_cost_weights", {}).get("icpErrorWeight", -1)
                )
                self.assertAlmostEqual(actual, 42.0, places=1)
        finally:
            root.destroy()

    # ──────────────────────────────────────────────────────────
    #  Constraints & Barriers
    # ──────────────────────────────────────────────────────────
    def test_foot_constraint_all_9_gains_have_sliders(self):
        """All 9 foot constraint gains must have sliders."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "model_settings")
            for g in self.FOOT_CONSTRAINT_GAINS:
                key = f"model_settings.foot_constraint.{g}"
                self.assertIn(key, tab.slider_rows, f"Missing slider for {key}")
        finally:
            root.destroy()

    def test_swing_trajectory_all_8_params_have_sliders(self):
        """All 8 swing trajectory parameters must have sliders."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "swing_trajectory_config")
            for p in self.SWING_TRAJECTORY_PARAMS:
                key = f"swing_trajectory_config.{p}"
                self.assertIn(key, tab.slider_rows, f"Missing slider for {key}")
        finally:
            root.destroy()

    def test_friction_cone_barrier_sliders_exist(self):
        """Friction cone frictionCoefficient, mu, delta sliders must exist."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "contacts")
            for p in self.FRICTION_CONE_PARAMS:
                key = f"contacts.frictionForceConeSoftConstraint.{p}"
                self.assertIn(key, tab.slider_rows, f"Missing slider for {key}")
        finally:
            root.destroy()

    def test_joint_limits_barrier_sliders_exist(self):
        """Joint limits mu, delta sliders must exist."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "jointLimits")
            for p in self.JOINT_LIMITS_PARAMS:
                key = f"jointLimits.{p}"
                self.assertIn(key, tab.slider_rows, f"Missing slider for {key}")
        finally:
            root.destroy()

    def test_contact_moment_xy_barrier_sliders_exist(self):
        """Contact moment XY mu, delta sliders must exist."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "contacts")
            for p in self.CONTACT_MOMENT_XY_PARAMS:
                key = f"contacts.contactMomentXYSoftConstraint.{p}"
                self.assertIn(key, tab.slider_rows, f"Missing slider for {key}")
        finally:
            root.destroy()

    def test_collision_constraint_barrier_sliders_exist(self):
        """Collision constraint mu, delta sliders must exist."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            tab = self._create_tab(root, "collision_constraint")
            for p in self.COLLISION_CONSTRAINT_PARAMS:
                key = f"collision_constraint.{p}"
                self.assertIn(key, tab.slider_rows, f"Missing slider for {key}")
        finally:
            root.destroy()

    def test_leg_torque_costs_sliders_exist(self):
        """Left and right leg joint torque costs scaling and per-joint sliders must exist."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            keys = self._all_slider_keys(root)
            self.assertIn("left_leg_torque_cost.weights.scaling", keys)
            self.assertIn("right_leg_torque_cost.weights.scaling", keys)
            for i in range(6):
                self.assertIn(f'left_leg_torque_cost.weights."({i},0)"', keys)
                self.assertIn(f'right_leg_torque_cost.weights."({i},0)"', keys)
        finally:
            root.destroy()

    def test_contact_geometry_and_collision_radii_sliders_exist(self):
        """Foot contact translation, contact rectangle, and collision sphere radii sliders must exist."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            keys = self._all_slider_keys(root)
            for axis in ["x", "y", "z"]:
                self.assertIn(f"contacts.contact_frame_translation.{axis}", keys)
            for k in ["x_min", "x_max", "y_min", "y_max"]:
                self.assertIn(f"contacts.contact_rectangle.{k}", keys)
            self.assertIn("collision_constraint.foot.footCollisionSphereRadius", keys)
            self.assertIn("collision_constraint.knee.kneeCollisionSphereRadius", keys)
        finally:
            root.destroy()

    def test_solver_and_horizon_sliders_exist(self):
        """MPC loop rates, horizon, SQP multiple shooting, and rollout sliders must exist."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            keys = self._all_slider_keys(root)
            self.assertIn("mpc.timeHorizon", keys)
            self.assertIn("mpc.mpcDesiredFrequency", keys)
            self.assertIn("mpc.mrtDesiredFrequency", keys)
            self.assertIn("contact_wrench_gate.debounceTime", keys)
            self.assertIn("contact_wrench_gate.rampTime", keys)
            for k in [
                "sqpIteration",
                "dt",
                "deltaTol",
                "g_max",
                "g_min",
                "inequalityConstraintMu",
                "inequalityConstraintDelta",
            ]:
                self.assertIn(f"multiple_shooting.{k}", keys)
            self.assertIn("model_settings.phaseTransitionStanceTime", keys)
            self.assertIn("rollout.timeStep", keys)
        finally:
            root.destroy()

    def test_constraints_and_barriers_full_roundtrip(self):
        """Modify every constraint/barrier parameter and verify YAML round-trip."""
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            # These parameters are spread over five blocks of the file, so the tab renders five different categories
            # in the course of this test; _set switches to each one and the values persist across the switches.
            tab = self._create_tab(root)

            # Foot constraint gains
            for idx, g in enumerate(self.FOOT_CONSTRAINT_GAINS):
                self._set(tab, f"model_settings.foot_constraint.{g}", 1.0 + idx)

            # Swing trajectory
            for idx, p in enumerate(self.SWING_TRAJECTORY_PARAMS):
                self._set(tab, f"swing_trajectory_config.{p}", 0.01 * (idx + 1))

            # Friction cone barrier
            for idx, p in enumerate(self.FRICTION_CONE_PARAMS):
                self._set(
                    tab,
                    f"contacts.frictionForceConeSoftConstraint.{p}",
                    0.1 * (idx + 1),
                )

            # Contact moment XY barrier
            for idx, p in enumerate(self.CONTACT_MOMENT_XY_PARAMS):
                self._set(
                    tab, f"contacts.contactMomentXYSoftConstraint.{p}", 0.3 + idx * 0.1
                )

            # Joint limits barrier
            for idx, p in enumerate(self.JOINT_LIMITS_PARAMS):
                self._set(tab, f"jointLimits.{p}", 500.0 + idx * 100)

            # Collision constraint barrier
            for idx, p in enumerate(self.COLLISION_CONSTRAINT_PARAMS):
                self._set(tab, f"collision_constraint.{p}", 1000.0 + idx * 500)

            # Force an explicit save
            tab.save_to_yaml()

            # Verify
            data = load_yaml_safe(self.tmp_task_file)

            for idx, g in enumerate(self.FOOT_CONSTRAINT_GAINS):
                actual = float(
                    data.get("model_settings", {}).get("foot_constraint", {}).get(g, -1)
                )
                self.assertAlmostEqual(
                    actual,
                    1.0 + idx,
                    places=1,
                    msg=f"foot_constraint.{g} mismatch",
                )

            for idx, p in enumerate(self.FRICTION_CONE_PARAMS):
                actual = float(
                    data.get("contacts", {})
                    .get("frictionForceConeSoftConstraint", {})
                    .get(p, -1)
                )
                self.assertAlmostEqual(
                    actual,
                    0.1 * (idx + 1),
                    places=2,
                    msg=f"frictionCone.{p} mismatch",
                )

            for idx, p in enumerate(self.CONTACT_MOMENT_XY_PARAMS):
                actual = float(
                    data.get("contacts", {})
                    .get("contactMomentXYSoftConstraint", {})
                    .get(p, -1)
                )
                self.assertAlmostEqual(
                    actual,
                    0.3 + idx * 0.1,
                    places=2,
                    msg=f"contactMomentXY.{p} mismatch",
                )

            for idx, p in enumerate(self.JOINT_LIMITS_PARAMS):
                actual = float(data.get("jointLimits", {}).get(p, -1))
                self.assertAlmostEqual(
                    actual,
                    500.0 + idx * 100,
                    places=1,
                    msg=f"jointLimits.{p} mismatch",
                )

            for idx, p in enumerate(self.COLLISION_CONSTRAINT_PARAMS):
                actual = float(data.get("collision_constraint", {}).get(p, -1))
                self.assertAlmostEqual(
                    actual,
                    1000.0 + idx * 500,
                    places=1,
                    msg=f"collision_constraint.{p} mismatch",
                )
        finally:
            root.destroy()

    # ──────────────────────────────────────────────────────────
    #  Summary: total slider count
    # ──────────────────────────────────────────────────────────
    def test_every_numeric_leaf_of_the_file_reaches_exactly_one_slider(self):
        """The sliders across all blocks are exactly the numeric leaves of the file - no more, no fewer.

        This used to be a hardcoded total (151 for this robot, split across five hand-written category names), which
        went stale the moment anybody added a parameter and said nothing about whether it had reached the GUI. The
        property that actually matters is a bijection between the file and the sliders, and it states itself: a leaf
        with no slider is a parameter that silently vanished from the GUI, and a slider with no leaf is a parameter
        the GUI invented.
        """
        import tkinter as tk

        from remote_control.tk_app.yaml_param_tree import tunables as read_tunables

        expected = {tunable.dotted for tunable in read_tunables(self.tmp_task_file)}
        self.assertGreater(
            len(expected),
            100,
            "this robot's task file should carry well over a hundred tunables; a much smaller number means the "
            "file failed to parse rather than that the robot got simpler",
        )

        root = tk.Tk()
        root.withdraw()
        try:
            rendered = {key.replace('"', "") for key in self._all_slider_keys(root)}
        finally:
            root.destroy()

        self.assertEqual(
            sorted(expected - rendered),
            [],
            "these parameters of the file reach no slider",
        )
        self.assertEqual(
            sorted(rendered - expected),
            [],
            "these sliders correspond to nothing in the file",
        )


if __name__ == "__main__":
    unittest.main()
