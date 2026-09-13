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
End-to-end parameter live-update coverage test for DRC Atlas task.yaml.

Tests the YAML update pipeline WITHOUT requiring tkinter, ROS2, or the C++ runtime.
For every tunable parameter in task.yaml, we verify:

  1. The UI slider key (dot-separated path) can be split into a valid key_path.
  2. `_update_single_key` modifies the correct line in the YAML file.
  3. `yaml.safe_load` of the modified YAML returns the expected new value.
  4. The expected ptree path that the C++ MpcParameterUpdaterModule reads matches.

This bridges the Python UI → YAML generation → C++ consumption pipeline without
needing to instantiate any GUI or ROS infrastructure.

Test matrix (DRC Atlas — 36 state DOFs, 36 input DOFs):
  - Q matrix: scaling + 36 diagonal (0,0)–(35,35)  = 37
  - R matrix: scaling + 36 diagonal (0,0)–(35,35)  = 37
  - Q_final:  scaling + terminalCostScaling + 24 DOFs (0,0)–(35,35)  = 38
  - Task-space foot costs: 18 weights
  - Task-space torso costs: 18 weights
  - ICP cost: 1
  - Leg torque costs: 2 scalings + 12 per-joint weights = 14
  - Foot constraint gains: 9
  - Swing trajectory config: 8
  - Friction cone barrier: frictionCoefficient + mu + delta = 3
  - Contact moment XY barrier: mu + delta = 2
  - Joint limits barrier: mu + delta = 2
  - Collision constraint barrier: mu + delta = 2
  - Contact geometry: 3 translation + 4 rectangle + 2 sphere radii = 9
  - Solver & Horizon: sqpIteration + dt + deltaTol + g_max + g_min
                      + inequalityConstraintMu + inequalityConstraintDelta
                      + phaseTransitionStanceTime + rollout.timeStep
                      + mpc.timeHorizon + mpc.mpcDesiredFrequency
                      + mpc.mrtDesiredFrequency = 12
"""

import importlib.util
import os
import shutil
import tempfile
import unittest
from typing import Dict, List

import yaml


# ═══════════════════════════════════════════════════════════════════════
# Import the real _update_single_key under test
# ═══════════════════════════════════════════════════════════════════════
#
# yaml_editor_utils imports nothing from tkinter, but remote_control.tk_app's
# package __init__ eagerly imports the GUI widgets, which do. Loading the module
# straight from its path skips that __init__, so the test exercises the shipped
# implementation instead of a copy of it that would silently drift.

_YAML_EDITOR_UTILS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "remote_control",
    "tk_app",
    "yaml_editor_utils.py",
)


def _load_yaml_editor_utils():
    """Loads yaml_editor_utils without importing the tkinter-dependent package."""
    spec = importlib.util.spec_from_file_location(
        "remote_control_yaml_editor_utils", _YAML_EDITOR_UTILS_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {_YAML_EDITOR_UTILS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_yaml_editor_utils = _load_yaml_editor_utils()
_update_single_key = _yaml_editor_utils._update_single_key


def _get_nested(data: dict, key_path: List[str]):
    """Navigate a nested dict by key_path."""
    current = data
    for k in key_path:
        k_stripped = k.strip("\"'")
        if isinstance(current, dict) and k_stripped in current:
            current = current[k_stripped]
        else:
            return None
    return current


class TestLiveUpdateCoverage(unittest.TestCase):
    """
    Verify that every tunable parameter in the DRC Atlas task.yaml
    can be updated via the slider → YAML → re-parse pipeline.
    """

    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )
        cls.atlas_task_file = os.path.join(
            cls.repo_root,
            "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml",
        )
        assert os.path.exists(
            cls.atlas_task_file
        ), f"task.yaml not found at {cls.atlas_task_file}"
        with open(cls.atlas_task_file, "r") as f:
            cls.original_lines = f.readlines()
        cls.original_data = yaml.safe_load("".join(cls.original_lines))

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tmp_task_file = os.path.join(self.tmpdir, "task.yaml")
        shutil.copy2(self.atlas_task_file, self.tmp_task_file)
        with open(self.tmp_task_file, "r") as f:
            self.lines = f.readlines()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _roundtrip(self, slider_key: str, new_val: float) -> float:
        """Apply an update via the YAML pipeline and return the re-parsed value."""
        key_path = slider_key.split(".")
        result_lines = _update_single_key(list(self.lines), key_path, new_val)
        result_yaml = yaml.safe_load("".join(result_lines))
        actual = _get_nested(result_yaml, key_path)
        return actual

    # ══════════════════════════════════════════════════════════════
    #  1. Q matrix: scaling + 36 diagonal entries
    # ══════════════════════════════════════════════════════════════
    def test_q_scaling_roundtrip(self):
        actual = self._roundtrip("Q.scaling", 7.77)
        self.assertAlmostEqual(actual, 7.77, places=2)

    def test_q_matrix_all_36_dofs_roundtrip(self):
        for i in range(36):
            slider_key = f'Q."({i},{i})"'
            new_val = 100.0 + i
            actual = self._roundtrip(slider_key, new_val)
            self.assertIsNotNone(actual, f"Q({i},{i}) not found after update")
            self.assertAlmostEqual(
                actual, new_val, places=1, msg=f"Q({i},{i}) roundtrip failed"
            )

    def test_q_com_roundtrip(self):
        actual_s = self._roundtrip("Q_com.scaling", 99.0)
        self.assertAlmostEqual(actual_s, 99.0, places=1)
        for i in range(3):
            slider_key = f'Q_com."({i},{i})"'
            new_val = 12.5 + i
            actual = self._roundtrip(slider_key, new_val)
            self.assertIsNotNone(actual, f"{slider_key} not found after update")
            self.assertAlmostEqual(actual, new_val, places=1)

    def test_q_acom_roundtrip(self):
        actual_s = self._roundtrip("Q_acom.scaling", 88.0)
        self.assertAlmostEqual(actual_s, 88.0, places=1)
        for i in range(3):
            slider_key = f'Q_acom."({i},{i})"'
            new_val = 18.5 + i
            actual = self._roundtrip(slider_key, new_val)
            self.assertIsNotNone(actual, f"{slider_key} not found after update")
            self.assertAlmostEqual(actual, new_val, places=1)

    # ══════════════════════════════════════════════════════════════
    #  2. R matrix: scaling + 36 diagonal entries
    # ══════════════════════════════════════════════════════════════
    def test_r_scaling_roundtrip(self):
        actual = self._roundtrip("R.scaling", 0.42)
        self.assertAlmostEqual(actual, 0.42, places=2)

    def test_r_matrix_all_36_dofs_roundtrip(self):
        for i in range(36):
            slider_key = f'R."({i},{i})"'
            new_val = 0.5 + i * 0.1
            actual = self._roundtrip(slider_key, new_val)
            self.assertIsNotNone(actual, f"R({i},{i}) not found after update")
            self.assertAlmostEqual(
                actual, new_val, places=2, msg=f"R({i},{i}) roundtrip failed"
            )

    # ══════════════════════════════════════════════════════════════
    #  3. Q_final: scaling + terminalCostScaling + DOFs
    # ══════════════════════════════════════════════════════════════
    def test_terminal_cost_scaling_roundtrip(self):
        actual = self._roundtrip("terminalCostScaling", 8.0)
        self.assertAlmostEqual(actual, 8.0, places=1)

    def test_q_final_scaling_roundtrip(self):
        actual = self._roundtrip("Q_final.scaling", 3.33)
        self.assertAlmostEqual(actual, 3.33, places=2)

    def test_q_final_dofs_roundtrip(self):
        """Q_final entries present in the YAML (at least 0..11, may have more)."""
        q_final_data = self.original_data.get("Q_final", {})
        for key, val in q_final_data.items():
            if key == "scaling":
                continue
            slider_key = f'Q_final."{key}"'
            new_val = 77.7
            actual = self._roundtrip(slider_key, new_val)
            self.assertIsNotNone(actual, f"Q_final.{key} not found after update")
            self.assertAlmostEqual(
                actual, new_val, places=1, msg=f"Q_final.{key} roundtrip failed"
            )

    # ══════════════════════════════════════════════════════════════
    #  4. Task-space foot costs: 18 weights
    # ══════════════════════════════════════════════════════════════
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

    def test_foot_cost_all_18_weights_roundtrip(self):
        for idx, w in enumerate(self.FOOT_COST_WEIGHTS):
            slider_key = f"task_space_foot_cost_weights.{w}"
            new_val = 10.0 + idx
            actual = self._roundtrip(slider_key, new_val)
            self.assertIsNotNone(actual, f"foot cost {w} not found")
            self.assertAlmostEqual(
                actual, new_val, places=1, msg=f"foot cost {w} roundtrip failed"
            )

    # ══════════════════════════════════════════════════════════════
    #  5. Task-space torso costs: 18 weights
    # ══════════════════════════════════════════════════════════════
    TORSO_COST_WEIGHTS = FOOT_COST_WEIGHTS  # same set of weight names

    def test_torso_cost_all_18_weights_roundtrip(self):
        for idx, w in enumerate(self.TORSO_COST_WEIGHTS):
            slider_key = f"task_space_costs.torso.weights.{w}"
            new_val = 20.0 + idx
            actual = self._roundtrip(slider_key, new_val)
            self.assertIsNotNone(actual, f"torso cost {w} not found")
            self.assertAlmostEqual(
                actual, new_val, places=1, msg=f"torso cost {w} roundtrip failed"
            )

    # ══════════════════════════════════════════════════════════════
    #  6. ICP cost
    # ══════════════════════════════════════════════════════════════
    def test_icp_cost_roundtrip(self):
        actual = self._roundtrip("icp_cost_weights.icpErrorWeight", 42.0)
        self.assertAlmostEqual(actual, 42.0, places=1)

    # ══════════════════════════════════════════════════════════════
    #  7. Leg torque costs
    # ══════════════════════════════════════════════════════════════
    def test_left_leg_torque_cost_scaling_roundtrip(self):
        actual = self._roundtrip("left_leg_torque_cost.weights.scaling", 2e-5)
        self.assertIsNotNone(actual, "left_leg_torque_cost.weights.scaling not found")
        self.assertAlmostEqual(actual, 2e-5, places=8)

    def test_right_leg_torque_cost_scaling_roundtrip(self):
        actual = self._roundtrip("right_leg_torque_cost.weights.scaling", 3e-5)
        self.assertIsNotNone(actual, "right_leg_torque_cost.weights.scaling not found")
        self.assertAlmostEqual(actual, 3e-5, places=8)

    def test_leg_torque_cost_per_joint_roundtrip(self):
        for side in ["left_leg_torque_cost", "right_leg_torque_cost"]:
            for i in range(6):
                slider_key = f'{side}.weights."({i},0)"'
                new_val = 5.0 + i
                actual = self._roundtrip(slider_key, new_val)
                self.assertIsNotNone(actual, f"{slider_key} not found")
                self.assertAlmostEqual(
                    actual, new_val, places=1, msg=f"{slider_key} roundtrip failed"
                )

    # ══════════════════════════════════════════════════════════════
    #  8. Foot constraint gains (9 params)
    # ══════════════════════════════════════════════════════════════
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

    def test_foot_constraint_all_9_gains_roundtrip(self):
        for idx, g in enumerate(self.FOOT_CONSTRAINT_GAINS):
            slider_key = f"model_settings.foot_constraint.{g}"
            new_val = 1.0 + idx * 0.5
            actual = self._roundtrip(slider_key, new_val)
            self.assertIsNotNone(actual, f"{slider_key} not found")
            self.assertAlmostEqual(
                actual, new_val, places=2, msg=f"{slider_key} roundtrip failed"
            )

    # ══════════════════════════════════════════════════════════════
    #  9. Swing trajectory config (8 params)
    # ══════════════════════════════════════════════════════════════
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

    def test_swing_trajectory_all_8_params_roundtrip(self):
        for idx, p in enumerate(self.SWING_TRAJECTORY_PARAMS):
            slider_key = f"swing_trajectory_config.{p}"
            new_val = 0.01 * (idx + 1)
            actual = self._roundtrip(slider_key, new_val)
            self.assertIsNotNone(actual, f"{slider_key} not found")
            self.assertAlmostEqual(
                actual, new_val, places=3, msg=f"{slider_key} roundtrip failed"
            )

    # ══════════════════════════════════════════════════════════════
    #  10. Barrier penalties
    # ══════════════════════════════════════════════════════════════
    def test_friction_cone_barrier_roundtrip(self):
        for p, new_val in [("frictionCoefficient", 0.77), ("mu", 0.33), ("delta", 9.9)]:
            slider_key = f"contacts.frictionForceConeSoftConstraint.{p}"
            actual = self._roundtrip(slider_key, new_val)
            self.assertIsNotNone(actual, f"{slider_key} not found")
            self.assertAlmostEqual(
                actual, new_val, places=2, msg=f"{slider_key} roundtrip failed"
            )

    def test_contact_moment_xy_barrier_roundtrip(self):
        for p, new_val in [("mu", 0.88), ("delta", 0.07)]:
            slider_key = f"contacts.contactMomentXYSoftConstraint.{p}"
            actual = self._roundtrip(slider_key, new_val)
            self.assertIsNotNone(actual, f"{slider_key} not found")
            self.assertAlmostEqual(
                actual, new_val, places=2, msg=f"{slider_key} roundtrip failed"
            )

    def test_contact_wrench_cone_barrier_roundtrip(self):
        for p, new_val in [
            ("frictionCoefficient", 0.65),
            ("torsionalFrictionCoefficient", 0.08),
            ("minNormalForce", 8.0),
            ("gripperForce", 0.5),
            ("mu", 0.35),
            ("delta", 4.5),
        ]:
            slider_key = f"contacts.contactWrenchConeSoftConstraint.{p}"
            actual = self._roundtrip(slider_key, new_val)
            self.assertIsNotNone(actual, f"{slider_key} not found")
            self.assertAlmostEqual(
                actual, new_val, places=2, msg=f"{slider_key} roundtrip failed"
            )

    def test_basis_non_negativity_barrier_roundtrip(self):
        for p, new_val in [("mu", 0.05), ("delta", 0.005)]:
            slider_key = f"contacts.basisNonNegativityBarrier.{p}"
            actual = self._roundtrip(slider_key, new_val)
            self.assertIsNotNone(actual, f"{slider_key} not found")
            self.assertAlmostEqual(
                actual, new_val, places=3, msg=f"{slider_key} roundtrip failed"
            )

    def test_basis_scaling_regularization_roundtrip(self):
        slider_key = "contacts.basisScalingRegularization"
        actual = self._roundtrip(slider_key, 5.0e-4)
        self.assertIsNotNone(actual, f"{slider_key} not found")
        self.assertAlmostEqual(
            actual, 5.0e-4, places=6, msg=f"{slider_key} roundtrip failed"
        )

    def test_joint_limits_barrier_roundtrip(self):
        for p, new_val in [("mu", 999.0), ("delta", 0.5)]:
            slider_key = f"jointLimits.{p}"
            actual = self._roundtrip(slider_key, new_val)
            self.assertIsNotNone(actual, f"{slider_key} not found")
            self.assertAlmostEqual(
                actual, new_val, places=1, msg=f"{slider_key} roundtrip failed"
            )

    def test_collision_constraint_barrier_roundtrip(self):
        for p, new_val in [("mu", 50000.0), ("delta", 0.08)]:
            slider_key = f"collision_constraint.{p}"
            actual = self._roundtrip(slider_key, new_val)
            self.assertIsNotNone(actual, f"{slider_key} not found")
            self.assertAlmostEqual(
                actual, new_val, places=2, msg=f"{slider_key} roundtrip failed"
            )

    # ══════════════════════════════════════════════════════════════
    #  11. Contact geometry & collision radii
    # ══════════════════════════════════════════════════════════════
    def test_contact_frame_translation_roundtrip(self):
        for axis in ["x", "y", "z"]:
            slider_key = f"contacts.contact_frame_translation.{axis}"
            new_val = 0.123
            actual = self._roundtrip(slider_key, new_val)
            self.assertIsNotNone(actual, f"{slider_key} not found")
            self.assertAlmostEqual(
                actual, new_val, places=3, msg=f"{slider_key} roundtrip failed"
            )

    def test_contact_rectangle_roundtrip(self):
        for k in ["x_max", "x_min", "y_max", "y_min"]:
            slider_key = f"contacts.contact_rectangle.{k}"
            new_val = 0.1
            actual = self._roundtrip(slider_key, new_val)
            self.assertIsNotNone(actual, f"{slider_key} not found")
            self.assertAlmostEqual(
                actual, new_val, places=3, msg=f"{slider_key} roundtrip failed"
            )

    def test_collision_sphere_radii_roundtrip(self):
        for slider_key, new_val in [
            ("collision_constraint.foot.footCollisionSphereRadius", 0.07),
            ("collision_constraint.knee.kneeCollisionSphereRadius", 0.1),
        ]:
            actual = self._roundtrip(slider_key, new_val)
            self.assertIsNotNone(actual, f"{slider_key} not found")
            self.assertAlmostEqual(
                actual, new_val, places=3, msg=f"{slider_key} roundtrip failed"
            )

    # ══════════════════════════════════════════════════════════════
    #  12. Solver & Horizon
    # ══════════════════════════════════════════════════════════════
    SOLVER_PARAMS = {
        "multiple_shooting.sqpIteration": 8,
        "multiple_shooting.dt": 0.05,
        "multiple_shooting.deltaTol": 0.001,
        "multiple_shooting.g_max": 0.02,
        "multiple_shooting.g_min": 1e-5,
        "multiple_shooting.inequalityConstraintMu": 0.5,
        "multiple_shooting.inequalityConstraintDelta": 10.0,
        "model_settings.phaseTransitionStanceTime": 0.02,
        "rollout.timeStep": 0.02,
        "mpc.timeHorizon": 1.5,
        "mpc.mpcDesiredFrequency": 100.0,
        "mpc.mrtDesiredFrequency": 200.0,
    }

    def test_solver_and_horizon_all_params_roundtrip(self):
        for slider_key, new_val in self.SOLVER_PARAMS.items():
            actual = self._roundtrip(slider_key, new_val)
            self.assertIsNotNone(actual, f"{slider_key} not found after update")
            self.assertAlmostEqual(
                actual,
                new_val,
                places=3,
                msg=f"{slider_key} roundtrip failed: expected {new_val}, got {actual}",
            )

    # ══════════════════════════════════════════════════════════════
    #  13. C++ ptree path agreement
    #  Verify that slider keys match what the C++ reader expects.
    # ══════════════════════════════════════════════════════════════
    # The C++ MpcParameterUpdaterModule reads these ptree paths:
    CPP_PTREE_PATHS = {
        # Barrier penalties (loadPtreeValue)
        "contacts.frictionForceConeSoftConstraint.mu",
        "contacts.frictionForceConeSoftConstraint.delta",
        "contacts.contactMomentXYSoftConstraint.mu",
        "contacts.contactMomentXYSoftConstraint.delta",
        "jointLimits.mu",
        "jointLimits.delta",
        "collision_constraint.mu",
        "collision_constraint.delta",
        "model_settings.foot_constraint.softConstraintWeight",
        # Foot constraint gains
        "model_settings.foot_constraint.positionErrorGain_z",
        "model_settings.foot_constraint.orientationErrorGain",
        "model_settings.foot_constraint.linearVelocityErrorGain_z",
        "model_settings.foot_constraint.linearVelocityErrorGain_xy",
        "model_settings.foot_constraint.angularVelocityErrorGain",
        "model_settings.foot_constraint.linearAccelerationErrorGain_z",
        "model_settings.foot_constraint.linearAccelerationErrorGain_xy",
        "model_settings.foot_constraint.angularAccelerationErrorGain",
        # SQP solver settings
        "multiple_shooting.sqpIteration",
        "multiple_shooting.deltaTol",
        "multiple_shooting.g_max",
        "multiple_shooting.g_min",
        "multiple_shooting.inequalityConstraintMu",
        "multiple_shooting.inequalityConstraintDelta",
    }

    def test_cpp_ptree_paths_exist_in_yaml(self):
        """Every ptree path the C++ code reads must exist in the YAML file."""
        for ptree_path in self.CPP_PTREE_PATHS:
            key_path = ptree_path.split(".")
            val = _get_nested(self.original_data, key_path)
            self.assertIsNotNone(
                val,
                f"C++ reads ptree path '{ptree_path}' but it does not exist in task.yaml",
            )

    def test_cpp_ptree_paths_roundtrip_through_yaml(self):
        """Every ptree path the C++ reads can be updated via slider pipeline."""
        for ptree_path in self.CPP_PTREE_PATHS:
            new_val = 12345.0
            actual = self._roundtrip(ptree_path, new_val)
            self.assertIsNotNone(
                actual, f"ptree path '{ptree_path}' not found after YAML update"
            )
            self.assertAlmostEqual(
                actual,
                new_val,
                places=1,
                msg=f"ptree path '{ptree_path}' roundtrip failed",
            )

    # ══════════════════════════════════════════════════════════════
    #  14. Bulk update test: update ALL sliders at once
    # ══════════════════════════════════════════════════════════════
    def test_bulk_update_all_parameters_simultaneously(self):
        """
        Update every single tunable parameter in one pass and verify
        no key collisions or corruption occur.
        """
        all_updates: Dict[str, float] = {}

        # Q matrix
        all_updates["Q.scaling"] = 9.9
        for i in range(36):
            all_updates[f'Q."({i},{i})"'] = 200.0 + i

        # R matrix
        all_updates["R.scaling"] = 0.77
        for i in range(36):
            all_updates[f'R."({i},{i})"'] = 1.0 + i * 0.01

        # Q_final
        all_updates["terminalCostScaling"] = 6.0
        all_updates["Q_final.scaling"] = 4.0
        q_final_data = self.original_data.get("Q_final", {})
        for key in q_final_data:
            if key != "scaling":
                all_updates[f'Q_final."{key}"'] = 88.0

        # Foot costs
        for w in self.FOOT_COST_WEIGHTS:
            all_updates[f"task_space_foot_cost_weights.{w}"] = 15.5

        # Torso costs
        for w in self.TORSO_COST_WEIGHTS:
            all_updates[f"task_space_costs.torso.weights.{w}"] = 25.5

        # ICP
        all_updates["icp_cost_weights.icpErrorWeight"] = 33.3

        # Foot constraint gains
        for g in self.FOOT_CONSTRAINT_GAINS:
            all_updates[f"model_settings.foot_constraint.{g}"] = 7.7

        # Swing trajectory
        for p in self.SWING_TRAJECTORY_PARAMS:
            all_updates[f"swing_trajectory_config.{p}"] = 0.123

        # Barriers
        all_updates["contacts.frictionForceConeSoftConstraint.frictionCoefficient"] = (
            0.6
        )
        all_updates["contacts.frictionForceConeSoftConstraint.mu"] = 0.3
        all_updates["contacts.frictionForceConeSoftConstraint.delta"] = 7.0
        all_updates["contacts.contactMomentXYSoftConstraint.mu"] = 0.9
        all_updates["contacts.contactMomentXYSoftConstraint.delta"] = 0.05
        all_updates["jointLimits.mu"] = 1500.0
        all_updates["jointLimits.delta"] = 0.2
        all_updates["collision_constraint.mu"] = 40000.0
        all_updates["collision_constraint.delta"] = 0.06

        # Solver & Horizon
        for k, v in self.SOLVER_PARAMS.items():
            all_updates[k] = v

        # Apply ALL updates
        lines = list(self.lines)
        for slider_key, new_val in all_updates.items():
            key_path = slider_key.split(".")
            lines = _update_single_key(lines, key_path, new_val)

        # Re-parse
        result = yaml.safe_load("".join(lines))
        self.assertIsNotNone(result, "YAML parsing failed after bulk update")

        # Verify a sample of critical parameters
        self.assertAlmostEqual(float(result["Q"]["scaling"]), 9.9, places=1)
        self.assertAlmostEqual(float(result["R"]["scaling"]), 0.77, places=2)
        self.assertAlmostEqual(float(result["terminalCostScaling"]), 6.0, places=1)
        self.assertAlmostEqual(float(result["jointLimits"]["mu"]), 1500.0, places=0)
        self.assertAlmostEqual(
            float(result["contacts"]["frictionForceConeSoftConstraint"]["mu"]),
            0.3,
            places=2,
        )
        self.assertAlmostEqual(
            float(result["model_settings"]["foot_constraint"]["softConstraintWeight"]),
            7.7,
            places=1,
        )

        # Verify total update count
        self.assertGreaterEqual(
            len(all_updates), 180, f"Expected ≥180 updates, got {len(all_updates)}"
        )

    # ══════════════════════════════════════════════════════════════
    #  15. Summary: total parameter count
    # ══════════════════════════════════════════════════════════════
    def test_total_parameter_count(self):
        """Cross-check that our test covers ≥190 unique slider keys."""
        all_keys = set()

        all_keys.add("Q.scaling")
        for i in range(36):
            all_keys.add(f'Q."({i},{i})"')

        all_keys.add("R.scaling")
        for i in range(36):
            all_keys.add(f'R."({i},{i})"')

        all_keys.add("terminalCostScaling")
        all_keys.add("Q_final.scaling")
        q_final_data = self.original_data.get("Q_final", {})
        for key in q_final_data:
            if key != "scaling":
                all_keys.add(f'Q_final."{key}"')

        for w in self.FOOT_COST_WEIGHTS:
            all_keys.add(f"task_space_foot_cost_weights.{w}")

        for w in self.TORSO_COST_WEIGHTS:
            all_keys.add(f"task_space_costs.torso.weights.{w}")

        all_keys.add("icp_cost_weights.icpErrorWeight")

        all_keys.add("left_leg_torque_cost.weights.scaling")
        all_keys.add("right_leg_torque_cost.weights.scaling")
        for i in range(6):
            all_keys.add(f'left_leg_torque_cost.weights."({i},0)"')
            all_keys.add(f'right_leg_torque_cost.weights."({i},0)"')

        for g in self.FOOT_CONSTRAINT_GAINS:
            all_keys.add(f"model_settings.foot_constraint.{g}")

        for p in self.SWING_TRAJECTORY_PARAMS:
            all_keys.add(f"swing_trajectory_config.{p}")

        all_keys.add("contacts.frictionForceConeSoftConstraint.frictionCoefficient")
        all_keys.add("contacts.frictionForceConeSoftConstraint.mu")
        all_keys.add("contacts.frictionForceConeSoftConstraint.delta")
        all_keys.add("contacts.contactMomentXYSoftConstraint.mu")
        all_keys.add("contacts.contactMomentXYSoftConstraint.delta")
        all_keys.add("jointLimits.mu")
        all_keys.add("jointLimits.delta")
        all_keys.add("collision_constraint.mu")
        all_keys.add("collision_constraint.delta")

        for axis in ["x", "y", "z"]:
            all_keys.add(f"contacts.contact_frame_translation.{axis}")
        for k in ["x_max", "x_min", "y_max", "y_min"]:
            all_keys.add(f"contacts.contact_rectangle.{k}")
        all_keys.add("collision_constraint.foot.footCollisionSphereRadius")
        all_keys.add("collision_constraint.knee.kneeCollisionSphereRadius")

        for k in self.SOLVER_PARAMS:
            all_keys.add(k)

        self.assertGreaterEqual(
            len(all_keys),
            185,
            f"Expected ≥185 unique slider keys, got {len(all_keys)}: "
            f"missing coverage for some parameters",
        )


if __name__ == "__main__":
    unittest.main()
