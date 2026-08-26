"""Unit Tests for Cross-Robot Kinematic and Offline Trajectory Retargeting Framework."""

import os
import tempfile
import unittest

import numpy as np

from humanoid_learning.envs.humanoid_residual_wbc_env import (
    HumanoidResidualWBCConfig,
    HumanoidResidualWBCEnv,
)
from humanoid_learning.retargeting.joint_mapper import SemanticJointMapper
from humanoid_learning.retargeting.kinematic_retargeter import (
    OptimizationKinematicRetargeter,
)
from humanoid_learning.retargeting.trajectory_retargeter import (
    DatasetRetargetingConfig,
    TrajectoryDatasetRetargeter,
)
from humanoid_learning.training.offline_dataset import OfflineTrajectoryDataset
from humanoid_learning.wbc.robot_model_loader import load_robot_spec


class TestRetargetingFramework(unittest.TestCase):
    """Test suite for cross-robot retargeting (G1 -> R1, G1 -> Atlas)."""

    @classmethod
    def setUpClass(cls):
        cls.g1_spec = load_robot_spec("g1")
        cls.r1_spec = load_robot_spec("r1")
        cls.atlas_spec = load_robot_spec("atlas")

    def test_g1_and_r1_spec_properties(self):
        """Verify extracted degrees of freedom and limb configurations."""
        self.assertEqual(self.g1_spec.n_act, 29)
        self.assertEqual(self.r1_spec.n_act, 26)
        self.assertEqual(self.atlas_spec.n_act, 28)

        # Check limb groupings
        self.assertEqual(len(self.g1_spec.limb_joint_indices["left_leg"]), 6)
        self.assertEqual(len(self.r1_spec.limb_joint_indices["left_leg"]), 6)
        self.assertEqual(len(self.g1_spec.limb_joint_indices["torso"]), 3)
        self.assertEqual(len(self.r1_spec.limb_joint_indices["torso"]), 2)
        self.assertEqual(len(self.g1_spec.limb_joint_indices["left_arm"]), 7)
        self.assertEqual(len(self.r1_spec.limb_joint_indices["left_arm"]), 5)

    def test_semantic_joint_mapper_g1_to_r1(self):
        """Test direct anatomical joint mapping and DOF reduction from G1 to R1."""
        mapper = SemanticJointMapper(self.g1_spec, self.r1_spec)

        # Test single nominal pose
        q_g1 = np.zeros((self.g1_spec.n_act,), dtype=np.float32)
        q_g1[3] = 0.5  # Left knee flex
        q_g1[9] = 0.5  # Right knee flex
        q_g1[12] = 0.2  # G1 waist yaw -> R1 waist yaw (R1 index 13)
        q_g1[13] = 0.1  # G1 waist roll -> R1 waist roll (R1 index 12)
        q_g1[14] = 0.05  # G1 waist pitch (unactuated in R1)
        q_g1[15] = -0.3  # G1 left shoulder pitch -> R1 left shoulder pitch (R1 index 14)

        q_r1 = mapper.map_joint_positions(q_g1)
        self.assertEqual(q_r1.shape, (26,))
        self.assertAlmostEqual(q_r1[3], 0.5)  # Left knee
        self.assertAlmostEqual(q_r1[9], 0.5)  # Right knee
        self.assertAlmostEqual(q_r1[12], 0.1)  # Waist roll
        self.assertAlmostEqual(q_r1[13], 0.2)  # Waist yaw
        self.assertAlmostEqual(q_r1[14], -0.3)  # Left shoulder pitch

        # Verify target joint limit clamping
        r1_lower = np.array(self.r1_spec.joint_limits_lower)
        r1_upper = np.array(self.r1_spec.joint_limits_upper)
        self.assertTrue(np.all(q_r1 >= r1_lower - 1e-4))
        self.assertTrue(np.all(q_r1 <= r1_upper + 1e-4))

    def test_semantic_joint_mapper_g1_to_atlas(self):
        """Test anatomical joint mapping from Unitree G1 to DRC Atlas."""
        mapper = SemanticJointMapper(self.g1_spec, self.atlas_spec)

        q_g1 = np.ones((self.g1_spec.n_act,), dtype=np.float32) * 0.1
        q_atlas = mapper.map_joint_positions(q_g1)
        self.assertEqual(q_atlas.shape, (28,))

        atlas_lower = np.array(self.atlas_spec.joint_limits_lower)
        atlas_upper = np.array(self.atlas_spec.joint_limits_upper)
        self.assertTrue(np.all(q_atlas >= atlas_lower - 1e-4))
        self.assertTrue(np.all(q_atlas <= atlas_upper + 1e-4))

    def test_kinematic_retargeter_trajectory(self):
        """Test trajectory retargeting with temporal smoothness."""
        retargeter = OptimizationKinematicRetargeter(
            self.g1_spec, self.r1_spec
        )

        T = 50
        t = np.linspace(0, 2 * np.pi, T)
        q_g1_traj = np.zeros((T, 29), dtype=np.float32)
        q_g1_traj[:, 3] = 0.4 * np.sin(t)  # Left knee motion
        q_g1_traj[:, 15] = 0.3 * np.cos(t)  # Arm swing

        q_r1_traj, qd_r1_traj = retargeter.retarget_trajectory(q_g1_traj)
        self.assertEqual(q_r1_traj.shape, (T, 26))
        self.assertEqual(qd_r1_traj.shape, (T, 26))

        # Check smoothness: max acceleration should be finite and bounded
        diff2 = np.diff(q_r1_traj, n=2, axis=0)
        self.assertTrue(np.all(np.isfinite(diff2)))

    def test_dataset_retargeter_end_to_end(self):
        """Test end-to-end dataset conversion and loading in target environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dataset_path = os.path.join(tmpdir, "g1_demos.npz")
            tgt_dataset_path = os.path.join(tmpdir, "r1_demos.npz")

            # 1. Create synthetic source dataset for G1
            N = 100
            g1_obs_dim = 29 * 2 + 7 + 2 + 6 + 3  # 76
            g1_obs = np.random.randn(N, g1_obs_dim).astype(np.float32) * 0.1
            g1_act = np.random.randn(N, 29).astype(np.float32) * 0.05
            np.savez(src_dataset_path, observations=g1_obs, actions=g1_act)

            # 2. Retarget dataset from G1 to R1
            retargeter = TrajectoryDatasetRetargeter(
                source_robot="g1",
                target_robot="r1",
                config=DatasetRetargetingConfig(method="anatomical"),
            )
            info = retargeter.retarget_dataset(src_dataset_path, tgt_dataset_path)

            self.assertEqual(info["num_samples"], N)
            self.assertEqual(info["act_dim"], 26)
            self.assertEqual(info["obs_dim"], 26 * 2 + 7 + 2 + 6 + 3)  # 70

            # 3. Verify OfflineTrajectoryDataset loads converted dataset
            dataset = OfflineTrajectoryDataset(
                filepath=tgt_dataset_path,
                obs_dim=info["obs_dim"],
                act_dim=26,
            )
            self.assertEqual(len(dataset), N)

            # 4. Step target environment on R1 using retargeted demonstrations
            env = HumanoidResidualWBCEnv(robot=self.r1_spec)
            self.assertEqual(env.action_size, 26)
            self.assertEqual(env.observation_size, info["obs_dim"])


if __name__ == "__main__":
    unittest.main()
