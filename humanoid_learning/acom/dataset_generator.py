"""Dataset generator for Angular Center of Mass (aCOM).

Samples robot joint configurations and computes the ground truth
locked-inertia normalized centroidal angular momentum matrix:
    A_bar_omega(q) = I_G^{-1}(q) * A_{omega, j}(q)
using MuJoCo rigid-body dynamics.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import mujoco


class AcomDatasetGenerator:
    """Generates training datasets for aCOM from MuJoCo robot models."""

    def __init__(self, xml_path: str):
        """Initializes generator from a MuJoCo XML file."""
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Determine floating base DoFs and joint DoFs
        self.nq = self.model.nq
        self.nv = self.model.nv

        # By convention in humanoid models: root joint is 6 DoF (free joint: 7 qpos, 6 qvel)
        # and remaining joints are internal joints.
        self.is_floating = self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        if self.is_floating:
            self.base_qpos_dim = 7
            self.base_qvel_dim = 6
            self.num_joints = self.nv - 6
        else:
            self.base_qpos_dim = 0
            self.base_qvel_dim = 0
            self.num_joints = self.nv

        # Extract joint limits for actuated internal joints
        self.joint_limits_lower = []
        self.joint_limits_upper = []

        start_jnt = 1 if self.is_floating else 0
        for j in range(start_jnt, self.model.njnt):
            if self.model.jnt_limited[j]:
                self.joint_limits_lower.append(self.model.jnt_range[j, 0])
                self.joint_limits_upper.append(self.model.jnt_range[j, 1])
            else:
                self.joint_limits_lower.append(-np.pi)
                self.joint_limits_upper.append(np.pi)

        self.joint_limits_lower = np.array(self.joint_limits_lower, dtype=np.float64)
        self.joint_limits_upper = np.array(self.joint_limits_upper, dtype=np.float64)

    def compute_centroidal_matrices(
        self, q_joints: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes I_G, A_omega, and locked normalized A_bar_omega for a given joint configuration.

        Returns:
            I_G: (3, 3) whole-body locked inertia at CoM
            A_omega_j: (3, n_j) centroidal angular momentum matrix for joint velocities
            A_bar_omega: (3, n_j) locked-inertia normalized matrix I_G^{-1} * A_omega_j
        """
        # Set configuration (base at origin, orientation identity)
        self.data.qpos[:] = 0.0
        if self.is_floating:
            self.data.qpos[2] = 0.8  # nominal height
            self.data.qpos[3] = 1.0  # quat w
            self.data.qpos[7:] = q_joints
        else:
            self.data.qpos[:] = q_joints

        # Forward kinematics and composite rigid body inertia
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        com = np.copy(self.data.subtree_com[0])  # whole-body CoM
        total_mass = self.model.body_mass.sum()

        # Compute centroidal momentum matrix A(q) via unit-velocity evaluations
        # For floating base: nv = 6 (3 lin, 3 ang) + n_j
        # We need the angular part (momentum about whole-body CoM)
        A_omega = np.zeros((3, self.nv), dtype=np.float64)

        for col in range(self.nv):
            self.data.qvel[:] = 0.0
            self.data.qvel[col] = 1.0
            mujoco.mj_forward(self.model, self.data)

            # Subtree angular momentum about world origin
            # Subtree angmom at root body in MuJoCo is about the subtree CoM!
            angmom_about_com = np.copy(self.data.subtree_angmom[0])
            A_omega[:, col] = angmom_about_com

        # Extract I_G (locked inertia for base angular velocity columns: col 3,4,5)
        # and A_omega_j (columns 6 to nv)
        if self.is_floating:
            I_G = A_omega[:, 3:6]
            A_omega_j = A_omega[:, 6:]
        else:
            I_G = np.eye(3)
            A_omega_j = A_omega

        # Invert locked inertia
        I_G_inv = np.linalg.pinv(I_G, rcond=1e-5)
        A_bar_omega = I_G_inv @ A_omega_j

        return I_G, A_omega_j, A_bar_omega

    def generate_dataset(
        self, num_samples: int = 10000, seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """Samples random joint configurations and generates training dataset.

        Returns:
            dict containing:
                - 'q_joints': (num_samples, n_j)
                - 'A_bar_omega': (num_samples, 3, n_j)
                - 'I_G': (num_samples, 3, 3)
        """
        rng = np.random.default_rng(seed)

        # Sample joint positions uniformly within 80% of joint limits to stay away from singularities
        margin = 0.1 * (self.joint_limits_upper - self.joint_limits_lower)
        low = self.joint_limits_lower + margin
        high = self.joint_limits_upper - margin

        q_samples = rng.uniform(low, high, size=(num_samples, self.num_joints))

        A_bar_samples = np.zeros((num_samples, 3, self.num_joints), dtype=np.float32)
        I_G_samples = np.zeros((num_samples, 3, 3), dtype=np.float32)

        for i in range(num_samples):
            I_G, _, A_bar = self.compute_centroidal_matrices(q_samples[i])
            A_bar_samples[i] = A_bar.astype(np.float32)
            I_G_samples[i] = I_G.astype(np.float32)

        return {
            "q_joints": q_samples.astype(np.float32),
            "A_bar_omega": A_bar_samples,
            "I_G": I_G_samples,
        }
