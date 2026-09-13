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

"""Dataset generator for Angular Center of Mass (aCOM).

Samples robot joint configurations and computes the ground truth
locked-inertia normalized centroidal angular momentum matrix:
    A_bar_omega(q) = I_G^{-1}(q) * A_{omega, j}(q)
using MuJoCo rigid-body dynamics.
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
import mujoco
import numpy as np


def resolve_xml_path(path: str) -> str:
    """Resolves XML path whether run directly or via Bazel runfiles."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)

    ws = os.environ.get("BUILD_WORKSPACE_DIRECTORY", "")
    if ws and os.path.exists(os.path.join(ws, path)):
        return os.path.join(ws, path)

    runfiles = os.environ.get("PYTHON_RUNFILES", "")
    if runfiles:
        p = os.path.join(runfiles, "_main", path)
        if os.path.exists(p):
            return p

    return path


class AcomDatasetGenerator:
    """Generates training datasets for aCOM from MuJoCo robot models.

    Args:
        xml_path: Path to MuJoCo XML model file.
        urdf_path: Optional path to URDF file. When provided, a joint permutation
            is computed so that all dataset outputs (q_joints, A_bar_omega) use
            URDF/Pinocchio joint ordering instead of MuJoCo joint ordering. This
            is critical for C++ runtime parity because the MPC uses Pinocchio.
    """

    def __init__(
        self,
        xml_path: str,
        urdf_path: Optional[str] = None,
        fixed_joints: Optional[List[str]] = None,
    ):
        """Initializes generator from a MuJoCo XML file."""
        resolved_path = resolve_xml_path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(resolved_path)
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
            self.num_mj_joints = self.nv - 6
        else:
            self.base_qpos_dim = 0
            self.base_qvel_dim = 0
            self.num_mj_joints = self.nv

        # Collect MuJoCo actuated joint names (skip the free joint if present)
        start_jnt = 1 if self.is_floating else 0
        self.mj_joint_names: List[str] = []
        for j in range(start_jnt, self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            self.mj_joint_names.append(name)

        self.fixed_joints = set(fixed_joints or [])
        self.mj_fixed_indices = [
            self.mj_joint_names.index(name)
            for name in self.fixed_joints
            if name in self.mj_joint_names
        ]

        # Extract joint limits for actuated internal joints (in MuJoCo order)
        self.joint_limits_lower = []
        self.joint_limits_upper = []

        for j in range(start_jnt, self.model.njnt):
            if self.model.jnt_limited[j]:
                self.joint_limits_lower.append(self.model.jnt_range[j, 0])
                self.joint_limits_upper.append(self.model.jnt_range[j, 1])
            else:
                self.joint_limits_lower.append(-np.pi)
                self.joint_limits_upper.append(np.pi)

        self.joint_limits_lower = np.array(self.joint_limits_lower, dtype=np.float64)
        self.joint_limits_upper = np.array(self.joint_limits_upper, dtype=np.float64)

        # Compute URDF ↔ MuJoCo joint permutation if URDF provided.
        # perm[i] = MuJoCo index of URDF's i-th joint.
        # Usage:
        #   MuJoCo→URDF (gather):   q_urdf = q_mj[perm]
        #   URDF→MuJoCo (scatter):  q_mj[perm] = q_urdf
        # LINT.IfChange(joint_permutation)
        if urdf_path is not None:
            resolved_urdf = resolve_xml_path(urdf_path)
            self.urdf_joint_names, self.joint_perm = self._compute_joint_permutation(
                resolved_urdf
            )
        else:
            self.urdf_joint_names = self.mj_joint_names
            self.joint_perm = None

        self.num_active_joints = len(
            [j for j in self.urdf_joint_names if j not in self.fixed_joints]
        )
        # LINT.ThenChange(//humanoid_learning/acom/tests/test_acom.py:joint_permutation_test)

    def _compute_joint_permutation(
        self, urdf_path: str
    ) -> Tuple[List[str], np.ndarray]:
        """Computes index permutation from URDF joint ordering to MuJoCo joint ordering.

        Returns:
            urdf_joint_names: list of non-fixed URDF joint names in URDF document order.
            perm: np.ndarray where perm[i] = MuJoCo joint index of URDF's i-th joint.
        """
        urdf_tree = ET.parse(urdf_path)
        urdf_joint_names = [
            j.attrib["name"]
            for j in urdf_tree.findall(".//joint")
            if j.attrib.get("type") not in ("fixed", "floating")
        ]

        # Validate that every URDF joint exists in MuJoCo (and vice-versa)
        mj_set = set(self.mj_joint_names)
        urdf_set = set(urdf_joint_names)
        missing_in_mj = urdf_set - mj_set
        missing_in_urdf = mj_set - urdf_set
        if missing_in_mj:
            raise ValueError(f"URDF joints not found in MuJoCo model: {missing_in_mj}")
        if missing_in_urdf:
            raise ValueError(
                f"MuJoCo joints not found in URDF model: {missing_in_urdf}"
            )

        perm = np.array(
            [self.mj_joint_names.index(name) for name in urdf_joint_names],
            dtype=np.int64,
        )
        return urdf_joint_names, perm

    def compute_centroidal_matrices(
        self, q_joints_mj: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes I_G, A_omega, and locked normalized A_bar_omega for a given joint configuration.

        Args:
            q_joints_mj: Joint angles in **MuJoCo** ordering.

        Returns:
            I_G: (3, 3) whole-body locked inertia at CoM
            A_omega_j: (3, n_j) centroidal angular momentum matrix for joint velocities (MuJoCo order)
            A_bar_omega: (3, n_j) locked-inertia normalized matrix I_G^{-1} * A_omega_j (MuJoCo order)
        """
        # Set configuration (base at origin, orientation identity)
        self.data.qpos[:] = 0.0
        if self.is_floating:
            self.data.qpos[2] = 0.8  # nominal height
            self.data.qpos[3] = 1.0  # quat w
            self.data.qpos[7:] = q_joints_mj
        else:
            self.data.qpos[:] = q_joints_mj

        # Forward kinematics and composite rigid body inertia
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Compute centroidal momentum matrix A(q) via unit-velocity evaluations
        # For floating base: nv = 6 (3 lin, 3 ang) + n_j
        # We need the angular part (momentum about whole-body CoM)
        A_omega = np.zeros((3, self.nv), dtype=np.float64)

        for col in range(self.nv):
            self.data.qvel[:] = 0.0
            self.data.qvel[col] = 1.0
            mujoco.mj_forward(self.model, self.data)
            # mj_subtreeVel must be called after mj_forward to populate
            # subtree_angmom; without it, angular momentum remains zero.
            mujoco.mj_subtreeVel(self.model, self.data)

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

        All outputs are in **URDF/Pinocchio** joint ordering when a URDF path was
        provided at construction, or in MuJoCo ordering otherwise.

        Returns:
            dict containing:
                - 'q_joints': (num_samples, n_j)
                - 'A_bar_omega': (num_samples, 3, n_j)
                - 'I_G': (num_samples, 3, 3)
        """
        rng = np.random.default_rng(seed)

        # Sample joint positions uniformly within 80% of joint limits to stay away from singularities
        # Limits are in MuJoCo order
        margin = 0.1 * (self.joint_limits_upper - self.joint_limits_lower)
        low = self.joint_limits_lower + margin
        high = self.joint_limits_upper - margin

        # Samples are generated in MuJoCo order (limits are in MuJoCo order)
        q_samples_mj = rng.uniform(low, high, size=(num_samples, self.num_mj_joints))

        # Zero out fixed joints
        if self.mj_fixed_indices:
            q_samples_mj[:, self.mj_fixed_indices] = 0.0

        A_bar_samples = np.zeros((num_samples, 3, self.num_mj_joints), dtype=np.float32)
        I_G_samples = np.zeros((num_samples, 3, 3), dtype=np.float32)

        for i in range(num_samples):
            I_G, _, A_bar_mj = self.compute_centroidal_matrices(q_samples_mj[i])
            A_bar_samples[i] = A_bar_mj.astype(np.float32)
            I_G_samples[i] = I_G.astype(np.float32)

        # Reorder from MuJoCo to URDF/Pinocchio joint ordering so that the
        # trained SIREN network is consistent with the C++ runtime, which
        # indexes joints via Pinocchio (URDF order).
        if self.joint_perm is not None:
            q_samples = q_samples_mj[:, self.joint_perm]  # gather: MuJoCo → URDF
            A_bar_samples = A_bar_samples[:, :, self.joint_perm]  # reorder columns
        else:
            q_samples = q_samples_mj

        # Drop fixed joints from the dataset
        if self.fixed_joints:
            active_indices = [
                i
                for i, name in enumerate(self.urdf_joint_names)
                if name not in self.fixed_joints
            ]
            q_samples = q_samples[:, active_indices]
            A_bar_samples = A_bar_samples[:, :, active_indices]

        return {
            "q_joints": q_samples.astype(np.float32),
            "A_bar_omega": A_bar_samples,
            "I_G": I_G_samples,
        }
