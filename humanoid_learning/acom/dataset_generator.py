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

import collections
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

# Joint types that carry no degree of freedom and are therefore absent from both
# the Pinocchio and the MuJoCo joint vectors.
_URDF_DOF_FREE_JOINT_TYPES = frozenset(("fixed", "floating"))

# MuJoCo body index of the robot's root link. Body 0 is the world body.
_ROOT_BODY_ID = 1

# Base height [m] at which configurations are sampled. The centroidal momentum
# matrix does not depend on it; it only keeps the model clear of the ground plane.
_NOMINAL_BASE_HEIGHT = 0.8

# Largest locked-inertia condition number that still yields a trustworthy solve.
# Real humanoids stay below 10; this leaves a wide margin before rejecting.
_MAX_INERTIA_COND = 1.0e6

# Half-width [rad] of the range a joint WITHOUT position limits is sampled over.
# The C++ acceptance test samples such a joint over the same range, so both sides
# grade the network on one distribution. No shipped robot has such a joint.
# LINT.IfChange(unlimited_joint_range)
_UNLIMITED_JOINT_HALF_RANGE = np.pi
# LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/test/testAcomAngularVelocityConsistency.cpp:acom_unlimited_joint_range)


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


def _parse_pinocchio_joint_order(urdf_path: str) -> List[str]:
    """Returns the non-fixed URDF joint names in Pinocchio's joint ordering.

    Pinocchio's URDF parser numbers joints by a pre-order depth-first traversal
    of the kinematic tree starting at the root link. It walks each link's
    `child_links`, which urdfdom fills by iterating its `std::map` of joints, so
    the children of a link are visited sorted by the NAME of the joint leading
    to them - not in the order the URDF lists them. That ordering is what
    `pinocchio::Model::names` exposes, and therefore what the C++ MPC feeds to
    the aCOM network. It is neither the order in which `<joint>` elements appear
    in the URDF document (the DRC Atlas URDF, for one, lists them
    alphabetically, which scrambles the tree) nor a tree walk in document order
    (the Unitree R1 URDF lists its arms before its head, while Pinocchio visits
    `head_*` first). testAcomAngularVelocityConsistency.cpp checks the rule
    against Pinocchio itself on the fixture test_acom.py uses.

    Args:
        urdf_path: Path to the URDF file.

    Returns:
        Names of all joints carrying at least one degree of freedom, ordered as
        Pinocchio orders them.

    Raises:
        ValueError: If the URDF does not describe a single-rooted kinematic tree.
    """
    root = ET.parse(urdf_path).getroot()
    link_names = {link.attrib["name"] for link in root.findall("link")}

    children = collections.defaultdict(list)
    child_links = set()
    for joint in root.findall("joint"):
        parent_link = joint.find("parent").attrib["link"]
        child_link = joint.find("child").attrib["link"]
        children[parent_link].append(
            (joint.attrib["name"], joint.attrib.get("type"), child_link)
        )
        child_links.add(child_link)

    root_links = sorted(link_names - child_links)
    if len(root_links) != 1:
        raise ValueError(
            f"Expected exactly one root link in {urdf_path}, found {root_links}."
        )

    # Pre-order depth-first walk: a joint is emitted when it is reached, and its
    # whole subtree is emitted before any of its siblings. Siblings are visited
    # sorted by joint name, as urdfdom's std::map of joints orders them; they are
    # pushed in reverse so that they come off the stack in that order.
    joint_order: List[str] = []
    stack = [(None, root_links[0])]
    visited = set()
    while stack:
        joint_type_and_name, link = stack.pop()
        if link in visited:
            raise ValueError(f"URDF {urdf_path} contains a kinematic loop at '{link}'.")
        visited.add(link)
        if joint_type_and_name is not None:
            name, joint_type = joint_type_and_name
            if joint_type not in _URDF_DOF_FREE_JOINT_TYPES:
                joint_order.append(name)
        for name, joint_type, child_link in reversed(sorted(children[link])):
            stack.append(((name, joint_type), child_link))
    return joint_order


class AcomDatasetGenerator:
    """Generates training datasets for aCOM from MuJoCo robot models.

    Args:
        xml_path: Path to MuJoCo XML model file.
        urdf_path: Optional path to URDF file. When provided, a joint permutation
            is computed so that all dataset outputs (q_joints, A_bar_omega) use
            Pinocchio joint ordering instead of MuJoCo joint ordering. This is
            critical for C++ runtime parity because the MPC indexes joints via
            Pinocchio.
        fixed_joints: Joint names to hold at zero and drop from the dataset. Must
            match the `fixedJointNames` list in the robot's MPC task.yaml, since
            the C++ runtime evaluates the network on the reduced joint vector;
            train_main.make_generator reads them from there.

    Raises:
        ValueError: If the model is not a floating-base model of single-DoF
            joints, if the URDF and the MJCF name different joints, or if
            `fixed_joints` names a joint the model does not have.
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

        # The aCOM decomposition theta_acom = theta_base + Delta_theta(q_j) is only
        # defined for a floating base, so a fixed-base model is rejected outright
        # rather than silently yielding un-normalized momentum as the target.
        free_joints = [
            j
            for j in range(self.model.njnt)
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if free_joints != [0]:
            raise ValueError(
                "The aCOM dataset generator requires a floating-base model whose "
                f"first joint is the free joint; found free joints at {free_joints}."
            )
        # The momentum readout below uses the root link's subtree, which is the
        # whole robot only if the free joint really does attach body _ROOT_BODY_ID
        # directly to the world.
        if self.model.jnt_bodyid[0] != _ROOT_BODY_ID:
            raise ValueError(
                f"Expected the free joint to belong to body {_ROOT_BODY_ID}, but it "
                f"belongs to body {self.model.jnt_bodyid[0]}."
            )
        # A MuJoCo free joint spans 7 qpos entries (position and quaternion) and 6
        # qvel entries.
        self.base_qpos_dim = 7
        self.base_qvel_dim = 6
        self.num_mj_joints = self.nv - self.base_qvel_dim
        if self.nq - self.base_qpos_dim != self.num_mj_joints:
            raise ValueError(
                f"Model has {self.nq - self.base_qpos_dim} internal qpos entries "
                f"but {self.num_mj_joints} internal velocity DoFs. The aCOM dataset "
                "generator only supports single-DoF internal joints."
            )

        # Collect MuJoCo internal joint names, skipping the free base joint.
        self.mj_joint_names: List[str] = []
        for j in range(1, self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            self.mj_joint_names.append(name)

        # Every downstream index computation treats a joint index and a velocity
        # index as interchangeable, which only holds for single-DoF joints. Ball
        # and free joints would silently shift every column of the dataset.
        if len(self.mj_joint_names) != self.num_mj_joints:
            raise ValueError(
                f"Model has {len(self.mj_joint_names)} internal joints but "
                f"{self.num_mj_joints} internal velocity DoFs. The aCOM dataset "
                "generator only supports single-DoF internal joints."
            )

        self.fixed_joints = set(fixed_joints or [])
        # A misspelt fixed joint would otherwise be dropped silently, leave the
        # joint it meant in the dataset, and only surface much later as an
        # input_dim mismatch in the C++ MPC.
        unknown_fixed = sorted(self.fixed_joints - set(self.mj_joint_names))
        if unknown_fixed:
            raise ValueError(
                f"fixed_joints names joints the model does not have: {unknown_fixed}. "
                f"Its joints are: {self.mj_joint_names}."
            )
        self.mj_fixed_indices = sorted(
            self.mj_joint_names.index(name) for name in self.fixed_joints
        )

        # Joint limits for the internal joints, in MuJoCo order.
        lower = []
        upper = []
        for j in range(1, self.model.njnt):
            if self.model.jnt_limited[j]:
                lower.append(self.model.jnt_range[j, 0])
                upper.append(self.model.jnt_range[j, 1])
            else:
                lower.append(-_UNLIMITED_JOINT_HALF_RANGE)
                upper.append(_UNLIMITED_JOINT_HALF_RANGE)

        self.joint_limits_lower = np.array(lower, dtype=np.float64)
        self.joint_limits_upper = np.array(upper, dtype=np.float64)

        # Permutation from Pinocchio to MuJoCo joint ordering, so that the dataset
        # is expressed in the ordering the C++ MPC uses at runtime.
        #   perm[i] = MuJoCo index of Pinocchio's i-th joint.
        #   MuJoCo -> Pinocchio (gather):  q_pin = q_mj[perm]
        #   Pinocchio -> MuJoCo (scatter): q_mj[perm] = q_pin
        # LINT.IfChange(joint_permutation)
        if urdf_path is not None:
            resolved_urdf = resolve_xml_path(urdf_path)
            (
                self.pinocchio_joint_names,
                self.joint_perm,
            ) = self._compute_joint_permutation(resolved_urdf)
        else:
            self.pinocchio_joint_names = self.mj_joint_names
            self.joint_perm = None

        self.active_joint_names = [
            name for name in self.pinocchio_joint_names if name not in self.fixed_joints
        ]
        self.num_active_joints = len(self.active_joint_names)
        # LINT.ThenChange(//humanoid_learning/acom/tests/test_acom.py:joint_permutation_test)

    def _to_dataset_order(self, per_mj_joint: np.ndarray) -> np.ndarray:
        """Reorders a per-joint array from MuJoCo order into the dataset's.

        The dataset's joint axis is Pinocchio's order when a URDF was given (and
        MuJoCo's otherwise), without the fixed joints; `active_joint_names` names
        its entries.
        """
        ordered = (
            per_mj_joint[..., self.joint_perm]
            if self.joint_perm is not None
            else per_mj_joint
        )
        active = [
            i
            for i, name in enumerate(self.pinocchio_joint_names)
            if name not in self.fixed_joints
        ]
        return ordered[..., active]

    def sampling_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """The box `generate_dataset` samples the active joints from.

        It is the joint-limit box itself, untrimmed, so every posture the MPC can
        reach within its limits - the robot's nominal stance included - is inside
        the training distribution, and the C++ acceptance test, which samples the
        URDF limits, grades the network on the distribution it was fit to.

        Returns:
            (low, high), each of shape (num_active_joints,), in the dataset's
            joint order (see `active_joint_names`).
        """
        low_mj, high_mj = self._sampling_bounds_mj()
        return self._to_dataset_order(low_mj), self._to_dataset_order(high_mj)

    def _sampling_bounds_mj(self) -> Tuple[np.ndarray, np.ndarray]:
        """The sampling box in MuJoCo order, including the fixed joints' ranges."""
        return self.joint_limits_lower, self.joint_limits_upper

    def _compute_joint_permutation(
        self, urdf_path: str
    ) -> Tuple[List[str], np.ndarray]:
        """Computes the index permutation from Pinocchio to MuJoCo joint ordering.

        Returns:
            pinocchio_joint_names: non-fixed joint names in Pinocchio ordering.
            perm: np.ndarray where perm[i] = MuJoCo joint index of Pinocchio's i-th
                joint.
        """
        pinocchio_joint_names = _parse_pinocchio_joint_order(urdf_path)

        # Validate that every URDF joint exists in MuJoCo (and vice-versa).
        mj_set = set(self.mj_joint_names)
        urdf_set = set(pinocchio_joint_names)
        missing_in_mj = urdf_set - mj_set
        missing_in_urdf = mj_set - urdf_set
        if missing_in_mj:
            raise ValueError(f"URDF joints not found in MuJoCo model: {missing_in_mj}")
        if missing_in_urdf:
            raise ValueError(
                f"MuJoCo joints not found in URDF model: {missing_in_urdf}"
            )

        perm = np.array(
            [self.mj_joint_names.index(name) for name in pinocchio_joint_names],
            dtype=np.int64,
        )
        return pinocchio_joint_names, perm

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
        # Place the base at a nominal height with identity orientation.
        #
        # The identity orientation is LOAD-BEARING, not cosmetic. A_bar_omega is
        # exactly invariant to base TRANSLATION - angular momentum about the CoM
        # does not see it - which is why the nominal height below is free. It is
        # NOT invariant to base ROTATION: under a base rotation R the locked
        # inertia becomes R I_G R^T and the joint block becomes R A_omega_j, so
        # A_bar_omega transforms as R A_bar_omega.
        #
        # Sampling at the identity is therefore what makes Delta_theta a pure
        # function of the joint angles, which is exactly the premise the additive
        # decomposition theta_acom = theta_base + Delta_theta(q_j) rests on: the
        # network is fitted to the connection expressed in the BASE frame, and
        # the C++ runtime adds it to the base Euler angles on that understanding.
        # Sampling a spread of base orientations here would not enrich the
        # dataset, it would make the target multivalued in q_j and the fit would
        # collapse towards the rotational average.
        self.data.qpos[:] = 0.0
        self.data.qpos[2] = _NOMINAL_BASE_HEIGHT
        self.data.qpos[3] = 1.0  # Quaternion w component.
        self.data.qpos[self.base_qpos_dim :] = q_joints_mj

        # Position-level kinematics depend only on qpos, so they are computed once
        # here and reused for every unit-velocity column below.
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Build the angular block of the centroidal momentum matrix A(q) column by
        # column, by evaluating the momentum induced by a unit velocity along each
        # generalized coordinate. For a floating base, nv = 3 linear + 3 angular +
        # n_j joint DoFs.
        A_omega = np.zeros((3, self.nv), dtype=np.float64)
        for col in range(self.nv):
            self.data.qvel[:] = 0.0
            self.data.qvel[col] = 1.0
            # mj_comVel propagates the velocity-level subtree quantities that
            # mj_subtreeVel needs. Without it subtree_angmom stays at its stale
            # value, which previously made every target identically zero.
            mujoco.mj_comVel(self.model, self.data)
            mujoco.mj_subtreeVel(self.model, self.data)

            # subtree_angmom is expressed about the subtree CoM. Index _ROOT_BODY_ID
            # is the robot's root link, so its subtree is the whole robot and the
            # momentum is about the whole-body CoM. Body 0 is MuJoCo's world body;
            # using it would silently fold in any other top-level body such as
            # terrain or a manipulated object.
            A_omega[:, col] = self.data.subtree_angmom[_ROOT_BODY_ID]

        # Columns 3..5 hold the momentum induced by unit base angular velocity with
        # the joints locked, which is by definition the whole-body locked inertia.
        # Columns 6.. hold the joint contribution. Columns 0..2 are identically
        # zero because angular momentum about the CoM is invariant to translation.
        I_G = A_omega[:, 3:6]
        A_omega_j = A_omega[:, 6:]

        # I_G is symmetric positive definite and very well conditioned for any real
        # humanoid, so a direct solve is both faster and safer than a pseudo-inverse,
        # which would silently discard a rotational axis if the model degenerated.
        condition_number = np.linalg.cond(I_G)
        if not np.isfinite(condition_number) or condition_number > _MAX_INERTIA_COND:
            raise ValueError(
                f"Locked inertia is ill-conditioned (cond = {condition_number:.3e}); "
                "the normalized centroidal momentum matrix would be unreliable."
            )
        A_bar_omega = np.linalg.solve(I_G, A_omega_j)

        return I_G, A_omega_j, A_bar_omega

    def generate_dataset(
        self, num_samples: int = 10000, seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """Samples random joint configurations and generates a training dataset.

        The joint axis of every output is in Pinocchio ordering when a URDF path
        was provided at construction, or in MuJoCo ordering otherwise, and
        excludes the fixed joints.

        Args:
            num_samples: Number of configurations to sample.
            seed: Seed for the joint configuration sampler.

        Returns:
            A dict with 'q_joints' of shape (num_samples, n_active),
            'A_bar_omega' of shape (num_samples, 3, n_active), and 'I_G' of shape
            (num_samples, 3, 3).
        """
        rng = np.random.default_rng(seed)

        # Sample uniformly over the full joint-limit box, in MuJoCo order. There
        # used to be a margin of 10 % of each range trimmed from both ends; it
        # excluded postures the robot really adopts (G1 stands with its knees at
        # 0.1 rad, below the trimmed floor of 0.21 rad), so the network was never
        # trained where the MPC evaluates it. The box is the one sampling_bounds()
        # reports.
        low_mj, high_mj = self._sampling_bounds_mj()
        q_samples_mj = rng.uniform(
            low_mj, high_mj, size=(num_samples, self.num_mj_joints)
        )

        # The C++ runtime evaluates the network on the reduced joint vector, in
        # which the fixed joints are held at zero.
        if self.mj_fixed_indices:
            q_samples_mj[:, self.mj_fixed_indices] = 0.0

        A_bar_samples = np.zeros((num_samples, 3, self.num_mj_joints), dtype=np.float32)
        I_G_samples = np.zeros((num_samples, 3, 3), dtype=np.float32)
        for i in range(num_samples):
            I_G, _, A_bar_mj = self.compute_centroidal_matrices(q_samples_mj[i])
            A_bar_samples[i] = A_bar_mj.astype(np.float32)
            I_G_samples[i] = I_G.astype(np.float32)

        # Reorder from MuJoCo to Pinocchio joint ordering so that the trained SIREN
        # network is indexed the same way as the C++ runtime, which reads joints
        # out of the Pinocchio model, and drop the fixed joints.
        q_samples = self._to_dataset_order(q_samples_mj)
        A_bar_samples = self._to_dataset_order(A_bar_samples)

        return {
            "q_joints": q_samples.astype(np.float32),
            "A_bar_omega": A_bar_samples,
            "I_G": I_G_samples,
        }
