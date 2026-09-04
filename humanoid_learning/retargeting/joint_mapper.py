"""Semantic Anatomical Joint Mapping and DOF Reduction for Cross-Robot Retargeting."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from humanoid_learning.wbc.robot_model_loader import (
    RobotModelSpec,
    load_robot_spec,
)

# Standard anatomical joint semantic categories
ANATOMICAL_CATEGORIES = [
    "left_leg",
    "right_leg",
    "torso",
    "left_arm",
    "right_arm",
    "head",
]


@dataclass
class JointMappingConfig:
    """Configuration for cross-robot joint retargeting."""

    clamp_to_limits: bool = True
    scale_range: bool = False  # Scale relative to joint limit percentage if True
    fill_unmapped_with_nominal: bool = True


class SemanticJointMapper:
    """Direct Anatomical Joint Mapper between source and target robot embodiments.

    Handles DOF reduction (e.g. 7-DOF arm to 5-DOF arm, 3-DOF waist to 2-DOF waist),
    joint axis alignments, sign flips, and joint limit clamping.
    """

    def __init__(
        self,
        source_spec: RobotModelSpec | str,
        target_spec: RobotModelSpec | str,
        config: JointMappingConfig | None = None,
    ):
        self.source_spec = (
            source_spec
            if isinstance(source_spec, RobotModelSpec)
            else load_robot_spec(source_spec)
        )
        self.target_spec = (
            target_spec
            if isinstance(target_spec, RobotModelSpec)
            else load_robot_spec(target_spec)
        )
        self.config = config or JointMappingConfig()

        self.n_src = self.source_spec.n_act
        self.n_tgt = self.target_spec.n_act

        # Build anatomical mapping index matrices
        self._build_mapping_tables()

    # LINT.IfChange(robot_limb_discovery)
    def _build_mapping_tables(self):
        """Construct matching index pairs between source and target joints."""
        self.mapping_pairs: List[Tuple[int, int, float]] = []  # (src_idx, tgt_idx, sign/scale)
        self.target_is_mapped = [False] * self.n_tgt

        # 1. Map legs (left and right)
        for limb in ["left_leg", "right_leg"]:
            src_joints = self.source_spec.limb_joint_indices.get(limb, [])
            tgt_joints = self.target_spec.limb_joint_indices.get(limb, [])
            # Map up to min length (typically 6 joints for humanoid legs)
            for i in range(min(len(src_joints), len(tgt_joints))):
                s_idx = src_joints[i]
                t_idx = tgt_joints[i]
                self.mapping_pairs.append((s_idx, t_idx, 1.0))
                self.target_is_mapped[t_idx] = True

        # 2. Map torso / waist
        src_torso = self.source_spec.limb_joint_indices.get("torso", [])
        tgt_torso = self.target_spec.limb_joint_indices.get("torso", [])

        # Match by name keywords if lengths differ (e.g. G1 3-DOF pitch/roll/yaw vs R1 2-DOF roll/yaw)
        for t_idx in tgt_torso:
            t_name = self.target_spec.joint_names[t_idx].lower()
            matched = False
            for s_idx in src_torso:
                s_name = self.source_spec.joint_names[s_idx].lower()
                if (
                    ("yaw" in t_name and "yaw" in s_name)
                    or ("roll" in t_name and "roll" in s_name)
                    or ("pitch" in t_name and "pitch" in s_name)
                    or ("bkz" in t_name and "yaw" in s_name)
                    or ("bkx" in t_name and "roll" in s_name)
                    or ("bky" in t_name and "pitch" in s_name)
                ):
                    self.mapping_pairs.append((s_idx, t_idx, 1.0))
                    self.target_is_mapped[t_idx] = True
                    matched = True
                    break
            if not matched and src_torso:
                # Default map to first torso joint
                self.mapping_pairs.append((src_torso[0], t_idx, 1.0))
                self.target_is_mapped[t_idx] = True

        # 3. Map arms (left and right)
        for arm in ["left_arm", "right_arm"]:
            src_arm = self.source_spec.limb_joint_indices.get(arm, [])
            tgt_arm = self.target_spec.limb_joint_indices.get(arm, [])

            # Map shoulder and elbow joints directly by sequence or keywords
            for t_idx in tgt_arm:
                t_name = self.target_spec.joint_names[t_idx].lower()
                matched = False
                for s_idx in src_arm:
                    s_name = self.source_spec.joint_names[s_idx].lower()
                    # Shoulder matches
                    if "shoulder" in t_name and "shoulder" in s_name:
                        if (
                            ("pitch" in t_name and "pitch" in s_name)
                            or ("roll" in t_name and "roll" in s_name)
                            or ("yaw" in t_name and "yaw" in s_name)
                            or ("shz" in t_name and "yaw" in s_name)
                            or ("shx" in t_name and "roll" in s_name)
                        ):
                            self.mapping_pairs.append((s_idx, t_idx, 1.0))
                            self.target_is_mapped[t_idx] = True
                            matched = True
                            break
                    # Elbow matches
                    elif "elbow" in t_name and "elbow" in s_name:
                        self.mapping_pairs.append((s_idx, t_idx, 1.0))
                        self.target_is_mapped[t_idx] = True
                        matched = True
                        break
                    # Wrist roll matches
                    elif "wrist_roll" in t_name and "wrist_roll" in s_name:
                        self.mapping_pairs.append((s_idx, t_idx, 1.0))
                        self.target_is_mapped[t_idx] = True
                        matched = True
                        break

                if not matched:
                    # Sequential fallback
                    arm_t_pos = tgt_arm.index(t_idx)
                    if arm_t_pos < len(src_arm):
                        self.mapping_pairs.append((src_arm[arm_t_pos], t_idx, 1.0))
                        self.target_is_mapped[t_idx] = True

        # Target lower and upper bounds
        self.tgt_lower = np.array(self.target_spec.joint_limits_lower, dtype=np.float32)
        self.tgt_upper = np.array(self.target_spec.joint_limits_upper, dtype=np.float32)
        self.tgt_nominal = np.array(self.target_spec.q_nominal, dtype=np.float32)
    # LINT.ThenChange(//humanoid_learning/wbc/robot_model_loader.py:robot_limb_discovery)

    def map_joint_positions(self, q_src: np.ndarray | jax.Array) -> np.ndarray:
        """Map single joint position vector or batch from source to target robot.

        Args:
            q_src: Source joint angles [..., n_src]

        Returns:
            Target joint angles [..., n_tgt]
        """
        is_jax = isinstance(q_src, (jax.Array, jnp.ndarray))
        if is_jax:
            q_src_np = np.array(q_src)
        else:
            q_src_np = np.asarray(q_src)

        orig_shape = q_src_np.shape
        flat_src = q_src_np.reshape(-1, self.n_src)
        batch_size = flat_src.shape[0]

        # Initialize with nominal target posture
        flat_tgt = np.tile(self.tgt_nominal, (batch_size, 1))

        # Apply mapped joint pairs
        for s_idx, t_idx, scale in self.mapping_pairs:
            flat_tgt[:, t_idx] = flat_src[:, s_idx] * scale

        # Enforce target joint limits
        if self.config.clamp_to_limits:
            flat_tgt = np.clip(flat_tgt, self.tgt_lower, self.tgt_upper)

        target_shape = orig_shape[:-1] + (self.n_tgt,)
        result = flat_tgt.reshape(target_shape)

        if is_jax:
            return jnp.array(result, dtype=jnp.float32)
        return result

    def map_trajectory(
        self, trajectory_src: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map complete trajectory [T, n_src] and compute target velocities [T, n_tgt].

        Args:
            trajectory_src: Source trajectory positions [T, n_src]

        Returns:
            Tuple of (target_positions [T, n_tgt], target_velocities [T, n_tgt])
        """
        T = trajectory_src.shape[0]
        q_tgt = self.map_joint_positions(trajectory_src)

        # Numerical differentiation for joint velocities
        qd_tgt = np.zeros_like(q_tgt)
        if T > 1:
            qd_tgt[1:-1] = (q_tgt[2:] - q_tgt[:-2]) / (2.0 * 0.02)
            qd_tgt[0] = (q_tgt[1] - q_tgt[0]) / 0.02
            qd_tgt[-1] = (q_tgt[-1] - q_tgt[-2]) / 0.02

        return q_tgt, qd_tgt
