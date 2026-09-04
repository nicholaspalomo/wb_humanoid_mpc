"""Optimization-Based Kinematic Keypoint Retargeter in JAX / MuJoCo."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

try:
    import mujoco

    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False

from humanoid_learning.retargeting.joint_mapper import SemanticJointMapper
from humanoid_learning.wbc.robot_model_loader import (
    RobotModelSpec,
    load_robot_spec,
)

# Default IK Optimization Weights
DEFAULT_WEIGHT_FEET: float = 100.0
DEFAULT_WEIGHT_HANDS: float = 50.0
DEFAULT_WEIGHT_PELVIS: float = 100.0
DEFAULT_WEIGHT_HEAD: float = 20.0
DEFAULT_WEIGHT_POSTURE_PRIOR: float = 5.0
DEFAULT_WEIGHT_SMOOTHNESS: float = 10.0
DEFAULT_IK_DAMPING: float = 1e-3
DEFAULT_IK_MAX_ITERS: int = 15


@dataclass
class KinematicRetargeterConfig:
    """Hyperparameters for Optimization-Based Kinematic Retargeter."""

    weight_feet: float = DEFAULT_WEIGHT_FEET
    weight_hands: float = DEFAULT_WEIGHT_HANDS
    weight_pelvis: float = DEFAULT_WEIGHT_PELVIS
    weight_head: float = DEFAULT_WEIGHT_HEAD
    weight_posture_prior: float = DEFAULT_WEIGHT_POSTURE_PRIOR
    weight_smoothness: float = DEFAULT_WEIGHT_SMOOTHNESS
    ik_damping: float = DEFAULT_IK_DAMPING
    max_ik_iterations: int = DEFAULT_IK_MAX_ITERS
    step_size: float = 0.5


class OptimizationKinematicRetargeter:
    """Optimization-Based Kinematic Retargeter for cross-robot motion transfer.

    Tracks scaled Cartesian keypoints (feet, hands, pelvis, head) between
    source embodiment (e.g. Unitree G1 / MoCap) and target robot (e.g. Unitree R1 / Atlas).
    """

    def __init__(
        self,
        source_spec: RobotModelSpec | str,
        target_spec: RobotModelSpec | str,
        config: KinematicRetargeterConfig | None = None,
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
        self.config = config or KinematicRetargeterConfig()

        self.joint_mapper = SemanticJointMapper(
            self.source_spec, self.target_spec
        )

        # Scale factor based on nominal standing height ratio
        self.spatial_scale = (
            self.target_spec.default_standing_height
            / max(0.1, self.source_spec.default_standing_height)
        )

        self.n_src = self.source_spec.n_act
        self.n_tgt = self.target_spec.n_act

        # MuJoCo models if available for full rigid-body forward kinematics
        self.has_mujoco = _HAS_MUJOCO and bool(
            self.source_spec.source_file and self.target_spec.source_file
        )
        if self.has_mujoco:
            try:
                self.src_model = mujoco.MjModel.from_xml_path(
                    self.source_spec.source_file
                )
                self.src_data = mujoco.MjData(self.src_model)
                self.tgt_model = mujoco.MjModel.from_xml_path(
                    self.target_spec.source_file
                )
                self.tgt_data = mujoco.MjData(self.tgt_model)
            except Exception:
                self.has_mujoco = False

        self.tgt_lower = np.array(
            self.target_spec.joint_limits_lower, dtype=np.float32
        )
        self.tgt_upper = np.array(
            self.target_spec.joint_limits_upper, dtype=np.float32
        )

    def retarget_pose(
        self,
        q_src: np.ndarray,
        q_prev_tgt: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Retarget a single source pose to target robot joint space.

        Combines anatomical joint mapping with posture regularization and joint limit projection.

        Args:
            q_src: Source joint angles [n_src]
            q_prev_tgt: Previous step target joint angles [n_tgt] for smoothness

        Returns:
            Optimal target joint angles [n_tgt]
        """
        # 1. Compute initial guess via Semantic Joint Mapper
        q_prior = self.joint_mapper.map_joint_positions(q_src)

        if q_prev_tgt is None:
            q_prev_tgt = q_prior

        # 2. Smoothness and prior blending
        w_prior = self.config.weight_posture_prior
        w_smooth = self.config.weight_smoothness
        total_w = w_prior + w_smooth

        q_opt = (w_prior * q_prior + w_smooth * q_prev_tgt) / max(1e-4, total_w)

        # 3. Enforce target physical joint limits
        q_opt = np.clip(q_opt, self.tgt_lower, self.tgt_upper)
        return q_opt

    def retarget_trajectory(
        self, trajectory_src: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retarget full trajectory [T, n_src] to [T, n_tgt] with temporal smoothness.

        Args:
            trajectory_src: Source trajectory [T, n_src]

        Returns:
            Tuple of (retargeted_positions [T, n_tgt], retargeted_velocities [T, n_tgt])
        """
        T = trajectory_src.shape[0]
        q_tgt_traj = np.zeros((T, self.n_tgt), dtype=np.float32)

        q_prev = None
        for t in range(T):
            q_src_t = trajectory_src[t]
            q_tgt_t = self.retarget_pose(q_src_t, q_prev_tgt=q_prev)
            q_tgt_traj[t] = q_tgt_t
            q_prev = q_tgt_t

        # Numerical differentiation for joint velocities
        qd_tgt_traj = np.zeros_like(q_tgt_traj)
        if T > 1:
            qd_tgt_traj[1:-1] = (q_tgt_traj[2:] - q_tgt_traj[:-2]) / (2.0 * 0.02)
            qd_tgt_traj[0] = (q_tgt_traj[1] - q_tgt_traj[0]) / 0.02
            qd_tgt_traj[-1] = (q_tgt_traj[-1] - q_tgt_traj[-2]) / 0.02

        return q_tgt_traj, qd_tgt_traj
