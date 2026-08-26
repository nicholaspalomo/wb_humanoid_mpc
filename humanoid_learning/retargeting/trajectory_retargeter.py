"""Offline Trajectory Dataset Retargeting Pipeline across Robot Embodiments."""

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from humanoid_learning.retargeting.joint_mapper import SemanticJointMapper
from humanoid_learning.retargeting.kinematic_retargeter import (
    KinematicRetargeterConfig,
    OptimizationKinematicRetargeter,
)
from humanoid_learning.wbc.robot_model_loader import (
    RobotModelSpec,
    load_robot_spec,
)


@dataclass
class DatasetRetargetingConfig:
    """Configuration for dataset retargeting pipeline."""

    method: str = "anatomical"  # 'anatomical' or 'optimization'
    smooth_velocities: bool = True
    action_scale: float = 0.25


class TrajectoryDatasetRetargeter:
    """Converts entire offline demonstration datasets between different robot embodiments."""

    def __init__(
        self,
        source_robot: str | RobotModelSpec,
        target_robot: str | RobotModelSpec,
        config: DatasetRetargetingConfig | None = None,
    ):
        self.source_spec = (
            source_robot
            if isinstance(source_robot, RobotModelSpec)
            else load_robot_spec(source_robot)
        )
        self.target_spec = (
            target_robot
            if isinstance(target_robot, RobotModelSpec)
            else load_robot_spec(target_robot)
        )
        self.config = config or DatasetRetargetingConfig()

        self.joint_mapper = SemanticJointMapper(
            self.source_spec, self.target_spec
        )
        self.opt_retargeter = OptimizationKinematicRetargeter(
            self.source_spec, self.target_spec
        )

        self.src_obs_dim = (
            self.source_spec.n_act * 2
            + self.source_spec.num_floating_base_pos
            + 2
            + self.source_spec.num_floating_base_vel
            + 3
        )
        self.tgt_obs_dim = (
            self.target_spec.n_act * 2
            + self.target_spec.num_floating_base_pos
            + 2
            + self.target_spec.num_floating_base_vel
            + 3
        )

    def retarget_dataset(
        self, input_path: str, output_path: str
    ) -> Dict[str, Any]:
        """Load source dataset, retarget transitions to target robot space, and save."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input dataset '{input_path}' not found.")

        data = np.load(input_path)
        obs_src = np.array(data["observations"], dtype=np.float32)
        act_src = np.array(data["actions"], dtype=np.float32)
        rewards = (
            np.array(data["rewards"], dtype=np.float32)
            if "rewards" in data
            else np.ones((len(obs_src),), dtype=np.float32)
        )

        N = len(obs_src)
        print(f"🔄 Retargeting {N} transitions from {self.source_spec.name} to {self.target_spec.name}...")

        # Retarget actions (joint residuals)
        if self.config.method == "optimization":
            act_tgt, _ = self.opt_retargeter.retarget_trajectory(act_src)
        else:
            act_tgt = self.joint_mapper.map_joint_positions(act_src)

        # Retarget observation vectors to target dimension
        obs_tgt = np.zeros((N, self.tgt_obs_dim), dtype=np.float32)

        # 1. Joint positions
        src_q_rel = obs_src[:, : self.source_spec.n_act]
        tgt_q_rel = self.joint_mapper.map_joint_positions(src_q_rel)
        obs_tgt[:, : self.target_spec.n_act] = tgt_q_rel

        # 2. Joint velocities
        src_qd = obs_src[
            :, self.source_spec.n_act : 2 * self.source_spec.n_act
        ]
        tgt_qd = self.joint_mapper.map_joint_positions(src_qd)
        obs_tgt[
            :, self.target_spec.n_act : 2 * self.target_spec.n_act
        ] = tgt_qd

        # 3. Base pos / height
        base_pos_start = 2 * self.source_spec.n_act
        base_pos_end = base_pos_start + self.source_spec.num_floating_base_pos
        tgt_base_pos_start = 2 * self.target_spec.n_act
        tgt_base_pos_end = (
            tgt_base_pos_start + self.target_spec.num_floating_base_pos
        )

        if self.source_spec.num_floating_base_pos == self.target_spec.num_floating_base_pos:
            obs_tgt[:, tgt_base_pos_start:tgt_base_pos_end] = obs_src[
                :, base_pos_start:base_pos_end
            ]
            # Scale base height by standing height ratio
            h_scale = (
                self.target_spec.default_standing_height
                / max(0.1, self.source_spec.default_standing_height)
            )
            obs_tgt[:, tgt_base_pos_start + 2] *= h_scale

        # 4. Remaining features (phase [2], base vel [6], cmd vel [3])
        obs_tgt[:, tgt_base_pos_end:] = obs_src[:, base_pos_end:]

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.savez_compressed(
            output_path,
            observations=obs_tgt,
            actions=act_tgt,
            rewards=rewards,
        )
        print(f"✓ Saved retargeted dataset to '{output_path}'.")

        return {
            "num_samples": N,
            "source_robot": self.source_spec.name,
            "target_robot": self.target_spec.name,
            "obs_dim": self.tgt_obs_dim,
            "act_dim": self.target_spec.n_act,
            "output_path": output_path,
        }
