"""Offline Trajectory Tracking Residual WBC Environment in JAX / Brax / MJX."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp

try:
    from brax.envs.base import PipelineEnv, State

    _HAS_BRAX = True
except ImportError:
    _HAS_BRAX = False

    class State:  # type: ignore
        def __init__(self, pipeline_state, obs, reward, done, metrics):
            self.pipeline_state = pipeline_state
            self.obs = obs
            self.reward = reward
            self.done = done
            self.metrics = metrics

        def replace(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            return self

from humanoid_learning.wbc.jax_wbc import JaxWBCConfig, JaxWholeBodyController
from humanoid_learning.wbc.robot_model_loader import (
    RobotModelSpec,
    load_robot_spec,
)

# ==============================================================================
# Environment Configuration & Timing Constants
# ==============================================================================
DEFAULT_CONTROL_DT_S: float = 0.02
DEFAULT_PHYSICS_DT_S: float = 0.002
DEFAULT_ACTION_SCALE_RAD: float = 0.25
DEFAULT_TARGET_VELOCITY_MPS: float = 1.0
DEFAULT_STANDING_HEIGHT_M: float = 0.85
DEFAULT_TRAJECTORY_LENGTH: int = 100

# Observation Feature Dimensions
PHASE_FEATURE_DIM: int = 2  # [cos(2*pi*t/T), sin(2*pi*t/T)]
CMD_VELOCITY_DIM: int = 3  # [vx_cmd, vy_cmd, wz_cmd]

# Numerical Physics Limits
RESET_PERTURBATION_RANGE_RAD: float = 0.02
PHYSICS_MAX_VELOCITY_CLIP_RAD_S: float = 15.0
PHYSICS_MAX_JOINT_POS_CLIP_RAD: float = 3.14159

# Offline Trajectory Imitation & Tracking Reward Scaling
REWARD_TRAJECTORY_TRACKING_WEIGHT: float = 2.0
REWARD_TRAJECTORY_TRACKING_EXP_SCALE: float = 5.0
REWARD_VEL_TRACKING_WEIGHT: float = 1.0
REWARD_VEL_TRACKING_EXP_SCALE: float = 3.0
REWARD_UPRIGHT_WEIGHT: float = 0.5
REWARD_HEIGHT_WEIGHT: float = 0.5
REWARD_HEIGHT_EXP_SCALE: float = 20.0

# Regularization Penalties
PENALTY_RESIDUAL_WEIGHT: float = 0.05
PENALTY_TORQUE_WEIGHT: float = 0.01
PENALTY_LATERAL_VEL_WEIGHT: float = 0.2
PENALTY_ANGULAR_VEL_WEIGHT: float = 0.1

# Early Termination Thresholds
TERMINATION_MIN_HEIGHT_FACTOR: float = 0.45
TERMINATION_MAX_HEIGHT_FACTOR: float = 1.65
TERMINATION_MIN_UPRIGHT_PROJECTION: float = 0.3


@dataclass
class HumanoidResidualWBCConfig:
    """Configuration for Offline Trajectory Tracking Residual WBC Environment."""

    robot: str | RobotModelSpec | None = None  # Robot model name or RobotModelSpec
    control_dt: float = DEFAULT_CONTROL_DT_S
    physics_dt: float = DEFAULT_PHYSICS_DT_S
    action_scale: float = DEFAULT_ACTION_SCALE_RAD
    target_velocity: float = DEFAULT_TARGET_VELOCITY_MPS
    nominal_height: float | None = None  # Auto-extracted from spec if None
    n_act: int | None = None  # Auto-extracted from spec if None
    nv: int | None = None  # Auto-extracted from spec if None
    nq: int | None = None  # Auto-extracted from spec if None
    trajectory_length: int = DEFAULT_TRAJECTORY_LENGTH


class HumanoidResidualWBCEnv(PipelineEnv if _HAS_BRAX else object):
    """General Robot Environment tracking Offline Reference Trajectories (from MPC / MoCap)

    The policy outputs residual corrections Delta q on top of the offline reference trajectory:
        q_des(t) = q_ref_offline(t) + action_scale * Delta q_t
    which are tracked in real-time by the model-based JAX Whole-Body Controller (WBC).
    """

    def __init__(
        self,
        config: Optional[HumanoidResidualWBCConfig] = None,
        robot: Optional[Union[str, RobotModelSpec]] = None,
        reference_trajectories: Optional[jax.Array] = None,
        **kwargs,
    ):
        self.config = config or HumanoidResidualWBCConfig()

        # Load robot model specification
        robot_spec_arg = robot or self.config.robot
        if robot_spec_arg is not None:
            if isinstance(robot_spec_arg, RobotModelSpec):
                self.spec = robot_spec_arg
            else:
                self.spec = load_robot_spec(str(robot_spec_arg))
        else:
            self.spec = load_robot_spec("generic_robot")

        # Dynamically extract all dimensions and properties from spec
        self.n_act = int(self.spec.n_act)
        self.nq = int(self.spec.nq)
        self.nv = int(self.spec.nv)
        self.n_fb_pos = self.spec.num_floating_base_pos
        self.n_fb_vel = self.spec.num_floating_base_vel
        self.n_contacts = self.spec.num_contact_points
        self.limb_joint_indices = dict(self.spec.limb_joint_indices)

        self.nominal_qpos = self.spec.get_q_nominal_jax()
        self.nominal_height = (
            self.config.nominal_height
            if self.config.nominal_height is not None
            else float(self.spec.default_standing_height)
        )

        # Initialize general Whole-Body Controller
        self.wbc = JaxWholeBodyController(spec=self.spec, config=JaxWBCConfig())

        # Set or initialize Offline Reference Trajectories [T, n_act]
        if reference_trajectories is not None:
            self.ref_trajectories = jnp.array(
                reference_trajectories, dtype=jnp.float32
            )
            self.traj_len = self.ref_trajectories.shape[0]
        else:
            # Default: Broadcast nominal standing posture across trajectory horizon
            self.traj_len = self.config.trajectory_length
            self.ref_trajectories = jnp.tile(
                self.nominal_qpos[None, :], (self.traj_len, 1)
            )

        # Observation size:
        # Joint positions relative [n_act] + Joint velocities [n_act] +
        # Base pos [n_fb_pos] + Phase features [PHASE_FEATURE_DIM] +
        # Base vel [n_fb_vel] + Command vel [CMD_VELOCITY_DIM]
        self._obs_size = (
            self.n_act * 2
            + self.n_fb_pos
            + PHASE_FEATURE_DIM
            + self.n_fb_vel
            + CMD_VELOCITY_DIM
        )
        self._act_size = self.n_act

    @property
    def observation_size(self) -> int:
        return self._obs_size

    @property
    def action_size(self) -> int:
        return self._act_size

    def get_reference_posture(self, step_idx: jax.Array) -> jax.Array:
        """Fetch reference joint positions directly from the offline trajectory buffer."""
        idx = jnp.mod(step_idx, self.traj_len)
        return self.ref_trajectories[idx]

    def reset(self, rng: jax.Array) -> State:
        """Reset environment to initial state of the offline trajectory with small perturbation."""
        rng_q, rng_noise = jax.random.split(rng)

        # Initial reference posture from offline trajectory at step 0
        step_count = jnp.array(0, dtype=jnp.int32)
        q_ref_0 = self.get_reference_posture(step_count)

        base_pos = jnp.array([0.0, 0.0, self.nominal_height], dtype=jnp.float32)
        base_quat = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        noise = jax.random.uniform(
            rng_noise,
            (self.n_act,),
            minval=-RESET_PERTURBATION_RANGE_RAD,
            maxval=RESET_PERTURBATION_RANGE_RAD,
        )
        joints = q_ref_0 + noise
        q = jnp.concatenate([base_pos, base_quat, joints])
        qd = jnp.zeros((self.nv,), dtype=jnp.float32)

        phase = jnp.array(0.0, dtype=jnp.float32)

        pipeline_state = {
            "q": q,
            "qd": qd,
            "phase": phase,
            "step_count": step_count,
        }
        obs = self._get_obs(q, qd, step_count)
        reward = jnp.zeros((), dtype=jnp.float32)
        done = jnp.zeros((), dtype=jnp.float32)
        metrics = {
            "reward": reward,
            "forward_vel": jnp.zeros(()),
            "tracking_reward": jnp.zeros(()),
            "imitation_reward": jnp.zeros(()),
            "residual_penalty": jnp.zeros(()),
            "torque_penalty": jnp.zeros(()),
            "torso_height": jnp.array(self.nominal_height),
        }

        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )

    def step(self, state: State, action: jax.Array) -> State:
        """Simulate one step with Residual WBC tracking of the offline trajectory."""
        q = state.pipeline_state["q"]
        qd = state.pipeline_state["qd"]
        step_count = state.pipeline_state["step_count"]

        # 1. Fetch offline reference joint posture at current time step
        q_ref = self.get_reference_posture(step_count)

        # 2. Add policy residual: q_des = q_ref_offline(t) + action_scale * delta_q
        delta_q = jnp.clip(action, -1.0, 1.0)
        q_des = q_ref + self.config.action_scale * delta_q

        # 3. Solve Whole-Body Controller (WBC) for motor torques
        wbc_out = self.wbc.solve(q=q, qd=qd, q_des=q_des)
        torques = wbc_out.torques

        # 4. Integrate dynamics forward (semi-implicit Euler physics update)
        n_substeps = int(
            round(self.config.control_dt / self.config.physics_dt)
        )
        dt = self.config.physics_dt

        def physics_substep(carry, _):
            curr_q, curr_qd = carry
            curr_qddot = wbc_out.qddot
            next_qd = curr_qd + dt * curr_qddot
            next_qd = jnp.clip(
                next_qd,
                -PHYSICS_MAX_VELOCITY_CLIP_RAD_S,
                PHYSICS_MAX_VELOCITY_CLIP_RAD_S,
            )

            # Update base & joint positions dynamically
            next_base_pos = curr_q[:3] + dt * next_qd[:3]
            next_base_quat = curr_q[3:7]
            next_joints = curr_q[self.n_fb_pos :] + dt * next_qd[self.n_fb_vel :]
            next_joints = jnp.clip(
                next_joints,
                -PHYSICS_MAX_JOINT_POS_CLIP_RAD,
                PHYSICS_MAX_JOINT_POS_CLIP_RAD,
            )

            next_q = jnp.concatenate(
                [next_base_pos, next_base_quat, next_joints]
            )
            return (next_q, next_qd), None

        (next_q, next_qd), _ = jax.lax.scan(
            physics_substep, (q, qd), None, length=n_substeps
        )

        next_step_count = step_count + 1

        # 5. Evaluate Trajectory Imitation and Locomotion Rewards
        joint_actual = next_q[self.n_fb_pos :]
        tracking_error = jnp.mean(jnp.square(joint_actual - q_ref))
        r_imitation = jnp.exp(
            -REWARD_TRAJECTORY_TRACKING_EXP_SCALE * tracking_error
        )

        forward_vel = next_qd[0]
        torso_height = next_q[2]
        base_quat = next_q[3:7]

        vel_err = forward_vel - self.config.target_velocity
        r_vel = jnp.exp(
            -REWARD_VEL_TRACKING_EXP_SCALE * jnp.square(vel_err)
        )

        upright_proj = 1.0 - 2.0 * (
            base_quat[1] ** 2 + base_quat[2] ** 2
        )
        r_upright = jnp.clip(upright_proj, 0.0, 1.0)
        r_height = jnp.exp(
            -REWARD_HEIGHT_EXP_SCALE
            * jnp.square(torso_height - self.nominal_height)
        )

        p_residual = PENALTY_RESIDUAL_WEIGHT * jnp.sum(jnp.square(delta_q))
        p_torque = PENALTY_TORQUE_WEIGHT * jnp.mean(
            jnp.square(torques / self.wbc.config.tau_max)
        )
        p_lateral = PENALTY_LATERAL_VEL_WEIGHT * jnp.square(next_qd[1])
        p_ang_vel = PENALTY_ANGULAR_VEL_WEIGHT * jnp.sum(
            jnp.square(next_qd[3:6])
        )

        reward = (
            REWARD_TRAJECTORY_TRACKING_WEIGHT * r_imitation
            + REWARD_VEL_TRACKING_WEIGHT * r_vel
            + REWARD_UPRIGHT_WEIGHT * r_upright
            + REWARD_HEIGHT_WEIGHT * r_height
            - p_residual
            - p_torque
            - p_lateral
            - p_ang_vel
        )

        # Dynamic termination condition based on robot nominal operating height
        is_fallen = (
            (torso_height < TERMINATION_MIN_HEIGHT_FACTOR * self.nominal_height)
            | (
                torso_height
                > TERMINATION_MAX_HEIGHT_FACTOR * self.nominal_height
            )
            | (upright_proj < TERMINATION_MIN_UPRIGHT_PROJECTION)
        )
        done = jnp.where(is_fallen, 1.0, 0.0)

        obs = self._get_obs(next_q, next_qd, next_step_count)
        metrics = {
            "reward": reward,
            "imitation_reward": r_imitation,
            "forward_vel": forward_vel,
            "tracking_reward": r_vel,
            "residual_penalty": p_residual,
            "torque_penalty": p_torque,
            "torso_height": torso_height,
            "mean_torque": jnp.mean(jnp.abs(torques)),
        }

        phase = jnp.array(
            2.0 * jnp.pi * (next_step_count % self.traj_len) / self.traj_len,
            dtype=jnp.float32,
        )

        pipeline_state = {
            "q": next_q,
            "qd": next_qd,
            "phase": phase,
            "step_count": next_step_count,
        }

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )

    def _get_obs(
        self, q: jax.Array, qd: jax.Array, step_count: jax.Array
    ) -> jax.Array:
        """Construct full continuous state observation vector dynamically."""
        joint_pos = q[self.n_fb_pos :] - self.nominal_qpos  # [n_act]
        joint_vel = qd[self.n_fb_vel :]  # [n_act]
        base_pos_coords = q[: self.n_fb_pos]  # [n_fb_pos] (e.g. z, quat)

        # Progress / Phase feature along offline trajectory: [cos(2*pi*t/T), sin(2*pi*t/T)]
        traj_phase = (
            2.0 * jnp.pi * (step_count % self.traj_len) / self.traj_len
        )
        phase_feat = jnp.array([jnp.cos(traj_phase), jnp.sin(traj_phase)])  # [2]

        base_vel_coords = qd[: self.n_fb_vel]  # [n_fb_vel] (e.g. lin/ang vel)
        cmd_vel = jnp.array(
            [self.config.target_velocity, 0.0, 0.0], dtype=jnp.float32
        )  # [3]

        return jnp.concatenate(
            [
                joint_pos,
                joint_vel,
                base_pos_coords,
                phase_feat,
                base_vel_coords,
                cmd_vel,
            ]
        )
