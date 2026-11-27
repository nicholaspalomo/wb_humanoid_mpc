# Copyright (c) 2026, Nicholas Palomo. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Base MJX Environment for legged-robot reinforcement learning with MuJoCo Playground & Brax.

The robot's root is a MuJoCo free joint: `qpos[0:3]` is the base position in the world, `qpos[3:7]` the base orientation
as a unit quaternion (w, x, y, z), `qvel[0:3]` the base's linear velocity in the world frame and `qvel[3:6]` its angular
velocity in the base frame. The velocity command is in the heading frame, the world frame turned by the base's twist
heading psi about the world z axis (the angle of the twist-swing split, as MpcRobotModelBase::getYawBaseRtWorld()), as the
MPC's velocity command is; its yaw rate is about the world z axis.
"""

import dataclasses
import math
from typing import Any

from brax.envs import base as brax_base
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

# The free joint's qpos: the base position, then its quaternion (w, x, y, z), then the joints.
_BASE_POSITION_END_INDEX = 3
_BASE_QUATERNION_END_INDEX = 7
# The free joint's qvel: the world-frame linear velocity, then the base-frame angular velocity, then the joints.
_BASE_LINEAR_VELOCITY_END_INDEX = 3
_BASE_ANGULAR_VELOCITY_END_INDEX = 6
# The relative tolerance within which control_dt must be a whole multiple of physics_dt.
_SUBSTEP_RATIO_TOLERANCE = 1e-9
# The base height below which the robot has fallen [m].
_MIN_POS_BASE_RT_WORLD_EWRT_WORLD_Z = 0.2

_DEFAULT_XML = """
<mujoco model="minimal_hopper">
    <compiler angle="radian"/>
    <option timestep="0.002" integrator="implicitfast"/>
    <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="10 10 0.1" rgba=".9 .9 .9 1"/>
        <body name="torso" pos="0 0 1.0">
            <freejoint name="root"/>
            <geom type="sphere" size="0.1" mass="10.0" rgba="0.2 0.6 0.8 1"/>
            <body name="leg" pos="0 0 -0.2">
                <joint name="hip" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04" mass="2.0"/>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor name="hip_actuator" joint="hip" gear="50" ctrlrange="-1 1"/>
    </actuator>
</mujoco>
"""


@dataclasses.dataclass
class LeggedEnvConfig:
    """Configuration for a legged-robot MJX environment.

    Attributes:
        xml_path: The MJCF file of the robot; with xml_string unset too, a minimal one-legged model.
        xml_string: The MJCF of the robot, instead of a file.
        control_dt: The policy's period [s]; a whole multiple of physics_dt.
        physics_dt: The simulation's time step [s].
        action_scale: The control is the action times this; the actuators' ctrlrange bounds it.
        cmd_linvel_base_rt_world_ewrt_heading_x: The commanded forward velocity in the heading frame [m/s].
        cmd_linvel_base_rt_world_ewrt_heading_y: The commanded lateral velocity in the heading frame [m/s].
        cmd_angvel_base_rt_world_ewrt_world_z: The commanded yaw rate about the world z axis [rad/s].
        tracking_sigma: The squared error at which a tracking reward has fallen to 1/e [(m/s)^2, (rad/s)^2].
        yaw_rate_weight: The weight of the yaw-rate tracking reward; the two linear ones weigh 1.
        energy_weight: The weight of the squared control.
        randomize_heading: Whether a reset turns the base to a uniformly random heading.
        reset_joint_position_noise: Half-width of the uniform noise a reset adds to every joint position [rad or m].
        reset_velocity_noise: Half-width of the uniform noise a reset adds to every velocity.
    """

    xml_path: str | None = None
    xml_string: str | None = None
    control_dt: float = 0.02
    physics_dt: float = 0.002
    action_scale: float = 0.5
    cmd_linvel_base_rt_world_ewrt_heading_x: float = 1.0
    cmd_linvel_base_rt_world_ewrt_heading_y: float = 0.0
    cmd_angvel_base_rt_world_ewrt_world_z: float = 0.0
    tracking_sigma: float = 0.25
    yaw_rate_weight: float = 0.5
    energy_weight: float = 0.001
    randomize_heading: bool = True
    reset_joint_position_noise: float = 0.05
    reset_velocity_noise: float = 0.05


def substep_count(control_dt: float, physics_dt: float) -> int:
    """Returns the number of physics steps in one control step.

    Args:
        control_dt: The policy's period [s].
        physics_dt: The simulation's time step [s].

    Raises:
        ValueError: When a period is not positive and finite, or control_dt is not a whole multiple of physics_dt: the
            policy would otherwise run at another rate than the one configured.
    """
    if not (math.isfinite(control_dt) and control_dt > 0.0):
        raise ValueError(
            f"control_dt must be positive and finite, got {control_dt}"
        )
    if not (math.isfinite(physics_dt) and physics_dt > 0.0):
        raise ValueError(
            f"physics_dt must be positive and finite, got {physics_dt}"
        )
    ratio = control_dt / physics_dt
    count = round(ratio)
    if count < 1 or abs(ratio - count) > _SUBSTEP_RATIO_TOLERANCE * ratio:
        raise ValueError(
            f"control_dt {control_dt} is not a whole multiple of physics_dt"
            f" {physics_dt} (ratio {ratio})"
        )
    return count


def _normalized(coefficients_wxyz: jax.Array) -> jax.Array:
    return coefficients_wxyz / jnp.linalg.norm(coefficients_wxyz)


def compute_rotmat_base_rt_world(
    quat_base_rt_world_wxyz: jax.Array,
) -> jax.Array:
    """Returns the rotation matrix, base to world, of a quaternion (w, x, y, z); it need not be of unit norm."""
    w, x, y, z = _normalized(quat_base_rt_world_wxyz)
    return jnp.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    )


def compute_yaw_base_rt_world(quat_base_rt_world_wxyz: jax.Array) -> jax.Array:
    """Returns the base's twist heading psi [rad] in [-pi, pi]: 2 atan2(q_z, q_w), wrapped.

    The angle of the twist about the world z axis of the twist-swing split; it equals the ZYX yaw only while the base is
    level.
    """
    w, _, _, z = quat_base_rt_world_wxyz
    yaw_base_rt_world = 2.0 * jnp.arctan2(z, w)
    return jnp.arctan2(jnp.sin(yaw_base_rt_world), jnp.cos(yaw_base_rt_world))


def unit_quaternion_of_yaw(psi: jax.Array) -> jax.Array:
    """Returns the quaternion (w, x, y, z) of a rotation by psi [rad] about the world z axis."""
    return jnp.array([jnp.cos(0.5 * psi), 0.0, 0.0, jnp.sin(0.5 * psi)])


def quaternion_product(
    left_wxyz: jax.Array, right_wxyz: jax.Array
) -> jax.Array:
    """Returns the Hamilton product left * right of two quaternions (w, x, y, z): right's rotation, then left's."""
    w1, x1, y1, z1 = left_wxyz
    w2, x2, y2, z2 = right_wxyz
    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def compute_linvel_base_rt_world_ewrt_heading(
    qpos: jax.Array, qvel: jax.Array
) -> jax.Array:
    """Returns the base's linear velocity in the heading frame, R_z(-psi) v_world: forward, lateral, vertical [m/s]."""
    yaw_base_rt_world = compute_yaw_base_rt_world(
        qpos[_BASE_POSITION_END_INDEX:_BASE_QUATERNION_END_INDEX]
    )
    cos_yaw_base_rt_world = jnp.cos(yaw_base_rt_world)
    sin_yaw_base_rt_world = jnp.sin(yaw_base_rt_world)
    linvel_base_rt_world_ewrt_world = qvel[:_BASE_LINEAR_VELOCITY_END_INDEX]
    return jnp.array(
        [
            cos_yaw_base_rt_world * linvel_base_rt_world_ewrt_world[0]
            + sin_yaw_base_rt_world * linvel_base_rt_world_ewrt_world[1],
            -sin_yaw_base_rt_world * linvel_base_rt_world_ewrt_world[0]
            + cos_yaw_base_rt_world * linvel_base_rt_world_ewrt_world[1],
            linvel_base_rt_world_ewrt_world[2],
        ]
    )


def compute_angvel_base_rt_world_ewrt_world_z(
    qpos: jax.Array, qvel: jax.Array
) -> jax.Array:
    """Returns the base's angular velocity about the world z axis [rad/s]: e_z . R omega_base."""
    rotmat_base_rt_world = compute_rotmat_base_rt_world(
        qpos[_BASE_POSITION_END_INDEX:_BASE_QUATERNION_END_INDEX]
    )
    return jnp.dot(
        rotmat_base_rt_world[2],
        qvel[_BASE_LINEAR_VELOCITY_END_INDEX:_BASE_ANGULAR_VELOCITY_END_INDEX],
    )


def compute_unit_gravity_ewrt_base(qpos: jax.Array) -> jax.Array:
    """Returns the unit gravity direction in the base frame, R^T (0, 0, -1): the base's tilt, without its heading."""
    rotmat_base_rt_world = compute_rotmat_base_rt_world(
        qpos[_BASE_POSITION_END_INDEX:_BASE_QUATERNION_END_INDEX]
    )
    return -rotmat_base_rt_world[2]


class LeggedMpxEnv(brax_base.PipelineEnv):
    """Base MJX velocity-tracking environment inheriting from Brax PipelineEnv.

    The observation is translation and heading invariant: the base height, the gravity direction in the base frame, the
    joint positions, the base's linear velocity in the heading frame, its angular velocity in the base frame, the joint
    velocities and the command. The reward tracks the command in the heading frame
    (compute_linvel_base_rt_world_ewrt_heading(), compute_angvel_base_rt_world_ewrt_world_z()), less the control's energy.
    A reset draws the heading and the noise from its key, so that the same key is the same initial state.
    """

    def __init__(
        self, config: LeggedEnvConfig | None = None, **kwargs: Any
    ) -> None:
        self.config = config or LeggedEnvConfig()

        if self.config.xml_string is not None:
            mj_model = mujoco.MjModel.from_xml_string(self.config.xml_string)
        elif self.config.xml_path is not None:
            mj_model = mujoco.MjModel.from_xml_path(self.config.xml_path)
        else:
            mj_model = mujoco.MjModel.from_xml_string(_DEFAULT_XML)
        if (
            mj_model.njnt < 1
            or mj_model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
            or mj_model.jnt_qposadr[0] != 0
        ):
            raise ValueError(
                "the robot's first joint must be its root free joint"
            )

        n_substeps = substep_count(
            self.config.control_dt, self.config.physics_dt
        )
        self.mj_model = mj_model
        self.mj_model.opt.timestep = self.config.physics_dt
        mjx_model = mjx.put_model(self.mj_model)
        super().__init__(sys=mjx_model, n_frames=n_substeps, **kwargs)

    @property
    def action_size(self) -> int:
        return self.sys.nu

    @property
    def observation_size(self) -> int:
        # The base height and gravity direction (4) replace the base pose (7), and the command adds 3.
        return self.sys.nq + self.sys.nv

    def _command(self) -> jax.Array:
        return jnp.array(
            [
                self.config.cmd_linvel_base_rt_world_ewrt_heading_x,
                self.config.cmd_linvel_base_rt_world_ewrt_heading_y,
                self.config.cmd_angvel_base_rt_world_ewrt_world_z,
            ]
        )

    def reset(self, rng: jax.Array) -> brax_base.State:
        """Resets the environment state to qpos0, turned and perturbed as drawn from `rng`."""
        rng_heading, rng_joints, rng_qvel = jax.random.split(rng, 3)
        qpos = jnp.array(self.sys.qpos0)
        if self.config.randomize_heading:
            # The initial minus qpos0's heading: a turn about the world z axis adds to the twist heading.
            init_err_yaw_base_rt_world = jax.random.uniform(
                rng_heading, (), minval=-jnp.pi, maxval=jnp.pi
            )
            quat_base_rt_world_wxyz = quaternion_product(
                unit_quaternion_of_yaw(init_err_yaw_base_rt_world),
                qpos[_BASE_POSITION_END_INDEX:_BASE_QUATERNION_END_INDEX],
            )
            qpos = qpos.at[
                _BASE_POSITION_END_INDEX:_BASE_QUATERNION_END_INDEX
            ].set(quat_base_rt_world_wxyz)
        noise = self.config.reset_joint_position_noise
        joint_noise = jax.random.uniform(
            rng_joints,
            (self.sys.nq - _BASE_QUATERNION_END_INDEX,),
            minval=-noise,
            maxval=noise,
        )
        qpos = qpos.at[_BASE_QUATERNION_END_INDEX:].add(joint_noise)
        velocity_noise = self.config.reset_velocity_noise
        qvel = jax.random.uniform(
            rng_qvel,
            (self.sys.nv,),
            minval=-velocity_noise,
            maxval=velocity_noise,
        )
        pipeline_state = self.pipeline_init(qpos, qvel)

        cmd = self._command()
        obs = self._get_obs(pipeline_state, cmd)
        reward = jnp.zeros(())
        done = jnp.zeros(())
        _, metrics = self._compute_reward(
            pipeline_state, cmd, jnp.zeros(self.sys.nu)
        )
        metrics = {key: jnp.zeros(()) for key in metrics}
        return brax_base.State(pipeline_state, obs, reward, done, metrics)

    def step(
        self, state: brax_base.State, action: jax.Array
    ) -> brax_base.State:
        """Simulates one control step with n_substeps physics steps."""
        # Brax PPO's actions are tanh-squashed into (-1, 1); the actuators' ctrlrange bounds the control.
        ctrl = action * self.config.action_scale
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)

        cmd = self._command()
        obs = self._get_obs(pipeline_state, cmd)
        reward, metrics = self._compute_reward(pipeline_state, cmd, ctrl)
        done = self._check_termination(pipeline_state)

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )

    def _get_obs(self, pipeline_state: mjx.Data, cmd: jax.Array) -> jax.Array:
        """Constructs the translation- and heading-invariant observation vector from MJX state."""
        qpos = pipeline_state.qpos
        qvel = pipeline_state.qvel
        return jnp.concatenate(
            [
                qpos[_BASE_POSITION_END_INDEX - 1 : _BASE_POSITION_END_INDEX],
                compute_unit_gravity_ewrt_base(qpos),
                qpos[_BASE_QUATERNION_END_INDEX:],
                compute_linvel_base_rt_world_ewrt_heading(qpos, qvel),
                qvel[_BASE_LINEAR_VELOCITY_END_INDEX:],
                cmd,
            ]
        )

    def _compute_reward(
        self, pipeline_state: mjx.Data, cmd: jax.Array, ctrl: jax.Array
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Computes the heading-frame tracking rewards and the energy penalty."""
        qpos = pipeline_state.qpos
        qvel = pipeline_state.qvel
        linvel_base_rt_world_ewrt_heading = (
            compute_linvel_base_rt_world_ewrt_heading(qpos, qvel)
        )
        sigma = self.config.tracking_sigma
        forward_reward = jnp.exp(
            -jnp.square(linvel_base_rt_world_ewrt_heading[0] - cmd[0]) / sigma
        )
        lateral_reward = jnp.exp(
            -jnp.square(linvel_base_rt_world_ewrt_heading[1] - cmd[1]) / sigma
        )
        yaw_rate_reward = jnp.exp(
            -jnp.square(
                compute_angvel_base_rt_world_ewrt_world_z(qpos, qvel) - cmd[2]
            )
            / sigma
        )
        energy_penalty = self.config.energy_weight * jnp.sum(jnp.square(ctrl))
        total_reward = (
            forward_reward
            + lateral_reward
            + self.config.yaw_rate_weight * yaw_rate_reward
            - energy_penalty
        )
        metrics = {
            "reward": total_reward,
            "forward_velocity_reward": forward_reward,
            "lateral_velocity_reward": lateral_reward,
            "yaw_rate_reward": yaw_rate_reward,
            "energy_penalty": energy_penalty,
        }
        return total_reward, metrics

    def _check_termination(self, pipeline_state: mjx.Data) -> jax.Array:
        """Checks termination: the base below _MIN_POS_BASE_RT_WORLD_EWRT_WORLD_Z."""
        pos_base_rt_world_ewrt_world_z = pipeline_state.qpos[
            _BASE_POSITION_END_INDEX - 1
        ]
        return jnp.where(
            pos_base_rt_world_ewrt_world_z
            < _MIN_POS_BASE_RT_WORLD_EWRT_WORLD_Z,
            1.0,
            0.0,
        )
