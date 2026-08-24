"""Base MJX Environment for humanoid reinforcement learning with MuJoCo Playground & Brax."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from brax.envs.base import PipelineEnv, State
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


@dataclass
class HumanoidEnvConfig:
    """Configuration for humanoid MJX environment."""

    xml_path: Optional[str] = None
    xml_string: Optional[str] = None
    control_dt: float = 0.02
    physics_dt: float = 0.002
    action_scale: float = 0.5
    default_joint_angles: Optional[Dict[str, float]] = None
    target_vx: float = 1.0
    target_vy: float = 0.0
    target_wz: float = 0.0


class HumanoidMpxEnv(PipelineEnv):
    """Base MJX velocity-tracking environment inheriting from Brax PipelineEnv."""

    def __init__(self, config: Optional[HumanoidEnvConfig] = None, **kwargs):
        self.config = config or HumanoidEnvConfig()

        if self.config.xml_string is not None:
            mj_model = mujoco.MjModel.from_xml_string(self.config.xml_string)
        elif self.config.xml_path is not None:
            mj_model = mujoco.MjModel.from_xml_path(self.config.xml_path)
        else:
            default_xml = """
            <mujoco model="minimal_humanoid">
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
                    <motor name="hip_actuator" joint="hip" gear="50"/>
                </actuator>
            </mujoco>
            """
            mj_model = mujoco.MjModel.from_xml_string(default_xml)

        self.mj_model = mj_model
        self.mj_model.opt.timestep = self.config.physics_dt
        n_substeps = int(round(self.config.control_dt / self.config.physics_dt))
        mjx_model = mjx.put_model(self.mj_model)
        super().__init__(sys=mjx_model, n_frames=n_substeps, **kwargs)

    @property
    def action_size(self) -> int:
        return self.sys.nu

    @property
    def observation_size(self) -> int:
        return self.sys.nq + self.sys.nv + 3

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment state."""
        pipeline_state = self.pipeline_init(
            jnp.array(self.sys.qpos0),
            jnp.zeros(self.sys.nv),
        )
        cmd = jnp.array(
            [self.config.target_vx, self.config.target_vy, self.config.target_wz]
        )
        obs = self._get_obs(pipeline_state, cmd)
        reward = jnp.zeros(())
        done = jnp.zeros(())
        metrics = {
            "reward": reward,
            "tracking_reward": jnp.zeros(()),
            "energy_penalty": jnp.zeros(()),
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Simulates one control step with n_substeps physics steps."""
        scaled_action = action * self.config.action_scale
        ctrl = jnp.clip(scaled_action, -1.0, 1.0)
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)

        cmd = jnp.array(
            [self.config.target_vx, self.config.target_vy, self.config.target_wz]
        )
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
        """Constructs observation vector from MJX state."""
        return jnp.concatenate([pipeline_state.qpos, pipeline_state.qvel, cmd])

    def _compute_reward(
        self, pipeline_state: mjx.Data, cmd: jax.Array, ctrl: jax.Array
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Computes tracking reward and regularization penalties."""
        vx = pipeline_state.qvel[0]
        tracking_err = jnp.square(vx - cmd[0])
        tracking_reward = jnp.exp(-tracking_err / 0.25)
        energy_penalty = 0.001 * jnp.sum(jnp.square(ctrl))
        total_reward = tracking_reward - energy_penalty
        metrics = {
            "reward": total_reward,
            "tracking_reward": tracking_reward,
            "energy_penalty": energy_penalty,
        }
        return total_reward, metrics

    def _check_termination(self, pipeline_state: mjx.Data) -> jax.Array:
        """Checks termination conditions (e.g. fallen torso height)."""
        torso_z = pipeline_state.qpos[2]
        return jnp.where(torso_z < 0.2, 1.0, 0.0)
