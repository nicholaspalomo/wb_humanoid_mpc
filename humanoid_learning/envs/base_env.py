"""Base MJX Environment for humanoid reinforcement learning with MuJoCo Playground."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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


@dataclass
class HumanoidEnvState:
    """MJX environment state."""

    data: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array]
    info: Dict[str, Any]


class HumanoidMpxEnv:
    """Base MJX velocity-tracking environment compatible with Brax and MuJoCo Playground."""

    def __init__(self, config: HumanoidEnvConfig):
        self.config = config

        if config.xml_string is not None:
            self._mj_model = mujoco.MjModel.from_xml_string(config.xml_string)
        elif config.xml_path is not None:
            self._mj_model = mujoco.MjModel.from_xml_path(config.xml_path)
        else:
            # Minimal default humanoid/pendulum for smoke test and initialization
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
            self._mj_model = mujoco.MjModel.from_xml_string(default_xml)

        self._mj_model.opt.timestep = config.physics_dt
        self._n_substeps = int(round(config.control_dt / config.physics_dt))
        self.mjx_model = mjx.put_model(self._mj_model)

    @property
    def action_size(self) -> int:
        return self._mj_model.nu

    @property
    def observation_size(self) -> int:
        # base qpos (except pos xy), qvel, commands, previous actions
        return self._mj_model.nq + self._mj_model.nv + 3

    def reset(self, rng: jax.Array) -> HumanoidEnvState:
        """Resets the environment state."""
        data = mjx.make_data(self.mjx_model)
        obs = self._get_obs(data, jnp.zeros((3,)))
        reward = jnp.zeros(())
        done = jnp.zeros((), dtype=jnp.bool_)
        metrics = {
            "tracking_reward": jnp.zeros(()),
            "energy_penalty": jnp.zeros(()),
        }
        return HumanoidEnvState(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info={},
        )

    def step(self, state: HumanoidEnvState, action: jax.Array) -> HumanoidEnvState:
        """Simulates one control step with n_substeps physics steps."""
        scaled_action = action * self.config.action_scale
        ctrl = jnp.clip(scaled_action, -1.0, 1.0)
        data = state.data.replace(ctrl=ctrl)

        def _substep(d, _):
            d = mjx.step(self.mjx_model, d)
            return d, None

        data, _ = jax.lax.scan(_substep, data, None, length=self._n_substeps)

        cmd = jnp.array(
            [self.config.target_vx, self.config.target_vy, self.config.target_wz]
        )
        obs = self._get_obs(data, cmd)
        reward, metrics = self._compute_reward(data, cmd, ctrl)
        done = self._check_termination(data)

        return HumanoidEnvState(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info={},
        )

    def _get_obs(self, data: mjx.Data, cmd: jax.Array) -> jax.Array:
        """Constructs observation vector from MJX state."""
        return jnp.concatenate([data.qpos, data.qvel, cmd])

    def _compute_reward(
        self, data: mjx.Data, cmd: jax.Array, ctrl: jax.Array
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Computes tracking reward and regularization penalties."""
        vx = data.qvel[0]
        tracking_err = jnp.square(vx - cmd[0])
        tracking_reward = jnp.exp(-tracking_err / 0.25)
        energy_penalty = 0.001 * jnp.sum(jnp.square(ctrl))
        total_reward = tracking_reward - energy_penalty
        metrics = {
            "tracking_reward": tracking_reward,
            "energy_penalty": energy_penalty,
        }
        return total_reward, metrics

    def _check_termination(self, data: mjx.Data) -> jax.Array:
        """Checks termination conditions (e.g. fallen torso height)."""
        torso_z = data.qpos[2]
        return torso_z < 0.2
