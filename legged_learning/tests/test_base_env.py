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

"""The heading-frame velocity tracking, observation, reset and time steps of the base MJX environment."""

import os

# JAX picks its platform when it is first imported, so the tests pin it to the CPU before the imports below.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# pylint: disable=wrong-import-position  # JAX_PLATFORMS is set above, before JAX is imported.
import unittest

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from legged_learning.envs import base_env

# pylint: enable=wrong-import-position

_FREE_BODY_XML = """
<mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
        <body name="base" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body>
    </worldbody>
</mujoco>
"""


def _random_unit_quaternion(rng: np.random.Generator) -> np.ndarray:
    coefficients = rng.standard_normal(4)
    return coefficients / np.linalg.norm(coefficients)


def _rotmat_of_yaw(psi: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(psi), -np.sin(psi), 0.0],
            [np.sin(psi), np.cos(psi), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _reward_terms(
    env: base_env.LeggedMpxEnv, qpos: np.ndarray, qvel: np.ndarray
) -> dict[str, float]:
    """The reward terms of `env` at the state (qpos, qvel), under its command and a zero action."""
    pipeline_state = env.pipeline_init(jnp.asarray(qpos), jnp.asarray(qvel))
    command = (
        env._command()
    )  # pylint: disable=protected-access  # The unit under test.
    action = jnp.zeros(env.action_size)
    _, metrics = (
        env._compute_reward(  # pylint: disable=protected-access  # Same.
            pipeline_state, command, action
        )
    )
    return {key: float(value) for key, value in metrics.items()}


def _observation(
    env: base_env.LeggedMpxEnv, qpos: np.ndarray, qvel: np.ndarray
) -> np.ndarray:
    pipeline_state = env.pipeline_init(jnp.asarray(qpos), jnp.asarray(qvel))
    return np.asarray(
        env._get_obs(
            pipeline_state, env._command()
        )  # pylint: disable=protected-access  # The unit under test.
    )


def _deterministic_env(
    cmd_linvel_base_rt_world_ewrt_heading_x: float = 1.0,
    cmd_linvel_base_rt_world_ewrt_heading_y: float = 0.0,
    cmd_angvel_base_rt_world_ewrt_world_z: float = 0.0,
) -> base_env.LeggedMpxEnv:
    """Returns an environment of the command given, which resets to qpos0 at rest at a heading of zero."""
    config = base_env.LeggedEnvConfig(
        cmd_linvel_base_rt_world_ewrt_heading_x=cmd_linvel_base_rt_world_ewrt_heading_x,
        cmd_linvel_base_rt_world_ewrt_heading_y=cmd_linvel_base_rt_world_ewrt_heading_y,
        cmd_angvel_base_rt_world_ewrt_world_z=cmd_angvel_base_rt_world_ewrt_world_z,
        randomize_heading=False,
        reset_joint_position_noise=0.0,
        reset_velocity_noise=0.0,
    )
    return base_env.LeggedMpxEnv(config)


class HeadingFrameTest(unittest.TestCase):
    def test_the_helpers_match_mujocos_own_frames(self) -> None:
        model = mujoco.MjModel.from_xml_string(_FREE_BODY_XML)
        data = mujoco.MjData(model)
        rng = np.random.default_rng(0)
        for _ in range(20):
            data.qpos[3:7] = _random_unit_quaternion(rng)
            data.qvel[:] = rng.standard_normal(6)
            mujoco.mj_forward(model, data)
            # mj_objectVelocity writes (angular, linear), world orientation, at the body's center of mass, which is
            # the base origin here: the sphere is centered on it.
            mj_object_output = np.zeros(6)
            mujoco.mj_objectVelocity(
                model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                1,
                mj_object_output,
                0,
            )
            angvel_linkcom_rt_world_ewrt_world = mj_object_output[:3]
            linvel_linkcom_rt_world_ewrt_world = mj_object_output[3:]
            rotmat_base_rt_world = data.xmat[1].reshape(3, 3)
            # The twist heading: the angle of the twist about the world z axis.
            psi = 2.0 * np.arctan2(data.qpos[6], data.qpos[3])
            qpos = jnp.asarray(data.qpos)
            qvel = jnp.asarray(data.qvel)

            np.testing.assert_allclose(
                base_env.compute_rotmat_base_rt_world(qpos[3:7]),
                rotmat_base_rt_world,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                base_env.compute_linvel_base_rt_world_ewrt_heading(qpos, qvel),
                _rotmat_of_yaw(psi).T @ linvel_linkcom_rt_world_ewrt_world,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                base_env.compute_angvel_base_rt_world_ewrt_world_z(qpos, qvel),
                angvel_linkcom_rt_world_ewrt_world[2],
                atol=1e-5,
            )
            np.testing.assert_allclose(
                base_env.compute_unit_gravity_ewrt_base(qpos),
                rotmat_base_rt_world.T @ np.array([0.0, 0.0, -1.0]),
                atol=1e-6,
            )

    def test_the_heading_of_a_tilted_base_is_its_twist_not_its_zyx_yaw(
        self,
    ) -> None:
        # A base turned by 0.7 rad about the world z axis, then pitched and rolled: the ZYX yaw of R moves with the tilt,
        # the twist heading does not.
        quat_base_rt_world_wxyz = np.array(
            base_env.quaternion_product(
                base_env.unit_quaternion_of_yaw(jnp.asarray(0.7)),
                jnp.asarray(
                    [np.cos(0.3), np.sin(0.3) * 0.6, np.sin(0.3) * 0.8, 0.0]
                ),
            )
        )
        rotmat_base_rt_world = np.asarray(
            base_env.compute_rotmat_base_rt_world(
                jnp.asarray(quat_base_rt_world_wxyz)
            )
        )
        eulerzyx_base_rt_world_yaw = np.arctan2(
            rotmat_base_rt_world[1, 0], rotmat_base_rt_world[0, 0]
        )
        yaw_base_rt_world = float(
            base_env.compute_yaw_base_rt_world(
                jnp.asarray(quat_base_rt_world_wxyz)
            )
        )
        self.assertAlmostEqual(
            yaw_base_rt_world,
            2.0
            * np.arctan2(
                quat_base_rt_world_wxyz[3], quat_base_rt_world_wxyz[0]
            ),
            places=6,
        )
        self.assertGreater(
            abs(yaw_base_rt_world - eulerzyx_base_rt_world_yaw), 1e-2
        )

    def test_moving_sideways_at_a_heading_of_ninety_degrees_is_not_forward_motion(
        self,
    ) -> None:
        env = _deterministic_env(cmd_linvel_base_rt_world_ewrt_heading_x=1.0)
        qpos = np.array(env.sys.qpos0)
        qpos[3:7] = np.asarray(base_env.unit_quaternion_of_yaw(jnp.pi / 2.0))
        moving = np.zeros(env.sys.nv)
        moving[0] = 1.0  # World x: the base's right side.
        still = np.zeros(env.sys.nv)

        np.testing.assert_allclose(
            base_env.compute_linvel_base_rt_world_ewrt_heading(
                jnp.asarray(qpos), jnp.asarray(moving)
            ),
            [0.0, -1.0, 0.0],
            atol=1e-6,
        )
        sideways = _reward_terms(env, qpos, moving)
        standing = _reward_terms(env, qpos, still)
        self.assertAlmostEqual(
            sideways["forward_velocity_reward"],
            standing["forward_velocity_reward"],
            places=6,
        )
        self.assertLess(sideways["reward"], standing["reward"])

        # Unturned, the same world velocity is forward motion, and tracks the command exactly.
        unturned = _reward_terms(env, np.array(env.sys.qpos0), moving)
        self.assertAlmostEqual(
            unturned["forward_velocity_reward"], 1.0, places=6
        )

    def test_the_reward_and_observation_do_not_depend_on_the_heading_or_position(
        self,
    ) -> None:
        env = _deterministic_env(
            cmd_linvel_base_rt_world_ewrt_heading_x=0.7,
            cmd_linvel_base_rt_world_ewrt_heading_y=-0.2,
            cmd_angvel_base_rt_world_ewrt_world_z=0.4,
        )
        rng = np.random.default_rng(1)
        for _ in range(10):
            qpos = np.array(env.sys.qpos0)
            qpos[3:7] = _random_unit_quaternion(rng)
            qpos[7:] = rng.uniform(-0.5, 0.5, env.sys.nq - 7)
            qvel = rng.standard_normal(env.sys.nv)

            psi = rng.uniform(-np.pi, np.pi)
            turned_qpos = qpos.copy()
            turned_qpos[:2] += rng.uniform(-5.0, 5.0, 2)
            turned_qpos[3:7] = np.asarray(
                base_env.quaternion_product(
                    base_env.unit_quaternion_of_yaw(psi), jnp.asarray(qpos[3:7])
                )
            )
            turned_qvel = qvel.copy()
            turned_qvel[:3] = (
                _rotmat_of_yaw(psi) @ qvel[:3]
            )  # The base-frame angular velocity is unchanged.

            reference = _reward_terms(env, qpos, qvel)
            turned = _reward_terms(env, turned_qpos, turned_qvel)
            for key, value in reference.items():
                self.assertAlmostEqual(turned[key], value, places=5, msg=key)
            np.testing.assert_allclose(
                _observation(env, turned_qpos, turned_qvel),
                _observation(env, qpos, qvel),
                atol=1e-5,
            )

    def test_the_yaw_rate_is_rewarded_about_the_world_z_axis(self) -> None:
        env = _deterministic_env(
            cmd_linvel_base_rt_world_ewrt_heading_x=0.0,
            cmd_angvel_base_rt_world_ewrt_world_z=0.5,
        )
        qpos = np.array(env.sys.qpos0)
        qpos[3:7] = np.asarray(base_env.unit_quaternion_of_yaw(1.0))
        qvel = np.zeros(env.sys.nv)
        qvel[5] = 0.5
        self.assertAlmostEqual(
            _reward_terms(env, qpos, qvel)["yaw_rate_reward"], 1.0, places=6
        )
        qvel[5] = 0.0
        self.assertLess(_reward_terms(env, qpos, qvel)["yaw_rate_reward"], 1.0)


class ObservationTest(unittest.TestCase):
    def test_the_observation_has_the_size_it_declares(self) -> None:
        env = base_env.LeggedMpxEnv()
        state = env.reset(jax.random.PRNGKey(0))
        self.assertEqual(state.obs.shape, (env.observation_size,))
        state = env.step(state, jnp.zeros(env.action_size))
        self.assertEqual(state.obs.shape, (env.observation_size,))
        self.assertTrue(np.all(np.isfinite(np.asarray(state.obs))))


class ResetTest(unittest.TestCase):
    def test_the_same_key_is_the_same_state_and_another_key_another(
        self,
    ) -> None:
        env = base_env.LeggedMpxEnv()
        first = env.reset(jax.random.PRNGKey(3))
        again = env.reset(jax.random.PRNGKey(3))
        other = env.reset(jax.random.PRNGKey(4))
        np.testing.assert_array_equal(
            first.pipeline_state.qpos, again.pipeline_state.qpos
        )
        np.testing.assert_array_equal(
            first.pipeline_state.qvel, again.pipeline_state.qvel
        )
        self.assertFalse(
            np.allclose(first.pipeline_state.qpos, other.pipeline_state.qpos)
        )
        self.assertFalse(
            np.allclose(first.pipeline_state.qvel, other.pipeline_state.qvel)
        )

    def test_a_reset_turns_the_base_and_keeps_its_height_and_unit_quaternion(
        self,
    ) -> None:
        env = base_env.LeggedMpxEnv()
        noise = env.config.reset_joint_position_noise
        yaw_base_rt_world_samples = []
        for seed in range(32):
            qpos = np.asarray(
                env.reset(jax.random.PRNGKey(seed)).pipeline_state.qpos
            )
            self.assertAlmostEqual(
                float(np.linalg.norm(qpos[3:7])), 1.0, places=6
            )
            np.testing.assert_allclose(qpos[:3], np.asarray(env.sys.qpos0)[:3])
            self.assertTrue(
                np.all(
                    np.abs(qpos[7:] - np.asarray(env.sys.qpos0)[7:]) <= noise
                )
            )
            yaw_base_rt_world_samples.append(
                float(
                    base_env.compute_yaw_base_rt_world(jnp.asarray(qpos[3:7]))
                )
            )
        self.assertGreater(
            max(yaw_base_rt_world_samples) - min(yaw_base_rt_world_samples),
            np.pi,
        )

    def test_without_randomization_the_reset_is_qpos0_at_rest(self) -> None:
        env = _deterministic_env()
        state = env.reset(jax.random.PRNGKey(5))
        np.testing.assert_allclose(
            state.pipeline_state.qpos, env.sys.qpos0, atol=1e-7
        )
        np.testing.assert_array_equal(
            state.pipeline_state.qvel, np.zeros(env.sys.nv)
        )


class SubstepTest(unittest.TestCase):
    def test_a_whole_ratio_is_the_substep_count(self) -> None:
        self.assertEqual(base_env.substep_count(0.02, 0.002), 10)
        self.assertEqual(
            base_env.substep_count(0.03, 0.01), 3
        )  # 0.03 / 0.01 rounds below 3.
        self.assertEqual(base_env.substep_count(0.002, 0.002), 1)

    def test_a_fractional_or_invalid_ratio_is_refused(self) -> None:
        for control_dt, physics_dt in (
            (0.025, 0.01),
            (0.001, 0.002),
            (0.0, 0.002),
            (0.02, -0.002),
            (float("nan"), 0.002),
            (0.02, float("inf")),
        ):
            with self.assertRaises(
                ValueError, msg=f"{control_dt}, {physics_dt}"
            ):
                base_env.substep_count(control_dt, physics_dt)
        with self.assertRaises(ValueError):
            base_env.LeggedMpxEnv(
                base_env.LeggedEnvConfig(control_dt=0.025, physics_dt=0.01)
            )

    def test_a_robot_without_a_root_free_joint_is_refused(self) -> None:
        xml = """<mujoco><worldbody><body><joint type="hinge"/><geom type="sphere" size="0.1"/></body>
                 </worldbody></mujoco>"""
        with self.assertRaises(ValueError):
            base_env.LeggedMpxEnv(base_env.LeggedEnvConfig(xml_string=xml))


if __name__ == "__main__":
    unittest.main()
