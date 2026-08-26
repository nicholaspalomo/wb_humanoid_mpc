"""Smoke tests to verify MuJoCo Playground, MJX, JAX, and RL stack imports and basic execution."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import brax
import flax
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import optax

try:
    import onnx

    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False

from humanoid_learning.envs.base_env import HumanoidEnvConfig, HumanoidMpxEnv


@jax.jit
def _square_add(x, y):
    """Module-level JIT-compiled helper for arithmetic tests."""
    return jnp.square(x) + y


@jax.jit
def _step_pendulum(mjx_model, d):
    """Module-level JIT-compiled MJX physics stepping helper."""
    return mjx.step(mjx_model, d)


class TestRLImportsAndBasics(unittest.TestCase):
    """Verifies that all core RL dependencies can be loaded and executed."""

    def test_jax_devices(self):
        """Reports available JAX acceleration devices (GPU/TPU/CPU)."""
        devices = jax.devices()
        self.assertGreater(len(devices), 0)
        print(
            f"JAX active backend platform: {jax.default_backend()} | Devices: {devices}"
        )

    def test_jax_and_jit(self):
        """Tests JAX installation and JIT compilation."""
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 5.0, 6.0])
        res = _square_add(a, b)
        expected = np.array([5.0, 9.0, 15.0])
        np.testing.assert_allclose(np.array(res), expected, rtol=1e-5)

    def test_mujoco_and_mjx(self):
        """Tests MuJoCo and MJX model compilation and stepping."""
        xml = """
        <mujoco model="test_pendulum">
            <worldbody>
                <body name="pole" pos="0 0 1">
                    <joint name="hinge" type="hinge" axis="0 1 0"/>
                    <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.05" mass="1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        mj_model = mujoco.MjModel.from_xml_string(xml)
        mjx_model = mjx.put_model(mj_model)
        mjx_data = mjx.make_data(mjx_model)

        next_d = _step_pendulum(mjx_model, mjx_data)
        self.assertIsNotNone(next_d)

    def test_humanoid_env(self):
        """Tests HumanoidMpxEnv reset and step."""
        config = HumanoidEnvConfig()
        env = HumanoidMpxEnv(config)

        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)
        self.assertEqual(state.obs.shape, (env.observation_size,))

        action = jnp.zeros((env.action_size,))
        next_state = env.step(state, action)
        self.assertEqual(next_state.obs.shape, (env.observation_size,))

    def test_framework_imports(self):
        """Tests importing auxiliary RL libraries."""
        self.assertIsNotNone(brax.__name__)
        self.assertIsNotNone(flax.__name__)
        self.assertIsNotNone(optax.__name__)
        if _HAS_ONNX:
            self.assertIsNotNone(onnx.__name__)


if __name__ == "__main__":
    unittest.main()
