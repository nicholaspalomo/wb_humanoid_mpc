"""Unit and integration tests for Cartpole Brax MJX environment and training pipeline."""

import os
import shutil
import tempfile
import unittest

from brax.envs.base import State
from brax.training import types
from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from humanoid_learning.examples.train_cartpole import (
    CartpoleBraxEnv,
    plot_training_curves,
    render_policy_rollout,
)


def _dummy_inference_fn(obs, rng):
    """Module-level dummy inference function for policy rollout tests."""
    return jnp.zeros((1,)), {}


class TestCartpoleBraxTrainingPipeline(unittest.TestCase):
    """Verifies all components of the Cartpole Brax MJX training and visualization pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="cartpole_test_")
        self.env = CartpoleBraxEnv()
        self.rng = jax.random.PRNGKey(42)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_env_initialization_and_dimensions(self):
        """Tests environment dimensions and model properties."""
        self.assertEqual(self.env.action_size, 1)
        self.assertEqual(self.env.observation_size, 5)
        self.assertIsNotNone(self.env.sys)

    def test_env_single_reset_and_step(self):
        """Tests single environment reset and physics stepping."""
        state = self.env.reset(self.rng)
        self.assertIsInstance(state, State)
        self.assertEqual(state.obs.shape, (5,))
        self.assertFalse(bool(state.done))
        self.assertFalse(np.isnan(np.array(state.obs)).any())

        # Step forward with zero control
        action = jnp.zeros((1,))
        next_state = self.env.step(state, action)
        self.assertEqual(next_state.obs.shape, (5,))
        self.assertFalse(np.isnan(np.array(next_state.obs)).any())
        self.assertFalse(np.isnan(float(next_state.reward)))

        # Step forward with maximum control
        action_max = jnp.ones((1,))
        next_state_max = self.env.step(state, action_max)
        self.assertEqual(next_state_max.obs.shape, (5,))

    def test_rendering_and_gif_generation(self):
        """Tests rendering policy rollout to animated GIF."""
        gif_path = os.path.join(self.temp_dir, "test_rollout.gif")
        eval_score = render_policy_rollout(
            self.env.mj_model, _dummy_inference_fn, gif_path, num_frames=20
        )

        self.assertTrue(os.path.exists(gif_path))
        self.assertGreater(os.path.getsize(gif_path), 1000)
        self.assertIsInstance(eval_score, float)

        # Verify GIF can be opened and contains frames
        with Image.open(gif_path) as img:
            self.assertEqual(img.format, "GIF")
            self.assertGreaterEqual(img.n_frames, 5)

    def test_plotting_training_curves(self):
        """Tests generating metric curves plot."""
        metrics_history = {
            "eval_steps": [1000, 2000, 3000, 4000],
            "eval_scores": [50.0, 70.0, 100.0, 119.0],
            "loss_steps": [1000, 2000, 3000, 4000],
            "loss": [10.0, 8.0, 5.0, 3.0],
        }
        plot_path = os.path.join(self.temp_dir, "test_curves.png")
        plot_training_curves(metrics_history, plot_path)

        self.assertTrue(os.path.exists(plot_path))
        self.assertGreater(os.path.getsize(plot_path), 1000)

    def test_ppo_policy_network_and_rollout(self):
        """Tests PPO policy network initialization, inference function, and environment rollout."""
        ppo_network = ppo_networks.make_ppo_networks(
            observation_size=self.env.observation_size,
            action_size=self.env.action_size,
            policy_hidden_layer_sizes=(32, 32),
            value_hidden_layer_sizes=(32, 32),
        )

        rng_policy, rng_value, rng_step = jax.random.split(self.rng, 3)
        dummy_obs = jnp.zeros((1, self.env.observation_size))
        policy_params = ppo_network.policy_network.init(rng_policy)
        value_params = ppo_network.value_network.init(rng_value)

        self.assertIsNotNone(policy_params)
        self.assertIsNotNone(value_params)

        # Test value network evaluation
        normalizer_params = ()
        value = ppo_network.value_network.apply(
            normalizer_params, value_params, dummy_obs
        )
        self.assertEqual(value.shape, (1,))
        self.assertFalse(np.isnan(np.array(value)).any())

        # Test inference function
        make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
        inference_fn = make_inference_fn((normalizer_params, policy_params))

        single_obs = jnp.zeros((self.env.observation_size,))
        action, extra = inference_fn(single_obs, rng_step)
        self.assertEqual(action.shape, (self.env.action_size,))
        self.assertFalse(np.isnan(np.array(action)).any())

        # Test rollout in environment with policy inference
        state = self.env.reset(self.rng)
        for _ in range(5):
            act, _ = inference_fn(state.obs, rng_step)
            state = self.env.step(state, act)
            self.assertFalse(np.isnan(np.array(state.obs)).any())
            self.assertFalse(np.isnan(float(state.reward)))


if __name__ == "__main__":
    unittest.main()
