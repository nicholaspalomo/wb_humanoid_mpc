"""Unit and integration tests for Cartpole Brax MJX environment and training pipeline."""

import os
import shutil
import tempfile
import unittest

from brax.envs.base import State
from brax.training.agents.ppo import train as ppo
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from humanoid_learning.examples.train_cartpole import (
    CartpoleBraxEnv,
    plot_training_curves,
    render_policy_rollout,
)


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

        # Simple identity/heuristic inference function
        def dummy_inference_fn(obs, rng):
            return jnp.zeros((1,)), {}

        gif_path = os.path.join(self.temp_dir, "test_rollout.gif")
        eval_score = render_policy_rollout(
            self.env.mj_model, dummy_inference_fn, gif_path
        )

        self.assertTrue(os.path.exists(gif_path))
        self.assertGreater(os.path.getsize(gif_path), 1000)
        self.assertIsInstance(eval_score, float)

        # Verify GIF can be opened and contains multiple frames
        with Image.open(gif_path) as img:
            self.assertEqual(img.format, "GIF")
            self.assertGreater(img.n_frames, 10)

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

    def test_brax_ppo_smoke_training(self):
        """Tests a short Brax PPO training run end-to-end."""
        progress_called = False

        def mock_progress(num_steps, metrics):
            nonlocal progress_called
            progress_called = True

        make_inference_fn, params, metrics = ppo.train(
            environment=self.env,
            num_timesteps=128,
            num_evals=2,
            reward_scaling=1.0,
            episode_length=32,
            normalize_observations=False,
            action_repeat=1,
            unroll_length=4,
            num_minibatches=2,
            num_updates_per_batch=1,
            discounting=0.99,
            learning_rate=1e-3,
            num_envs=8,
            batch_size=8,
            seed=0,
            progress_fn=mock_progress,
        )

        self.assertIsNotNone(params)
        self.assertTrue(progress_called)

        inference_fn = make_inference_fn(params)
        dummy_obs = jnp.zeros((5,))
        action, _ = inference_fn(dummy_obs, self.rng)
        self.assertEqual(action.shape, (1,))
        self.assertFalse(np.isnan(np.array(action)).any())


if __name__ == "__main__":
    unittest.main()
