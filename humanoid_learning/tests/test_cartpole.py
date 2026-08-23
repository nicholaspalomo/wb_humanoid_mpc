"""Unit and integration tests for Cartpole MJX environment, neural network, and training pipeline."""

import os
import shutil
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image

from humanoid_learning.examples.train_cartpole import (
    ActorCritic,
    CartpoleEnvState,
    CartpoleMJXEnv,
    plot_training_curves,
    render_policy_rollout,
)


class TestCartpoleTrainingPipeline(unittest.TestCase):
    """Verifies all components of the Cartpole MJX training and visualization pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="cartpole_test_")
        self.env = CartpoleMJXEnv()
        self.rng = jax.random.PRNGKey(42)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_env_initialization_and_dimensions(self):
        """Tests environment dimensions and model properties."""
        self.assertEqual(self.env.obs_dim, 5)
        self.assertEqual(self.env.act_dim, 1)
        self.assertIsNotNone(self.env.mj_model)
        self.assertIsNotNone(self.env.mjx_model)

    def test_env_single_reset_and_step(self):
        """Tests single environment reset and physics stepping."""
        state = self.env.reset(self.rng)
        self.assertIsInstance(state, CartpoleEnvState)
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

    def test_env_vectorized_vmap(self):
        """Tests batch simulation across parallel environments using jax.vmap."""
        batch_size = 16
        v_reset = jax.vmap(self.env.reset)
        v_step = jax.vmap(self.env.step)

        reset_keys = jax.random.split(self.rng, batch_size)
        batch_state = v_reset(reset_keys)
        self.assertEqual(batch_state.obs.shape, (batch_size, 5))
        self.assertEqual(batch_state.reward.shape, (batch_size,))
        self.assertEqual(batch_state.done.shape, (batch_size,))

        # Vectorized step
        batch_actions = jax.random.uniform(
            self.rng, (batch_size, 1), minval=-1.0, maxval=1.0
        )
        next_batch_state = v_step(batch_state, batch_actions)
        self.assertEqual(next_batch_state.obs.shape, (batch_size, 5))
        self.assertEqual(next_batch_state.reward.shape, (batch_size,))
        self.assertEqual(next_batch_state.done.shape, (batch_size,))

    def test_actor_critic_network(self):
        """Tests ActorCritic architecture and forward pass."""
        network = ActorCritic(action_dim=self.env.act_dim)
        batch_size = 8
        dummy_obs = jnp.zeros((batch_size, self.env.obs_dim))

        params = network.init(self.rng, dummy_obs)
        mean, log_std, value = network.apply(params, dummy_obs)

        self.assertEqual(mean.shape, (batch_size, 1))
        self.assertEqual(log_std.shape, (1,))
        self.assertEqual(value.shape, (batch_size,))
        self.assertFalse(np.isnan(np.array(mean)).any())
        self.assertFalse(np.isnan(np.array(value)).any())

    def test_rendering_and_gif_generation(self):
        """Tests rendering policy rollout to animated GIF."""
        network = ActorCritic(action_dim=self.env.act_dim)
        dummy_obs = jnp.zeros((1, self.env.obs_dim))
        params = network.init(self.rng, dummy_obs)

        gif_path = os.path.join(self.temp_dir, "test_rollout.gif")
        eval_score = render_policy_rollout(self.env.mj_model, network, params, gif_path)

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
            "updates": [1, 2, 3, 4],
            "rewards": [0.5, 0.7, 0.9, 0.95],
            "loss": [10.0, 8.0, 5.0, 3.0],
        }
        plot_path = os.path.join(self.temp_dir, "test_curves.png")
        plot_training_curves(metrics_history, plot_path)

        self.assertTrue(os.path.exists(plot_path))
        self.assertGreater(os.path.getsize(plot_path), 1000)

    def test_single_ppo_update_step(self):
        """Tests one complete PPO rollout and optimizer update."""
        num_envs = 8
        rollout_len = 16

        v_reset = jax.vmap(self.env.reset)
        v_step = jax.vmap(self.env.step)
        network = ActorCritic(action_dim=self.env.act_dim)
        optimizer = optax.adam(learning_rate=1e-3)

        rng, init_rng = jax.random.split(self.rng)
        params = network.init(init_rng, jnp.zeros((num_envs, self.env.obs_dim)))
        opt_state = optimizer.init(params)

        reset_keys = jax.random.split(rng, num_envs)
        env_state = v_reset(reset_keys)

        # Collect rollout
        def _step_fn(carry, _):
            e_state, k = carry
            k, act_key = jax.random.split(k)
            mean, log_std, value = network.apply(params, e_state.obs)
            std = jnp.exp(log_std)
            action = mean + std * jax.random.normal(act_key, mean.shape)
            next_e_state = v_step(e_state, action)
            transition = (
                e_state.obs,
                action,
                e_state.reward,
                value,
                mean,
                log_std,
            )
            return (next_e_state, k), transition

        (final_env_state, _), traj = jax.lax.scan(
            _step_fn, (env_state, rng), None, length=rollout_len
        )
        obs, actions, rewards, values, old_means, old_log_stds = traj
        returns = jnp.cumsum(rewards[::-1], axis=0)[::-1]
        advantages = returns - values

        def loss_fn(p):
            new_means, new_log_stds, new_values = network.apply(p, obs)
            diff = jnp.square(new_means - actions) - jnp.square(old_means - actions)
            policy_loss = jnp.mean(jnp.sum(diff, axis=-1) * advantages)
            value_loss = 0.5 * jnp.mean(jnp.square(new_values - returns))
            return policy_loss + value_loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)

        self.assertFalse(np.isnan(float(loss)))
        self.assertIsNotNone(next_params)
        self.assertIsNotNone(next_opt_state)


if __name__ == "__main__":
    unittest.main()
