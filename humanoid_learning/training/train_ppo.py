"""PPO training pipeline for MuJoCo Playground humanoid environments."""

import argparse
import time
from typing import NamedTuple, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from humanoid_learning.envs.base_env import HumanoidEnvConfig, HumanoidMpxEnv


class ActorCritic(nn.Module):
    """Actor-Critic network architecture."""

    action_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        # Shared torso
        h = nn.Dense(256)(x)
        h = nn.elu(h)
        h = nn.Dense(256)(h)
        h = nn.elu(h)

        # Actor head (mean & log_std)
        actor_mean = nn.Dense(self.action_dim)(h)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))

        # Critic head (value)
        value = nn.Dense(1)(h)

        return actor_mean, log_std, jnp.squeeze(value, axis=-1)


class Transition(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    info: dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Humanoid PPO Policy with MJX/Playground"
    )
    parser.add_argument(
        "--num_envs", type=int, default=64, help="Number of parallel MJX environments"
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=100_000, help="Total environment steps"
    )
    parser.add_argument(
        "--num_steps", type=int, default=10, help="Rollout steps per iteration"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints", help="Output directory"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("🚀 Initializing Humanoid PPO Training with MuJoCo Playground / MJX")
    print(f"   Parallel Envs:    {args.num_envs}")
    print(f"   Total Timesteps:  {args.total_timesteps}")
    print(f"   Rollout Length:   {args.num_steps}")
    print(f"   Seed:             {args.seed}")
    print("=" * 60)

    # Initialize environment
    config = HumanoidEnvConfig()
    env = HumanoidMpxEnv(config)

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    obs_dim = env.observation_size
    act_dim = env.action_size

    # Initialize network
    network = ActorCritic(action_dim=act_dim)
    dummy_obs = jnp.zeros((args.num_envs, obs_dim))
    params = network.init(init_rng, dummy_obs)

    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=args.lr),
    )
    opt_state = tx.init(params)

    # Vectorized reset
    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)

    reset_rngs = jax.random.split(rng, args.num_envs)
    env_state = v_reset(reset_rngs)

    print(f"✅ Environment initialized: obs_dim={obs_dim}, act_dim={act_dim}")
    print(
        f"✅ Neural network parameters initialized ({sum(x.size for x in jax.tree_util.tree_leaves(params))} weights)"
    )

    # Execute a small smoke rollout step under JIT
    @jax.jit
    def single_step(env_state, params, key):
        mean, log_std, val = network.apply(params, env_state.obs)
        std = jnp.exp(log_std)
        noise = jax.random.normal(key, mean.shape)
        action = mean + std * noise
        next_env_state = v_step(env_state, action)
        return next_env_state, val

    t0 = time.time()
    rng, step_key = jax.random.split(rng)
    env_state, val = single_step(env_state, params, step_key)
    print(
        f"⚡ JIT compilation and first rollout step completed in {time.time() - t0:.3f}s"
    )
    print("✅ PPO training pipeline initialized successfully.")


if __name__ == "__main__":
    main()
