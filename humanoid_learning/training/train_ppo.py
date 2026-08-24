"""PPO training pipeline for MuJoCo Playground humanoid environments using Google Brax."""

import argparse
import os
import time
from typing import Dict

from brax.training.agents.ppo import train as ppo
import jax

from humanoid_learning.envs.base_env import HumanoidEnvConfig, HumanoidMpxEnv
import humanoid_learning.training  # Registers JAX/Brax compatibility polyfills


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Humanoid PPO Policy with Brax and MJX"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=64,
        help="Number of parallel MJX environments",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100_000,
        help="Total environment steps",
    )
    parser.add_argument(
        "--num_evals",
        type=int,
        default=10,
        help="Number of evaluation checkpoints",
    )
    parser.add_argument(
        "--episode_length", type=int, default=1000, help="Episode step length"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve output directory relative to workspace root if invoked via Bazel
    workspace_dir = os.environ.get(
        "BUILD_WORKSPACE_DIRECTORY", os.path.abspath(".")
    )
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(workspace_dir, args.output_dir)

    print("=" * 70)
    print("🚀 Google Brax & MuJoCo MJX Humanoid PPO Training")
    print(f"   Parallel Envs:       {args.num_envs}")
    print(f"   Total Timesteps:     {args.total_timesteps:,}")
    print(f"   Evaluation Epochs:   {args.num_evals}")
    print(f"   Episode Length:      {args.episode_length}")
    print(f"   Output Directory:    {os.path.abspath(args.output_dir)}")
    print("=" * 70)

    # Initialize environment
    config = HumanoidEnvConfig()
    env = HumanoidMpxEnv(config)

    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()

    def progress_callback(num_steps: int, metrics: Dict[str, float]):
        eval_reward = float(
            metrics.get("eval/episode_reward", metrics.get("eval/reward", 0.0))
        )
        loss_val = float(metrics.get("training/total_loss", 0.0))
        print(
            f"Step {num_steps:>6d} | Eval Reward: {eval_reward:>+7.2f} | Loss: {loss_val:.4f}"
        )

    print("\n🏁 Starting Brax PPO Training Loop...\n")

    make_inference_fn, params, final_metrics = ppo.train(
        environment=env,
        num_timesteps=args.total_timesteps,
        num_evals=args.num_evals,
        reward_scaling=1.0,
        episode_length=args.episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=16,
        num_updates_per_batch=4,
        discounting=args.gamma,
        learning_rate=args.lr,
        entropy_cost=1e-2,
        num_envs=args.num_envs,
        batch_size=32,
        seed=args.seed,
        progress_fn=progress_callback,
    )

    inference_fn = make_inference_fn(params)
    print(f"\n⚡ Training finished in {time.time() - t_start:.2f}s.")
    print("✅ Brax PPO humanoid policy trained and ready for inference.")


if __name__ == "__main__":
    main()
