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

"""PPO training pipeline for MuJoCo Playground humanoid environments using Google Brax."""

import argparse
import os
import time

from brax.training.agents.ppo import train as ppo

from humanoid_learning.envs import base_env

# pylint: disable-next=unused-import  # Installs the JAX polyfill Brax PPO needs (training/__init__.py).
import humanoid_learning.training


def parse_args() -> argparse.Namespace:
    """The command line of the training run."""
    parser = argparse.ArgumentParser(
        description="Train Humanoid PPO Policy with Brax and MJX"
    )
    parser.add_argument(
        "--num_envs", type=int, default=64, help="Number of parallel MJX environments"
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=100_000, help="Total environment steps"
    )
    parser.add_argument(
        "--num_evals", type=int, default=10, help="Number of evaluation checkpoints"
    )
    parser.add_argument(
        "--episode_length", type=int, default=1000, help="Episode step length"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints", help="Output directory"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve output directory relative to workspace root if invoked via Bazel
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.path.abspath("."))
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
    config = base_env.HumanoidEnvConfig()
    env = base_env.HumanoidMpxEnv(config)

    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()

    def progress_callback(num_steps: int, metrics: dict[str, float]) -> None:
        eval_reward = float(
            metrics.get("eval/episode_reward", metrics.get("eval/reward", 0.0))
        )
        loss_val = float(metrics.get("training/total_loss", 0.0))
        print(
            f"Step {num_steps:>6d} | Eval Reward: {eval_reward:>+7.2f} | Loss: {loss_val:.4f}"
        )

    print("\n🏁 Starting Brax PPO Training Loop...\n")

    make_inference_fn, params, _ = ppo.train(
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

    make_inference_fn(params)  # Builds the policy once: the trained parameters load.
    print(f"\n⚡ Training finished in {time.time() - t_start:.2f}s.")
    print("✅ Brax PPO humanoid policy trained and ready for inference.")


if __name__ == "__main__":
    main()
