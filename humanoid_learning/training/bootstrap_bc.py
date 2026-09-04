"""Behavioral Cloning (BC) Bootstrapping from Offline MPC Demonstrations."""

import argparse
import os
import time
from typing import Tuple

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from humanoid_learning.training.offline_dataset import OfflineTrajectoryDataset
from humanoid_learning.training.policy_network import HumanoidResidualPolicy


# ==============================================================================
# Behavioral Cloning Training Defaults
# ==============================================================================
DEFAULT_BC_EPOCHS: int = 15
DEFAULT_BC_BATCH_SIZE: int = 128
DEFAULT_BC_LEARNING_RATE: float = 1e-3
DEFAULT_BC_OBS_DIM: int = 40
DEFAULT_BC_ACT_DIM: int = 12
DEFAULT_BC_OUTPUT_PATH: str = "checkpoints/bootstrapped_policy.npz"
BC_NLL_LOSS_WEIGHT: float = 0.1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bootstrap Humanoid Residual Policy via Behavioral Cloning"
    )
    parser.add_argument(
        "--demos_path",
        type=str,
        default="",
        help="Path to HDF5 demonstration dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_BC_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BC_BATCH_SIZE,
        help="Minibatch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_BC_LEARNING_RATE,
        help="Learning rate for AdamW",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_BC_OUTPUT_PATH,
        help="Path to save bootstrapped weights",
    )
    parser.add_argument(
        "--obs_dim",
        type=int,
        default=DEFAULT_BC_OBS_DIM,
        help="Observation dimension",
    )
    parser.add_argument(
        "--act_dim",
        type=int,
        default=DEFAULT_BC_ACT_DIM,
        help="Action residual dimension",
    )
    return parser.parse_args()


def bootstrap_policy(
    demos_path: str = "",
    epochs: int = DEFAULT_BC_EPOCHS,
    batch_size: int = DEFAULT_BC_BATCH_SIZE,
    lr: float = DEFAULT_BC_LEARNING_RATE,
    output_path: str = DEFAULT_BC_OUTPUT_PATH,
    obs_dim: int = DEFAULT_BC_OBS_DIM,
    act_dim: int = DEFAULT_BC_ACT_DIM,
) -> Tuple[HumanoidResidualPolicy, dict]:
    """Train policy on offline demonstration data to bootstrap weights."""
    print("=" * 70)
    print(
        "🚀 Initializing Behavioral Cloning Bootstrapping from Offline Trajectories"
    )
    print(f"   Demos Path:   {demos_path or '(Synthetic Reference Gait)'}")
    print(f"   Epochs:       {epochs}")
    print(f"   Batch Size:   {batch_size}")
    print(f"   Learning Rate:{lr}")
    print(f"   Output Path:  {output_path}")
    print("=" * 70)

    # 1. Load Dataset
    dataset = OfflineTrajectoryDataset(
        filepath=demos_path if demos_path else None,
        obs_dim=obs_dim,
        act_dim=act_dim,
    )

    # 2. Initialize Policy and Optimizer
    rng_key = jax.random.PRNGKey(42)
    policy_rngs = nnx.Rngs(params=rng_key)
    policy = HumanoidResidualPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=256,
        num_blocks=3,
        rngs=policy_rngs,
    )

    optimizer = nnx.Optimizer(
        policy,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr, weight_decay=1e-4),
        ),
        wrt=nnx.Param,
    )

    # 3. Supervised Negative Log-Likelihood / MSE Loss Step
    @nnx.jit
    def train_step(
        model: HumanoidResidualPolicy,
        opt: nnx.Optimizer,
        obs: jax.Array,
        target_acts: jax.Array,
    ):
        def loss_fn(m: HumanoidResidualPolicy):
            mean, std = m.forward_dist(obs)
            # Gaussian negative log likelihood + MSE
            mse = jnp.mean(jnp.square(mean - target_acts))
            nll = -jnp.mean(m.evaluate_log_prob(obs, target_acts))
            return mse + 0.1 * nll

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    steps_per_epoch = max(1, dataset.size // batch_size)
    t_start = time.time()

    print("\nStarting Behavioral Cloning Optimization...")
    for epoch in range(epochs):
        epoch_losses = []
        for step in range(steps_per_epoch):
            rng_key, step_rng = jax.random.split(rng_key)
            batch = dataset.sample_batch(step_rng, batch_size=batch_size)
            loss_val = train_step(
                policy, optimizer, batch.observations, batch.actions
            )
            epoch_losses.append(float(loss_val))

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch + 1:>3d}/{epochs} | BC Loss: {mean_loss:.6f} | Elapsed: {time.time() - t_start:.2f}s"
            )

    # 4. Save Bootstrapped Model Weights
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    state = nnx.state(policy)
    flat_state = {
        str(path): np.array(val) for path, val in dict(state.flat_state()).items()
    }
    np.savez(output_path, **flat_state)
    print(f"\n✓ Bootstrapped policy checkpoint saved to '{output_path}'.")

    return policy, flat_state


def main():
    args = parse_args()
    bootstrap_policy(
        demos_path=args.demos_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output_path,
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
    )


if __name__ == "__main__":
    main()
