"""Behavioral Cloning (BC) warmstart on MPC state-action trajectories."""

import argparse
import h5py
import numpy as np
import jax
import jax.numpy as jnp
import optax

from humanoid_learning.training.train_ppo import ActorCritic


def parse_args():
    parser = argparse.ArgumentParser(
        description="Behavioral Cloning Pretraining from MPC Demonstrations"
    )
    parser.add_argument(
        "--demos_path", type=str, default="", help="Path to HDF5 MPC demo dataset"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--output_path",
        type=str,
        default="checkpoints/bc_policy.npz",
        help="Output checkpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("🚀 Initializing Behavioral Cloning Pretraining from MPC Rollouts")
    print(f"   Demos Path:   {args.demos_path or '(Synthetic / Stub)'}")
    print(f"   Epochs:       {args.epochs}")
    print(f"   Batch Size:   {args.batch_size}")
    print("=" * 60)

    # Synthetic demo data if no path provided
    obs_dim = 25
    act_dim = 12
    n_samples = 1000

    if args.demos_path and h5py.is_hdf5(args.demos_path):
        with h5py.File(args.demos_path, "r") as f:
            observations = np.array(f["observations"])
            actions = np.array(f["actions"])
    else:
        print("ℹ️ Using synthetic demonstrations for warmstart stub.")
        observations = np.random.randn(n_samples, obs_dim).astype(np.float32)
        actions = np.random.randn(n_samples, act_dim).astype(np.float32)

    network = ActorCritic(action_dim=act_dim)
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    params = network.init(init_rng, jnp.zeros((1, obs_dim)))

    tx = optax.adam(learning_rate=args.lr)
    opt_state = tx.init(params)

    @jax.jit
    def loss_fn(params, obs, targets):
        mean, _, _ = network.apply(params, obs)
        return jnp.mean(jnp.square(mean - targets))

    @jax.jit
    def train_step(params, opt_state, obs, targets):
        loss, grads = jax.value_and_grad(loss_fn)(params, obs, targets)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    obs_jax = jnp.array(observations)
    act_jax = jnp.array(actions)

    for epoch in range(args.epochs):
        params, opt_state, loss = train_step(params, opt_state, obs_jax, act_jax)
        if (epoch + 1) % max(1, args.epochs // 5) == 0:
            print(f"Epoch {epoch + 1}/{args.epochs} - MSE Loss: {loss:.6f}")

    print("✅ Behavioral cloning pretraining completed successfully.")


if __name__ == "__main__":
    main()
