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


"""Behavioral Cloning (BC) warmstart on MPC state-action trajectories."""

import argparse
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax

from humanoid_learning.training import actor_critic


def parse_args() -> argparse.Namespace:
    """The command line of the warmstart."""
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


def train_behavioral_cloning(
    observations: np.ndarray,
    actions: np.ndarray,
    epochs: int,
    learning_rate: float,
    verbose: bool = True,
) -> tuple[Any, list[float]]:
    """Fits the actor's mean action to the demonstrated actions, one full-batch Adam step per epoch.

    Args:
        observations: The demonstrations' observations, (num_samples, observation_dim).
        actions: The demonstrated actions, (num_samples, action_dim).
        epochs: Number of training epochs.
        learning_rate: Adam's learning rate.
        verbose: Whether to print the loss five times over the run.

    Returns:
        The trained ActorCritic parameters, and the mean squared error of every epoch, before its step.
    """
    network = actor_critic.ActorCritic(action_dim=actions.shape[1])
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    params = network.init(init_rng, jnp.zeros((1, observations.shape[1])))

    tx = optax.adam(learning_rate=learning_rate)
    opt_state = tx.init(params)

    @jax.jit
    def loss_fn(params: Any, obs: jax.Array, targets: jax.Array) -> jax.Array:
        mean, _, _ = network.apply(params, obs)
        return jnp.mean(jnp.square(mean - targets))

    @jax.jit
    def train_step(
        params: Any, opt_state: optax.OptState, obs: jax.Array, targets: jax.Array
    ) -> tuple[Any, optax.OptState, jax.Array]:
        loss, grads = jax.value_and_grad(loss_fn)(params, obs, targets)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    obs_jax = jnp.array(observations)
    act_jax = jnp.array(actions)

    losses: list[float] = []
    for epoch in range(epochs):
        params, opt_state, loss = train_step(params, opt_state, obs_jax, act_jax)
        losses.append(float(loss))
        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"Epoch {epoch + 1}/{epochs} - MSE Loss: {loss:.6f}")
    return params, losses


def main() -> None:
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

    train_behavioral_cloning(observations, actions, args.epochs, args.lr)

    print("✅ Behavioral cloning pretraining completed successfully.")


if __name__ == "__main__":
    main()
