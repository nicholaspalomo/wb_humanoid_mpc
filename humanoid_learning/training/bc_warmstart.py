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


"""Behavioral Cloning (BC) warmstart on MPC state-action trajectories.

Fits the deterministic policy actor_critic.policy_action() to demonstrated actions by minibatch Adam, and writes the
trained parameters and the observation normalizer as a checkpoint (actor_critic.save_checkpoint()) that export_onnx.py
reads. The demonstrated actions are in the policy's action range [-1, 1], which the environments scale to controls.
"""

import argparse
from collections.abc import Sequence
import dataclasses
import math
import os
from typing import Any

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax

from humanoid_learning.training import actor_critic

# The dimensions of the synthetic demonstrations main() trains on without a data set.
_SYNTHETIC_OBS_DIM = 25
_SYNTHETIC_ACT_DIM = 12
_SYNTHETIC_SAMPLES = 1000
# How many times a verbose training prints its loss.
_LOSS_REPORTS = 5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Returns the command line of the warmstart; `argv` None reads sys.argv."""
    parser = argparse.ArgumentParser(
        description="Behavioral Cloning Pretraining from MPC Demonstrations"
    )
    parser.add_argument(
        "--demos_path",
        type=str,
        default="",
        help="Path to HDF5 MPC demo dataset",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Minibatch size of one Adam step",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed of the initialization and the shuffles",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="checkpoints/bc_policy.npz",
        help=(
            "Output checkpoint; a relative path is relative to the workspace"
            " under bazel run"
        ),
    )
    return parser.parse_args(argv)


@dataclasses.dataclass
class BehavioralCloningResult:
    """What train_behavioral_cloning() trained.

    Attributes:
        params: The trained ActorCritic parameters.
        normalizer: The observation normalizer of the demonstrations, which the parameters take their input through.
        epoch_losses: The mean squared error of every epoch, the mean over its minibatches before their steps.
        num_updates: The number of Adam steps taken.
    """

    params: Any
    normalizer: actor_critic.ObservationNormalizer
    epoch_losses: list[float]
    num_updates: int


def train_behavioral_cloning(
    observations: np.ndarray,
    actions: np.ndarray,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int = 0,
    verbose: bool = True,
) -> BehavioralCloningResult:
    """Fits the deterministic policy to the demonstrated actions, one Adam step per shuffled minibatch.

    Args:
        observations: The demonstrations' observations, (num_samples, observation_dim).
        actions: The demonstrated actions in [-1, 1], (num_samples, action_dim).
        epochs: Number of passes over the demonstrations.
        learning_rate: Adam's learning rate.
        batch_size: The samples of one step; the last minibatch of an epoch holds the remainder.
        seed: The seed of the initialization and of every epoch's shuffle.
        verbose: Whether to print the loss five times over the run.

    Returns:
        The trained parameters and normalizer, the loss of every epoch and the number of steps taken.

    Raises:
        ValueError: On mismatched or empty demonstrations, actions outside [-1, 1], or a batch size below one.
    """
    if (
        observations.ndim != 2
        or actions.ndim != 2
        or len(observations) != len(actions)
    ):
        raise ValueError(
            f"observations {observations.shape} and actions {actions.shape} are"
            " not (num_samples, dim) of one length"
        )
    num_samples = len(observations)
    if num_samples == 0:
        raise ValueError("no demonstrations")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if not np.all(np.abs(actions) <= 1.0):
        raise ValueError(
            "the demonstrated actions must be in the policy's range [-1, 1]"
        )

    network = actor_critic.ActorCritic(action_dim=actions.shape[1])
    normalizer = actor_critic.normalizer_from_observations(observations)
    params = network.init(
        jax.random.PRNGKey(seed), jnp.zeros((1, observations.shape[1]))
    )

    tx = optax.adam(learning_rate=learning_rate)
    opt_state = tx.init(params)

    def loss_fn(params: Any, obs: jax.Array, targets: jax.Array) -> jax.Array:
        mean, _, _ = network.apply(params, obs)
        return jnp.mean(jnp.square(jnp.tanh(mean) - targets))

    @jax.jit
    def train_step(
        params: Any,
        opt_state: optax.OptState,
        obs: jax.Array,
        targets: jax.Array,
    ) -> tuple[Any, optax.OptState, jax.Array]:
        loss, grads = jax.value_and_grad(loss_fn)(params, obs, targets)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    normalized = np.asarray(
        normalizer.normalize(np.asarray(observations, dtype=np.float32))
    )
    targets = np.asarray(actions, dtype=np.float32)
    shuffle = np.random.default_rng(seed)

    losses: list[float] = []
    num_updates = 0
    for epoch in range(epochs):
        order = shuffle.permutation(num_samples)
        weighted_loss = 0.0
        for start in range(0, num_samples, batch_size):
            batch = order[start : start + batch_size]
            params, opt_state, loss = train_step(
                params,
                opt_state,
                jnp.asarray(normalized[batch]),
                jnp.asarray(targets[batch]),
            )
            weighted_loss += float(loss) * len(batch)
            num_updates += 1
        losses.append(weighted_loss / num_samples)
        if verbose and (epoch + 1) % max(1, epochs // _LOSS_REPORTS) == 0:
            print(f"Epoch {epoch + 1}/{epochs} - MSE Loss: {losses[-1]:.6f}")
    return BehavioralCloningResult(
        params=params,
        normalizer=normalizer,
        epoch_losses=losses,
        num_updates=num_updates,
    )


def updates_per_epoch(num_samples: int, batch_size: int) -> int:
    """Returns the number of Adam steps of one epoch of train_behavioral_cloning()."""
    return math.ceil(num_samples / batch_size)


def _resolved_output_path(output_path: str) -> str:
    """Returns `output_path`, a relative one taken relative to the workspace under bazel run."""
    if os.path.isabs(output_path):
        return output_path
    workspace = os.environ.get(
        "BUILD_WORKSPACE_DIRECTORY", os.path.abspath(".")
    )
    return os.path.join(workspace, output_path)


def main(argv: Sequence[str] | None = None) -> None:
    """Trains on the demonstrations, or synthetic ones without --demos_path, and writes the checkpoint."""
    args = parse_args(argv)
    output_path = _resolved_output_path(args.output_path)
    print("=" * 60)
    print("🚀 Initializing Behavioral Cloning Pretraining from MPC Rollouts")
    print(f"   Demos Path:   {args.demos_path or '(Synthetic / Stub)'}")
    print(f"   Epochs:       {args.epochs}")
    print(f"   Batch Size:   {args.batch_size}")
    print(f"   Output:       {output_path}")
    print("=" * 60)

    if args.demos_path and h5py.is_hdf5(args.demos_path):
        with h5py.File(args.demos_path, "r") as f:
            observations = np.array(f["observations"])
            actions = np.array(f["actions"])
    else:
        print("ℹ️ Using synthetic demonstrations for warmstart stub.")
        rng = np.random.default_rng(args.seed)
        observations = rng.standard_normal(
            (_SYNTHETIC_SAMPLES, _SYNTHETIC_OBS_DIM)
        ).astype(np.float32)
        actions = rng.uniform(
            -1.0, 1.0, (_SYNTHETIC_SAMPLES, _SYNTHETIC_ACT_DIM)
        ).astype(np.float32)

    result = train_behavioral_cloning(
        observations,
        actions,
        args.epochs,
        args.lr,
        args.batch_size,
        seed=args.seed,
    )
    actor_critic.save_checkpoint(output_path, result.params, result.normalizer)
    print(
        "✅ Behavioral cloning pretraining completed; checkpoint saved to"
        f" {output_path}"
    )


if __name__ == "__main__":
    main()
