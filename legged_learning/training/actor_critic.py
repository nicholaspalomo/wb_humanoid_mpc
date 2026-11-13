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


"""The actor-critic network that behavioral cloning warm-starts and the ONNX export serializes, and its checkpoint.

It is the network train_ppo.py defined until PPO training moved to Brax's own networks; bc_warmstart.py and
export_onnx.py are built around its layer names. The deterministic policy is policy_action(): the observation
normalized by an ObservationNormalizer, the actor's mean, squashed by tanh into the action range (-1, 1) as Brax PPO's
deterministic policy squashes its mean. A checkpoint (save_checkpoint(), load_checkpoint()) holds the network's
parameters and the normalizer; bc_warmstart.py writes it and export_onnx.py reads it.
"""

import dataclasses
import os
from typing import Any

from flax import linen
import jax
import jax.numpy as jnp
import numpy as np

# Width of the two shared torso layers.
_HIDDEN_DIM = 256
# The standard deviation below which an observation counts as constant, and is centered but not scaled.
_MIN_OBSERVATION_STD = 1e-6
# The parameter leaves of ActorCritic, as a checkpoint names them: "params/<layer>/<leaf>" and "params/log_std".
CHECKPOINT_PARAMETER_KEYS = (
    "params/Dense_0/kernel",
    "params/Dense_0/bias",
    "params/Dense_1/kernel",
    "params/Dense_1/bias",
    "params/Dense_2/kernel",
    "params/Dense_2/bias",
    "params/Dense_3/kernel",
    "params/Dense_3/bias",
    "params/log_std",
)
_NORMALIZER_MEAN_KEY = "normalizer/mean"
_NORMALIZER_STD_KEY = "normalizer/std"


class ActorCritic(linen.Module):
    """An actor-critic network whose actor and critic share a two-layer torso.

    The torso is two ELU layers (`Dense_0`, `Dense_1`); the actor's mean is `Dense_2` and the critic's value
    `Dense_3`, and the actor's log standard deviation `log_std` is a parameter of its own, independent of the
    observation. The network takes normalized observations (ObservationNormalizer); policy_action() is its
    deterministic policy, which export_onnx.py serializes from `Dense_0` to `Dense_2`.

    Attributes:
        action_dim: The number of actions.
    """

    action_dim: int

    @linen.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Evaluates the actor and the critic.

        Args:
            x: Normalized observations, (batch, observation_dim).

        Returns:
            The actor's mean action before the tanh (batch, action_dim), its log standard deviation (action_dim,), and
            the critic's value (batch,).
        """
        # Shared torso
        h = linen.Dense(_HIDDEN_DIM)(x)
        h = linen.elu(h)
        h = linen.Dense(_HIDDEN_DIM)(h)
        h = linen.elu(h)

        # Actor head (mean & log_std)
        actor_mean = linen.Dense(self.action_dim)(h)
        log_std = self.param(
            "log_std", linen.initializers.zeros, (self.action_dim,)
        )

        # Critic head (value)
        value = linen.Dense(1)(h)

        return actor_mean, log_std, jnp.squeeze(value, axis=-1)


@dataclasses.dataclass(frozen=True)
class ObservationNormalizer:
    """The affine map (observation - mean) / std the policy applies to its observations first.

    Attributes:
        mean: The mean observation, (observation_dim,), float32.
        std: The standard deviation of every observation, (observation_dim,), float32; positive.
    """

    mean: np.ndarray
    std: np.ndarray

    def normalize(self, observations: jax.Array) -> jax.Array:
        """Returns the normalized observations, with the same shape as `observations`."""
        return (observations - self.mean) / self.std


def normalizer_from_observations(
    observations: np.ndarray,
) -> ObservationNormalizer:
    """Returns the normalizer of a data set's observations, (num_samples, observation_dim).

    An observation whose standard deviation is below _MIN_OBSERVATION_STD (a constant one) is centered and not scaled,
    so that the normalizer does not amplify what the data never varied.

    Args:
        observations: The data set's observations, (num_samples, observation_dim).

    Returns:
        The normalizer to the data set's mean and population standard deviation, float32.
    """
    data = np.asarray(observations, dtype=np.float64)
    std = data.std(axis=0)
    std = np.where(std > _MIN_OBSERVATION_STD, std, 1.0)
    return ObservationNormalizer(
        mean=data.mean(axis=0).astype(np.float32), std=std.astype(np.float32)
    )


def policy_action(
    network: ActorCritic,
    params: Any,
    normalizer: ObservationNormalizer,
    observations: jax.Array,
) -> jax.Array:
    """Returns the deterministic policy's actions tanh(mean(normalize(observations))), (batch, action_dim), in (-1, 1)."""
    mean, _, _ = network.apply(params, normalizer.normalize(observations))
    return jnp.tanh(mean)


def save_checkpoint(
    path: str, params: Any, normalizer: ObservationNormalizer
) -> None:
    """Writes the network's parameters and the normalizer to an .npz file at `path`, keyed as load_checkpoint() reads.

    Args:
        path: The file to write; its directory is created.
        params: ActorCritic parameters, as `ActorCritic.init` returns them.
        normalizer: The normalizer the parameters were trained on.
    """
    arrays = {
        _NORMALIZER_MEAN_KEY: np.asarray(normalizer.mean),
        _NORMALIZER_STD_KEY: np.asarray(normalizer.std),
    }
    for key in CHECKPOINT_PARAMETER_KEYS:
        leaf: Any = params
        for name in key.split("/"):
            leaf = leaf[name]
        arrays[key] = np.asarray(leaf)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as f:
        np.savez(f, **arrays)


def load_checkpoint(path: str) -> tuple[dict[str, Any], ObservationNormalizer]:
    """Reads a checkpoint save_checkpoint() wrote.

    Args:
        path: The .npz file.

    Returns:
        The ActorCritic parameters, as `ActorCritic.init` returns them, and the normalizer.

    Raises:
        KeyError: When the file lacks one of the parameters or the normalizer, naming every missing key.
        ValueError: When the normalizer's standard deviation is not positive and finite.
    """
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    required = (
        *CHECKPOINT_PARAMETER_KEYS,
        _NORMALIZER_MEAN_KEY,
        _NORMALIZER_STD_KEY,
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"checkpoint {path} lacks {', '.join(missing)}")
    params: dict[str, Any] = {}
    for key in CHECKPOINT_PARAMETER_KEYS:
        names = key.split("/")
        node = params
        for name in names[:-1]:
            node = node.setdefault(name, {})
        node[names[-1]] = jnp.asarray(arrays[key])
    normalizer = ObservationNormalizer(
        mean=arrays[_NORMALIZER_MEAN_KEY].astype(np.float32),
        std=arrays[_NORMALIZER_STD_KEY].astype(np.float32),
    )
    if not np.all(np.isfinite(normalizer.std) & (normalizer.std > 0.0)):
        raise ValueError(
            f"checkpoint {path}: the normalizer's std is not positive and"
            " finite"
        )
    return params, normalizer
