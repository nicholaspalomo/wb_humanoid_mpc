"""Offline Trajectory Dataset Loader for Humanoid MPC & Kinematic Demonstrations."""

import os
from typing import Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np

try:
    import h5py

    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


# ==============================================================================
# Offline Dataset Defaults
# ==============================================================================
DEFAULT_DATASET_OBS_DIM: int = 40
DEFAULT_DATASET_ACT_DIM: int = 12
DEFAULT_SYNTHETIC_SAMPLE_COUNT: int = 2000
DEFAULT_MINIBATCH_SIZE: int = 64


class TrajectoryBatch(NamedTuple):
    """Batch of state-action demonstration transitions."""

    observations: jax.Array  # [B, obs_dim]
    actions: jax.Array  # [B, act_dim]
    rewards: jax.Array  # [B]
    masks: jax.Array  # [B]


class OfflineTrajectoryDataset:
    """Offline Dataset containing whole-body MPC demonstrations or reference trajectories."""

    def __init__(
        self,
        filepath: str | None = None,
        obs_dim: int = DEFAULT_DATASET_OBS_DIM,
        act_dim: int = DEFAULT_DATASET_ACT_DIM,
        synthetic_samples: int = DEFAULT_SYNTHETIC_SAMPLE_COUNT,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        loaded = False
        if filepath and os.path.exists(filepath):
            if filepath.endswith(".npz"):
                data = np.load(filepath)
                self.observations = np.array(
                    data["observations"], dtype=np.float32
                )
                self.actions = np.array(data["actions"], dtype=np.float32)
                self.rewards = (
                    np.array(data["rewards"], dtype=np.float32)
                    if "rewards" in data
                    else np.ones((len(self.observations),), dtype=np.float32)
                )
                loaded = True
                print(
                    f"✓ Loaded {len(self.observations)} offline demonstration transitions from NPZ '{filepath}'."
                )
            elif _HAS_H5PY and h5py.is_hdf5(filepath):
                with h5py.File(filepath, "r") as f:
                    self.observations = np.array(
                        f["observations"], dtype=np.float32
                    )
                    self.actions = np.array(f["actions"], dtype=np.float32)
                    if "rewards" in f:
                        self.rewards = np.array(f["rewards"], dtype=np.float32)
                    else:
                        self.rewards = np.ones(
                            (len(self.observations),), dtype=np.float32
                        )
                loaded = True
                print(
                    f"✓ Loaded {len(self.observations)} offline demonstration transitions from HDF5 '{filepath}'."
                )

        if not loaded:
            # Generate synthetic offline reference demonstration dataset
            print(
                f"ℹ️ Generating {synthetic_samples} synthetic reference transitions for offline bootstrapping."
            )
            rng = np.random.RandomState(42)
            # Smooth reference states around nominal standing & walking
            self.observations = (
                rng.randn(synthetic_samples, obs_dim).astype(np.float32) * 0.1
            )
            # Near-zero residual targets (matching nominal reference walking gait)
            self.actions = (
                rng.randn(synthetic_samples, act_dim).astype(np.float32) * 0.05
            )
            self.rewards = np.ones((synthetic_samples,), dtype=np.float32)

        self.size = len(self.observations)

    def __len__(self) -> int:
        """Return total number of dataset transitions."""
        return len(self.observations)

    def sample_batch(
        self, rng: jax.Array, batch_size: int = DEFAULT_MINIBATCH_SIZE
    ) -> TrajectoryBatch:
        """Sample random minibatch of transitions."""
        indices = jax.random.randint(
            rng, (batch_size,), minval=0, maxval=self.size
        )
        return TrajectoryBatch(
            observations=jnp.array(self.observations[indices]),
            actions=jnp.array(self.actions[indices]),
            rewards=jnp.array(self.rewards[indices]),
            masks=jnp.ones((batch_size,), dtype=jnp.float32),
        )

    def save_dataset(self, output_path: str):
        """Save dataset to NPZ or HDF5 file."""
        os.makedirs(
            os.path.dirname(os.path.abspath(output_path)), exist_ok=True
        )
        if output_path.endswith(".h5") or output_path.endswith(".hdf5"):
            if _HAS_H5PY:
                with h5py.File(output_path, "w") as f:
                    f.create_dataset("observations", data=self.observations)
                    f.create_dataset("actions", data=self.actions)
                    f.create_dataset("rewards", data=self.rewards)
                print(f"✓ Dataset saved to '{output_path}'.")
                return
        np.savez(
            output_path,
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
        )
        print(f"✓ Dataset saved to '{output_path}'.")
