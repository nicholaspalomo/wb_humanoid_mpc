"""Reinforcement learning training pipelines and behavioral cloning warmstart."""

import jax
import jax.numpy as jnp

# Backward compatibility polyfill for Brax PPO multi-device replication on newer JAX versions
if not hasattr(jax, "device_put_replicated"):

    def _device_put_replicated(x, devices):
        def _replicate_leaf(leaf):
            arr = jnp.asarray(leaf)
            return jax.device_put(jnp.broadcast_to(arr, (len(devices),) + arr.shape))

        return jax.tree_util.tree_map(_replicate_leaf, x)

    jax.device_put_replicated = _device_put_replicated

from humanoid_learning.training.bootstrap_bc import bootstrap_policy
from humanoid_learning.training.offline_dataset import (
    OfflineTrajectoryDataset,
    TrajectoryBatch,
)
from humanoid_learning.training.policy_network import HumanoidResidualPolicy
from humanoid_learning.training.train_grpo import (
    GRPORolloutBatch,
    HumanoidGRPOConfig,
    HumanoidGRPOTrainer,
    train_humanoid_grpo,
)

__all__ = [
    "OfflineTrajectoryDataset",
    "TrajectoryBatch",
    "HumanoidResidualPolicy",
    "bootstrap_policy",
    "HumanoidGRPOTrainer",
    "HumanoidGRPOConfig",
    "GRPORolloutBatch",
    "train_humanoid_grpo",
]
