"""MuJoCo Playground and MJX training examples and tutorials."""

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
