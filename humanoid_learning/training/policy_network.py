"""Gaussian Residual Policy Network for Humanoid Whole-Body Locomotion."""

from typing import Tuple

from flax import nnx
import jax
import jax.numpy as jnp

# ==============================================================================
# Policy Network Architectural Defaults
# ==============================================================================
DEFAULT_POLICY_OBS_DIM: int = 40
DEFAULT_POLICY_ACT_DIM: int = 12
DEFAULT_POLICY_HIDDEN_DIM: int = 256
DEFAULT_POLICY_NUM_BLOCKS: int = 3
DEFAULT_POLICY_INIT_LOG_STD: float = -1.0

# Numerical Clamping Limits
LOG_STD_MIN_CLIP: float = -5.0
LOG_STD_MAX_CLIP: float = 1.0
ACTION_BOUND_LOWER: float = -1.0
ACTION_BOUND_UPPER: float = 1.0
KL_NUMERICAL_TOLERANCE_EPS: float = 1e-8


class ResidualMLPBlock(nnx.Module):
    """Residual Dense MLP block with LayerNorm and SiLU."""

    def __init__(self, hidden_dim: int, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(hidden_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        h = nnx.silu(self.dense1(x))
        h = self.dense2(h)
        return self.norm(h + residual)


class HumanoidResidualPolicy(nnx.Module):
    """Stochastic Gaussian Policy outputting joint position residuals for Whole-Body Control.

    Computes action distribution pi_theta(Delta q | s) = N(mu(s), diag(sigma^2))
    with tanh squashing to [-1, 1] bounds.
    """

    def __init__(
        self,
        obs_dim: int = DEFAULT_POLICY_OBS_DIM,
        act_dim: int = DEFAULT_POLICY_ACT_DIM,
        hidden_dim: int = DEFAULT_POLICY_HIDDEN_DIM,
        num_blocks: int = DEFAULT_POLICY_NUM_BLOCKS,
        init_log_std: float = DEFAULT_POLICY_INIT_LOG_STD,
        *,
        rngs: nnx.Rngs,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Encoder input projection
        self.input_proj = nnx.Linear(obs_dim, hidden_dim, rngs=rngs)

        # Residual backbone blocks in nnx.List container
        self.blocks = nnx.List(
            [ResidualMLPBlock(hidden_dim, rngs=rngs) for _ in range(num_blocks)]
        )

        # Action mean output head
        self.mean_head = nnx.Linear(hidden_dim, act_dim, rngs=rngs)

        # State-independent learnable log standard deviation parameter
        self.log_std = nnx.Param(
            jnp.full((act_dim,), init_log_std, dtype=jnp.float32)
        )

    def __call__(self, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Compute Gaussian distribution mean and standard deviation.

        Args:
            obs: Observation tensor [..., obs_dim]

        Returns:
            Tuple of (mean [..., act_dim], std [..., act_dim])
        """
        h = nnx.silu(self.input_proj(obs))
        for block in self.blocks:
            h = block(h)
        mean = jnp.tanh(self.mean_head(h))
        log_std = jnp.clip(self.log_std[:], LOG_STD_MIN_CLIP, LOG_STD_MAX_CLIP)
        std = jnp.exp(log_std)
        return mean, std

    def sample_action(
        self, rng: jax.Array, obs: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Sample actions using reparameterization and evaluate log probabilities.

        Args:
            rng: PRNGKey for stochastic Gaussian action sampling
            obs: State observation tensor [..., obs_dim]

        Returns:
            Tuple of (action [..., act_dim], log_prob [...])
        """
        mean, std = self(obs)
        eps = jax.random.normal(rng, shape=mean.shape)
        raw_action = mean + std * eps
        action = jnp.clip(raw_action, ACTION_BOUND_LOWER, ACTION_BOUND_UPPER)

        # Log probability density under diagonal Gaussian
        var = jnp.square(std)
        log_prob = -0.5 * (
            jnp.sum(jnp.square(raw_action - mean) / (var + 1e-8), axis=-1)
            + jnp.sum(2.0 * jnp.log(std) + jnp.log(2.0 * jnp.pi), axis=-1)
        )
        return action, log_prob

    def log_prob(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Evaluate log probability of given actions under current policy."""
        mean, std = self(obs)
        var = jnp.square(std)
        log_p = -0.5 * (
            jnp.sum(jnp.square(action - mean) / (var + 1e-8), axis=-1)
            + jnp.sum(2.0 * jnp.log(std) + jnp.log(2.0 * jnp.pi), axis=-1)
        )
        return log_p

    def evaluate_log_prob(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Evaluate log probability of given actions."""
        return self.log_prob(obs, action)

    def kl_divergence(
        self, other_policy: "HumanoidResidualPolicy", obs: jax.Array
    ) -> jax.Array:
        """Analytical KL divergence D_KL(pi_current || pi_other) between two diagonal Gaussians."""
        mu_p, std_p = self(obs)
        mu_q, std_q = other_policy(obs)

        var_p = jnp.square(std_p)
        var_q = jnp.square(std_q)

        kl = jnp.sum(
            jnp.log(std_q / (std_p + 1e-8))
            + (var_p + jnp.square(mu_p - mu_q)) / (2.0 * var_q + 1e-8)
            - 0.5,
            axis=-1,
        )
        return jnp.maximum(0.0, kl)

    def evaluate_kl(
        self, obs: jax.Array, other_policy: "HumanoidResidualPolicy"
    ) -> jax.Array:
        """Evaluate analytical KL divergence given observation tensor and reference policy."""
        return self.kl_divergence(other_policy, obs)
