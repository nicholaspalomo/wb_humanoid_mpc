"""Angular Center of Mass (aCOM) JAX Neural Models.

Implements Sinusoidal Representation Networks (SIREN) for learning
integrable whole-body orientation coordinates from centroidal angular momentum.
"""

from typing import Any, Callable, Dict, List, Tuple
import jax
import jax.numpy as jnp


def siren_init(
    key: jax.Array,
    in_dim: int,
    out_dim: int,
    is_first: bool = False,
    omega_0: float = 30.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Initialize weights and biases for a SIREN layer with sinusoidal activation."""
    if is_first:
        w_bound = 1.0 / in_dim
    else:
        w_bound = jnp.sqrt(6.0 / in_dim) / omega_0

    k1, k2 = jax.random.split(key)
    w = jax.random.uniform(k1, (out_dim, in_dim), minval=-w_bound, maxval=w_bound)
    b = jax.random.uniform(k2, (out_dim,), minval=-w_bound, maxval=w_bound)
    return w, b


class SirenACOM:
    """SIREN-based Angular Center of Mass network mapping joint positions to orientation offset.

    Delta_theta(q_j) in R^3 represents the whole-body orientation contribution from internal joints.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        out_dim: int = 3,
        omega_0: float = 30.0,
    ):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.omega_0 = omega_0

    def init_params(self, key: jax.Array) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Initialize all layer parameters."""
        params = []
        keys = jax.random.split(key, self.num_layers + 1)

        # First layer
        w, b = siren_init(
            keys[0], self.in_dim, self.hidden_dim, is_first=True, omega_0=self.omega_0
        )
        params.append((w, b))

        # Hidden layers
        for i in range(1, self.num_layers):
            w, b = siren_init(
                keys[i],
                self.hidden_dim,
                self.hidden_dim,
                is_first=False,
                omega_0=self.omega_0,
            )
            params.append((w, b))

        # Output layer (linear readout with small initialization)
        w_bound = jnp.sqrt(6.0 / self.hidden_dim) / self.omega_0
        k_w, k_b = jax.random.split(keys[-1])
        w_out = jax.random.uniform(
            k_w, (self.out_dim, self.hidden_dim), minval=-w_bound, maxval=w_bound
        )
        b_out = jnp.zeros((self.out_dim,))
        params.append((w_out, b_out))

        return params

    def forward(
        self, params: List[Tuple[jnp.ndarray, jnp.ndarray]], q_j: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward pass computing Delta_theta(q_j) in R^3."""
        x = q_j
        for i, (w, b) in enumerate(params[:-1]):
            x = jnp.sin(self.omega_0 * (jnp.dot(w, x) + b))

        w_out, b_out = params[-1]
        out = jnp.dot(w_out, x) + b_out
        return out

    def jacobian_qj(
        self, params: List[Tuple[jnp.ndarray, jnp.ndarray]], q_j: jnp.ndarray
    ) -> jnp.ndarray:
        """Computes J_Delta_theta = d(Delta_theta)/d(q_j) in R^(3 x n_j)."""
        return jax.jacobian(lambda q: self.forward(params, q))(q_j)

    def full_acom_pose(
        self, params: List[Tuple[jnp.ndarray, jnp.ndarray]], q: jnp.ndarray
    ) -> jnp.ndarray:
        """Computes full aCOM orientation theta_aCOM(q) = theta_base + Delta_theta(q_j).

        Args:
            q: generalized coordinates [pos_base (3), rpy_base (3), q_joints (n_j)]
        Returns:
            theta_acom: RPY aCOM orientation in R^3
        """
        rpy_base = q[3:6]
        q_j = q[6:]
        delta_theta = self.forward(params, q_j)
        return rpy_base + delta_theta

    def full_acom_jacobian(
        self, params: List[Tuple[jnp.ndarray, jnp.ndarray]], q: jnp.ndarray
    ) -> jnp.ndarray:
        """Computes full aCOM Jacobian J_aCOM = [0_(3x3), I_(3x3), J_Delta_theta_(3xn_j)]."""
        q_j = q[6:]
        j_delta = self.jacobian_qj(params, q_j)
        n_j = q_j.shape[0]

        zeros_pos = jnp.zeros((3, 3))
        eye_rot = jnp.eye(3)
        return jnp.concatenate([zeros_pos, eye_rot, j_delta], axis=1)
