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

"""Angular Center of Mass (aCOM) JAX Neural Models.

Implements Sinusoidal Representation Networks (SIREN) for learning
integrable whole-body orientation coordinates from centroidal angular momentum.
"""

from typing import TypeAlias

import jax
import jax.numpy as jnp

# The parameters of a SirenACOM: one (weight, bias) pair per layer, the linear readout last.
SirenParams: TypeAlias = list[tuple[jnp.ndarray, jnp.ndarray]]


def siren_init(
    key: jax.Array,
    in_dim: int,
    out_dim: int,
    is_first: bool = False,
    omega_0: float = 30.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Initialize weights and biases for a SIREN layer with sinusoidal activation."""
    if is_first:
        w_bound = 1.0 / in_dim
    else:
        w_bound = jnp.sqrt(6.0 / in_dim) / omega_0

    k1, k2 = jax.random.split(key)
    w = jax.random.uniform(k1, (out_dim, in_dim), minval=-w_bound, maxval=w_bound)
    b = jax.random.uniform(k2, (out_dim,), minval=-w_bound, maxval=w_bound)
    return w, b


# At or below this squared angle Exp uses its series, as ocs2::quaternionExp does (UnitQuaternionMath.h).
_EXP_SERIES_THRESHOLD_SQUARED = 1.0e-12
# At or below this squared angle the right Jacobian's (a - sin a) / a^3 uses its series, as ocs2::so3RightJacobian does.
_JACOBIAN_SERIES_THRESHOLD_SQUARED = 1.0e-2


def quaternion_product(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Returns the Hamilton product a (x) b of quaternion coefficients (x, y, z, w)."""
    a_vector, a_scalar = a[:3], a[3]
    b_vector, b_scalar = b[:3], b[3]
    vector = a_scalar * b_vector + b_scalar * a_vector + jnp.cross(a_vector, b_vector)
    return jnp.concatenate(
        [vector, jnp.reshape(a_scalar * b_scalar - jnp.dot(a_vector, b_vector), (1,))]
    )


def quaternion_exp(phi: jnp.ndarray) -> jnp.ndarray:
    """Returns Exp(phi), the unit quaternion (x, y, z, w) rotating by |phi| about phi."""
    angle_squared = jnp.dot(phi, phi)
    small = angle_squared <= _EXP_SERIES_THRESHOLD_SQUARED
    angle = jnp.sqrt(jnp.where(small, 1.0, angle_squared))
    # sin(a / 2) / a = 1/2 - a^2/48 + a^4/3840, cos(a / 2) = 1 - a^2/8 + a^4/384.
    vector_scale = jnp.where(
        small,
        0.5 - angle_squared / 48.0 + angle_squared**2 / 3840.0,
        jnp.sin(0.5 * angle) / angle,
    )
    scalar = jnp.where(
        small,
        1.0 - angle_squared / 8.0 + angle_squared**2 / 384.0,
        jnp.cos(0.5 * angle),
    )
    return jnp.concatenate([vector_scale * phi, jnp.reshape(scalar, (1,))])


def quaternion_rotation_matrix(xi: jnp.ndarray) -> jnp.ndarray:
    """Returns R(xi), the rotation (body to world) of unit quaternion coefficients (x, y, z, w)."""
    x, y, z, w = xi[0], xi[1], xi[2], xi[3]
    return jnp.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _cross_product_matrix(v: jnp.ndarray) -> jnp.ndarray:
    """Returns [v]x, the matrix of the cross product with v."""
    return jnp.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def so3_right_jacobian(phi: jnp.ndarray) -> jnp.ndarray:
    """Returns the right Jacobian of SO(3), Exp(phi + d) = Exp(phi) (x) Exp(J_r(phi) d) to first order.

    J_r(phi) = I - (1 - cos a) / a^2 [phi]x + (a - sin a) / a^3 [phi]x^2 with
    a = |phi|, the coefficients by their series near zero, as
    ocs2::so3RightJacobian computes them.

    Args:
        phi: A rotation vector.

    Returns:
        The (3, 3) right Jacobian.
    """
    angle_squared = jnp.dot(phi, phi)
    exp_small = angle_squared <= _EXP_SERIES_THRESHOLD_SQUARED
    jacobian_small = angle_squared <= _JACOBIAN_SERIES_THRESHOLD_SQUARED
    angle = jnp.sqrt(jnp.where(exp_small, 1.0, angle_squared))
    angle_for_second = jnp.sqrt(jnp.where(jacobian_small, 1.0, angle_squared))
    sin_half = jnp.sin(0.5 * angle)
    first = jnp.where(
        exp_small,
        0.5 - angle_squared / 24.0 + angle_squared**2 / 720.0,
        2.0 * sin_half * sin_half / (angle * angle),
    )
    second = jnp.where(
        jacobian_small,
        1.0 / 6.0
        - angle_squared / 120.0
        + angle_squared**2 / 5040.0
        - angle_squared**3 / 362880.0
        + angle_squared**4 / 39916800.0,
        (angle_for_second - jnp.sin(angle_for_second)) / angle_for_second**3,
    )
    cross = _cross_product_matrix(phi)
    return jnp.eye(3) - first * cross + second * jnp.dot(cross, cross)


class SirenACOM:
    """SIREN-based Angular Center of Mass network mapping joint positions to orientation offset.

    Delta_theta(q_j) in R^3 represents the whole-body orientation contribution from internal joints.

    The network has `num_layers` sinusoidal layers followed by a linear readout,
    so `init_params` returns `num_layers + 1` weight/bias pairs. The C++ evaluator
    counts the readout as a layer, so the exported header's `num_layers` is one
    greater than this one; see `export_acom.export_to_cpp_header`.

    Attributes:
        in_dim: Number of joints the network takes as input.
        hidden_dim: Width of every sinusoidal layer.
        num_layers: Number of sinusoidal layers, excluding the linear readout.
            Defaults to 2, the only count the C++ evaluator loads (see
            train_main._CPP_SUPPORTED_NUM_LAYERS); a 3 here exports a header that
            fails to compile.
        out_dim: Output dimension, 3 for an orientation offset.
        omega_0: Frequency scaling applied inside every sinusoidal activation.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        out_dim: int = 3,
        omega_0: float = 30.0,
    ):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.omega_0 = omega_0

    def init_params(self, key: jax.Array) -> SirenParams:
        """Initialize all layer parameters."""
        params: SirenParams = []
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
        k_w, _ = jax.random.split(keys[-1])  # The readout's bias starts at zero.
        w_out = jax.random.uniform(
            k_w, (self.out_dim, self.hidden_dim), minval=-w_bound, maxval=w_bound
        )
        b_out = jnp.zeros((self.out_dim,))
        params.append((w_out, b_out))

        return params

    def forward(self, params: SirenParams, q_j: jnp.ndarray) -> jnp.ndarray:
        """Forward pass computing Delta_theta(q_j) in R^3."""
        x = q_j
        for w, b in params[:-1]:
            x = jnp.sin(self.omega_0 * (jnp.dot(w, x) + b))

        w_out, b_out = params[-1]
        return jnp.dot(w_out, x) + b_out

    def jacobian_qj(self, params: SirenParams, q_j: jnp.ndarray) -> jnp.ndarray:
        """Computes J_Delta_theta = d(Delta_theta)/d(q_j) in R^(3 x n_j)."""
        return jax.jacobian(lambda q: self.forward(params, q))(q_j)

    # The C++ evaluator the MPC runs computes the same composition and tangent Jacobian.
    # LINT.IfChange(acom_attitude)
    def acom_attitude(
        self, params: SirenParams, xi: jnp.ndarray, q_j: jnp.ndarray
    ) -> jnp.ndarray:
        """Computes the ACoM attitude xi_a = xi_hat (x) Exp(Delta_theta(q_j)).

        The base orientation composed with the network's rotation, as
        AngularCenterOfMass::acomAttitude does in C++ (design section 2.8.3 of
        humanoid_nmpc/docs/quaternion_base_orientation/README.md). Delta_theta
        is the rotation vector, in the base frame, the network emits: its
        components are about the base's x, y and z axes.

        Args:
            params: The network parameters.
            xi: The base orientation, quaternion coefficients (x, y, z, w) of a
                nonzero quaternion; it is normalized here.
            q_j: The joint positions, in the network's joint order.

        Returns:
            The ACoM attitude, unit quaternion coefficients (x, y, z, w).
        """
        xi_hat = xi / jnp.linalg.norm(xi)
        return quaternion_product(xi_hat, quaternion_exp(self.forward(params, q_j)))

    def acom_attitude_tangent_jacobian(
        self, params: SirenParams, q_j: jnp.ndarray
    ) -> jnp.ndarray:
        """Computes the body tangent Jacobian of acom_attitude.

        When the base orientation moves to xi_hat (x) Exp(dphi) and the joints
        to q_j + dq_j, the attitude moves to xi_a (x) Exp(dphi_a) with, to first
        order, dphi_a = [R(Delta_theta)^T, J_r(Delta_theta) J_Delta_theta]
        [dphi; dq_j]: the adjoint of the network's rotation on the base and the
        right Jacobian of SO(3) times the network's Jacobian on the joints, the
        two blocks of AngularCenterOfMass::acomAttitudeTangentJacobian in C++.
        Neither depends on the base orientation.

        Args:
            params: The network parameters.
            q_j: The joint positions, in the network's joint order.

        Returns:
            The (3, 3 + n_j) Jacobian, columns (dphi of the base, dq_j).
        """
        delta_theta = self.forward(params, q_j)
        base_block = quaternion_rotation_matrix(quaternion_exp(delta_theta)).T
        joint_block = jnp.dot(
            so3_right_jacobian(delta_theta), self.jacobian_qj(params, q_j)
        )
        return jnp.concatenate([base_block, joint_block], axis=1)

    # LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/acom/AngularCenterOfMass.cpp:acom_attitude)
