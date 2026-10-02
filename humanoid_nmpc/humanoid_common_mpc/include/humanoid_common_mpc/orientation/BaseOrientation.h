/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#pragma once

#include <Eigen/Core>

#include <ocs2_core/manifold/UnitQuaternionMath.h>

#include "humanoid_common_mpc/common/Types.h"

/*
 * The base orientation of the humanoid MPCs as a unit quaternion: the safe normalization every function of the state
 * reads the quaternion through, the lift of tangent Jacobians to the stored coefficients, and the heading-tilt split that
 * keeps the Euler-space tuning of the base orientation (task.yaml Q, Q_final, Q_acom rows yaw, pitch, roll) meaningful on
 * a quaternion state. Sections 2.1, 2.6 and 2.8.1 of humanoid_nmpc/docs/quaternion_base_orientation/README.md derive
 * everything here.
 *
 * Conventions, the same as RobotState::getRootRotationLocalToWorldFrame():
 *   - A quaternion is a vector4_t of coefficients xi = (x, y, z, w), Eigen's coeffs() order and Pinocchio's
 *     configuration order, with the Hamilton product. R(xi) maps base to world. Build an Eigen quaternion from the
 *     coefficient vector, never from four scalars: that constructor takes (w, x, y, z).
 *   - A rotation is perturbed on the right, in the body frame: xi (x) Exp(dphi). The "body tangent" Jacobian of a
 *     function g is dg/d(dphi) at dphi = 0.
 *   - Every function of the orientation reads the quaternion through safelyNormalizedQuaternion (design decision D5), so
 *     it is invariant to the scale of xi and maps the zero quaternion to the identity rotation instead of NaN. The one
 *     exception is the kinematic row d(xi)/dt = 1/2 G(xi) w_B of the flow maps, which uses the raw xi.
 *
 * The quaternion algebra itself - the rate matrix G(xi) (ocs2::quaternionRateMatrix), the Hamilton product, Exp, Log,
 * the right Jacobians and slerp - is lib/ocs2's ocs2_core/manifold/UnitQuaternionMath.h, which this header includes, so
 * the humanoid code and the manifold solver share one implementation of each rule. Nothing here is Euler: the
 * conversions to and from ZYX Euler angles are in orientation/EulerBoundary.h.
 */

namespace ocs2::humanoid {

/******************************************************************************************************/
/* Safe normalization (design section 2.6)                                                            */
/******************************************************************************************************/

/**
 * epsilon of design section 2.6: a quaternion with |xi| <= epsilon is read as the zero quaternion, which is the identity.
 * The manifold solver's ocs2::kQuaternionNormalizationEpsilon, so that the solver and the humanoid terms normalize alike.
 */
inline constexpr scalar_t kQuaternionNormGuard = kQuaternionNormalizationEpsilon;

/**
 * n_safe(xi) = |xi| if |xi|^2 > epsilon^2, else 1: the divisor of the safe normalization.
 *
 * The comparison is on the squared norm and the square root is taken of the selected branch. On an AD scalar the
 * comparison is a conditional expression (CppAD::CondExpGt), which tapes no comparison between variables, so CppADCodeGen
 * can generate code for it, and both branches have bounded derivatives: guarding the norm instead,
 * CondExpGt(|xi|, epsilon, |xi|, 1), has a NaN reverse-mode Jacobian at the zero quaternion in generated code (the square
 * root's reverse sweep multiplies the zero partial of the unselected branch by 1 / (2 |xi|); Step 0,
 * test/testQuaternionRootJointCppAd.cpp). Never call Eigen's normalized() / normalize() on an AD scalar: they record a
 * comparison between variables.
 *
 * The rule is ocs2::quaternionSafeNorm's; these overloads are the humanoid formulation's entry points to it, for the two
 * scalar types the MPCs use.
 */
scalar_t safeQuaternionNorm(const vector4_t& xi);
ad_scalar_t safeQuaternionNorm(const ad_vector4_t& xi);

/**
 * xi_hat = xi / n_safe(xi). A unit quaternion for every |xi| > epsilon; the zero quaternion stays zero, which the
 * polynomial rotation matrix of a quaternion (Eigen's toRotationMatrix()) and every function in this file read as the
 * identity. At the CppAD tape point x = ones it is the unit quaternion (1, 1, 1, 1) / 2. -xi gives exactly -xi_hat.
 */
vector4_t safelyNormalizedQuaternion(const vector4_t& xi);
ad_vector4_t safelyNormalizedQuaternion(const ad_vector4_t& xi);

/******************************************************************************************************/
/* The configuration tangent lift (design section 2.1)                                                */
/******************************************************************************************************/

/**
 * E^+(xi) = 2 G^T(xi_hat) / n_safe(xi) = 2 G^T(xi) / n_safe(xi)^2, the 3 x 4 left inverse of the map E(xi) = 1/2 G(xi)
 * from a body right perturbation to the quaternion coefficients: E^+ E = I_3 for every |xi| > epsilon, and E^+ xi = 0.
 *
 * It lifts a body tangent Jacobian J_t of a function g of xi_hat to the ambient Jacobian with respect to the stored
 * coefficients, J_a = J_t E^+ (design decision D4): g is invariant to the scale of xi, so its ambient gradient has no
 * radial part, and J_t E^+ is exactly that gradient. On the unit sphere E^+ = 2 G^T(xi). It is zero, not NaN, at the
 * zero quaternion. ocs2::quaternionTangentMapPseudoInverse, which the manifold solver's pull-backs use.
 */
Eigen::Matrix<scalar_t, 3, 4> configurationTangentLift(const vector4_t& xi);

/******************************************************************************************************/
/* Heading and tilt (design section 2.8.1)                                                            */
/******************************************************************************************************/

/** Below this norm n = |s_xy| of the swing's vector part, the tilt is evaluated with its series (design section 2.8.1). */
inline constexpr scalar_t kTiltSeriesThreshold = 1e-6;
/**
 * The twist norm rho = |(xi_z, xi_w)| is at least this in every division (the upside-down guard of design section
 * 2.8.1). The heading is undefined, and its derivatives unbounded, only upside down (rho = 0): there the values stay
 * finite, the heading is read as zero, and the Jacobians are bounded by about 1 / kMinimumTwistNorm.
 */
inline constexpr scalar_t kMinimumTwistNorm = 1e-9;

/**
 * The heading psi_h(xi) = 2 atan2(xi_z, xi_w): the angle of the twist about the world z axis in the twist-swing split
 * xi = q_t (x) s (design decision D9), in (-pi, pi].
 *
 * (xi_z, xi_w) is first brought to xi_w >= 0, so psi_h(-xi) = psi_h(xi) exactly and no wrap is needed; the design writes
 * the raw 2 atan2, which differs from this by a multiple of 2 pi. For a level attitude q_z(psi) it is psi wrapped to
 * (-pi, pi]; for ZYX Euler angles (psi, theta, phi) it is psi - 2 atan(tan(theta / 2) tan(phi / 2)), so it equals the
 * ZYX yaw at single-axis tilt and differs from it by about -theta phi / 2 otherwise. Unlike the ZYX yaw it is regular at
 * a pitch of 90 degrees; it is singular only upside down, where it is zero. Invariant to the scale of xi.
 */
scalar_t twistHeading(const vector4_t& xi);

/**
 * The heading quaternion q_t(xi) = (0, 0, xi_z, xi_w) / rho of xi_hat: the rotation about the world z axis by
 * twistHeading(xi), in the same hemisphere as xi (so q_t(-xi) = -q_t(xi)). A unit quaternion for every xi; upside down
 * (rho below kMinimumTwistNorm) it is computed from the angle instead of by the division, and is the identity at rho = 0.
 */
vector4_t headingQuaternion(const vector4_t& xi);

/**
 * The swing s(xi) = q_t(xi)^-1 (x) xi_hat = (s_x, s_y, 0, s_w), s_w = rho >= 0: the tilt of the base relative to its
 * heading frame, a rotation about a horizontal axis of that frame, with xi_hat = q_t (x) s. s(-xi) = s(xi) exactly, and
 * s(q_z(alpha) (x) xi) = s(xi) for every rotation q_z(alpha) about the world z axis.
 */
vector4_t swingQuaternion(const vector4_t& xi);

/**
 * The tilt tau(xi) = Log(s(xi))_xy = kappa(n, rho) s_xy, the rotation vector of the swing in the heading frame, as
 * (tau_x, tau_y): (roll, pitch) about the heading frame's axes. kappa = 2 atan2(n, rho) / n with n = |s_xy|, never
 * sqrt(1 - rho^2), and its series (2 / rho)(1 - n^2 / (3 rho^2)) below kTiltSeriesThreshold, so it is exactly zero and
 * not 0 / 0 at a level attitude (the judges' fix of design section 2.8.1). |tau| <= pi, with pi only upside down.
 *
 * For a single-axis tilt it is exact: tau(q_z(psi) (x) q_y(theta)) = (0, theta) and tau(q_z(psi) (x) q_x(phi)) = (phi, 0).
 * It is exactly independent of the heading, and therefore of a yaw error. Invariant to the scale and the sign of xi.
 */
vector2_t tiltVector(const vector4_t& xi);

/**
 * The heading-tilt residual of xi against xi_r in the rows of the Euler tuning layout (yaw, pitch, roll), design
 * decision D7 and section 2.8.1:
 *
 *   e_HT(xi, xi_r) = [ wrap_pi(psi_h(xi) - psi_h(xi_r)),  tau_y(xi) - tau_y(xi_r),  tau_x(xi) - tau_x(xi_r) ].
 *
 * The yaw row is wrapped to (-pi, pi] and is discontinuous only at a yaw error of exactly +-pi. The tilt rows are exactly
 * independent of the yaw error, unlike the components of Log(R_r^T R), which leak tilt between the axes under a yaw
 * error. Invariant to the scale and the sign of xi and of xi_r.
 */
vector3_t headingTiltResidual(const vector4_t& xi, const vector4_t& xiReference);

/**
 * J_HT(xi) = d e_HT(xi (x) Exp(dphi), xi_r) / d(dphi) at dphi = 0, the 3 x 3 Jacobian of the heading-tilt residual with
 * respect to the body right perturbation of xi (it does not depend on xi_r). Closed form, from the ambient gradients
 * [grad psi_h; grad tau_y; grad tau_x] of design section 2.8.1 times 1/2 G(xi_hat):
 *
 *   grad psi_h = (2 / rho^2) (0, 0, w, -z),     N = (w x + z y, w y - z x),     s_xy = N / rho,
 *   dN_x = (w, z, y, x),   dN_y = (-z, w, -x, y),   d rho = (0, 0, z, w) / rho,   d s_xy = (dN - s_xy d rho) / rho,
 *   d kappa / dn = (2 rho / (n^2 + rho^2) - kappa) / n  (series -4 n / (3 rho^3)),   d kappa / d rho = -2 / (n^2 + rho^2),
 *   d tau = kappa d s_xy + s_xy (d kappa / dn  s_xy^T d s_xy / n  +  d kappa / d rho  d rho).
 *
 * At every level attitude q_z(psi) it is the permutation P_3 = antidiag(1, 1, 1), so J_HT T_B(0) = I and the
 * Gauss-Newton Hessians of the Euler-weighted rows equal those of the Euler formulation at a level STATE, whatever the
 * reference. At a tilted state the Euler-chart Jacobian J_HT(phi(Theta)) T_B(Theta) is not I: it deviates by about
 * |tilt| / 2 (the yaw row picks up -tan(theta / 2) and -tan(phi / 2)), even against a level reference. It is finite
 * and of full rank at a pitch of 90 degrees (singular values about 1.95, 1.00, 0.80). Upside down it stays finite,
 * bounded through kMinimumTwistNorm. The ambient Jacobian with respect to the stored coefficients is
 * J_HT(xi) * configurationTangentLift(xi).
 */
matrix3_t headingTiltJacobian(const vector4_t& xi);

/** The heading-tilt residual and its body tangent Jacobian, as the tuning deviations need them together. */
struct HeadingTiltError {
  /** e_HT(xi, xi_r), rows (yaw, pitch, roll); see headingTiltResidual. */
  vector3_t residual;
  /** J_HT(xi) = d residual / d(dphi) for the body right perturbation dphi of xi; see headingTiltJacobian. */
  matrix3_t jacobianBodyTangent;
};

/** headingTiltResidual(xi, xi_r) and headingTiltJacobian(xi), sharing the split of xi. */
HeadingTiltError headingTiltError(const vector4_t& xi, const vector4_t& xiReference);

}  // namespace ocs2::humanoid
