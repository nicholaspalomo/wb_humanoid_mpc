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

#include <cmath>

#include <Eigen/Core>

/*
 * Unit-quaternion algebra for a rotation stored in a state vector.
 *
 * Conventions (humanoid_nmpc/docs/quaternion_base_orientation/README.md, section 2.1):
 *  - A quaternion is a coefficient vector xi = (x, y, z, w), the order of Eigen's coeffs() and of Pinocchio's
 *    configuration vector, with the Hamilton product. Build quaternions from coefficient vectors only: Eigen's
 *    four-scalar constructor takes (w, x, y, z).
 *  - R(xi) maps the body frame to the world frame, and a tangent vector delta is a body (right) perturbation:
 *    xi (+) delta = xi (x) Exp(delta).
 *  - Exp and Log map between rotation vectors and unit quaternions. Log takes the shortest path (it returns a rotation
 *    vector of norm at most pi), so every function here is invariant under xi -> -xi (the double cover).
 *
 * Every function is a template on the scalar, so the same code runs in double and on a CppAD tape. On a tape, no branch
 * may be a plain `if` (the tape would record only the branch taken at the taping point): branches go through
 * ifGreaterThan(), which is a CppAD conditional expression for AD scalars, and both of its branches are kept finite at
 * every input, so neither a forward nor a reverse sweep can produce 0 * inf. In particular a norm is never taken as
 * sqrt of a quantity that can be 0 inside a selected-away branch: the squared quantity is guarded before the square root
 * (the reverse sweep of sqrt at 0 is 0 * inf, which gives a NaN Jacobian even when the branch is not selected).
 * For an AD scalar, include <cppad/cppad.hpp> (or <cppad/cg.hpp>) before instantiating these templates.
 */

namespace ocs2 {

template <typename SCALAR_T>
using quaternion_coeffs_t = Eigen::Matrix<SCALAR_T, 4, 1>;  // (x, y, z, w)

template <typename SCALAR_T>
using rotation_vector_t = Eigen::Matrix<SCALAR_T, 3, 1>;

/** Below this norm a quaternion is not normalized (it is treated as the identity by the rotation-matrix polynomial). */
constexpr double kQuaternionNormalizationEpsilon = 1e-6;

namespace unit_quaternion_internal {

/** left > right ? ifTrue : ifFalse, as a plain conditional for double. */
inline double ifGreaterThan(double left, double right, double ifTrue, double ifFalse) {
  return left > right ? ifTrue : ifFalse;
}

/** left > right ? ifTrue : ifFalse, as a CppAD conditional expression for AD scalars (found by argument-dependent lookup). */
template <typename SCALAR_T>
SCALAR_T ifGreaterThan(const SCALAR_T& left, const SCALAR_T& right, const SCALAR_T& ifTrue, const SCALAR_T& ifFalse) {
  return CondExpGt(left, right, ifTrue, ifFalse);
}

/** [v]x, the cross-product matrix. */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, 3, 3> crossProductMatrix(const rotation_vector_t<SCALAR_T>& v) {
  Eigen::Matrix<SCALAR_T, 3, 3> matrix;
  matrix << SCALAR_T(0.0), -v(2), v(1), v(2), SCALAR_T(0.0), -v(0), -v(1), v(0), SCALAR_T(0.0);
  return matrix;
}

/**
 * sqrt(squaredNorm), or 1 when squaredNorm <= threshold^2. The guard is applied to the squared norm, before the square
 * root, so that the result and its derivatives are finite everywhere, also at 0 and in reverse mode.
 */
template <typename SCALAR_T>
SCALAR_T guardedNorm(const SCALAR_T& squaredNorm, double threshold) {
  using std::sqrt;
  return sqrt(ifGreaterThan(squaredNorm, SCALAR_T(threshold * threshold), squaredNorm, SCALAR_T(1.0)));
}

// Thresholds below which a coefficient is evaluated from its Taylor series. For Exp and Log the closed form has no
// cancellation and the series only avoids 0 / 0. The (alpha - sin alpha) / alpha^3 coefficient of Jr and the
// 1 / alpha^2 - cot(alpha / 2) / (2 alpha) coefficient of Jr^-1 cancel in double precision as alpha -> 0, so they switch
// to the series at alpha = 0.1, where the series, truncated after alpha^8 resp. alpha^6, is exact to round-off and the
// closed form has lost fewer than four digits.
constexpr double kExpLogSeriesThreshold = 1e-6;
constexpr double kJacobianSeriesThreshold = 0.1;

}  // namespace unit_quaternion_internal

/**
 * The rate matrix G(xi), with xi (x) (omega, 0) = G(xi) * omega:
 *   G = [ w -z  y ;  z  w -x ; -y  x  w ; -x -y -z ].
 * It is linear in xi, satisfies G' G = |xi|^2 I and G' xi = 0, and xi_dot = 0.5 G(xi) omega_body is the attitude
 * kinematics. Polynomial, so it is safe on any tape and at any xi, including the raw (unnormalized) state.
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, 4, 3> quaternionRateMatrix(const quaternion_coeffs_t<SCALAR_T>& xi) {
  const SCALAR_T& x = xi(0);
  const SCALAR_T& y = xi(1);
  const SCALAR_T& z = xi(2);
  const SCALAR_T& w = xi(3);
  Eigen::Matrix<SCALAR_T, 4, 3> G;
  G << w, -z, y, z, w, -x, -y, x, w, -x, -y, -z;
  return G;
}

/** The Hamilton product a (x) b of coefficient vectors. */
template <typename SCALAR_T>
quaternion_coeffs_t<SCALAR_T> quaternionProduct(const quaternion_coeffs_t<SCALAR_T>& a, const quaternion_coeffs_t<SCALAR_T>& b) {
  quaternion_coeffs_t<SCALAR_T> product;
  product(0) = a(3) * b(0) + a(0) * b(3) + a(1) * b(2) - a(2) * b(1);
  product(1) = a(3) * b(1) - a(0) * b(2) + a(1) * b(3) + a(2) * b(0);
  product(2) = a(3) * b(2) + a(0) * b(1) - a(1) * b(0) + a(2) * b(3);
  product(3) = a(3) * b(3) - a(0) * b(0) - a(1) * b(1) - a(2) * b(2);
  return product;
}

/** The conjugate (-x, -y, -z, w); the inverse of a unit quaternion. */
template <typename SCALAR_T>
quaternion_coeffs_t<SCALAR_T> quaternionConjugate(const quaternion_coeffs_t<SCALAR_T>& xi) {
  quaternion_coeffs_t<SCALAR_T> conjugate;
  conjugate << -xi(0), -xi(1), -xi(2), xi(3);
  return conjugate;
}

/**
 * |xi|, or 1 when |xi| <= kQuaternionNormalizationEpsilon. The guard acts on |xi|^2 (see guardedNorm()), so the norm
 * and its derivatives are finite at every xi, the zero quaternion included, in forward and reverse mode.
 */
template <typename SCALAR_T>
SCALAR_T quaternionSafeNorm(const quaternion_coeffs_t<SCALAR_T>& xi) {
  return unit_quaternion_internal::guardedNorm(xi.squaredNorm(), kQuaternionNormalizationEpsilon);
}

/**
 * xi / |xi|, safe on a tape and at every xi: a quaternion shorter than kQuaternionNormalizationEpsilon is returned
 * unchanged, so the zero quaternion stays zero and Eigen's polynomial toRotationMatrix() maps it to the identity. Never
 * use Eigen's normalized() / normalize() on an AD scalar: they record a comparison between variables.
 */
template <typename SCALAR_T>
quaternion_coeffs_t<SCALAR_T> quaternionSafeNormalize(const quaternion_coeffs_t<SCALAR_T>& xi) {
  return xi / quaternionSafeNorm(xi);
}

/**
 * E(xi) = 0.5 G(xi) (4 x 3): maps a body perturbation delta to the first-order change of xi (x) Exp(delta).
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, 4, 3> quaternionTangentMap(const quaternion_coeffs_t<SCALAR_T>& xi) {
  return SCALAR_T(0.5) * quaternionRateMatrix(xi);
}

/**
 * E+(xi) = 2 G(xi)' / |xi|^2 (3 x 4), the left inverse of E(xi) (E+ E = I) whose rows are orthogonal to xi. On the
 * unit sphere it is 2 G(xi)'. A row vector J_t on the tangent lifts to the ambient row vector J_t E+.
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, 3, 4> quaternionTangentMapPseudoInverse(const quaternion_coeffs_t<SCALAR_T>& xi) {
  const SCALAR_T norm = quaternionSafeNorm(xi);
  return (SCALAR_T(2.0) / (norm * norm)) * quaternionRateMatrix(xi).transpose();
}

/**
 * R(xi), the rotation matrix (body to world) of a unit quaternion, as Eigen's polynomial toRotationMatrix() computes it:
 * no division, so it is safe on any tape, and the zero quaternion maps to the identity. For a quaternion that is not
 * unit, pass quaternionSafeNormalize(xi).
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, 3, 3> quaternionRotationMatrix(const quaternion_coeffs_t<SCALAR_T>& xi) {
  const SCALAR_T& x = xi(0);
  const SCALAR_T& y = xi(1);
  const SCALAR_T& z = xi(2);
  const SCALAR_T& w = xi(3);
  const SCALAR_T one(1.0);
  const SCALAR_T two(2.0);
  Eigen::Matrix<SCALAR_T, 3, 3> R;
  R << one - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w),  //
      two * (x * y + z * w), one - two * (x * x + z * z), two * (y * z - x * w),   //
      two * (x * z - y * w), two * (y * z + x * w), one - two * (x * x + y * y);
  return R;
}

/** Exp(phi) = (sin(|phi| / 2) phi / |phi|, cos(|phi| / 2)), a unit quaternion rotating by |phi| about phi. */
template <typename SCALAR_T>
quaternion_coeffs_t<SCALAR_T> quaternionExp(const rotation_vector_t<SCALAR_T>& phi) {
  using std::cos;
  using std::sin;
  using unit_quaternion_internal::ifGreaterThan;
  const SCALAR_T angleSquared = phi.squaredNorm();
  const SCALAR_T angle = unit_quaternion_internal::guardedNorm(angleSquared, unit_quaternion_internal::kExpLogSeriesThreshold);
  const SCALAR_T threshold(unit_quaternion_internal::kExpLogSeriesThreshold * unit_quaternion_internal::kExpLogSeriesThreshold);
  // sin(a / 2) / a = 1/2 - a^2/48 + a^4/3840, cos(a / 2) = 1 - a^2/8 + a^4/384.
  const SCALAR_T vectorScale =
      ifGreaterThan(angleSquared, threshold, sin(SCALAR_T(0.5) * angle) / angle,
                    SCALAR_T(0.5) - angleSquared / SCALAR_T(48.0) + angleSquared * angleSquared / SCALAR_T(3840.0));
  const SCALAR_T scalarPart = ifGreaterThan(angleSquared, threshold, cos(SCALAR_T(0.5) * angle),
                                            SCALAR_T(1.0) - angleSquared / SCALAR_T(8.0) + angleSquared * angleSquared / SCALAR_T(384.0));
  quaternion_coeffs_t<SCALAR_T> xi;
  xi << vectorScale * phi(0), vectorScale * phi(1), vectorScale * phi(2), scalarPart;
  return xi;
}

/**
 * Log(xi), the rotation vector of xi along the shortest path: xi is replaced by -xi when w < 0, so |Log(xi)| <= pi and
 * Log(-xi) = Log(xi). It is homogeneous of degree 0 (Log(c xi) = Log(xi) for c > 0), so xi need not be unit. The zero
 * quaternion maps to the zero vector.
 */
template <typename SCALAR_T>
rotation_vector_t<SCALAR_T> quaternionLog(const quaternion_coeffs_t<SCALAR_T>& xi) {
  using std::atan2;
  using unit_quaternion_internal::ifGreaterThan;
  const SCALAR_T sign = ifGreaterThan(SCALAR_T(0.0), xi(3), SCALAR_T(-1.0), SCALAR_T(1.0));
  const rotation_vector_t<SCALAR_T> v = sign * xi.template head<3>();
  const SCALAR_T w = sign * xi(3);  // >= 0
  const SCALAR_T vectorNormSquared = v.squaredNorm();
  const SCALAR_T vectorNorm = unit_quaternion_internal::guardedNorm(vectorNormSquared, unit_quaternion_internal::kExpLogSeriesThreshold);
  const SCALAR_T threshold(unit_quaternion_internal::kExpLogSeriesThreshold * unit_quaternion_internal::kExpLogSeriesThreshold);
  // 2 atan2(n, w) / n = (2 / w) (1 - n^2 / (3 w^2)) for n << w. The guarded w keeps that branch finite at the zero
  // quaternion (where v = 0 and the result is 0 either way) and for a half-turn (w = 0), where it is not selected.
  const SCALAR_T guardedW = ifGreaterThan(w, SCALAR_T(unit_quaternion_internal::kExpLogSeriesThreshold), w, SCALAR_T(1.0));
  const SCALAR_T scale =
      ifGreaterThan(vectorNormSquared, threshold, SCALAR_T(2.0) * atan2(vectorNorm, w) / vectorNorm,
                    (SCALAR_T(2.0) / guardedW) * (SCALAR_T(1.0) - vectorNormSquared / (SCALAR_T(3.0) * guardedW * guardedW)));
  return scale * v;
}

/**
 * The right Jacobian of SO(3): Exp(phi + d) = Exp(phi) (x) Exp(Jr(phi) d) to first order.
 *   Jr(phi) = I - (1 - cos a) / a^2 [phi]x + (a - sin a) / a^3 [phi]x^2,  a = |phi|.
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, 3, 3> so3RightJacobian(const rotation_vector_t<SCALAR_T>& phi) {
  using std::sin;
  using unit_quaternion_internal::ifGreaterThan;
  const SCALAR_T a2 = phi.squaredNorm();
  const SCALAR_T a = unit_quaternion_internal::guardedNorm(a2, unit_quaternion_internal::kExpLogSeriesThreshold);
  const SCALAR_T aJacobian = unit_quaternion_internal::guardedNorm(a2, unit_quaternion_internal::kJacobianSeriesThreshold);
  // (1 - cos a) / a^2 = 2 sin^2(a / 2) / a^2, which has no cancellation; series 1/2 - a^2/24 + a^4/720.
  const SCALAR_T sinHalf = sin(SCALAR_T(0.5) * a);
  const SCALAR_T first =
      ifGreaterThan(a2, SCALAR_T(unit_quaternion_internal::kExpLogSeriesThreshold * unit_quaternion_internal::kExpLogSeriesThreshold),
                    SCALAR_T(2.0) * sinHalf * sinHalf / (a * a), SCALAR_T(0.5) - a2 / SCALAR_T(24.0) + a2 * a2 / SCALAR_T(720.0));
  // (a - sin a) / a^3 = 1/6 - a^2/120 + a^4/5040 - a^6/362880 + a^8/39916800.
  const SCALAR_T a4 = a2 * a2;
  const SCALAR_T second = ifGreaterThan(
      a2, SCALAR_T(unit_quaternion_internal::kJacobianSeriesThreshold * unit_quaternion_internal::kJacobianSeriesThreshold),
      (aJacobian - sin(aJacobian)) / (aJacobian * aJacobian * aJacobian),
      SCALAR_T(1.0 / 6.0) - a2 / SCALAR_T(120.0) + a4 / SCALAR_T(5040.0) - a4 * a2 / SCALAR_T(362880.0) + a4 * a4 / SCALAR_T(39916800.0));
  const Eigen::Matrix<SCALAR_T, 3, 3> cross = unit_quaternion_internal::crossProductMatrix(phi);
  return Eigen::Matrix<SCALAR_T, 3, 3>::Identity() - first * cross + second * cross * cross;
}

/**
 * The inverse of the right Jacobian: Log(Exp(phi) (x) Exp(d)) = phi + Jr^-1(phi) d to first order.
 *   Jr^-1(phi) = I + 1/2 [phi]x + (1 / a^2 - cot(a / 2) / (2 a)) [phi]x^2,  a = |phi| < 2 pi.
 * The coefficient is written with cot(a / 2), so it is regular at a = pi (where (1 + cos a) / (2 a sin a) is 0 / 0).
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, 3, 3> so3RightJacobianInverse(const rotation_vector_t<SCALAR_T>& phi) {
  using std::cos;
  using std::sin;
  using unit_quaternion_internal::ifGreaterThan;
  const SCALAR_T a2 = phi.squaredNorm();
  const SCALAR_T a = unit_quaternion_internal::guardedNorm(a2, unit_quaternion_internal::kJacobianSeriesThreshold);
  const SCALAR_T halfAngle = SCALAR_T(0.5) * a;
  // 1/a^2 - cot(a/2) / (2a) = 1/12 + a^2/720 + a^4/30240 + a^6/1209600.
  const SCALAR_T a4 = a2 * a2;
  const SCALAR_T coefficient =
      ifGreaterThan(a2, SCALAR_T(unit_quaternion_internal::kJacobianSeriesThreshold * unit_quaternion_internal::kJacobianSeriesThreshold),
                    SCALAR_T(1.0) / (a * a) - cos(halfAngle) / (SCALAR_T(2.0) * a * sin(halfAngle)),
                    SCALAR_T(1.0 / 12.0) + a2 / SCALAR_T(720.0) + a4 / SCALAR_T(30240.0) + a4 * a2 / SCALAR_T(1209600.0));
  const Eigen::Matrix<SCALAR_T, 3, 3> cross = unit_quaternion_internal::crossProductMatrix(phi);
  return Eigen::Matrix<SCALAR_T, 3, 3>::Identity() + SCALAR_T(0.5) * cross + coefficient * cross * cross;
}

/**
 * Shortest-path spherical interpolation: xi0 at alpha = 0 and the rotation of xi1 at alpha = 1, through
 * xi0 (x) Exp(alpha Log(xi0^-1 (x) xi1)). The result is normalized. Rotations about one fixed axis interpolate linearly in
 * the angle. Invariant under xi0 -> -xi0 and xi1 -> -xi1 up to the sign of the result.
 */
template <typename SCALAR_T>
quaternion_coeffs_t<SCALAR_T> quaternionSlerp(const quaternion_coeffs_t<SCALAR_T>& xi0,
                                              const quaternion_coeffs_t<SCALAR_T>& xi1,
                                              const SCALAR_T& alpha) {
  const rotation_vector_t<SCALAR_T> relative = quaternionLog(quaternionProduct(quaternionConjugate(xi0), xi1));
  const rotation_vector_t<SCALAR_T> step = alpha * relative;
  return quaternionSafeNormalize(quaternionProduct(xi0, quaternionExp(step)));
}

}  // namespace ocs2
