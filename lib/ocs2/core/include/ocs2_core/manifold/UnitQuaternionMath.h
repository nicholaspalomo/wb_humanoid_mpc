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

#include "ocs2_core/Dimensions.h"

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
 *    vector of norm at most pi), so the functions of a rotation here - R(xi), Log(xi) and slerp up to the sign of its
 *    result - are invariant under xi -> -xi (the double cover), except Log on its cut locus, an exact half turn, where
 *    it returns the other of the two logarithms (quaternionLog()), and with it slerp where the relative rotation of its
 *    ends is one (quaternionSlerp()). The linear maps G, E and E+ change sign with xi.
 *
 * Every function is a template on the scalar, so the same code runs in double and on a CppAD tape. On a tape, no branch
 * may be a plain `if` (the tape would record only the branch taken at the taping point): branches go through
 * ifGreaterThan(), which is a CppAD conditional expression for AD scalars, and both of its branches are kept finite at
 * every input whose squared norms are normal doubles - for a quaternion 1e-150 <= |xi| <= 1e150, for a rotation vector
 * |phi| <= 1e150 - so that neither a forward nor a reverse sweep can produce 0 * inf there. Outside that range a squared
 * norm under- or overflows, and quaternionLog()'s guards, which scale with |xi|^2, no longer select the branch they
 * would at the same rotation: its value loses its precision and then its meaning (a rotation by 1 rad reads as 1.00004
 * rad at |xi| = 1e-160, as 1e-165 rad at |xi| = 1e-165 and as -inf at 1e155) and its tape Jacobian can be NaN (at
 * xi = (5e-164, 0, 0, 1e-154), for one). The solver's quaternions are near the unit sphere. In particular a norm is
 * never taken as sqrt of a quantity that can be 0 inside a selected-away branch: the squared quantity is guarded before
 * the square root (the reverse sweep of sqrt at 0 is 0 * inf, which gives a NaN Jacobian even when the branch is not
 * selected).
 * For an AD scalar, include <cppad/cppad.hpp> (or <cppad/cg.hpp>) before instantiating these templates.
 */

namespace ocs2 {

template <typename SCALAR_T>
using quaternion_coeffs_t = Eigen::Matrix<SCALAR_T, kQuaternionCoefficients, 1>;  // (x, y, z, w)

template <typename SCALAR_T>
using rotation_vector_t = Eigen::Matrix<SCALAR_T, kRotationTangentDim, 1>;

/**
 * At or below this norm a quaternion is not normalized: quaternionSafeNormalize() returns it unchanged, and the
 * rotation-matrix polynomial reads it as the identity, to within 2 |xi|^2 <= 2e-12 per entry (exactly at the zero
 * quaternion). Log is the exception: it reads every nonzero quaternion as the rotation of xi / |xi| (quaternionLog()).
 */
constexpr double kUnitQuaternionNormalizationEpsilon = 1e-6;

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
Eigen::Matrix<SCALAR_T, kRotationTangentDim, kRotationTangentDim> crossProductMatrix(const rotation_vector_t<SCALAR_T>& v) {
  const SCALAR_T& x = v(kXAxisIndex);
  const SCALAR_T& y = v(kYAxisIndex);
  const SCALAR_T& z = v(kZAxisIndex);
  Eigen::Matrix<SCALAR_T, kRotationTangentDim, kRotationTangentDim> matrix;
  matrix << SCALAR_T(0.0), -z, y, z, SCALAR_T(0.0), -x, -y, x, SCALAR_T(0.0);
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
// cancellation and the series only avoids 0 / 0; Log's threshold is relative, |v| <= 1e-6 |xi| (quaternionLog()). The
// (alpha - sin alpha) / alpha^3 coefficient of Jr and the 1 / alpha^2 - cot(alpha / 2) / (2 alpha) coefficient of Jr^-1
// cancel in double precision as alpha -> 0, so they switch to the series at alpha = 0.1, where the series, truncated
// after alpha^8 resp. alpha^6, is exact to round-off and the closed form has lost fewer than four digits.
constexpr double kExpLogSeriesThreshold = 1e-6;
constexpr double kJacobianSeriesThreshold = 0.1;

}  // namespace unit_quaternion_internal

/**
 * A unit quaternion represents the rotation by an angle a through the half angle a / 2 (the double cover): Exp(phi) has
 * the coefficients sin(a / 2) and cos(a / 2), E(xi) = G(xi) / 2 and with it the attitude kinematics
 * xi' = 1/2 G(xi) omega_body, a quaternion-distance orientation error (the vector part of an error quaternion, half the
 * rotation angle to first order) moves at half the angular velocity, and Jr^-1 has the terms (a / 2) [phi / a]x and
 * (1 - (a / 2) cot(a / 2)) [phi / a]x^2. kUnitQuaternionDoubleAngleFactor takes a half angle back to its angle: Log(xi) =
 * 2 atan2(|v|, w) v / |v|, E+(xi) = 2 G(xi)' / |xi|^2, the cot(a / 2) / (2 a) of Jr^-1, and the double-angle identities
 * sin a = 2 sin(a / 2) cos(a / 2) and 1 - cos a = 2 sin^2(a / 2), which give R(xi) and Jr their factors 2.
 */
inline constexpr double kUnitQuaternionHalfAngleFactor = 0.5;
inline constexpr double kUnitQuaternionDoubleAngleFactor = 2.0;

/**
 * The rate matrix G(xi), with xi (x) (omega, 0) = G(xi) * omega:
 *   G = [ w -z  y ;  z  w -x ; -y  x  w ; -x -y -z ].
 * It is linear in xi, satisfies G' G = |xi|^2 I and G' xi = 0, and xi_dot = 0.5 G(xi) omega_body is the attitude
 * kinematics. Polynomial, so it is safe on any tape and at any xi, including the raw (unnormalized) state.
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, kQuaternionCoefficients, kRotationTangentDim> quaternionRateMatrix(const quaternion_coeffs_t<SCALAR_T>& xi) {
  const SCALAR_T& x = xi(kQuaternionXIndex);
  const SCALAR_T& y = xi(kQuaternionYIndex);
  const SCALAR_T& z = xi(kQuaternionZIndex);
  const SCALAR_T& w = xi(kQuaternionWIndex);
  Eigen::Matrix<SCALAR_T, kQuaternionCoefficients, kRotationTangentDim> G;
  G << w, -z, y, z, w, -x, -y, x, w, -x, -y, -z;
  return G;
}

/** The Hamilton product a (x) b of coefficient vectors. */
template <typename SCALAR_T>
quaternion_coeffs_t<SCALAR_T> quaternionProduct(const quaternion_coeffs_t<SCALAR_T>& a, const quaternion_coeffs_t<SCALAR_T>& b) {
  const SCALAR_T& ax = a(kQuaternionXIndex);
  const SCALAR_T& ay = a(kQuaternionYIndex);
  const SCALAR_T& az = a(kQuaternionZIndex);
  const SCALAR_T& aw = a(kQuaternionWIndex);
  const SCALAR_T& bx = b(kQuaternionXIndex);
  const SCALAR_T& by = b(kQuaternionYIndex);
  const SCALAR_T& bz = b(kQuaternionZIndex);
  const SCALAR_T& bw = b(kQuaternionWIndex);
  quaternion_coeffs_t<SCALAR_T> product;
  product(kQuaternionXIndex) = aw * bx + ax * bw + ay * bz - az * by;
  product(kQuaternionYIndex) = aw * by - ax * bz + ay * bw + az * bx;
  product(kQuaternionZIndex) = aw * bz + ax * by - ay * bx + az * bw;
  product(kQuaternionWIndex) = aw * bw - ax * bx - ay * by - az * bz;
  return product;
}

/** The conjugate (-x, -y, -z, w); the inverse of a unit quaternion. */
template <typename SCALAR_T>
quaternion_coeffs_t<SCALAR_T> quaternionConjugate(const quaternion_coeffs_t<SCALAR_T>& xi) {
  quaternion_coeffs_t<SCALAR_T> conjugate;
  conjugate << -xi(kQuaternionXIndex), -xi(kQuaternionYIndex), -xi(kQuaternionZIndex), xi(kQuaternionWIndex);
  return conjugate;
}

/**
 * |xi|, or 1 when |xi| <= kUnitQuaternionNormalizationEpsilon. The guard acts on |xi|^2 (see guardedNorm()), so the norm
 * and its derivatives are finite at every xi, the zero quaternion included, in forward and reverse mode.
 */
template <typename SCALAR_T>
SCALAR_T quaternionSafeNorm(const quaternion_coeffs_t<SCALAR_T>& xi) {
  return unit_quaternion_internal::guardedNorm(xi.squaredNorm(), kUnitQuaternionNormalizationEpsilon);
}

/**
 * xi / |xi|, safe on a tape and at every xi: a quaternion shorter than kUnitQuaternionNormalizationEpsilon is returned
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
Eigen::Matrix<SCALAR_T, kQuaternionCoefficients, kRotationTangentDim> quaternionTangentMap(const quaternion_coeffs_t<SCALAR_T>& xi) {
  return SCALAR_T(kUnitQuaternionHalfAngleFactor) * quaternionRateMatrix(xi);
}

/**
 * E+(xi) = 2 G(xi)' / |xi|^2 (3 x 4), the left inverse of E(xi) (E+ E = I) for |xi| > kUnitQuaternionNormalizationEpsilon,
 * whose rows are orthogonal to xi (E+ xi = 0 at every xi). On the unit sphere it is 2 G(xi)'. At or below the epsilon
 * the divisor is the guarded 1, so E+ = 2 G(xi)' and E+ E = |xi|^2 I: the zero quaternion lifts to zero, not NaN. A row
 * vector J_t on the tangent lifts to the ambient row vector J_t E+.
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, kRotationTangentDim, kQuaternionCoefficients> quaternionTangentMapPseudoInverse(
    const quaternion_coeffs_t<SCALAR_T>& xi) {
  const SCALAR_T norm = quaternionSafeNorm(xi);
  return (SCALAR_T(kUnitQuaternionDoubleAngleFactor) / (norm * norm)) * quaternionRateMatrix(xi).transpose();
}

/**
 * R(xi), the rotation matrix (body to world) of a unit quaternion, as Eigen's polynomial toRotationMatrix() computes it:
 * no division, so it is safe on any tape, and the zero quaternion maps to the identity. For a quaternion that is not
 * unit, pass quaternionSafeNormalize(xi).
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, kPositionDim, kPositionDim> quaternionRotationMatrix(const quaternion_coeffs_t<SCALAR_T>& xi) {
  const SCALAR_T& x = xi(kQuaternionXIndex);
  const SCALAR_T& y = xi(kQuaternionYIndex);
  const SCALAR_T& z = xi(kQuaternionZIndex);
  const SCALAR_T& w = xi(kQuaternionWIndex);
  const SCALAR_T one(1.0);
  // The factors 2 of the double-angle identities sin a = 2 sin(a / 2) cos(a / 2) and 1 - cos a = 2 sin^2(a / 2).
  const SCALAR_T doubleAngleFactor(kUnitQuaternionDoubleAngleFactor);
  Eigen::Matrix<SCALAR_T, kPositionDim, kPositionDim> R;
  R << one - doubleAngleFactor * (y * y + z * z), doubleAngleFactor * (x * y - z * w), doubleAngleFactor * (x * z + y * w),  //
      doubleAngleFactor * (x * y + z * w), one - doubleAngleFactor * (x * x + z * z), doubleAngleFactor * (y * z - x * w),   //
      doubleAngleFactor * (x * z - y * w), doubleAngleFactor * (y * z + x * w), one - doubleAngleFactor * (x * x + y * y);
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
  // sin(a / 2) / a = 1/2 - a^2/48 + a^4/3840, cos(a / 2) = 1 - a^2/8 + a^4/384, the Taylor series of sin and cos at the
  // half angle: sin(a / 2) / a = (1 - (a / 2)^2 / 3! + (a / 2)^4 / 5!) / 2 and cos(a / 2) = 1 - (a / 2)^2 / 2! + (a / 2)^4 / 4!.
  constexpr double kVectorScaleSecondOrderDenominator = 48.0;
  constexpr double kVectorScaleFourthOrderDenominator = 3840.0;
  constexpr double kScalarPartSecondOrderDenominator = 8.0;
  constexpr double kScalarPartFourthOrderDenominator = 384.0;
  const SCALAR_T vectorScale =
      ifGreaterThan(angleSquared, threshold, sin(SCALAR_T(kUnitQuaternionHalfAngleFactor) * angle) / angle,
                    SCALAR_T(kUnitQuaternionHalfAngleFactor) - angleSquared / SCALAR_T(kVectorScaleSecondOrderDenominator) +
                        angleSquared * angleSquared / SCALAR_T(kVectorScaleFourthOrderDenominator));
  const SCALAR_T scalarPart = ifGreaterThan(angleSquared, threshold, cos(SCALAR_T(kUnitQuaternionHalfAngleFactor) * angle),
                                            SCALAR_T(1.0) - angleSquared / SCALAR_T(kScalarPartSecondOrderDenominator) +
                                                angleSquared * angleSquared / SCALAR_T(kScalarPartFourthOrderDenominator));
  quaternion_coeffs_t<SCALAR_T> xi;
  xi << vectorScale * phi(kXAxisIndex), vectorScale * phi(kYAxisIndex), vectorScale * phi(kZAxisIndex), scalarPart;
  return xi;
}

/**
 * Log(xi), the rotation vector of xi along the shortest path: xi is replaced by -xi when w < 0, so |Log(xi)| <= pi.
 *
 * Log(xi) = 2 atan2(|v|, w) v / |v| (v the vector part) depends on xi only through xi / |xi|: it is homogeneous of
 * degree 0, Log(c xi) = Log(xi) for every c > 0, so xi need not be unit. Every guard below is therefore relative to
 * |xi|^2, never an absolute threshold (which would select the series for every short quaternion, whatever its angle), and
 * the identity holds to round-off for 1e-150 <= |xi| <= 1e150, where |xi|^2 is a normal double. A nonzero xi is read as
 * the rotation of xi / |xi| also at or below kUnitQuaternionNormalizationEpsilon, where the safe normalization leaves it
 * unchanged. The zero quaternion maps to the zero vector.
 *
 * The double cover: Log(-xi) = Log(xi) bit for bit for every xi with w != 0, subnormal w included, and
 * Log(conj(xi)) = -Log(xi) bit for bit for every xi (conj(xi) is xi^-1 up to the positive scale |xi|^2). At an exact
 * half turn, w = +0 or -0, both cannot hold: -xi and conj(xi) then have the same coefficients up to the sign of the zero
 * w, so a rule on the coefficients' values that kept Log(-xi) = Log(xi) as well as the inversion would give
 * Log = -Log = 0, which is no logarithm of a half turn. The rule here keeps the inversion: Log(xi) = pi v / |v| for
 * w = +-0, so Log(-xi) = -Log(xi) there. Both are logarithms of the same rotation, and the linearization Jr^-1(Log(xi))
 * is the derivative of the branch returned, that of the hemisphere w >= 0 (testStateManifold). A tie-break in the
 * double overload alone would make the double path and the tape disagree.
 */
template <typename SCALAR_T>
rotation_vector_t<SCALAR_T> quaternionLog(const quaternion_coeffs_t<SCALAR_T>& xi) {
  using std::atan2;
  using std::sqrt;
  using unit_quaternion_internal::ifGreaterThan;
  const SCALAR_T sign = ifGreaterThan(SCALAR_T(0.0), xi(kQuaternionWIndex), SCALAR_T(-1.0), SCALAR_T(1.0));
  const rotation_vector_t<SCALAR_T> v = sign * xi.template head<kRotationTangentDim>();
  const SCALAR_T w = sign * xi(kQuaternionWIndex);  // >= 0
  const SCALAR_T vectorNormSquared = v.squaredNorm();
  // th^2 |xi|^2: the series threshold th = kExpLogSeriesThreshold scaled with xi, so that no selection below changes when
  // xi is multiplied by c > 0.
  const SCALAR_T threshold =
      SCALAR_T(unit_quaternion_internal::kExpLogSeriesThreshold * unit_quaternion_internal::kExpLogSeriesThreshold) * xi.squaredNorm();
  // |v|, guarded to 1 where the series is selected (also at the zero quaternion), so the closed form stays finite there.
  const SCALAR_T vectorNorm = sqrt(ifGreaterThan(vectorNormSquared, threshold, vectorNormSquared, SCALAR_T(1.0)));
  // 2 atan2(n, w) / n = (2 / w) (1 - n^2 / (3 w^2)) + O(n^4 / w^5) for n = |v| << w. The series is selected where
  // n^2 <= th^2 |xi|^2, i.e. n / w <= th / sqrt(1 - th^2), so its truncation error is below 1e-24 relative. There
  // w^2 = |xi|^2 - n^2 > th^2 |xi|^2, so w is kept unless xi = 0; the guarded w is 1 only at the zero quaternion (where
  // v = 0 and the result is 0 either way) and near a half turn, where the series is not selected, so both branches are
  // finite at every xi with 1e-150 <= |xi| <= 1e150, as the tape requires, and at the zero quaternion. Below that range
  // |xi|^2, n^2 and th^2 |xi|^2 are subnormal or zero, so the selection no longer follows the angle and a reverse sweep
  // can divide by an underflowed w^3 (0 * inf, a NaN Jacobian); above it |xi|^2 overflows (the class comment).
  const SCALAR_T guardedW = ifGreaterThan(w * w, threshold, w, SCALAR_T(1.0));
  // The series is that of atan(t) = t - t^3 / 3 + O(t^5) at t = n / w.
  constexpr double kArctanThirdOrderDenominator = 3.0;
  const SCALAR_T scale =
      ifGreaterThan(vectorNormSquared, threshold, SCALAR_T(kUnitQuaternionDoubleAngleFactor) * atan2(vectorNorm, w) / vectorNorm,
                    (SCALAR_T(kUnitQuaternionDoubleAngleFactor) / guardedW) *
                        (SCALAR_T(1.0) - vectorNormSquared / (SCALAR_T(kArctanThirdOrderDenominator) * guardedW * guardedW)));
  return scale * v;
}

/**
 * The right Jacobian of SO(3): Exp(phi + d) = Exp(phi) (x) Exp(Jr(phi) d) to first order.
 *   Jr(phi) = I - (1 - cos a) / a^2 [phi]x + (a - sin a) / a^3 [phi]x^2,  a = |phi|.
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, kRotationTangentDim, kRotationTangentDim> so3RightJacobian(const rotation_vector_t<SCALAR_T>& phi) {
  using std::sin;
  using unit_quaternion_internal::ifGreaterThan;
  // The Taylor coefficients of first and second, below their thresholds, are inverse factorials.
  constexpr double kInverseFactorial2 = 0.5;
  constexpr double kInverseFactorial3 = 1.0 / 6.0;
  constexpr double kFactorial4 = 24.0;
  constexpr double kFactorial5 = 120.0;
  constexpr double kFactorial6 = 720.0;
  constexpr double kFactorial7 = 5040.0;
  constexpr double kFactorial9 = 362880.0;
  constexpr double kFactorial11 = 39916800.0;
  const SCALAR_T a2 = phi.squaredNorm();
  const SCALAR_T a = unit_quaternion_internal::guardedNorm(a2, unit_quaternion_internal::kExpLogSeriesThreshold);
  const SCALAR_T aJacobian = unit_quaternion_internal::guardedNorm(a2, unit_quaternion_internal::kJacobianSeriesThreshold);
  // (1 - cos a) / a^2 = 2 sin^2(a / 2) / a^2, which has no cancellation; series 1/2! - a^2/4! + a^4/6!.
  const SCALAR_T sinHalf = sin(SCALAR_T(kUnitQuaternionHalfAngleFactor) * a);
  const SCALAR_T first =
      ifGreaterThan(a2, SCALAR_T(unit_quaternion_internal::kExpLogSeriesThreshold * unit_quaternion_internal::kExpLogSeriesThreshold),
                    SCALAR_T(kUnitQuaternionDoubleAngleFactor) * sinHalf * sinHalf / (a * a),
                    SCALAR_T(kInverseFactorial2) - a2 / SCALAR_T(kFactorial4) + a2 * a2 / SCALAR_T(kFactorial6));
  // (a - sin a) / a^3 = 1/3! - a^2/5! + a^4/7! - a^6/9! + a^8/11!.
  const SCALAR_T a4 = a2 * a2;
  const SCALAR_T second =
      ifGreaterThan(a2, SCALAR_T(unit_quaternion_internal::kJacobianSeriesThreshold * unit_quaternion_internal::kJacobianSeriesThreshold),
                    (aJacobian - sin(aJacobian)) / (aJacobian * aJacobian * aJacobian),
                    SCALAR_T(kInverseFactorial3) - a2 / SCALAR_T(kFactorial5) + a4 / SCALAR_T(kFactorial7) -
                        a4 * a2 / SCALAR_T(kFactorial9) + a4 * a4 / SCALAR_T(kFactorial11));
  const Eigen::Matrix<SCALAR_T, kRotationTangentDim, kRotationTangentDim> cross = unit_quaternion_internal::crossProductMatrix(phi);
  return Eigen::Matrix<SCALAR_T, kRotationTangentDim, kRotationTangentDim>::Identity() - first * cross + second * cross * cross;
}

/**
 * The inverse of the right Jacobian: Log(Exp(phi) (x) Exp(d)) = phi + Jr^-1(phi) d to first order.
 *   Jr^-1(phi) = I + 1/2 [phi]x + (1 / a^2 - cot(a / 2) / (2 a)) [phi]x^2,  a = |phi| < 2 pi.
 * The coefficient is written with cot(a / 2), so it is regular at a = pi (where (1 + cos a) / (2 a sin a) is 0 / 0).
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, kRotationTangentDim, kRotationTangentDim> so3RightJacobianInverse(const rotation_vector_t<SCALAR_T>& phi) {
  using std::cos;
  using std::sin;
  using unit_quaternion_internal::ifGreaterThan;
  // The series of the coefficient, whose terms are |B_2k| a^(2k - 2) / (2k)! with B the Bernoulli numbers.
  constexpr double kSeriesConstantTerm = 1.0 / 12.0;
  constexpr double kSeriesSecondOrderDenominator = 720.0;
  constexpr double kSeriesFourthOrderDenominator = 30240.0;
  constexpr double kSeriesSixthOrderDenominator = 1209600.0;
  const SCALAR_T a2 = phi.squaredNorm();
  const SCALAR_T a = unit_quaternion_internal::guardedNorm(a2, unit_quaternion_internal::kJacobianSeriesThreshold);
  const SCALAR_T halfAngle = SCALAR_T(kUnitQuaternionHalfAngleFactor) * a;
  // 1/a^2 - cot(a/2) / (2a) = 1/12 + a^2/720 + a^4/30240 + a^6/1209600.
  const SCALAR_T a4 = a2 * a2;
  const SCALAR_T coefficient =
      ifGreaterThan(a2, SCALAR_T(unit_quaternion_internal::kJacobianSeriesThreshold * unit_quaternion_internal::kJacobianSeriesThreshold),
                    SCALAR_T(1.0) / (a * a) - cos(halfAngle) / (SCALAR_T(kUnitQuaternionDoubleAngleFactor) * a * sin(halfAngle)),
                    SCALAR_T(kSeriesConstantTerm) + a2 / SCALAR_T(kSeriesSecondOrderDenominator) +
                        a4 / SCALAR_T(kSeriesFourthOrderDenominator) + a4 * a2 / SCALAR_T(kSeriesSixthOrderDenominator));
  const Eigen::Matrix<SCALAR_T, kRotationTangentDim, kRotationTangentDim> cross = unit_quaternion_internal::crossProductMatrix(phi);
  return Eigen::Matrix<SCALAR_T, kRotationTangentDim, kRotationTangentDim>::Identity() + SCALAR_T(kUnitQuaternionHalfAngleFactor) * cross +
         coefficient * cross * cross;
}

/**
 * Shortest-path spherical interpolation: xi0 at alpha = 0 and the rotation of xi1 at alpha = 1, through
 * xi0 (x) Exp(alpha Log(xi0^-1 (x) xi1)). The result is normalized (for |xi0| > kUnitQuaternionNormalizationEpsilon).
 * Rotations about one fixed axis interpolate linearly in the angle. Invariant under xi0 -> -xi0 and xi1 -> -xi1 up to the
 * sign of the result, except where xi0^-1 (x) xi1 is an exact half turn (w = +-0): both geodesics are then shortest, of
 * length pi, and xi1 and -xi1 take opposite ones (quaternionLog()), so slerp(xi0, xi1, 1/2) and slerp(xi0, -xi1, 1/2)
 * are rotations pi apart.
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
