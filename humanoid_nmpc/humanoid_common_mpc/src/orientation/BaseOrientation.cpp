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

#include "humanoid_common_mpc/orientation/BaseOrientation.h"

#include <algorithm>
#include <cmath>

namespace ocs2::humanoid {

namespace {

constexpr scalar_t kMinimumTwistNormSquared = kMinimumTwistNorm * kMinimumTwistNorm;

/** The tilt tau = (tau_x, tau_y) and the swing's vector part s_xy: rotations about the two horizontal axes. */
constexpr int kTiltDim = kHorizontalPositionDim;

/** The rows of the heading-tilt residual and of its gradients, in the order of the Euler tuning layout. */
constexpr int kHeadingRow = 0;
constexpr int kPitchRow = 1;
constexpr int kRollRow = 2;

/**
 * The small-tilt series of kappa(n, rho) = (2 / rho) atan(u) / u with u = n / rho, from atan(u) / u = 1 - u^2 / 3 + O(u^4):
 * kappa = (2 / rho)(1 - n^2 / (3 rho^2)), and (d kappa / dn) / n = -4 / (3 rho^3).
 */
constexpr scalar_t kTiltSeriesCubicDivisor = 3.0;
constexpr scalar_t kTiltSeriesSlopeNumerator = 4.0;

/** A difference of two angles in (-pi, pi], wrapped back to (-pi, pi]. */
scalar_t wrapAngleDifference(scalar_t angle) {
  if (angle > M_PI) return angle - 2.0 * M_PI;
  if (angle <= -M_PI) return angle + 2.0 * M_PI;
  return angle;
}

/**
 * |xi|^2 <= epsilon^2: the finite quaternions the safe normalization (ocs2::quaternionSafeNorm) leaves unchanged, which
 * every heading-tilt function reads as the zero quaternion, i.e. the identity. False for a quaternion with a NaN
 * coefficient, whose squared norm compares false with anything: it is normalized like any other and its NaN reaches
 * every function that reads that coefficient, rather than being read as the identity.
 */
bool isInsideNormalizationGuard(const vector4_t& xi) {
  return xi.squaredNorm() <= kQuaternionNormGuard * kQuaternionNormGuard;
}

/**
 * -1 or 1: the sign that brings the twist coefficients (z, w) to the hemisphere w > 0, or w = 0 < z, of design decision
 * D9. It flips with (z, w), so sign * (z, w) is the same pair for xi and -xi, bit for bit. 1 at z = w = 0.
 */
scalar_t twistHemisphereSign(scalar_t z, scalar_t w) {
  return (w < 0.0 || (w == 0.0 && z < 0.0)) ? -1.0 : 1.0;
}

/** The twist-swing split xi_hat = q_t (x) s of design section 2.8.1, shared by the heading, the tilt and their Jacobian. */
struct TwistSwingSplit {
  /** xi_hat = (x, y, z, w). */
  vector4_t unitQuaternion;
  /** rho = |(z, w)| of xi_hat. */
  scalar_t twistNorm = 0.0;
  /** q_t = (0, 0, headingSin, headingCos), the heading quaternion, a unit quaternion in the hemisphere of xi_hat. */
  scalar_t headingCos = 1.0;
  scalar_t headingSin = 0.0;
  /** s = (swingXy, 0, swingW) = q_t^-1 (x) xi_hat. */
  vector2_t swingXy = vector2_t::Zero();
  scalar_t swingW = 1.0;
};

TwistSwingSplit splitTwistSwing(const vector4_t& xi) {
  TwistSwingSplit split;
  // Inside the guard the safe normalization leaves xi unchanged, and xi_hat = xi is not a unit quaternion: its swing
  // would have n and rho of the order of |xi|, and the absolute series test of tiltScale() would apply the small-tilt
  // series at every tilt (sign-flipped and unbounded near upside down). It is read as the zero quaternion instead, the
  // identity the rotation matrix also reads it as, so the residual is constant there and the zero Jacobian below is its
  // derivative.
  split.unitQuaternion = isInsideNormalizationGuard(xi) ? vector4_t(vector4_t::Zero()) : safelyNormalizedQuaternion(xi);
  const scalar_t x = split.unitQuaternion(kQuaternionXIndex);
  const scalar_t y = split.unitQuaternion(kQuaternionYIndex);
  const scalar_t z = split.unitQuaternion(kQuaternionZIndex);
  const scalar_t w = split.unitQuaternion(kQuaternionWIndex);
  split.twistNorm = std::sqrt(z * z + w * w);
  if (split.twistNorm > kMinimumTwistNorm) {
    split.headingCos = w / split.twistNorm;
    split.headingSin = z / split.twistNorm;
  } else if (z != 0.0 || w != 0.0) {
    // Upside down: the same unit quaternion from the angle, without dividing by a vanishing rho. The angle is taken of
    // (z, w) brought to the hemisphere w > 0 (or w = 0 < z) and the sign is put back afterwards, so that -xi gives
    // exactly -q_t and therefore exactly the same swing; atan2(z, w) of the raw coefficients would differ between xi and
    // -xi by pi only to round-off.
    const scalar_t hemisphere = twistHemisphereSign(z, w);
    const scalar_t halfHeading = std::atan2(hemisphere * z, hemisphere * w);
    split.headingCos = hemisphere * std::cos(halfHeading);
    split.headingSin = hemisphere * std::sin(halfHeading);
  }
  // At rho = 0 exactly the heading quaternion stays the identity of the initializers, for xi and -xi alike: atan2 of two
  // zeros would give +-pi for a negative zero w (atan2(+-0, -0) = +-pi), i.e. minus the identity. The swing of -xi is
  // then -s(xi), the other logarithm of the same half turn (BaseOrientation.h).
  // s = q_t^-1 (x) xi_hat with q_t^-1 = (0, 0, -headingSin, headingCos); its z coefficient is zero.
  const scalar_t c = split.headingCos;
  const scalar_t d = split.headingSin;
  split.swingXy = vector2_t(c * x + d * y, c * y - d * x);
  split.swingW = c * w + d * z;
  return split;
}

/** kappa(n, rho) of design section 2.8.1, with tau = kappa s_xy; n = |s_xy|, rho = s_w. */
scalar_t tiltScale(scalar_t swingVectorNorm, scalar_t swingW) {
  if (swingVectorNorm >= kTiltSeriesThreshold) {
    return 2.0 * std::atan2(swingVectorNorm, swingW) / swingVectorNorm;
  }
  const scalar_t rho = std::max(swingW, kMinimumTwistNorm);
  return (2.0 / rho) * (1.0 - swingVectorNorm * swingVectorNorm / (kTiltSeriesCubicDivisor * rho * rho));
}

vector2_t tiltFromSplit(const TwistSwingSplit& split) {
  return tiltScale(split.swingXy.norm(), split.swingW) * split.swingXy;
}

/** The heading psi_h in (-pi, pi], from the twist coefficients brought to the hemisphere z_w >= 0. */
scalar_t headingFromTwist(scalar_t z, scalar_t w) {
  if (twistHemisphereSign(z, w) < 0.0) {
    z = -z;
    w = -w;
  }
  // Now w >= 0 by value, but a negative zero survives the comparisons, and atan2(+-0, -0) = +-pi would put the heading
  // of an exactly upside-down quaternion such as -(1, 0, 0, 0) = (-1, -0, -0, -0) at +-2 pi. |w| makes the zero positive.
  return 2.0 * std::atan2(z, std::abs(w));
}

/**
 * The 3 x 4 ambient gradients [grad psi_h; grad tau_y; grad tau_x] at xi_hat, rows in the order of the residual (yaw,
 * pitch, roll), coefficients (x, y, z, w); design section 2.8.1.
 */
Eigen::Matrix<scalar_t, kRotationTangentDim, kQuaternionCoefficients> headingTiltAmbientGradient(const TwistSwingSplit& split) {
  const scalar_t x = split.unitQuaternion(kQuaternionXIndex);
  const scalar_t y = split.unitQuaternion(kQuaternionYIndex);
  const scalar_t z = split.unitQuaternion(kQuaternionZIndex);
  const scalar_t w = split.unitQuaternion(kQuaternionWIndex);
  const scalar_t rho = std::max(split.twistNorm, kMinimumTwistNorm);

  Eigen::Matrix<scalar_t, kRotationTangentDim, kQuaternionCoefficients> gradient;
  const scalar_t headingGradientScale = 2.0 / (rho * rho);
  gradient.row(kHeadingRow) << 0.0, 0.0, headingGradientScale * w, -headingGradientScale * z;

  // s_xy = N / rho with N = (w x + z y, w y - z x).
  Eigen::Matrix<scalar_t, kTiltDim, kQuaternionCoefficients> numeratorGradient;
  // clang-format off
  numeratorGradient <<  w, z,  y, x,
                       -z, w, -x, y;
  // clang-format on
  const Eigen::Matrix<scalar_t, 1, kQuaternionCoefficients> twistNormGradient(0.0, 0.0, z / rho, w / rho);
  const vector2_t& swingXy = split.swingXy;
  const Eigen::Matrix<scalar_t, kTiltDim, kQuaternionCoefficients> swingGradient = (numeratorGradient - swingXy * twistNormGradient) / rho;

  // tau = kappa(n, rho) s_xy: d tau = kappa d s_xy + s_xy ((d kappa / dn) / n  s_xy^T d s_xy + d kappa / d rho  d rho).
  const scalar_t swingVectorNorm = swingXy.norm();
  const scalar_t swingW = split.swingW;
  const scalar_t kappa = tiltScale(swingVectorNorm, swingW);
  scalar_t kappaNormDerivativeOverNorm = 0.0;
  if (swingVectorNorm >= kTiltSeriesThreshold) {
    const scalar_t squaredNorm = swingVectorNorm * swingVectorNorm;
    kappaNormDerivativeOverNorm = (2.0 * swingW / (squaredNorm + swingW * swingW) - kappa) / squaredNorm;
  } else {
    const scalar_t seriesRho = std::max(swingW, kMinimumTwistNorm);
    kappaNormDerivativeOverNorm = -kTiltSeriesSlopeNumerator / (kTiltSeriesCubicDivisor * seriesRho * seriesRho * seriesRho);
  }
  const scalar_t swingSquaredNorm = std::max(swingVectorNorm * swingVectorNorm + swingW * swingW, kMinimumTwistNormSquared);
  const scalar_t kappaTwistDerivative = -2.0 / swingSquaredNorm;
  const Eigen::Matrix<scalar_t, kTiltDim, kQuaternionCoefficients> tiltGradient =
      kappa * swingGradient +
      swingXy * (kappaNormDerivativeOverNorm * swingXy.transpose() * swingGradient + kappaTwistDerivative * twistNormGradient);

  gradient.row(kPitchRow) = tiltGradient.row(kYAxisIndex);  // pitch: tau_y
  gradient.row(kRollRow) = tiltGradient.row(kXAxisIndex);   // roll: tau_x
  return gradient;
}

matrix3_t headingTiltJacobianFromSplit(const TwistSwingSplit& split) {
  return headingTiltAmbientGradient(split) * (0.5 * quaternionRateMatrix<scalar_t>(split.unitQuaternion));
}

/** e_HT(xi, xi_r), with `split` the split of xi. */
vector3_t headingTiltResidualFromSplit(const vector4_t& xi, const TwistSwingSplit& split, const vector4_t& xiReference) {
  const vector2_t tilt = tiltFromSplit(split);
  const vector2_t tiltReference = tiltVector(xiReference);
  return vector3_t(wrapAngleDifference(twistHeading(xi) - twistHeading(xiReference)), tilt(1) - tiltReference(1),
                   tilt(0) - tiltReference(0));
}

}  // namespace

scalar_t safeQuaternionNorm(const vector4_t& xi) {
  return quaternionSafeNorm<scalar_t>(xi);
}

ad_scalar_t safeQuaternionNorm(const ad_vector4_t& xi) {
  return quaternionSafeNorm<ad_scalar_t>(xi);
}

vector4_t safelyNormalizedQuaternion(const vector4_t& xi) {
  return quaternionSafeNormalize<scalar_t>(xi);
}

ad_vector4_t safelyNormalizedQuaternion(const ad_vector4_t& xi) {
  return quaternionSafeNormalize<ad_scalar_t>(xi);
}

Eigen::Matrix<scalar_t, kRotationTangentDim, kQuaternionCoefficients> configurationTangentLift(const vector4_t& xi) {
  return quaternionTangentMapPseudoInverse<scalar_t>(xi);
}

scalar_t twistHeading(const vector4_t& xi) {
  // The zero quaternion's heading inside the guard, as splitTwistSwing() reads it. Above it atan2 is invariant to the
  // scale of its arguments, so the raw coefficients serve.
  if (isInsideNormalizationGuard(xi)) return 0.0;
  return headingFromTwist(xi(kQuaternionZIndex), xi(kQuaternionWIndex));
}

vector4_t headingQuaternion(const vector4_t& xi) {
  const TwistSwingSplit split = splitTwistSwing(xi);
  return vector4_t(0.0, 0.0, split.headingSin, split.headingCos);
}

vector4_t swingQuaternion(const vector4_t& xi) {
  const TwistSwingSplit split = splitTwistSwing(xi);
  return vector4_t(split.swingXy(0), split.swingXy(1), 0.0, split.swingW);
}

vector2_t tiltVector(const vector4_t& xi) {
  return tiltFromSplit(splitTwistSwing(xi));
}

vector3_t headingTiltResidual(const vector4_t& xi, const vector4_t& xiReference) {
  return headingTiltResidualFromSplit(xi, splitTwistSwing(xi), xiReference);
}

matrix3_t headingTiltJacobian(const vector4_t& xi) {
  return headingTiltJacobianFromSplit(splitTwistSwing(xi));
}

HeadingTiltError headingTiltError(const vector4_t& xi, const vector4_t& xiReference) {
  const TwistSwingSplit split = splitTwistSwing(xi);
  return HeadingTiltError{
      .residual = headingTiltResidualFromSplit(xi, split, xiReference),
      .jacobianBodyTangent = headingTiltJacobianFromSplit(split),
  };
}

}  // namespace ocs2::humanoid
