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

#include "humanoid_nmpc/humanoid_wb_mpc/test/OrientationErrorRateOracle.h"

#include <algorithm>
#include <array>
#include <cmath>

#include "Eigen/Geometry"
#include "ocs2_robotic_tools/common/RotationTransforms.h"

namespace ocs2::humanoid::test_support {

namespace {

/** Below this angle the coefficients of the left Jacobian are their Taylor series (leftJacobianSo3()). */
constexpr scalar_t kSeriesAngle = 1.0e-2;
/** The largest step of the differences, h, in units of the motion's natural time 1 / max(1, rateScale). */
constexpr scalar_t kRelativeStep = 4.0e-2;
/** The steps of the differences in units of h: central differences at h, h / 2 and h / 4. */
constexpr std::array<scalar_t, 3> kStepFractions = {1.0, 0.5, 0.25};

/** The rotation of the rotation vector `rotationVector`. */
matrix3_t exponential(const vector3_t& rotationVector) {
  const scalar_t angle = rotationVector.norm();
  if (angle == 0.0) return matrix3_t::Identity();
  return Eigen::AngleAxisd(angle, rotationVector / angle).toRotationMatrix();
}

/** The orientation error of the production definition and the normal row's integrand along `motion` at `time`. */
struct ErrorSample {
  vector3_t error = vector3_t::Zero();
  scalar_t halfNormalSpinRate = 0.0;
};

ErrorSample sampleAt(const AttitudeMotion& motion, const vector3_t& planeNormal, scalar_t time) {
  const matrix3_t rotation = rotationAlong(motion, time);
  return ErrorSample{.error = rotationMatrixDistanceToPlane<scalar_t>(rotation, planeNormal),
                     .halfNormalSpinRate = 0.5 * rotation.col(2).dot(angularVelocityAlong(motion, time))};
}

/**
 * Returns Richardson's extrapolation of `estimates`, one central difference at each step of kStepFractions, whose error
 * is a series in even powers of the step: R(h / 2) = (4 D(h / 2) - D(h)) / 3 and R(h / 4) = (4 D(h / 4) - D(h / 2)) / 3,
 * then (16 R(h / 4) - R(h / 2)) / 15, whose truncation is O(h^6). For a vector and for a scalar.
 */
vector3_t extrapolated(const std::array<vector3_t, kStepFractions.size()>& estimates) {
  const vector3_t coarse = (4.0 * estimates[1] - estimates[0]) / 3.0;
  const vector3_t fine = (4.0 * estimates[2] - estimates[1]) / 3.0;
  return (16.0 * fine - coarse) / 15.0;
}
scalar_t extrapolated(const std::array<scalar_t, kStepFractions.size()>& estimates) {
  const scalar_t coarse = (4.0 * estimates[1] - estimates[0]) / 3.0;
  const scalar_t fine = (4.0 * estimates[2] - estimates[1]) / 3.0;
  return (16.0 * fine - coarse) / 15.0;
}

}  // namespace

matrix3_t crossProductMatrix(const vector3_t& vector) {
  matrix3_t matrix;
  matrix << 0.0, -vector(2), vector(1), vector(2), 0.0, -vector(0), -vector(1), vector(0), 0.0;
  return matrix;
}

matrix3_t leftJacobianSo3(const vector3_t& rotationVector) {
  const scalar_t angle = rotationVector.norm();
  const scalar_t squared = angle * angle;
  scalar_t first = 0.0;   // (1 - cos a) / a^2
  scalar_t second = 0.0;  // (a - sin a) / a^3
  if (angle < kSeriesAngle) {
    // Truncated after a^6: the next terms are below 1e-14 / 9! and 1e-14 / 10! at a = 1e-2.
    first = 0.5 - squared / 24.0 + squared * squared / 720.0 - squared * squared * squared / 40320.0;
    second = 1.0 / 6.0 - squared / 120.0 + squared * squared / 5040.0 - squared * squared * squared / 362880.0;
  } else {
    first = (1.0 - std::cos(angle)) / squared;
    second = (angle - std::sin(angle)) / (squared * angle);
  }
  const matrix3_t cross = crossProductMatrix(rotationVector);
  return matrix3_t::Identity() + first * cross + second * cross * cross;
}

matrix3_t rotationAlong(const AttitudeMotion& motion, scalar_t time) {
  const vector3_t rotationVector = motion.angularVelocity * time + 0.5 * time * time * motion.angularAcceleration;
  return exponential(rotationVector) * motion.initialRotation;
}

vector3_t angularVelocityAlong(const AttitudeMotion& motion, scalar_t time) {
  const vector3_t rotationVector = motion.angularVelocity * time + 0.5 * time * time * motion.angularAcceleration;
  const vector3_t rotationVectorRate = motion.angularVelocity + time * motion.angularAcceleration;
  return leftJacobianSo3(rotationVector) * rotationVectorRate;
}

scalar_t orientationErrorRateScale(const AttitudeMotion& motion, const vector3_t& planeNormal) {
  const scalar_t onePlusCosine = 1.0 + motion.initialRotation.col(2).dot(planeNormal);
  const scalar_t turningAngle = std::min(1.0, std::sqrt(2.0 * onePlusCosine));
  return std::max(motion.angularVelocity.norm(), std::sqrt(turningAngle * motion.angularAcceleration.norm())) / turningAngle;
}

DifferencedOrientationErrorRates differencedOrientationErrorRates(const AttitudeMotion& motion, const vector3_t& planeNormal) {
  DifferencedOrientationErrorRates rates;
  rates.rateScale = orientationErrorRateScale(motion, planeNormal);
  const scalar_t largestStep = kRelativeStep / std::max(1.0, rates.rateScale);

  const ErrorSample atZero = sampleAt(motion, planeNormal, /*time=*/0.0);
  std::array<vector3_t, kStepFractions.size()> errorRates;
  std::array<vector3_t, kStepFractions.size()> errorAccelerations;
  std::array<scalar_t, kStepFractions.size()> spinAccelerations = {};
  for (size_t i = 0; i < kStepFractions.size(); ++i) {
    const scalar_t step = kStepFractions[i] * largestStep;
    const ErrorSample plus = sampleAt(motion, planeNormal, step);
    const ErrorSample minus = sampleAt(motion, planeNormal, -step);
    errorRates[i] = (plus.error - minus.error) / (2.0 * step);
    errorAccelerations[i] = (plus.error - 2.0 * atZero.error + minus.error) / (step * step);
    spinAccelerations[i] = (plus.halfNormalSpinRate - minus.halfNormalSpinRate) / (2.0 * step);
  }

  rates.rate = extrapolated(errorRates) + atZero.halfNormalSpinRate * planeNormal;
  rates.acceleration = extrapolated(errorAccelerations) + extrapolated(spinAccelerations) * planeNormal;
  return rates;
}

}  // namespace ocs2::humanoid::test_support
