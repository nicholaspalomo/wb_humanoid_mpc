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

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "Eigen/SVD"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"

/*
 * Step 5 of humanoid_nmpc/docs/quaternion_base_orientation/README.md: the heading-tilt split of design section 2.8.1.
 * The closed-form Jacobian is checked against central finite differences in the body tangent, xi (x) Exp(h e_i), and in
 * the stored coefficients, xi + h e_i; then the residual at a level attitude, at a pitch of 90 degrees and upside down;
 * and every row of the design's property table with its stated value. Quaternions are composed here with Eigen's own
 * Hamilton product and AngleAxis, independently of the library.
 */

namespace ocs2::humanoid {
namespace {

using QuaternionFunction = std::function<vector3_t(const vector4_t&)>;

constexpr scalar_t kFiniteDifferenceStep = 1.0e-6;
/** Central differences at kFiniteDifferenceStep: truncation about 1e-12, round-off about 1e-10. */
constexpr scalar_t kFiniteDifferenceTolerance = 1.0e-8;
constexpr scalar_t kRoundOffTolerance = 1.0e-14;
constexpr scalar_t kDegree = M_PI / 180.0;
constexpr size_t kNumChosenAttitudes = 7;
constexpr size_t kRandomAttitudes = 24;
constexpr scalar_t kMinimumTestTwistNorm = 0.3;

/** a (x) b, with Eigen's Hamilton product. */
vector4_t product(const vector4_t& a, const vector4_t& b) {
  return (quaternion_t(a) * quaternion_t(b)).coeffs();
}

vector4_t conjugate(const vector4_t& xi) {
  return vector4_t(-xi(0), -xi(1), -xi(2), xi(3));
}

vector4_t axisRotation(const vector3_t& axis, scalar_t angle) {
  return quaternion_t(Eigen::AngleAxis<scalar_t>(angle, axis)).coeffs();
}

vector4_t yawRotation(scalar_t yaw) {
  return axisRotation(vector3_t::UnitZ(), yaw);
}

vector4_t pitchRotation(scalar_t pitch) {
  return axisRotation(vector3_t::UnitY(), pitch);
}

vector4_t rollRotation(scalar_t roll) {
  return axisRotation(vector3_t::UnitX(), roll);
}

/** q_z(yaw) (x) q_y(pitch) (x) q_x(roll). */
vector4_t eulerZyxQuaternion(scalar_t yaw, scalar_t pitch, scalar_t roll) {
  return product(yawRotation(yaw), product(pitchRotation(pitch), rollRotation(roll)));
}

/** Exp(phi), the rotation by |phi| about phi / |phi|. */
vector4_t expMap(const vector3_t& rotationVector) {
  const scalar_t angle = rotationVector.norm();
  if (angle == 0.0) return vector4_t(0.0, 0.0, 0.0, 1.0);
  return axisRotation(rotationVector / angle, angle);
}

/** Log(xi) of a unit quaternion, from Eigen's AngleAxis. */
vector3_t logMap(const vector4_t& xi) {
  const Eigen::AngleAxis<scalar_t> angleAxis{quaternion_t(xi)};
  return angleAxis.angle() * angleAxis.axis();
}

vector4_t randomUnitQuaternion(std::mt19937& generator) {
  std::normal_distribution<scalar_t> normal(0.0, 1.0);
  vector4_t xi;
  for (Eigen::Index i = 0; i < 4; ++i) xi(i) = normal(generator);
  return xi / xi.norm();
}

/**
 * Attitudes for the Jacobian checks: a few chosen ones, tilts up to 2.5 rad, and random ones at least 0.3 in twist norm
 * rho. Closer to upside down J_HT grows as 1 / rho and the error of the finite differences as 1 / rho^3; that region has
 * its own test (SingularOnlyUpsideDownAndAtAYawErrorOfPi).
 */
std::vector<vector4_t> testAttitudes() {
  std::vector<vector4_t> attitudes = {
      eulerZyxQuaternion(0.3, 0.2, -0.1),       eulerZyxQuaternion(-2.5, -0.6, 1.2), eulerZyxQuaternion(3.0, 1.2, 2.5),
      eulerZyxQuaternion(1.0, 0.01, 0.02),      eulerZyxQuaternion(-1.7, 2.5, 0.0),  eulerZyxQuaternion(0.4, 0.0, -2.4),
      eulerZyxQuaternion(-0.9, 1.0e-4, 3.0e-4),
  };
  std::mt19937 generator(7);
  while (attitudes.size() < kNumChosenAttitudes + kRandomAttitudes) {
    const vector4_t xi = randomUnitQuaternion(generator);
    if (std::hypot(xi(2), xi(3)) >= kMinimumTestTwistNorm) attitudes.push_back(xi);
  }
  return attitudes;
}

/** A reference whose heading is `headingOffset` from that of xi, tilted by an arbitrary swing. */
vector4_t referenceFor(const vector4_t& xi, scalar_t headingOffset) {
  return product(yawRotation(twistHeading(xi) + headingOffset), product(rollRotation(0.2), pitchRotation(-0.3)));
}

/** d f(xi (x) Exp(dphi)) / d(dphi) at dphi = 0, by central differences. */
matrix_t tangentFiniteDifference(const QuaternionFunction& function, const vector4_t& xi) {
  matrix_t jacobian(3, 3);
  for (Eigen::Index i = 0; i < 3; ++i) {
    const vector3_t step = kFiniteDifferenceStep * vector3_t::Unit(i);
    jacobian.col(i) = (function(product(xi, expMap(step))) - function(product(xi, expMap(-step)))) / (2.0 * kFiniteDifferenceStep);
  }
  return jacobian;
}

/** d f(xi) / d xi in the stored coefficients, by central differences. */
matrix_t ambientFiniteDifference(const QuaternionFunction& function, const vector4_t& xi) {
  matrix_t jacobian(3, 4);
  for (Eigen::Index i = 0; i < 4; ++i) {
    vector4_t forward = xi;
    vector4_t backward = xi;
    forward(i) += kFiniteDifferenceStep;
    backward(i) -= kFiniteDifferenceStep;
    jacobian.col(i) = (function(forward) - function(backward)) / (2.0 * kFiniteDifferenceStep);
  }
  return jacobian;
}

matrix3_t permutationP3() {
  matrix3_t permutation;
  // clang-format off
  permutation << 0.0, 0.0, 1.0,
                 0.0, 1.0, 0.0,
                 1.0, 0.0, 0.0;
  // clang-format on
  return permutation;
}

bool allFinite(const matrix_t& matrix) {
  return matrix.allFinite();
}

/******************************************************************************************************/
/* Safe normalization, rate matrix and tangent lift                                                   */
/******************************************************************************************************/

TEST(BaseOrientation, SafeNormalizationIsTheUnitQuaternionAndKeepsTheZeroQuaternion) {
  std::mt19937 generator(1);
  for (int i = 0; i < 8; ++i) {
    const vector4_t unit = randomUnitQuaternion(generator);
    for (const scalar_t scale : {1.0e-3, 0.5, 1.0, 2.0, 100.0}) {
      const vector4_t xi = scale * unit;
      const vector4_t normalized = safelyNormalizedQuaternion(xi);
      EXPECT_NEAR(normalized.norm(), 1.0, kRoundOffTolerance);
      EXPECT_LT((normalized - unit).cwiseAbs().maxCoeff(), kRoundOffTolerance);
      EXPECT_NEAR(safeQuaternionNorm(xi), scale, kRoundOffTolerance * scale);
      // -xi gives exactly -xi_hat: no sign alignment anywhere (design decision D12).
      EXPECT_EQ(safelyNormalizedQuaternion(vector4_t(-xi)), vector4_t(-normalized));
    }
    // At or below epsilon the quaternion is divided by one: unchanged, so the zero quaternion stays zero.
    const vector4_t tiny = 0.5 * kQuaternionNormGuard * unit;
    EXPECT_EQ(safeQuaternionNorm(tiny), 1.0);
    EXPECT_EQ(safelyNormalizedQuaternion(tiny), tiny);
  }
  EXPECT_EQ(safelyNormalizedQuaternion(vector4_t(vector4_t::Zero())), vector4_t::Zero());
  EXPECT_EQ(safeQuaternionNorm(vector4_t(vector4_t::Zero())), 1.0);
  // The CppAD tape point x = ones is a valid unit quaternion.
  EXPECT_LT((safelyNormalizedQuaternion(vector4_t(vector4_t::Ones())) - 0.5 * vector4_t::Ones()).cwiseAbs().maxCoeff(), kRoundOffTolerance);
}

// The convention of ocs2::quaternionRateMatrix that the closed-form Jacobian and the tangent lift rely on, checked
// against Eigen's own Hamilton product.
TEST(BaseOrientation, QuaternionRateMatrixIsTheHamiltonProductWithAPureQuaternion) {
  std::mt19937 generator(2);
  std::uniform_real_distribution<scalar_t> uniform(-2.0, 2.0);
  for (int i = 0; i < 8; ++i) {
    const vector4_t xi = uniform(generator) * randomUnitQuaternion(generator);
    const vector3_t omega(uniform(generator), uniform(generator), uniform(generator));
    const Eigen::Matrix<scalar_t, 4, 3> rateMatrix = quaternionRateMatrix<scalar_t>(xi);
    const vector4_t pureOmega(omega(0), omega(1), omega(2), 0.0);
    EXPECT_LT((rateMatrix * omega - product(xi, pureOmega)).cwiseAbs().maxCoeff(), kRoundOffTolerance);
    EXPECT_LT((rateMatrix.transpose() * rateMatrix - xi.squaredNorm() * matrix3_t::Identity()).cwiseAbs().maxCoeff(), kRoundOffTolerance);
    EXPECT_LT((rateMatrix.transpose() * xi).cwiseAbs().maxCoeff(), kRoundOffTolerance);
    // d/dt xi (x) Exp(t w) at t = 0 is 1/2 G(xi) w.
    const vector4_t forward = product(xi, expMap(kFiniteDifferenceStep * omega));
    const vector4_t backward = product(xi, expMap(-kFiniteDifferenceStep * omega));
    EXPECT_LT(((forward - backward) / (2.0 * kFiniteDifferenceStep) - 0.5 * rateMatrix * omega).cwiseAbs().maxCoeff(),
              kFiniteDifferenceTolerance);
  }
}

TEST(BaseOrientation, ConfigurationTangentLiftIsTheLeftInverseOfTheTangentMap) {
  std::mt19937 generator(3);
  for (int i = 0; i < 8; ++i) {
    const vector4_t unit = randomUnitQuaternion(generator);
    for (const scalar_t scale : {0.5, 1.0, 2.0}) {
      const vector4_t xi = scale * unit;
      const Eigen::Matrix<scalar_t, 3, 4> lift = configurationTangentLift(xi);
      const Eigen::Matrix<scalar_t, 4, 3> tangentMap = 0.5 * quaternionRateMatrix<scalar_t>(xi);
      EXPECT_LT((lift * tangentMap - matrix3_t::Identity()).cwiseAbs().maxCoeff(), kRoundOffTolerance);
      EXPECT_LT((lift * xi).cwiseAbs().maxCoeff(), kRoundOffTolerance);
    }
    // On the unit sphere E^+ = 2 G^T.
    EXPECT_LT((configurationTangentLift(unit) - 2.0 * quaternionRateMatrix<scalar_t>(unit).transpose()).cwiseAbs().maxCoeff(),
              kRoundOffTolerance);
  }
  EXPECT_TRUE(allFinite(configurationTangentLift(vector4_t::Zero())));
  EXPECT_EQ(configurationTangentLift(vector4_t::Zero()).cwiseAbs().maxCoeff(), 0.0);
}

/******************************************************************************************************/
/* Heading, heading quaternion, swing and tilt                                                        */
/******************************************************************************************************/

TEST(BaseOrientation, HeadingOfALevelAttitudeIsItsYawWrapped) {
  for (const scalar_t yaw : {0.0, 0.3, -2.9, 3.1, M_PI - 1.0e-9, -M_PI + 1.0e-9, 4.0, -7.0, 10.0 * M_PI + 0.1}) {
    const scalar_t wrapped = std::remainder(yaw, 2.0 * M_PI);
    EXPECT_NEAR(twistHeading(yawRotation(yaw)), wrapped, 1.0e-13) << "yaw " << yaw;
    EXPECT_GT(twistHeading(yawRotation(yaw)), -M_PI);
    EXPECT_LE(twistHeading(yawRotation(yaw)), M_PI);
  }
  // Half a turn is pi from either side of the double cover.
  EXPECT_EQ(twistHeading(vector4_t(0.0, 0.0, 1.0, 0.0)), M_PI);
  EXPECT_EQ(twistHeading(vector4_t(0.0, 0.0, -1.0, 0.0)), M_PI);
  // Invariant to the scale and, bit for bit, to the sign of xi.
  for (const vector4_t& xi : testAttitudes()) {
    EXPECT_EQ(twistHeading(vector4_t(-xi)), twistHeading(xi));
    EXPECT_NEAR(twistHeading(vector4_t(3.0 * xi)), twistHeading(xi), kRoundOffTolerance);
  }
  // The zero quaternion is the identity.
  EXPECT_EQ(twistHeading(vector4_t::Zero()), 0.0);
}

TEST(BaseOrientation, HeadingTimesSwingIsTheAttitude) {
  for (const vector4_t& xi : testAttitudes()) {
    const vector4_t heading = headingQuaternion(xi);
    const vector4_t swing = swingQuaternion(xi);
    EXPECT_NEAR(heading.norm(), 1.0, kRoundOffTolerance);
    EXPECT_NEAR(swing.norm(), 1.0, kRoundOffTolerance);
    // q_t is a rotation about z by the heading, in the hemisphere of xi; s is a rotation about a horizontal axis.
    EXPECT_EQ(heading(0), 0.0);
    EXPECT_EQ(heading(1), 0.0);
    EXPECT_GT(heading(2) * xi(2) + heading(3) * xi(3), 0.0);
    EXPECT_NEAR(std::remainder(2.0 * std::atan2(heading(2), heading(3)) - twistHeading(xi), 2.0 * M_PI), 0.0, 1.0e-13);
    EXPECT_EQ(swing(2), 0.0);
    EXPECT_GE(swing(3), 0.0);
    EXPECT_LT((product(heading, swing) - xi).cwiseAbs().maxCoeff(), kRoundOffTolerance);
    // tau is the rotation vector of the swing.
    EXPECT_LT((tiltVector(xi) - logMap(swing).head<2>()).cwiseAbs().maxCoeff(), 1.0e-13);
    // s(-xi) = s(xi) exactly; a rotation about world z changes the heading only.
    EXPECT_EQ(swingQuaternion(vector4_t(-xi)), swing);
    EXPECT_LT((swingQuaternion(product(yawRotation(1.1), xi)) - swing).cwiseAbs().maxCoeff(), kRoundOffTolerance);
  }
}

TEST(BaseOrientation, TiltOfASingleAxisTiltIsExact) {
  for (const scalar_t yaw : {0.0, 1.3, -3.0}) {
    for (const scalar_t angle : {1.0e-9, -1.0e-7, 2.0e-6, 1.0e-3, 0.3, -1.5, 2.0, 3.0}) {
      const vector2_t pitchTilt = tiltVector(product(yawRotation(yaw), pitchRotation(angle)));
      const vector2_t rollTilt = tiltVector(product(yawRotation(yaw), rollRotation(angle)));
      const scalar_t tolerance = kRoundOffTolerance * std::max(1.0, std::abs(angle));
      EXPECT_NEAR(pitchTilt(0), 0.0, tolerance) << yaw << ", " << angle;
      EXPECT_NEAR(pitchTilt(1), angle, tolerance) << yaw << ", " << angle;
      EXPECT_NEAR(rollTilt(0), angle, tolerance) << yaw << ", " << angle;
      EXPECT_NEAR(rollTilt(1), 0.0, tolerance) << yaw << ", " << angle;
    }
  }
}

TEST(BaseOrientation, TiltIsExactlyZeroAndRegularAtALevelAttitude) {
  // n = |s_xy| = 0 at a level attitude: the tilt is the series, exactly zero, never 0 / 0 (the judges' fix).
  for (const scalar_t yaw : {0.0, 0.7, -2.2, M_PI}) {
    const vector4_t level = yawRotation(yaw);
    ASSERT_EQ(level(0), 0.0);
    ASSERT_EQ(level(1), 0.0);
    EXPECT_EQ(tiltVector(level), vector2_t::Zero());
    const HeadingTiltError error = headingTiltError(level, vector4_t(0.0, 0.0, 0.0, 1.0));
    EXPECT_TRUE(error.residual.allFinite());
    EXPECT_TRUE(allFinite(error.jacobianBodyTangent));
    EXPECT_EQ(error.residual(1), 0.0);
    EXPECT_EQ(error.residual(2), 0.0);
  }
  EXPECT_EQ(tiltVector(vector4_t(0.0, 0.0, 0.0, 1.0)), vector2_t::Zero());
  // The zero quaternion is the identity, with a zero (not NaN) Jacobian.
  EXPECT_EQ(headingTiltResidual(vector4_t::Zero(), vector4_t(0.0, 0.0, 0.0, 1.0)), vector3_t::Zero());
  EXPECT_EQ(headingTiltJacobian(vector4_t::Zero()), matrix3_t::Zero());

  // The value and the Jacobian are continuous across the series threshold: n = sin(angle / 2) on either side of it.
  const scalar_t thresholdAngle = 2.0 * std::asin(kTiltSeriesThreshold);
  const vector4_t below = product(yawRotation(0.4), product(pitchRotation(thresholdAngle * (1.0 - 1.0e-6)), rollRotation(0.0)));
  const vector4_t above = product(yawRotation(0.4), product(pitchRotation(thresholdAngle * (1.0 + 1.0e-6)), rollRotation(0.0)));
  EXPECT_NEAR(tiltVector(below)(1), thresholdAngle * (1.0 - 1.0e-6), 1.0e-20);
  EXPECT_NEAR(tiltVector(above)(1), thresholdAngle * (1.0 + 1.0e-6), 1.0e-20);
  EXPECT_LT((headingTiltJacobian(below) - headingTiltJacobian(above)).cwiseAbs().maxCoeff(), 1.0e-11);
  const vector4_t reference = yawRotation(-0.3);
  const QuaternionFunction residual = [&reference](const vector4_t& attitude) { return headingTiltResidual(attitude, reference); };
  for (const vector4_t& xi : {below, above}) {
    EXPECT_LT((headingTiltJacobian(xi) - tangentFiniteDifference(residual, xi)).cwiseAbs().maxCoeff(), kFiniteDifferenceTolerance);
  }
}

/******************************************************************************************************/
/* The closed-form Jacobian                                                                           */
/******************************************************************************************************/

TEST(BaseOrientation, JacobianMatchesFiniteDifferencesInTheBodyTangent) {
  int checked = 0;
  for (const vector4_t& xi : testAttitudes()) {
    for (const scalar_t headingOffset : {-2.5, 0.0, 0.7}) {
      const vector4_t reference = referenceFor(xi, headingOffset);
      const QuaternionFunction residual = [&reference](const vector4_t& attitude) { return headingTiltResidual(attitude, reference); };
      const matrix3_t jacobian = headingTiltJacobian(xi);
      const matrix_t finiteDifference = tangentFiniteDifference(residual, xi);
      EXPECT_LT((jacobian - finiteDifference).cwiseAbs().maxCoeff(), kFiniteDifferenceTolerance)
          << "xi " << xi.transpose() << "\nclosed form\n"
          << jacobian << "\nfinite differences\n"
          << finiteDifference;
      ++checked;
    }
  }
  EXPECT_EQ(checked, 3 * static_cast<int>(testAttitudes().size()));
}

TEST(BaseOrientation, JacobianLiftedByTheTangentLiftMatchesFiniteDifferencesInTheCoefficients) {
  for (const vector4_t& unit : testAttitudes()) {
    const vector4_t reference = referenceFor(unit, /*headingOffset=*/-0.9);
    const QuaternionFunction residual = [&reference](const vector4_t& attitude) { return headingTiltResidual(attitude, reference); };
    for (const scalar_t scale : {0.5, 1.0, 2.0}) {
      // The residual reads xi through xi_hat, so its ambient gradient has no radial part and is J_HT E^+ (decision D4).
      const vector4_t xi = scale * unit;
      const Eigen::Matrix<scalar_t, 3, 4> lifted = headingTiltJacobian(xi) * configurationTangentLift(xi);
      const matrix_t finiteDifference = ambientFiniteDifference(residual, xi);
      EXPECT_LT((lifted - finiteDifference).cwiseAbs().maxCoeff(), kFiniteDifferenceTolerance / scale)
          << "xi " << xi.transpose() << "\nlifted\n"
          << lifted << "\nfinite differences\n"
          << finiteDifference;
      EXPECT_LT((lifted * xi).cwiseAbs().maxCoeff(), kRoundOffTolerance);
    }
  }
}

TEST(BaseOrientation, ResidualAndJacobianAreInvariantToTheScaleOfTheQuaternion) {
  for (const vector4_t& xi : testAttitudes()) {
    const vector4_t reference = referenceFor(xi, /*headingOffset=*/1.1);
    for (const scalar_t scale : {0.5, 2.0}) {
      EXPECT_LT((headingTiltResidual(vector4_t(scale * xi), reference) - headingTiltResidual(xi, reference)).cwiseAbs().maxCoeff(),
                kRoundOffTolerance);
      EXPECT_LT((headingTiltResidual(xi, vector4_t(scale * reference)) - headingTiltResidual(xi, reference)).cwiseAbs().maxCoeff(),
                kRoundOffTolerance);
      EXPECT_LT((headingTiltJacobian(vector4_t(scale * xi)) - headingTiltJacobian(xi)).cwiseAbs().maxCoeff(), 1.0e-13);
    }
  }
}

TEST(BaseOrientation, HeadingTiltErrorIsTheResidualAndTheJacobian) {
  for (const vector4_t& xi : testAttitudes()) {
    const vector4_t reference = referenceFor(xi, /*headingOffset=*/0.3);
    const HeadingTiltError error = headingTiltError(xi, reference);
    EXPECT_EQ(error.residual, headingTiltResidual(xi, reference));
    EXPECT_EQ(error.jacobianBodyTangent, headingTiltJacobian(xi));
  }
}

/******************************************************************************************************/
/* The property table of design section 2.8.1                                                         */
/******************************************************************************************************/

TEST(BaseOrientationProperties, JacobianAtALevelAttitudeIsThePermutation) {
  // J_HT(q_z(psi)) = P_3, so J_HT T_B(0) = I: at a level state (whatever the reference, here a tilted one) the
  // Euler-coordinate Gauss-Newton Hessians of the Euler-weighted rows equal those of the Euler formulation.
  const matrix3_t eulerRateMap = eulerZyxRateToLocalAngularVelocityMatrix(vector3_t::Zero());
  for (const scalar_t yaw : {0.0, 0.4, -1.9, 3.0, M_PI}) {
    const matrix3_t jacobian = headingTiltJacobian(yawRotation(yaw));
    EXPECT_LT((jacobian - permutationP3()).cwiseAbs().maxCoeff(), 1.0e-15) << "yaw " << yaw << "\n" << jacobian;
    EXPECT_LT((jacobian * eulerRateMap - matrix3_t::Identity()).cwiseAbs().maxCoeff(), 1.0e-15);
    // The same, end to end through the Euler chart: d e_HT(phi(Theta), xi_r) / d Theta = I at Theta = (yaw, 0, 0).
    const vector4_t reference = product(yawRotation(yaw - 0.2), pitchRotation(0.05));
    matrix3_t eulerJacobian;
    for (Eigen::Index i = 0; i < 3; ++i) {
      const vector3_t step = kFiniteDifferenceStep * vector3_t::Unit(i);
      const vector3_t level(yaw, 0.0, 0.0);
      eulerJacobian.col(i) = (headingTiltResidual(quaternionFromEulerZyx(vector3_t(level + step)), reference) -
                              headingTiltResidual(quaternionFromEulerZyx(vector3_t(level - step)), reference)) /
                             (2.0 * kFiniteDifferenceStep);
    }
    EXPECT_LT((eulerJacobian - matrix3_t::Identity()).cwiseAbs().maxCoeff(), kFiniteDifferenceTolerance) << eulerJacobian;
  }
}

/** d e_HT(phi(Theta), xi_r) / d Theta at the Euler angles Theta = (yaw, pitch, roll), by central differences. */
matrix3_t eulerChartJacobian(const vector3_t& eulerAngles, const vector4_t& reference) {
  matrix3_t jacobian;
  for (Eigen::Index i = 0; i < 3; ++i) {
    const vector3_t step = kFiniteDifferenceStep * vector3_t::Unit(i);
    jacobian.col(i) = (headingTiltResidual(quaternionFromEulerZyx(vector3_t(eulerAngles + step)), reference) -
                       headingTiltResidual(quaternionFromEulerZyx(vector3_t(eulerAngles - step)), reference)) /
                      (2.0 * kFiniteDifferenceStep);
  }
  return jacobian;
}

TEST(BaseOrientationProperties, EulerChartJacobianIsTheIdentityOnlyAtALevelState) {
  // J_e = J_HT(phi(Theta)) T_B(Theta) is evaluated at the STATE: it is I at a level state whatever the reference
  // (previous test), but not at a tilted state, even against a level reference. There it deviates from I at first
  // order in the tilt, with coefficient 1/2 (the yaw row picks up -tan(theta / 2) d phi and -tan(phi / 2) d theta), so the
  // Euler-coordinate Gauss-Newton Hessians J_e' Q J_e differ from Q by at most about |tilt| |Q|.
  const vector4_t levelReference = yawRotation(0.1);
  const matrix3_t sa01Weights = vector3_t(0.0, 5.0, 85.0).asDiagonal();
  const std::vector<vector3_t> tiltedStates = {vector3_t(0.3, 0.02, 0.0), vector3_t(0.3, 0.05, 0.0),   vector3_t(0.3, 0.2, 0.0),
                                               vector3_t(0.3, 0.0, 0.2),  vector3_t(1.3, 0.05, -0.03), vector3_t(-3.0, 0.2, 0.1)};
  for (const vector3_t& state : tiltedStates) {
    const scalar_t tilt = state.tail<2>().norm();
    const matrix3_t jacobian = eulerChartJacobian(state, levelReference);
    const scalar_t deviation = (jacobian - matrix3_t::Identity()).cwiseAbs().maxCoeff();
    EXPECT_GT(deviation, 0.4 * tilt) << state.transpose() << "\n" << jacobian;
    EXPECT_LT(deviation, 0.6 * tilt) << state.transpose() << "\n" << jacobian;
    const scalar_t hessianDeviation = (jacobian.transpose() * sa01Weights * jacobian - sa01Weights).cwiseAbs().maxCoeff();
    EXPECT_LT(hessianDeviation, tilt * sa01Weights.maxCoeff()) << state.transpose();
  }
}

TEST(BaseOrientationProperties, MixedTiltDeviatesFromTheEulerAnglesByTheStatedFraction) {
  // |tau(q_y(a) q_x(a)) - (phi, theta)| / |(phi, theta)| = 0.066 % at 0.05 rad, 0.26 % at 0.1, 1.06 % at 0.2.
  const std::vector<std::pair<scalar_t, scalar_t>> statedFractions = {{0.05, 0.00066}, {0.1, 0.0026}, {0.2, 0.0106}};
  const std::vector<scalar_t> statedResolution = {0.000005, 0.00005, 0.00005};
  for (size_t i = 0; i < statedFractions.size(); ++i) {
    const scalar_t angle = statedFractions[i].first;
    const vector2_t tilt = tiltVector(eulerZyxQuaternion(/*yaw=*/0.0, angle, angle));
    const vector2_t euler(angle, angle);  // (phi, theta)
    EXPECT_NEAR((tilt - euler).norm() / euler.norm(), statedFractions[i].second, statedResolution[i]) << "tilt " << angle;
  }
}

TEST(BaseOrientationProperties, YawRowDiffersFromTheEulerYawByTheStatedTerm) {
  // e_yaw - (psi - psi_r) = -2 atan(tan(theta / 2) tan(phi / 2)) + 2 atan(tan(theta_r / 2) tan(phi_r / 2)), modulo 2 pi.
  std::mt19937 generator(4);
  std::uniform_real_distribution<scalar_t> yawDistribution(-M_PI, M_PI);
  std::uniform_real_distribution<scalar_t> tiltDistribution(-0.6, 0.6);
  for (int i = 0; i < 16; ++i) {
    const vector3_t euler(yawDistribution(generator), tiltDistribution(generator), tiltDistribution(generator));
    const vector3_t eulerReference(yawDistribution(generator), tiltDistribution(generator), tiltDistribution(generator));
    const scalar_t yawRow = headingTiltResidual(eulerZyxQuaternion(euler(0), euler(1), euler(2)),
                                                eulerZyxQuaternion(eulerReference(0), eulerReference(1), eulerReference(2)))(0);
    const scalar_t stated = -2.0 * std::atan(std::tan(euler(1) / 2.0) * std::tan(euler(2) / 2.0)) +
                            2.0 * std::atan(std::tan(eulerReference(1) / 2.0) * std::tan(eulerReference(2) / 2.0));
    EXPECT_NEAR(std::remainder(yawRow - (euler(0) - eulerReference(0)) - stated, 2.0 * M_PI), 0.0, 1.0e-13);
  }
  // About -theta phi / 2: 0.0113 rad at theta = phi = 0.15 (design decision D9).
  EXPECT_NEAR(twistHeading(eulerZyxQuaternion(0.0, 0.15, 0.15)), -0.0113, 0.00005);
}

TEST(BaseOrientationProperties, TiltRowsAreIndependentOfTheYawError) {
  std::mt19937 generator(5);
  for (int i = 0; i < 8; ++i) {
    const vector4_t xi = randomUnitQuaternion(generator);
    const vector4_t reference = randomUnitQuaternion(generator);
    const vector3_t residual = headingTiltResidual(xi, reference);
    for (const scalar_t yawError : {0.35, M_PI - 0.01, -M_PI + 0.01, -3.0, 10.0 * M_PI + 0.1}) {
      const vector4_t rotated = product(yawRotation(yawError), xi);
      EXPECT_LT((tiltVector(rotated) - tiltVector(xi)).cwiseAbs().maxCoeff(), kRoundOffTolerance);
      const vector3_t rotatedResidual = headingTiltResidual(rotated, reference);
      EXPECT_NEAR(rotatedResidual(1), residual(1), kRoundOffTolerance);
      EXPECT_NEAR(rotatedResidual(2), residual(2), kRoundOffTolerance);
      EXPECT_NEAR(std::remainder(rotatedResidual(0) - residual(0) - yawError, 2.0 * M_PI), 0.0, 1.0e-13);
    }
  }
  // Decision D7's contrast: under a 0.35 rad yaw error the components of Log(R_r^T R) report (0.108, 0.032) for a true
  // (pitch, roll) = (0.1, 0.05); the tilt rows report the tilt without the yaw error.
  const vector4_t yawed = eulerZyxQuaternion(0.35, 0.1, 0.05);
  const vector3_t logResidual = logMap(product(conjugate(vector4_t(0.0, 0.0, 0.0, 1.0)), yawed));
  EXPECT_NEAR(logResidual(1), 0.108, 0.0005);
  EXPECT_NEAR(logResidual(0), 0.032, 0.0005);
  const vector3_t residual = headingTiltResidual(yawed, vector4_t(0.0, 0.0, 0.0, 1.0));
  const vector2_t unyawedTilt = tiltVector(eulerZyxQuaternion(0.0, 0.1, 0.05));
  EXPECT_NEAR(residual(1), unyawedTilt(1), kRoundOffTolerance);
  EXPECT_NEAR(residual(2), unyawedTilt(0), kRoundOffTolerance);
  EXPECT_NEAR(residual(1), 0.1, 0.0005);
  EXPECT_NEAR(residual(2), 0.05, 0.0005);
}

TEST(BaseOrientationProperties, PitchOfNinetyDegreesIsFiniteAndOfFullRank) {
  const vector4_t reference = eulerZyxQuaternion(0.2, -0.1, 0.05);
  const QuaternionFunction residual = [&reference](const vector4_t& attitude) { return headingTiltResidual(attitude, reference); };
  for (const scalar_t pitch : {89.9 * kDegree, 90.0 * kDegree, 90.1 * kDegree, -90.0 * kDegree}) {
    for (const vector2_t& yawRoll : {vector2_t(0.0, 0.0), vector2_t(0.7, -0.4), vector2_t(-2.5, 1.0)}) {
      const vector4_t xi = eulerZyxQuaternion(yawRoll(0), pitch, yawRoll(1));
      const HeadingTiltError error = headingTiltError(xi, reference);
      ASSERT_TRUE(error.residual.allFinite()) << pitch;
      ASSERT_TRUE(allFinite(error.jacobianBodyTangent)) << pitch;
      EXPECT_LT((error.jacobianBodyTangent - tangentFiniteDifference(residual, xi)).cwiseAbs().maxCoeff(), kFiniteDifferenceTolerance)
          << "pitch " << pitch / kDegree << ", yaw " << yawRoll(0) << ", roll " << yawRoll(1);
      const vector3_t singularValues = Eigen::JacobiSVD<matrix3_t>(error.jacobianBodyTangent).singularValues();
      EXPECT_GT(singularValues(2), 0.75);
      if (pitch == 90.0 * kDegree) {
        // The stated singular values, at every yaw and roll.
        EXPECT_NEAR(singularValues(0), 1.95, 0.01);
        EXPECT_NEAR(singularValues(1), 1.00, 0.01);
        EXPECT_NEAR(singularValues(2), 0.80, 0.01);
      }
    }
    // A pure pitch is a single-axis tilt, exact even here.
    EXPECT_NEAR(tiltVector(pitchRotation(pitch))(1), pitch, kRoundOffTolerance);
  }
}

TEST(BaseOrientationProperties, ResidualIsInvariantToTheDoubleCover) {
  for (const vector4_t& xi : testAttitudes()) {
    const vector4_t reference = referenceFor(xi, /*headingOffset=*/-1.4);
    const vector3_t residual = headingTiltResidual(xi, reference);
    // s(-xi) = s(xi) and the heading is read in one hemisphere, so the residual and J_HT are the same bit for bit.
    EXPECT_EQ(headingTiltResidual(vector4_t(-xi), reference), residual);
    EXPECT_EQ(headingTiltResidual(xi, vector4_t(-reference)), residual);
    EXPECT_EQ(headingTiltResidual(vector4_t(-xi), vector4_t(-reference)), residual);
    EXPECT_EQ(headingTiltJacobian(vector4_t(-xi)), headingTiltJacobian(xi));
    EXPECT_EQ(tiltVector(vector4_t(-xi)), tiltVector(xi));
  }
}

TEST(BaseOrientationProperties, HeadingKeepsItsRangeAndTheDoubleCoverAtSignedZeros) {
  // An exactly upside-down attitude and its negation differ only in the signs of zeros: -(1, 0, 0, 0) is
  // (-1, -0, -0, -0). atan2(+-0, -0) = +-pi, so a heading read without care would come out as +-2 pi, and the heading
  // quaternion as minus the identity.
  const vector4_t identity(0.0, 0.0, 0.0, 1.0);
  for (const vector4_t& upsideDown : {vector4_t(1.0, 0.0, 0.0, 0.0), vector4_t(0.6, -0.8, 0.0, 0.0)}) {
    const vector4_t negated = -upsideDown;
    ASSERT_TRUE(std::signbit(negated(2)) == !std::signbit(upsideDown(2)) && std::signbit(negated(3)) == !std::signbit(upsideDown(3)));
    for (const vector4_t& xi : {upsideDown, negated}) {
      const scalar_t heading = twistHeading(xi);
      EXPECT_GT(heading, -M_PI) << xi.transpose();
      EXPECT_LE(heading, M_PI) << xi.transpose();
      EXPECT_EQ(heading, 0.0) << xi.transpose();
      EXPECT_EQ(headingQuaternion(xi), identity) << xi.transpose();
    }
    EXPECT_EQ(twistHeading(negated), twistHeading(upsideDown));
  }
  // Every sign combination of zero twist coefficients, and of a zero w with a nonzero z.
  for (const scalar_t z : {0.0, -0.0}) {
    for (const scalar_t w : {0.0, -0.0}) {
      EXPECT_EQ(twistHeading(vector4_t(1.0, 0.0, z, w)), 0.0) << z << " " << w;
    }
  }
  for (const scalar_t w : {0.0, -0.0}) {
    EXPECT_EQ(twistHeading(vector4_t(0.0, 0.0, 1.0, w)), M_PI);
    EXPECT_EQ(twistHeading(vector4_t(0.0, 0.0, -1.0, w)), M_PI);
  }
}

TEST(BaseOrientationProperties, SingularOnlyUpsideDownAndAtAYawErrorOfPi) {
  // |tau| < pi short of upside down.
  for (const vector4_t& xi : testAttitudes()) {
    EXPECT_LT(tiltVector(xi).norm(), M_PI);
  }
  // Upside down (rho = 0) the heading is read as zero, the tilt is pi about the horizontal axis, and the Jacobian is
  // finite, bounded through kMinimumTwistNorm.
  for (const scalar_t axisAngle : {0.0, 0.6, -2.0}) {
    const vector4_t upsideDown(std::cos(axisAngle), std::sin(axisAngle), 0.0, 0.0);
    EXPECT_EQ(twistHeading(upsideDown), 0.0);
    EXPECT_EQ(headingQuaternion(upsideDown), vector4_t(0.0, 0.0, 0.0, 1.0));
    EXPECT_LT((tiltVector(upsideDown) - M_PI * vector2_t(std::cos(axisAngle), std::sin(axisAngle))).cwiseAbs().maxCoeff(),
              kRoundOffTolerance);
    const HeadingTiltError error = headingTiltError(upsideDown, vector4_t(0.0, 0.0, 0.0, 1.0));
    EXPECT_TRUE(error.residual.allFinite());
    EXPECT_TRUE(allFinite(error.jacobianBodyTangent));
    EXPECT_LT(error.jacobianBodyTangent.cwiseAbs().maxCoeff(), 10.0 / kMinimumTwistNorm);
    // Approaching it: finite, with |tau| -> pi, on both sides of the clamp.
    for (const scalar_t twistNorm : {1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6}) {
      const vector4_t nearlyUpsideDown =
          safelyNormalizedQuaternion(vector4_t(std::cos(axisAngle), std::sin(axisAngle), 0.6 * twistNorm, 0.8 * twistNorm));
      const HeadingTiltError nearError = headingTiltError(nearlyUpsideDown, vector4_t(0.0, 0.0, 0.0, 1.0));
      EXPECT_TRUE(nearError.residual.allFinite()) << twistNorm;
      EXPECT_TRUE(allFinite(nearError.jacobianBodyTangent)) << twistNorm;
      EXPECT_NEAR(tiltVector(nearlyUpsideDown).norm(), M_PI, 3.0 * twistNorm);
    }
  }
  // The yaw row jumps from pi to -pi across a yaw error of pi, and is pi there.
  const vector4_t identity(0.0, 0.0, 0.0, 1.0);
  EXPECT_NEAR(headingTiltResidual(yawRotation(M_PI - 1.0e-6), identity)(0), M_PI - 1.0e-6, 1.0e-12);
  EXPECT_NEAR(headingTiltResidual(yawRotation(M_PI + 1.0e-6), identity)(0), -M_PI + 1.0e-6, 1.0e-12);
  EXPECT_EQ(headingTiltResidual(vector4_t(0.0, 0.0, 1.0, 0.0), identity)(0), M_PI);
}

TEST(BaseOrientationProperties, Sa01WeightedCostChangesByTheStatedFraction) {
  // SA01's shipped Q rows (yaw, pitch, roll) = (0, 5, 85). At a mixed tilt theta = phi = a against a level reference,
  // 1/2 e^T Q e changes by -0.15 % at 0.05 rad, -0.61 % at 0.1 and -2.4 % at 0.2 against the Euler residual.
  const vector3_t weights(0.0, 5.0, 85.0);
  const std::vector<std::pair<scalar_t, scalar_t>> statedFractions = {{0.05, -0.0015}, {0.1, -0.0061}, {0.2, -0.024}};
  const std::vector<scalar_t> statedResolution = {0.00005, 0.00005, 0.0005};
  for (size_t i = 0; i < statedFractions.size(); ++i) {
    const scalar_t angle = statedFractions[i].first;
    const vector3_t residual = headingTiltResidual(eulerZyxQuaternion(/*yaw=*/0.0, angle, angle), vector4_t(0.0, 0.0, 0.0, 1.0));
    const vector3_t eulerResidual(0.0, angle, angle);
    const scalar_t cost = residual.dot(weights.cwiseProduct(residual));
    const scalar_t eulerCost = eulerResidual.dot(weights.cwiseProduct(eulerResidual));
    EXPECT_NEAR((cost - eulerCost) / eulerCost, statedFractions[i].second, statedResolution[i]) << "tilt " << angle;
  }
}

}  // namespace
}  // namespace ocs2::humanoid
