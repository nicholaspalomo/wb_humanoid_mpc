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
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"
#include "ocs2_robotic_tools/common/RotationDerivativesTransforms.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"

#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"

/*
 * The Euler boundary of
 * humanoid_nmpc/docs/quaternion_base_orientation/README.md (section 2.9). Round
 * trips in both directions (the rotation at every attitude, gimbal lock
 * included), the clamped asin at +-90 degrees - for a sine of the pitch past
 * one by round-off and for a measured quaternion slightly longer than one -
 * parity with an independent ZYX conversion with the same clamp
 * (legacyEulerZyxFromQuaternion()), T_B, its inverse and its derivative against
 * the rate of the rotation matrix, and the tuning layout: its placement on the
 * state, the lookup of its blocks, its conversion of task-file-shaped vectors
 * and the errors that name the expected layout.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kRoundOffTolerance = 1.0e-14;
constexpr scalar_t kFiniteDifferenceStep = 1.0e-6;
constexpr scalar_t kFiniteDifferenceTolerance = 1.0e-8;
constexpr scalar_t kDegree = M_PI / 180.0;
/** The G1's joint count, for layouts of the shipped sizes (centroidal 35 -> 36,
 * whole body 58 -> 59). */
constexpr size_t kNumJoints = 23;

/** An independent quaternion-to-ZYX conversion with the clamp of the asin: the
 * reference of the parity cases. */
vector3_t legacyEulerZyxFromQuaternion(const vector4_t& xi) {
  const scalar_t x = xi(0);
  const scalar_t y = xi(1);
  const scalar_t z = xi(2);
  const scalar_t w = xi(3);
  const scalar_t yaw =
      std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
  const scalar_t pitch =
      std::asin(std::clamp(2.0 * (w * y - z * x), -1.0, 1.0));
  const scalar_t roll =
      std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
  return vector3_t(yaw, pitch, roll);
}

/** q_z(yaw) (x) q_y(pitch) (x) q_x(roll) from the half angles, independently of
 * Eigen's AngleAxis. */
vector4_t quaternionFromEulerZyxByHalfAngles(const vector3_t& eulerZyx) {
  const scalar_t cy = std::cos(eulerZyx(0) / 2.0);
  const scalar_t sy = std::sin(eulerZyx(0) / 2.0);
  const scalar_t cp = std::cos(eulerZyx(1) / 2.0);
  const scalar_t sp = std::sin(eulerZyx(1) / 2.0);
  const scalar_t cr = std::cos(eulerZyx(2) / 2.0);
  const scalar_t sr = std::sin(eulerZyx(2) / 2.0);
  return vector4_t(sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                   cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy);
}

/** The distance between the rotations of two unit quaternions, |a - b| or |a +
 * b|, whichever is smaller. */
scalar_t rotationDistance(const vector4_t& a, const vector4_t& b) {
  return std::min((a - b).norm(), (a + b).norm());
}

vector3_t randomEulerAngles(std::mt19937& generator, scalar_t maximumPitch) {
  std::uniform_real_distribution<scalar_t> angle(-M_PI, M_PI);
  std::uniform_real_distribution<scalar_t> pitch(-maximumPitch, maximumPitch);
  const scalar_t yaw = angle(generator);
  const scalar_t pitchAngle = pitch(generator);
  const scalar_t roll = angle(generator);
  return vector3_t(yaw, pitchAngle, roll);
}

/** The Euler angles near and at gimbal lock that the round trips are checked
 * at. */
std::vector<vector3_t> gimbalLockAngles() {
  std::vector<vector3_t> angles;
  for (const scalar_t pitchOffset :
       {0.0, 1.0e-12, 1.0e-9, 1.0e-7, 1.0e-5, 1.0e-4, 1.0e-3, 2.0e-3}) {
    for (const scalar_t sign : {1.0, -1.0}) {
      for (const vector2_t& yawRoll :
           {vector2_t(0.0, 0.0), vector2_t(0.3, 0.7), vector2_t(-2.0, 1.1),
            vector2_t(2.9, -3.0)}) {
        angles.emplace_back(yawRoll(0), sign * (M_PI_2 - pitchOffset),
                            yawRoll(1));
      }
    }
  }
  return angles;
}

/** The centroidal tuning layout of design section 2.3, not placed: [h(6),
 * p_W(3), (yaw, pitch, roll), q_j(nj)]. */
TuningLayout unplacedCentroidalLayout() {
  return TuningLayout{
      .name = "centroidal",
      .segments =
          {
              {.name = "h",
               .kind = TuningSegmentKind::kEuclidean,
               .tuningDim = 6},
              {.name = "p_W",
               .kind = TuningSegmentKind::kEuclidean,
               .tuningDim = 3},
              {.name = "eulerzyx_base_rt_world",
               .kind = TuningSegmentKind::kEulerZyxOrientation,
               .tuningDim = 3},
              {.name = "q_j",
               .kind = TuningSegmentKind::kEuclidean,
               .tuningDim = kNumJoints},
          },
  };
}

/** The centroidal tuning layout placed on a quaternion state [h(6), p_W(3),
 * xi(4), q_j(nj)]. */
TuningLayout centroidalLayout() {
  return placeTuningLayout(unplacedCentroidalLayout());
}

/** The whole-body tuning layout of design section 2.3, placed on a quaternion
 * state [p_W, xi, q_j, pd_W, w_B, qd_j]. */
TuningLayout wholeBodyLayout() {
  return placeTuningLayout(TuningLayout{
      .name = "whole-body",
      .segments =
          {
              {.name = "p_W",
               .kind = TuningSegmentKind::kEuclidean,
               .tuningDim = 3},
              {.name = "eulerzyx_base_rt_world",
               .kind = TuningSegmentKind::kEulerZyxOrientation,
               .tuningDim = 3},
              {.name = "q_j",
               .kind = TuningSegmentKind::kEuclidean,
               .tuningDim = kNumJoints},
              {.name = "pd_W",
               .kind = TuningSegmentKind::kEuclidean,
               .tuningDim = 3},
              {.name = "w_B",
               .kind = TuningSegmentKind::kZyxOrderedAngularVelocity,
               .tuningDim = 3},
              {.name = "qd_j",
               .kind = TuningSegmentKind::kEuclidean,
               .tuningDim = kNumJoints},
          },
  });
}

/** A tuning-layout vector with distinct entries and a valid (yaw, pitch, roll)
 * at `eulerStart`. */
vector_t tuningVector(size_t dim,
                      Eigen::Index eulerStart,
                      const vector3_t& eulerZyx) {
  vector_t vector(static_cast<Eigen::Index>(dim));
  for (Eigen::Index i = 0; i < vector.size(); ++i)
    vector(i) = 0.1 * static_cast<scalar_t>(i) - 1.3;
  vector.segment<3>(eulerStart) = eulerZyx;
  return vector;
}

/******************************************************************************************************/
/* Euler angles and quaternions */
/******************************************************************************************************/

TEST(EulerBoundary, QuaternionFromEulerZyxIsTheZyxProduct) {
  std::mt19937 generator(11);
  for (int i = 0; i < 32; ++i) {
    const vector3_t euler = randomEulerAngles(generator, /*maximumPitch=*/M_PI);
    const vector4_t xi = quaternionFromEulerZyx(euler);
    // Bit for bit the quaternion the human-facing code builds today.
    EXPECT_EQ(xi, getQuaternionFromEulerAnglesZyx<scalar_t>(euler).coeffs());
    EXPECT_LT(
        (xi - quaternionFromEulerZyxByHalfAngles(euler)).cwiseAbs().maxCoeff(),
        kRoundOffTolerance);
    EXPECT_LT((quaternion_t(xi).toRotationMatrix() -
               getRotationMatrixFromZyxEulerAngles<scalar_t>(euler))
                  .cwiseAbs()
                  .maxCoeff(),
              kRoundOffTolerance);
  }
}

TEST(EulerBoundary, EulerRoundTripIsExactBelowNinetyDegrees) {
  std::mt19937 generator(12);
  for (int i = 0; i < 64; ++i) {
    const vector3_t euler =
        randomEulerAngles(generator, /*maximumPitch=*/M_PI_2 - 1.0e-3);
    EXPECT_LT((eulerZyxFromQuaternion(quaternionFromEulerZyx(euler)) - euler)
                  .cwiseAbs()
                  .maxCoeff(),
              1.0e-12)
        << euler.transpose();
  }
  // Yaw and roll come back in (-pi, pi].
  const vector3_t wrapped =
      eulerZyxFromQuaternion(quaternionFromEulerZyx(vector3_t(4.0, 0.2, -3.5)));
  EXPECT_NEAR(wrapped(0), 4.0 - 2.0 * M_PI, 1.0e-12);
  EXPECT_NEAR(wrapped(1), 0.2, 1.0e-12);
  EXPECT_NEAR(wrapped(2), -3.5 + 2.0 * M_PI, 1.0e-12);
}

TEST(EulerBoundary, QuaternionRoundTripReproducesTheRotationAtEveryAttitude) {
  std::mt19937 generator(13);
  std::normal_distribution<scalar_t> normal(0.0, 1.0);
  for (int i = 0; i < 64; ++i) {
    vector4_t xi;
    for (Eigen::Index j = 0; j < 4; ++j) xi(j) = normal(generator);
    xi /= xi.norm();
    const vector3_t euler = eulerZyxFromQuaternion(xi);
    EXPECT_LT(rotationDistance(quaternionFromEulerZyx(euler), xi), 1.0e-12)
        << xi.transpose();
    EXPECT_LE(std::abs(euler(1)), M_PI_2);
  }
  // At and near gimbal lock, where the yaw and roll formulas alone lose the
  // rotation.
  for (const vector3_t& angles : gimbalLockAngles()) {
    const vector4_t xi = quaternionFromEulerZyx(angles);
    const vector3_t euler = eulerZyxFromQuaternion(xi);
    ASSERT_TRUE(euler.allFinite()) << angles.transpose();
    EXPECT_LE(std::abs(euler(1)), M_PI_2);
    EXPECT_LT(rotationDistance(quaternionFromEulerZyx(euler), xi), 1.0e-12)
        << "angles " << angles.transpose() << ", read back as "
        << euler.transpose();
  }
}

TEST(EulerBoundary, AsinIsClampedAtNinetyDegrees) {
  // A pitch of +-90 degrees whose 2 (w y - z x) exceeds 1 by round-off, both
  // coefficients the double just above sqrt(1/2): the unclamped asin is NaN.
  for (const scalar_t sign : {1.0, -1.0}) {
    const vector4_t xi =
        vector4_t(0.0, 0.7071067811865476, 0.0, 0.7071067811865476)
            .cwiseProduct(vector4_t(1.0, sign, 1.0, 1.0));
    ASSERT_GT(std::abs(2.0 * (xi(3) * xi(1) - xi(2) * xi(0))), 1.0);
    ASSERT_TRUE(std::isnan(std::asin(2.0 * (xi(3) * xi(1) - xi(2) * xi(0)))));
    const vector3_t euler = eulerZyxFromQuaternion(xi);
    ASSERT_TRUE(euler.allFinite());
    EXPECT_NEAR(euler(1), sign * M_PI_2, 1.0e-15);
    EXPECT_LT(rotationDistance(quaternionFromEulerZyx(euler), xi), 1.0e-12);
  }
  // Every unit quaternion at exactly +-90 degrees, whichever way its round-off
  // falls.
  for (const scalar_t sign : {1.0, -1.0}) {
    for (int yawStep = -6; yawStep <= 6; ++yawStep) {
      for (int rollStep = -6; rollStep <= 6; ++rollStep) {
        const vector3_t angles(0.5 * yawStep, sign * M_PI_2, 0.5 * rollStep);
        const vector4_t xi = quaternionFromEulerZyx(angles);
        const vector3_t euler = eulerZyxFromQuaternion(xi);
        ASSERT_TRUE(euler.allFinite()) << angles.transpose();
        EXPECT_NEAR(euler(1), sign * M_PI_2, 1.0e-15) << angles.transpose();
        EXPECT_LT(rotationDistance(quaternionFromEulerZyx(euler), xi), 1.0e-12)
            << angles.transpose();
      }
    }
  }

  // A measured quaternion normalized only to round-off - a part in 1e12 or 1e9
  // too long - at exactly +-90 degrees: the sine of the pitch exceeds one by
  // far more than a rounding step, and the angles are still finite and describe
  // its rotation.
  for (const scalar_t sign : {1.0, -1.0}) {
    for (const scalar_t scale : {1.0 + 1.0e-12, 1.0 + 1.0e-9}) {
      const vector4_t xi =
          scale * vector4_t(0.0, sign * std::sqrt(0.5), 0.0, std::sqrt(0.5));
      ASSERT_GT(std::abs(2.0 * (xi(3) * xi(1) - xi(2) * xi(0))), 1.0 + 1.0e-13)
          << "scale " << scale;
      const vector3_t euler = eulerZyxFromQuaternion(xi);
      ASSERT_TRUE(euler.allFinite())
          << "scale " << scale << ": " << euler.transpose();
      EXPECT_LE(std::abs(euler(1)), M_PI_2) << "scale " << scale;
      EXPECT_NEAR(euler(1), sign * M_PI_2, 1.0e-8) << "scale " << scale;
      EXPECT_LT(
          rotationDistance(quaternionFromEulerZyx(euler), xi.normalized()),
          1.0e-8)
          << "scale " << scale;
    }
  }

  // A measured quaternion slightly off the unit sphere near 90 degrees takes
  // the asin formula, and its argument exceeds one by more than round-off: the
  // clamp gives exactly +-pi/2.
  for (const scalar_t sign : {1.0, -1.0}) {
    const vector4_t xi =
        (1.0 + 5.0e-4) *
        quaternionFromEulerZyx(vector3_t(0.4, sign * 89.0 * kDegree, -0.2));
    ASSERT_GT(std::abs(2.0 * (xi(3) * xi(1) - xi(2) * xi(0))), 1.0);
    const vector3_t euler = eulerZyxFromQuaternion(xi);
    ASSERT_TRUE(euler.allFinite());
    EXPECT_EQ(euler(1), sign * M_PI_2);
    EXPECT_EQ(euler, legacyEulerZyxFromQuaternion(xi));
  }
}

TEST(EulerBoundary, MatchesTheConversionOfBeforeTheSwitchBitForBit) {
  // Below kGimbalLockCosPitch's neighborhood of 90 degrees the conversion is
  // the old one, so a call site moved to it (the MRTs, the telemetry)
  // reproduces its old output exactly.
  std::mt19937 generator(14);
  const scalar_t maximumPitch = std::acos(2.0 * kGimbalLockCosPitch);
  for (int i = 0; i < 256; ++i) {
    const vector3_t euler = randomEulerAngles(generator, maximumPitch);
    const vector4_t xi = quaternionFromEulerZyx(euler);
    EXPECT_EQ(eulerZyxFromQuaternion(xi), legacyEulerZyxFromQuaternion(xi))
        << euler.transpose();
  }
  for (const scalar_t pitch : {89.9 * kDegree, -89.9 * kDegree}) {
    const vector4_t xi = quaternionFromEulerZyx(vector3_t(1.0, pitch, -0.5));
    EXPECT_EQ(eulerZyxFromQuaternion(xi), legacyEulerZyxFromQuaternion(xi));
  }
}

TEST(EulerBoundary, LocalAngularVelocityIsTheRateOfTheRotation) {
  std::mt19937 generator(15);
  std::uniform_real_distribution<scalar_t> rate(-2.0, 2.0);
  for (int i = 0; i < 16; ++i) {
    const vector3_t euler = randomEulerAngles(generator, /*maximumPitch=*/1.4);
    const vector3_t eulerRates(rate(generator), rate(generator),
                               rate(generator));
    // w_B = vee(R^T dR/dt), with dR/dt by central differences along Theta + t
    // dTheta/dt.
    const matrix3_t rotation =
        getRotationMatrixFromZyxEulerAngles<scalar_t>(euler);
    const matrix3_t rateRotmatLocalRtWorld =
        (getRotationMatrixFromZyxEulerAngles<scalar_t>(
             vector3_t(euler + kFiniteDifferenceStep * eulerRates)) -
         getRotationMatrixFromZyxEulerAngles<scalar_t>(
             vector3_t(euler - kFiniteDifferenceStep * eulerRates))) /
        (2.0 * kFiniteDifferenceStep);
    const matrix3_t skew = rotation.transpose() * rateRotmatLocalRtWorld;
    const vector3_t omega(skew(2, 1), skew(0, 2), skew(1, 0));
    const vector3_t angvelLocalRtWorldEwrtLocal =
        localAngularVelocityFromEulerZyxRates(euler, eulerRates);
    EXPECT_LT((angvelLocalRtWorldEwrtLocal - omega).cwiseAbs().maxCoeff(),
              kFiniteDifferenceTolerance);
    EXPECT_LT((eulerZyxRateToBodyRateMatrix(euler) * eulerRates -
               angvelLocalRtWorldEwrtLocal)
                  .cwiseAbs()
                  .maxCoeff(),
              kRoundOffTolerance);

    // Design section 2.12: d quaternionFromEulerZyx(Theta) / d Theta = 1/2
    // G(xi) T_B(Theta).
    const vector4_t xi = quaternionFromEulerZyx(euler);
    const Eigen::Matrix<scalar_t, 4, 3> chartJacobian =
        0.5 * quaternionRateMatrix<scalar_t>(xi) *
        eulerZyxRateToBodyRateMatrix(euler);
    for (Eigen::Index j = 0; j < 3; ++j) {
      const vector3_t step = kFiniteDifferenceStep * vector3_t::Unit(j);
      const vector4_t column =
          (quaternionFromEulerZyx(vector3_t(euler + step)) -
           quaternionFromEulerZyx(vector3_t(euler - step))) /
          (2.0 * kFiniteDifferenceStep);
      EXPECT_LT((column - chartJacobian.col(j)).cwiseAbs().maxCoeff(),
                kFiniteDifferenceTolerance);
    }
  }
  // T_B(0) = P_3: at zero attitude the Euler rates (yaw, pitch, roll) are the
  // body rates (z, y, x).
  matrix3_t permutation;
  // clang-format off
  permutation << 0.0, 0.0, 1.0,
                 0.0, 1.0, 0.0,
                 1.0, 0.0, 0.0;
  // clang-format on
  EXPECT_EQ(eulerZyxRateToBodyRateMatrix(vector3_t::Zero()), permutation);
}

TEST(EulerBoundary, EulerRatesAreTheInverseOfTheLocalAngularVelocity) {
  std::mt19937 generator(16);
  std::uniform_real_distribution<scalar_t> rate(-2.0, 2.0);
  for (int i = 0; i < 16; ++i) {
    const vector3_t euler = randomEulerAngles(generator, /*maximumPitch=*/1.4);
    const vector3_t angvelLocalRtWorldEwrtLocal(
        rate(generator), rate(generator), rate(generator));
    const vector3_t eulerRates =
        eulerZyxRatesFromBodyRates(euler, angvelLocalRtWorldEwrtLocal);
    // Bit for bit OCS2's own conversion of the rates.
    EXPECT_EQ(eulerRates,
              // NOLINTNEXTLINE(frame-naming): a generic Euler conversion
              getEulerAnglesZyxDerivativesFromLocalAngularVelocity<scalar_t>(
                  euler, angvelLocalRtWorldEwrtLocal));
    EXPECT_LT((localAngularVelocityFromEulerZyxRates(euler, eulerRates) -
               angvelLocalRtWorldEwrtLocal)
                  .cwiseAbs()
                  .maxCoeff(),
              1.0e-12);
    EXPECT_LT((eulerZyxRateToBodyRateMatrix(euler).inverse() *
                   angvelLocalRtWorldEwrtLocal -
               eulerRates)
                  .cwiseAbs()
                  .maxCoeff(),
              1.0e-12);
  }
}

TEST(EulerBoundary, LocalAngularVelocityJacobianMatchesFiniteDifferences) {
  std::mt19937 generator(17);
  std::uniform_real_distribution<scalar_t> rate(-2.0, 2.0);
  for (int i = 0; i < 16; ++i) {
    const vector3_t euler = randomEulerAngles(generator, /*maximumPitch=*/1.4);
    const vector3_t eulerRates(rate(generator), rate(generator),
                               rate(generator));
    const matrix3_t jacobian = bodyRateEulerZyxJacobian(euler, eulerRates);
    for (Eigen::Index j = 0; j < 3; ++j) {
      const vector3_t step = kFiniteDifferenceStep * vector3_t::Unit(j);
      const vector3_t column = (localAngularVelocityFromEulerZyxRates(
                                    vector3_t(euler + step), eulerRates) -
                                localAngularVelocityFromEulerZyxRates(
                                    vector3_t(euler - step), eulerRates)) /
                               (2.0 * kFiniteDifferenceStep);
      EXPECT_LT((column - jacobian.col(j)).cwiseAbs().maxCoeff(),
                kFiniteDifferenceTolerance)
          << "column " << j;
    }
    // T_B does not depend on the yaw.
    EXPECT_EQ(jacobian.col(0), vector3_t::Zero());
  }
}

/******************************************************************************************************/
/* The tuning layout */
/******************************************************************************************************/

TEST(EulerBoundaryTuningLayout, PlacementPutsTheBlocksConsecutivelyOnTheState) {
  const TuningLayout unplaced = unplacedCentroidalLayout();
  for (const TuningSegment& segment : unplaced.segments) {
    EXPECT_EQ(segment.stateStart, 0U) << segment.name;
    EXPECT_EQ(segment.stateDim, 0U) << segment.name;
  }
  // Every block as many state rows as tuning rows but the orientation, which
  // gets the quaternion's 4: one row more from the orientation on.
  const TuningLayout placedLayout = placeTuningLayout(unplaced);
  ASSERT_EQ(placedLayout.segments.size(), 4U);
  const std::vector<size_t> placedStateStarts = {0, 6, 9, 13};
  const std::vector<size_t> placedStateDims = {6, 3, 4, kNumJoints};
  for (size_t i = 0; i < placedLayout.segments.size(); ++i) {
    EXPECT_EQ(placedLayout.segments[i].stateStart, placedStateStarts[i])
        << placedLayout.segments[i].name;
    EXPECT_EQ(placedLayout.segments[i].stateDim, placedStateDims[i])
        << placedLayout.segments[i].name;
    EXPECT_EQ(placedLayout.segments[i].tuningDim,
              unplaced.segments[i].tuningDim)
        << placedLayout.segments[i].name;
  }
  EXPECT_EQ(getTuningLayoutStateDim(placedLayout),
            getTuningLayoutDim(placedLayout) + 1);
  EXPECT_EQ(getTuningLayoutDim(placedLayout), getTuningLayoutDim(unplaced));

  // The whole body: the angular velocity has as many state rows as tuning rows.
  const TuningLayout wholeBody = wholeBodyLayout();
  EXPECT_EQ(wholeBody.segments[4].stateStart, 10 + kNumJoints);
  EXPECT_EQ(wholeBody.segments[4].stateDim, 3U);
}

TEST(EulerBoundaryTuningLayout, SegmentStartsAreTheTuningOffsetsByName) {
  const TuningLayout layout = wholeBodyLayout();
  const std::vector<std::pair<std::string, size_t>> expected = {
      {"p_W", 0},
      {"eulerzyx_base_rt_world", 3},
      {"q_j", 6},
      {"pd_W", 6 + kNumJoints},
      {"w_B", 9 + kNumJoints},
      {"qd_j", 12 + kNumJoints}};
  for (const std::pair<std::string, size_t>& block : expected) {
    const absl::StatusOr<size_t> start =
        tuningSegmentStart(layout, block.first);
    ASSERT_TRUE(start.ok()) << start.status();
    EXPECT_EQ(*start, block.second) << block.first;
  }
  // The tuning offsets do not depend on the placement.
  for (const TuningLayout& centroidal :
       {unplacedCentroidalLayout(), centroidalLayout()}) {
    const absl::StatusOr<size_t> jointsStart =
        tuningSegmentStart(centroidal, "q_j");
    ASSERT_TRUE(jointsStart.ok()) << jointsStart.status();
    EXPECT_EQ(*jointsStart, 12U);
  }

  const absl::StatusOr<size_t> missing =
      tuningSegmentStart(layout, "base_twist");
  ASSERT_FALSE(missing.ok());
  EXPECT_EQ(missing.status().code(), absl::StatusCode::kNotFound);
  for (const std::string& part :
       {std::string("'base_twist'"), std::string("whole-body tuning layout"),
        std::string("q_j 23")}) {
    EXPECT_TRUE(absl::StrContains(missing.status().message(), part))
        << "'" << part << "' missing from: " << missing.status();
  }
}

TEST(EulerBoundaryTuningLayout, OnlyThePlacedLayoutIsConverted) {
  const vector_t tuning = tuningVector(12 + kNumJoints, /*eulerStart=*/9,
                                       vector3_t(0.4, 0.1, -0.2));

  // The orientation placed by hand on the 3 rows of its Euler angles, the
  // blocks after it moved up to follow on: the placement mismatch names the
  // block and the quaternion's rows.
  TuningLayout threeRows = centroidalLayout();
  threeRows.segments[2].stateDim = 3;
  threeRows.segments[3].stateStart -= 1;
  ASSERT_EQ(getTuningLayoutStateDim(threeRows), getTuningLayoutDim(threeRows));
  const absl::StatusOr<vector_t> threeRowsState =
      stateFromTuningLayout(tuning, threeRows);
  ASSERT_FALSE(threeRowsState.ok());
  EXPECT_EQ(threeRowsState.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(threeRowsState.status().message(),
                                "'eulerzyx_base_rt_world'"))
      << threeRowsState.status();
  EXPECT_TRUE(absl::StrContains(
      threeRowsState.status().message(),
      "with 3 rows; placeTuningLayout() puts it at row 9 with 4."))
      << threeRowsState.status();
  const absl::StatusOr<vector_t> threeRowsTuning =
      tuningLayoutFromState(tuning, threeRows);
  ASSERT_FALSE(threeRowsTuning.ok());
  EXPECT_EQ(threeRowsTuning.status().code(),
            absl::StatusCode::kInvalidArgument);

  // A layout that is not placed, or placed by hand elsewhere.
  const absl::StatusOr<vector_t> unplacedState =
      stateFromTuningLayout(tuning, unplacedCentroidalLayout());
  ASSERT_FALSE(unplacedState.ok());
  EXPECT_EQ(unplacedState.status().code(), absl::StatusCode::kInvalidArgument);
  TuningLayout shifted = centroidalLayout();
  shifted.segments[3].stateStart += 1;
  const absl::StatusOr<vector_t> shiftedState =
      stateFromTuningLayout(tuning, shifted);
  ASSERT_FALSE(shiftedState.ok());
  EXPECT_EQ(shiftedState.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(shiftedState.status().message(), "'q_j'"))
      << shiftedState.status();
}

TEST(EulerBoundaryTuningLayout, CentroidalLayoutConvertsTheEulerRowsOnly) {
  const TuningLayout layout = centroidalLayout();
  ASSERT_EQ(getTuningLayoutDim(layout), 12 + kNumJoints);
  ASSERT_EQ(getTuningLayoutStateDim(layout), 13 + kNumJoints);
  const vector3_t euler(0.8, -0.3, 0.25);
  const vector_t tuning =
      tuningVector(getTuningLayoutDim(layout), /*eulerStart=*/9, euler);
  const absl::StatusOr<vector_t> state = stateFromTuningLayout(tuning, layout);
  ASSERT_TRUE(state.ok()) << state.status();
  ASSERT_EQ(static_cast<size_t>(state->size()),
            getTuningLayoutStateDim(layout));
  // h and p_W at 0..8, the quaternion at 9..12, the joints shifted by one row
  // to 13.., all copied bit for bit.
  EXPECT_EQ(state->head<9>(), tuning.head<9>());
  EXPECT_EQ(vector4_t(state->segment<4>(9)), quaternionFromEulerZyx(euler));
  EXPECT_EQ(state->tail(kNumJoints), tuning.tail(kNumJoints));

  const absl::StatusOr<vector_t> roundTrip =
      tuningLayoutFromState(*state, layout);
  ASSERT_TRUE(roundTrip.ok()) << roundTrip.status();
  EXPECT_EQ(roundTrip->head<9>(), tuning.head<9>());
  EXPECT_LT((roundTrip->segment<3>(9) - euler).cwiseAbs().maxCoeff(), 1.0e-14);
  EXPECT_EQ(roundTrip->tail(kNumJoints), tuning.tail(kNumJoints));

  // The state is read through the safe normalization: a quaternion off the unit
  // sphere reads the same angles.
  vector_t scaledState = *state;
  scaledState.segment<4>(9) *= 1.5;
  const absl::StatusOr<vector_t> scaledRoundTrip =
      tuningLayoutFromState(scaledState, layout);
  ASSERT_TRUE(scaledRoundTrip.ok()) << scaledRoundTrip.status();
  EXPECT_LT((scaledRoundTrip->segment<3>(9) - euler).cwiseAbs().maxCoeff(),
            1.0e-14);
}

TEST(EulerBoundaryTuningLayout, WholeBodyLayoutReordersTheAngularVelocityRows) {
  const TuningLayout layout = wholeBodyLayout();
  ASSERT_EQ(getTuningLayoutDim(layout), 12 + 2 * kNumJoints);
  ASSERT_EQ(getTuningLayoutStateDim(layout), 13 + 2 * kNumJoints);
  const Eigen::Index nj = static_cast<Eigen::Index>(kNumJoints);
  const vector3_t euler(-2.4, 0.1, 0.05);
  const vector_t tuning =
      tuningVector(getTuningLayoutDim(layout), /*eulerStart=*/3, euler);
  const absl::StatusOr<vector_t> state = stateFromTuningLayout(tuning, layout);
  ASSERT_TRUE(state.ok()) << state.status();
  EXPECT_EQ(state->head<3>(), tuning.head<3>());
  EXPECT_EQ(vector4_t(state->segment<4>(3)), quaternionFromEulerZyx(euler));
  EXPECT_EQ(state->segment(7, nj), tuning.segment(6, nj));
  EXPECT_EQ(state->segment(7 + nj, 3), tuning.segment(6 + nj, 3));
  // Tuning rows 9 + nj .. 11 + nj (omega_base_z, omega_base_y, omega_base_x)
  // are w_B = (x, y, z) at 10 + nj .. 12 + nj.
  EXPECT_EQ(state->segment(10 + nj, 3),
            vector3_t(tuning(11 + nj), tuning(10 + nj), tuning(9 + nj)));
  EXPECT_EQ(state->tail(nj), tuning.tail(nj));

  const absl::StatusOr<vector_t> roundTrip =
      tuningLayoutFromState(*state, layout);
  ASSERT_TRUE(roundTrip.ok()) << roundTrip.status();
  EXPECT_LT((*roundTrip - tuning).cwiseAbs().maxCoeff(), 1.0e-14);
  EXPECT_EQ(roundTrip->segment(9 + nj, 3), tuning.segment(9 + nj, 3));
}

TEST(EulerBoundaryTuningLayout, EuclideanLayoutIsTheIdentity) {
  // A layout of Euclidean blocks only: both directions copy the vector bit for
  // bit.
  const TuningLayout layout = placeTuningLayout(TuningLayout{
      .name = "euler_state",
      .segments = {{.name = "x",
                    .kind = TuningSegmentKind::kEuclidean,
                    .tuningDim = 12 + kNumJoints}},
  });
  const vector_t tuning = tuningVector(
      getTuningLayoutDim(layout), /*eulerStart=*/9, vector3_t(0.1, 0.2, 0.3));
  const absl::StatusOr<vector_t> state = stateFromTuningLayout(tuning, layout);
  ASSERT_TRUE(state.ok()) << state.status();
  EXPECT_EQ(*state, tuning);
  const absl::StatusOr<vector_t> back = tuningLayoutFromState(*state, layout);
  ASSERT_TRUE(back.ok()) << back.status();
  EXPECT_EQ(*back, tuning);

  const absl::StatusOr<vector_t> empty =
      stateFromTuningLayout(vector_t(0), TuningLayout{});
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_EQ(empty->size(), 0);
}

TEST(EulerBoundaryTuningLayout, WrongSizesAreRejectedNamingTheExpectedLayout) {
  const TuningLayout layout = centroidalLayout();
  // A quaternion-sized state_weights or initial_state (36 rows) in the 35-row
  // tuning layout, and the reverse.
  const absl::StatusOr<vector_t> tooLong =
      stateFromTuningLayout(vector_t::Zero(13 + kNumJoints), layout);
  ASSERT_FALSE(tooLong.ok());
  EXPECT_EQ(tooLong.status().code(), absl::StatusCode::kInvalidArgument);
  const std::string tooLongMessage(tooLong.status().message());
  for (const std::string& expected :
       {std::string("36 entries"),
        std::string("centroidal tuning layout (35 rows"), std::string("h 6"),
        std::string("p_W 3"),
        std::string("eulerzyx_base_rt_world 3 (yaw, pitch, roll)"),
        std::string("q_j 23"), std::string("initial_state")}) {
    EXPECT_TRUE(absl::StrContains(tooLongMessage, expected))
        << "'" << expected << "' missing from: " << tooLongMessage;
  }
  const absl::StatusOr<vector_t> tooShort =
      stateFromTuningLayout(vector_t::Zero(11 + kNumJoints), layout);
  ASSERT_FALSE(tooShort.ok());
  EXPECT_EQ(tooShort.status().code(), absl::StatusCode::kInvalidArgument);

  const absl::StatusOr<vector_t> eulerSizedState =
      tuningLayoutFromState(vector_t::Zero(12 + kNumJoints), layout);
  ASSERT_FALSE(eulerSizedState.ok());
  EXPECT_EQ(eulerSizedState.status().code(),
            absl::StatusCode::kInvalidArgument);
  const std::string stateMessage(eulerSizedState.status().message());
  for (const std::string& expected :
       {std::string("35 entries"),
        std::string("centroidal state layout (36 rows"),
        std::string("eulerzyx_base_rt_world 4 (quaternion x, y, z, w)")}) {
    EXPECT_TRUE(absl::StrContains(stateMessage, expected))
        << "'" << expected << "' missing from: " << stateMessage;
  }

  const std::string wholeBodyDescription =
      describeTuningLayout(wholeBodyLayout());
  EXPECT_TRUE(
      absl::StrContains(wholeBodyDescription, "w_B 3 (body w_z, w_y, w_x)"))
      << wholeBodyDescription;
}

TEST(EulerBoundaryTuningLayout, RotationBlocksOfTheWrongSizeAreRejected) {
  for (const TuningSegmentKind kind :
       {TuningSegmentKind::kEulerZyxOrientation,
        TuningSegmentKind::kZyxOrderedAngularVelocity}) {
    const TuningLayout layout = placeTuningLayout(TuningLayout{
        .name = "malformed",
        .segments = {{.name = "p_W",
                      .kind = TuningSegmentKind::kEuclidean,
                      .tuningDim = 3},
                     {.name = "rotation", .kind = kind, .tuningDim = 2}},
    });
    const absl::StatusOr<vector_t> state =
        stateFromTuningLayout(vector_t::Zero(5), layout);
    ASSERT_FALSE(state.ok());
    EXPECT_EQ(state.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(absl::StrContains(state.status().message(), "'rotation'"))
        << state.status();
    const absl::StatusOr<vector_t> tuning = tuningLayoutFromState(
        vector_t::Zero(
            static_cast<Eigen::Index>(getTuningLayoutStateDim(layout))),
        layout);
    ASSERT_FALSE(tuning.ok());
    EXPECT_EQ(tuning.status().code(), absl::StatusCode::kInvalidArgument);
  }
}

}  // namespace
}  // namespace ocs2::humanoid
