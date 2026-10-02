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

#include "humanoid_common_mpc/orientation/EulerBoundary.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"

#include "humanoid_common_mpc/orientation/BaseOrientation.h"

/*
 * Step 5 of humanoid_nmpc/docs/quaternion_base_orientation/README.md: the Euler boundary. Round trips in both
 * directions (the rotation at every attitude, gimbal lock included), the clamped asin at +-90 degrees, parity with the
 * conversion of before the switch, T_B against the rate of the rotation matrix, and the tuning layout: its conversion of
 * task.yaml-shaped vectors and the errors that name the expected layout.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kRoundOffTolerance = 1e-14;
constexpr scalar_t kFiniteDifferenceStep = 1e-6;
constexpr scalar_t kFiniteDifferenceTolerance = 1e-8;
constexpr scalar_t kDegree = M_PI / 180.0;
/** The G1's joint count, for layouts of the shipped sizes (centroidal 35 -> 36, whole body 58 -> 59). */
constexpr size_t kNumJoints = 23;

/** The conversion of before the switch, pinocchio_model/DynamicsHelperFunctions.h quaternionToEulerZYX, with the clamp. */
vector3_t legacyEulerZyxFromQuaternion(const vector4_t& xi) {
  const scalar_t x = xi(0);
  const scalar_t y = xi(1);
  const scalar_t z = xi(2);
  const scalar_t w = xi(3);
  const scalar_t yaw = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
  const scalar_t pitch = std::asin(std::clamp(2.0 * (w * y - z * x), -1.0, 1.0));
  const scalar_t roll = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
  return vector3_t(yaw, pitch, roll);
}

/** q_z(yaw) (x) q_y(pitch) (x) q_x(roll) from the half angles, independently of Eigen's AngleAxis. */
vector4_t halfAngleEulerZyxQuaternion(const vector3_t& eulerZyx) {
  const scalar_t cy = std::cos(eulerZyx(0) / 2.0);
  const scalar_t sy = std::sin(eulerZyx(0) / 2.0);
  const scalar_t cp = std::cos(eulerZyx(1) / 2.0);
  const scalar_t sp = std::sin(eulerZyx(1) / 2.0);
  const scalar_t cr = std::cos(eulerZyx(2) / 2.0);
  const scalar_t sr = std::sin(eulerZyx(2) / 2.0);
  return vector4_t(sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy);
}

/** The distance between the rotations of two unit quaternions, |a - b| or |a + b|, whichever is smaller. */
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

/** The Euler angles near and at gimbal lock that the round trips are checked at. */
std::vector<vector3_t> gimbalLockAngles() {
  std::vector<vector3_t> angles;
  for (const scalar_t pitchOffset : {0.0, 1e-12, 1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 2e-3}) {
    for (const scalar_t sign : {1.0, -1.0}) {
      for (const vector2_t& yawRoll : {vector2_t(0.0, 0.0), vector2_t(0.3, 0.7), vector2_t(-2.0, 1.1), vector2_t(2.9, -3.0)}) {
        angles.emplace_back(yawRoll(0), sign * (M_PI_2 - pitchOffset), yawRoll(1));
      }
    }
  }
  return angles;
}

/** The centroidal tuning layout of design section 2.3: [h(6), p_W(3), (yaw, pitch, roll), q_j(nj)]. */
TuningLayout centroidalLayout() {
  return TuningLayout{
      .name = "centroidal",
      .segments =
          {
              {.name = "h", .kind = TuningSegmentKind::kEuclidean, .tuningDim = 6},
              {.name = "p_W", .kind = TuningSegmentKind::kEuclidean, .tuningDim = 3},
              {.name = "base_orientation", .kind = TuningSegmentKind::kEulerZyxOrientation, .tuningDim = 3},
              {.name = "q_j", .kind = TuningSegmentKind::kEuclidean, .tuningDim = kNumJoints},
          },
  };
}

/** The whole-body tuning layout of design section 2.3: [p_W, (yaw, pitch, roll), q_j, pd_W, (w_z, w_y, w_x), qd_j]. */
TuningLayout wholeBodyLayout() {
  return TuningLayout{
      .name = "whole-body",
      .segments =
          {
              {.name = "p_W", .kind = TuningSegmentKind::kEuclidean, .tuningDim = 3},
              {.name = "base_orientation", .kind = TuningSegmentKind::kEulerZyxOrientation, .tuningDim = 3},
              {.name = "q_j", .kind = TuningSegmentKind::kEuclidean, .tuningDim = kNumJoints},
              {.name = "pd_W", .kind = TuningSegmentKind::kEuclidean, .tuningDim = 3},
              {.name = "w_B", .kind = TuningSegmentKind::kZyxOrderedAngularVelocity, .tuningDim = 3},
              {.name = "qd_j", .kind = TuningSegmentKind::kEuclidean, .tuningDim = kNumJoints},
          },
  };
}

/** A tuning-layout vector with distinct entries and a valid (yaw, pitch, roll) at `eulerStart`. */
vector_t tuningVector(size_t dim, Eigen::Index eulerStart, const vector3_t& eulerZyx) {
  vector_t vector(static_cast<Eigen::Index>(dim));
  for (Eigen::Index i = 0; i < vector.size(); ++i) vector(i) = 0.1 * static_cast<scalar_t>(i) - 1.3;
  vector.segment<3>(eulerStart) = eulerZyx;
  return vector;
}

/******************************************************************************************************/
/* Euler angles and quaternions                                                                       */
/******************************************************************************************************/

TEST(EulerBoundary, QuaternionFromEulerZyxIsTheZyxProduct) {
  std::mt19937 generator(11);
  for (int i = 0; i < 32; ++i) {
    const vector3_t euler = randomEulerAngles(generator, /*maximumPitch=*/M_PI);
    const vector4_t xi = quaternionFromEulerZyx(euler);
    // Bit for bit the quaternion the human-facing code builds today.
    EXPECT_EQ(xi, getQuaternionFromEulerAnglesZyx<scalar_t>(euler).coeffs());
    EXPECT_LT((xi - halfAngleEulerZyxQuaternion(euler)).cwiseAbs().maxCoeff(), kRoundOffTolerance);
    EXPECT_LT((quaternion_t(xi).toRotationMatrix() - getRotationMatrixFromZyxEulerAngles<scalar_t>(euler)).cwiseAbs().maxCoeff(),
              kRoundOffTolerance);
  }
}

TEST(EulerBoundary, EulerRoundTripIsExactBelowNinetyDegrees) {
  std::mt19937 generator(12);
  for (int i = 0; i < 64; ++i) {
    const vector3_t euler = randomEulerAngles(generator, /*maximumPitch=*/M_PI_2 - 1e-3);
    EXPECT_LT((eulerZyxFromQuaternion(quaternionFromEulerZyx(euler)) - euler).cwiseAbs().maxCoeff(), 1e-12) << euler.transpose();
  }
  // Yaw and roll come back in (-pi, pi].
  const vector3_t wrapped = eulerZyxFromQuaternion(quaternionFromEulerZyx(vector3_t(4.0, 0.2, -3.5)));
  EXPECT_NEAR(wrapped(0), 4.0 - 2.0 * M_PI, 1e-12);
  EXPECT_NEAR(wrapped(1), 0.2, 1e-12);
  EXPECT_NEAR(wrapped(2), -3.5 + 2.0 * M_PI, 1e-12);
}

TEST(EulerBoundary, QuaternionRoundTripReproducesTheRotationAtEveryAttitude) {
  std::mt19937 generator(13);
  std::normal_distribution<scalar_t> normal(0.0, 1.0);
  for (int i = 0; i < 64; ++i) {
    vector4_t xi;
    for (Eigen::Index j = 0; j < 4; ++j) xi(j) = normal(generator);
    xi /= xi.norm();
    const vector3_t euler = eulerZyxFromQuaternion(xi);
    EXPECT_LT(rotationDistance(quaternionFromEulerZyx(euler), xi), 1e-12) << xi.transpose();
    EXPECT_LE(std::abs(euler(1)), M_PI_2);
  }
  // At and near gimbal lock, where the yaw and roll formulas alone lose the rotation.
  for (const vector3_t& angles : gimbalLockAngles()) {
    const vector4_t xi = quaternionFromEulerZyx(angles);
    const vector3_t euler = eulerZyxFromQuaternion(xi);
    ASSERT_TRUE(euler.allFinite()) << angles.transpose();
    EXPECT_LE(std::abs(euler(1)), M_PI_2);
    EXPECT_LT(rotationDistance(quaternionFromEulerZyx(euler), xi), 1e-12)
        << "angles " << angles.transpose() << ", read back as " << euler.transpose();
  }
}

TEST(EulerBoundary, AsinIsClampedAtNinetyDegrees) {
  // A pitch of +-90 degrees whose 2 (w y - z x) exceeds 1 by round-off, both coefficients the double just above
  // sqrt(1/2): the unclamped asin is NaN.
  for (const scalar_t sign : {1.0, -1.0}) {
    const vector4_t xi = vector4_t(0.0, 0.7071067811865476, 0.0, 0.7071067811865476).cwiseProduct(vector4_t(1.0, sign, 1.0, 1.0));
    ASSERT_GT(std::abs(2.0 * (xi(3) * xi(1) - xi(2) * xi(0))), 1.0);
    ASSERT_TRUE(std::isnan(std::asin(2.0 * (xi(3) * xi(1) - xi(2) * xi(0)))));
    const vector3_t euler = eulerZyxFromQuaternion(xi);
    ASSERT_TRUE(euler.allFinite());
    EXPECT_NEAR(euler(1), sign * M_PI_2, 1e-15);
    EXPECT_LT(rotationDistance(quaternionFromEulerZyx(euler), xi), 1e-12);
  }
  // Every unit quaternion at exactly +-90 degrees, whichever way its round-off falls.
  for (const scalar_t sign : {1.0, -1.0}) {
    for (int yawStep = -6; yawStep <= 6; ++yawStep) {
      for (int rollStep = -6; rollStep <= 6; ++rollStep) {
        const vector3_t angles(0.5 * yawStep, sign * M_PI_2, 0.5 * rollStep);
        const vector4_t xi = quaternionFromEulerZyx(angles);
        const vector3_t euler = eulerZyxFromQuaternion(xi);
        ASSERT_TRUE(euler.allFinite()) << angles.transpose();
        EXPECT_NEAR(euler(1), sign * M_PI_2, 1e-15) << angles.transpose();
        EXPECT_LT(rotationDistance(quaternionFromEulerZyx(euler), xi), 1e-12) << angles.transpose();
      }
    }
  }

  // A measured quaternion slightly off the unit sphere near 90 degrees takes the asin formula, and its argument exceeds
  // one by more than round-off: the clamp gives exactly +-pi/2.
  for (const scalar_t sign : {1.0, -1.0}) {
    const vector4_t xi = (1.0 + 5e-4) * quaternionFromEulerZyx(vector3_t(0.4, sign * 89.0 * kDegree, -0.2));
    ASSERT_GT(std::abs(2.0 * (xi(3) * xi(1) - xi(2) * xi(0))), 1.0);
    const vector3_t euler = eulerZyxFromQuaternion(xi);
    ASSERT_TRUE(euler.allFinite());
    EXPECT_EQ(euler(1), sign * M_PI_2);
    EXPECT_EQ(euler, legacyEulerZyxFromQuaternion(xi));
  }
}

TEST(EulerBoundary, MatchesTheConversionOfBeforeTheSwitchBitForBit) {
  // Below kGimbalLockCosPitch's neighborhood of 90 degrees the conversion is the old one, so a call site moved to it
  // (the MRTs, the telemetry) reproduces its old output exactly.
  std::mt19937 generator(14);
  const scalar_t maximumPitch = std::acos(2.0 * kGimbalLockCosPitch);
  for (int i = 0; i < 256; ++i) {
    const vector3_t euler = randomEulerAngles(generator, maximumPitch);
    const vector4_t xi = quaternionFromEulerZyx(euler);
    EXPECT_EQ(eulerZyxFromQuaternion(xi), legacyEulerZyxFromQuaternion(xi)) << euler.transpose();
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
    const vector3_t eulerRates(rate(generator), rate(generator), rate(generator));
    // w_B = vee(R^T dR/dt), with dR/dt by central differences along Theta + t dTheta/dt.
    const matrix3_t rotation = getRotationMatrixFromZyxEulerAngles<scalar_t>(euler);
    const matrix3_t rotationRate = (getRotationMatrixFromZyxEulerAngles<scalar_t>(vector3_t(euler + kFiniteDifferenceStep * eulerRates)) -
                                    getRotationMatrixFromZyxEulerAngles<scalar_t>(vector3_t(euler - kFiniteDifferenceStep * eulerRates))) /
                                   (2.0 * kFiniteDifferenceStep);
    const matrix3_t skew = rotation.transpose() * rotationRate;
    const vector3_t omega(skew(2, 1), skew(0, 2), skew(1, 0));
    const vector3_t localAngularVelocity = localAngularVelocityFromEulerZyxRates(euler, eulerRates);
    EXPECT_LT((localAngularVelocity - omega).cwiseAbs().maxCoeff(), kFiniteDifferenceTolerance);
    EXPECT_LT((eulerZyxRateToLocalAngularVelocityMatrix(euler) * eulerRates - localAngularVelocity).cwiseAbs().maxCoeff(),
              kRoundOffTolerance);

    // Design section 2.12: d quaternionFromEulerZyx(Theta) / d Theta = 1/2 G(xi) T_B(Theta).
    const vector4_t xi = quaternionFromEulerZyx(euler);
    const Eigen::Matrix<scalar_t, 4, 3> chartJacobian =
        0.5 * quaternionRateMatrix<scalar_t>(xi) * eulerZyxRateToLocalAngularVelocityMatrix(euler);
    for (Eigen::Index j = 0; j < 3; ++j) {
      const vector3_t step = kFiniteDifferenceStep * vector3_t::Unit(j);
      const vector4_t column = (quaternionFromEulerZyx(vector3_t(euler + step)) - quaternionFromEulerZyx(vector3_t(euler - step))) /
                               (2.0 * kFiniteDifferenceStep);
      EXPECT_LT((column - chartJacobian.col(j)).cwiseAbs().maxCoeff(), kFiniteDifferenceTolerance);
    }
  }
  // T_B(0) = P_3: at zero attitude the Euler rates (yaw, pitch, roll) are the body rates (z, y, x).
  matrix3_t permutation;
  // clang-format off
  permutation << 0.0, 0.0, 1.0,
                 0.0, 1.0, 0.0,
                 1.0, 0.0, 0.0;
  // clang-format on
  EXPECT_EQ(eulerZyxRateToLocalAngularVelocityMatrix(vector3_t::Zero()), permutation);
}

/******************************************************************************************************/
/* The tuning layout                                                                                  */
/******************************************************************************************************/

TEST(EulerBoundaryTuningLayout, CentroidalLayoutConvertsTheEulerRowsOnly) {
  const TuningLayout layout = centroidalLayout();
  ASSERT_EQ(getTuningLayoutDim(layout), 12 + kNumJoints);
  ASSERT_EQ(getTuningLayoutStateDim(layout), 13 + kNumJoints);
  const vector3_t euler(0.8, -0.3, 0.25);
  const vector_t tuning = tuningVector(getTuningLayoutDim(layout), /*eulerStart=*/9, euler);
  const absl::StatusOr<vector_t> state = stateFromTuningLayout(tuning, layout);
  ASSERT_TRUE(state.ok()) << state.status();
  ASSERT_EQ(static_cast<size_t>(state->size()), getTuningLayoutStateDim(layout));
  // h and p_W at 0..8, the quaternion at 9..12, the joints shifted by one row to 13.., all copied bit for bit.
  EXPECT_EQ(state->head<9>(), tuning.head<9>());
  EXPECT_EQ(vector4_t(state->segment<4>(9)), quaternionFromEulerZyx(euler));
  EXPECT_EQ(state->tail(kNumJoints), tuning.tail(kNumJoints));

  const absl::StatusOr<vector_t> roundTrip = tuningLayoutFromState(*state, layout);
  ASSERT_TRUE(roundTrip.ok()) << roundTrip.status();
  EXPECT_EQ(roundTrip->head<9>(), tuning.head<9>());
  EXPECT_LT((roundTrip->segment<3>(9) - euler).cwiseAbs().maxCoeff(), 1e-14);
  EXPECT_EQ(roundTrip->tail(kNumJoints), tuning.tail(kNumJoints));

  // The state is read through the safe normalization: a quaternion off the unit sphere reads the same angles.
  vector_t scaledState = *state;
  scaledState.segment<4>(9) *= 1.5;
  const absl::StatusOr<vector_t> scaledRoundTrip = tuningLayoutFromState(scaledState, layout);
  ASSERT_TRUE(scaledRoundTrip.ok()) << scaledRoundTrip.status();
  EXPECT_LT((scaledRoundTrip->segment<3>(9) - euler).cwiseAbs().maxCoeff(), 1e-14);
}

TEST(EulerBoundaryTuningLayout, WholeBodyLayoutReordersTheAngularVelocityRows) {
  const TuningLayout layout = wholeBodyLayout();
  ASSERT_EQ(getTuningLayoutDim(layout), 12 + 2 * kNumJoints);
  ASSERT_EQ(getTuningLayoutStateDim(layout), 13 + 2 * kNumJoints);
  const Eigen::Index nj = static_cast<Eigen::Index>(kNumJoints);
  const vector3_t euler(-2.4, 0.1, 0.05);
  const vector_t tuning = tuningVector(getTuningLayoutDim(layout), /*eulerStart=*/3, euler);
  const absl::StatusOr<vector_t> state = stateFromTuningLayout(tuning, layout);
  ASSERT_TRUE(state.ok()) << state.status();
  EXPECT_EQ(state->head<3>(), tuning.head<3>());
  EXPECT_EQ(vector4_t(state->segment<4>(3)), quaternionFromEulerZyx(euler));
  EXPECT_EQ(state->segment(7, nj), tuning.segment(6, nj));
  EXPECT_EQ(state->segment(7 + nj, 3), tuning.segment(6 + nj, 3));
  // Tuning rows 9 + nj .. 11 + nj (omega_base_z, omega_base_y, omega_base_x) are w_B = (x, y, z) at 10 + nj .. 12 + nj.
  EXPECT_EQ(state->segment(10 + nj, 3), vector3_t(tuning(11 + nj), tuning(10 + nj), tuning(9 + nj)));
  EXPECT_EQ(state->tail(nj), tuning.tail(nj));

  const absl::StatusOr<vector_t> roundTrip = tuningLayoutFromState(*state, layout);
  ASSERT_TRUE(roundTrip.ok()) << roundTrip.status();
  EXPECT_LT((*roundTrip - tuning).cwiseAbs().maxCoeff(), 1e-14);
  EXPECT_EQ(roundTrip->segment(9 + nj, 3), tuning.segment(9 + nj, 3));
}

TEST(EulerBoundaryTuningLayout, EuclideanLayoutIsTheIdentity) {
  // The Euler state of before the switch: every block Euclidean, so both directions copy the vector bit for bit.
  const TuningLayout layout{
      .name = "euler_state",
      .segments = {{.name = "x", .kind = TuningSegmentKind::kEuclidean, .tuningDim = 12 + kNumJoints}},
  };
  const vector_t tuning = tuningVector(getTuningLayoutDim(layout), /*eulerStart=*/9, vector3_t(0.1, 0.2, 0.3));
  const absl::StatusOr<vector_t> state = stateFromTuningLayout(tuning, layout);
  ASSERT_TRUE(state.ok()) << state.status();
  EXPECT_EQ(*state, tuning);
  const absl::StatusOr<vector_t> back = tuningLayoutFromState(*state, layout);
  ASSERT_TRUE(back.ok()) << back.status();
  EXPECT_EQ(*back, tuning);

  const absl::StatusOr<vector_t> empty = stateFromTuningLayout(vector_t(0), TuningLayout{});
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_EQ(empty->size(), 0);
}

TEST(EulerBoundaryTuningLayout, WrongSizesAreRejectedNamingTheExpectedLayout) {
  const TuningLayout layout = centroidalLayout();
  // A quaternion-sized Q or initialState (36 rows) in the 35-row tuning layout, and the reverse.
  const absl::StatusOr<vector_t> tooLong = stateFromTuningLayout(vector_t::Zero(13 + kNumJoints), layout);
  ASSERT_FALSE(tooLong.ok());
  EXPECT_EQ(tooLong.status().code(), absl::StatusCode::kInvalidArgument);
  const std::string tooLongMessage(tooLong.status().message());
  for (const std::string& expected :
       {std::string("36 entries"), std::string("centroidal tuning layout (35 rows"), std::string("h 6"), std::string("p_W 3"),
        std::string("base_orientation 3 (yaw, pitch, roll)"), std::string("q_j 23"), std::string("initialState")}) {
    EXPECT_TRUE(absl::StrContains(tooLongMessage, expected)) << "'" << expected << "' missing from: " << tooLongMessage;
  }
  const absl::StatusOr<vector_t> tooShort = stateFromTuningLayout(vector_t::Zero(11 + kNumJoints), layout);
  ASSERT_FALSE(tooShort.ok());
  EXPECT_EQ(tooShort.status().code(), absl::StatusCode::kInvalidArgument);

  const absl::StatusOr<vector_t> eulerSizedState = tuningLayoutFromState(vector_t::Zero(12 + kNumJoints), layout);
  ASSERT_FALSE(eulerSizedState.ok());
  EXPECT_EQ(eulerSizedState.status().code(), absl::StatusCode::kInvalidArgument);
  const std::string stateMessage(eulerSizedState.status().message());
  for (const std::string& expected : {std::string("35 entries"), std::string("centroidal state layout (36 rows"),
                                      std::string("base_orientation 4 (quaternion x, y, z, w)")}) {
    EXPECT_TRUE(absl::StrContains(stateMessage, expected)) << "'" << expected << "' missing from: " << stateMessage;
  }

  const std::string wholeBodyDescription = describeTuningLayout(wholeBodyLayout());
  EXPECT_TRUE(absl::StrContains(wholeBodyDescription, "w_B 3 (body w_z, w_y, w_x)")) << wholeBodyDescription;
}

TEST(EulerBoundaryTuningLayout, RotationBlocksOfTheWrongSizeAreRejected) {
  for (const TuningSegmentKind kind : {TuningSegmentKind::kEulerZyxOrientation, TuningSegmentKind::kZyxOrderedAngularVelocity}) {
    const TuningLayout layout{
        .name = "malformed",
        .segments = {{.name = "p_W", .kind = TuningSegmentKind::kEuclidean, .tuningDim = 3},
                     {.name = "rotation", .kind = kind, .tuningDim = 2}},
    };
    const absl::StatusOr<vector_t> state = stateFromTuningLayout(vector_t::Zero(5), layout);
    ASSERT_FALSE(state.ok());
    EXPECT_EQ(state.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(absl::StrContains(state.status().message(), "'rotation'")) << state.status();
    const absl::StatusOr<vector_t> tuning =
        tuningLayoutFromState(vector_t::Zero(static_cast<Eigen::Index>(getTuningLayoutStateDim(layout))), layout);
    ASSERT_FALSE(tuning.ok());
    EXPECT_EQ(tuning.status().code(), absl::StatusCode::kInvalidArgument);
  }
}

}  // namespace
}  // namespace ocs2::humanoid
