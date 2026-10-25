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
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"
#include "ocs2_core/PreComputation.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_nmpc/humanoid_wb_mpc/test/AffineEndEffectorDynamics.h"
#include "humanoid_wb_mpc/constraint/EndEffectorDynamicsAccelerationsConstraint.h"
#include "humanoid_wb_mpc/constraint/EndEffectorDynamicsLinearAccConstraint.h"
#include "humanoid_wb_mpc/end_effector/EndEffectorDynamics.h"

/**
 * The stance-foot constraint of the whole-body MPC, g = b + Ax [p; e] + Av d/dt
 * [p; e] + Aa d2/dt2 [p; e], p the position and e the orientation error wrt the
 * ground, driven by an affine stand-in for the end-effector dynamics so that
 * its algebra is checked without a robot model: the velocity and acceleration
 * gains act on the pose-wrt-plane rate and acceleration of the dynamics, not on
 * its linear velocity and acceleration. The linearization is the value's
 * derivative for any Ax and any number of rows; it used to read only the two
 * diagonal 3x3 blocks of Ax and to assume six rows. The rates themselves are
 * testOrientationErrorWrtPlaneRates' and testStanceFootOrientationRates'.
 */
namespace ocs2::humanoid {
namespace {

using test_support::AffineEndEffectorDynamics;
using test_support::kAffineInputDim;
using test_support::kAffineStateDim;

constexpr scalar_t kQueryTime = 0.0;

/** The gains WBMpcInterface::getStanceFootConstraint builds, with distinct
 * values so that no two rows coincide. */
EndEffectorDynamicsAccelerationsConstraint::Config stanceFootConfig() {
  EndEffectorDynamicsAccelerationsConstraint::Config config;
  config.b = (vector_t(6) << 0.01, -0.02, 0.03, -0.04, 0.05, -0.06).finished();
  config.Ax.setZero(6, 6);
  config.Ax(2, 2) = 100.0;  // positionErrorGain_z
  config.Ax.block(3, 3, 3, 3) =
      80.0 * matrix3_t::Identity();  // orientationErrorGain
  config.Av.setIdentity(6, 6);
  config.Av.block(0, 0, 2, 2) = 20.0 * matrix_t::Identity(2, 2);
  config.Av(2, 2) = 10.0;
  config.Av.block(3, 3, 3, 3) = 20.0 * matrix3_t::Identity();
  config.Aa.setIdentity(6, 6);
  config.Aa(2, 2) = 2.0;
  config.Aa.block(3, 3, 3, 3) = 3.0 * matrix3_t::Identity();
  return config;
}

vector_t testState() {
  return (vector_t(kAffineStateDim) << 0.3, -0.7, 1.2, 0.4).finished();
}

vector_t testInput() {
  return (vector_t(kAffineInputDim) << -0.5, 0.9, 0.2, -1.1, 0.6).finished();
}

/** The constraint Create() makes for `config` on `dynamics`, which has its one
 * end effector; null after a failure. */
std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint>
stanceFootConstraint(
    const EndEffectorDynamics<scalar_t>& dynamics,
    EndEffectorDynamicsAccelerationsConstraint::Config config) {
  absl::StatusOr<std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint>>
      constraint = EndEffectorDynamicsAccelerationsConstraint::Create(
          dynamics, /*numConstraints=*/6, std::move(config));
  EXPECT_TRUE(constraint.ok()) << constraint.status();
  return constraint.ok() ? *std::move(constraint) : nullptr;
}

/** End-effector dynamics of none and of two end effectors, which the
 * single-end-effector constraints refuse. */
std::vector<std::vector<std::string>> otherThanOneEndEffector() {
  return {{}, {"left_foot", "right_foot"}};
}

/** Central differences of getValue() in the state (`withRespectToState`) or in
 * the input. */
matrix_t valueJacobian(
    const EndEffectorDynamicsAccelerationsConstraint& constraint,
    const vector_t& state,
    const vector_t& input,
    bool withRespectToState) {
  const scalar_t step = 1.0e-6;
  const Eigen::Index dim = withRespectToState ? state.size() : input.size();
  matrix_t jacobian(constraint.getNumConstraints(kQueryTime), dim);
  for (Eigen::Index i = 0; i < dim; ++i) {
    vector_t statePlus = state;
    vector_t stateMinus = state;
    vector_t inputPlus = input;
    vector_t inputMinus = input;
    if (withRespectToState) {
      statePlus(i) += step;
      stateMinus(i) -= step;
    } else {
      inputPlus(i) += step;
      inputMinus(i) -= step;
    }
    jacobian.col(i) = (constraint.getValue(kQueryTime, statePlus, inputPlus,
                                           PreComputation()) -
                       constraint.getValue(kQueryTime, stateMinus, inputMinus,
                                           PreComputation())) /
                      (2.0 * step);
  }
  return jacobian;
}

TEST(EndEffectorDynamicsAccelerationsConstraint,
     TheLinearizationIsExactForTheStanceFootGains) {
  const AffineEndEffectorDynamics dynamics;
  const std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint> created =
      stanceFootConstraint(dynamics, stanceFootConfig());
  ASSERT_NE(created, nullptr);
  const EndEffectorDynamicsAccelerationsConstraint& constraint = *created;
  const vector_t state = testState();
  const vector_t input = testInput();

  const vector_t value =
      constraint.getValue(kQueryTime, state, input, PreComputation());
  const VectorFunctionLinearApproximation approximation =
      constraint.getLinearApproximation(kQueryTime, state, input,
                                        PreComputation());
  EXPECT_TRUE(approximation.f.isApprox(value, /*prec=*/1.0e-12))
      << approximation.f.transpose() << "\nvs\n"
      << value.transpose();
  EXPECT_TRUE(approximation.dfdx.isApprox(
      valueJacobian(constraint, state, input, /*withRespectToState=*/true),
      /*prec=*/1.0e-8));
  EXPECT_TRUE(approximation.dfdu.isApprox(
      valueJacobian(constraint, state, input, /*withRespectToState=*/false),
      /*prec=*/1.0e-8));

  // The orientation rows of Ax are applied: without the orientation gain the
  // last three rows change by exactly its contribution.
  EndEffectorDynamicsAccelerationsConstraint::Config withoutOrientationGain =
      stanceFootConfig();
  withoutOrientationGain.Ax.block(3, 3, 3, 3).setZero();
  const std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint>
      withoutOrientation =
          stanceFootConstraint(dynamics, withoutOrientationGain);
  ASSERT_NE(withoutOrientation, nullptr);
  const VectorFunctionLinearApproximation reduced =
      withoutOrientation->getLinearApproximation(kQueryTime, state, input,
                                                 PreComputation());
  const vector3_t orientationTerm =
      80.0 * dynamics.orientationErrorWrtPlane(state);
  EXPECT_TRUE((approximation.f - reduced.f)
                  .tail<3>()
                  .isApprox(orientationTerm, /*prec=*/1.0e-12));
  EXPECT_GT(orientationTerm.norm(), 1.0)
      << "the case under test: an orientation error the gain acts on";
}

TEST(EndEffectorDynamicsAccelerationsConstraint,
     TheVelocityAndAccelerationGainsActOnThePoseRatesNotOnTheTwist) {
  const AffineEndEffectorDynamics dynamics;
  const std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint> created =
      stanceFootConstraint(dynamics, stanceFootConfig());
  ASSERT_NE(created, nullptr);
  const vector_t state = testState();
  const vector_t input = testInput();
  const EndEffectorDynamicsAccelerationsConstraint::Config config =
      stanceFootConfig();

  vector6_t footPose;
  footPose << dynamics.position(state),
      dynamics.orientationErrorWrtPlane(state);
  const vector_t expected =
      config.b + config.Ax * footPose +
      config.Av * dynamics.poseWrtPlaneRate(state, input) +
      config.Aa * dynamics.poseWrtPlaneAcceleration(state, input);
  const vector_t value =
      created->getValue(kQueryTime, state, input, PreComputation());
  EXPECT_TRUE(value.isApprox(expected, /*prec=*/1.0e-12))
      << value.transpose() << "\nvs\n"
      << expected.transpose();
  const VectorFunctionLinearApproximation approximation =
      created->getLinearApproximation(kQueryTime, state, input,
                                      PreComputation());
  // The rate maps of the stand-in, with the block-diagonal Ax the linearization
  // reads.
  const std::vector<vector3_t> ground = {vector3_t::UnitZ()};
  const VectorFunctionLinearApproximation rate =
      dynamics.getPoseWrtPlaneRateLinearApproximation(state, input, ground)
          .front();
  const VectorFunctionLinearApproximation acceleration =
      dynamics
          .getPoseWrtPlaneAccelerationLinearApproximation(state, input, ground)
          .front();
  matrix_t expectedDx = config.Av * rate.dfdx + config.Aa * acceleration.dfdx;
  expectedDx.topRows(3) +=
      config.Ax.topLeftCorner(3, 3) *
      dynamics.getPositionLinearApproximation(state).front().dfdx;
  expectedDx.bottomRows(3) +=
      config.Ax.bottomRightCorner(3, 3) *
      dynamics.getOrientationErrorWrtPlaneLinearApproximation(state, ground)
          .front()
          .dfdx;
  const matrix_t expectedDu =
      config.Av * rate.dfdu + config.Aa * acceleration.dfdu;
  EXPECT_TRUE(approximation.dfdx.isApprox(expectedDx, /*prec=*/1.0e-12));
  EXPECT_TRUE(approximation.dfdu.isApprox(expectedDu, /*prec=*/1.0e-12));

  // Positive control: on the stand-in's linear velocity and acceleration, which
  // the swing foot's constraint reads, the linear rows would be other values.
  const vector3_t onTheVelocity =
      (config.b + config.Ax * footPose).head<3>() +
      config.Av.topLeftCorner<3, 3>() * dynamics.velocity(state, input) +
      config.Aa.topLeftCorner<3, 3>() *
          dynamics.linearAcceleration(state, input);
  EXPECT_GT((value.head<3>() - onTheVelocity).cwiseAbs().maxCoeff(), 1.0)
      << "the case under test: rates other than the velocity";
}

/** Returns a rows x cols matrix of coefficients drawn uniformly from [-2, 2]
 * with `generator`. */
matrix_t randomMatrix(Eigen::Index rows,
                      Eigen::Index cols,
                      std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> coefficient(-2.0, 2.0);
  matrix_t matrix(rows, cols);
  for (Eigen::Index i = 0; i < matrix.size(); ++i)
    matrix(i) = coefficient(generator);
  return matrix;
}

TEST(EndEffectorDynamicsAccelerationsConstraint,
     TheLinearizationIsTheDerivativeOfTheValueForAnyAxAndAnyRowCount) {
  // getValue() applies all of Ax to the pose [p; e]; the linearization used to
  // read only Ax's two diagonal 3x3 blocks and to assume six rows. With a
  // random full Ax, Av and Aa of every row count, it is the value and its
  // central differences.
  const AffineEndEffectorDynamics dynamics;
  std::mt19937 generator(/*sd=*/20'261'019);
  for (size_t rows = 1; rows <= 6; ++rows) {
    SCOPED_TRACE(rows);
    const Eigen::Index numRows = static_cast<Eigen::Index>(rows);
    EndEffectorDynamicsAccelerationsConstraint::Config config;
    config.b = randomMatrix(numRows, /*cols=*/1, generator);
    config.Ax = 50.0 * randomMatrix(numRows, /*cols=*/6, generator);
    config.Av = 10.0 * randomMatrix(numRows, /*cols=*/6, generator);
    config.Aa = randomMatrix(numRows, /*cols=*/6, generator);
    absl::StatusOr<std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint>>
        created = EndEffectorDynamicsAccelerationsConstraint::Create(
            dynamics, rows, config);
    ASSERT_TRUE(created.ok()) << created.status();
    const EndEffectorDynamicsAccelerationsConstraint& constraint = **created;
    for (int sample = 0; sample < 3; ++sample) {
      const vector_t state =
          testState() + randomMatrix(kAffineStateDim, /*cols=*/1, generator);
      const vector_t input =
          testInput() + randomMatrix(kAffineInputDim, /*cols=*/1, generator);
      const vector_t value =
          constraint.getValue(kQueryTime, state, input, PreComputation());
      ASSERT_EQ(value.size(), numRows);
      // The value is the formula of the class comment.
      vector6_t footPose;
      footPose << dynamics.position(state),
          dynamics.orientationErrorWrtPlane(state);
      const vector_t expected =
          config.b + config.Ax * footPose +
          config.Av * dynamics.poseWrtPlaneRate(state, input) +
          config.Aa * dynamics.poseWrtPlaneAcceleration(state, input);
      EXPECT_TRUE(value.isApprox(expected, /*prec=*/1.0e-12));
      const VectorFunctionLinearApproximation approximation =
          constraint.getLinearApproximation(kQueryTime, state, input,
                                            PreComputation());
      EXPECT_TRUE(approximation.f.isApprox(value, /*prec=*/1.0e-12))
          << approximation.f.transpose() << "\nvs\n"
          << value.transpose();
      const matrix_t dx =
          valueJacobian(constraint, state, input, /*withRespectToState=*/true);
      const matrix_t du =
          valueJacobian(constraint, state, input, /*withRespectToState=*/false);
      EXPECT_LT((approximation.dfdx - dx).cwiseAbs().maxCoeff(),
                1.0e-6 * std::max(1.0, dx.cwiseAbs().maxCoeff()));
      EXPECT_LT((approximation.dfdu - du).cwiseAbs().maxCoeff(),
                1.0e-6 * std::max(1.0, du.cwiseAbs().maxCoeff()));
      // Positive control: the off-diagonal blocks of Ax carry a state
      // derivative of their own, which the former block-diagonal linearization
      // dropped.
      if (rows == 6) {
        matrix_t blockDiagonal = matrix_t::Zero(6, 6);
        blockDiagonal.topLeftCorner(3, 3) = config.Ax.topLeftCorner(3, 3);
        blockDiagonal.bottomRightCorner(3, 3) =
            config.Ax.bottomRightCorner(3, 3);
        const std::vector<vector3_t> ground = {vector3_t::UnitZ()};
        matrix_t poseJacobian(6, state.size());
        poseJacobian
            << dynamics.getPositionLinearApproximation(state).front().dfdx,
            dynamics
                .getOrientationErrorWrtPlaneLinearApproximation(state, ground)
                .front()
                .dfdx;
        EXPECT_GT(
            ((config.Ax - blockDiagonal) * poseJacobian).cwiseAbs().maxCoeff(),
            1.0e-3);
      }
    }
  }
}

TEST(EndEffectorDynamicsAccelerationsConstraint,
     CreateRefusesCoefficientsOfOtherShapesThanItsRows) {
  const AffineEndEffectorDynamics dynamics;
  for (const size_t rows : {size_t{0}, size_t{7}}) {
    EndEffectorDynamicsAccelerationsConstraint::Config config;
    config.b = vector_t::Zero(static_cast<Eigen::Index>(rows));
    EXPECT_EQ(EndEffectorDynamicsAccelerationsConstraint::Create(dynamics, rows,
                                                                 config)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument)
        << rows;
  }
  EndEffectorDynamicsAccelerationsConstraint::Config shortB =
      stanceFootConfig();
  shortB.b = vector_t::Zero(5);
  EXPECT_EQ(EndEffectorDynamicsAccelerationsConstraint::Create(
                dynamics, /*numConstraints=*/6, shortB)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EndEffectorDynamicsAccelerationsConstraint::Config narrowAv =
      stanceFootConfig();
  narrowAv.Av = matrix_t::Identity(6, 3);
  EXPECT_EQ(EndEffectorDynamicsAccelerationsConstraint::Create(
                dynamics, /*numConstraints=*/6, narrowAv)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EndEffectorDynamicsAccelerationsConstraint::Config threeRows;
  threeRows.b = vector_t::Zero(3);
  threeRows.Ax = matrix_t::Ones(3, 6);
  EXPECT_TRUE(EndEffectorDynamicsAccelerationsConstraint::Create(
                  dynamics, /*numConstraints=*/3, threeRows)
                  .ok());
  // An empty config is the one configure() sets later.
  EXPECT_TRUE(EndEffectorDynamicsAccelerationsConstraint::Create(
                  dynamics, /*numConstraints=*/6, /*config=*/{})
                  .ok());
}

// The stance-foot constraint reads the first end effector of its dynamics:
// dynamics of none or of several are refused by Create(), naming what it was
// given, where the constructor used to throw.
TEST(EndEffectorDynamicsAccelerationsConstraint,
     CreateRefusesDynamicsOfOtherThanOneEndEffector) {
  for (const std::vector<std::string>& ids : otherThanOneEndEffector()) {
    const AffineEndEffectorDynamics dynamics(ids);
    const absl::StatusOr<
        std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint>>
        constraint = EndEffectorDynamicsAccelerationsConstraint::Create(
            dynamics, /*numConstraints=*/6, stanceFootConfig());
    ASSERT_FALSE(constraint.ok()) << ids.size() << " end effectors";
    EXPECT_EQ(constraint.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(absl::StrContains(constraint.status().message(),
                                  "only accepts a single end-effector"))
        << constraint.status();
    for (const std::string& id : ids) {
      EXPECT_TRUE(absl::StrContains(constraint.status().message(), id))
          << constraint.status();
    }
  }
}

// The swing foot's normal-motion constraint likewise.
TEST(EndEffectorDynamicsLinearAccConstraint,
     CreateRefusesDynamicsOfOtherThanOneEndEffector) {
  for (const std::vector<std::string>& ids : otherThanOneEndEffector()) {
    const AffineEndEffectorDynamics dynamics(ids);
    const absl::StatusOr<
        std::unique_ptr<EndEffectorDynamicsLinearAccConstraint>>
        constraint = EndEffectorDynamicsLinearAccConstraint::Create(
            dynamics, /*numConstraints=*/1);
    ASSERT_FALSE(constraint.ok()) << ids.size() << " end effectors";
    EXPECT_EQ(constraint.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(absl::StrContains(constraint.status().message(),
                                  "only accepts a single end-effector"))
        << constraint.status();
  }
  const AffineEndEffectorDynamics oneFoot;
  const absl::StatusOr<std::unique_ptr<EndEffectorDynamicsLinearAccConstraint>>
      constraint = EndEffectorDynamicsLinearAccConstraint::Create(
          oneFoot, /*numConstraints=*/1);
  ASSERT_TRUE(constraint.ok()) << constraint.status();
  EXPECT_EQ((*constraint)->getNumConstraints(kQueryTime), 1u);
}

}  // namespace
}  // namespace ocs2::humanoid
