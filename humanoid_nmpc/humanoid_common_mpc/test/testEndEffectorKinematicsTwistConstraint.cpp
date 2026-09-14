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

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/EndEffectorKinematicsTwistConstraint.h"

/**
 * Tests of the stance-foot twist constraint, driven by an analytic stand-in for the end-effector kinematics so that
 * the constraint algebra can be checked without a robot model.
 *
 * The property under test is that a 6D constraint actually pins all six degrees of freedom of the foot. The
 * orientation error with respect to a plane only measures the tilt of the foot normal, so its rate spans two
 * dimensions and, used on its own, leaves the foot free to pivot about the contact normal.
 */
namespace ocs2::humanoid {
namespace {

constexpr size_t kStateDim = 6;  // [position(3), rotation vector(3)]
constexpr size_t kInputDim = 6;  // [linear velocity(3), angular velocity(3)]
constexpr scalar_t kTol = 1e-9;

matrix3_t rotationFromRotationVector(const vector3_t& rotationVector) {
  const scalar_t angle = rotationVector.norm();
  if (angle < 1e-12) return matrix3_t::Identity();
  return Eigen::AngleAxis<scalar_t>(angle, rotationVector / angle).toRotationMatrix();
}

/**
 * End-effector kinematics whose pose is read straight out of the state and whose twist is the input. The linear
 * approximations are exact for the position and the twist; the orientation-error approximation uses central
 * differences, which is enough for the tests that exercise it.
 */
class AnalyticEndEffectorKinematics final : public EndEffectorKinematics<scalar_t> {
 public:
  AnalyticEndEffectorKinematics() : ids_{"test_foot"} {}
  AnalyticEndEffectorKinematics* clone() const override { return new AnalyticEndEffectorKinematics(*this); }
  const std::vector<std::string>& getIds() const override { return ids_; }

  static vector3_t positionOf(const vector_t& state) { return state.head<3>(); }
  static matrix3_t rotationOf(const vector_t& state) { return rotationFromRotationVector(state.segment<3>(3)); }

  std::vector<vector3_t> getPosition(const vector_t& state) const override { return {positionOf(state)}; }
  std::vector<quaternion_t> getOrientation(const vector_t& state) const override { return {quaternion_t(rotationOf(state))}; }
  std::vector<vector3_t> getVelocity(const vector_t& /*state*/, const vector_t& input) const override { return {input.head<3>()}; }
  std::vector<vector3_t> getAngularVelocity(const vector_t& /*state*/, const vector_t& input) const override { return {input.tail<3>()}; }
  std::vector<vector6_t> getTwist(const vector_t& /*state*/, const vector_t& input) const override { return {vector6_t(input)}; }

  std::vector<vector3_t> getOrientationErrorWrtPlane(const vector_t& state, const std::vector<vector3_t>& planeNormals) const override {
    return {rotationMatrixDistanceToPlane<scalar_t>(rotationOf(state), planeNormals.front())};
  }
  std::vector<vector3_t> getOrientationError(const vector_t& state, const std::vector<quaternion_t>& references) const override {
    return {quaternionDistance<scalar_t>(quaternion_t(rotationOf(state)), references.front())};
  }

  std::vector<VectorFunctionLinearApproximation> getPositionLinearApproximation(const vector_t& state) const override {
    VectorFunctionLinearApproximation approx;
    approx.f = positionOf(state);
    approx.dfdx = matrix_t::Zero(3, state.size());
    approx.dfdx.leftCols(3).setIdentity();
    return {approx};
  }

  std::vector<VectorFunctionLinearApproximation> getTwistLinearApproximation(const vector_t& state, const vector_t& input) const override {
    VectorFunctionLinearApproximation approx;
    approx.f = input;
    approx.dfdx = matrix_t::Zero(6, state.size());
    approx.dfdu = matrix_t::Identity(6, input.size());
    return {approx};
  }

  std::vector<VectorFunctionLinearApproximation> getVelocityLinearApproximation(const vector_t& state,
                                                                                const vector_t& input) const override {
    VectorFunctionLinearApproximation approx;
    approx.f = input.head<3>();
    approx.dfdx = matrix_t::Zero(3, state.size());
    approx.dfdu = matrix_t::Zero(3, input.size());
    approx.dfdu.leftCols(3).setIdentity();
    return {approx};
  }

  std::vector<VectorFunctionLinearApproximation> getAngularVelocityLinearApproximation(const vector_t& state,
                                                                                       const vector_t& input) const override {
    VectorFunctionLinearApproximation approx;
    approx.f = input.tail<3>();
    approx.dfdx = matrix_t::Zero(3, state.size());
    approx.dfdu = matrix_t::Zero(3, input.size());
    approx.dfdu.rightCols(3).setIdentity();
    return {approx};
  }

  std::vector<VectorFunctionLinearApproximation> getOrientationErrorWrtPlaneLinearApproximation(
      const vector_t& state, const std::vector<vector3_t>& planeNormals) const override {
    VectorFunctionLinearApproximation approx;
    approx.f = getOrientationErrorWrtPlane(state, planeNormals).front();
    approx.dfdx = matrix_t::Zero(3, state.size());
    const scalar_t eps = 1e-6;
    for (Eigen::Index i = 0; i < state.size(); ++i) {
      vector_t plus = state, minus = state;
      plus(i) += eps;
      minus(i) -= eps;
      approx.dfdx.col(i) =
          (getOrientationErrorWrtPlane(plus, planeNormals).front() - getOrientationErrorWrtPlane(minus, planeNormals).front()) /
          (2.0 * eps);
    }
    return {approx};
  }

  std::vector<VectorFunctionLinearApproximation> getOrientationErrorLinearApproximation(
      const vector_t& state, const std::vector<quaternion_t>& references) const override {
    VectorFunctionLinearApproximation approx;
    approx.f = getOrientationError(state, references).front();
    approx.dfdx = matrix_t::Zero(3, state.size());
    return {approx};
  }

 private:
  std::vector<std::string> ids_;
};

/** The gains the interface builds for a stance foot, with the defaults of ModelSettings::FootConstraintConfig. */
EndEffectorKinematicsTwistConstraint::Config makeFootConfig(scalar_t positionGainZ = 1.0,
                                                            scalar_t orientationGain = 0.1,
                                                            scalar_t linearVelocityGain = 0.1,
                                                            scalar_t angularVelocityGain = 0.01) {
  EndEffectorKinematicsTwistConstraint::Config config;
  config.b.setZero(6);
  config.Ax.setZero(6, 6);
  config.Av.setZero(6, 6);
  config.Ax(2, 2) = positionGainZ;
  config.Ax.block(3, 3, 3, 3) = matrix3_t::Identity() * orientationGain;
  config.Av(0, 0) = linearVelocityGain;
  config.Av(1, 1) = linearVelocityGain;
  config.Av(2, 2) = linearVelocityGain;
  config.Av(3, 3) = angularVelocityGain;
  config.Av(4, 4) = angularVelocityGain;
  config.Av(5, 5) = angularVelocityGain;
  return config;
}

vector_t makeState(const vector3_t& position, const vector3_t& rotationVector) {
  vector_t state(kStateDim);
  state << position, rotationVector;
  return state;
}

vector_t makeInput(const vector3_t& linearVelocity, const vector3_t& angularVelocity) {
  vector_t input(kInputDim);
  input << linearVelocity, angularVelocity;
  return input;
}

class TwistConstraintTest : public ::testing::Test {
 protected:
  AnalyticEndEffectorKinematics kinematics_;
};

// ==================== The yaw-rate row is opt-in ====================

TEST_F(TwistConstraintTest, YawRateRowIsInactiveByDefault) {
  // The default has to reproduce the behaviour controllers were tuned against: the plane-normal row of the orientation
  // block stays empty, so a stance foot may pivot about the contact normal without any residual.
  EndEffectorKinematicsTwistConstraint constraint(kinematics_, 6, makeFootConfig());
  EXPECT_FALSE(constraint.getConstrainYawRateAboutNormal());

  const vector_t state = makeState(vector3_t::Zero(), vector3_t(0.15, -0.2, 0.5));
  const vector3_t footNormal = AnalyticEndEffectorKinematics::rotationOf(state) * vector3_t::UnitZ();
  const vector_t atRest = constraint.getValue(0.0, state, makeInput(vector3_t::Zero(), vector3_t::Zero()), PreComputation());
  const vector_t spinning = constraint.getValue(0.0, state, makeInput(vector3_t::Zero(), 0.5 * footNormal), PreComputation());
  EXPECT_TRUE((spinning - atRest).isZero(kTol)) << "the default must not react to a spin about the contact normal";

  const auto approx = constraint.getLinearApproximation(0.0, state, makeInput(vector3_t::Zero(), vector3_t::Zero()), PreComputation());
  EXPECT_EQ(approx.dfdu.bottomRightCorner(3, 3).fullPivLu().rank(), 2) << "the orientation rows are rank two by default";
}

// ==================== The 6D constraint has to pin all six degrees of freedom ====================

TEST_F(TwistConstraintTest, YawRateAboutTheContactNormalIsConstrained) {
  // A foot spinning about its own normal keeps its position and its tilt, so every term except the yaw rate is zero.
  // Without the yaw-rate row the last row of a 6D constraint is identically zero and the foot is free to pivot while
  // the constraint reports perfect satisfaction.
  EndEffectorKinematicsTwistConstraint constraint(kinematics_, 6, makeFootConfig());
  constraint.setConstrainYawRateAboutNormal(true);

  for (const vector3_t& rotationVector : {vector3_t(0.0, 0.0, 0.0), vector3_t(0.0, 0.0, 0.8), vector3_t(0.15, -0.2, 0.5)}) {
    const vector_t state = makeState(vector3_t::Zero(), rotationVector);
    const matrix3_t rotation = AnalyticEndEffectorKinematics::rotationOf(state);
    const vector3_t footNormal = rotation * vector3_t::UnitZ();

    const vector_t atRest = constraint.getValue(0.0, state, makeInput(vector3_t::Zero(), vector3_t::Zero()), PreComputation());
    const vector_t spinning = constraint.getValue(0.0, state, makeInput(vector3_t::Zero(), 0.5 * footNormal), PreComputation());

    EXPECT_GT((spinning - atRest).norm(), 1e-4) << "spinning about the contact normal must violate the constraint";
    EXPECT_NEAR(spinning(5) - atRest(5), 0.01 * 0.5 * 0.5, 1e-9)
        << "the yaw row should be the angular velocity gain times half the rate about the contact normal";
  }
}

TEST_F(TwistConstraintTest, AngularBlockOfTheJacobianHasFullRank) {
  EndEffectorKinematicsTwistConstraint constraint(kinematics_, 6, makeFootConfig());
  constraint.setConstrainYawRateAboutNormal(true);
  for (const vector3_t& rotationVector : {vector3_t(0.0, 0.0, 0.0), vector3_t(0.3, 0.1, -0.7), vector3_t(-0.4, 0.25, 1.2)}) {
    const vector_t state = makeState(vector3_t(0.1, -0.2, 0.0), rotationVector);
    const auto approx = constraint.getLinearApproximation(0.0, state, makeInput(vector3_t::Zero(), vector3_t::Zero()), PreComputation());
    const matrix_t angularBlock = approx.dfdu.bottomRightCorner(3, 3);
    EXPECT_EQ(angularBlock.fullPivLu().rank(), 3) << "the orientation rows leave a rotation unconstrained:\n" << angularBlock;
    EXPECT_EQ(approx.dfdu.fullPivLu().rank(), 6) << "the 6D constraint should pin the whole twist";
  }
}

TEST_F(TwistConstraintTest, ZeroTwistSatisfiesAFlatFootAtTheReferenceHeight) {
  EndEffectorKinematicsTwistConstraint constraint(kinematics_, 6, makeFootConfig());
  const vector_t state = makeState(vector3_t(0.3, -0.1, 0.0), vector3_t(0.0, 0.0, 0.9));
  const vector_t value = constraint.getValue(0.0, state, makeInput(vector3_t::Zero(), vector3_t::Zero()), PreComputation());
  EXPECT_TRUE(value.isZero(kTol)) << "a flat, still foot on the ground plane should satisfy the constraint: " << value.transpose();
}

TEST_F(TwistConstraintTest, TiltRowsStillActAsAProportionalDerivativeLaw) {
  // A tilted foot whose tilt is being removed at exactly the rate the PD law asks for should satisfy the orientation
  // rows, which is only true if the rate mapping is the true derivative of the orientation error.
  const scalar_t orientationGain = 0.1;
  const scalar_t angularVelocityGain = 0.01;
  EndEffectorKinematicsTwistConstraint constraint(kinematics_, 6, makeFootConfig(1.0, orientationGain, 0.1, angularVelocityGain));
  constraint.setConstrainYawRateAboutNormal(true);

  const vector_t state = makeState(vector3_t::Zero(), vector3_t(0.12, -0.09, 0.4));
  const vector3_t error = kinematics_.getOrientationErrorWrtPlane(state, {vector3_t::UnitZ()}).front();

  // Numerically find the angular velocity whose residual is zero by solving the linearized rows.
  const auto approx = constraint.getLinearApproximation(0.0, state, makeInput(vector3_t::Zero(), vector3_t::Zero()), PreComputation());
  const matrix_t angularBlock = approx.dfdu.bottomRightCorner(3, 3);
  const vector3_t residualAtRest = approx.f.tail(3);
  const vector3_t omega = angularBlock.fullPivLu().solve(-residualAtRest);

  const vector_t value = constraint.getValue(0.0, state, makeInput(vector3_t::Zero(), omega), PreComputation());
  EXPECT_TRUE(value.tail(3).isZero(1e-9)) << "orientation rows should vanish at the PD solution: " << value.tail(3).transpose();
  EXPECT_GT(error.norm(), 1e-3) << "the test state should actually be tilted";
  // Removing the tilt requires rotating about an axis perpendicular to the plane normal.
  EXPECT_GT(omega.head<2>().norm(), 1e-3);
}

// ==================== Value and linearization agree ====================

TEST_F(TwistConstraintTest, LinearApproximationMatchesTheValue) {
  for (size_t numConstraints : {size_t(3), size_t(6)}) {
    EndEffectorKinematicsTwistConstraint constraint(kinematics_, numConstraints, makeFootConfig());
    const vector_t state = makeState(vector3_t(0.2, 0.1, 0.03), vector3_t(0.1, -0.05, 0.6));
    const vector_t input = makeInput(vector3_t(0.4, -0.2, 0.1), vector3_t(0.3, 0.2, -0.5));

    const vector_t value = constraint.getValue(0.0, state, input, PreComputation());
    const auto approx = constraint.getLinearApproximation(0.0, state, input, PreComputation());
    EXPECT_EQ(constraint.getNumConstraints(0.0), numConstraints);
    EXPECT_EQ(static_cast<size_t>(value.size()), numConstraints);
    EXPECT_TRUE(approx.f.isApprox(value, 1e-9))
        << "n=" << numConstraints << ": value " << value.transpose() << " vs approximation " << approx.f.transpose();
  }
}

TEST_F(TwistConstraintTest, InputJacobianMatchesFiniteDifferences) {
  EndEffectorKinematicsTwistConstraint constraint(kinematics_, 6, makeFootConfig());
  constraint.setConstrainYawRateAboutNormal(true);
  const vector_t state = makeState(vector3_t(0.2, 0.1, 0.03), vector3_t(0.1, -0.05, 0.6));
  const vector_t input = makeInput(vector3_t(0.4, -0.2, 0.1), vector3_t(0.3, 0.2, -0.5));

  const auto approx = constraint.getLinearApproximation(0.0, state, input, PreComputation());
  matrix_t finiteDifference = matrix_t::Zero(6, kInputDim);
  const scalar_t eps = 1e-6;
  for (size_t i = 0; i < kInputDim; ++i) {
    vector_t plus = input, minus = input;
    plus(i) += eps;
    minus(i) -= eps;
    finiteDifference.col(i) =
        (constraint.getValue(0.0, state, plus, PreComputation()) - constraint.getValue(0.0, state, minus, PreComputation())) / (2.0 * eps);
  }
  EXPECT_TRUE(approx.dfdu.isApprox(finiteDifference, 1e-6)) << "dfdu =\n" << approx.dfdu << "\nfinite differences =\n" << finiteDifference;
}

TEST_F(TwistConstraintTest, OffDiagonalGainsAreAppliedByBothValueAndLinearization) {
  // The two code paths used to disagree for non-diagonal gain matrices: the value applied the whole gain row while
  // the linearization only used the diagonal 3x3 blocks.
  EndEffectorKinematicsTwistConstraint::Config config = makeFootConfig();
  config.Ax(0, 4) = 0.7;  // x position row driven by the second orientation error component
  config.Av(1, 5) = 0.3;  // y velocity row driven by the third orientation rate component
  config.Av(4, 0) = 0.2;  // orientation row driven by the linear velocity
  EndEffectorKinematicsTwistConstraint constraint(kinematics_, 6, config);
  constraint.setConstrainYawRateAboutNormal(true);

  const vector_t state = makeState(vector3_t(0.2, 0.1, 0.03), vector3_t(0.1, -0.05, 0.6));
  const vector_t input = makeInput(vector3_t(0.4, -0.2, 0.1), vector3_t(0.3, 0.2, -0.5));
  const vector_t value = constraint.getValue(0.0, state, input, PreComputation());
  const auto approx = constraint.getLinearApproximation(0.0, state, input, PreComputation());
  EXPECT_TRUE(approx.f.isApprox(value, 1e-9)) << "value " << value.transpose() << " vs approximation " << approx.f.transpose();

  matrix_t finiteDifference = matrix_t::Zero(6, kInputDim);
  const scalar_t eps = 1e-6;
  for (size_t i = 0; i < kInputDim; ++i) {
    vector_t plus = input, minus = input;
    plus(i) += eps;
    minus(i) -= eps;
    finiteDifference.col(i) =
        (constraint.getValue(0.0, state, plus, PreComputation()) - constraint.getValue(0.0, state, minus, PreComputation())) / (2.0 * eps);
  }
  EXPECT_TRUE(approx.dfdu.isApprox(finiteDifference, 1e-6)) << "dfdu =\n" << approx.dfdu << "\nfinite differences =\n" << finiteDifference;
}

// ==================== Translation-only behaviour is unchanged ====================

TEST_F(TwistConstraintTest, TranslationOnlyConstraintIgnoresOrientation) {
  EndEffectorKinematicsTwistConstraint constraint(kinematics_, 3, makeFootConfig());
  const vector_t state = makeState(vector3_t(0.2, 0.1, 0.0), vector3_t(0.2, -0.1, 0.4));
  const vector_t input = makeInput(vector3_t::Zero(), vector3_t(0.5, -0.3, 0.9));

  const vector_t value = constraint.getValue(0.0, state, input, PreComputation());
  EXPECT_EQ(value.size(), 3);
  EXPECT_TRUE(value.isZero(kTol)) << "a translation-only constraint should not react to a tilted, rotating foot: " << value.transpose();

  const auto approx = constraint.getLinearApproximation(0.0, state, input, PreComputation());
  EXPECT_TRUE(approx.dfdu.rightCols(3).isZero(kTol)) << "the angular columns should stay empty for a 3-row constraint";
}

TEST_F(TwistConstraintTest, HeightOffsetEntersThroughTheConstantTerm) {
  // ZeroVelocityConstraintCppAd writes the terrain height into b[2]; the foot has to sit at that height.
  EndEffectorKinematicsTwistConstraint::Config config = makeFootConfig();
  const scalar_t referenceHeight = 0.07;
  config.b(2) = -config.Ax(2, 2) * referenceHeight;
  EndEffectorKinematicsTwistConstraint constraint(kinematics_, 6, config);

  const vector_t atHeight = constraint.getValue(0.0, makeState(vector3_t(0.0, 0.0, referenceHeight), vector3_t::Zero()),
                                                makeInput(vector3_t::Zero(), vector3_t::Zero()), PreComputation());
  EXPECT_TRUE(atHeight.isZero(kTol)) << atHeight.transpose();

  const vector_t tooLow = constraint.getValue(0.0, makeState(vector3_t::Zero(), vector3_t::Zero()),
                                              makeInput(vector3_t::Zero(), vector3_t::Zero()), PreComputation());
  EXPECT_NEAR(tooLow(2), -config.Ax(2, 2) * referenceHeight, kTol);
}

// ==================== Non-flat ground ====================

TEST_F(TwistConstraintTest, GroundPlaneNormalIsRespected) {
  EndEffectorKinematicsTwistConstraint constraint(kinematics_, 6, makeFootConfig());
  constraint.setConstrainYawRateAboutNormal(true);
  const vector3_t slopeNormal = vector3_t(0.2, 0.0, 1.0).normalized();
  constraint.setGroundPlaneNormal(slopeNormal);
  EXPECT_TRUE(constraint.getGroundPlaneNormal().isApprox(slopeNormal, kTol));

  // A foot aligned with the slope and at rest satisfies the orientation rows.
  const vector3_t axis = vector3_t::UnitZ().cross(slopeNormal);
  const scalar_t angle = std::asin(axis.norm());
  const vector_t alignedState = makeState(vector3_t::Zero(), axis.normalized() * angle);
  const vector_t value = constraint.getValue(0.0, alignedState, makeInput(vector3_t::Zero(), vector3_t::Zero()), PreComputation());
  EXPECT_TRUE(value.tail(3).isZero(1e-9)) << "a foot aligned with the slope should have no orientation error: "
                                          << value.tail(3).transpose();

  // Spinning about the slope normal is still detected.
  const vector3_t footNormal = AnalyticEndEffectorKinematics::rotationOf(alignedState) * vector3_t::UnitZ();
  const vector_t spinning = constraint.getValue(0.0, alignedState, makeInput(vector3_t::Zero(), 0.4 * footNormal), PreComputation());
  EXPECT_GT(std::abs(spinning(5)), 1e-6);
}

}  // namespace
}  // namespace ocs2::humanoid
