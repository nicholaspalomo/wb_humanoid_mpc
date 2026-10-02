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

#include <stdexcept>
#include <string>
#include <vector>

#include <ocs2_core/PreComputation.h>

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_wb_mpc/constraint/EndEffectorDynamicsAccelerationsConstraint.h"
#include "humanoid_wb_mpc/end_effector/EndEffectorDynamics.h"

/**
 * The stance-foot constraint of the whole-body MPC, g = Ax * [position; orientation error wrt the ground] + Av * twist
 * + Aa * accelerations + b, driven by an affine stand-in for the end-effector dynamics so that its algebra is checked
 * without a robot model. The linearization reads only the diagonal blocks of Ax (its comment said, wrongly, that the
 * orientation gains were ignored altogether); for the block-diagonal Ax the interface builds it is exact.
 */
namespace ocs2::humanoid {
namespace {

constexpr Eigen::Index kStateDim = 4;
constexpr Eigen::Index kInputDim = 5;
constexpr scalar_t kQueryTime = 0.0;

/**
 * End-effector dynamics whose position, orientation error wrt the plane, twist and accelerations are fixed affine maps
 * of the state and the input, with exact linear approximations. The orientation error ignores the plane normal: only
 * the composition the constraint performs is under test.
 */
class AffineEndEffectorDynamics final : public EndEffectorDynamics<scalar_t> {
 public:
  AffineEndEffectorDynamics() : ids_{"test_foot"} {
    // Fixed, dense and of no particular structure, so that every row and column of every block is exercised.
    positionDx_ = matrix_t::Zero(3, kStateDim);
    positionDx_ << 1.0, 0.2, -0.3, 0.4, -0.5, 1.1, 0.6, -0.2, 0.3, -0.7, 0.9, 0.5;
    positionOffset_ = vector3_t(0.1, -0.2, 0.05);
    orientationDx_ = matrix_t::Zero(3, kStateDim);
    orientationDx_ << 0.3, -0.1, 0.8, 0.2, 0.6, 0.4, -0.5, 0.1, -0.2, 0.7, 0.3, -0.9;
    twistDx_ = matrix_t::Zero(6, kStateDim);
    twistDu_ = matrix_t::Zero(6, kInputDim);
    accelerationDx_ = matrix_t::Zero(6, kStateDim);
    accelerationDu_ = matrix_t::Zero(6, kInputDim);
    for (Eigen::Index row = 0; row < 6; ++row) {
      for (Eigen::Index col = 0; col < kStateDim; ++col) {
        twistDx_(row, col) = 0.1 * static_cast<scalar_t>((row + 2 * col) % 5) - 0.2;
        accelerationDx_(row, col) = 0.05 * static_cast<scalar_t>((3 * row + col) % 7) - 0.15;
      }
      for (Eigen::Index col = 0; col < kInputDim; ++col) {
        twistDu_(row, col) = 0.2 * static_cast<scalar_t>((2 * row + col) % 3) - 0.1;
        accelerationDu_(row, col) = 0.3 * static_cast<scalar_t>((row + 3 * col) % 4) - 0.4;
      }
    }
  }
  AffineEndEffectorDynamics* clone() const override { return new AffineEndEffectorDynamics(*this); }
  const std::vector<std::string>& getIds() const override { return ids_; }

  vector3_t position(const vector_t& state) const { return positionDx_ * state + positionOffset_; }
  vector3_t orientationErrorWrtPlane(const vector_t& state) const { return orientationDx_ * state; }
  vector6_t twist(const vector_t& state, const vector_t& input) const { return twistDx_ * state + twistDu_ * input; }
  vector6_t accelerations(const vector_t& state, const vector_t& input) const { return accelerationDx_ * state + accelerationDu_ * input; }

  std::vector<vector3_t> getPosition(const vector_t& state) const override { return {position(state)}; }
  std::vector<vector3_t> getOrientationErrorWrtPlane(const vector_t& state, const std::vector<vector3_t>& /*planeNormals*/) const override {
    return {orientationErrorWrtPlane(state)};
  }
  std::vector<vector6_t> getTwist(const vector_t& state, const vector_t& input) const override { return {twist(state, input)}; }
  std::vector<vector3_t> getVelocity(const vector_t& state, const vector_t& input) const override {
    return {twist(state, input).head<3>()};
  }
  std::vector<vector3_t> getAngularVelocity(const vector_t& state, const vector_t& input) const override {
    return {twist(state, input).tail<3>()};
  }
  std::vector<vector6_t> getAccelerations(const vector_t& state, const vector_t& input) const override {
    return {accelerations(state, input)};
  }
  std::vector<vector3_t> getLinearAcceleration(const vector_t& state, const vector_t& input) const override {
    return {accelerations(state, input).head<3>()};
  }
  std::vector<vector3_t> getAngularAcceleration(const vector_t& state, const vector_t& input) const override {
    return {accelerations(state, input).tail<3>()};
  }

  std::vector<VectorFunctionLinearApproximation> getPositionLinearApproximation(const vector_t& state) const override {
    VectorFunctionLinearApproximation approximation;
    approximation.f = position(state);
    approximation.dfdx = positionDx_;
    return {approximation};
  }
  std::vector<VectorFunctionLinearApproximation> getOrientationErrorWrtPlaneLinearApproximation(
      const vector_t& state, const std::vector<vector3_t>& /*planeNormals*/) const override {
    VectorFunctionLinearApproximation approximation;
    approximation.f = orientationErrorWrtPlane(state);
    approximation.dfdx = orientationDx_;
    return {approximation};
  }
  std::vector<VectorFunctionLinearApproximation> getTwistLinearApproximation(const vector_t& state, const vector_t& input) const override {
    return {affine(twist(state, input), twistDx_, twistDu_)};
  }
  std::vector<VectorFunctionLinearApproximation> getVelocityLinearApproximation(const vector_t& state,
                                                                                const vector_t& input) const override {
    return {affine(twist(state, input).head<3>(), twistDx_.topRows(3), twistDu_.topRows(3))};
  }
  std::vector<VectorFunctionLinearApproximation> getAngularVelocityLinearApproximation(const vector_t& state,
                                                                                       const vector_t& input) const override {
    return {affine(twist(state, input).tail<3>(), twistDx_.bottomRows(3), twistDu_.bottomRows(3))};
  }
  std::vector<VectorFunctionLinearApproximation> getAccelerationsLinearApproximation(const vector_t& state,
                                                                                     const vector_t& input) const override {
    return {affine(accelerations(state, input), accelerationDx_, accelerationDu_)};
  }
  std::vector<VectorFunctionLinearApproximation> getLinearAccelerationLinearApproximation(const vector_t& state,
                                                                                          const vector_t& input) const override {
    return {affine(accelerations(state, input).head<3>(), accelerationDx_.topRows(3), accelerationDu_.topRows(3))};
  }
  std::vector<VectorFunctionLinearApproximation> getAngularAccelerationLinearApproximation(const vector_t& state,
                                                                                           const vector_t& input) const override {
    return {affine(accelerations(state, input).tail<3>(), accelerationDx_.bottomRows(3), accelerationDu_.bottomRows(3))};
  }

  // The constraint never asks for an orientation, only for its error with respect to a plane.
  std::vector<quaternion_t> getOrientation(const vector_t& /*state*/) const override { throw std::logic_error("not used"); }
  std::vector<vector3_t> getOrientationError(const vector_t& /*state*/,
                                             const std::vector<quaternion_t>& /*referenceOrientations*/) const override {
    throw std::logic_error("not used");
  }
  std::vector<VectorFunctionLinearApproximation> getOrientationErrorLinearApproximation(
      const vector_t& /*state*/, const std::vector<quaternion_t>& /*referenceOrientations*/) const override {
    throw std::logic_error("not used");
  }

 private:
  static VectorFunctionLinearApproximation affine(const vector_t& value, const matrix_t& dfdx, const matrix_t& dfdu) {
    VectorFunctionLinearApproximation approximation;
    approximation.f = value;
    approximation.dfdx = dfdx;
    approximation.dfdu = dfdu;
    return approximation;
  }

  std::vector<std::string> ids_;
  matrix_t positionDx_;
  vector3_t positionOffset_;
  matrix_t orientationDx_;
  matrix_t twistDx_;
  matrix_t twistDu_;
  matrix_t accelerationDx_;
  matrix_t accelerationDu_;
};

/** The gains WBMpcInterface::getStanceFootConstraint builds, with distinct values so that no two rows coincide. */
EndEffectorDynamicsAccelerationsConstraint::Config stanceFootConfig() {
  EndEffectorDynamicsAccelerationsConstraint::Config config;
  config.b = (vector_t(6) << 0.01, -0.02, 0.03, -0.04, 0.05, -0.06).finished();
  config.Ax.setZero(6, 6);
  config.Ax(2, 2) = 100.0;                                     // positionErrorGain_z
  config.Ax.block(3, 3, 3, 3) = 80.0 * matrix3_t::Identity();  // orientationErrorGain
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
  return (vector_t(kStateDim) << 0.3, -0.7, 1.2, 0.4).finished();
}

vector_t testInput() {
  return (vector_t(kInputDim) << -0.5, 0.9, 0.2, -1.1, 0.6).finished();
}

/** Central differences of getValue() in the state (`withRespectToState`) or in the input. */
matrix_t valueJacobian(const EndEffectorDynamicsAccelerationsConstraint& constraint,
                       const vector_t& state,
                       const vector_t& input,
                       bool withRespectToState) {
  const scalar_t step = 1e-6;
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
    jacobian.col(i) = (constraint.getValue(kQueryTime, statePlus, inputPlus, PreComputation()) -
                       constraint.getValue(kQueryTime, stateMinus, inputMinus, PreComputation())) /
                      (2.0 * step);
  }
  return jacobian;
}

TEST(EndEffectorDynamicsAccelerationsConstraint, TheLinearizationIsExactForTheStanceFootGains) {
  const AffineEndEffectorDynamics dynamics;
  const EndEffectorDynamicsAccelerationsConstraint constraint(dynamics, /*numConstraints=*/6, stanceFootConfig());
  const vector_t state = testState();
  const vector_t input = testInput();

  const vector_t value = constraint.getValue(kQueryTime, state, input, PreComputation());
  const VectorFunctionLinearApproximation approximation = constraint.getLinearApproximation(kQueryTime, state, input, PreComputation());
  EXPECT_TRUE(approximation.f.isApprox(value, /*prec=*/1e-12)) << approximation.f.transpose() << "\nvs\n" << value.transpose();
  EXPECT_TRUE(approximation.dfdx.isApprox(valueJacobian(constraint, state, input, /*withRespectToState=*/true), /*prec=*/1e-8));
  EXPECT_TRUE(approximation.dfdu.isApprox(valueJacobian(constraint, state, input, /*withRespectToState=*/false), /*prec=*/1e-8));

  // The orientation rows of Ax are applied: without the orientation gain the last three rows change by exactly its
  // contribution.
  EndEffectorDynamicsAccelerationsConstraint::Config withoutOrientationGain = stanceFootConfig();
  withoutOrientationGain.Ax.block(3, 3, 3, 3).setZero();
  const EndEffectorDynamicsAccelerationsConstraint withoutOrientation(dynamics, /*numConstraints=*/6, withoutOrientationGain);
  const VectorFunctionLinearApproximation reduced = withoutOrientation.getLinearApproximation(kQueryTime, state, input, PreComputation());
  const vector3_t orientationTerm = 80.0 * dynamics.orientationErrorWrtPlane(state);
  EXPECT_TRUE((approximation.f - reduced.f).tail<3>().isApprox(orientationTerm, /*prec=*/1e-12));
  EXPECT_GT(orientationTerm.norm(), 1.0) << "the case under test: an orientation error the gain acts on";
}

TEST(EndEffectorDynamicsAccelerationsConstraint, TheLinearizationReadsOnlyTheDiagonalBlocksOfAx) {
  // getValue() applies all of Ax, the linearization its two diagonal blocks: a coupling block shows up in the value
  // and in nothing the linearization returns.
  const AffineEndEffectorDynamics dynamics;
  const vector_t state = testState();
  const vector_t input = testInput();
  EndEffectorDynamicsAccelerationsConstraint::Config coupled = stanceFootConfig();
  coupled.Ax.block(0, 3, 3, 3) = (matrix3_t() << 1.0, 2.0, 0.5, -1.0, 0.3, 0.7, 0.2, -0.4, 1.5).finished();
  coupled.Ax.block(3, 0, 3, 3) = (matrix3_t() << 0.6, -0.1, 0.9, 0.4, 1.2, -0.3, -0.8, 0.5, 0.1).finished();
  const EndEffectorDynamicsAccelerationsConstraint withCoupling(dynamics, /*numConstraints=*/6, coupled);
  const EndEffectorDynamicsAccelerationsConstraint blockDiagonal(dynamics, /*numConstraints=*/6, stanceFootConfig());

  const VectorFunctionLinearApproximation coupledApproximation =
      withCoupling.getLinearApproximation(kQueryTime, state, input, PreComputation());
  const VectorFunctionLinearApproximation diagonalApproximation =
      blockDiagonal.getLinearApproximation(kQueryTime, state, input, PreComputation());
  EXPECT_TRUE(coupledApproximation.f == diagonalApproximation.f);
  EXPECT_TRUE(coupledApproximation.dfdx == diagonalApproximation.dfdx);
  EXPECT_TRUE(coupledApproximation.dfdu == diagonalApproximation.dfdu);

  vector6_t footPose;
  footPose << dynamics.position(state), dynamics.orientationErrorWrtPlane(state);
  const vector6_t couplingTerm = (coupled.Ax - stanceFootConfig().Ax) * footPose;
  const vector_t valueDifference = withCoupling.getValue(kQueryTime, state, input, PreComputation()) -
                                   blockDiagonal.getValue(kQueryTime, state, input, PreComputation());
  EXPECT_TRUE(valueDifference.isApprox(couplingTerm, /*prec=*/1e-12));
  EXPECT_GT(couplingTerm.norm(), 1.0) << "the case under test: a coupling the value sees";
}

}  // namespace
}  // namespace ocs2::humanoid
