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

#include <memory>
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
 * The stance-foot constraint of the whole-body MPC, g = Ax * [position; orientation error wrt the ground] + Av * twist
 * + Aa * accelerations + b, driven by an affine stand-in for the end-effector dynamics so that its algebra is checked
 * without a robot model. The linearization reads only the diagonal blocks of Ax (its comment said, wrongly, that the
 * orientation gains were ignored altogether); for the block-diagonal Ax the interface builds it is exact.
 */
namespace ocs2::humanoid {
namespace {

using test_support::AffineEndEffectorDynamics;
using test_support::kAffineInputDim;
using test_support::kAffineStateDim;

constexpr scalar_t kQueryTime = 0.0;

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
  return (vector_t(kAffineStateDim) << 0.3, -0.7, 1.2, 0.4).finished();
}

vector_t testInput() {
  return (vector_t(kAffineInputDim) << -0.5, 0.9, 0.2, -1.1, 0.6).finished();
}

/** The constraint Create() makes for `config` on `dynamics`, which has its one end effector; null after a failure. */
std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint> stanceFootConstraint(
    const EndEffectorDynamics<scalar_t>& dynamics, EndEffectorDynamicsAccelerationsConstraint::Config config) {
  absl::StatusOr<std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint>> constraint =
      EndEffectorDynamicsAccelerationsConstraint::Create(dynamics, /*numConstraints=*/6, std::move(config));
  EXPECT_TRUE(constraint.ok()) << constraint.status();
  return constraint.ok() ? *std::move(constraint) : nullptr;
}

/** End-effector dynamics of none and of two end effectors, which the single-end-effector constraints refuse. */
std::vector<std::vector<std::string>> otherThanOneEndEffector() {
  return {{}, {"left_foot", "right_foot"}};
}

/** Central differences of getValue() in the state (`withRespectToState`) or in the input. */
matrix_t valueJacobian(const EndEffectorDynamicsAccelerationsConstraint& constraint,
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
    jacobian.col(i) = (constraint.getValue(kQueryTime, statePlus, inputPlus, PreComputation()) -
                       constraint.getValue(kQueryTime, stateMinus, inputMinus, PreComputation())) /
                      (2.0 * step);
  }
  return jacobian;
}

TEST(EndEffectorDynamicsAccelerationsConstraint, TheLinearizationIsExactForTheStanceFootGains) {
  const AffineEndEffectorDynamics dynamics;
  const std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint> created = stanceFootConstraint(dynamics, stanceFootConfig());
  ASSERT_NE(created, nullptr);
  const EndEffectorDynamicsAccelerationsConstraint& constraint = *created;
  const vector_t state = testState();
  const vector_t input = testInput();

  const vector_t value = constraint.getValue(kQueryTime, state, input, PreComputation());
  const VectorFunctionLinearApproximation approximation = constraint.getLinearApproximation(kQueryTime, state, input, PreComputation());
  EXPECT_TRUE(approximation.f.isApprox(value, /*prec=*/1.0e-12)) << approximation.f.transpose() << "\nvs\n" << value.transpose();
  EXPECT_TRUE(approximation.dfdx.isApprox(valueJacobian(constraint, state, input, /*withRespectToState=*/true), /*prec=*/1.0e-8));
  EXPECT_TRUE(approximation.dfdu.isApprox(valueJacobian(constraint, state, input, /*withRespectToState=*/false), /*prec=*/1.0e-8));

  // The orientation rows of Ax are applied: without the orientation gain the last three rows change by exactly its
  // contribution.
  EndEffectorDynamicsAccelerationsConstraint::Config withoutOrientationGain = stanceFootConfig();
  withoutOrientationGain.Ax.block(3, 3, 3, 3).setZero();
  const std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint> withoutOrientation =
      stanceFootConstraint(dynamics, withoutOrientationGain);
  ASSERT_NE(withoutOrientation, nullptr);
  const VectorFunctionLinearApproximation reduced = withoutOrientation->getLinearApproximation(kQueryTime, state, input, PreComputation());
  const vector3_t orientationTerm = 80.0 * dynamics.orientationErrorWrtPlane(state);
  EXPECT_TRUE((approximation.f - reduced.f).tail<3>().isApprox(orientationTerm, /*prec=*/1.0e-12));
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
  const std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint> createdWithCoupling = stanceFootConstraint(dynamics, coupled);
  const std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint> createdBlockDiagonal =
      stanceFootConstraint(dynamics, stanceFootConfig());
  ASSERT_NE(createdWithCoupling, nullptr);
  ASSERT_NE(createdBlockDiagonal, nullptr);
  const EndEffectorDynamicsAccelerationsConstraint& withCoupling = *createdWithCoupling;
  const EndEffectorDynamicsAccelerationsConstraint& blockDiagonal = *createdBlockDiagonal;

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
  EXPECT_TRUE(valueDifference.isApprox(couplingTerm, /*prec=*/1.0e-12));
  EXPECT_GT(couplingTerm.norm(), 1.0) << "the case under test: a coupling the value sees";
}

// The stance-foot constraint reads the first end effector of its dynamics: dynamics of none or of several are refused by
// Create(), naming what it was given, where the constructor used to throw.
TEST(EndEffectorDynamicsAccelerationsConstraint, CreateRefusesDynamicsOfOtherThanOneEndEffector) {
  for (const std::vector<std::string>& ids : otherThanOneEndEffector()) {
    const AffineEndEffectorDynamics dynamics(ids);
    const absl::StatusOr<std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint>> constraint =
        EndEffectorDynamicsAccelerationsConstraint::Create(dynamics, /*numConstraints=*/6, stanceFootConfig());
    ASSERT_FALSE(constraint.ok()) << ids.size() << " end effectors";
    EXPECT_EQ(constraint.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(absl::StrContains(constraint.status().message(), "only accepts a single end-effector")) << constraint.status();
    for (const std::string& id : ids) {
      EXPECT_TRUE(absl::StrContains(constraint.status().message(), id)) << constraint.status();
    }
  }
}

// The swing foot's normal-motion constraint likewise.
TEST(EndEffectorDynamicsLinearAccConstraint, CreateRefusesDynamicsOfOtherThanOneEndEffector) {
  for (const std::vector<std::string>& ids : otherThanOneEndEffector()) {
    const AffineEndEffectorDynamics dynamics(ids);
    const absl::StatusOr<std::unique_ptr<EndEffectorDynamicsLinearAccConstraint>> constraint =
        EndEffectorDynamicsLinearAccConstraint::Create(dynamics, /*numConstraints=*/1);
    ASSERT_FALSE(constraint.ok()) << ids.size() << " end effectors";
    EXPECT_EQ(constraint.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(absl::StrContains(constraint.status().message(), "only accepts a single end-effector")) << constraint.status();
  }
  const AffineEndEffectorDynamics oneFoot;
  const absl::StatusOr<std::unique_ptr<EndEffectorDynamicsLinearAccConstraint>> constraint =
      EndEffectorDynamicsLinearAccConstraint::Create(oneFoot, /*numConstraints=*/1);
  ASSERT_TRUE(constraint.ok()) << constraint.status();
  EXPECT_EQ((*constraint)->getNumConstraints(kQueryTime), 1u);
}

}  // namespace
}  // namespace ocs2::humanoid
