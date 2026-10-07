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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_centroidal_model/CentroidalModelPinocchioMapping.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid {

namespace {

/// Layout of the centroidal state x = [h_norm(6), p_base(3), euler_zyx_base(3), q_joints(nj)]. Written out here
/// rather than shared with the cost, so that a shifted index in the cost is caught instead of copied.
constexpr Eigen::Index kGeneralizedCoordinatesStartIndex = 6;
constexpr Eigen::Index kBasePositionStateIndex = 6;
constexpr Eigen::Index kBaseOrientationStateIndex = 9;
constexpr Eigen::Index kJointStateIndex = 12;

/// Index of the first base orientation coordinate within the Pinocchio q vector, and the base's coordinate count.
constexpr Eigen::Index kBaseOrientationOffset = 3;
constexpr Eigen::Index kGeneralizedBaseDim = 6;

/// Central-difference step, and the agreement it buys on a smooth cost of order one.
constexpr scalar_t kFiniteDifferenceStep = 1.0e-6;
constexpr scalar_t kFiniteDifferenceTolerance = 1.0e-5;

}  // namespace

/**
 * Tests ComAndAcomTrackingCost itself: it is constructed on the same REDUCED Atlas model the MPC builds (via
 * ModelSettings and loadCustomPinocchioInterface) and evaluated through getValue and getQuadraticApproximation.
 *
 * The central check is that getQuadraticApproximation is the Gauss-Newton model of getValue: its value equals
 * getValue, its gradient equals central differences of getValue at a tilted base with bent joints, and its Hessian is
 * J^T Q J with J the finite-difference Jacobians of the CoM position and the aCOM orientation. That pins every piece
 * of the hand-assembled Jacobians - the state index of each block, the absence of an Euler-rate to angular-velocity
 * mapping on the CoM's base orientation columns, the XYZ-to-ZYX reordering of the network's rows - against the value
 * the solver is actually minimizing.
 */
class ComAndAcomTrackingCostTest : public ::testing::Test {
 protected:
  void SetUp() override {
    task_ = loadTaskFile(robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto").value())
                .value();
    urdfFile_ = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf").value();

    modelSettingsPtr_ =
        std::make_unique<ModelSettings>(ModelSettings::Create(task_, urdfFile_, "testComAndAcomTrackingCost", /*verbose=*/false).value());
    pinocchioInterfacePtr_ =
        std::make_unique<PinocchioInterface>(loadCustomPinocchioInterface(task_, urdfFile_, *modelSettingsPtr_).value());

    const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
    nJoints_ = model.nq - kGeneralizedBaseDim;
    info_ = makeInfo(model);

    // An independent evaluator of the same network, to compute expectations with.
    absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> acom =
        AngularCenterOfMass::Create(modelSettingsPtr_->robotName, modelSettingsPtr_->mpcModelJointNames);
    ASSERT_TRUE(acom.ok()) << acom.status();
    acomPtr_ = *std::move(acom);
  }

  static CentroidalModelInfo makeInfo(const pinocchio::Model& model) {
    CentroidalModelInfo info;
    info.generalizedCoordinatesNum = static_cast<size_t>(model.nq);
    info.actuatedDofNum = static_cast<size_t>(model.nq - kGeneralizedBaseDim);
    info.stateDim = static_cast<size_t>(kGeneralizedCoordinatesStartIndex + model.nq);
    info.inputDim = info.actuatedDofNum;
    info.robotMass = pinocchio::computeTotalMass(model);
    return info;
  }

  /** Creates the cost on the fixture's model, failing the test if Create refuses it. */
  std::unique_ptr<ComAndAcomTrackingCost> makeCost(const matrix_t& Q_com, const matrix_t& Q_acom) const {
    absl::StatusOr<std::unique_ptr<ComAndAcomTrackingCost>> cost =
        ComAndAcomTrackingCost::Create(Q_com, Q_acom, *pinocchioInterfacePtr_, info_, modelSettingsPtr_->robotName);
    EXPECT_TRUE(cost.ok()) << cost.status();
    return cost.ok() ? *std::move(cost) : nullptr;
  }

  /** Distinct, non-diagonal-friendly weights, so a swapped row or column changes the value. */
  static matrix_t weights(scalar_t a, scalar_t b, scalar_t c) {
    matrix_t Q = matrix_t::Zero(3, 3);
    Q.diagonal() << a, b, c;
    return Q;
  }

  /** A state with the base off level (so an Euler-rate mapping would matter) and the joints bent by `jointOffset`. */
  vector_t makeState(const vector3_t& eulerZyx, scalar_t jointOffset) const {
    vector_t state = vector_t::Zero(info_.stateDim);
    state.head(kGeneralizedCoordinatesStartIndex) << 0.1, -0.2, 0.05, 0.3, -0.1, 0.2;  // must not affect the cost
    state.segment<3>(kBasePositionStateIndex) << 0.05, -0.03, 0.82;
    state.segment<3>(kBaseOrientationStateIndex) = eulerZyx;
    for (Eigen::Index joint = 0; joint < nJoints_; ++joint) {
      state(kJointStateIndex + joint) = jointOffset + 0.01 * static_cast<scalar_t>(joint % 7);
    }
    return state;
  }

  TargetTrajectories reference(const vector_t& stateRef) const {
    return TargetTrajectories({0.0}, {stateRef}, {vector_t::Zero(info_.inputDim)});
  }

  vector_t generalizedCoordinates(const vector_t& state) const { return state.tail(info_.generalizedCoordinatesNum); }

  /** The CoM of `state`, from Pinocchio on a private data object. */
  vector3_t centerOfMass(const vector_t& state) const {
    const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
    pinocchio::Data data(model);
    return pinocchio::centerOfMass(model, data, generalizedCoordinates(state));
  }

  /** theta_aCOM of `state`, in the state's ZYX order, from the independent evaluator. */
  vector3_t acomOrientation(const vector_t& state) const { return acomPtr_->computeAcomOrientation(generalizedCoordinates(state)); }

  /** Central-difference Jacobian of a 3-vector function of the state. */
  template <typename Function>
  matrix_t finiteDifferenceJacobian(const vector_t& state, const Function& function) const {
    matrix_t jacobian = matrix_t::Zero(3, state.size());
    for (Eigen::Index i = 0; i < state.size(); ++i) {
      vector_t plus = state;
      vector_t minus = state;
      plus(i) += kFiniteDifferenceStep;
      minus(i) -= kFiniteDifferenceStep;
      jacobian.col(i) = (function(plus) - function(minus)) / (2.0 * kFiniteDifferenceStep);
    }
    return jacobian;
  }

  mpc_config::TaskFile task_;
  std::string urdfFile_;
  std::unique_ptr<ModelSettings> modelSettingsPtr_;
  std::unique_ptr<PinocchioInterface> pinocchioInterfacePtr_;
  std::unique_ptr<AngularCenterOfMass> acomPtr_;
  CentroidalModelInfo info_;
  Eigen::Index nJoints_ = 0;
  PreComputation preComputation_;
};

/**
 * The trained weights must be indexed by exactly the joints the MPC optimizes over, and the cost is built on the
 * shipped model without complaint.
 */
TEST_F(ComAndAcomTrackingCostTest, CreateAcceptsTheShippedModel) {
  EXPECT_EQ(acomPtr_->getInputDim(), static_cast<size_t>(nJoints_));
  EXPECT_EQ(modelSettingsPtr_->mpcModelJointNames.size(), static_cast<size_t>(nJoints_));
  EXPECT_NE(makeCost(weights(1.0, 2.0, 3.0), weights(4.0, 5.0, 6.0)), nullptr);
}

/** The quadratic approximation's value is the cost's value, not a separately assembled number. */
TEST_F(ComAndAcomTrackingCostTest, QuadraticApproximationValueIsGetValue) {
  const std::unique_ptr<ComAndAcomTrackingCost> cost = makeCost(weights(30.0, 20.0, 40.0), weights(25.0, 15.0, 10.0));
  ASSERT_NE(cost, nullptr);
  const vector_t state = makeState(vector3_t(0.30, 0.20, -0.15), /*jointOffset=*/0.2);
  const TargetTrajectories target = reference(makeState(vector3_t(0.10, -0.05, 0.05), /*jointOffset=*/-0.1));

  const scalar_t value = cost->getValue(/*time=*/0.0, state, target, preComputation_);
  const ScalarFunctionQuadraticApproximation approximation = cost->getQuadraticApproximation(/*time=*/0.0, state, target, preComputation_);
  EXPECT_GT(value, 0.0) << "the state and the reference differ, so the cost cannot be zero";
  EXPECT_NEAR(approximation.f, value, 1.0e-12 * std::max(1.0, std::abs(value)));
}

/**
 * The gradient is exactly the derivative of getValue. The Gauss-Newton gradient J^T Q e of a least-squares cost is
 * exact, so this holds to finite-difference precision, and it holds at a tilted base with bent joints, where an
 * Euler-rate to angular-velocity mapping chained onto the CoM Jacobian, a block placed at the wrong state index, a
 * flipped sign or a dropped XYZ-to-ZYX reordering would each show up.
 */
TEST_F(ComAndAcomTrackingCostTest, GradientMatchesFiniteDifferencesOfGetValue) {
  const std::unique_ptr<ComAndAcomTrackingCost> cost = makeCost(weights(30.0, 20.0, 40.0), weights(25.0, 15.0, 10.0));
  ASSERT_NE(cost, nullptr);
  const vector_t state = makeState(vector3_t(0.30, 0.20, -0.15), /*jointOffset=*/0.2);
  const TargetTrajectories target = reference(makeState(vector3_t(0.10, -0.05, 0.05), /*jointOffset=*/-0.1));

  const vector_t gradient = cost->getQuadraticApproximation(/*time=*/0.0, state, target, preComputation_).dfdx;
  ASSERT_EQ(gradient.size(), state.size());
  vector_t finiteDifference = vector_t::Zero(state.size());
  for (Eigen::Index i = 0; i < state.size(); ++i) {
    vector_t plus = state;
    vector_t minus = state;
    plus(i) += kFiniteDifferenceStep;
    minus(i) -= kFiniteDifferenceStep;
    finiteDifference(i) =
        (cost->getValue(/*time=*/0.0, plus, target, preComputation_) - cost->getValue(/*time=*/0.0, minus, target, preComputation_)) /
        (2.0 * kFiniteDifferenceStep);
  }
  ASSERT_GT(finiteDifference.norm(), 1.0e-3) << "a vanishing gradient would make this comparison vacuous";
  EXPECT_TRUE(gradient.head(kGeneralizedCoordinatesStartIndex).isZero(0.0)) << "the momentum does not enter this cost";
  for (Eigen::Index i = 0; i < state.size(); ++i) {
    EXPECT_NEAR(gradient(i), finiteDifference(i), kFiniteDifferenceTolerance * std::max(1.0, finiteDifference.cwiseAbs().maxCoeff()))
        << "state index " << i;
  }
}

/** The Hessian is the Gauss-Newton product J^T Q J, with each J the true Jacobian of its residual. */
TEST_F(ComAndAcomTrackingCostTest, HessianIsTheGaussNewtonProductOfTheTrueJacobians) {
  const matrix_t Q_com = weights(30.0, 20.0, 40.0);
  const matrix_t Q_acom = weights(25.0, 15.0, 10.0);
  const std::unique_ptr<ComAndAcomTrackingCost> cost = makeCost(Q_com, Q_acom);
  ASSERT_NE(cost, nullptr);
  const vector_t state = makeState(vector3_t(0.30, 0.20, -0.15), /*jointOffset=*/0.2);
  const TargetTrajectories target = reference(makeState(vector3_t(0.10, -0.05, 0.05), /*jointOffset=*/-0.1));

  const matrix_t J_com = finiteDifferenceJacobian(state, [this](const vector_t& x) -> vector3_t { return centerOfMass(x); });
  const matrix_t J_acom = finiteDifferenceJacobian(state, [this](const vector_t& x) -> vector3_t { return acomOrientation(x); });
  const matrix_t expected = J_com.transpose() * Q_com * J_com + J_acom.transpose() * Q_acom * J_acom;

  const matrix_t hessian = cost->getQuadraticApproximation(/*time=*/0.0, state, target, preComputation_).dfdxx;
  ASSERT_EQ(hessian.rows(), state.size());
  ASSERT_EQ(hessian.cols(), state.size());
  EXPECT_LT((hessian - expected).cwiseAbs().maxCoeff(), kFiniteDifferenceTolerance * std::max(1.0, expected.cwiseAbs().maxCoeff()))
      << "max abs error " << (hessian - expected).cwiseAbs().maxCoeff();
}

/**
 * The yaw error is wrapped: a heading of +3.1 rad tracking a reference of -3.1 rad is 0.083 rad off, not 6.2. The
 * joints are equal, so the network's offsets cancel and the error is the base yaw difference alone.
 */
TEST_F(ComAndAcomTrackingCostTest, YawErrorIsWrappedAcrossPi) {
  const std::unique_ptr<ComAndAcomTrackingCost> cost = makeCost(matrix_t::Zero(3, 3), weights(1.0, 0.0, 0.0));
  ASSERT_NE(cost, nullptr);
  const vector_t state = makeState(vector3_t(3.1, 0.0, 0.0), /*jointOffset=*/0.2);
  const scalar_t shortWay = 2.0 * M_PI - 6.2;

  const scalar_t across =
      cost->getValue(/*time=*/0.0, state, reference(makeState(vector3_t(-3.1, 0.0, 0.0), /*jointOffset=*/0.2)), preComputation_);
  const scalar_t direct =
      cost->getValue(/*time=*/0.0, state, reference(makeState(vector3_t(3.1 + shortWay, 0.0, 0.0), /*jointOffset=*/0.2)), preComputation_);
  EXPECT_NEAR(across, 0.5 * shortWay * shortWay, 1.0e-9);
  EXPECT_NEAR(across, direct, 1.0e-9);
}

/**
 * Row 0 of Q_acom weights YAW, which the network emits as its third (z) component. With only that row weighted and the
 * base identical in state and reference, the cost is half the square of the change in the network's z output - read
 * straight from the evaluator in its native XYZ order, so a cost that forgot to reorder would weight roll instead.
 */
TEST_F(ComAndAcomTrackingCostTest, EachQacomRowWeightsItsOwnEulerAngle) {
  const vector_t state = makeState(vector3_t(0.2, 0.1, -0.1), /*jointOffset=*/0.25);
  const vector_t stateRef = makeState(vector3_t(0.2, 0.1, -0.1), /*jointOffset=*/-0.15);
  const vector3_t offsetChangeXyz =
      acomPtr_->computeJointOrientationOffset(state.tail(nJoints_)) - acomPtr_->computeJointOrientationOffset(stateRef.tail(nJoints_));
  ASSERT_GT(std::abs(offsetChangeXyz.z() - offsetChangeXyz.x()), 1.0e-4) << "roll and yaw must differ for this to tell them apart";

  const std::unique_ptr<ComAndAcomTrackingCost> yawOnly = makeCost(matrix_t::Zero(3, 3), weights(1.0, 0.0, 0.0));
  const std::unique_ptr<ComAndAcomTrackingCost> rollOnly = makeCost(matrix_t::Zero(3, 3), weights(0.0, 0.0, 1.0));
  ASSERT_NE(yawOnly, nullptr);
  ASSERT_NE(rollOnly, nullptr);
  EXPECT_NEAR(yawOnly->getValue(/*time=*/0.0, state, reference(stateRef), preComputation_), 0.5 * offsetChangeXyz.z() * offsetChangeXyz.z(),
              1.0e-12);
  EXPECT_NEAR(rollOnly->getValue(/*time=*/0.0, state, reference(stateRef), preComputation_),
              0.5 * offsetChangeXyz.x() * offsetChangeXyz.x(), 1.0e-12);

  // And a base roll the yaw row must ignore.
  const vector_t rolled = makeState(vector3_t(0.2, 0.1, 0.4), /*jointOffset=*/-0.15);
  EXPECT_NEAR(yawOnly->getValue(/*time=*/0.0, rolled, reference(stateRef), preComputation_), 0.0, 1.0e-15);
}

/** clone() is an independent copy that evaluates identically. */
TEST_F(ComAndAcomTrackingCostTest, CloneEvaluatesIdenticallyAndIndependently) {
  const std::unique_ptr<ComAndAcomTrackingCost> cost = makeCost(weights(30.0, 20.0, 40.0), weights(25.0, 15.0, 10.0));
  ASSERT_NE(cost, nullptr);
  const std::unique_ptr<ComAndAcomTrackingCost> clone(cost->clone());
  const vector_t state = makeState(vector3_t(0.30, 0.20, -0.15), /*jointOffset=*/0.2);
  const TargetTrajectories target = reference(makeState(vector3_t(0.10, -0.05, 0.05), /*jointOffset=*/-0.1));

  const ScalarFunctionQuadraticApproximation original = cost->getQuadraticApproximation(/*time=*/0.0, state, target, preComputation_);
  const ScalarFunctionQuadraticApproximation cloned = clone->getQuadraticApproximation(/*time=*/0.0, state, target, preComputation_);
  EXPECT_EQ(cloned.f, original.f);
  EXPECT_TRUE(cloned.dfdx.isApprox(original.dfdx, 0.0));
  EXPECT_TRUE(cloned.dfdxx.isApprox(original.dfdxx, 0.0));

  ASSERT_TRUE(clone->setWeights(weights(1.0, 1.0, 1.0), weights(1.0, 1.0, 1.0)).ok());
  EXPECT_EQ(cost->getValue(/*time=*/0.0, state, target, preComputation_), original.f) << "re-weighting the clone changed the original";
  EXPECT_NE(clone->getValue(/*time=*/0.0, state, target, preComputation_), original.f);
}

TEST_F(ComAndAcomTrackingCostTest, WeightsMustBe3x3) {
  const absl::StatusOr<std::unique_ptr<ComAndAcomTrackingCost>> cost = ComAndAcomTrackingCost::Create(
      matrix_t::Identity(2, 2), matrix_t::Identity(3, 3), *pinocchioInterfacePtr_, info_, modelSettingsPtr_->robotName);
  ASSERT_FALSE(cost.ok());
  EXPECT_EQ(cost.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(cost.status().message(), "Q_com")) << cost.status();

  const std::unique_ptr<ComAndAcomTrackingCost> valid = makeCost(matrix_t::Identity(3, 3), matrix_t::Identity(3, 3));
  ASSERT_NE(valid, nullptr);
  const scalar_t weightBefore = valid->getQAcom()(0, 0);
  const absl::Status refused = valid->setWeights(matrix_t::Identity(3, 3), 2.0 * matrix_t::Identity(4, 4));
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.message(), "Q_acom")) << refused;
  EXPECT_EQ(valid->getQAcom().rows(), 3) << "a refused update changed the weights";
  EXPECT_EQ(valid->getQAcom()(0, 0), weightBefore);
  EXPECT_TRUE(ComAndAcomTrackingCost::validateWeights(matrix_t::Identity(3, 3), matrix_t::Identity(3, 3)).ok());
}

TEST_F(ComAndAcomTrackingCostTest, CreateRejectsAnInfoThatDoesNotDescribeTheModel) {
  CentroidalModelInfo wrong = info_;
  wrong.actuatedDofNum = info_.actuatedDofNum - 1;
  const absl::StatusOr<std::unique_ptr<ComAndAcomTrackingCost>> cost = ComAndAcomTrackingCost::Create(
      matrix_t::Identity(3, 3), matrix_t::Identity(3, 3), *pinocchioInterfacePtr_, wrong, modelSettingsPtr_->robotName);
  ASSERT_FALSE(cost.ok());
  EXPECT_EQ(cost.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(cost.status().message(), "actuatedDofNum")) << cost.status();
}

TEST_F(ComAndAcomTrackingCostTest, CreateRejectsARobotWithoutANetwork) {
  const absl::StatusOr<std::unique_ptr<ComAndAcomTrackingCost>> cost =
      ComAndAcomTrackingCost::Create(matrix_t::Identity(3, 3), matrix_t::Identity(3, 3), *pinocchioInterfacePtr_, info_, "r1");
  ASSERT_FALSE(cost.ok());
  EXPECT_EQ(cost.status().code(), absl::StatusCode::kNotFound);
  EXPECT_TRUE(absl::StrContains(cost.status().message(), "com_and_acom_tracking_cost")) << cost.status();
}

/**
 * The runtime joint check on the model the cost is actually built on. The task file is the shipped one with a
 * different fourth joint fixed - neck_ry instead of r_arm_wrx - so the MPC model still has 24 joints, and a check that
 * only counted them would pass. The network was trained with the head active, so from index 7 on every joint it is
 * fed is a different one; Create has to refuse, and name the first joint that differs.
 */
TEST_F(ComAndAcomTrackingCostTest, CreateRejectsAModelWithADifferentFixedJointSet) {
  mpc_config::TaskFile drifted = task_;
  std::vector<std::string>& fixedJoints = drifted.model_settings.fixed_joint_names;
  ASSERT_EQ(std::count(fixedJoints.begin(), fixedJoints.end(), "r_arm_wrx"), 1)
      << "the shipped task file no longer fixes r_arm_wrx once; update this test";
  *std::find(fixedJoints.begin(), fixedJoints.end(), "r_arm_wrx") = "neck_ry";

  const ModelSettings driftedSettings = ModelSettings::Create(drifted, urdfFile_, "testComAndAcomTrackingCost", /*verbose=*/false).value();
  ASSERT_EQ(driftedSettings.mpcModelJointNames.size(), static_cast<size_t>(nJoints_)) << "the joint COUNT must not change";
  const PinocchioInterface driftedModel = loadCustomPinocchioInterface(drifted, urdfFile_, driftedSettings).value();
  const CentroidalModelInfo driftedInfo = makeInfo(driftedModel.getModel());

  const absl::StatusOr<std::unique_ptr<ComAndAcomTrackingCost>> cost =
      ComAndAcomTrackingCost::Create(matrix_t::Identity(3, 3), matrix_t::Identity(3, 3), driftedModel, driftedInfo, "atlas");
  ASSERT_FALSE(cost.ok()) << "a model with the head fixed and a wrist active was accepted";
  EXPECT_EQ(cost.status().code(), absl::StatusCode::kFailedPrecondition);
  const std::string message(cost.status().message());
  EXPECT_TRUE(absl::StrContains(message, "model_settings.robot_name 'atlas'")) << message;
  EXPECT_TRUE(absl::StrContains(message, "joint 7 is 'neck_ry'")) << message;
  EXPECT_TRUE(absl::StrContains(message, "'r_arm_shz' in the MPC model")) << message;
  EXPECT_TRUE(absl::StrContains(message, "model_settings.fixed_joint_names")) << message;
}

/**
 * The base-pose weights the factory and the live updater drop while this cost runs are exactly the state block whose
 * error this cost re-expresses - p_base and the base Euler angles - and nothing else.
 */
TEST_F(ComAndAcomTrackingCostTest, ZeroBasePoseWeightsClearsExactlyTheBasePoseBlock) {
  const Eigen::Index n = static_cast<Eigen::Index>(info_.stateDim);
  matrix_t Q = matrix_t::Zero(n, n);
  for (Eigen::Index row = 0; row < n; ++row) {
    for (Eigen::Index col = 0; col < n; ++col) {
      Q(row, col) = 1.0 + static_cast<scalar_t>(row * n + col);
    }
  }
  const matrix_t original = Q;
  ComAndAcomTrackingCost::zeroBasePoseWeights(Q);
  for (Eigen::Index row = 0; row < n; ++row) {
    for (Eigen::Index col = 0; col < n; ++col) {
      const bool inBasePose =
          row >= kBasePositionStateIndex && row < kJointStateIndex && col >= kBasePositionStateIndex && col < kJointStateIndex;
      EXPECT_EQ(Q(row, col), inBasePose ? 0.0 : original(row, col)) << "(" << row << ", " << col << ")";
    }
  }
  // Idempotent: a live reload may apply it to an already-zeroed matrix.
  const matrix_t once = Q;
  ComAndAcomTrackingCost::zeroBasePoseWeights(Q);
  EXPECT_EQ(Q, once);
}

/**
 * The aCOM Jacobian rows must follow the centroidal state's ZYX Euler ordering, while the network natively emits XYZ.
 * Getting this backwards silently swaps the roll and yaw tracking weights.
 */
TEST_F(ComAndAcomTrackingCostTest, AcomJacobianRowsAreInZyxOrder) {
  const vector_t q = generalizedCoordinates(makeState(vector3_t(0.30, 0.20, -0.15), /*jointOffset=*/0.2));
  const vector_t qJoints = q.tail(nJoints_);

  const matrix_t J_xyz = acomPtr_->computeJointOffsetJacobian(qJoints);
  const matrix_t J_zyx = acomJacobianXyzToZyx(J_xyz);

  EXPECT_TRUE(J_zyx.row(0).isApprox(J_xyz.row(2)));
  EXPECT_TRUE(J_zyx.row(1).isApprox(J_xyz.row(1)));
  EXPECT_TRUE(J_zyx.row(2).isApprox(J_xyz.row(0)));

  // computeAcomJacobian must apply the same reordering to its joint block.
  const matrix_t J_acom = acomPtr_->computeAcomJacobian(q);
  ASSERT_EQ(J_acom.cols(), q.size());
  EXPECT_TRUE(J_acom.leftCols<3>().isZero(1.0e-15));
  EXPECT_TRUE((J_acom.block<3, 3>(0, kBaseOrientationOffset).isApprox(matrix_t::Identity(3, 3), 1.0e-15)));
  EXPECT_TRUE(J_acom.rightCols(nJoints_).isApprox(J_zyx));
}

/**
 * The whole-body aCOM orientation must reduce to the base Euler triple plus the reordered joint offset, in the state's
 * ZYX convention.
 */
TEST_F(ComAndAcomTrackingCostTest, AcomOrientationIsBaseEulerPlusReorderedOffset) {
  const vector_t q = generalizedCoordinates(makeState(vector3_t(0.30, 0.20, -0.15), /*jointOffset=*/0.2));
  const vector3_t expected = q.segment<3>(kBaseOrientationOffset) + acomXyzToZyx(acomPtr_->computeJointOrientationOffset(q.tail(nJoints_)));
  EXPECT_TRUE(acomPtr_->computeAcomOrientation(q).isApprox(expected, 1.0e-12));
}

/**
 * The reduced model must place the CoM somewhere physically sensible, which catches a mis-built or mis-scaled Pinocchio
 * model before it reaches the cost.
 */
TEST_F(ComAndAcomTrackingCostTest, CenterOfMassIsPhysicallyPlausible) {
  vector_t state = vector_t::Zero(info_.stateDim);
  state(kGeneralizedCoordinatesStartIndex + 2) = 0.8;

  CentroidalModelPinocchioMapping mapping(info_);
  const vector_t q = mapping.getPinocchioJointPosition(state);
  ASSERT_EQ(static_cast<size_t>(q.size()), info_.generalizedCoordinatesNum);

  const vector3_t com = centerOfMass(state);
  EXPECT_NEAR(com[0], 0.0, 0.5);
  EXPECT_NEAR(com[1], 0.0, 0.5);
  EXPECT_GT(com[2], 0.3);
  EXPECT_GT(info_.robotMass, 0.0);
}

}  // namespace ocs2::humanoid
