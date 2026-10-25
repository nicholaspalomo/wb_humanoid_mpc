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
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_centroidal_model/CentroidalModelPinocchioMapping.h"
#include "ocs2_centroidal_model/FactoryFunctions.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StateManifoldFactory.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/QuaternionLayoutRobotModel.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid {

namespace {

/// Layout of the centroidal state x = [h_norm(6), p_base(3), xi_base(4) = (x,
/// y, z, w), q_joints(nj)], and of its tangent and tuning rows [h_norm(6),
/// p_base(3), base rotation(3), q_joints(nj)]. Written out here rather than
/// shared with the cost, so that a shifted index in the cost is caught instead
/// of copied.
constexpr Eigen::Index kGeneralizedCoordinatesStartIndex = 6;
constexpr Eigen::Index kBasePositionStateIndex = 6;
constexpr Eigen::Index kBaseOrientationStateIndex = 9;
constexpr Eigen::Index kJointStateIndex = 13;
constexpr Eigen::Index kBasePositionTuningRow = 6;
constexpr Eigen::Index kJointTuningRow = 12;

/// Central-difference step, and the agreement it buys on a smooth cost of order
/// one.
constexpr scalar_t kFiniteDifferenceStep = 1.0e-6;
constexpr scalar_t kFiniteDifferenceTolerance = 1.0e-5;

}  // namespace

/**
 * Tests ComAndAcomTrackingCost itself: it is constructed on the same REDUCED
 * Atlas model the MPC builds (via ModelSettings and
 * loadCustomPinocchioInterface) and evaluated through getValue and
 * getQuadraticApproximation.
 *
 * The central check is that getQuadraticApproximation is the Gauss-Newton model
 * of getValue: its value equals getValue, its gradient equals central
 * differences of getValue at a tilted base with bent joints, and its Hessian is
 * J^T Q J with J the finite-difference Jacobians of the CoM position and of the
 * ACoM heading-tilt residual. That pins every piece of the assembled Jacobians
 * - the state index of each block, the robot model's tangent maps, the adjoint
 * R(Delta_theta)^T and the right Jacobian J_r(Delta_theta) of the
 * multiplicative ACoM - against the value the solver is actually minimizing.
 * The expectations recompute the ACoM attitude from the state's slots with an
 * independent evaluator of the same network, so a cost that read the wrong
 * slots or composed in the wrong order fails here.
 *
 * The state is the quaternion layout of the centroidal MPC
 * (test/support/QuaternionLayoutRobotModel.h): the base orientation is four
 * slots xi, read through xi / |xi|. Derivatives with respect to the stored
 * slots (ambient) are checked against finite differences of the slots, and
 * their pull-back to the tangent (the solver's coordinates) against finite
 * differences along the state manifold's retraction x (+) dx; the cost is
 * invariant under xi -> -xi.
 */
class ComAndAcomTrackingCostTest : public ::testing::Test {
 protected:
  void SetUp() override {
    HUMANOID_ASSERT_OK_AND_ASSIGN(
        const std::string taskFile,
        robot::resolveResourcePath(
            "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/"
            "task.textproto"));
    HUMANOID_ASSERT_OK_AND_ASSIGN(task_, loadTaskFile(taskFile));
    HUMANOID_ASSERT_OK_AND_ASSIGN(
        urdfFile_,
        robot::resolveResourcePath(
            "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf"));

    modelSettingsPtr_ = std::make_unique<ModelSettings>(checkedValue(
        ModelSettings::Create(task_, urdfFile_, "testComAndAcomTrackingCost",
                              /*verbose=*/false)));
    pinocchioInterfacePtr_ = std::make_unique<PinocchioInterface>(checkedValue(
        loadCustomPinocchioInterface(task_, urdfFile_, *modelSettingsPtr_)));

    info_ = makeInfo(*pinocchioInterfacePtr_, *modelSettingsPtr_);
    nJoints_ = static_cast<Eigen::Index>(info_.actuatedDofNum);
    // The robot model the cost reads the state and interpolates its reference
    // with: the centroidal quaternion layout of the same settings, and its
    // state manifold.
    modelPtr_ = std::make_unique<test::QuaternionLayoutRobotModel>(
        *modelSettingsPtr_, StateInputLayout::Mpc::kCentroidal);
    ASSERT_EQ(modelPtr_->getStateDim(), info_.stateDim);
    ASSERT_EQ(modelPtr_->getStateTangentDim(), info_.getStateTangentDim());
    absl::StatusOr<std::shared_ptr<const StateManifold>> manifold =
        stateManifoldOf(*modelPtr_);
    ASSERT_TRUE(manifold.ok()) << manifold.status();
    manifoldPtr_ = *std::move(manifold);

    // An independent evaluator of the same network, to compute expectations
    // with.
    absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> acom =
        AngularCenterOfMass::Create(modelSettingsPtr_->robotName,
                                    modelSettingsPtr_->mpcModelJointNames);
    ASSERT_TRUE(acom.ok()) << acom.status();
    acomPtr_ = *std::move(acom);
  }

  /** The full centroidal model info of `pinocchioInterface`, as the MPC builds
   * it, at zero nominal joint angles. */
  static CentroidalModelInfo makeInfo(
      const PinocchioInterface& pinocchioInterface,
      const ModelSettings& settings) {
    return centroidal_model::createCentroidalModelInfo(
        pinocchioInterface, CentroidalModelType::FullCentroidalDynamics,
        vector_t::Zero(
            static_cast<Eigen::Index>(settings.mpcModelJointNames.size())),
        settings.contactNames3DoF, settings.contactNames6DoF);
  }

  /** Creates the cost on the fixture's model, failing the test if Create
   * refuses it. */
  std::unique_ptr<ComAndAcomTrackingCost> makeCost(
      const matrix_t& Q_com, const matrix_t& Q_acom) const {
    absl::StatusOr<std::unique_ptr<ComAndAcomTrackingCost>> cost =
        ComAndAcomTrackingCost::Create(Q_com, Q_acom, *pinocchioInterfacePtr_,
                                       info_, modelPtr_.get(),
                                       modelSettingsPtr_->robotName);
    EXPECT_TRUE(cost.ok()) << cost.status();
    return cost.ok() ? *std::move(cost) : nullptr;
  }

  /** Distinct, non-diagonal-friendly weights, so a swapped row or column
   * changes the value. */
  static matrix_t weights(scalar_t a, scalar_t b, scalar_t c) {
    matrix_t Q = matrix_t::Zero(3, 3);
    Q.diagonal() << a, b, c;
    return Q;
  }

  /**
   * A state with the base off level at the ZYX Euler angles `eulerZyx` (so a
   * tangent map left out would matter), stored as the quaternion scaled by
   * `quaternionScale`, and the joints bent by `jointOffset`.
   */
  vector_t makeState(const vector3_t& eulerZyx,
                     scalar_t jointOffset,
                     scalar_t quaternionScale = 1.0) const {
    vector_t state = vector_t::Zero(info_.stateDim);
    state.head(kGeneralizedCoordinatesStartIndex) << 0.1, -0.2, 0.05, 0.3, -0.1,
        0.2;  // must not affect the cost
    state.segment<3>(kBasePositionStateIndex) << 0.05, -0.03, 0.82;
    state.segment<4>(kBaseOrientationStateIndex) =
        quaternionScale * quaternionFromEulerZyx(eulerZyx);
    for (Eigen::Index joint = 0; joint < nJoints_; ++joint) {
      state(kJointStateIndex + joint) =
          jointOffset + 0.01 * static_cast<scalar_t>(joint % 7);
    }
    return state;
  }

  TargetTrajectories reference(const vector_t& stateRef) const {
    return TargetTrajectories({0.0}, {stateRef},
                              {vector_t::Zero(info_.inputDim)});
  }

  /** Pinocchio's q = [p, xi / |xi|, q_j] of `state`: its last nq slots, the
   * quaternion normalized. */
  vector_t generalizedCoordinates(const vector_t& state) const {
    vector_t q =
        state.tail(static_cast<Eigen::Index>(info_.getConfigurationDim()));
    q.segment<4>(3).normalize();
    return q;
  }

  /** The CoM of `state`, from Pinocchio on a private data object. */
  vector3_t centerOfMass(const vector_t& state) const {
    const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
    pinocchio::Data data(model);
    return pinocchio::centerOfMass(model, data, generalizedCoordinates(state));
  }

  /**
   * The ACoM attitude xi_a = xi_hat (x) Exp(Delta_theta(q_j)) of `state`, from
   * its quaternion slots (normalized) and its joint slots, through the
   * independent evaluator's raw network output.
   */
  vector4_t acomAttitude(const vector_t& state) const {
    const vector4_t base =
        vector4_t(state.segment<4>(kBaseOrientationStateIndex)).normalized();
    return quaternionProduct(
        base, quaternionExp(acomPtr_->computeJointOrientationOffset(
                  state.tail(nJoints_))));
  }

  /** The ACoM heading-tilt residual e_HT(xi_a(state), xi_a(stateRef)), rows
   * (yaw, pitch, roll). */
  vector3_t acomResidual(const vector_t& state,
                         const vector_t& stateRef) const {
    return headingTiltResidual(acomAttitude(state), acomAttitude(stateRef));
  }

  /** Central-difference Jacobian of a 3-vector function of the state. */
  template <typename Function>
  matrix_t finiteDifferenceJacobian(const vector_t& state,
                                    const Function& function) const {
    matrix_t jacobian = matrix_t::Zero(3, state.size());
    for (Eigen::Index i = 0; i < state.size(); ++i) {
      vector_t plus = state;
      vector_t minus = state;
      plus(i) += kFiniteDifferenceStep;
      minus(i) -= kFiniteDifferenceStep;
      jacobian.col(i) =
          (function(plus) - function(minus)) / (2.0 * kFiniteDifferenceStep);
    }
    return jacobian;
  }

  /** Central-difference Jacobian of a 3-vector function of the state along the
   * retraction: tangent columns. */
  template <typename Function>
  matrix_t tangentFiniteDifferenceJacobian(const vector_t& state,
                                           const Function& function) const {
    const Eigen::Index tangentDim =
        static_cast<Eigen::Index>(manifoldPtr_->getTangentDim());
    matrix_t jacobian = matrix_t::Zero(3, tangentDim);
    for (Eigen::Index i = 0; i < tangentDim; ++i) {
      jacobian.col(i) =
          (function(retracted(state, i, kFiniteDifferenceStep)) -
           function(retracted(state, i, -kFiniteDifferenceStep))) /
          (2.0 * kFiniteDifferenceStep);
    }
    return jacobian;
  }

  /** state (+) step e_i on the fixture's state manifold. */
  vector_t retracted(const vector_t& state,
                     Eigen::Index i,
                     scalar_t step) const {
    vector_t direction = vector_t::Zero(
        static_cast<Eigen::Index>(manifoldPtr_->getTangentDim()));
    direction(i) = 1.0;
    vector_t result;
    manifoldPtr_->retract(state, direction, step, result);
    return result;
  }

  mpc_config::TaskFile task_;
  std::string urdfFile_;
  std::unique_ptr<ModelSettings> modelSettingsPtr_;
  std::unique_ptr<PinocchioInterface> pinocchioInterfacePtr_;
  std::unique_ptr<AngularCenterOfMass> acomPtr_;
  CentroidalModelInfo info_;
  std::unique_ptr<test::QuaternionLayoutRobotModel> modelPtr_;
  std::shared_ptr<const StateManifold> manifoldPtr_;
  Eigen::Index nJoints_ = 0;
  PreComputation preComputation_;
};

/**
 * The trained weights must be indexed by exactly the joints the MPC optimizes
 * over, and the cost is built on the shipped model without complaint.
 */
TEST_F(ComAndAcomTrackingCostTest, CreateAcceptsTheShippedModel) {
  EXPECT_EQ(acomPtr_->getInputDim(), static_cast<size_t>(nJoints_));
  EXPECT_EQ(modelSettingsPtr_->mpcModelJointNames.size(),
            static_cast<size_t>(nJoints_));
  EXPECT_NE(makeCost(weights(1.0, 2.0, 3.0), weights(4.0, 5.0, 6.0)), nullptr);
}

/** The quadratic approximation's value is the cost's value, not a separately
 * assembled number. */
TEST_F(ComAndAcomTrackingCostTest, QuadraticApproximationValueIsGetValue) {
  const std::unique_ptr<ComAndAcomTrackingCost> cost =
      makeCost(weights(30.0, 20.0, 40.0), weights(25.0, 15.0, 10.0));
  ASSERT_NE(cost, nullptr);
  const vector_t state =
      makeState(vector3_t(0.30, 0.20, -0.15), /*jointOffset=*/0.2);
  const TargetTrajectories target =
      reference(makeState(vector3_t(0.10, -0.05, 0.05), /*jointOffset=*/-0.1));

  const scalar_t value =
      cost->getValue(/*time=*/0.0, state, target, preComputation_);
  const ScalarFunctionQuadraticApproximation approximation =
      cost->getQuadraticApproximation(/*time=*/0.0, state, target,
                                      preComputation_);
  EXPECT_GT(value, 0.0)
      << "the state and the reference differ, so the cost cannot be zero";
  EXPECT_NEAR(approximation.f, value, 1.0e-12 * std::max(1.0, std::abs(value)));
}

/**
 * The gradient with respect to the stored slots is exactly the derivative of
 * getValue. The Gauss-Newton gradient J^T Q e of a least-squares cost is exact,
 * so this holds to finite-difference precision, and it holds at a tilted base
 * with bent joints, where a dropped or doubled tangent lift E^+(xi) of the
 * quaternion slots, a block placed at the wrong state index, a flipped sign, or
 * a dropped adjoint or right Jacobian of the ACoM attitude would each show up.
 */
TEST_F(ComAndAcomTrackingCostTest, GradientMatchesFiniteDifferencesOfGetValue) {
  const std::unique_ptr<ComAndAcomTrackingCost> cost =
      makeCost(weights(30.0, 20.0, 40.0), weights(25.0, 15.0, 10.0));
  ASSERT_NE(cost, nullptr);
  const vector_t state =
      makeState(vector3_t(0.30, 0.20, -0.15), /*jointOffset=*/0.2);
  const TargetTrajectories target =
      reference(makeState(vector3_t(0.10, -0.05, 0.05), /*jointOffset=*/-0.1));

  const vector_t gradient =
      cost->getQuadraticApproximation(/*time=*/0.0, state, target,
                                      preComputation_)
          .dfdx;
  ASSERT_EQ(gradient.size(), state.size());
  vector_t finiteDifference = vector_t::Zero(state.size());
  for (Eigen::Index i = 0; i < state.size(); ++i) {
    vector_t plus = state;
    vector_t minus = state;
    plus(i) += kFiniteDifferenceStep;
    minus(i) -= kFiniteDifferenceStep;
    finiteDifference(i) =
        (cost->getValue(/*time=*/0.0, plus, target, preComputation_) -
         cost->getValue(/*time=*/0.0, minus, target, preComputation_)) /
        (2.0 * kFiniteDifferenceStep);
  }
  ASSERT_GT(finiteDifference.norm(), 1.0e-3)
      << "a vanishing gradient would make this comparison vacuous";
  EXPECT_TRUE(gradient.head(kGeneralizedCoordinatesStartIndex).isZero(0.0))
      << "the momentum does not enter this cost";
  for (Eigen::Index i = 0; i < state.size(); ++i) {
    EXPECT_NEAR(gradient(i), finiteDifference(i),
                kFiniteDifferenceTolerance *
                    std::max(1.0, finiteDifference.cwiseAbs().maxCoeff()))
        << "state index " << i;
  }
}

/** The Hessian is the Gauss-Newton product J^T Q J, with each J the true
 * Jacobian of its residual. */
TEST_F(ComAndAcomTrackingCostTest,
       HessianIsTheGaussNewtonProductOfTheTrueJacobians) {
  const matrix_t Q_com = weights(30.0, 20.0, 40.0);
  const matrix_t Q_acom = weights(25.0, 15.0, 10.0);
  const std::unique_ptr<ComAndAcomTrackingCost> cost = makeCost(Q_com, Q_acom);
  ASSERT_NE(cost, nullptr);
  const vector_t state =
      makeState(vector3_t(0.30, 0.20, -0.15), /*jointOffset=*/0.2);
  const TargetTrajectories target =
      reference(makeState(vector3_t(0.10, -0.05, 0.05), /*jointOffset=*/-0.1));

  const vector_t stateRef = target.stateTrajectory.front();
  const matrix_t J_com = finiteDifferenceJacobian(
      state,
      [this](const vector_t& x) -> vector3_t { return centerOfMass(x); });
  const matrix_t J_acom = finiteDifferenceJacobian(
      state, [this, &stateRef](const vector_t& x) -> vector3_t {
        return acomResidual(x, stateRef);
      });
  const matrix_t expected =
      J_com.transpose() * Q_com * J_com + J_acom.transpose() * Q_acom * J_acom;

  const matrix_t hessian = cost->getQuadraticApproximation(
                                   /*time=*/0.0, state, target, preComputation_)
                               .dfdxx;
  ASSERT_EQ(hessian.rows(), state.size());
  ASSERT_EQ(hessian.cols(), state.size());
  EXPECT_LT((hessian - expected).cwiseAbs().maxCoeff(),
            kFiniteDifferenceTolerance *
                std::max(1.0, expected.cwiseAbs().maxCoeff()))
      << "max abs error " << (hessian - expected).cwiseAbs().maxCoeff();
}

/**
 * The yaw error is wrapped: a heading of +3.1 rad tracking a reference of -3.1
 * rad is 0.083 rad off, not 6.2. The joints are equal and the base level, so
 * the ACoM headings differ by the base yaw difference alone (a rotation about
 * the vertical adds to the twist heading).
 */
TEST_F(ComAndAcomTrackingCostTest, YawErrorIsWrappedAcrossPi) {
  const std::unique_ptr<ComAndAcomTrackingCost> cost =
      makeCost(matrix_t::Zero(3, 3), weights(1.0, 0.0, 0.0));
  ASSERT_NE(cost, nullptr);
  const vector_t state =
      makeState(vector3_t(3.1, 0.0, 0.0), /*jointOffset=*/0.2);
  const scalar_t shortWay = 2.0 * M_PI - 6.2;

  const scalar_t across = cost->getValue(
      /*time=*/0.0, state,
      reference(makeState(vector3_t(-3.1, 0.0, 0.0), /*jointOffset=*/0.2)),
      preComputation_);
  const scalar_t direct =
      cost->getValue(/*time=*/0.0, state,
                     reference(makeState(vector3_t(3.1 + shortWay, 0.0, 0.0),
                                         /*jointOffset=*/0.2)),
                     preComputation_);
  EXPECT_NEAR(across, 0.5 * shortWay * shortWay, 1.0e-9);
  EXPECT_NEAR(across, direct, 1.0e-9);
}

/**
 * Each row of Q_acom weights its own row of the heading-tilt residual (yaw,
 * pitch, roll): with one row weighted the cost is half the square of that row
 * of e_HT, recomputed here from the state's slots. The joints differ between
 * the state and the reference, so all three rows are nonzero and distinct, and
 * a swapped row would weight another one.
 */
TEST_F(ComAndAcomTrackingCostTest, EachQacomRowWeightsItsOwnHeadingTiltRow) {
  const vector_t state =
      makeState(vector3_t(0.2, 0.1, -0.1), /*jointOffset=*/0.25);
  const vector_t stateRef =
      makeState(vector3_t(0.2, 0.1, -0.1), /*jointOffset=*/-0.15);
  const vector3_t residual = acomResidual(state, stateRef);
  ASSERT_GT(std::abs(residual.x() - residual.z()), 1.0e-4)
      << "the yaw and roll rows must differ for this to tell them apart";
  ASSERT_GT(std::abs(residual.y()), 1.0e-4);

  for (Eigen::Index row = 0; row < 3; ++row) {
    matrix_t Q_acom = matrix_t::Zero(3, 3);
    Q_acom(row, row) = 1.0;
    const std::unique_ptr<ComAndAcomTrackingCost> cost =
        makeCost(matrix_t::Zero(3, 3), Q_acom);
    ASSERT_NE(cost, nullptr);
    EXPECT_NEAR(cost->getValue(/*time=*/0.0, state, reference(stateRef),
                               preComputation_),
                0.5 * residual(row) * residual(row), 1.0e-12)
        << "row " << row;
  }
}

/**
 * A turn of the whole robot about the vertical moves the yaw row alone: adding
 * alpha to the base's ZYX yaw rotates the base, and so the ACoM, by alpha about
 * the world z axis, which adds alpha to the twist heading and leaves the tilt
 * of the heading frame unchanged - at a tilted base and bent joints, where the
 * additive Euler rule mixed them.
 */
TEST_F(ComAndAcomTrackingCostTest, ATurnAboutTheVerticalMovesOnlyTheYawRow) {
  constexpr scalar_t kTurn = 0.4;
  const vector_t stateRef =
      makeState(vector3_t(0.3, 0.2, -0.15), /*jointOffset=*/0.2);
  const vector_t turned =
      makeState(vector3_t(0.3 + kTurn, 0.2, -0.15), /*jointOffset=*/0.2);
  const std::unique_ptr<ComAndAcomTrackingCost> yawOnly =
      makeCost(matrix_t::Zero(3, 3), weights(1.0, 0.0, 0.0));
  const std::unique_ptr<ComAndAcomTrackingCost> tiltOnly =
      makeCost(matrix_t::Zero(3, 3), weights(0.0, 1.0, 1.0));
  ASSERT_NE(yawOnly, nullptr);
  ASSERT_NE(tiltOnly, nullptr);
  EXPECT_NEAR(yawOnly->getValue(/*time=*/0.0, turned, reference(stateRef),
                                preComputation_),
              0.5 * kTurn * kTurn, 1.0e-12);
  EXPECT_NEAR(tiltOnly->getValue(/*time=*/0.0, turned, reference(stateRef),
                                 preComputation_),
              0.0, 1.0e-24);
}

/** clone() is an independent copy that evaluates identically. */
TEST_F(ComAndAcomTrackingCostTest, CloneEvaluatesIdenticallyAndIndependently) {
  const std::unique_ptr<ComAndAcomTrackingCost> cost =
      makeCost(weights(30.0, 20.0, 40.0), weights(25.0, 15.0, 10.0));
  ASSERT_NE(cost, nullptr);
  const std::unique_ptr<ComAndAcomTrackingCost> clone(cost->clone());
  const vector_t state =
      makeState(vector3_t(0.30, 0.20, -0.15), /*jointOffset=*/0.2);
  const TargetTrajectories target =
      reference(makeState(vector3_t(0.10, -0.05, 0.05), /*jointOffset=*/-0.1));

  const ScalarFunctionQuadraticApproximation original =
      cost->getQuadraticApproximation(/*time=*/0.0, state, target,
                                      preComputation_);
  const ScalarFunctionQuadraticApproximation cloned =
      clone->getQuadraticApproximation(/*time=*/0.0, state, target,
                                       preComputation_);
  EXPECT_EQ(cloned.f, original.f);
  EXPECT_TRUE(cloned.dfdx.isApprox(original.dfdx, 0.0));
  EXPECT_TRUE(cloned.dfdxx.isApprox(original.dfdxx, 0.0));

  ASSERT_TRUE(
      clone->setWeights(weights(1.0, 1.0, 1.0), weights(1.0, 1.0, 1.0)).ok());
  EXPECT_EQ(cost->getValue(/*time=*/0.0, state, target, preComputation_),
            original.f)
      << "re-weighting the clone changed the original";
  EXPECT_NE(clone->getValue(/*time=*/0.0, state, target, preComputation_),
            original.f);
}

TEST_F(ComAndAcomTrackingCostTest, WeightsMustBe3x3) {
  const absl::StatusOr<std::unique_ptr<ComAndAcomTrackingCost>> cost =
      ComAndAcomTrackingCost::Create(
          matrix_t::Identity(2, 2), matrix_t::Identity(3, 3),
          *pinocchioInterfacePtr_, info_, modelPtr_.get(),
          modelSettingsPtr_->robotName);
  ASSERT_FALSE(cost.ok());
  EXPECT_EQ(cost.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(cost.status().message(), "Q_com"))
      << cost.status();

  const std::unique_ptr<ComAndAcomTrackingCost> valid =
      makeCost(matrix_t::Identity(3, 3), matrix_t::Identity(3, 3));
  ASSERT_NE(valid, nullptr);
  const scalar_t weightBefore = valid->getQAcom()(0, 0);
  const absl::Status refused = valid->setWeights(
      matrix_t::Identity(3, 3), 2.0 * matrix_t::Identity(4, 4));
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.message(), "Q_acom")) << refused;
  EXPECT_EQ(valid->getQAcom().rows(), 3)
      << "a refused update changed the weights";
  EXPECT_EQ(valid->getQAcom()(0, 0), weightBefore);
  EXPECT_TRUE(ComAndAcomTrackingCost::validateWeights(matrix_t::Identity(3, 3),
                                                      matrix_t::Identity(3, 3))
                  .ok());
}

TEST_F(ComAndAcomTrackingCostTest,
       CreateRejectsAnInfoThatDoesNotDescribeTheModel) {
  CentroidalModelInfo wrong = info_;
  wrong.actuatedDofNum = info_.actuatedDofNum - 1;
  const absl::StatusOr<std::unique_ptr<ComAndAcomTrackingCost>> cost =
      ComAndAcomTrackingCost::Create(
          matrix_t::Identity(3, 3), matrix_t::Identity(3, 3),
          *pinocchioInterfacePtr_, wrong, modelPtr_.get(),
          modelSettingsPtr_->robotName);
  ASSERT_FALSE(cost.ok());
  EXPECT_EQ(cost.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(cost.status().message(), "actuatedDofNum"))
      << cost.status();
}

TEST_F(ComAndAcomTrackingCostTest, CreateRejectsARobotWithoutANetwork) {
  const absl::StatusOr<std::unique_ptr<ComAndAcomTrackingCost>> cost =
      ComAndAcomTrackingCost::Create(
          matrix_t::Identity(3, 3), matrix_t::Identity(3, 3),
          *pinocchioInterfacePtr_, info_, modelPtr_.get(), "r1");
  ASSERT_FALSE(cost.ok());
  EXPECT_EQ(cost.status().code(), absl::StatusCode::kNotFound);
  EXPECT_TRUE(
      absl::StrContains(cost.status().message(), "com_and_acom_tracking_cost"))
      << cost.status();
}

/**
 * The runtime joint check on the model the cost is actually built on. The task
 * file is the shipped one with a different fourth joint fixed - neck_ry instead
 * of r_arm_wrx - so the MPC model still has 24 joints, and a check that only
 * counted them would pass. The network was trained with the head active, so
 * from index 7 on every joint it is fed is a different one; Create has to
 * refuse, and name the first joint that differs.
 */
TEST_F(ComAndAcomTrackingCostTest,
       CreateRejectsAModelWithADifferentFixedJointSet) {
  mpc_config::TaskFile drifted = task_;
  std::vector<std::string>& fixedJoints =
      drifted.model_settings.fixed_joint_names;
  ASSERT_EQ(std::count(fixedJoints.begin(), fixedJoints.end(), "r_arm_wrx"), 1)
      << "the shipped task file no longer fixes r_arm_wrx once; update this "
         "test";
  *std::find(fixedJoints.begin(), fixedJoints.end(), "r_arm_wrx") = "neck_ry";

  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const ModelSettings driftedSettings,
      ModelSettings::Create(drifted, urdfFile_, "testComAndAcomTrackingCost",
                            /*verbose=*/false));
  ASSERT_EQ(driftedSettings.mpcModelJointNames.size(),
            static_cast<size_t>(nJoints_))
      << "the joint COUNT must not change";
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const PinocchioInterface driftedModel,
      loadCustomPinocchioInterface(drifted, urdfFile_, driftedSettings));
  const CentroidalModelInfo driftedInfo =
      makeInfo(driftedModel, driftedSettings);

  const absl::StatusOr<std::unique_ptr<ComAndAcomTrackingCost>> cost =
      ComAndAcomTrackingCost::Create(matrix_t::Identity(3, 3),
                                     matrix_t::Identity(3, 3), driftedModel,
                                     driftedInfo, modelPtr_.get(), "atlas");
  ASSERT_FALSE(cost.ok())
      << "a model with the head fixed and a wrist active was accepted";
  EXPECT_EQ(cost.status().code(), absl::StatusCode::kFailedPrecondition);
  const std::string message(cost.status().message());
  EXPECT_TRUE(absl::StrContains(message, "model_settings.robot_name 'atlas'"))
      << message;
  EXPECT_TRUE(absl::StrContains(message, "joint 7 is 'neck_ry'")) << message;
  EXPECT_TRUE(absl::StrContains(message, "'r_arm_shz' in the MPC model"))
      << message;
  EXPECT_TRUE(absl::StrContains(message, "model_settings.fixed_joint_names"))
      << message;
}

/**
 * The base-pose weights the factory and the live updater drop while this cost
 * runs are exactly the tuning rows whose error this cost re-expresses -
 * base_position and base_orientation - and nothing else.
 */
TEST_F(ComAndAcomTrackingCostTest,
       ZeroBasePoseWeightsClearsExactlyTheBasePoseBlock) {
  // Q is written in the tuning layout, n_t = the state's tangent dimension
  // rows.
  const Eigen::Index n = static_cast<Eigen::Index>(info_.getStateTangentDim());
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
          row >= kBasePositionTuningRow && row < kJointTuningRow &&
          col >= kBasePositionTuningRow && col < kJointTuningRow;
      EXPECT_EQ(Q(row, col), inBasePose ? 0.0 : original(row, col))
          << "(" << row << ", " << col << ")";
    }
  }
  // Idempotent: a live reload may apply it to an already-zeroed matrix.
  const matrix_t once = Q;
  ComAndAcomTrackingCost::zeroBasePoseWeights(Q);
  EXPECT_EQ(Q, once);

  // The block is the rows of base_position and base_orientation of the
  // centroidal tuning layout, which Q is written in.
  const TuningLayout tuningLayout = tuningLayoutOf(
      stateInputLayout(*modelSettingsPtr_, StateInputLayout::Mpc::kCentroidal));
  const absl::StatusOr<size_t> positionRow =
      tuningSegmentStart(tuningLayout, "base_position");
  const absl::StatusOr<size_t> orientationRow =
      tuningSegmentStart(tuningLayout, "base_orientation");
  ASSERT_TRUE(positionRow.ok()) << positionRow.status();
  ASSERT_TRUE(orientationRow.ok()) << orientationRow.status();
  EXPECT_EQ(*positionRow, static_cast<size_t>(kBasePositionTuningRow));
  EXPECT_EQ(*orientationRow + 3, static_cast<size_t>(kJointTuningRow));
}

/**
 * The reference of a node is the robot model's interpolation of the target
 * (MpcRobotModelBase::getDesiredState), the one that follows the state layout:
 * between two knots the cost measures from the interpolated state.
 */
TEST_F(ComAndAcomTrackingCostTest,
       TheReferenceIsTheRobotModelsInterpolationOfTheTarget) {
  const std::unique_ptr<ComAndAcomTrackingCost> cost =
      makeCost(weights(30.0, 20.0, 40.0), weights(25.0, 15.0, 10.0));
  ASSERT_NE(cost, nullptr);
  const vector_t first =
      makeState(vector3_t(0.1, 0.05, -0.02), /*jointOffset=*/0.1);
  const vector_t second =
      makeState(vector3_t(0.3, -0.05, 0.04), /*jointOffset=*/-0.1);
  const TargetTrajectories twoKnots(
      {0.0, 1.0}, {first, second},
      {vector_t::Zero(info_.inputDim), vector_t::Zero(info_.inputDim)});
  const vector_t midpoint = modelPtr_->getDesiredState(twoKnots, /*time=*/0.5);
  EXPECT_EQ(cost->getValue(/*time=*/0.5, midpoint, twoKnots, preComputation_),
            0.0)
      << "the reference is not the interpolated knot";
  EXPECT_GT(cost->getValue(/*time=*/0.5, first, twoKnots, preComputation_),
            0.0);
}

/**
 * At a level base the ACoM rows' Jacobian with respect to the base's body
 * rotation (the tangent of its quaternion slots) is J_HT(xi_a)
 * R(Delta_theta)^T: for the network's rotation at the zero posture it is P_3 =
 * antidiag(1, 1, 1) - the heading-tilt rows (yaw, pitch, roll) of a body
 * rotation (x, y, z) - to first order in |Delta_theta|, and at |Delta_theta| ->
 * 0 it would be P_3 exactly; through the Euler chart (times T_B(0) = P_3) that
 * is the identity block. The joints' columns are J_HT(xi_a) J_r(Delta_theta)
 * J_Delta_theta.
 */
TEST_F(
    ComAndAcomTrackingCostTest,
    AtALevelBaseTheBaseBlockIsTheHeadingTiltPermutationToFirstOrderInTheJointRotation) {
  const std::unique_ptr<ComAndAcomTrackingCost> cost =
      makeCost(matrix_t::Zero(3, 3), matrix_t::Identity(3, 3));
  ASSERT_NE(cost, nullptr);
  vector_t state = makeState(vector3_t(1.2, 0.0, 0.0), /*jointOffset=*/0.0);
  state.tail(nJoints_).setZero();
  const vector_t stateRef = state;
  const vector3_t offset =
      acomPtr_->computeJointOrientationOffset(state.tail(nJoints_));

  // J along the retraction by finite differences of the residual; the base
  // rotation's tangent columns start at the quaternion's first slot.
  const matrix_t tangentJacobian = tangentFiniteDifferenceJacobian(
      state, [this, &stateRef](const vector_t& x) -> vector3_t {
        return acomResidual(x, stateRef);
      });
  const matrix_t baseBlock =
      tangentJacobian.block(0, kBaseOrientationStateIndex, 3, 3);
  matrix_t headingTiltPermutation = matrix_t::Zero(3, 3);
  headingTiltPermutation(0, 2) = headingTiltPermutation(1, 1) =
      headingTiltPermutation(2, 0) = 1.0;
  EXPECT_LT((baseBlock - headingTiltPermutation).cwiseAbs().maxCoeff(),
            2.0 * offset.norm() + 1.0e-6)
      << "|Delta_theta| = " << offset.norm() << "\n"
      << baseBlock;
  // The cost's Gauss-Newton Hessian of an identity Q_acom is J^T J, in the
  // stored slots as in the tangent.
  const matrix_t J = finiteDifferenceJacobian(
      state, [this, &stateRef](const vector_t& x) -> vector3_t {
        return acomResidual(x, stateRef);
      });
  matrix_t hessian =
      cost->getQuadraticApproximation(/*time=*/0.0, state, reference(stateRef),
                                      preComputation_)
          .dfdxx;
  EXPECT_LT((hessian - J.transpose() * J).cwiseAbs().maxCoeff(),
            kFiniteDifferenceTolerance *
                std::max(1.0, hessian.cwiseAbs().maxCoeff()));
  manifoldPtr_->pullBackHessian(state, hessian);
  EXPECT_LT((hessian - tangentJacobian.transpose() * tangentJacobian)
                .cwiseAbs()
                .maxCoeff(),
            kFiniteDifferenceTolerance *
                std::max(1.0, hessian.cwiseAbs().maxCoeff()));
}

/**
 * The solver works in the tangent: the cost's gradient and Gauss-Newton Hessian
 * pulled back through the state manifold (E(x)' g, E(x)' H E(x)) are the
 * derivative of getValue along the retraction x (+) dx and J_t' Q J_t of the
 * residuals' tangent Jacobians, at a tilted base with bent joints. The cost
 * reads the quaternion through xi / |xi|, so its ambient gradient has no radial
 * part, g' xi = 0, and a stored quaternion of any length - or its negative -
 * gives the same value.
 */
TEST_F(ComAndAcomTrackingCostTest,
       ThePulledBackDerivativesAreTheDerivativesAlongTheRetraction) {
  const matrix_t Q_com = weights(30.0, 20.0, 40.0);
  const matrix_t Q_acom = weights(25.0, 15.0, 10.0);
  const std::unique_ptr<ComAndAcomTrackingCost> cost = makeCost(Q_com, Q_acom);
  ASSERT_NE(cost, nullptr);
  const vector_t state =
      makeState(vector3_t(0.30, 0.20, -0.15), /*jointOffset=*/0.2);
  const TargetTrajectories target =
      reference(makeState(vector3_t(0.10, -0.05, 0.05), /*jointOffset=*/-0.1));
  const vector_t stateRef = target.stateTrajectory.front();

  const ScalarFunctionQuadraticApproximation approximation =
      cost->getQuadraticApproximation(/*time=*/0.0, state, target,
                                      preComputation_);
  EXPECT_LT(std::abs(approximation.dfdx.segment<4>(kBaseOrientationStateIndex)
                         .dot(state.segment<4>(kBaseOrientationStateIndex))),
            1.0e-12 * std::max(1.0, approximation.dfdx.cwiseAbs().maxCoeff()))
      << "the ambient gradient has a radial part";

  vector_t gradient = approximation.dfdx;
  manifoldPtr_->pullBackGradient(state, gradient);
  const Eigen::Index tangentDim =
      static_cast<Eigen::Index>(manifoldPtr_->getTangentDim());
  ASSERT_EQ(gradient.size(), tangentDim);
  vector_t finiteDifference(tangentDim);
  for (Eigen::Index i = 0; i < tangentDim; ++i) {
    finiteDifference(i) =
        (cost->getValue(/*time=*/0.0,
                        retracted(state, i, kFiniteDifferenceStep), target,
                        preComputation_) -
         cost->getValue(/*time=*/0.0,
                        retracted(state, i, -kFiniteDifferenceStep), target,
                        preComputation_)) /
        (2.0 * kFiniteDifferenceStep);
  }
  ASSERT_GT(finiteDifference.segment<3>(kBaseOrientationStateIndex).norm(),
            1.0e-3)
      << "the base rotation must move the cost";
  EXPECT_LT((gradient - finiteDifference).cwiseAbs().maxCoeff(),
            kFiniteDifferenceTolerance *
                std::max(1.0, finiteDifference.cwiseAbs().maxCoeff()));

  const matrix_t J_com = tangentFiniteDifferenceJacobian(
      state,
      [this](const vector_t& x) -> vector3_t { return centerOfMass(x); });
  const matrix_t J_acom = tangentFiniteDifferenceJacobian(
      state, [this, &stateRef](const vector_t& x) -> vector3_t {
        return acomResidual(x, stateRef);
      });
  const matrix_t expected =
      J_com.transpose() * Q_com * J_com + J_acom.transpose() * Q_acom * J_acom;
  matrix_t hessian = approximation.dfdxx;
  manifoldPtr_->pullBackHessian(state, hessian);
  EXPECT_LT((hessian - expected).cwiseAbs().maxCoeff(),
            kFiniteDifferenceTolerance *
                std::max(1.0, expected.cwiseAbs().maxCoeff()));

  // Radial invariance: the same attitude stored at another length, or as the
  // other quaternion of the same rotation.
  const scalar_t value =
      cost->getValue(/*time=*/0.0, state, target, preComputation_);
  for (const scalar_t scale : {0.5, 2.0}) {
    const vector_t scaled =
        makeState(vector3_t(0.30, 0.20, -0.15), /*jointOffset=*/0.2, scale);
    EXPECT_NEAR(cost->getValue(/*time=*/0.0, scaled, target, preComputation_),
                value, 1.0e-12 * value)
        << "scale " << scale;
  }
  const vector_t negated =
      makeState(vector3_t(0.30, 0.20, -0.15), /*jointOffset=*/0.2,
                /*quaternionScale=*/-1.0);
  EXPECT_EQ(cost->getValue(/*time=*/0.0, negated, target, preComputation_),
            value)
      << "-xi is the same attitude, bit for bit";
  vector_t negatedGradient =
      cost->getQuadraticApproximation(/*time=*/0.0, negated, target,
                                      preComputation_)
          .dfdx;
  manifoldPtr_->pullBackGradient(negated, negatedGradient);
  EXPECT_LT((negatedGradient - gradient).cwiseAbs().maxCoeff(),
            1.0e-12 * std::max(1.0, gradient.cwiseAbs().maxCoeff()));
}

/**
 * The cost reads the state only through the robot model, which must therefore
 * read the Pinocchio model the cost evaluates: a model of one joint fewer is
 * refused, naming both sets of dimensions.
 */
TEST_F(ComAndAcomTrackingCostTest,
       CreateRejectsARobotModelOfAnotherPinocchioModel) {
  mpc_config::TaskFile fewerJoints = task_;
  fewerJoints.model_settings.fixed_joint_names.emplace_back("neck_ry");
  const ModelSettings fewerSettings = checkedValue(ModelSettings::Create(
      fewerJoints, urdfFile_, "testComAndAcomTrackingCost", /*verbose=*/false));
  ASSERT_EQ(fewerSettings.mpcModelJointNames.size() + 1,
            static_cast<size_t>(nJoints_))
      << "neck_ry is no longer an active joint";
  const test::QuaternionLayoutRobotModel otherModel(
      fewerSettings, StateInputLayout::Mpc::kCentroidal);
  const absl::StatusOr<std::unique_ptr<ComAndAcomTrackingCost>> cost =
      ComAndAcomTrackingCost::Create(matrix_t::Identity(3, 3),
                                     matrix_t::Identity(3, 3),
                                     *pinocchioInterfacePtr_, info_,
                                     &otherModel, modelSettingsPtr_->robotName);
  ASSERT_FALSE(cost.ok());
  EXPECT_EQ(cost.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(
      cost.status().message(),
      "robot model does not read the Pinocchio model's state"))
      << cost.status();
}

/**
 * The reduced model must place the CoM somewhere physically sensible, which
 * catches a mis-built or mis-scaled Pinocchio model before it reaches the cost.
 */
TEST_F(ComAndAcomTrackingCostTest, CenterOfMassIsPhysicallyPlausible) {
  vector_t state = modelPtr_->getNeutralState();
  state(kGeneralizedCoordinatesStartIndex + 2) = 0.8;

  CentroidalModelPinocchioMapping mapping(info_);
  const vector_t q = mapping.getPinocchioJointPosition(state);
  ASSERT_EQ(static_cast<size_t>(q.size()), info_.getConfigurationDim());

  const vector3_t com = centerOfMass(state);
  EXPECT_NEAR(com[0], 0.0, 0.5);
  EXPECT_NEAR(com[1], 0.0, 0.5);
  EXPECT_GT(com[2], 0.3);
  EXPECT_GT(info_.robotMass, 0.0);
}

namespace {

/** One draw of a base attitude and an ACoM offset for the E2 bounds: ZYX angles
 * with |tilt| = hypot(pitch, roll). */
struct AcomDraw {
  vector3_t eulerZyx = vector3_t::Zero();
  vector3_t offset = vector3_t::Zero();  // Delta_theta, the joints' rotation of
                                         // the ACoM in the base frame
  scalar_t tilt = 0.0;
};

AcomDraw drawAcom(std::mt19937& engine, scalar_t maxTilt, scalar_t maxOffset) {
  std::uniform_real_distribution<scalar_t> unit(0.0, 1.0);
  std::normal_distribution<scalar_t> normal(0.0, 1.0);
  AcomDraw draw;
  const scalar_t direction = 2.0 * M_PI * unit(engine);
  draw.tilt = maxTilt * unit(engine);
  draw.eulerZyx = vector3_t(M_PI * (2.0 * unit(engine) - 1.0),
                            draw.tilt * std::sin(direction),
                            draw.tilt * std::cos(direction));
  const vector3_t axis(normal(engine), normal(engine), normal(engine));
  draw.offset = (maxOffset * unit(engine) / axis.norm()) * axis;
  return draw;
}

/** The yaw, pitch, roll rows (P3) of a body rotation vector (x, y, z). */
vector3_t permutedToHeadingTiltRows(const vector3_t& rotation) {
  return vector3_t(rotation(2), rotation(1), rotation(0));
}

/** The heading-tilt residual of the ACoM attitudes xi(Theta) (x)
 * Exp(Delta_theta) of a state and a reference. */
vector3_t acomResidual(const vector3_t& eulerZyx,
                       const vector3_t& offset,
                       const vector3_t& referenceEulerZyx,
                       const vector3_t& referenceOffset) {
  return headingTiltResidual(
      quaternionProduct(quaternionFromEulerZyx(eulerZyx),
                        quaternionExp(offset)),
      quaternionProduct(quaternionFromEulerZyx(referenceEulerZyx),
                        quaternionExp(referenceOffset)));
}

/** `angle` wrapped into [-pi, pi]. */
scalar_t wrappedAngle(scalar_t angle) {
  return std::remainder(angle, 2.0 * M_PI);
}

/** The ranges the bounds are drawn over: (max |tilt|, max |Delta_theta|), both
 * within README 4.4's 0.2 rad. */
constexpr std::array<std::pair<scalar_t, scalar_t>, 5> kAcomRanges = {
    {{0.05, 0.05}, {0.1, 0.1}, {0.2, 0.2}, {0.05, 0.2}, {0.2, 0.05}}};
constexpr int kAcomDrawsPerRange = 2000;

}  // namespace

/**
 * README section 4.4, E2 ComAndAcom: the ACoM rows e_HT(xi (x)
 * Exp(Delta_theta), xi_r (x) Exp(Delta_theta_r)) differ from the base
 * attitudes' own heading-tilt residual plus the additive Euler rule's P3
 * (Delta_theta - Delta_theta_r), per row (yaw wrapped), by at most c
 * (|Delta_theta| |tilt| + |Delta_theta|^2) for the state plus the same for the
 * reference, c = 1 + max |Delta_theta| + max |tilt|; and so the rows against
 * the additive Euler rule Theta - Theta_r + P3 (Delta_theta - Delta_theta_r) by
 * at most that plus the base attitudes' own change. The bound is not vacuous:
 * some draw comes within a quarter of it (a probe reached 0.81 of it).
 */
TEST(ComAndAcomEulerTolerance,
     RowsStayWithinTheStatedBoundOfTheAdditiveEulerRule) {
  std::mt19937 engine(9U);
  scalar_t largestRatio = 0.0;
  for (const std::pair<scalar_t, scalar_t>& range : kAcomRanges) {
    for (int i = 0; i < kAcomDrawsPerRange; ++i) {
      const AcomDraw state = drawAcom(engine, range.first, range.second);
      const AcomDraw reference = drawAcom(engine, range.first, range.second);
      const scalar_t c =
          1.0 + std::max(state.offset.norm(), reference.offset.norm()) +
          std::max(state.tilt, reference.tilt);
      const scalar_t slack =
          c * (state.offset.norm() * state.tilt + state.offset.squaredNorm()) +
          c * (reference.offset.norm() * reference.tilt +
               reference.offset.squaredNorm());
      const vector3_t acom = acomResidual(state.eulerZyx, state.offset,
                                          reference.eulerZyx, reference.offset);
      const vector3_t base =
          headingTiltResidual(quaternionFromEulerZyx(state.eulerZyx),
                              quaternionFromEulerZyx(reference.eulerZyx));
      vector3_t composition =
          acom - base -
          permutedToHeadingTiltRows(state.offset - reference.offset);
      composition(0) = wrappedAngle(composition(0));
      vector3_t eulerDifference = state.eulerZyx - reference.eulerZyx;
      eulerDifference(0) = wrappedAngle(eulerDifference(0));
      vector3_t additive =
          acom - eulerDifference -
          permutedToHeadingTiltRows(state.offset - reference.offset);
      additive(0) = wrappedAngle(additive(0));
      vector3_t own = base - eulerDifference;
      own(0) = wrappedAngle(own(0));
      for (Eigen::Index row = 0; row < 3; ++row) {
        EXPECT_LE(std::abs(composition(row)), slack + 1.0e-12)
            << "row " << row << " range " << range.first << ", "
            << range.second;
        EXPECT_LE(std::abs(additive(row)), std::abs(own(row)) + slack + 1.0e-12)
            << "row " << row;
      }
      if (slack > 0.0)
        largestRatio =
            std::max(largestRatio, composition.cwiseAbs().maxCoeff() / slack);
    }
  }
  RecordProperty("largest_ratio_to_the_bound", absl::StrCat(largestRatio));
  EXPECT_GT(largestRatio, 0.25) << "the bound is not tested near its value";
  EXPECT_LE(largestRatio, 1.0);
}

/**
 * README section 4.4, E2 ComAndAcom Jacobian: in the Euler chart, d e / d Theta
 * of the ACoM rows is within c (|tilt| + |Delta_theta|) of the identity per
 * entry, and d e / d Delta_theta (J_HT(xi_a) J_r(Delta_theta)) within the same
 * of P3, at a level reference of any heading, c = 1 + |Delta_theta| + |tilt|. A
 * probe reached 0.98 of the bound, so the stated constant is used; the bound is
 * not vacuous (some draw within a half of it).
 */
TEST(ComAndAcomEulerTolerance, TheEulerChartJacobianStaysWithinTheStatedBound) {
  constexpr scalar_t kStep = 1.0e-6;
  std::mt19937 engine(11U);
  std::uniform_real_distribution<scalar_t> heading(-M_PI, M_PI);
  matrix3_t permutation = matrix3_t::Zero();
  permutation(0, 2) = 1.0;
  permutation(1, 1) = 1.0;
  permutation(2, 0) = 1.0;
  scalar_t largestRatio = 0.0;
  for (const std::pair<scalar_t, scalar_t>& range : kAcomRanges) {
    for (int i = 0; i < kAcomDrawsPerRange / 4; ++i) {
      const AcomDraw state = drawAcom(engine, range.first, range.second);
      const vector3_t level(heading(engine), 0.0, 0.0);
      const scalar_t bound = (1.0 + state.offset.norm() + state.tilt) *
                             (state.tilt + state.offset.norm());
      matrix3_t baseBlock = matrix3_t::Zero();
      matrix3_t jointBlock = matrix3_t::Zero();
      for (Eigen::Index k = 0; k < 3; ++k) {
        const vector3_t step = kStep * vector3_t::Unit(k);
        vector3_t alongTheta = acomResidual(state.eulerZyx + step, state.offset,
                                            level, vector3_t::Zero()) -
                               acomResidual(state.eulerZyx - step, state.offset,
                                            level, vector3_t::Zero());
        alongTheta(0) = wrappedAngle(alongTheta(0));
        baseBlock.col(k) = alongTheta / (2.0 * kStep);
        vector3_t alongOffset =
            acomResidual(state.eulerZyx, state.offset + step, level,
                         vector3_t::Zero()) -
            acomResidual(state.eulerZyx, state.offset - step, level,
                         vector3_t::Zero());
        alongOffset(0) = wrappedAngle(alongOffset(0));
        jointBlock.col(k) = alongOffset / (2.0 * kStep);
      }
      // The differences' own error, O(kStep^2) and round-off 1e-16 / kStep, is
      // below 1e-8.
      EXPECT_LE((baseBlock - matrix3_t::Identity()).cwiseAbs().maxCoeff(),
                bound + 1.0e-8);
      EXPECT_LE((jointBlock - permutation).cwiseAbs().maxCoeff(),
                bound + 1.0e-8);
      if (bound > 1.0e-3) {
        largestRatio = std::max(
            {largestRatio,
             (baseBlock - matrix3_t::Identity()).cwiseAbs().maxCoeff() / bound,
             (jointBlock - permutation).cwiseAbs().maxCoeff() / bound});
      }
    }
  }
  RecordProperty("largest_ratio_to_the_bound", absl::StrCat(largestRatio));
  EXPECT_GT(largestRatio, 0.5) << "the bound is not tested near its value";
}

/**
 * README section 4.4, E2 planner heading: the contact planner's heading with
 * the ACoM, twistHeading(xi (x) Exp(Delta_theta)), differs from the additive
 * Euler rule's psi + Delta_theta_z by at most |o(theta, phi)| + c
 * (|Delta_theta| |tilt| + |Delta_theta|^2), o(theta, phi) = -2 atan(tan(theta /
 * 2) tan(phi / 2)) the twist's offset from the ZYX yaw (a probe reached 0.49 of
 * the second term).
 */
TEST(ComAndAcomEulerTolerance, ThePlannerHeadingStaysWithinTheStatedBound) {
  std::mt19937 engine(13U);
  for (const std::pair<scalar_t, scalar_t>& range : kAcomRanges) {
    for (int i = 0; i < kAcomDrawsPerRange; ++i) {
      const AcomDraw state = drawAcom(engine, range.first, range.second);
      const scalar_t pitch = state.eulerZyx(1);
      const scalar_t roll = state.eulerZyx(2);
      const scalar_t twistOffset =
          -2.0 * std::atan(std::tan(0.5 * pitch) * std::tan(0.5 * roll));
      const scalar_t c = 1.0 + state.offset.norm() + state.tilt;
      const scalar_t plannerHeading = twistHeading(quaternionProduct(
          quaternionFromEulerZyx(state.eulerZyx), quaternionExp(state.offset)));
      EXPECT_LE(std::abs(wrappedAngle(plannerHeading -
                                      (state.eulerZyx(0) + state.offset(2)))),
                std::abs(twistOffset) +
                    c * (state.offset.norm() * state.tilt +
                         state.offset.squaredNorm()) +
                    1.0e-12);
    }
  }
}

}  // namespace ocs2::humanoid
