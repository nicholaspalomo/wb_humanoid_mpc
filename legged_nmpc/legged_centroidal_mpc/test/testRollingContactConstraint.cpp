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

#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <string>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/ComputationRequest.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/multibody/fwd.hpp"

#include "legged_centroidal_mpc/CentroidalMpcInterface.h"
#include "legged_centroidal_mpc/constraint/RollingContactConstraintCppAd.h"
#include "legged_common_mpc/common/ContactTermNames.h"
#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/MpcRobotModelBase.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_nmpc/legged_common_mpc/test/support/StatusTestMacros.h"
#include "support/FiniteDifferenceChecks.h"
#include "support/RollingContactTestSupport.h"
#include "support/TypedConfigFiles.h"

/**
 * The rolling no-slip constraint (RollingContactConstraintCppAd) of Tron1's
 * two wheels, as CentroidalMpcInterface builds it from the shipped task file:
 *  - its linear approximation is the central finite difference of its value
 *    at random states and inputs;
 *  - it is zero while the robot rolls forward on its wheels and not zero for
 *    a longitudinal or a lateral slip;
 *  - its z row is the position gain times the height error of the contact
 *    point at rest;
 *  - it is active in stance only;
 *  - its value is the velocity of the wheel material at the contact point,
 *    which Pinocchio's kinematics give on their own.
 */
namespace ocs2::legged {
namespace {

using rolling_contact_test::kLeftWheel;
using rolling_contact_test::kRightWheel;

constexpr size_t kNumWheels = 2;
constexpr int kNumRandomPoints = 5;
// [m/s] A forward and a lateral speed of the mass-normalized linear momentum.
constexpr scalar_t kForwardSpeed = 0.5;
constexpr scalar_t kLateralSpeed = 0.3;
// [rad/s] The extra spin of a slipping wheel.
constexpr scalar_t kSlipSpin = 2.0;
// [m] The base raised above its standing height.
constexpr scalar_t kRaisedHeight = 0.01;
// The weight of a leg joint's velocity against a wheel's in the rolling
// solve, which turns the wheels rather than the legs.
constexpr scalar_t kLegVelocityWeight = 100.0;
// [s] The time step of the finite difference of the material point.
constexpr scalar_t kMaterialPointTimeStep = 1.0e-6;
constexpr scalar_t kTolerance = 1.0e-9;
constexpr scalar_t kKinematicsTolerance = 1.0e-6;

/** The interface, built once for the suite: it generates the CppAD libraries
 * of the dynamics and of the two wheel frames. */
class RollingContactConstraintTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    interface_ = checkedValue(CentroidalMpcInterface::Create(
                                  rolling_contact_test::tron1Config(),
                                  tron1Files().urdfFile))
                     .release();
  }

  static void TearDownTestSuite() {
    delete interface_;
    interface_ = nullptr;
  }

  void SetUp() override {
    rolling_contact_test::prepareSolve(mpc(),
                                       rolling_contact_test::bothWheelsStand());
  }

  static CentroidalMpcInterface& mpc() { return *interface_; }

  /** Returns the rolling constraint of wheel `contactIndex`. */
  static const RollingContactConstraintCppAd& constraintOf(
      size_t contactIndex) {
    return mpc()
        .getOptimalControlProblemRef()
        .equalityConstraintPtr->get<RollingContactConstraintCppAd>(
            contact_term::name(mpc().modelSettings().contactNames[contactIndex],
                               contact_term::kRollingNoSlipName));
  }

  /** Returns the problem's pre-computation requested at (`time`, `state`,
   * `input`). */
  static const PreComputation& requestedPreComputation(scalar_t time,
                                                       const vector_t& state,
                                                       const vector_t& input) {
    PreComputation& preComp =
        *mpc().getOptimalControlProblemRef().preComputationPtr;
    preComp.request(Request::Constraint, time, state, input);
    return preComp;
  }

  /** Returns the value of the constraint of wheel `contactIndex`. */
  static vector_t valueOf(size_t contactIndex,
                          const vector_t& state,
                          const vector_t& input) {
    return constraintOf(contactIndex)
        .getValue(rolling_contact_test::kStanceTime, state, input,
                  requestedPreComputation(rolling_contact_test::kStanceTime,
                                          state, input));
  }

 private:
  static CentroidalMpcInterface* absl_nullable interface_;
};

CentroidalMpcInterface* absl_nullable RollingContactConstraintTest::interface_ =
    nullptr;

/** Returns a state near the initial state of `mpc`: random momenta, a tilted
 * and turned base and random joint angles, the wheels' anywhere in a turn. */
vector_t randomState(CentroidalMpcInterface& mpc, std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> unit(-1.0, 1.0);
  const MpcRobotModelBase<scalar_t>& model = mpc.getMpcRobotModel();
  vector_t state = mpc.getInitialState();
  for (Eigen::Index i = 0; i < 6; ++i) state(i) = 0.5 * unit(generator);
  const size_t quaternionStart = model.getBaseOrientationStartIndex();
  vector4_t quatBaseRtWorld =
      state.segment<kQuaternionCoefficients>(quaternionStart);
  for (Eigen::Index i = 0; i < kQuaternionCoefficients; ++i) {
    quatBaseRtWorld(i) += 0.1 * unit(generator);
  }
  state.segment<kQuaternionCoefficients>(quaternionStart) =
      quatBaseRtWorld.normalized();
  const Eigen::Index jointStart =
      static_cast<Eigen::Index>(model.getJointStartindex());
  for (Eigen::Index joint = 0;
       joint < static_cast<Eigen::Index>(model.getJointDim()); ++joint) {
    state(jointStart + joint) += 0.3 * unit(generator);
  }
  // The wheels anywhere in a turn.
  for (const std::string& wheelJoint :
       mpc.modelSettings().rollingContactJointNames()) {
    state(jointStart + static_cast<Eigen::Index>(
                           mpc.modelSettings().jointIndexMap.at(wheelJoint))) =
        3.0 * unit(generator);
  }
  return state;
}

/** Returns a random input: contact forces about the weight and joint
 * velocities up to 2 rad/s. */
vector_t randomInput(CentroidalMpcInterface& mpc, std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> unit(-1.0, 1.0);
  vector_t input = vector_t::Zero(
      static_cast<Eigen::Index>(mpc.getMpcRobotModel().getInputDim()));
  for (Eigen::Index i = 0; i < input.size(); ++i) {
    input(i) = 2.0 * unit(generator);
  }
  return input;
}

TEST_F(RollingContactConstraintTest, itsLinearApproximationIsItsDerivative) {
  std::mt19937 generator(41);
  for (int point = 0; point < kNumRandomPoints; ++point) {
    const vector_t state = randomState(mpc(), generator);
    const vector_t input = randomInput(mpc(), generator);
    for (const size_t wheel : {kLeftWheel, kRightWheel}) {
      SCOPED_TRACE(absl::StrCat("point ", point, ", wheel ", wheel));
      expectStateInputDerivativesMatchFiniteDifferencesAt(
          constraintOf(wheel), rolling_contact_test::kStanceTime, state, input,
          requestedPreComputation(rolling_contact_test::kStanceTime, state,
                                  input));
    }
  }
}

TEST_F(RollingContactConstraintTest, itIsZeroForPureRollingAndNotForSlip) {
  const MpcRobotModelBase<scalar_t>& model = mpc().getMpcRobotModel();
  const ModelSettings& settings = mpc().modelSettings();
  vector_t state = mpc().getInitialState();
  state(0) = kForwardSpeed;
  const vector_t zeroInput =
      vector_t::Zero(static_cast<Eigen::Index>(model.getInputDim()));
  const PreComputation& preComp = requestedPreComputation(
      rolling_contact_test::kStanceTime, state, zeroInput);

  // Both wheels' rows at zero joint velocities and their Jacobian in the
  // joint velocities.
  const Eigen::Index numJoints = static_cast<Eigen::Index>(model.getJointDim());
  const Eigen::Index jointStart =
      static_cast<Eigen::Index>(model.getJointVelocitiesStartindex());
  vector_t residual(kNumWheels * kPositionDim);
  matrix_t jacobian(kNumWheels * kPositionDim, numJoints);
  for (const size_t wheel : {kLeftWheel, kRightWheel}) {
    const VectorFunctionLinearApproximation approximation =
        constraintOf(wheel).getLinearApproximation(
            rolling_contact_test::kStanceTime, state, zeroInput, preComp);
    residual.segment<kPositionDim>(wheel * kPositionDim) = approximation.f;
    jacobian.middleRows<kPositionDim>(wheel * kPositionDim) =
        approximation.dfdu.middleCols(jointStart, numJoints);
  }
  // min |W qdot| subject to J qdot = -g: qdot = W^-1 (J W^-1)^+ (-g).
  vector_t inverseWeights =
      vector_t::Constant(numJoints, 1.0 / kLegVelocityWeight);
  const size_t leftWheelJoint = settings.jointIndexMap.at("wheel_L_Joint");
  const size_t rightWheelJoint = settings.jointIndexMap.at("wheel_R_Joint");
  inverseWeights(static_cast<Eigen::Index>(leftWheelJoint)) = 1.0;
  inverseWeights(static_cast<Eigen::Index>(rightWheelJoint)) = 1.0;
  const matrix_t scaledJacobian = jacobian * inverseWeights.asDiagonal();
  const vector_t jointVelocities =
      inverseWeights.asDiagonal() *
      scaledJacobian.completeOrthogonalDecomposition().solve(-residual);
  vector_t rolling = zeroInput;
  rolling.segment(jointStart, numJoints) = jointVelocities;

  for (const size_t wheel : {kLeftWheel, kRightWheel}) {
    EXPECT_LT(valueOf(wheel, state, rolling).norm(), kTolerance) << wheel;
  }
  // The robot rolls forward on its wheels: the base moves at about the
  // commanded speed and each wheel turns at about speed / radius, forward
  // (+y axles), while the legs barely move.
  std::unique_ptr<MpcRobotModelBase<scalar_t>> modelCopy(model.clone());
  const vector_t generalizedVelocities =
      modelCopy->getGeneralizedVelocities(state, rolling);
  EXPECT_GT(generalizedVelocities(0), 0.5 * kForwardSpeed);
  const scalar_t rollingSpin =
      generalizedVelocities(0) / settings.wheelRadius(kLeftWheel);
  for (const size_t joint : {leftWheelJoint, rightWheelJoint}) {
    EXPECT_NEAR(jointVelocities(static_cast<Eigen::Index>(joint)), rollingSpin,
                0.2 * rollingSpin)
        << joint;
  }

  // A longitudinal slip: the left wheel turns faster than it rolls.
  vector_t slipping = rolling;
  slipping(jointStart + static_cast<Eigen::Index>(leftWheelJoint)) += kSlipSpin;
  const vector_t longitudinal = valueOf(kLeftWheel, state, slipping);
  EXPECT_GT(std::abs(longitudinal(kXAxisIndex)),
            0.5 * settings.wheelRadius(kLeftWheel) * kSlipSpin);
  EXPECT_LT(std::abs(longitudinal(kYAxisIndex)),
            0.1 * std::abs(longitudinal(kXAxisIndex)));

  // A lateral slip: the robot moves sideways on wheels that only roll.
  vector_t sideways = state;
  sideways(1) = kLateralSpeed;
  for (const size_t wheel : {kLeftWheel, kRightWheel}) {
    const vector_t lateral = valueOf(wheel, sideways, rolling);
    EXPECT_GT(std::abs(lateral(kYAxisIndex)), 0.5 * kLateralSpeed) << wheel;
  }
}

TEST_F(RollingContactConstraintTest, itsZRowIsTheGainTimesTheHeightError) {
  vector_t state = mpc().getInitialState();
  const size_t baseStart = mpc().getMpcRobotModel().getBaseStartindex();
  state(static_cast<Eigen::Index>(baseStart + kZAxisIndex)) += kRaisedHeight;
  const vector_t input = vector_t::Zero(
      static_cast<Eigen::Index>(mpc().getMpcRobotModel().getInputDim()));
  const ModelSettings& settings = mpc().modelSettings();
  for (const size_t wheel : {kLeftWheel, kRightWheel}) {
    SCOPED_TRACE(wheel);
    const scalar_t errPosContactRtSurfaceEwrtWorldZ =
        rolling_contact_test::posWheelRtWorldEwrtWorld(
            mpc(), state, settings.contactNames[wheel])
            .z() -
        settings.wheelRadius(wheel);
    // At rest: no momentum and no joint velocity, so no velocity anywhere.
    const vector_t value = valueOf(wheel, state, input);
    EXPECT_NEAR(value(kXAxisIndex), 0.0, kTolerance);
    EXPECT_NEAR(value(kYAxisIndex), 0.0, kTolerance);
    EXPECT_NEAR(value(kZAxisIndex),
                settings.footConstraintConfig.positionErrorGain_z *
                    errPosContactRtSurfaceEwrtWorldZ,
                kTolerance);
    EXPECT_GT(errPosContactRtSurfaceEwrtWorldZ, 0.5 * kRaisedHeight);
  }
}

TEST_F(RollingContactConstraintTest, itIsActiveInStanceOnly) {
  rolling_contact_test::prepareSolve(mpc(),
                                     rolling_contact_test::rightWheelSwing());
  EXPECT_TRUE(
      constraintOf(kLeftWheel).isActive(rolling_contact_test::kMidSwingTime));
  EXPECT_FALSE(
      constraintOf(kRightWheel).isActive(rolling_contact_test::kMidSwingTime));
  EXPECT_TRUE(
      constraintOf(kRightWheel).isActive(rolling_contact_test::kStanceTime));
}

/**
 * Returns the velocity of the wheel material at the contact point of the
 * rolling contact `frameName` (radius `wheelRadius`) by Pinocchio alone: the
 * point of the wheel link under the wheel center at `q`, moved along the
 * generalized velocity `v` by a central difference in time.
 */
vector3_t linvelContactRtWorldEwrtWorld(const pinocchio::Model& model,
                                        const vector_t& q,
                                        const vector_t& v,
                                        absl::string_view frameName,
                                        scalar_t wheelRadius) {
  const pinocchio::FrameIndex frame = model.getFrameId(std::string(frameName));
  pinocchio::Data data(model);
  pinocchio::framesForwardKinematics(model, data, q);
  // The material point in the wheel frame: below the center, in world axes.
  const vector3_t posContactRtWheelEwrtWheel =
      data.oMf[frame].rotation().transpose() *
      (-wheelRadius * vector3_t::UnitZ());
  const vector_t forward =
      pinocchio::integrate(model, q, kMaterialPointTimeStep * v);
  const vector_t backward =
      pinocchio::integrate(model, q, -kMaterialPointTimeStep * v);
  pinocchio::framesForwardKinematics(model, data, forward);
  vector3_t posContactRtWorldEwrtWorld =
      data.oMf[frame].act(posContactRtWheelEwrtWheel);
  pinocchio::framesForwardKinematics(model, data, backward);
  posContactRtWorldEwrtWorld -= data.oMf[frame].act(posContactRtWheelEwrtWheel);
  return posContactRtWorldEwrtWorld / (2.0 * kMaterialPointTimeStep);
}

TEST_F(RollingContactConstraintTest, itIsTheVelocityOfTheMaterialPoint) {
  std::mt19937 generator(43);
  const ModelSettings& settings = mpc().modelSettings();
  const ModelSettings::FootConstraintConfig& gains =
      settings.footConstraintConfig;
  std::unique_ptr<MpcRobotModelBase<scalar_t>> model(
      mpc().getMpcRobotModel().clone());
  for (int point = 0; point < kNumRandomPoints; ++point) {
    const vector_t state = randomState(mpc(), generator);
    const vector_t input = randomInput(mpc(), generator);
    const vector_t q = model->getGeneralizedCoordinates(state);
    const vector_t v = model->getGeneralizedVelocities(state, input);
    for (const size_t wheel : {kLeftWheel, kRightWheel}) {
      SCOPED_TRACE(absl::StrCat("point ", point, ", wheel ", wheel));
      const std::string& frameName = settings.contactNames[wheel];
      const vector3_t linvelMaterial = linvelContactRtWorldEwrtWorld(
          mpc().getPinocchioInterface().getModel(), q, v, frameName,
          settings.wheelRadius(wheel));
      // h_ref is the ground in stance (the swing planner's terrain, 0).
      const scalar_t errPosContactRtSurfaceEwrtWorldZ =
          rolling_contact_test::posWheelRtWorldEwrtWorld(mpc(), state,
                                                         frameName)
              .z() -
          settings.wheelRadius(wheel);
      const vector_t value = valueOf(wheel, state, input);
      EXPECT_NEAR(value(kXAxisIndex),
                  gains.linearVelocityErrorGain_xy * linvelMaterial.x(),
                  kKinematicsTolerance);
      EXPECT_NEAR(value(kYAxisIndex),
                  gains.linearVelocityErrorGain_xy * linvelMaterial.y(),
                  kKinematicsTolerance);
      EXPECT_NEAR(
          value(kZAxisIndex),
          gains.linearVelocityErrorGain_z * linvelMaterial.z() +
              gains.positionErrorGain_z * errPosContactRtSurfaceEwrtWorldZ,
          kKinematicsTolerance);
    }
  }
}

}  // namespace
}  // namespace ocs2::legged
