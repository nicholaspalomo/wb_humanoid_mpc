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
#include <optional>
#include <string>

#include "absl/base/nullability.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_core/ComputationRequest.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/fwd.hpp"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"
#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "support/FourPointContactSet.h"
#include "support/RollingContactTestSupport.h"
#include "support/TypedConfigFiles.h"

/**
 * The references of rolling contacts (humanoid_nmpc/docs/rolling_contact/
 * README.md) on Tron1's two wheels, against a fixed point contact set:
 *  - the wheel joint velocity reference of getDesiredInput() is the rolling
 *    speed of the commanded twist at each contact point in stance, with the
 *    sign of the wheel joint axis, and 0 in swing;
 *  - a rolling contact in stance moves with the predicted base
 *    (getSwingFootReference() in stance), a swing of it lifts off from that
 *    moving point, and task_space_foot_cost in stance tracks it;
 *  - the contacts of the legged four-point set on the DRC Atlas, fixed points,
 *    keep no stance reference, lift off where they were measured and get no
 *    joint velocity reference.
 */
namespace ocs2::humanoid {
namespace {

using rolling_contact_test::kLeftWheel;
using rolling_contact_test::kMidSwingTime;
using rolling_contact_test::kRightWheel;
using rolling_contact_test::kStanceTime;
using rolling_contact_test::kSwingStartTime;

constexpr scalar_t kTolerance = 1.0e-9;
// [m/s] The forward command.
constexpr scalar_t kCmdLinvelBaseRtWorldEwrtWorldX = 0.5;
// [kg m^2 / kg / s] The normalized angular momentum of the turning command.
constexpr scalar_t kCmdMnormAngmomZ = 0.05;
// [s] The end of the target, and the horizon the reference manager prepares.
constexpr scalar_t kTargetEndTime = 2.0;
constexpr scalar_t kHorizonEndTime = 1.0;
// [s] Times in the stance of a standing schedule.
constexpr scalar_t kEarlyStanceTime = 0.1;
constexpr scalar_t kLateStanceTime = 0.9;
// [s] Just after the lift-off, where the swing blend has barely started.
constexpr scalar_t kJustAfterLiftOff = 1.0e-7;
constexpr scalar_t kLiftOffTolerance = 1.0e-6;

/** Returns `state` with the normalized centroidal momentum of the command
 * (v_x, h_z): the linear part is the CoM velocity, the angular part the turn
 * (CentroidalMpcRobotModel::getRefPlanarMotion()). */
vector_t commandedState(const vector_t& state,
                        scalar_t cmdLinvelX,
                        scalar_t cmdMnormAngmomZ) {
  vector_t commanded = state;
  commanded(kXAxisIndex) = cmdLinvelX;
  commanded(centroidal_model::kAngularMomentumStartIndex + kZAxisIndex) =
      cmdMnormAngmomZ;
  return commanded;
}

/**
 * Makes `schedule` the reference manager's with a target that holds the
 * initial state under the command (v_x, h_z), solves the references at 0 the
 * way the MPC loop does, and points the problem at the target.
 */
void prepareCommandedSolve(CentroidalMpcInterface& mpc,
                           const ModeSchedule& schedule,
                           scalar_t cmdLinvelX,
                           scalar_t cmdMnormAngmomZ) {
  SwitchedModelReferenceManager& referenceManager =
      *mpc.getSwitchedModelReferenceManagerPtr();
  referenceManager.getGaitSchedule()->updateModeSchedule(schedule);
  const vector_t& state = mpc.getInitialState();
  const vector_t target = commandedState(state, cmdLinvelX, cmdMnormAngmomZ);
  const vector_t input = vector_t::Zero(
      static_cast<Eigen::Index>(mpc.getMpcRobotModel().getInputDim()));
  referenceManager.setTargetTrajectories(TargetTrajectories(
      {0.0, kTargetEndTime}, {target, target}, {input, input}));
  referenceManager.preSolverRun(/*initTime=*/0.0,
                                /*finalTime=*/kHorizonEndTime, state,
                                schedule.modeAtTime(0.0));
  mpc.getOptimalControlProblemRef().targetTrajectoriesPtr =
      &referenceManager.getTargetTrajectories();
}

/** Returns the reference input of `mpc` at `time`. */
vector_t desiredInputAt(CentroidalMpcInterface& mpc, scalar_t time) {
  const SwitchedModelReferenceManager& referenceManager =
      *mpc.getSwitchedModelReferenceManagerPtr();
  return referenceManager.getDesiredInput(
      referenceManager.getTargetTrajectories(), mpc.getInitialState(), time);
}

/** Returns the input row of the velocity of contact `contactIndex`'s wheel
 * joint. */
Eigen::Index wheelJointInputIndex(CentroidalMpcInterface& mpc,
                                  size_t contactIndex) {
  const ModelSettings& settings = mpc.modelSettings();
  return static_cast<Eigen::Index>(
      mpc.getMpcRobotModel().getJointVelocitiesStartindex() +
      settings.jointIndexMap.at(
          settings.pointContacts[contactIndex].parentJointName));
}

/**
 * Returns the rolling direction of contact `contactIndex`'s wheel at the
 * initial state, from Pinocchio's velocity of the wheel frame under a unit
 * velocity of the wheel joint alone: t = normalize(w x e_z).
 */
vector3_t dirWheelEwrtWorld(CentroidalMpcInterface& mpc, size_t contactIndex) {
  const ModelSettings& settings = mpc.modelSettings();
  const pinocchio::Model& model = mpc.getPinocchioInterface().getModel();
  pinocchio::Data data(model);
  vector_t jointVelocities = vector_t::Zero(model.nv);
  jointVelocities(model
                      .joints[model.getJointId(
                          settings.pointContacts[contactIndex].parentJointName)]
                      .idx_v()) = 1.0;
  pinocchio::forwardKinematics(
      model, data,
      mpc.getMpcRobotModel().getGeneralizedCoordinates(mpc.getInitialState()),
      jointVelocities);
  const vector3_t angvelWheelRtWorldEwrtWorld =
      pinocchio::getFrameVelocity(
          model, data, model.getFrameId(settings.contactNames[contactIndex]),
          pinocchio::LOCAL_WORLD_ALIGNED)
          .angular();
  return angvelWheelRtWorldEwrtWorld.cross(vector3_t::UnitZ()).normalized();
}

/** Returns where contact `contactIndex` stands relative to the base at the
 * initial state, in world axes (the heading's, the initial yaw being 0). */
vector2_t posContactRtBaseEwrtWorldXy(CentroidalMpcInterface& mpc,
                                      size_t contactIndex) {
  const vector_t& state = mpc.getInitialState();
  return (rolling_contact_test::posWheelRtWorldEwrtWorld(
              mpc, state, mpc.modelSettings().contactNames[contactIndex]) -
          mpc.getMpcRobotModel().getPosBaseRtWorldEwrtWorld(state))
      .head<kHorizontalPositionDim>();
}

/** Returns the base position at the initial state. */
vector2_t posBaseRtWorldEwrtWorldXy(CentroidalMpcInterface& mpc) {
  return mpc.getMpcRobotModel()
      .getPosBaseRtWorldEwrtWorld(mpc.getInitialState())
      .head<kHorizontalPositionDim>();
}

/** Tron1's configuration with the foot cost of every wheel active in swing and
 * in stance. */
CentroidalMpcConfig tron1FootCostConfig() {
  CentroidalMpcConfig config = rolling_contact_test::tron1Config();
  config.task.costs.emplace_back("task_space_foot_cost");
  config.task.task_space_foot_cost.active_phases = "swing_and_stance";
  return config;
}

/** The Tron1 interface, built once for the suite. */
class RollingContactReferencesTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    interface_ = checkedValue(CentroidalMpcInterface::Create(
                                  tron1FootCostConfig(), tron1Files().urdfFile))
                     .release();
  }

  static void TearDownTestSuite() {
    delete interface_;
    interface_ = nullptr;
  }

  void SetUp() override {
    ASSERT_NEAR(
        mpc().getMpcRobotModel().getYawBaseRtWorld(mpc().getInitialState()),
        0.0, kTolerance);
  }

  static CentroidalMpcInterface& mpc() { return *interface_; }

  static const SwitchedModelReferenceManager& referenceManager() {
    return *mpc().getSwitchedModelReferenceManagerPtr();
  }

 private:
  static CentroidalMpcInterface* absl_nullable interface_;
};

CentroidalMpcInterface* absl_nullable RollingContactReferencesTest::interface_ =
    nullptr;

TEST_F(RollingContactReferencesTest, aForwardCommandRollsBothWheels) {
  prepareCommandedSolve(mpc(), rolling_contact_test::bothWheelsStand(),
                        kCmdLinvelBaseRtWorldEwrtWorldX,
                        /*cmdMnormAngmomZ=*/0.0);
  const vector_t input = desiredInputAt(mpc(), kStanceTime);
  for (const size_t wheel : {kLeftWheel, kRightWheel}) {
    SCOPED_TRACE(mpc().modelSettings().contactNames[wheel]);
    const vector3_t dirRolling = dirWheelEwrtWorld(mpc(), wheel);
    // Tron1's axles are horizontal and across the heading at the nominal
    // posture, so the wheel rolls straight ahead or straight back.
    ASSERT_NEAR(std::abs(dirRolling.x()), 1.0, kTolerance);
    const scalar_t radius = mpc().modelSettings().wheelRadius(wheel);
    EXPECT_NEAR(
        input(wheelJointInputIndex(mpc(), wheel)),
        std::copysign(kCmdLinvelBaseRtWorldEwrtWorldX / radius, dirRolling.x()),
        kTolerance);
  }
}

TEST_F(RollingContactReferencesTest, aTurnInPlaceRollsTheWheelsApart) {
  prepareCommandedSolve(mpc(), rolling_contact_test::bothWheelsStand(),
                        /*cmdLinvelX=*/0.0, kCmdMnormAngmomZ);
  const scalar_t cmdRateYawBaseRtWorld =
      referenceManager().getCmdRateYawBaseRtWorld(kStanceTime);
  ASSERT_GT(std::abs(cmdRateYawBaseRtWorld), 0.0);
  const vector_t input = desiredInputAt(mpc(), kStanceTime);
  for (const size_t wheel : {kLeftWheel, kRightWheel}) {
    SCOPED_TRACE(mpc().modelSettings().contactNames[wheel]);
    // v = w e_z x p at the contact point, along the rolling direction.
    const vector2_t nomPosContactRtBaseEwrtWorldXy =
        posContactRtBaseEwrtWorldXy(mpc(), wheel);
    const vector3_t dirRolling = dirWheelEwrtWorld(mpc(), wheel);
    const scalar_t expected =
        (cmdRateYawBaseRtWorld *
         (-nomPosContactRtBaseEwrtWorldXy.y() * dirRolling.x() +
          nomPosContactRtBaseEwrtWorldXy.x() * dirRolling.y())) /
        mpc().modelSettings().wheelRadius(wheel);
    EXPECT_NEAR(input(wheelJointInputIndex(mpc(), wheel)), expected,
                kTolerance);
  }
  // The wheels sit on either side of the base: their references have opposite
  // signs (with the same joint axis on both sides) in the ratio of their half
  // tracks.
  const scalar_t left = input(wheelJointInputIndex(mpc(), kLeftWheel));
  const scalar_t right = input(wheelJointInputIndex(mpc(), kRightWheel));
  const scalar_t signs = dirWheelEwrtWorld(mpc(), kLeftWheel).x() *
                         dirWheelEwrtWorld(mpc(), kRightWheel).x();
  EXPECT_LT(left * right * signs, 0.0);
  EXPECT_NEAR(left / right,
              signs * posContactRtBaseEwrtWorldXy(mpc(), kLeftWheel).y() /
                  posContactRtBaseEwrtWorldXy(mpc(), kRightWheel).y(),
              kTolerance);
}

TEST_F(RollingContactReferencesTest, noCommandGivesNoWheelVelocity) {
  prepareCommandedSolve(mpc(), rolling_contact_test::bothWheelsStand(),
                        /*cmdLinvelX=*/0.0, /*cmdMnormAngmomZ=*/0.0);
  const vector_t input = desiredInputAt(mpc(), kStanceTime);
  EXPECT_EQ(input(wheelJointInputIndex(mpc(), kLeftWheel)), 0.0);
  EXPECT_EQ(input(wheelJointInputIndex(mpc(), kRightWheel)), 0.0);
}

TEST_F(RollingContactReferencesTest, aWheelInSwingHasNoVelocityReference) {
  prepareCommandedSolve(mpc(), rolling_contact_test::rightWheelSwing(),
                        kCmdLinvelBaseRtWorldEwrtWorldX,
                        /*cmdMnormAngmomZ=*/0.0);
  const vector_t input = desiredInputAt(mpc(), kMidSwingTime);
  EXPECT_EQ(input(wheelJointInputIndex(mpc(), kRightWheel)), 0.0);
  EXPECT_NE(input(wheelJointInputIndex(mpc(), kLeftWheel)), 0.0);
}

TEST_F(RollingContactReferencesTest, theStanceReferenceMovesWithTheBase) {
  prepareCommandedSolve(mpc(), rolling_contact_test::bothWheelsStand(),
                        kCmdLinvelBaseRtWorldEwrtWorldX,
                        /*cmdMnormAngmomZ=*/0.0);
  for (const size_t wheel : {kLeftWheel, kRightWheel}) {
    for (const scalar_t time : {kEarlyStanceTime, kLateStanceTime}) {
      SCOPED_TRACE(absl::StrCat(mpc().modelSettings().contactNames[wheel],
                                " at ", time, " s"));
      const std::optional<SwingFootReference> reference =
          referenceManager().getSwingFootReference(wheel, time);
      if (!reference.has_value()) {
        GTEST_FAIL() << "no swing reference";
      }
      // The predicted base, advanced by the command, plus the nominal offset.
      const vector2_t expected =
          posBaseRtWorldEwrtWorldXy(mpc()) +
          vector2_t(kCmdLinvelBaseRtWorldEwrtWorldX * time, 0.0) +
          posContactRtBaseEwrtWorldXy(mpc(), wheel);
      EXPECT_TRUE(
          reference->desPosFootRtWorldEwrtWorld.head<kHorizontalPositionDim>()
              .isApprox(expected, kTolerance))
          << reference->desPosFootRtWorldEwrtWorld.transpose();
      EXPECT_NEAR(reference->desPosFootRtWorldEwrtWorld.z(),
                  referenceManager().getAppliedPosSurfaceRtWorldEwrtWorldZ(),
                  kTolerance);
      EXPECT_TRUE(reference->desLinvelFootRtWorldEwrtWorld.isApprox(
          vector3_t(kCmdLinvelBaseRtWorldEwrtWorldX, 0.0, 0.0), kTolerance))
          << reference->desLinvelFootRtWorldEwrtWorld.transpose();
    }
  }
}

TEST_F(RollingContactReferencesTest, aSwingLiftsOffFromTheMovingPoint) {
  prepareCommandedSolve(mpc(), rolling_contact_test::rightWheelSwing(),
                        kCmdLinvelBaseRtWorldEwrtWorldX,
                        /*cmdMnormAngmomZ=*/0.0);
  const scalar_t time = kSwingStartTime + kJustAfterLiftOff;
  ASSERT_FALSE(referenceManager().isInContact(time, kRightWheel));
  const std::optional<SwingFootReference> reference =
      referenceManager().getSwingFootReference(kRightWheel, time);
  if (!reference.has_value()) {
    GTEST_FAIL() << "no swing reference";
  }
  const vector2_t movingPoint =
      posBaseRtWorldEwrtWorldXy(mpc()) +
      vector2_t(kCmdLinvelBaseRtWorldEwrtWorldX * kSwingStartTime, 0.0) +
      posContactRtBaseEwrtWorldXy(mpc(), kRightWheel);
  EXPECT_TRUE(
      reference->desPosFootRtWorldEwrtWorld.head<kHorizontalPositionDim>()
          .isApprox(movingPoint, kLiftOffTolerance))
      << reference->desPosFootRtWorldEwrtWorld.transpose();
  // Not where the wheel was measured at the solve.
  EXPECT_GT(
      (reference->desPosFootRtWorldEwrtWorld.head<kHorizontalPositionDim>() -
       (posBaseRtWorldEwrtWorldXy(mpc()) +
        posContactRtBaseEwrtWorldXy(mpc(), kRightWheel)))
          .norm(),
      kCmdLinvelBaseRtWorldEwrtWorldX * kSwingStartTime / 2.0);
}

TEST_F(RollingContactReferencesTest, theFootCostTracksTheMovingStancePoint) {
  prepareCommandedSolve(mpc(), rolling_contact_test::bothWheelsStand(),
                        kCmdLinvelBaseRtWorldEwrtWorldX,
                        /*cmdMnormAngmomZ=*/0.0);
  OptimalControlProblem& problem = mpc().getOptimalControlProblemRef();
  const ModelSettings& settings = mpc().modelSettings();
  const CentroidalMpcEndEffectorFootCost& cost =
      problem.costPtr->get<CentroidalMpcEndEffectorFootCost>(
          taskSpaceKinematicsCostName(settings.contactNames[kLeftWheel]));
  ASSERT_TRUE(cost.isActive(kLateStanceTime));
  const vector_t input = vector_t::Zero(
      static_cast<Eigen::Index>(mpc().getMpcRobotModel().getInputDim()));
  PreComputation& preComp = *problem.preComputationPtr;
  preComp.request(Request::Cost, kLateStanceTime, mpc().getInitialState(),
                  input);
  const vector_t parameters = cost.getParameters(
      kLateStanceTime, *problem.targetTrajectoriesPtr, preComp);
  const std::optional<SwingFootReference> reference =
      referenceManager().getSwingFootReference(kLeftWheel, kLateStanceTime);
  if (!reference.has_value()) {
    GTEST_FAIL() << "no swing reference";
  }
  EXPECT_TRUE(parameters.head<kPositionDim>().isApprox(
      reference->desPosFootRtWorldEwrtWorld, kTolerance))
      << parameters.head<kPositionDim>().transpose();
  EXPECT_GT(parameters.x(),
            posBaseRtWorldEwrtWorldXy(mpc()).x() +
                posContactRtBaseEwrtWorldXy(mpc(), kLeftWheel).x());
}

/** The legged four-point set on the DRC Atlas with the nominal_stance anchor:
 * fixed point contacts. */
CentroidalMpcConfig fixedPointConfig() {
  CentroidalMpcConfig config = four_point_set::leggedFourPointConfig(
      checkedValue(loadConfigOf(atlasFiles())));
  config.task.nominal_foothold.anchor = "nominal_stance";
  return config;
}

/** A schedule on which the front-right and hind-left contacts swing from
 * kSwingStartTime to kSwingEndTime. */
ModeSchedule diagonalSwing() {
  return ModeSchedule(
      {kSwingStartTime, rolling_contact_test::kSwingEndTime},
      {four_point_set::kAllFour, four_point_set::kFrontLeftHindRight,
       four_point_set::kAllFour});
}

TEST(FixedPointContactReferencesTest, aFixedContactDoesNotMoveInStance) {
  const std::unique_ptr<CentroidalMpcInterface> mpc =
      checkedValue(CentroidalMpcInterface::Create(fixedPointConfig(),
                                                  atlasFiles().urdfFile));
  prepareCommandedSolve(*mpc, diagonalSwing(), kCmdLinvelBaseRtWorldEwrtWorldX,
                        /*cmdMnormAngmomZ=*/0.0);
  const SwitchedModelReferenceManager& referenceManager =
      *mpc->getSwitchedModelReferenceManagerPtr();
  const ModelSettings& settings = mpc->modelSettings();
  ASSERT_FALSE(settings.hasRollingContacts());
  for (size_t contact = 0; contact < settings.contactNames.size(); ++contact) {
    SCOPED_TRACE(settings.contactNames[contact]);
    EXPECT_FALSE(referenceManager.getSwingFootReference(contact, kStanceTime)
                     .has_value());
  }
  // The swinging contacts lift off where they were measured at the solve.
  const scalar_t time = kSwingStartTime + kJustAfterLiftOff;
  for (size_t contact = 0; contact < settings.contactNames.size(); ++contact) {
    if (referenceManager.isInContact(time, contact)) continue;
    SCOPED_TRACE(settings.contactNames[contact]);
    const std::optional<SwingFootReference> reference =
        referenceManager.getSwingFootReference(contact, time);
    if (!reference.has_value()) {
      GTEST_FAIL() << "no swing reference";
    }
    const vector2_t measured =
        rolling_contact_test::posWheelRtWorldEwrtWorld(
            *mpc, mpc->getInitialState(), settings.contactNames[contact])
            .head<kHorizontalPositionDim>();
    EXPECT_TRUE(
        reference->desPosFootRtWorldEwrtWorld.head<kHorizontalPositionDim>()
            .isApprox(measured, kLiftOffTolerance))
        << reference->desPosFootRtWorldEwrtWorld.transpose();
  }
  // No joint velocity reference without rolling contacts.
  const vector_t input = desiredInputAt(*mpc, kStanceTime);
  EXPECT_TRUE(
      input
          .segment(
              static_cast<Eigen::Index>(
                  mpc->getMpcRobotModel().getJointVelocitiesStartindex()),
              static_cast<Eigen::Index>(mpc->getMpcRobotModel().getJointDim()))
          .isZero(0.0));
}

}  // namespace
}  // namespace ocs2::humanoid
