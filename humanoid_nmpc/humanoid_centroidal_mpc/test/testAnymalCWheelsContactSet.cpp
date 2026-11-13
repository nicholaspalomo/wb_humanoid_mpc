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
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/multibody/fwd.hpp"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/common/ContactTermNames.h"
#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/constraint/JointLimitsSoftConstraint.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "support/TypedConfigFiles.h"

/**
 * The centroidal MPC on the wheeled ANYmal C's four rolling contacts, from the
 * shipped configuration (robot_models/anybotics_anymal_c_wheels):
 *  - the input is a force per wheel and the sixteen joint velocities;
 *  - zero_velocity builds the rolling no-slip constraint on each wheel, beside
 *    its normal-velocity servo and its friction cone, and no stance-foot
 *    constraint;
 *  - the joint-limit barrier pushes each of the twelve leg joints back from
 *    its upper limit and has no term for a wheel;
 *  - a forward command gives every wheel in stance the reference velocity
 *    v / r.
 */
namespace ocs2::humanoid {
namespace {

using ::testing::ElementsAre;
using ::testing::IsSupersetOf;

constexpr size_t kNumWheels = 4;
constexpr size_t kNumJoints = 16;
constexpr size_t kNumLegJoints = 12;
// Every wheel in contact: the mode of the four mode bits LF, RF, LH, RH.
constexpr size_t kAllWheelsMode = 15;
constexpr scalar_t kTolerance = 1.0e-9;
// [rad] Wheel angles the joint-limit barrier must not see.
constexpr scalar_t kWheelTurns = 100.0;
// [m/s] The forward command.
constexpr scalar_t kCmdLinvelBaseRtWorldEwrtWorldX = 0.5;
// [s] The end of the target, the horizon the reference manager prepares and a
// time in its stance.
constexpr scalar_t kTargetEndTime = 2.0;
constexpr scalar_t kHorizonEndTime = 1.0;
constexpr scalar_t kStanceTime = 0.1;

/** The interface, built once for the suite. */
class AnymalCWheelsContactSetTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    interface_ =
        checkedValue(CentroidalMpcInterface::Create(
                         checkedValue(loadConfigOf(anymalCWheelsFiles())),
                         anymalCWheelsFiles().urdfFile))
            .release();
  }

  static void TearDownTestSuite() {
    delete interface_;
    interface_ = nullptr;
  }

  static CentroidalMpcInterface& mpc() { return *interface_; }

 private:
  static CentroidalMpcInterface* absl_nullable interface_;
};

CentroidalMpcInterface* absl_nullable AnymalCWheelsContactSetTest::interface_ =
    nullptr;

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

TEST_F(AnymalCWheelsContactSetTest,
       theInputIsAForcePerWheelAndTheJointVelocities) {
  const ModelSettings& settings = mpc().modelSettings();
  EXPECT_THAT(settings.contactNames,
              ElementsAre("LF_wheel_contact", "RF_wheel_contact",
                          "LH_wheel_contact", "RH_wheel_contact"));
  for (size_t i = 0; i < kNumWheels; ++i) {
    EXPECT_TRUE(settings.isRollingContact(i)) << i;
  }
  EXPECT_THAT(settings.rollingContactJointNames(),
              ElementsAre("LF_wheel", "RF_wheel", "LH_wheel", "RH_wheel"));
  EXPECT_EQ(mpc().getMpcRobotModel().getInputDim(),
            kNumWheels * kPositionDim + kNumJoints);
  EXPECT_EQ(mpc().getMpcRobotModel().getJointDim(), kNumJoints);
}

TEST_F(AnymalCWheelsContactSetTest, eachWheelHasTheRollingNoSlipAndSwingTerms) {
  const std::vector<std::string> equalities =
      mpc().getEqualityConstraintNames();
  const std::vector<std::string> softConstraints =
      mpc().getSoftConstraintNames();
  for (const std::string& name : mpc().modelSettings().contactNames) {
    SCOPED_TRACE(name);
    EXPECT_EQ(
        std::count(equalities.begin(), equalities.end(),
                   contact_term::name(name, contact_term::kRollingNoSlipName)),
        1);
    EXPECT_THAT(equalities,
                IsSupersetOf({contact_term::name(
                                  name, contact_term::kNormalVelocityName),
                              absl::StrCat(name, "_zeroWrench")}));
    EXPECT_EQ(std::count(equalities.begin(), equalities.end(),
                         zeroVelocityTermName(name)),
              0)
        << "a wheel has no stance-foot constraint on its center";
    EXPECT_THAT(softConstraints, IsSupersetOf({contact_term::name(
                                     name, contact_term::kFrictionForceCone)}));
  }
}

TEST_F(AnymalCWheelsContactSetTest, theJointLimitsHaveARowPerLegJointOnly) {
  const OptimalControlProblem& problem = mpc().getOptimalControlProblemRef();
  const JointLimitsSoftConstraint& jointLimits =
      problem.stateSoftConstraintPtr->get<JointLimitsSoftConstraint>(
          kJointLimitsTerm);
  const ModelSettings& settings = mpc().modelSettings();
  const vector_t standing =
      mpc().getMpcRobotModel().getJointAngles(mpc().getInitialState());
  vector_t turned = standing;
  absl::flat_hash_set<Eigen::Index> wheelJoints;
  for (const std::string& wheelJoint : settings.rollingContactJointNames()) {
    const Eigen::Index joint =
        static_cast<Eigen::Index>(settings.jointIndexMap.at(wheelJoint));
    wheelJoints.insert(joint);
    turned(joint) = kWheelTurns;
  }
  ASSERT_EQ(wheelJoints.size(), kNumWheels);
  EXPECT_EQ(jointLimits.getValue(turned), jointLimits.getValue(standing));
  const ScalarFunctionQuadraticApproximation approximation =
      jointLimits.getQuadraticApproximation(turned);
  const pinocchio::Model& model = mpc().getPinocchioInterface().getModel();
  // The approximation is over the whole state; the joints start at this row.
  const Eigen::Index jointStart =
      static_cast<Eigen::Index>(mpc().getMpcRobotModel().getJointStartindex());
  size_t numBarrierJoints = 0;
  for (Eigen::Index joint = 0; joint < static_cast<Eigen::Index>(kNumJoints);
       ++joint) {
    const std::string& name =
        settings.mpcModelJointNames[static_cast<size_t>(joint)];
    SCOPED_TRACE(name);
    const Eigen::Index row = jointStart + joint;
    if (wheelJoints.contains(joint)) {
      EXPECT_EQ(approximation.dfdx(row), 0.0);
      EXPECT_EQ(approximation.dfdxx(row, row), 0.0);
      continue;
    }
    // A leg joint at its upper limit is inside the barrier's band.
    const Eigen::Index configurationIndex =
        model.joints[model.getJointId(name)].idx_q();
    const double upperLimit =
        model
            .upperPositionLimit(  // NOLINT(frame-naming): Pinocchio's joint-space limit vector
                configurationIndex);
    ASSERT_TRUE(std::isfinite(upperLimit));
    vector_t atLimit = standing;
    atLimit(joint) = upperLimit;
    EXPECT_GT(jointLimits.getQuadraticApproximation(atLimit).dfdx(row), 0.0);
    ++numBarrierJoints;
  }
  EXPECT_EQ(numBarrierJoints, kNumLegJoints);
}

TEST_F(AnymalCWheelsContactSetTest, aForwardCommandRollsEveryWheel) {
  SwitchedModelReferenceManager& referenceManager =
      *mpc().getSwitchedModelReferenceManagerPtr();
  const ModeSchedule schedule({kTargetEndTime},
                              {kAllWheelsMode, kAllWheelsMode});
  referenceManager.getGaitSchedule()->updateModeSchedule(schedule);
  const vector_t& state = mpc().getInitialState();
  vector_t target = state;
  // The normalized linear momentum of the command is the CoM velocity.
  target(kXAxisIndex) = kCmdLinvelBaseRtWorldEwrtWorldX;
  const vector_t zeroInput = vector_t::Zero(
      static_cast<Eigen::Index>(mpc().getMpcRobotModel().getInputDim()));
  referenceManager.setTargetTrajectories(TargetTrajectories(
      {0.0, kTargetEndTime}, {target, target}, {zeroInput, zeroInput}));
  referenceManager.preSolverRun(/*initTime=*/0.0,
                                /*finalTime=*/kHorizonEndTime, state,
                                schedule.modeAtTime(0.0));
  const vector_t input = referenceManager.getDesiredInput(
      referenceManager.getTargetTrajectories(), state, kStanceTime);
  for (size_t wheel = 0; wheel < kNumWheels; ++wheel) {
    SCOPED_TRACE(mpc().modelSettings().contactNames[wheel]);
    // Every axle is the base's +y axis at the default posture, so a positive
    // wheel velocity rolls the wheel forward: t = e_y x e_z = e_x.
    EXPECT_NEAR(input(wheelJointInputIndex(mpc(), wheel)),
                kCmdLinvelBaseRtWorldEwrtWorldX /
                    mpc().modelSettings().wheelRadius(wheel),
                kTolerance);
  }
}

}  // namespace
}  // namespace ocs2::humanoid
