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

#include "absl/base/nullability.h"
#include "gtest/gtest.h"
#include "ocs2_core/ComputationRequest.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/constraint/NormalVelocityConstraintCppAd.h"
#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"
#include "humanoid_common_mpc/common/ContactTermNames.h"
#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "support/RollingContactTestSupport.h"
#include "support/TypedConfigFiles.h"

/**
 * A wheel of Tron1 in swing: a rolling contact is then a swinging point
 * contact whose frame is the wheel center, so
 *  - its normal-velocity servo is active, the other wheel's is not, and its
 *    residual references the wheel center a wheel radius above the swing
 *    reference of the contact point;
 *  - its swing cost (task_space_foot_cost) compares the contact point, a
 *    wheel radius below the wheel center, with that reference.
 */
namespace ocs2::humanoid {
namespace {

using rolling_contact_test::kLeftWheel;
using rolling_contact_test::kMidSwingTime;
using rolling_contact_test::kRightWheel;

constexpr scalar_t kTolerance = 1.0e-9;
// [m] The swing reference of the contact point at mid-swing is above this.
constexpr scalar_t kMinSwingClearance = 0.01;
// The index of the z position weight in EndEffectorKinematicsWeights.
constexpr Eigen::Index kPosZWeightIndex = kZAxisIndex;

/** Tron1's configuration with the swing cost of every contact. */
CentroidalMpcConfig swingConfig() {
  CentroidalMpcConfig config = rolling_contact_test::tron1Config();
  config.task.costs.emplace_back("task_space_foot_cost");
  return config;
}

/** The interface, built once for the suite, swinging the right wheel. */
class RollingContactSwingTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    interface_ = checkedValue(CentroidalMpcInterface::Create(
                                  swingConfig(), tron1Files().urdfFile))
                     .release();
  }

  static void TearDownTestSuite() {
    delete interface_;
    interface_ = nullptr;
  }

  void SetUp() override {
    rolling_contact_test::prepareSolve(mpc(),
                                       rolling_contact_test::rightWheelSwing());
  }

  static CentroidalMpcInterface& mpc() { return *interface_; }

 private:
  static CentroidalMpcInterface* absl_nullable interface_;
};

CentroidalMpcInterface* absl_nullable RollingContactSwingTest::interface_ =
    nullptr;

TEST_F(RollingContactSwingTest, theServoReferencesTheWheelCenterARadiusUp) {
  OptimalControlProblem& problem = mpc().getOptimalControlProblemRef();
  const ModelSettings& settings = mpc().modelSettings();
  const vector_t& state = mpc().getInitialState();
  const vector_t input = vector_t::Zero(
      static_cast<Eigen::Index>(mpc().getMpcRobotModel().getInputDim()));
  PreComputation& preComp = *problem.preComputationPtr;
  preComp.request(Request::Constraint, kMidSwingTime, state, input);

  const NormalVelocityConstraintCppAd& leftServo =
      problem.equalityConstraintPtr->get<NormalVelocityConstraintCppAd>(
          contact_term::name(settings.contactNames[kLeftWheel],
                             contact_term::kNormalVelocityName));
  const NormalVelocityConstraintCppAd& rightServo =
      problem.equalityConstraintPtr->get<NormalVelocityConstraintCppAd>(
          contact_term::name(settings.contactNames[kRightWheel],
                             contact_term::kNormalVelocityName));
  EXPECT_FALSE(leftServo.isActive(kMidSwingTime));
  ASSERT_TRUE(rightServo.isActive(kMidSwingTime));

  const SwingTrajectoryPlanner& swingPlanner =
      *mpc().getSwitchedModelReferenceManagerPtr()->getSwingTrajectoryPlanner();
  const scalar_t desPosContactRtWorldEwrtWorldZ =
      swingPlanner.getDesPosFootRtWorldEwrtWorldZ(kRightWheel, kMidSwingTime);
  const scalar_t desLinvelContactRtWorldEwrtWorldZ =
      swingPlanner.getDesLinvelFootRtWorldEwrtWorldZ(kRightWheel,
                                                     kMidSwingTime);
  EXPECT_GT(desPosContactRtWorldEwrtWorldZ, kMinSwingClearance);
  const scalar_t posWheelRtWorldEwrtWorldZ =
      rolling_contact_test::posWheelRtWorldEwrtWorld(
          mpc(), state, settings.contactNames[kRightWheel])
          .z();
  // At rest the wheel center does not move: the residual is the reference
  // velocity and the height error of the wheel center against the swing
  // reference of the contact point plus the wheel radius.
  const vector_t value =
      rightServo.getValue(kMidSwingTime, state, input, preComp);
  ASSERT_EQ(value.size(), 1);
  EXPECT_NEAR(
      value(0),
      -desLinvelContactRtWorldEwrtWorldZ +
          settings.footConstraintConfig.positionErrorGain_z *
              (posWheelRtWorldEwrtWorldZ - (desPosContactRtWorldEwrtWorldZ +
                                            settings.wheelRadius(kRightWheel))),
      kTolerance);
}

TEST_F(RollingContactSwingTest, theSwingCostTracksTheContactPoint) {
  OptimalControlProblem& problem = mpc().getOptimalControlProblemRef();
  const ModelSettings& settings = mpc().modelSettings();
  CentroidalMpcEndEffectorFootCost& cost =
      problem.costPtr->get<CentroidalMpcEndEffectorFootCost>(
          taskSpaceKinematicsCostName(settings.contactNames[kRightWheel]));
  ASSERT_TRUE(cost.isActive(kMidSwingTime));
  // Only the height is weighed.
  vector12_t weights = vector12_t::Zero();
  weights(kPosZWeightIndex) = 1.0;
  cost.setWeights(weights);
  cost.setSwingVelocityReferenceWeight(0.0);

  const vector_t& state = mpc().getInitialState();
  const vector_t input = vector_t::Zero(
      static_cast<Eigen::Index>(mpc().getMpcRobotModel().getInputDim()));
  PreComputation& preComp = *problem.preComputationPtr;
  preComp.request(Request::Cost, kMidSwingTime, state, input);
  const vector_t parameters = cost.getParameters(
      kMidSwingTime, *problem.targetTrajectoriesPtr, preComp);
  const scalar_t desPosContactRtWorldEwrtWorldZ = parameters(kZAxisIndex);
  EXPECT_GT(desPosContactRtWorldEwrtWorldZ, kMinSwingClearance);
  const scalar_t posContactRtWorldEwrtWorldZ =
      rolling_contact_test::posWheelRtWorldEwrtWorld(
          mpc(), state, settings.contactNames[kRightWheel])
          .z() -
      settings.wheelRadius(kRightWheel);
  const scalar_t errPosContactRtWorldEwrtWorldZ =
      posContactRtWorldEwrtWorldZ - desPosContactRtWorldEwrtWorldZ;
  EXPECT_NEAR(
      cost.getValue(kMidSwingTime, state, input, *problem.targetTrajectoriesPtr,
                    preComp),
      0.5 * errPosContactRtWorldEwrtWorldZ * errPosContactRtWorldEwrtWorldZ,
      kTolerance);
}

}  // namespace
}  // namespace ocs2::humanoid
