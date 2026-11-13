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

// StateEstimatorRegistry and the two estimators behind it: an unknown name
// lists the names, ground_truth hands the measured state itself on, and kalman,
// fed the states a backend without a position sensor publishes (zero base
// position and velocity), hands on the base of the walk.

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
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_state_estimation/humanoid_state_estimator/test/MeasuredState.h"
#include "humanoid_state_estimation/humanoid_state_estimator/test/WalkingTrajectory.h"
#include "humanoid_state_estimator/ContactEstimationInputs.h"
#include "humanoid_state_estimator/ContactKinematics.h"
#include "humanoid_state_estimator/KalmanBaseEstimator.h"
#include "humanoid_state_estimator/StateEstimator.h"
#include "humanoid_state_estimator/StateEstimatorRegistry.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid::estimation {
namespace {

using test_support::WalkingSample;
using test_support::walkingSample;

constexpr scalar_t kDt = 0.002;

/** The prismatic biped's feet, as contact points at their links' origins. */
std::vector<ContactPointSpec> footContacts() {
  std::vector<ContactPointSpec> contacts;
  for (const std::string& link : test_support::prismaticBipedFootLinks()) {
    contacts.push_back(ContactPointSpec{.linkName = link});
  }
  return contacts;
}

robot::model::RobotDescription bipedDescription() {
  absl::StatusOr<robot::model::RobotDescription> description =
      robot::model::RobotDescription::Create(test_support::kPrismaticBipedUrdf);
  EXPECT_TRUE(description.ok()) << description.status();
  return *std::move(description);
}

StateEstimatorContext contextOf(
    const robot::model::RobotDescription& description) {
  return StateEstimatorContext{.urdfFile = test_support::kPrismaticBipedUrdf,
                               .contacts = footContacts(),
                               .description = &description,
                               .kalman = KalmanBaseEstimator::Config()};
}

TEST(StateEstimatorRegistryTest, NamesItsEstimatorsAndRefusesAnUnknownOne) {
  const robot::model::RobotDescription description = bipedDescription();
  const StateEstimatorRegistry registry;
  EXPECT_EQ(registry.names(),
            (std::vector<std::string>{"ground_truth", "kalman"}));
  EXPECT_TRUE(registry.has("kalman"));
  EXPECT_FALSE(registry.has("particle"));
  const absl::StatusOr<std::unique_ptr<StateEstimator>> unknown =
      registry.create("particle", contextOf(description));
  ASSERT_EQ(unknown.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(unknown.status().message(), "particle"))
      << unknown.status();
  EXPECT_TRUE(absl::StrContains(unknown.status().message(), "ground_truth ("))
      << unknown.status();
  EXPECT_TRUE(absl::StrContains(unknown.status().message(), "kalman ("))
      << unknown.status();
}

TEST(StateEstimatorRegistryTest, TheGroundTruthHandsTheMeasuredStateItselfOn) {
  const robot::model::RobotDescription description = bipedDescription();
  absl::StatusOr<std::unique_ptr<StateEstimator>> estimator =
      StateEstimatorRegistry().create("ground_truth", contextOf(description));
  ASSERT_TRUE(estimator.ok()) << estimator.status();
  EXPECT_EQ((*estimator)->name(), "ground_truth");
  robot::model::RobotState measured(description);
  (*estimator)->reset(measured);
  const ContactEstimationInputs contactInputs(test_support::kNumFeet);
  EXPECT_EQ(&(*estimator)->estimate(measured, contactInputs), &measured);
  EXPECT_TRUE((*estimator)->estPosContactsRtWorldEwrtWorld().empty());
}

TEST(StateEstimatorRegistryTest,
     KalmanRefusesAContactLinkTheUrdfDoesNotHaveAndABadNoiseModel) {
  const robot::model::RobotDescription description = bipedDescription();
  StateEstimatorContext context = contextOf(description);
  EXPECT_TRUE(StateEstimatorRegistry().create("kalman", context).ok());
  context.contacts[0].linkName = "foot_l_contact";
  const absl::StatusOr<std::unique_ptr<StateEstimator>> noLink =
      StateEstimatorRegistry().create("kalman", context);
  EXPECT_EQ(noLink.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(noLink.status().message(), "foot_l_contact"))
      << noLink.status();
  context = contextOf(description);
  context.kalman.imuAccelerationNoise = 0.0;
  const absl::StatusOr<std::unique_ptr<StateEstimator>> refused =
      StateEstimatorRegistry().create("kalman", context);
  EXPECT_TRUE(
      absl::StrContains(refused.status().message(), "imu_acceleration_noise"))
      << refused.status();
}

TEST(StateEstimatorRegistryTest,
     KalmanRefusesARollingContactAndTheGroundTruthTakesIt) {
  // A wheel's lowest point is no point of its link: the filter would treat the
  // wheel center as a foot at rest on the ground.
  const robot::model::RobotDescription description = bipedDescription();
  StateEstimatorContext context = contextOf(description);
  context.contacts[1].kind = ContactPointKind::kRolling;
  context.contacts[1].wheelRadius = 0.127;
  context.contacts[1].contactName = "wheel_R_contact";
  const absl::StatusOr<std::unique_ptr<StateEstimator>> refused =
      StateEstimatorRegistry().create("kalman", context);
  ASSERT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.status().message(), "wheel_R_contact"))
      << refused.status();
  EXPECT_TRUE(absl::StrContains(refused.status().message(),
                                "the kalman state estimator models contact "
                                "points fixed on their links only"))
      << refused.status();
  EXPECT_TRUE(StateEstimatorRegistry().create("ground_truth", context).ok());
}

TEST(StateEstimatorRegistryTest,
     KalmanEstimatesTheBaseABackendWithoutAPositionSensorDoesNotReport) {
  const robot::model::RobotDescription description = bipedDescription();
  absl::StatusOr<std::unique_ptr<StateEstimator>> created =
      StateEstimatorRegistry().create("kalman", contextOf(description));
  ASSERT_TRUE(created.ok()) << created.status();
  StateEstimator& estimator = **created;
  EXPECT_EQ(estimator.name(), "kalman");

  // The start: the truth, as a simulator would give it, from which the
  // estimator takes over.
  robot::model::RobotState measured(description);
  const WalkingSample start = walkingSample(0.0);
  test_support::writeMeasuredState(start, measured);
  measured.setPosBaseRtWorldEwrtWorld(start.posBaseRtWorldEwrtWorld);
  measured.setLinvelBaseRtWorldEwrtBase(start.quatBaseRtWorld.inverse() *
                                        start.linvelBaseRtWorldEwrtWorld);
  estimator.reset(measured);
  const ContactEstimationInputs contactInputs(test_support::kNumFeet);

  scalar_t maxErrPosBaseRtWorldEwrtWorld = 0.0;
  scalar_t maxErrLinvelBaseRtWorldEwrtBase = 0.0;
  for (int step = 1; step <= 2000; ++step) {
    const WalkingSample sample = walkingSample(step * kDt);
    test_support::writeMeasuredState(sample, measured);
    const robot::model::RobotState& estimated =
        estimator.estimate(measured, contactInputs);
    ASSERT_NE(&estimated, &measured);
    maxErrPosBaseRtWorldEwrtWorld = std::max(
        maxErrPosBaseRtWorldEwrtWorld, (estimated.getPosBaseRtWorldEwrtWorld() -
                                        sample.posBaseRtWorldEwrtWorld)
                                           .norm());
    // The controller's velocity is in the base frame.
    const vector3_t trueLinvelBaseRtWorldEwrtBase =
        sample.quatBaseRtWorld.inverse() * sample.linvelBaseRtWorldEwrtWorld;
    maxErrLinvelBaseRtWorldEwrtBase =
        std::max(maxErrLinvelBaseRtWorldEwrtBase,
                 (estimated.getLinvelBaseRtWorldEwrtBase() -
                  trueLinvelBaseRtWorldEwrtBase)
                     .norm());
    // Everything else is the backend's.
    EXPECT_EQ(estimated.getJointPosition(0), measured.getJointPosition(0));
    EXPECT_EQ(estimated.getTime(), measured.getTime());
  }
  // As KalmanBaseEstimatorTest.TracksTheWalkWithExactSensors: the sensors are
  // exact.
  EXPECT_LT(maxErrPosBaseRtWorldEwrtWorld, 1.0e-3);
  EXPECT_LT(maxErrLinvelBaseRtWorldEwrtBase, 5.0e-3);
  EXPECT_EQ(estimator.failedUpdates(), 0u);
  // The contact positions the robot process looks surfaces up with: the
  // walk's feet.
  ASSERT_EQ(estimator.estPosContactsRtWorldEwrtWorld().size(),
            test_support::kNumFeet);
  const WalkingSample last = walkingSample(2000 * kDt);
  for (size_t foot = 0; foot < test_support::kNumFeet; ++foot) {
    EXPECT_LT((estimator.estPosContactsRtWorldEwrtWorld()[foot] -
               last.posFeetRtWorldEwrtWorld[foot])
                  .norm(),
              2.0e-3)
        << "foot " << foot;
  }

  // A state that does not advance in time is handed on with the previous
  // estimate; a non-finite one is counted.
  const vector3_t posBaseRtWorldEwrtWorld =
      estimator.estimate(measured, contactInputs).getPosBaseRtWorldEwrtWorld();
  EXPECT_EQ(
      estimator.estimate(measured, contactInputs).getPosBaseRtWorldEwrtWorld(),
      posBaseRtWorldEwrtWorld);
  measured.setTime(measured.getTime() + kDt);
  measured.setSpecforceBaseRtWorldEwrtBase(vector3_t::Constant(std::nan("")));
  EXPECT_TRUE(estimator.estimate(measured, contactInputs)
                  .getPosBaseRtWorldEwrtWorld()
                  .allFinite());
  EXPECT_EQ(estimator.failedUpdates(), 1u);
  // Contact inputs of another number of contacts are counted too.
  measured.setTime(measured.getTime() + kDt);
  measured.setSpecforceBaseRtWorldEwrtBase(last.specforceBaseRtWorldEwrtBase);
  estimator.estimate(measured,
                     ContactEstimationInputs(test_support::kNumFeet + 1));
  EXPECT_EQ(estimator.failedUpdates(), 2u);
}

}  // namespace
}  // namespace ocs2::humanoid::estimation
