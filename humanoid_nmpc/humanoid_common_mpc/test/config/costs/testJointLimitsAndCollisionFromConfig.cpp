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

// The conversions of the joint limits and the foot collision constraint (JointLimitsFromConfig.h,
// CollisionConstraintFromConfig.h): an absent barrier is PieceWisePolynomialBarrierPenalty's own, the frames and radii
// the file sets are carried over beside the constraint's fixed frames, a value out of its range (a barrier's mu and
// delta and a sphere's radius are positive) is refused by its path, and so is a frame the model does not have.

#include "pinocchio/fwd.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/config/costs/CollisionConstraintFromConfig.h"
#include "humanoid_common_mpc/config/costs/JointLimitsFromConfig.h"
#include "humanoid_common_mpc/constraint/FootCollisionConstraint.h"
#include "humanoid_mpc_config/collision_constraint_config.nproto.h"
#include "humanoid_mpc_config/joint_limits_config.nproto.h"

namespace ocs2::humanoid {
namespace {

TEST(JointLimitsFromConfigTest, AnAbsentBarrierIsThePenaltysDefault) {
  const PieceWisePolynomialBarrierPenalty::Config defaults;
  const absl::StatusOr<PieceWisePolynomialBarrierPenalty::Config> barrier = jointLimitsBarrierFromConfig(mpc_config::JointLimitsConfig{});
  ASSERT_TRUE(barrier.ok()) << barrier.status();
  EXPECT_EQ(barrier->mu, defaults.mu);
  EXPECT_EQ(barrier->delta, defaults.delta);
}

TEST(JointLimitsFromConfigTest, CarriesTheBarrierAndRefusesANonFiniteOne) {
  mpc_config::JointLimitsConfig jointLimits;
  jointLimits.mu = 1200.0;
  jointLimits.delta = 0.1;
  const absl::StatusOr<PieceWisePolynomialBarrierPenalty::Config> barrier = jointLimitsBarrierFromConfig(jointLimits);
  ASSERT_TRUE(barrier.ok()) << barrier.status();
  EXPECT_EQ(barrier->mu, 1200.0);
  EXPECT_EQ(barrier->delta, 0.1);
  jointLimits.delta = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(absl::StrContains(jointLimitsBarrierFromConfig(jointLimits).status().message(), "joint_limits.delta"));
}

TEST(JointLimitsFromConfigTest, ANonPositiveOrNanBarrierIsRefused) {
  for (const double bad : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN()}) {
    mpc_config::JointLimitsConfig mu;
    mu.mu = bad;
    EXPECT_TRUE(absl::StrContains(jointLimitsBarrierFromConfig(mu).status().message(), "joint_limits.mu: ")) << bad;
    mpc_config::JointLimitsConfig delta;
    delta.delta = bad;
    EXPECT_TRUE(absl::StrContains(jointLimitsBarrierFromConfig(delta).status().message(), "joint_limits.delta: ")) << bad;
  }
}

/** A collision block that names every frame and both radii. */
mpc_config::CollisionConstraintConfig completeCollision() {
  mpc_config::CollisionConstraintConfig collision;
  collision.foot.left_ankle_frame = "l_leg_akx";
  collision.foot.right_ankle_frame = "r_leg_akx";
  collision.foot.foot_collision_sphere_radius = 0.05;
  collision.knee.left_knee_frame = "l_leg_kny";
  collision.knee.right_knee_frame = "r_leg_kny";
  collision.knee.knee_collision_sphere_radius = 0.07;
  return collision;
}

TEST(CollisionConstraintFromConfigTest, AnEmptyBlockIsRefusedAndAnAbsentBarrierIsThePenaltysDefault) {
  // The empty block has radii of 0, which keep nothing apart, and no ankle or knee frames.
  EXPECT_EQ(footCollisionConstraintConfigFromConfig(mpc_config::CollisionConstraintConfig{}).status().code(),
            absl::StatusCode::kInvalidArgument);
  const PieceWisePolynomialBarrierPenalty::Config barrierDefaults;
  const absl::StatusOr<PieceWisePolynomialBarrierPenalty::Config> barrier =
      footCollisionBarrierFromConfig(mpc_config::CollisionConstraintConfig{});
  ASSERT_TRUE(barrier.ok()) << barrier.status();
  EXPECT_EQ(barrier->mu, barrierDefaults.mu);
  EXPECT_EQ(barrier->delta, barrierDefaults.delta);
}

TEST(CollisionConstraintFromConfigTest, ANonPositiveOrNanRadiusOrBarrierAndAnEmptyFrameAreRefused) {
  ASSERT_TRUE(footCollisionConstraintConfigFromConfig(completeCollision()).ok());
  for (const double bad : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN()}) {
    mpc_config::CollisionConstraintConfig foot = completeCollision();
    foot.foot.foot_collision_sphere_radius = bad;
    EXPECT_TRUE(absl::StrContains(footCollisionConstraintConfigFromConfig(foot).status().message(),
                                  "collision_constraint.foot.foot_collision_sphere_radius: "))
        << bad;
    mpc_config::CollisionConstraintConfig knee = completeCollision();
    knee.knee.knee_collision_sphere_radius = bad;
    EXPECT_TRUE(absl::StrContains(footCollisionConstraintConfigFromConfig(knee).status().message(),
                                  "collision_constraint.knee.knee_collision_sphere_radius: "))
        << bad;
    mpc_config::CollisionConstraintConfig mu = completeCollision();
    mu.mu = bad;
    EXPECT_TRUE(absl::StrContains(footCollisionBarrierFromConfig(mu).status().message(), "collision_constraint.mu: ")) << bad;
    mpc_config::CollisionConstraintConfig delta = completeCollision();
    delta.delta = bad;
    EXPECT_TRUE(absl::StrContains(footCollisionBarrierFromConfig(delta).status().message(), "collision_constraint.delta: ")) << bad;
  }
  mpc_config::CollisionConstraintConfig unnamed = completeCollision();
  unnamed.knee.right_knee_frame.clear();
  EXPECT_TRUE(absl::StrContains(footCollisionConstraintConfigFromConfig(unnamed).status().message(),
                                "collision_constraint.knee.right_knee_frame is empty"));
}

TEST(CollisionConstraintFromConfigTest, EveryFrameMustBeAFrameOfTheModel) {
  const absl::StatusOr<FootCollisionConstraint::Config> config = footCollisionConstraintConfigFromConfig(completeCollision());
  ASSERT_TRUE(config.ok()) << config.status();
  const std::vector<std::string> frames = {
      config->leftAnkleFrame,  config->rightAnkleFrame, config->leftFootCenterFrame, config->rightFootCenterFrame, config->leftFootFrame1,
      config->rightFootFrame1, config->leftFootFrame2,  config->rightFootFrame2,     config->leftKneeFrame,        config->rightKneeFrame};
  // A model with every frame of the constraint passes; leaving out any one of them is refused, naming it.
  for (size_t left = 0; left <= frames.size(); ++left) {
    PinocchioInterface::Model model;
    for (size_t i = 0; i < frames.size(); ++i) {
      if (i == left) continue;
      model.addFrame(pinocchio::Frame(frames[i], /*parentJoint=*/0, /*parentFrame=*/0, pinocchio::SE3::Identity(), pinocchio::OP_FRAME));
    }
    const absl::Status status = checkFootCollisionFramesExist(*config, model);
    if (left == frames.size()) {
      EXPECT_TRUE(status.ok()) << status;
    } else {
      EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << frames[left];
      EXPECT_TRUE(absl::StrContains(status.message(), absl::StrCat("'", frames[left], "'"))) << status;
    }
  }
}

TEST(CollisionConstraintFromConfigTest, CarriesTheFramesAndRadiiBesideTheFixedFrames) {
  const absl::StatusOr<FootCollisionConstraint::Config> config = footCollisionConstraintConfigFromConfig(completeCollision());
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->leftAnkleFrame, "l_leg_akx");
  EXPECT_EQ(config->rightAnkleFrame, "r_leg_akx");
  EXPECT_EQ(config->leftKneeFrame, "l_leg_kny");
  EXPECT_EQ(config->rightKneeFrame, "r_leg_kny");
  EXPECT_EQ(config->footCollisionSphereRadius, 0.05);
  EXPECT_EQ(config->kneeCollisionSphereRadius, 0.07);
  EXPECT_EQ(config->leftFootCenterFrame, FootCollisionConstraint::Config{}.leftFootCenterFrame);
}

}  // namespace
}  // namespace ocs2::humanoid
