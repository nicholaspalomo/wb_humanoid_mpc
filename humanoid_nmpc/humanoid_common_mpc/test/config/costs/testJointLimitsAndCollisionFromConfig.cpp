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
// the file sets are carried over beside the constraint's fixed frames, and a value that is not finite is refused by its
// path.

#include "pinocchio/fwd.hpp"

#include <limits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"
#include "ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h"

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

TEST(CollisionConstraintFromConfigTest, AnEmptyBlockIsTheConstraintsDefaultConfig) {
  const absl::StatusOr<FootCollisionConstraint::Config> config =
      footCollisionConstraintConfigFromConfig(mpc_config::CollisionConstraintConfig{});
  ASSERT_TRUE(config.ok()) << config.status();
  const FootCollisionConstraint::Config defaults;
  EXPECT_EQ(config->leftAnkleFrame, defaults.leftAnkleFrame);
  EXPECT_EQ(config->rightKneeFrame, defaults.rightKneeFrame);
  EXPECT_EQ(config->footCollisionSphereRadius, defaults.footCollisionSphereRadius);
  EXPECT_EQ(config->kneeCollisionSphereRadius, defaults.kneeCollisionSphereRadius);
  const PieceWisePolynomialBarrierPenalty::Config barrierDefaults;
  const absl::StatusOr<PieceWisePolynomialBarrierPenalty::Config> barrier =
      footCollisionBarrierFromConfig(mpc_config::CollisionConstraintConfig{});
  ASSERT_TRUE(barrier.ok()) << barrier.status();
  EXPECT_EQ(barrier->mu, barrierDefaults.mu);
  EXPECT_EQ(barrier->delta, barrierDefaults.delta);
}

TEST(CollisionConstraintFromConfigTest, CarriesTheFramesAndRadiiBesideTheFixedFrames) {
  mpc_config::CollisionConstraintConfig collision;
  collision.foot.left_ankle_frame = "l_leg_akx";
  collision.foot.right_ankle_frame = "r_leg_akx";
  collision.foot.foot_collision_sphere_radius = 0.05;
  collision.knee.left_knee_frame = "l_leg_kny";
  collision.knee.right_knee_frame = "r_leg_kny";
  collision.knee.knee_collision_sphere_radius = 0.07;
  const absl::StatusOr<FootCollisionConstraint::Config> config = footCollisionConstraintConfigFromConfig(collision);
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->leftAnkleFrame, "l_leg_akx");
  EXPECT_EQ(config->rightAnkleFrame, "r_leg_akx");
  EXPECT_EQ(config->leftKneeFrame, "l_leg_kny");
  EXPECT_EQ(config->rightKneeFrame, "r_leg_kny");
  EXPECT_EQ(config->footCollisionSphereRadius, 0.05);
  EXPECT_EQ(config->kneeCollisionSphereRadius, 0.07);
  EXPECT_EQ(config->leftFootCenterFrame, FootCollisionConstraint::Config{}.leftFootCenterFrame);

  collision.knee.knee_collision_sphere_radius = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(absl::StrContains(footCollisionConstraintConfigFromConfig(collision).status().message(),
                                "collision_constraint.knee.knee_collision_sphere_radius"));
  collision.mu = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(absl::StrContains(footCollisionBarrierFromConfig(collision).status().message(), "collision_constraint.mu"));
}

}  // namespace
}  // namespace ocs2::humanoid
