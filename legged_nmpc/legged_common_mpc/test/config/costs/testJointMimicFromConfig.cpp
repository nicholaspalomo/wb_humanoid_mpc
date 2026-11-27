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

// The conversion of the knee mimic joints (JointMimicFromConfig.h): the left
// knee first, every value carried over, a value that is not finite refused by
// its path, and the centroidal variant refusing the velocity gain it has no use
// for, and both refusing a contact set that is not two feet.

#include <array>
#include <cstddef>
#include <limits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"

#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/config/costs/JointMimicFromConfig.h"
#include "legged_mpc_config/mimic_joints_config.nproto.h"

namespace ocs2::legged {
namespace {

// The biped's contact count, which the knee blocks are written for.
constexpr size_t kTwoFeet = static_cast<size_t>(kBipedFootCount);
// A quadruped-like set of four point contacts.
constexpr size_t kFourContacts = 4;

/** Knee mimic joints whose two legs differ in every value. */
mpc_config::MimicJointsConfig kneeMimicJoints() {
  mpc_config::MimicJointsConfig mimicJoints;
  mimicJoints.left_knee.parent_joint_name = "l_knee";
  mimicJoints.left_knee.child_joint_name = "l_knee_mimic";
  mimicJoints.left_knee.multiplier = 0.5;
  mimicJoints.left_knee.position_gain = 2.0;
  mimicJoints.right_knee.parent_joint_name = "r_knee";
  mimicJoints.right_knee.child_joint_name = "r_knee_mimic";
  mimicJoints.right_knee.multiplier = -1.0;
  mimicJoints.right_knee.position_gain = 3.0;
  return mimicJoints;
}

TEST(JointMimicFromConfigTest, TheLegsComeInTheOrderOfTheContacts) {
  mpc_config::MimicJointsConfig mimicJoints = kneeMimicJoints();
  mimicJoints.right_knee.velocity_gain = 0.25;
  const absl::StatusOr<BipedFeetArray<JointMimicSettings>> joints =
      kneeMimicJointsFromConfig(mimicJoints, kTwoFeet);
  ASSERT_TRUE(joints.ok()) << joints.status();
  EXPECT_EQ((*joints)[0].parentJointName, "l_knee");
  EXPECT_EQ((*joints)[0].childJointName, "l_knee_mimic");
  EXPECT_EQ((*joints)[0].multiplier, 0.5);
  EXPECT_EQ((*joints)[0].positionGain, 2.0);
  EXPECT_EQ((*joints)[0].velocityGain, 0.0);
  EXPECT_EQ((*joints)[1].parentJointName, "r_knee");
  EXPECT_EQ((*joints)[1].multiplier, -1.0);
  EXPECT_EQ((*joints)[1].velocityGain, 0.25);
}

TEST(JointMimicFromConfigTest, AnEmptyBlockIsEmptyNamesAndZeros) {
  const absl::StatusOr<BipedFeetArray<JointMimicSettings>> joints =
      kneeMimicJointsFromConfig(mpc_config::MimicJointsConfig{}, kTwoFeet);
  ASSERT_TRUE(joints.ok()) << joints.status();
  for (const JointMimicSettings& joint : *joints) {
    EXPECT_TRUE(joint.parentJointName.empty());
    EXPECT_EQ(joint.multiplier, 0.0);
    EXPECT_EQ(joint.positionGain, 0.0);
  }
}

TEST(JointMimicFromConfigTest, RefusesANonFiniteValueByItsPath) {
  mpc_config::MimicJointsConfig mimicJoints = kneeMimicJoints();
  mimicJoints.right_knee.position_gain =
      std::numeric_limits<double>::infinity();
  const absl::StatusOr<BipedFeetArray<JointMimicSettings>> refused =
      kneeMimicJointsFromConfig(mimicJoints, kTwoFeet);
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.status().message(),
                                "mimic_joints.right_knee.position_gain"))
      << refused.status();
}

TEST(JointMimicFromConfigTest, TheCentroidalVariantRefusesAVelocityGain) {
  mpc_config::MimicJointsConfig mimicJoints = kneeMimicJoints();
  EXPECT_TRUE(kneeMimicKinematicJointsFromConfig(mimicJoints, kTwoFeet).ok());
  mimicJoints.left_knee.velocity_gain = 1.0;
  const absl::StatusOr<BipedFeetArray<JointMimicSettings>> refused =
      kneeMimicKinematicJointsFromConfig(mimicJoints, kTwoFeet);
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.status().message(),
                                "mimic_joints.left_knee.velocity_gain"))
      << refused.status();
}

TEST(JointMimicFromConfigTest, BothVariantsRefuseAContactSetThatIsNotTwoFeet) {
  const absl::StatusOr<BipedFeetArray<JointMimicSettings>> dynamic =
      kneeMimicJointsFromConfig(kneeMimicJoints(), kFourContacts);
  const absl::StatusOr<BipedFeetArray<JointMimicSettings>> kinematic =
      kneeMimicKinematicJointsFromConfig(kneeMimicJoints(), kFourContacts);
  const std::array<absl::Status, 2> statuses = {dynamic.status(),
                                                kinematic.status()};
  for (const absl::Status& status : statuses) {
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(absl::StrContains(status.message(), "knee joint mimic"))
        << status;
    EXPECT_TRUE(
        absl::StrContains(status.message(), "the contact set has 4 contacts"))
        << status;
  }
}

}  // namespace
}  // namespace ocs2::legged
