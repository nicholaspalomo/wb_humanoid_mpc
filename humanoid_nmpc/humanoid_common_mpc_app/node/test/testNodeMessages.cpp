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

#include <array>
#include <cmath>
#include <limits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/command/WalkingVelocityCommand.h"
#include "humanoid_common_mpc/contact_planning/TargetContactPose.h"
#include "humanoid_common_mpc_app/node/ViewerAnnotations.h"
#include "humanoid_common_mpc_app/node/WalkingVelocityCommandConversions.h"
#include "humanoid_mpc_msgs/viewer_annotations.pb.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"

/*
 * The messages the MPC node converts: the operator's walking command, clamped
 * to the normalized ranges the motion manager scales and refused when a value
 * is not finite, and the annotations sent with every policy.
 */

namespace ocs2::humanoid::node {
namespace {

humanoid_mpc_msgs::WalkingVelocityCommand commandMessage(double vx,
                                                         double vy,
                                                         double height,
                                                         double yawRate) {
  humanoid_mpc_msgs::WalkingVelocityCommand message;
  message.set_cmd_linvel_base_rt_world_ewrt_heading_x(vx);
  message.set_cmd_linvel_base_rt_world_ewrt_heading_y(vy);
  message.set_cmd_pos_base_rt_world_ewrt_world_z(height);
  message.set_cmd_angvel_base_rt_world_ewrt_world_z(yawRate);
  return message;
}

TEST(WalkingVelocityCommandFromProto, PassesACommandInsideTheRangesThrough) {
  const absl::StatusOr<WalkingVelocityCommand> command =
      walkingVelocityCommandFromProto(commandMessage(
          /*vx=*/0.5, /*vy=*/-0.25, /*height=*/0.8, /*yawRate=*/0.125));
  ASSERT_TRUE(command.ok()) << command.status();
  EXPECT_EQ(command->cmd_linvel_base_rt_world_ewrt_heading_x, 0.5);
  EXPECT_EQ(command->cmd_linvel_base_rt_world_ewrt_heading_y, -0.25);
  EXPECT_EQ(command->cmd_pos_base_rt_world_ewrt_world_z, 0.8);
  EXPECT_EQ(command->cmd_angvel_base_rt_world_ewrt_world_z, 0.125);
}

TEST(WalkingVelocityCommandFromProto, ClampsToTheNormalizedRanges) {
  const absl::StatusOr<WalkingVelocityCommand> high =
      walkingVelocityCommandFromProto(commandMessage(
          /*vx=*/3.0, /*vy=*/2.0, /*height=*/5.0, /*yawRate=*/7.0));
  ASSERT_TRUE(high.ok()) << high.status();
  EXPECT_EQ(high->cmd_linvel_base_rt_world_ewrt_heading_x,
            kMaxNormalizedVelocity);
  EXPECT_EQ(high->cmd_linvel_base_rt_world_ewrt_heading_y,
            kMaxNormalizedVelocity);
  EXPECT_EQ(high->cmd_pos_base_rt_world_ewrt_world_z, kMaxPelvisHeight);
  EXPECT_EQ(high->cmd_angvel_base_rt_world_ewrt_world_z,
            kMaxNormalizedVelocity);

  const absl::StatusOr<WalkingVelocityCommand> low =
      walkingVelocityCommandFromProto(commandMessage(
          /*vx=*/-3.0, /*vy=*/-2.0, /*height=*/0.0, /*yawRate=*/-7.0));
  ASSERT_TRUE(low.ok()) << low.status();
  EXPECT_EQ(low->cmd_linvel_base_rt_world_ewrt_heading_x,
            -kMaxNormalizedVelocity);
  EXPECT_EQ(low->cmd_linvel_base_rt_world_ewrt_heading_y,
            -kMaxNormalizedVelocity);
  EXPECT_EQ(low->cmd_pos_base_rt_world_ewrt_world_z, kMinPelvisHeight);
  EXPECT_EQ(low->cmd_angvel_base_rt_world_ewrt_world_z,
            -kMaxNormalizedVelocity);
}

TEST(WalkingVelocityCommandFromProto,
     RefusesAValueThatIsNotFiniteNamingTheField) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  const absl::StatusOr<WalkingVelocityCommand> notANumber =
      walkingVelocityCommandFromProto(commandMessage(
          /*vx=*/nan, /*vy=*/0.0, /*height=*/0.8, /*yawRate=*/0.0));
  EXPECT_EQ(notANumber.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(notANumber.status().message().find(
                "cmd_linvel_base_rt_world_ewrt_heading_x"),
            absl::string_view::npos)
      << notANumber.status();

  const absl::StatusOr<WalkingVelocityCommand> infinite =
      walkingVelocityCommandFromProto(commandMessage(
          /*vx=*/0.0, /*vy=*/0.0, /*height=*/infinity, /*yawRate=*/0.0));
  EXPECT_EQ(infinite.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(
      infinite.status().message().find("cmd_pos_base_rt_world_ewrt_world_z"),
      absl::string_view::npos)
      << infinite.status();
}

TEST(ClampWalkingVelocityCommand, IsTheClampOfTheConversion) {
  // The clamp the lockstep closed loop applies to the GUI's messages is the one
  // a received message goes through.
  for (const WalkingVelocityCommand& command :
       {WalkingVelocityCommand(/*v_x=*/0.3, /*v_y=*/-0.2,
                               /*desired_pelvis_h=*/0.8, /*v_yaw=*/0.1),
        WalkingVelocityCommand(/*v_x=*/2.0, /*v_y=*/-3.0,
                               /*desired_pelvis_h=*/0.05, /*v_yaw=*/1.5),
        WalkingVelocityCommand(/*v_x=*/-7.0, /*v_y=*/4.0,
                               /*desired_pelvis_h=*/1.7, /*v_yaw=*/-2.5)}) {
    const WalkingVelocityCommand clamped = clampWalkingVelocityCommand(command);
    const absl::StatusOr<WalkingVelocityCommand> converted =
        walkingVelocityCommandFromProto(
            commandMessage(command.cmd_linvel_base_rt_world_ewrt_heading_x,
                           command.cmd_linvel_base_rt_world_ewrt_heading_y,
                           command.cmd_pos_base_rt_world_ewrt_world_z,
                           command.cmd_angvel_base_rt_world_ewrt_world_z));
    ASSERT_TRUE(converted.ok()) << converted.status();
    EXPECT_EQ(clamped.cmd_linvel_base_rt_world_ewrt_heading_x,
              converted->cmd_linvel_base_rt_world_ewrt_heading_x);
    EXPECT_EQ(clamped.cmd_linvel_base_rt_world_ewrt_heading_y,
              converted->cmd_linvel_base_rt_world_ewrt_heading_y);
    EXPECT_EQ(clamped.cmd_pos_base_rt_world_ewrt_world_z,
              converted->cmd_pos_base_rt_world_ewrt_world_z);
    EXPECT_EQ(clamped.cmd_angvel_base_rt_world_ewrt_world_z,
              converted->cmd_angvel_base_rt_world_ewrt_world_z);
    EXPECT_LE(std::abs(clamped.cmd_linvel_base_rt_world_ewrt_heading_x),
              kMaxNormalizedVelocity);
    EXPECT_GE(clamped.cmd_pos_base_rt_world_ewrt_world_z, kMinPelvisHeight);
    EXPECT_LE(clamped.cmd_pos_base_rt_world_ewrt_world_z, kMaxPelvisHeight);
  }
  // A value that is not finite is the conversion's to refuse; the clamp passes
  // it on.
  const WalkingVelocityCommand notANumber(
      /*v_x=*/std::numeric_limits<double>::quiet_NaN(), /*v_y=*/0.0,
      /*desired_pelvis_h=*/0.8,
      /*v_yaw=*/0.0);
  EXPECT_TRUE(std::isnan(clampWalkingVelocityCommand(notANumber)
                             .cmd_linvel_base_rt_world_ewrt_heading_x));
}

TargetContactPose pose(TargetContactPose::Kind kind, scalar_t x) {
  TargetContactPose result;
  result.valid = true;
  result.kind = kind;
  result.position = vector2_t(x, -x);
  result.height = 0.02;
  result.yaw = 0.3;
  result.yawPlanned = true;
  return result;
}

TEST(FillViewerAnnotations, SendsOnePatchPerFootAndTheScaledCommand) {
  const feet_array_t<TargetContactPose> poses = {
      pose(TargetContactPose::Kind::kSwingInFlight, /*x=*/0.4),
      pose(TargetContactPose::Kind::kNextSwing, /*x=*/0.1)};
  const WalkingVelocityCommand scaled(/*v_x=*/0.3, /*v_y=*/-0.1,
                                      /*desired_pelvis_h=*/0.8, /*v_yaw=*/0.2);
  humanoid_mpc_msgs::ViewerAnnotations annotations;
  fillViewerAnnotations(&poses, scaled, &annotations);

  ASSERT_EQ(annotations.target_contact_patches_size(),
            static_cast<int>(kNumContacts));
  const humanoid_mpc_msgs::TargetContactPatch& left =
      annotations.target_contact_patches(0);
  ASSERT_TRUE(left.has_des_pose_foot_rt_world());
  EXPECT_EQ(left.kind(),
            humanoid_mpc_msgs::TargetContactPatch::KIND_SWING_IN_FLIGHT);
  EXPECT_EQ(left.des_pose_foot_rt_world().x(), 0.4);
  EXPECT_EQ(left.des_pose_foot_rt_world().y(), -0.4);
  EXPECT_EQ(left.des_pose_foot_rt_world().z(), 0.02);
  EXPECT_EQ(left.des_pose_foot_rt_world().yaw(), 0.3);
  EXPECT_TRUE(left.des_pose_foot_rt_world().yaw_planned());
  EXPECT_EQ(annotations.target_contact_patches(1).kind(),
            humanoid_mpc_msgs::TargetContactPatch::KIND_NEXT_SWING);
  EXPECT_EQ(annotations.scaled_cmd_linvel_base_rt_world_ewrt_heading_x(), 0.3);
  EXPECT_EQ(annotations.scaled_cmd_linvel_base_rt_world_ewrt_heading_y(), -0.1);
  EXPECT_EQ(annotations.scaled_cmd_angvel_base_rt_world_ewrt_world_z(), 0.2);
}

TEST(FillViewerAnnotations,
     SendsNoPatchWithoutAContactPlannerAndReplacesWhatTheMessageHeld) {
  humanoid_mpc_msgs::ViewerAnnotations annotations;
  annotations.add_target_contact_patches()
      ->mutable_des_pose_foot_rt_world()
      ->set_x(1.0);
  fillViewerAnnotations(/*targetContactPoses=*/nullptr,
                        WalkingVelocityCommand(), &annotations);
  EXPECT_EQ(annotations.target_contact_patches_size(), 0);
  EXPECT_EQ(annotations.scaled_cmd_linvel_base_rt_world_ewrt_heading_x(), 0.0);
}

TEST(TargetContactPatchToProto, SendsNoPoseForAPoseThatIsInvalidOrNotFinite) {
  TargetContactPose broken = pose(TargetContactPose::Kind::kStance, /*x=*/0.0);
  broken.yaw = std::numeric_limits<scalar_t>::quiet_NaN();
  humanoid_mpc_msgs::TargetContactPatch patch;
  targetContactPatchToProto(broken, &patch);
  EXPECT_FALSE(patch.has_des_pose_foot_rt_world());
  EXPECT_EQ(patch.kind(), humanoid_mpc_msgs::TargetContactPatch::KIND_STANCE);

  // A patch that had a pose loses it: the message is replaced, not merged.
  targetContactPatchToProto(
      pose(TargetContactPose::Kind::kNextSwing, /*x=*/0.2), &patch);
  ASSERT_TRUE(patch.has_des_pose_foot_rt_world());
  TargetContactPose invalid = pose(TargetContactPose::Kind::kStance, /*x=*/0.0);
  invalid.valid = false;
  targetContactPatchToProto(invalid, &patch);
  EXPECT_FALSE(patch.has_des_pose_foot_rt_world());
  EXPECT_EQ(patch.kind(), humanoid_mpc_msgs::TargetContactPatch::KIND_STANCE);
}

TEST(TargetContactPatchToProto,
     EveryKindIsSentAsItsOwnEnumeratorNeverUnspecified) {
  const std::array<std::pair<TargetContactPose::Kind,
                             humanoid_mpc_msgs::TargetContactPatch::Kind>,
                   3>
      kinds = {{
          {TargetContactPose::Kind::kStance,
           humanoid_mpc_msgs::TargetContactPatch::KIND_STANCE},
          {TargetContactPose::Kind::kSwingInFlight,
           humanoid_mpc_msgs::TargetContactPatch::KIND_SWING_IN_FLIGHT},
          {TargetContactPose::Kind::kNextSwing,
           humanoid_mpc_msgs::TargetContactPatch::KIND_NEXT_SWING},
      }};
  for (const std::pair<TargetContactPose::Kind,
                       humanoid_mpc_msgs::TargetContactPatch::Kind>& kind :
       kinds) {
    humanoid_mpc_msgs::TargetContactPatch patch;
    targetContactPatchToProto(pose(kind.first, /*x=*/0.1), &patch);
    EXPECT_EQ(patch.kind(), kind.second);
    EXPECT_NE(patch.kind(),
              humanoid_mpc_msgs::TargetContactPatch::KIND_UNSPECIFIED);
    EXPECT_TRUE(patch.has_des_pose_foot_rt_world());
  }
}

}  // namespace
}  // namespace ocs2::humanoid::node
