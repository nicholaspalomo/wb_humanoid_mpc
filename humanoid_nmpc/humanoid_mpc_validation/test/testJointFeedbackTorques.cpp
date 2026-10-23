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

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"

#include "humanoid_mpc_validation/closed_loop/LockstepClosedLoop.h"
#include "robot_core/Types.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"

/*
 * The joint torques a control cycle of the lockstep closed loop records: the PD torque plus the feedforward of each MPC
 * joint, in the order of the MPC's joints, and an error rather than an exception for a joint the action leaves out.
 */

namespace ocs2::humanoid::validation {
namespace {

constexpr char kG1Urdf[] = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";

TEST(JointFeedbackTorques, AreTheCommandedTorquesOfTheJointsInTheirOrder) {
  absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus = robot::model::RobotDescription::Create(kG1Urdf);
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  robot::model::RobotJointAction action(description);
  robot::model::RobotState state(description, /*contactSize=*/2);
  for (const robot::joint_index_t joint : description.getJointIndices()) {
    robot::model::JointAction& jointAction = action[joint].emplace();
    jointAction.q_des = 0.1 * static_cast<double>(joint);
    jointAction.qd_des = -0.2;
    jointAction.kp = 10.0;
    jointAction.kd = 2.0;
    jointAction.feed_forward_effort = 0.5;
    state.setJointPosition(joint, 0.05 * static_cast<double>(joint));
    state.setJointVelocity(joint, /*jointVelocity=*/0.3);
  }
  const std::vector<robot::joint_index_t> joints = {4, 0, 2};
  const absl::StatusOr<Eigen::VectorXd> torques = jointFeedbackTorques(action, state, joints);
  ASSERT_TRUE(torques.ok()) << torques.status();
  ASSERT_EQ(torques->size(), 3);
  for (Eigen::Index i = 0; i < torques->size(); ++i) {
    const robot::joint_index_t joint = joints[static_cast<size_t>(i)];
    // kp (q_des - q) + kd (qd_des - qd) + feedforward.
    const double expected = 10.0 * (0.05 * static_cast<double>(joint)) + 2.0 * (-0.2 - 0.3) + 0.5;
    EXPECT_DOUBLE_EQ((*torques)(i), expected) << "joint " << joint;
  }
  const absl::StatusOr<Eigen::VectorXd> none = jointFeedbackTorques(action, state, /*joints=*/{});
  ASSERT_TRUE(none.ok()) << none.status();
  EXPECT_EQ(none->size(), 0);
}

TEST(JointFeedbackTorques, AJointWithoutAnActionIsAnInternalErrorNamingIt) {
  absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus = robot::model::RobotDescription::Create(kG1Urdf);
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  robot::model::RobotJointAction action(description);
  const robot::model::RobotState state(description, /*contactSize=*/2);
  action[3].reset();
  const std::vector<robot::joint_index_t> joints = {0, 3};
  const absl::StatusOr<Eigen::VectorXd> torques = jointFeedbackTorques(action, state, joints);
  ASSERT_FALSE(torques.ok());
  EXPECT_EQ(torques.status().code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(torques.status().message(), "joint 3")) << torques.status();
}

}  // namespace
}  // namespace ocs2::humanoid::validation
