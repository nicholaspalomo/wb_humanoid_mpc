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

#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/costs/CollisionConstraintFromConfig.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/config/costs/TaskSpaceCostFromConfig.h"
#include "humanoid_common_mpc/config/swing/SwingTrajectoryFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/constraint/FootCollisionConstraint.h"
#include "humanoid_common_mpc/contact/ContactCenterPoint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"
#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/task_file.nproto.h"

/**
 * The blocks of the task file the MPC interfaces build their terms from, read as they read them: a textproto through
 * loadTaskFile() and the typed conversion of each block. What they read, that a value that does not parse is refused
 * naming the file, its line and its column, and that a value the term cannot take is refused naming its field.
 */
namespace ocs2::humanoid {
namespace {

using ::testing::HasSubstr;

/** Writes `content` to `name` in the test's scratch directory and returns its path. */
std::string writeFile(absl::string_view name, absl::string_view content) {
  const std::string path = absl::StrCat(::testing::TempDir(), "/", name);
  std::ofstream(path) << content;
  return path;
}

/** The task file of the textproto `content`, written to `name`; fails the test unless it loads. */
mpc_config::TaskFile taskFileOf(absl::string_view name, absl::string_view content) {
  absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(writeFile(name, content));
  EXPECT_TRUE(task.ok()) << task.status();
  return task.ok() ? *std::move(task) : mpc_config::TaskFile{};
}

constexpr char kInertial[] =
    R"(<inertial><mass value="1.0"/><origin xyz="0 0 0"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>)";

/** A floating base with two revolute joints, `shoulder` and `hip`; $0 is each link's inertial block. */
constexpr char kTwoJointUrdf[] = R"(<robot name="two_joints">
  <link name="base">$0</link>
  <link name="arm">$0</link>
  <link name="leg">$0</link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="arm"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <joint name="hip" type="revolute">
    <parent link="base"/><child link="leg"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>
)";

/** The model settings of the two-joint robot, with a contact at the end of each of its links. */
ModelSettings twoContactSettings() {
  const std::string urdf = writeFile("loaders_two_joints.urdf", absl::Substitute(kTwoJointUrdf, kInertial));
  const mpc_config::TaskFile task = taskFileOf("loaders_model_settings.textproto",
                                               "model_settings {\n"
                                               "  robot_name: \"two_joints\"\n"
                                               "  contact_names_6dof: [\"arm_contact\", \"leg_contact\"]\n"
                                               "  contact_parent_joint_names: [\"shoulder\", \"hip\"]\n"
                                               "}\n");
  absl::StatusOr<ModelSettings> settings = ModelSettings::Create(task, urdf, "loaders_", /*verbose=*/false);
  EXPECT_TRUE(settings.ok()) << settings.status();
  return *std::move(settings);
}

TEST(SwingTrajectorySettings, AreReadFromTheirBlock) {
  const mpc_config::TaskFile task = taskFileOf(
      "swing.textproto", "swing_trajectory_config {\n  swing_height: 0.12\n  lift_off_velocity: 0.2\n  touch_down_velocity: -0.3\n}\n");
  const absl::StatusOr<SwingTrajectoryPlanner::Config> config = swingTrajectorySettingsFromConfig(task.swing_trajectory_config);
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_DOUBLE_EQ(config->swingHeight, 0.12);
  EXPECT_DOUBLE_EQ(config->liftOffVelocity, 0.2);
  EXPECT_DOUBLE_EQ(config->touchDownVelocity, -0.3);
}

TEST(TaskFile, AValueThatDoesNotParseIsRefusedNamingItsLineAndAMissingFileIsNotFound) {
  const std::string file = writeFile("swing_bad.textproto", "swing_trajectory_config {\n  swing_height: high\n}\n");
  const absl::StatusOr<mpc_config::TaskFile> badValue = loadTaskFile(file);
  EXPECT_EQ(badValue.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(badValue.status().message(), HasSubstr(absl::StrCat(file, ":2:")));

  const std::string missing = absl::StrCat(::testing::TempDir(), "/no_such_task.textproto");
  const absl::StatusOr<mpc_config::TaskFile> notFound = loadTaskFile(missing);
  EXPECT_EQ(notFound.status().code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(notFound.status().message(), HasSubstr(missing));
}

TEST(EndEffectorKinematicsWeights, AreReadFromTheirBlockAndAWeightThatIsNotFiniteIsRefusedByName) {
  const mpc_config::TaskFile task =
      taskFileOf("weights.textproto", "task_space_foot_cost {\n  weights {\n    pos_x: 1.5\n    ang_velocity_z: 0.25\n  }\n}\n");
  const absl::StatusOr<EndEffectorKinematicsWeights> weights =
      endEffectorKinematicsWeightsFromConfig(task.task_space_foot_cost.weights, "task_space_foot_cost.weights");
  ASSERT_TRUE(weights.ok()) << weights.status();
  EXPECT_DOUBLE_EQ(weights->contactPositionErrorWeight(0), 1.5);
  EXPECT_DOUBLE_EQ(weights->contactPositionErrorWeight(1), 0.0) << "a weight the block does not carry is 0";
  EXPECT_DOUBLE_EQ(weights->contactAngularVelocityErrorWeight(2), 0.25);

  mpc_config::TaskSpaceWeights infinite = task.task_space_foot_cost.weights;
  infinite.pos_x = std::numeric_limits<double>::infinity();
  const absl::StatusOr<EndEffectorKinematicsWeights> refused =
      endEffectorKinematicsWeightsFromConfig(infinite, "task_space_foot_cost.weights");
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(refused.status().message(), HasSubstr("task_space_foot_cost.weights.pos_x"));
}

TEST(ContactRectangle, IsReadAroundTheContactsCenterPoint) {
  const ModelSettings settings = twoContactSettings();
  const mpc_config::TaskFile task = taskFileOf("rectangle.textproto",
                                               "contacts {\n"
                                               "  contact_rectangle { x_min: -0.1 x_max: 0.15 y_min: -0.05 y_max: 0.05 }\n"
                                               "  contact_frame_translation { z: -0.02 }\n"
                                               "}\n");
  const absl::StatusOr<ContactRectangle> rectangle = contactRectangleFromConfig(task.contacts, settings, /*contactIndex=*/1);
  ASSERT_TRUE(rectangle.ok()) << rectangle.status();
  EXPECT_DOUBLE_EQ(rectangle->getBounds().x_max, 0.15);
  EXPECT_EQ(rectangle->getNumberOfContactPoints(), 4U);
  const absl::StatusOr<ContactCenterPoint> center = contactCenterPointFromConfig(task.contacts, settings, /*contactIndex=*/1);
  ASSERT_TRUE(center.ok()) << center.status();
  EXPECT_EQ(center->frameName, "leg_contact");
  EXPECT_EQ(center->parentJointName, "hip");
  EXPECT_DOUBLE_EQ(center->translationFromParent.z(), -0.02);
}

TEST(ContactRectangle, AContactOutOfRangeOrABoundThatIsNotFiniteIsAnInvalidArgument) {
  const ModelSettings settings = twoContactSettings();
  const mpc_config::TaskFile task = taskFileOf("rectangle_good.textproto", "contacts {\n  contact_rectangle { x_min: -0.1 }\n}\n");
  // The index used to be checked by an assert only, and then indexed the contact names out of range.
  const absl::StatusOr<ContactRectangle> outOfRange = contactRectangleFromConfig(task.contacts, settings, /*contactIndex=*/2);
  EXPECT_EQ(outOfRange.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(outOfRange.status().message(), HasSubstr("contact 2"));

  mpc_config::ContactsConfig infinite = task.contacts;
  infinite.contact_rectangle.x_min = -std::numeric_limits<double>::infinity();
  const absl::StatusOr<ContactRectangle> notFinite = contactRectangleFromConfig(infinite, settings, /*contactIndex=*/0);
  EXPECT_EQ(notFinite.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(notFinite.status().message(), HasSubstr("contacts.contact_rectangle"));
}

TEST(FootCollisionConstraintConfig, IsReadFromItsBlockAndARadiusThatIsNotFiniteIsRefused) {
  const mpc_config::TaskFile task =
      taskFileOf("collision.textproto", "collision_constraint {\n  foot { foot_collision_sphere_radius: 0.07 }\n}\n");
  const absl::StatusOr<FootCollisionConstraint::Config> config = footCollisionConstraintConfigFromConfig(task.collision_constraint);
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_DOUBLE_EQ(config->footCollisionSphereRadius, 0.07);

  mpc_config::CollisionConstraintConfig notANumber = task.collision_constraint;
  notANumber.foot.foot_collision_sphere_radius = std::numeric_limits<double>::quiet_NaN();
  const absl::StatusOr<FootCollisionConstraint::Config> refused = footCollisionConstraintConfigFromConfig(notANumber);
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(refused.status().message(), HasSubstr("foot_collision_sphere_radius"));
}

TEST(ExternalTorqueQuadraticCostConfig, IsReadFromItsBlockByJointNameAndAJointOfNoModelIsRefusedByName) {
  const ModelSettings settings = twoContactSettings();
  const StateInputLayout layout = stateInputLayout(settings, StateInputLayout::Mpc::kCentroidal);
  const mpc_config::TaskFile task = taskFileOf("torque.textproto",
                                               "left_leg_torque_cost {\n"
                                               "  joints { joint: \"shoulder\" value: 2.0 }\n"
                                               "  joints { joint: \"hip\" value: 3.0 }\n"
                                               "}\n");
  const absl::StatusOr<ExternalTorqueQuadraticCostAD::Config> config =
      legTorqueCostFromConfig(task.left_leg_torque_cost, layout, "left_leg_torque_cost");
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->activeJointNames, (std::vector<std::string>{"shoulder", "hip"}));
  ASSERT_EQ(config->weights.size(), 2);
  EXPECT_DOUBLE_EQ(config->weights(1), 3.0);

  mpc_config::JointWeights misspelled = task.left_leg_torque_cost;
  misspelled.joints[1].joint = "knee";
  const absl::StatusOr<ExternalTorqueQuadraticCostAD::Config> refused = legTorqueCostFromConfig(misspelled, layout, "left_leg_torque_cost");
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(refused.status().message(), HasSubstr("knee"));
}

}  // namespace
}  // namespace ocs2::humanoid
