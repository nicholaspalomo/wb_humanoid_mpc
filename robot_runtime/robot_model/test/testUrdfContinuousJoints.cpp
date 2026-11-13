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

// makeContinuousJointsUnboundedRevolute() and parseUrdfFileForPinocchio(): a
// continuous joint becomes a revolute joint with infinite position limits and
// its effort and velocity limits, which Pinocchio builds with one coordinate;
// other joints and URDFs without a continuous joint are left as they are.

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "pinocchio/multibody/fwd.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "urdf_model/joint.h"
#include "urdf_world/types.h"
#include "urdfdom/urdf_parser/urdf_parser.h"

#include "robot_model/UrdfContinuousJoints.h"

namespace robot::model {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

constexpr double kInfinity = std::numeric_limits<double>::infinity();

/** A knee (revolute, limited) carrying a wheel (continuous) and a hub without
 * a <limit> element (continuous). */
constexpr char kWheelUrdf[] = R"(<?xml version="1.0"?>
<robot name="wheel">
  <link name="base_link">
    <inertial><mass value="1"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="knee_link">
    <inertial><mass value="1"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="wheel_link">
    <inertial><mass value="0.2"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="hub_link">
    <inertial><mass value="0.1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="knee_joint" type="revolute">
    <parent link="base_link"/><child link="knee_link"/><axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0" effort="80" velocity="15"/>
  </joint>
  <joint name="wheel_joint" type="continuous">
    <parent link="knee_link"/><child link="wheel_link"/><axis xyz="0 1 0"/>
    <limit effort="40" velocity="30"/>
  </joint>
  <joint name="hub_joint" type="continuous">
    <parent link="knee_link"/><child link="hub_link"/><axis xyz="1 0 0"/>
  </joint>
</robot>)";

TEST(UrdfContinuousJointsTest, ContinuousJointsBecomeUnboundedRevoluteJoints) {
  const urdf::ModelInterfaceSharedPtr model = urdf::parseURDF(kWheelUrdf);
  ASSERT_NE(model, nullptr);
  EXPECT_THAT(makeContinuousJointsUnboundedRevolute(*model),
              ElementsAre("hub_joint", "wheel_joint"));

  const urdf::Joint& wheel = *model->getJoint("wheel_joint");
  EXPECT_EQ(wheel.type, urdf::Joint::REVOLUTE);
  EXPECT_EQ(wheel.limits->lower, -kInfinity);
  EXPECT_EQ(wheel.limits->upper, kInfinity);
  EXPECT_EQ(wheel.limits->effort, 40.0);
  // NOLINTNEXTLINE(frame-naming): urdfdom JointLimits field, a joint-space bound
  EXPECT_EQ(wheel.limits->velocity, 30.0);

  // Without a <limit>, nothing bounds the joint.
  const urdf::Joint& hub = *model->getJoint("hub_joint");
  EXPECT_EQ(hub.type, urdf::Joint::REVOLUTE);
  EXPECT_EQ(hub.limits->lower, -kInfinity);
  EXPECT_EQ(hub.limits->upper, kInfinity);
  EXPECT_EQ(hub.limits->effort, kInfinity);
  // NOLINTNEXTLINE(frame-naming): urdfdom JointLimits field, a joint-space bound
  EXPECT_EQ(hub.limits->velocity, kInfinity);

  const urdf::Joint& knee = *model->getJoint("knee_joint");
  EXPECT_EQ(knee.type, urdf::Joint::REVOLUTE);
  EXPECT_EQ(knee.limits->lower, -1.0);
  EXPECT_EQ(knee.limits->upper, 1.0);
}

TEST(UrdfContinuousJointsTest, AUrdfWithoutContinuousJointsIsUnchanged) {
  const urdf::ModelInterfaceSharedPtr model = urdf::parseURDF(kWheelUrdf);
  ASSERT_NE(model, nullptr);
  ASSERT_FALSE(makeContinuousJointsUnboundedRevolute(*model).empty());
  // A second pass finds no continuous joint left.
  EXPECT_TRUE(makeContinuousJointsUnboundedRevolute(*model).empty());
  EXPECT_EQ(model->getJoint("knee_joint")->limits->upper, 1.0);
}

/** Writes `content` to a file of the test's temporary directory named
 * `name`, and returns its path. */
std::string writeTemporaryFile(absl::string_view name,
                               absl::string_view content) {
  const std::filesystem::path path =
      std::filesystem::path(::testing::TempDir()) / std::string(name);
  std::ofstream file(path);
  file << content;
  return path.string();
}

TEST(UrdfContinuousJointsTest, PinocchioBuildsAWheelWithOneCoordinate) {
  const absl::StatusOr<urdf::ModelInterfaceSharedPtr> tree =
      parseUrdfFileForPinocchio(writeTemporaryFile("wheel.urdf", kWheelUrdf));
  ASSERT_TRUE(tree.ok()) << tree.status();
  pinocchio::Model model;
  pinocchio::urdf::buildModel(*tree, model);
  ASSERT_TRUE(model.existJointName("wheel_joint"));
  const pinocchio::JointIndex wheel = model.getJointId("wheel_joint");
  EXPECT_EQ(model.nqs[wheel], 1);
  EXPECT_EQ(model.nvs[wheel], 1);
  EXPECT_EQ(model.nq, model.nv);
  const int q = model.idx_qs[wheel];
  const int knee = model.idx_qs[model.getJointId("knee_joint")];
  // NOLINTBEGIN(frame-naming): Pinocchio's joint limits (joint space)
  EXPECT_EQ(model.lowerPositionLimit[q], -kInfinity);
  EXPECT_EQ(model.upperPositionLimit[q], kInfinity);
  EXPECT_EQ(model.lowerPositionLimit[knee], -1.0);
  EXPECT_EQ(model.upperPositionLimit[knee], 1.0);
  // NOLINTEND(frame-naming)
}

TEST(UrdfContinuousJointsTest, AFileThatIsNotAUrdfIsInvalidArgumentNamingIt) {
  const std::string path =
      writeTemporaryFile("not_a_urdf.urdf", "This is not a URDF");
  const absl::StatusOr<urdf::ModelInterfaceSharedPtr> tree =
      parseUrdfFileForPinocchio(path);
  EXPECT_EQ(tree.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(tree.status().message(), HasSubstr(path));
}

}  // namespace
}  // namespace robot::model
