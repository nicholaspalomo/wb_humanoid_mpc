/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

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

#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "robot_model/RobotDescription.h"

namespace robot::model {
namespace {

using ::testing::HasSubstr;

class RobotDescriptionTest : public ::testing::Test {
 protected:
  // Create a temporary URDF file for testing
  void SetUp() override {
    tempDir_ = std::filesystem::temp_directory_path() / "robot_model_test";
    std::filesystem::create_directories(tempDir_);
    urdf_path_ = tempDir_ / "test_robot.urdf";

    // Create a simple test URDF with a few joints
    std::ofstream urdfFile(urdf_path_);
    urdfFile << R"(<?xml version="1.0"?>
        <robot name="test_robot">
            <link name="base_link"/>
            <link name="shoulder_link"/>
            <link name="elbow_link"/>
            <link name="wrist_link"/>
            <link name="hand_link"/>

            <joint name="shoulder_joint" type="revolute">
                <parent link="base_link"/>
                <child link="shoulder_link"/>
                <axis xyz="0 0 1"/>
                <limit lower="-1.57" upper="1.57" effort="100" velocity="2.0"/>
            </joint>

            <joint name="elbow_joint" type="revolute">
                <parent link="shoulder_link"/>
                <child link="elbow_link"/>
                <axis xyz="0 1 0"/>
                <limit lower="-2.0" upper="2.0" effort="80" velocity="1.5"/>
            </joint>

            <joint name="wrist_joint" type="revolute">
                <parent link="elbow_link"/>
                <child link="wrist_link"/>
                <axis xyz="1 0 0"/>
                <limit lower="-1.0" upper="1.0" effort="50" velocity="3.0"/>
            </joint>

            <joint name="fixed_joint" type="fixed">
                <parent link="base_link"/>
                <child link="hand_link"/>
            </joint>
        </robot>)";
    urdfFile.close();
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tempDir_, ec);
  }

  RobotDescription description() const {
    absl::StatusOr<RobotDescription> description = RobotDescription::Create(urdf_path_.string());
    EXPECT_TRUE(description.ok()) << description.status();
    return *std::move(description);
  }

  std::filesystem::path tempDir_;
  std::filesystem::path urdf_path_;
};

TEST_F(RobotDescriptionTest, CreateReadsTheRevoluteAndPrismaticJoints) {
  const absl::StatusOr<RobotDescription> robotDesc = RobotDescription::Create(urdf_path_.string());
  ASSERT_TRUE(robotDesc.ok()) << robotDesc.status();
  EXPECT_EQ(robotDesc->getURDFPath(), urdf_path_.string());
  EXPECT_EQ(robotDesc->getNumJoints(), 3u);  // Should only count the revolute joints
}

TEST_F(RobotDescriptionTest, CreateOfAMissingFileIsNotFoundNamingIt) {
  const std::string nonexistentPath = tempDir_ / "nonexistent.urdf";
  const absl::StatusOr<RobotDescription> robotDesc = RobotDescription::Create(nonexistentPath);
  EXPECT_EQ(robotDesc.status().code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(robotDesc.status().message(), HasSubstr(nonexistentPath));
}

TEST_F(RobotDescriptionTest, CreateOfAFileThatIsNotAUrdfIsInvalidArgument) {
  const std::filesystem::path invalidPath = tempDir_ / "invalid.urdf";
  std::ofstream invalidFile(invalidPath);
  invalidFile << "This is not a valid URDF file";
  invalidFile.close();

  const absl::StatusOr<RobotDescription> robotDesc = RobotDescription::Create(invalidPath.string());
  EXPECT_EQ(robotDesc.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(robotDesc.status().message(), HasSubstr("Failed to parse URDF file"));
}

TEST_F(RobotDescriptionTest, CreateRefusesAUrdfWithFewerJointsThanAJointMapTakes) {
  const std::filesystem::path onePath = tempDir_ / "one_joint.urdf";
  std::ofstream oneFile(onePath);
  oneFile << R"(<?xml version="1.0"?>
        <robot name="one_joint">
            <link name="base_link"/>
            <link name="arm_link"/>
            <joint name="arm_joint" type="revolute">
                <parent link="base_link"/>
                <child link="arm_link"/>
                <axis xyz="0 0 1"/>
                <limit lower="-1.0" upper="1.0" effort="10" velocity="1.0"/>
            </joint>
        </robot>)";
  oneFile.close();

  const absl::StatusOr<RobotDescription> robotDesc = RobotDescription::Create(onePath.string());
  EXPECT_EQ(robotDesc.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(robotDesc.status().message(), HasSubstr("has 1 revolute and prismatic joints"));
}

TEST_F(RobotDescriptionTest, GetURDFName) {
  EXPECT_EQ(description().getURDFName(), "test_robot.urdf");
}

TEST_F(RobotDescriptionTest, ContainsJoint) {
  const RobotDescription robotDesc = description();

  EXPECT_TRUE(robotDesc.containsJoint("shoulder_joint"));
  EXPECT_TRUE(robotDesc.containsJoint("elbow_joint"));
  EXPECT_TRUE(robotDesc.containsJoint("wrist_joint"));
  EXPECT_FALSE(robotDesc.containsJoint("fixed_joint"));  // Fixed joints should be excluded
  EXPECT_FALSE(robotDesc.containsJoint("nonexistent_joint"));
}

TEST_F(RobotDescriptionTest, FindJointDescription) {
  const RobotDescription robotDesc = description();

  const JointDescription* absl_nullable shoulderDesc = robotDesc.findJointDescription("shoulder_joint");
  ASSERT_NE(shoulderDesc, nullptr);
  EXPECT_EQ(shoulderDesc->min_angle, -1.57);
  EXPECT_EQ(shoulderDesc->max_angle, 1.57);
  EXPECT_EQ(shoulderDesc->max_effort, 100.0);
  EXPECT_EQ(shoulderDesc->max_velocity, 2.0);

  const JointDescription* absl_nullable elbowDesc = robotDesc.findJointDescription("elbow_joint");
  ASSERT_NE(elbowDesc, nullptr);
  EXPECT_EQ(elbowDesc->min_angle, -2.0);
  EXPECT_EQ(elbowDesc->max_angle, 2.0);
  EXPECT_EQ(elbowDesc->max_effort, 80.0);
  EXPECT_EQ(elbowDesc->max_velocity, 1.5);

  EXPECT_EQ(robotDesc.findJointDescription("nonexistent_joint"), nullptr);
}

TEST_F(RobotDescriptionTest, JointsAreNumberedInTheOrderOfTheirNames) {
  const RobotDescription robotDesc = description();

  EXPECT_EQ(robotDesc.getJointIndex("elbow_joint"), 0u);
  EXPECT_EQ(robotDesc.getJointIndex("shoulder_joint"), 1u);
  EXPECT_EQ(robotDesc.getJointIndex("wrist_joint"), 2u);
  EXPECT_EQ(robotDesc.getJointIndices(), (std::vector<joint_index_t>{0, 1, 2}));
  EXPECT_EQ(robotDesc.getJointNames(), (std::vector<std::string>{"elbow_joint", "shoulder_joint", "wrist_joint"}));
  for (joint_index_t index = 0; index < robotDesc.getNumJoints(); ++index) {
    EXPECT_EQ(robotDesc.getJointName(index), robotDesc.getJointNames()[index]);
    const JointDescription* absl_nullable joint = robotDesc.findJointDescription(robotDesc.getJointName(index));
    ASSERT_NE(joint, nullptr);
    EXPECT_EQ(joint->id, index);
  }
}

TEST_F(RobotDescriptionTest, FindJointIndicesNamesTheJointTheUrdfDoesNotHave) {
  const RobotDescription robotDesc = description();

  EXPECT_EQ(robotDesc.findJointIndex("wrist_joint"), std::optional<joint_index_t>(2));
  EXPECT_EQ(robotDesc.findJointIndex("nonexistent_joint"), std::nullopt);

  const absl::StatusOr<std::vector<joint_index_t>> found = robotDesc.findJointIndices({"wrist_joint", "elbow_joint"});
  ASSERT_TRUE(found.ok()) << found.status();
  EXPECT_EQ(*found, (std::vector<joint_index_t>{2, 0}));
  EXPECT_EQ(robotDesc.getJointIndices({"wrist_joint", "elbow_joint"}), *found);

  const absl::StatusOr<std::vector<joint_index_t>> missing = robotDesc.findJointIndices({"wrist_joint", "knee_joint"});
  EXPECT_EQ(missing.status().code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(missing.status().message(), HasSubstr("'knee_joint'"));
}

TEST_F(RobotDescriptionTest, MovingADescriptionKeepsItsJoints) {
  RobotDescription original = description();
  const RobotDescription moved(std::move(original));
  EXPECT_EQ(moved.getNumJoints(), 3u);
  EXPECT_EQ(moved.getJointIndex("wrist_joint"), 2u);
}

TEST_F(RobotDescriptionTest, JointDescriptionStreamOperator) {
  JointDescription jointDesc;
  jointDesc.id = 42;
  jointDesc.min_angle = -1.5;
  jointDesc.max_angle = 1.5;
  jointDesc.max_velocity = 2.0;
  jointDesc.max_effort = 100.0;

  std::stringstream ss;
  ss << jointDesc;
  EXPECT_THAT(ss.str(), HasSubstr("id: 42"));
  EXPECT_THAT(ss.str(), HasSubstr("min_angle: -1.5"));
}

TEST(JointDescriptionTest, ADefaultJointHasNoLimits) {
  const JointDescription joint;
  EXPECT_EQ(joint.min_angle, std::numeric_limits<scalar_t>::lowest()) << "below every angle, not the smallest positive double";
  EXPECT_EQ(joint.max_angle, std::numeric_limits<scalar_t>::max());
}

TEST_F(RobotDescriptionTest, RobotDescriptionStreamOperatorListsTheJointsInIndexOrder) {
  std::stringstream ss;
  ss << description();
  const std::string output = ss.str();
  EXPECT_TRUE(absl::StrContains(output, "test_robot.urdf"));
  EXPECT_TRUE(absl::StrContains(output, "shoulder_joint"));
  EXPECT_TRUE(absl::StrContains(output, "elbow_joint"));
  EXPECT_TRUE(absl::StrContains(output, "wrist_joint"));
  EXPECT_LT(output.find("elbow_joint"), output.find("shoulder_joint"));
  EXPECT_LT(output.find("shoulder_joint"), output.find("wrist_joint"));
}

using RobotDescriptionDeathTest = RobotDescriptionTest;

TEST_F(RobotDescriptionDeathTest, AJointNameOrIndexTheDescriptionDoesNotHaveIsAProgrammingError) {
  const RobotDescription robotDesc = description();
  EXPECT_DEATH(robotDesc.getJointIndex("nonexistent_joint"), "has no revolute or prismatic joint 'nonexistent_joint'");
  EXPECT_DEATH(robotDesc.getJointIndices({"elbow_joint", "nonexistent_joint"}), "'nonexistent_joint'");
  EXPECT_DEATH(robotDesc.getJointName(/*jointIndex=*/999), "has no joint of index 999");
}

}  // namespace
}  // namespace robot::model
