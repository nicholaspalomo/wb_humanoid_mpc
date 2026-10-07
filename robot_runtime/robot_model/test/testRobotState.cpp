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

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "robot_model/IDMapBase.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"

namespace robot::model {
namespace {

using ::testing::DoubleEq;
using ::testing::Field;
using ::testing::Optional;

// A biped with three joints, so that the ids are 0, 1 and 2.
constexpr char kUrdf[] = R"(<?xml version="1.0"?>
    <robot name="biped">
        <link name="base_link"/>
        <link name="foot_l"/>
        <link name="foot_r"/>
        <link name="head"/>
        <joint name="hip_l" type="revolute">
            <parent link="base_link"/>
            <child link="foot_l"/>
            <axis xyz="0 1 0"/>
            <limit lower="-1.57" upper="1.57" effort="100" velocity="2.0"/>
        </joint>
        <joint name="hip_r" type="revolute">
            <parent link="base_link"/>
            <child link="foot_r"/>
            <axis xyz="0 1 0"/>
            <limit lower="-1.57" upper="1.57" effort="100" velocity="2.0"/>
        </joint>
        <joint name="neck" type="revolute">
            <parent link="base_link"/>
            <child link="head"/>
            <axis xyz="0 0 1"/>
            <limit lower="-1.0" upper="1.0" effort="10" velocity="2.0"/>
        </joint>
    </robot>)";

/** The biped of kUrdf, read from a file in the test's temporary directory. */
RobotDescription bipedDescription() {
  const std::filesystem::path urdfPath = std::filesystem::temp_directory_path() / "robot_model_robot_state_test.urdf";
  std::ofstream(urdfPath) << kUrdf;
  absl::StatusOr<RobotDescription> description = RobotDescription::Create(urdfPath.string());
  CHECK_OK(description.status());
  std::error_code ec;
  std::filesystem::remove(urdfPath, ec);
  return *std::move(description);
}

class RobotStateTest : public ::testing::Test {
 protected:
  const RobotDescription description_ = bipedDescription();
};

TEST_F(RobotStateTest, EveryJointOfTheDescriptionHasAStateStartingAtZero) {
  const RobotState state(description_, /*contactSize=*/2);
  for (joint_index_t joint = 0; joint < description_.getNumJoints(); ++joint) {
    EXPECT_TRUE(state.hasJoint(joint));
    EXPECT_EQ(state.getJointPosition(joint), 0.0);
    EXPECT_EQ(state.getJointVelocity(joint), 0.0);
  }
  EXPECT_FALSE(state.hasJoint(description_.getNumJoints()));
  EXPECT_EQ(state.getTime(), 0.0) << "a fresh state has a time";
  EXPECT_EQ(state.getContactFlags(), (std::vector<bool>{true, true}));
}

TEST_F(RobotStateTest, SettingAJointThatIsNotThereIsIgnored) {
  RobotState state(description_, /*contactSize=*/2);
  state.setJointPosition(/*jointId=*/1, /*jointPosition=*/0.5);
  state.setJointVelocity(/*jointId=*/1, /*jointVelocity=*/-0.25);
  state.setJointPosition(/*jointId=*/7, /*jointPosition=*/1.0);
  state.setJointVelocity(/*jointId=*/7, /*jointVelocity=*/1.0);
  EXPECT_EQ(state.getJointPosition(1), 0.5);
  EXPECT_EQ(state.getJointVelocity(1), -0.25);
  EXPECT_FALSE(state.hasJoint(7));
}

TEST_F(RobotStateTest, TheCheckedAndTheUncheckedReadsOfAJointAgree) {
  RobotState state(description_, /*contactSize=*/2);
  for (const joint_index_t joint : description_.getJointIndices()) {
    state.setJointPosition(joint, /*jointPosition=*/0.1 * static_cast<scalar_t>(joint + 1));
    state.setJointVelocity(joint, /*jointVelocity=*/-0.2 * static_cast<scalar_t>(joint + 1));
    EXPECT_EQ(state.getJointPosition(joint), state.getCheckedJointPosition(joint));
    EXPECT_EQ(state.getJointVelocity(joint), state.getCheckedJointVelocity(joint));
  }
}

TEST_F(RobotStateTest, CopyingAStateIntoAnotherCopiesEveryJointAndFlag) {
  RobotState source(description_, /*contactSize=*/2);
  source.setJointPosition(/*jointId=*/2, /*jointPosition=*/0.75);
  source.setContactFlag(/*index=*/1, /*contactFlag=*/false);
  source.setTime(3.5);
  RobotState target(description_, /*contactSize=*/2);
  target = source;
  EXPECT_EQ(target.getJointPosition(2), 0.75);
  EXPECT_FALSE(target.getContactFlag(1));
  EXPECT_EQ(target.getTime(), 3.5);

  RobotState moved(std::move(source));
  EXPECT_EQ(moved.getJointPosition(2), 0.75);
}

TEST_F(RobotStateTest, AJointActionStartsAtZeroForEveryJoint) {
  RobotJointAction action(description_);
  EXPECT_EQ(action.size(), description_.getNumJoints());
  EXPECT_EQ(action.capacity(), description_.getNumJoints());
  for (joint_index_t joint = 0; joint < description_.getNumJoints(); ++joint) {
    EXPECT_THAT(action[joint], Optional(Field(&JointAction::kp, DoubleEq(0.0))));
    EXPECT_THAT(action.at(joint), Optional(Field(&JointAction::feed_forward_effort, DoubleEq(0.0))));
  }
  action[1].emplace().kp = 10.0;
  size_t visited = 0;
  for (const JointAction& joint : action) {
    visited += joint.kp == 10.0 ? 1 : 0;
  }
  EXPECT_EQ(visited, 1u) << "iteration visits every engaged slot";
}

TEST_F(RobotStateTest, AnEmptySlotReadsAsTheDefaultAndIsSkippedByIteration) {
  RobotJointAction action(description_);
  action[0].emplace().q_des = 0.1;
  action[1].reset();
  action[2].emplace().q_des = 0.3;
  const vector_t targets = action.toVector({0, 1, 2}, [](const JointAction& joint) { return joint.q_des; }, /*defaultValue=*/-1.0);
  EXPECT_EQ(targets, (vector_t(3) << 0.1, -1.0, 0.3).finished());
  EXPECT_EQ(action.size(), 2u);
  size_t visited = 0;
  for (const JointAction& joint : action) {
    EXPECT_NE(joint.q_des, 0.0);
    ++visited;
  }
  EXPECT_EQ(visited, 2u);
}

using RobotStateDeathTest = RobotStateTest;

TEST_F(RobotStateDeathTest, ReadingAJointOrAContactPointThatIsNotThereIsAProgrammingError) {
  RobotState state(description_, /*contactSize=*/2);
  EXPECT_DEATH(state.getCheckedJointPosition(/*jointId=*/7), "RobotState: no joint 7");
  EXPECT_DEATH(state.getCheckedJointVelocity(/*jointId=*/7), "RobotState: no joint 7");
  EXPECT_DEATH(state.getContactFlag(/*index=*/2), "RobotState: no contact point 2");
  EXPECT_DEATH(state.setContactFlag(/*index=*/2, /*contactFlag=*/true), "RobotState: no contact point 2");
}

TEST_F(RobotStateDeathTest, TheCheckedAccessOfAJointActionRefusesAnIdOutOfRange) {
  RobotJointAction action(description_);
  EXPECT_DEATH(action.at(/*element_id=*/3), "IDMapBase: no id 3");
}

// IDMapBase's constructor is for the maps built from a description; this one takes any size.
class SizedMap : public IDMapBase<int> {
 public:
  explicit SizedMap(size_t size) : IDMapBase<int>(size) {}
};

TEST(IDMapBaseDeathTest, AMapHasTwoTo255Ids) {
  EXPECT_EQ(SizedMap(/*size=*/2).capacity(), 2u);
  EXPECT_EQ(SizedMap(/*size=*/255).capacity(), 255u);
  EXPECT_DEATH(SizedMap(/*size=*/1), "IDMapBase: 1 ids; a map has 2 to 255");
  EXPECT_DEATH(SizedMap(/*size=*/256), "IDMapBase: 256 ids; a map has 2 to 255");
}

TEST(IDMapBase, CopyingKeepsTheCapacity) {
  // The copy is unchecked, as every copy the realtime loop makes is between maps of one robot description; a larger
  // source is copied up to the target's capacity.
  SizedMap target(/*size=*/3);
  SizedMap larger(/*size=*/4);
  larger[0] = 1;
  larger[3] = 4;
  target = larger;
  EXPECT_EQ(target.capacity(), 3u) << "the capacity is kept; the extra slot is not copied";
  EXPECT_EQ(target[0], std::optional<int>(1));
  EXPECT_EQ(target.size(), 1u);
}

}  // namespace
}  // namespace robot::model
