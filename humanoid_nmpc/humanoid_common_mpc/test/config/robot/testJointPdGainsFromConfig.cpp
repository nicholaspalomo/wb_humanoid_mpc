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

// The joint PD gains of a typed joint PD gains file: an empty file is the controller's defaults exactly, each level
// inherits what it leaves out from the one above, the torque limits of a controller that commands none are not read,
// and a gain that is not one, a joint named twice and an entry without a joint are refused by their field.

#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/robot/JointPdGainsFromConfig.h"
#include "humanoid_common_mpc/mrt/JointPdGains.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.pb.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid {
namespace {

using ::testing::HasSubstr;

std::vector<std::string> mpcJoints() {
  return {"hip", "knee", "ankle"};
}
std::vector<std::string> otherJoints() {
  return {"neck", "wrist"};
}

JointPdGainsDefaults controllerDefaults() {
  JointPdGainsDefaults defaults;
  defaults.kp = 250.0;
  defaults.kd = 15.0;
  defaults.torqueLimit = 500.0;
  return defaults;
}

JointPdGainsDefaults withoutTorqueLimit() {
  JointPdGainsDefaults defaults = controllerDefaults();
  defaults.torqueLimit = std::nullopt;
  return defaults;
}

mpc_config::JointPdGainsFile::JointGains joint(const std::string& name) {
  mpc_config::JointPdGainsFile::JointGains entry;
  entry.joint = name;
  return entry;
}

void expectSameGains(const JointPdGains& actual, const JointPdGains& expected) {
  EXPECT_EQ(actual.mpcJointKp, expected.mpcJointKp);
  EXPECT_EQ(actual.mpcJointKd, expected.mpcJointKd);
  EXPECT_EQ(actual.mpcJointTorqueLimit, expected.mpcJointTorqueLimit);
  EXPECT_EQ(actual.otherJointKp, expected.otherJointKp);
  EXPECT_EQ(actual.otherJointKd, expected.otherJointKd);
  EXPECT_EQ(actual.otherJointTorqueLimit, expected.otherJointTorqueLimit);
  EXPECT_EQ(actual.defaults.kp, expected.defaults.kp);
  EXPECT_EQ(actual.defaults.kd, expected.defaults.kd);
  EXPECT_EQ(actual.defaults.torqueLimit, expected.defaults.torqueLimit);
}

TEST(JointPdGainsFromConfigTest, AnEmptyFileIsTheControllersDefaults) {
  for (const JointPdGainsDefaults& defaults : {controllerDefaults(), withoutTorqueLimit()}) {
    const absl::StatusOr<JointPdGains> gains = jointPdGainsFromConfig(mpc_config::JointPdGainsFile{}, defaults, mpcJoints(), otherJoints());
    ASSERT_TRUE(gains.ok()) << gains.status();
    EXPECT_TRUE(gains->hasDimensions(mpcJoints().size(), otherJoints().size()));
    expectSameGains(*gains, defaultJointPdGains(defaults, mpcJoints(), otherJoints()));
  }
}

TEST(JointPdGainsFromConfigTest, EachLevelInheritsWhatItLeavesOut) {
  mpc_config::JointPdGainsFile file;
  file.default_gains.kp = 100.0;
  mpc_config::JointPdGainsFile::JointGains knee = joint("knee");
  knee.kd = 3.0;
  mpc_config::JointPdGainsFile::JointGains wrist = joint("wrist");
  wrist.torque_limit = 7.0;
  file.joint_gains = {knee, wrist};

  const absl::StatusOr<JointPdGains> gains = jointPdGainsFromConfig(file, controllerDefaults(), mpcJoints(), otherJoints());
  ASSERT_TRUE(gains.ok()) << gains.status();
  EXPECT_EQ(gains->defaults.kp, 100.0) << "default_gains over the controller's";
  EXPECT_EQ(gains->defaults.kd, 15.0);
  EXPECT_EQ(gains->defaults.torqueLimit, std::optional<double>(500.0));
  // An unnamed MPC joint has default_gains, a named one its own over them.
  EXPECT_EQ(gains->mpcJointKp[0], 100.0);
  EXPECT_EQ(gains->mpcJointKd[0], 15.0);
  EXPECT_EQ(gains->mpcJointKp[1], 100.0);
  EXPECT_EQ(gains->mpcJointKd[1], 3.0);
  EXPECT_EQ(gains->mpcJointTorqueLimit[1], 500.0);
  // An unnamed other joint has the scaled default kp and kd, a named one its own gains over default_gains.
  EXPECT_EQ(gains->otherJointKp[0], 100.0 * kOtherJointDefaultGainScale);
  EXPECT_EQ(gains->otherJointKd[0], 15.0 * kOtherJointDefaultGainScale);
  EXPECT_EQ(gains->otherJointTorqueLimit[0], 500.0);
  EXPECT_EQ(gains->otherJointKp[1], 100.0);
  EXPECT_EQ(gains->otherJointKd[1], 15.0);
  EXPECT_EQ(gains->otherJointTorqueLimit[1], 7.0);
}

TEST(JointPdGainsFromConfigTest, AControllerWithoutTorqueLimitsReadsNone) {
  mpc_config::JointPdGainsFile file;
  file.default_gains.torque_limit = -1.0;
  mpc_config::JointPdGainsFile::JointGains hip = joint("hip");
  hip.torque_limit = std::numeric_limits<double>::quiet_NaN();
  file.joint_gains = {hip};

  const absl::StatusOr<JointPdGains> gains = jointPdGainsFromConfig(file, withoutTorqueLimit(), mpcJoints(), otherJoints());
  ASSERT_TRUE(gains.ok()) << gains.status() << ": a limit the controller would not use cannot make it refuse the file";
  EXPECT_FALSE(gains->defaults.torqueLimit.has_value());
  for (Eigen::Index i = 0; i < gains->mpcJointTorqueLimit.size(); ++i) {
    EXPECT_TRUE(std::isinf(gains->mpcJointTorqueLimit[i]));
  }
  for (Eigen::Index i = 0; i < gains->otherJointTorqueLimit.size(); ++i) {
    EXPECT_TRUE(std::isinf(gains->otherJointTorqueLimit[i]));
  }
}

TEST(JointPdGainsFromConfigTest, AFileThatNamesJointsThisRobotDoesNotHaveIsAnotherRobotsAndRefused) {
  mpc_config::JointPdGainsFile file;
  mpc_config::JointPdGainsFile::JointGains tail = joint("tail");
  tail.kp = 1.0;
  mpc_config::JointPdGainsFile::JointGains wing = joint("wing");
  wing.kd = 2.0;
  file.joint_gains = {tail, joint(mpcJoints()[0]), wing};
  const absl::StatusOr<JointPdGains> gains = jointPdGainsFromConfig(file, controllerDefaults(), mpcJoints(), otherJoints());
  EXPECT_EQ(gains.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(gains.status().message(), HasSubstr("joint_gains names tail, wing, which this robot does not have"));
}

TEST(JointPdGainsFromConfigTest, AGainThatIsNotOneIsRefusedByItsField) {
  mpc_config::JointPdGainsFile negativeDefault;
  negativeDefault.default_gains.kd = -0.5;
  absl::StatusOr<JointPdGains> gains = jointPdGainsFromConfig(negativeDefault, controllerDefaults(), mpcJoints(), otherJoints());
  EXPECT_EQ(gains.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(gains.status().message(), HasSubstr("default_gains.kd is -0.5"));

  for (const double bad : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(), -1.0}) {
    mpc_config::JointPdGainsFile file;
    mpc_config::JointPdGainsFile::JointGains knee = joint("knee");
    knee.torque_limit = bad;
    file.joint_gains = {knee};
    gains = jointPdGainsFromConfig(file, controllerDefaults(), mpcJoints(), otherJoints());
    EXPECT_EQ(gains.status().code(), absl::StatusCode::kInvalidArgument) << bad;
    EXPECT_THAT(gains.status().message(), HasSubstr("joint_gains[joint=knee].torque_limit"));
  }
}

TEST(JointPdGainsFromConfigTest, AJointIsNamedAtMostOnceAndEveryEntryNamesOne) {
  mpc_config::JointPdGainsFile twice;
  twice.joint_gains = {joint("hip"), joint("knee"), joint("hip")};
  absl::StatusOr<JointPdGains> gains = jointPdGainsFromConfig(twice, controllerDefaults(), mpcJoints(), otherJoints());
  EXPECT_EQ(gains.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(gains.status().message(), HasSubstr("names the joint hip twice"));

  mpc_config::JointPdGainsFile unnamed;
  unnamed.joint_gains = {joint("hip"), joint(/*name=*/"")};
  gains = jointPdGainsFromConfig(unnamed, controllerDefaults(), mpcJoints(), otherJoints());
  EXPECT_EQ(gains.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(gains.status().message(), HasSubstr("joint_gains[1] names no joint"));
}

TEST(JointPdGainsFromConfigTest, TheRetiredDefaultBlockNamesItsReplacement) {
  const absl::StatusOr<humanoid_mpc_config::JointPdGainsFile> parsed =
      nproto::ParseTextproto<humanoid_mpc_config::JointPdGainsFile>("default {\n  kp: 1\n}\n", "joint_pd_gains.textproto");
  EXPECT_EQ(parsed.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(parsed.status().message(), HasSubstr("'default' is retired: the gains every joint starts from are default_gains"));
}

TEST(JointPdGainsFromConfigTest, AGainTheFileLeavesOutIsAbsentInTheStruct) {
  const absl::StatusOr<humanoid_mpc_config::JointPdGainsFile> message =
      nproto::ParseTextproto<humanoid_mpc_config::JointPdGainsFile>("joint_gains { joint: \"hip\" kd: 2 }\n", "joint_pd_gains.textproto");
  ASSERT_TRUE(message.ok()) << message.status();
  mpc_config::JointPdGainsFile file;
  ASSERT_TRUE(FromProto(*message, &file).ok());
  EXPECT_FALSE(file.default_gains.kp.has_value());
  ASSERT_EQ(file.joint_gains.size(), 1u);
  EXPECT_FALSE(file.joint_gains[0].kp.has_value()) << "inherited from default_gains, not 0";
  EXPECT_EQ(file.joint_gains[0].kd, std::optional<double>(2.0));
  EXPECT_FALSE(file.joint_gains[0].torque_limit.has_value());
}

}  // namespace
}  // namespace ocs2::humanoid
