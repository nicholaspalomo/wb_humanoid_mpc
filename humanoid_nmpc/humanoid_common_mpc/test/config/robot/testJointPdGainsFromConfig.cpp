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

// The joint PD gains of a typed joint PD gains file: every gain is the file's -
// default_gains sets kp, kd and max_joint_torque, each joint inherits what it
// leaves out from default_gains -, the controller constants are read with their
// ranges, and a gain that is not one, a joint named twice and an entry without
// a joint are refused by their field.

#include <cmath>
#include <limits>
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

/** A file whose default_gains sets every gain, and names no joint. */
mpc_config::JointPdGainsFile completeDefaults() {
  mpc_config::JointPdGainsFile file;
  file.default_gains.kp = 250.0;
  file.default_gains.kd = 15.0;
  file.default_gains.max_joint_torque = 500.0;
  return file;
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
  EXPECT_EQ(actual.defaults.maxJointTorque, expected.defaults.maxJointTorque);
  EXPECT_EQ(actual.gravityCompDampingFraction,
            expected.gravityCompDampingFraction);
  EXPECT_EQ(actual.maxPolicyJointError, expected.maxPolicyJointError);
  EXPECT_EQ(actual.divergedPolicyJointTargetBound,
            expected.divergedPolicyJointTargetBound);
}

TEST(JointPdGainsFromConfigTest,
     AFileNamingNoJointGivesEveryJointItsDefaultGains) {
  const absl::StatusOr<JointPdGains> gains =
      jointPdGainsFromConfig(completeDefaults(), mpcJoints(), otherJoints());
  ASSERT_TRUE(gains.ok()) << gains.status();
  EXPECT_TRUE(gains->hasDimensions(mpcJoints().size(), otherJoints().size()));
  expectSameGains(*gains,
                  defaultJointPdGains(
                      JointPdGainsDefaults{
                          .kp = 250.0, .kd = 15.0, .maxJointTorque = 500.0},
                      mpcJoints(), otherJoints()));
}

TEST(JointPdGainsFromConfigTest, DefaultGainsMustSetEveryGain) {
  EXPECT_THAT(jointPdGainsFromConfig(mpc_config::JointPdGainsFile{},
                                     mpcJoints(), otherJoints())
                  .status()
                  .message(),
              HasSubstr("default_gains.kp is not set"));
  mpc_config::JointPdGainsFile noKd = completeDefaults();
  noKd.default_gains.kd.reset();
  EXPECT_THAT(jointPdGainsFromConfig(noKd, mpcJoints(), otherJoints())
                  .status()
                  .message(),
              HasSubstr("default_gains.kd is not set"));
  mpc_config::JointPdGainsFile noLimit = completeDefaults();
  noLimit.default_gains.max_joint_torque.reset();
  const absl::StatusOr<JointPdGains> refused =
      jointPdGainsFromConfig(noLimit, mpcJoints(), otherJoints());
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(refused.status().message(),
              HasSubstr("default_gains.max_joint_torque is not set"));
}

TEST(JointPdGainsFromConfigTest,
     EachJointInheritsWhatItLeavesOutFromDefaultGains) {
  mpc_config::JointPdGainsFile file = completeDefaults();
  file.default_gains.kp = 100.0;
  mpc_config::JointPdGainsFile::JointGains knee = joint("knee");
  knee.kd = 3.0;
  mpc_config::JointPdGainsFile::JointGains wrist = joint("wrist");
  wrist.max_joint_torque = 7.0;
  file.joint_gains = {knee, wrist};

  const absl::StatusOr<JointPdGains> gains =
      jointPdGainsFromConfig(file, mpcJoints(), otherJoints());
  ASSERT_TRUE(gains.ok()) << gains.status();
  EXPECT_EQ(gains->defaults.kp, 100.0);
  EXPECT_EQ(gains->defaults.kd, 15.0);
  EXPECT_EQ(gains->defaults.maxJointTorque, 500.0);
  // An unnamed MPC joint has default_gains, a named one its own over them.
  EXPECT_EQ(gains->mpcJointKp[0], 100.0);
  EXPECT_EQ(gains->mpcJointKd[0], 15.0);
  EXPECT_EQ(gains->mpcJointKp[1], 100.0);
  EXPECT_EQ(gains->mpcJointKd[1], 3.0);
  EXPECT_EQ(gains->mpcJointTorqueLimit[1], 500.0);
  // An unnamed other joint has the scaled default kp and kd, a named one its
  // own gains over default_gains.
  EXPECT_EQ(gains->otherJointKp[0], 100.0 * kOtherJointDefaultGainScale);
  EXPECT_EQ(gains->otherJointKd[0], 15.0 * kOtherJointDefaultGainScale);
  EXPECT_EQ(gains->otherJointTorqueLimit[0], 500.0);
  EXPECT_EQ(gains->otherJointKp[1], 100.0);
  EXPECT_EQ(gains->otherJointKd[1], 15.0);
  EXPECT_EQ(gains->otherJointTorqueLimit[1], 7.0);
}

TEST(JointPdGainsFromConfigTest, TheControllerConstantsAreReadWithTheirRanges) {
  // Absent: the schema's defaults, the values the controllers had hard-coded.
  const absl::StatusOr<JointPdGains> defaults =
      jointPdGainsFromConfig(completeDefaults(), mpcJoints(), otherJoints());
  ASSERT_TRUE(defaults.ok()) << defaults.status();
  EXPECT_EQ(defaults->gravityCompDampingFraction, 0.2);
  EXPECT_EQ(defaults->maxPolicyJointError, 0.5);
  EXPECT_EQ(defaults->divergedPolicyJointTargetBound, 0.2);

  mpc_config::JointPdGainsFile file = completeDefaults();
  file.gravity_comp_damping_fraction = 0.35;
  file.max_policy_joint_error = 0.8;
  file.diverged_policy_joint_target_bound = 0.0;
  const absl::StatusOr<JointPdGains> set =
      jointPdGainsFromConfig(file, mpcJoints(), otherJoints());
  ASSERT_TRUE(set.ok()) << set.status();
  EXPECT_EQ(set->gravityCompDampingFraction, 0.35);
  EXPECT_EQ(set->maxPolicyJointError, 0.8);
  EXPECT_EQ(set->divergedPolicyJointTargetBound, 0.0);

  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const double bad : {-0.1, 1.5, nan}) {
    mpc_config::JointPdGainsFile badFraction = completeDefaults();
    badFraction.gravity_comp_damping_fraction = bad;
    EXPECT_THAT(jointPdGainsFromConfig(badFraction, mpcJoints(), otherJoints())
                    .status()
                    .message(),
                HasSubstr("gravity_comp_damping_fraction"))
        << bad;
  }
  for (const double bad :
       {0.0, -0.1, std::numeric_limits<double>::infinity(), nan}) {
    mpc_config::JointPdGainsFile badError = completeDefaults();
    badError.max_policy_joint_error = bad;
    EXPECT_THAT(jointPdGainsFromConfig(badError, mpcJoints(), otherJoints())
                    .status()
                    .message(),
                HasSubstr("max_policy_joint_error"))
        << bad;
  }
  for (const double bad :
       {-0.1, std::numeric_limits<double>::infinity(), nan}) {
    mpc_config::JointPdGainsFile badBound = completeDefaults();
    badBound.diverged_policy_joint_target_bound = bad;
    EXPECT_THAT(jointPdGainsFromConfig(badBound, mpcJoints(), otherJoints())
                    .status()
                    .message(),
                HasSubstr("diverged_policy_joint_target_bound"))
        << bad;
  }
}

TEST(JointPdGainsFromConfigTest,
     AFileThatNamesJointsThisRobotDoesNotHaveIsAnotherRobotsAndRefused) {
  mpc_config::JointPdGainsFile file = completeDefaults();
  mpc_config::JointPdGainsFile::JointGains tail = joint("tail");
  tail.kp = 1.0;
  mpc_config::JointPdGainsFile::JointGains wing = joint("wing");
  wing.kd = 2.0;
  file.joint_gains = {tail, joint(mpcJoints()[0]), wing};
  const absl::StatusOr<JointPdGains> gains =
      jointPdGainsFromConfig(file, mpcJoints(), otherJoints());
  EXPECT_EQ(gains.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      gains.status().message(),
      HasSubstr(
          "joint_gains names tail, wing, which this robot does not have"));
}

TEST(JointPdGainsFromConfigTest, AGainThatIsNotOneIsRefusedByItsField) {
  mpc_config::JointPdGainsFile negativeDefault = completeDefaults();
  negativeDefault.default_gains.kd = -0.5;
  absl::StatusOr<JointPdGains> gains =
      jointPdGainsFromConfig(negativeDefault, mpcJoints(), otherJoints());
  EXPECT_EQ(gains.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(gains.status().message(), HasSubstr("default_gains.kd is -0.5"));

  for (const double bad : {std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN(), -1.0}) {
    mpc_config::JointPdGainsFile file = completeDefaults();
    mpc_config::JointPdGainsFile::JointGains knee = joint("knee");
    knee.max_joint_torque = bad;
    file.joint_gains = {knee};
    gains = jointPdGainsFromConfig(file, mpcJoints(), otherJoints());
    EXPECT_EQ(gains.status().code(), absl::StatusCode::kInvalidArgument) << bad;
    EXPECT_THAT(gains.status().message(),
                HasSubstr("joint_gains[joint=knee].max_joint_torque"));
  }
}

TEST(JointPdGainsFromConfigTest, AJointIsNamedAtMostOnceAndEveryEntryNamesOne) {
  mpc_config::JointPdGainsFile twice = completeDefaults();
  twice.joint_gains = {joint("hip"), joint("knee"), joint("hip")};
  absl::StatusOr<JointPdGains> gains =
      jointPdGainsFromConfig(twice, mpcJoints(), otherJoints());
  EXPECT_EQ(gains.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(gains.status().message(), HasSubstr("names the joint hip twice"));

  mpc_config::JointPdGainsFile unnamed = completeDefaults();
  unnamed.joint_gains = {joint("hip"), joint(/*name=*/"")};
  gains = jointPdGainsFromConfig(unnamed, mpcJoints(), otherJoints());
  EXPECT_EQ(gains.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(gains.status().message(),
              HasSubstr("joint_gains[1] names no joint"));
}

TEST(JointPdGainsFromConfigTest, TheRetiredDefaultBlockNamesItsReplacement) {
  const absl::StatusOr<humanoid_mpc_config::JointPdGainsFile> parsed =
      nproto::ParseTextproto<humanoid_mpc_config::JointPdGainsFile>(
          "default {\n  kp: 1\n}\n", "joint_pd_gains.textproto");
  EXPECT_EQ(parsed.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(parsed.status().message(),
              HasSubstr("'default' is retired: the gains every joint starts "
                        "from are default_gains"));
}

}  // namespace
}  // namespace ocs2::humanoid
