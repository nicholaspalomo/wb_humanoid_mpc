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

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/mrt/JointPdGains.h"
#include "robot_core/ResourcePaths.h"

/*
 * The joint_pd_gains.yaml parser the MRT joint controllers share. It used to live twice, inside each controller's
 * loadPdGains(), on the control thread: a document that failed part-way logged a warning and applied whatever it had
 * read before the error, with the hard-coded defaults for the rest. It is now a pure function the caller's thread runs,
 * and a document it refuses changes nothing.
 */

namespace ocs2::humanoid {
namespace {

const std::vector<std::string> kMpcJoints{"left_knee", "right_knee"};
const std::vector<std::string> kOtherJoints{"left_wrist", "right_wrist"};

JointPdGainsDefaults controllerDefaults() {
  JointPdGainsDefaults defaults;
  defaults.kp = 250.0;
  defaults.kd = 15.0;
  defaults.torqueLimit = 500.0;
  return defaults;
}

absl::StatusOr<JointPdGains> parse(absl::string_view yaml) {
  return parseJointPdGainsYaml(yaml, controllerDefaults(), kMpcJoints, kOtherJoints);
}

TEST(JointPdGains, WithoutADocumentTheMpcJointsGetTheDefaultsAndTheOthersAFractionOfThem) {
  const JointPdGains gains = defaultJointPdGains(controllerDefaults(), kMpcJoints, kOtherJoints);
  ASSERT_TRUE(gains.hasDimensions(kMpcJoints.size(), kOtherJoints.size()));
  for (size_t i = 0; i < kMpcJoints.size(); ++i) {
    EXPECT_EQ(gains.mpcJointKp[i], 250.0);
    EXPECT_EQ(gains.mpcJointKd[i], 15.0);
    EXPECT_EQ(gains.mpcJointTorqueLimit[i], 500.0);
  }
  for (size_t i = 0; i < kOtherJoints.size(); ++i) {
    EXPECT_EQ(gains.otherJointKp[i], 250.0 * kOtherJointDefaultGainScale);
    EXPECT_EQ(gains.otherJointKd[i], 15.0 * kOtherJointDefaultGainScale);
    EXPECT_EQ(gains.otherJointTorqueLimit[i], 500.0);
  }
}

TEST(JointPdGains, ANamedJointGetsItsOwnGainsOverTheDocumentsDefaults) {
  const absl::StatusOr<JointPdGains> gains = parse(
      "default_gains:\n"
      "  kp: 100.0\n"
      "  kd: 5.0\n"
      "joint_gains:\n"
      "  left_knee: {kp: 1.0}\n"
      "  left_wrist: {kd: 2.0, torque_limit: 3.0}\n"
      "  a_joint_of_another_robot: {kp: 9.0}\n");
  ASSERT_TRUE(gains.ok()) << gains.status();
  EXPECT_EQ(gains->defaults.kp, 100.0);
  EXPECT_EQ(gains->defaults.kd, 5.0);
  EXPECT_EQ(gains->defaults.torqueLimit, 500.0) << "a default the document does not set is the controller's";
  // A named MPC joint: its own kp, the document's default kd and the controller's torque limit.
  EXPECT_EQ(gains->mpcJointKp[0], 1.0);
  EXPECT_EQ(gains->mpcJointKd[0], 5.0);
  EXPECT_EQ(gains->mpcJointTorqueLimit[0], 500.0);
  // An MPC joint the document does not name: the document's defaults.
  EXPECT_EQ(gains->mpcJointKp[1], 100.0);
  EXPECT_EQ(gains->mpcJointKd[1], 5.0);
  // A named other joint: its entry over the defaults, unscaled.
  EXPECT_EQ(gains->otherJointKp[0], 100.0);
  EXPECT_EQ(gains->otherJointKd[0], 2.0);
  EXPECT_EQ(gains->otherJointTorqueLimit[0], 3.0);
  // An other joint the document does not name: a fraction of the document's defaults.
  EXPECT_EQ(gains->otherJointKp[1], 100.0 * kOtherJointDefaultGainScale);
  EXPECT_EQ(gains->otherJointKd[1], 5.0 * kOtherJointDefaultGainScale);
  EXPECT_EQ(gains->otherJointTorqueLimit[1], 500.0);
}

TEST(JointPdGains, ADocumentWithoutGainsOrWithEmptySectionsIsTheDefaults) {
  const JointPdGains defaults = defaultJointPdGains(controllerDefaults(), kMpcJoints, kOtherJoints);
  for (const char* yaml : {"some_other_key: 1\n", "default_gains:\njoint_gains:\n", "joint_gains:\n  left_knee:\n"}) {
    const absl::StatusOr<JointPdGains> gains = parse(yaml);
    ASSERT_TRUE(gains.ok()) << yaml << ": " << gains.status();
    EXPECT_EQ(gains->mpcJointKp, defaults.mpcJointKp) << yaml;
    EXPECT_EQ(gains->mpcJointKd, defaults.mpcJointKd) << yaml;
    EXPECT_EQ(gains->otherJointKp, defaults.otherJointKp) << yaml;
    EXPECT_EQ(gains->otherJointTorqueLimit, defaults.otherJointTorqueLimit) << yaml;
  }
}

TEST(JointPdGains, AMalformedDocumentIsRefusedNamingWhatIsWrong) {
  struct Case {
    const char* yaml;
    const char* expected;  // in the message
  };
  for (const Case& c : std::vector<Case>{
           {"default_gains: {kp: [1, 2\n", "not YAML"},
           {"", "empty"},
           {"# only a comment\n", "empty"},
           {"42\n", "not a map"},
           {"- default_gains\n", "not a map"},
           {"default_gains: 5\n", "default_gains is not a map"},
           {"joint_gains: [left_knee, right_knee]\n", "joint_gains is not a map"},
           {"joint_gains:\n  left_knee: 5\n", "joint_gains.left_knee is not a map"},
           {"default_gains: {kp: stiff}\n", "default_gains.kp is 'stiff'"},
           {"default_gains: {kd: -1.0}\n", "default_gains.kd"},
           {"joint_gains: {right_knee: {kp: .nan}}\n", "joint_gains.right_knee.kp"},
           {"joint_gains: {left_wrist: {torque_limit: .inf}}\n", "joint_gains.left_wrist.torque_limit"},
           {"joint_gains: {left_knee: {kd: }}\n", "joint_gains.left_knee.kd"},
       }) {
    const absl::StatusOr<JointPdGains> gains = parse(c.yaml);
    ASSERT_FALSE(gains.ok()) << "accepted: " << c.yaml;
    EXPECT_EQ(gains.status().code(), absl::StatusCode::kInvalidArgument) << c.yaml;
    EXPECT_NE(gains.status().message().find(c.expected), absl::string_view::npos) << c.yaml << ": " << gains.status().message();
  }
}

// The whole-body controller commands no torque limit. Its loader never read the key, so a value it would not use must
// not make it refuse a document, not even the +infinity that is its own default.
TEST(JointPdGains, WithoutATorqueLimitInTheDefaultsTheKeyIsNotRead) {
  JointPdGainsDefaults noTorqueLimit = controllerDefaults();
  noTorqueLimit.torqueLimit = std::nullopt;
  for (const char* yaml :
       {"default_gains: {kp: 100.0, torque_limit: -1.0}\n", "default_gains: {kp: 100.0, torque_limit: strong}\n",
        "default_gains: {kp: 100.0, torque_limit: .inf}\n", "joint_gains: {left_wrist: {kp: 100.0, torque_limit: .nan}}\n"}) {
    const absl::StatusOr<JointPdGains> gains = parseJointPdGainsYaml(yaml, noTorqueLimit, kMpcJoints, kOtherJoints);
    ASSERT_TRUE(gains.ok()) << yaml << ": " << gains.status();
    EXPECT_FALSE(gains->defaults.torqueLimit.has_value()) << yaml;
    for (Eigen::Index i = 0; i < gains->mpcJointTorqueLimit.size(); ++i) {
      EXPECT_EQ(gains->mpcJointTorqueLimit[i], std::numeric_limits<scalar_t>::infinity()) << yaml;
    }
    for (Eigen::Index i = 0; i < gains->otherJointTorqueLimit.size(); ++i) {
      EXPECT_EQ(gains->otherJointTorqueLimit[i], std::numeric_limits<scalar_t>::infinity()) << yaml;
    }
  }
  // The gains it does read are still checked.
  EXPECT_FALSE(parseJointPdGainsYaml("default_gains: {kp: -1.0}\n", noTorqueLimit, kMpcJoints, kOtherJoints).ok());
  // Positive control: with a torque limit in the defaults the same key is read, and refused.
  EXPECT_FALSE(parse("default_gains: {kp: 100.0, torque_limit: -1.0}\n").ok());
}

TEST(JointPdGains, TheWriteTimeOfAGainsFileIsTheEpochWhenThereIsNone) {
  EXPECT_EQ(jointPdGainsFileWriteTime(""), std::filesystem::file_time_type());
  EXPECT_EQ(jointPdGainsFileWriteTime("/nonexistent/joint_pd_gains.yaml"), std::filesystem::file_time_type());

  const std::string file = absl::StrCat(::testing::TempDir(), "/write_time_joint_pd_gains.yaml");
  std::ofstream(file) << "default_gains: {kp: 1.0}\n";
  const std::filesystem::file_time_type written = std::filesystem::last_write_time(file) - std::chrono::hours(1);
  std::filesystem::last_write_time(file, written);
  EXPECT_EQ(jointPdGainsFileWriteTime(file), written);
  std::filesystem::remove(file);
}

TEST(JointPdGains, HasDimensionsChecksEveryVector) {
  JointPdGains gains = defaultJointPdGains(controllerDefaults(), kMpcJoints, kOtherJoints);
  EXPECT_TRUE(gains.hasDimensions(/*numMpcJoints=*/2, /*numOtherJoints=*/2));
  EXPECT_FALSE(gains.hasDimensions(/*numMpcJoints=*/2, /*numOtherJoints=*/3));
  gains.otherJointTorqueLimit.resize(1);
  EXPECT_FALSE(gains.hasDimensions(/*numMpcJoints=*/2, /*numOtherJoints=*/2));
}

TEST(JointPdGains, ReadingAFileThatCannotBeOpenedIsNotFound) {
  const absl::StatusOr<std::string> text = readJointPdGainsFile("/nonexistent/joint_pd_gains.yaml");
  EXPECT_EQ(text.status().code(), absl::StatusCode::kNotFound);
}

// Every robot's shipped gains file, which the refusal of a malformed document must not catch: the parse is stricter
// than the one it replaced (a negative, non-finite or non-numeric gain used to be applied or turned into defaults).
TEST(JointPdGains, EveryShippedGainsFileParses) {
  // LINT.IfChange(shipped_pd_gains_files)
  const std::vector<std::string> shipped{
      "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/controller/joint_pd_gains.yaml",
      "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/controller/joint_pd_gains.yaml",
      "robot_models/unitree_g1/g1_centroidal_mpc/config/controller/joint_pd_gains.yaml",
      "robot_models/unitree_g1/g1_wb_mpc/config/controller/joint_pd_gains.yaml",
      "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/controller/joint_pd_gains.yaml",
  };
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/BUILD.bazel:shipped_pd_gains_files_data)
  for (const std::string& relativePath : shipped) {
    const absl::StatusOr<std::string> file = robot::resolveResourcePath(relativePath);
    ASSERT_TRUE(file.ok()) << file.status();
    const absl::StatusOr<std::string> text = readJointPdGainsFile(*file);
    ASSERT_TRUE(text.ok()) << text.status();
    const absl::StatusOr<JointPdGains> gains = parse(*text);
    EXPECT_TRUE(gains.ok()) << relativePath << ": " << gains.status();
  }
}

}  // namespace
}  // namespace ocs2::humanoid
