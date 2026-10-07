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

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/robot/JointPdGainsFromConfig.h"
#include "humanoid_common_mpc/mrt/JointPdGains.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"
#include "robot_core/ResourcePaths.h"

/*
 * The joint PD gains the MRT joint controllers share: the gains file they start from and watch (loadJointPdGains(), a
 * joint_pd_gains.textproto read strictly) and its write time. The parser of the gains used to live twice, inside each
 * controller's loadPdGains(), on the control thread: a document that failed part-way logged a warning and applied
 * whatever it had read before the error, with the hard-coded defaults for the rest. It is now a pure function the
 * caller's thread runs (jointPdGainsFromConfig(), test/config/robot/), and a file it refuses changes nothing.
 */

namespace ocs2::humanoid {
namespace {

const absl::NoDestructor<std::vector<std::string>> kMpcJoints(std::vector<std::string>{"left_knee", "right_knee"});
const absl::NoDestructor<std::vector<std::string>> kOtherJoints(std::vector<std::string>{"left_wrist", "right_wrist"});

JointPdGainsDefaults controllerDefaults() {
  JointPdGainsDefaults defaults;
  defaults.kp = 250.0;
  defaults.kd = 15.0;
  defaults.torqueLimit = 500.0;
  return defaults;
}

absl::StatusOr<std::optional<JointPdGains>> load(const std::string& file) {
  return loadJointPdGains(file, controllerDefaults(), *kMpcJoints, *kOtherJoints);
}

/** `content` written to `name` in the test's temporary directory; its path. */
std::string writeTemporaryFile(absl::string_view name, absl::string_view content) {
  const std::string file = absl::StrCat(::testing::TempDir(), "/", name);
  std::ofstream(file, std::ios::trunc) << content;
  return file;
}

/** Whether `a` and `b` are the same gains, bit for bit. */
bool sameGains(const JointPdGains& a, const JointPdGains& b) {
  return a.mpcJointKp == b.mpcJointKp && a.mpcJointKd == b.mpcJointKd && a.mpcJointTorqueLimit == b.mpcJointTorqueLimit &&
         a.otherJointKp == b.otherJointKp && a.otherJointKd == b.otherJointKd && a.otherJointTorqueLimit == b.otherJointTorqueLimit &&
         a.defaults.kp == b.defaults.kp && a.defaults.kd == b.defaults.kd && a.defaults.torqueLimit == b.defaults.torqueLimit;
}

TEST(JointPdGains, WithoutADocumentTheMpcJointsGetTheDefaultsAndTheOthersAFractionOfThem) {
  const JointPdGains gains = defaultJointPdGains(controllerDefaults(), *kMpcJoints, *kOtherJoints);
  ASSERT_TRUE(gains.hasDimensions(kMpcJoints->size(), kOtherJoints->size()));
  for (size_t i = 0; i < kMpcJoints->size(); ++i) {
    EXPECT_EQ(gains.mpcJointKp[i], 250.0);
    EXPECT_EQ(gains.mpcJointKd[i], 15.0);
    EXPECT_EQ(gains.mpcJointTorqueLimit[i], 500.0);
  }
  for (size_t i = 0; i < kOtherJoints->size(); ++i) {
    EXPECT_EQ(gains.otherJointKp[i], 250.0 * kOtherJointDefaultGainScale);
    EXPECT_EQ(gains.otherJointKd[i], 15.0 * kOtherJointDefaultGainScale);
    EXPECT_EQ(gains.otherJointTorqueLimit[i], 500.0);
  }
}

TEST(JointPdGains, TheWriteTimeOfAGainsFileIsTheEpochWhenThereIsNone) {
  EXPECT_EQ(jointPdGainsFileWriteTime(""), std::filesystem::file_time_type());
  EXPECT_EQ(jointPdGainsFileWriteTime("/nonexistent/joint_pd_gains.textproto"), std::filesystem::file_time_type());

  const std::string file = absl::StrCat(::testing::TempDir(), "/write_time_joint_pd_gains.textproto");
  std::ofstream(file) << "default_gains { kp: 1.0 }\n";
  const std::filesystem::file_time_type written = std::filesystem::last_write_time(file) - std::chrono::hours(1);
  std::filesystem::last_write_time(file, written);
  EXPECT_EQ(jointPdGainsFileWriteTime(file), written);
  std::filesystem::remove(file);
}

TEST(JointPdGains, HasDimensionsChecksEveryVector) {
  JointPdGains gains = defaultJointPdGains(controllerDefaults(), *kMpcJoints, *kOtherJoints);
  EXPECT_TRUE(gains.hasDimensions(/*numMpcJoints=*/2, /*numOtherJoints=*/2));
  EXPECT_FALSE(gains.hasDimensions(/*numMpcJoints=*/2, /*numOtherJoints=*/3));
  gains.otherJointTorqueLimit.resize(1);
  EXPECT_FALSE(gains.hasDimensions(/*numMpcJoints=*/2, /*numOtherJoints=*/2));
}

TEST(JointPdGains, AGainsFileIsATextprotoReadStrictly) {
  const std::string file = writeTemporaryFile("strict_joint_pd_gains.textproto",
                                              "default_gains { kp: 100.0 kd: 5.0 }\n"
                                              "joint_gains { joint: \"left_knee\" kp: 1.0 }\n"
                                              "joint_gains { joint: \"left_wrist\" kd: 2.0 torque_limit: 3.0 }\n");
  const absl::StatusOr<std::optional<JointPdGains>> gains = load(file);
  ASSERT_TRUE(gains.ok()) << gains.status();
  if (!gains->has_value()) GTEST_FAIL() << "the gains file was not found";
  // The gains of the file as jointPdGainsFromConfig() resolves them.
  const absl::StatusOr<mpc_config::JointPdGainsFile> typed = loadJointPdGainsFile(file);
  ASSERT_TRUE(typed.ok()) << typed.status();
  const absl::StatusOr<JointPdGains> converted = jointPdGainsFromConfig(*typed, controllerDefaults(), *kMpcJoints, *kOtherJoints);
  ASSERT_TRUE(converted.ok()) << converted.status();
  EXPECT_TRUE(sameGains(**gains, *converted));
  EXPECT_EQ((*gains)->mpcJointKp[0], 1.0) << "a named joint gets its own gains";
  EXPECT_EQ((*gains)->otherJointTorqueLimit[0], 3.0);

  // An unknown field is refused where it is written, a gain jointPdGainsFromConfig() refuses by its field; both name the file.
  const std::string unknown = writeTemporaryFile("unknown_joint_pd_gains.textproto", "default_gains { kp: 1.0 }\nstiffness: 2.0\n");
  const absl::StatusOr<std::optional<JointPdGains>> refused = load(unknown);
  ASSERT_FALSE(refused.ok());
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(refused.status().message().find(absl::StrCat(unknown, ":2:")), absl::string_view::npos) << refused.status();
  const std::string negative = writeTemporaryFile("negative_joint_pd_gains.textproto", "default_gains { kd: -1.0 }\n");
  const absl::StatusOr<std::optional<JointPdGains>> negativeGain = load(negative);
  ASSERT_FALSE(negativeGain.ok());
  EXPECT_EQ(negativeGain.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(negativeGain.status().message().find(negative), absl::string_view::npos) << negativeGain.status();
  EXPECT_NE(negativeGain.status().message().find("default_gains.kd"), absl::string_view::npos) << negativeGain.status();
}

TEST(JointPdGains, WithoutAGainsFileThereAreNoGainsToLoad) {
  for (const std::string& file : {std::string(), std::string("/nonexistent/joint_pd_gains.textproto")}) {
    const absl::StatusOr<std::optional<JointPdGains>> gains = load(file);
    ASSERT_TRUE(gains.ok()) << file << ": " << gains.status();
    EXPECT_FALSE(gains->has_value()) << file;
  }
}

// Every robot's shipped gains file, which the refusal of a malformed file must not catch: the parse is stricter than
// the one it replaced (a negative, non-finite or non-numeric gain used to be applied or turned into defaults).
TEST(JointPdGains, EveryShippedGainsFileParses) {
  // LINT.IfChange(shipped_pd_gains_files)
  const std::vector<std::string> shipped{
      "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/controller/joint_pd_gains.textproto",
      "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/controller/joint_pd_gains.textproto",
      "robot_models/unitree_g1/g1_centroidal_mpc/config/controller/joint_pd_gains.textproto",
      "robot_models/unitree_g1/g1_wb_mpc/config/controller/joint_pd_gains.textproto",
      "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/controller/joint_pd_gains.textproto",
  };
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/BUILD.bazel:shipped_pd_gains_files_data)
  for (const std::string& relativePath : shipped) {
    const absl::StatusOr<std::string> file = robot::resolveResourcePath(relativePath);
    ASSERT_TRUE(file.ok()) << file.status();
    EXPECT_NE(jointPdGainsFileWriteTime(*file), std::filesystem::file_time_type()) << relativePath;
    const absl::StatusOr<mpc_config::JointPdGainsFile> typed = loadJointPdGainsFile(*file);
    ASSERT_TRUE(typed.ok()) << typed.status();
    // A file of another robot's joints is refused (jointPdGainsFromConfig()): the file is read as its own robot's,
    // whose joints it names, all of them on the MPC side here.
    std::vector<std::string> joints;
    for (const mpc_config::JointPdGainsFile::JointGains& entry : typed->joint_gains) joints.push_back(entry.joint);
    const std::vector<std::string> noOtherJoints;
    const absl::StatusOr<std::optional<JointPdGains>> loaded = loadJointPdGains(*file, controllerDefaults(), joints, noOtherJoints);
    ASSERT_TRUE(loaded.ok()) << relativePath << ": " << loaded.status();
    if (!loaded->has_value()) GTEST_FAIL() << relativePath << ": the gains file was not found";
    const absl::StatusOr<JointPdGains> converted = jointPdGainsFromConfig(*typed, controllerDefaults(), joints, noOtherJoints);
    ASSERT_TRUE(converted.ok()) << converted.status();
    EXPECT_TRUE(sameGains(**loaded, *converted)) << relativePath;
    // Read as the two-knee robot of the other tests, it is another robot's file.
    EXPECT_EQ(load(*file).status().code(), absl::StatusCode::kInvalidArgument) << relativePath;
  }
}

}  // namespace
}  // namespace ocs2::humanoid
