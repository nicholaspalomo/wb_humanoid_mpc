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

// The configuration files as text (ConfigFiles.h): parseTaskFile(), parseReferenceFile() and parseJointPdGainsFile()
// read a text as strictly as the loaders read the file, and configFileIdentity() names a configuration file by its path
// from robot_models/ on, wherever the tree lies.

#include <fstream>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {
namespace {

using ::testing::AllOf;
using ::testing::HasSubstr;

constexpr char kTaskText[] = R"(# proto-file: humanoid_nmpc/humanoid_mpc_config/task_file.proto
# proto-message: humanoid_mpc_config.TaskFile
model_settings { robot_name: "g1" }
contact_estimator: "always_in_contact"
)";

/** Writes `content` to `name` in the test's scratch directory and returns its path. */
std::string writeFile(absl::string_view name, absl::string_view content) {
  const std::string path = absl::StrCat(::testing::TempDir(), "/", name);
  std::ofstream(path) << content;
  return path;
}

TEST(ConfigFileText, ATextParsesToWhatTheLoaderReadsFromTheSameFile) {
  const absl::StatusOr<mpc_config::TaskFile> parsed = parseTaskFile(kTaskText, /*source=*/"saved task file");
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  const absl::StatusOr<mpc_config::TaskFile> loaded = loadTaskFile(writeFile("task.textproto", kTaskText));
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_TRUE(*parsed == *loaded);
  EXPECT_EQ(parsed->model_settings.robot_name, "g1");
  EXPECT_EQ(parsed->contact_estimator, "always_in_contact");
}

TEST(ConfigFileText, AnErrorNamesTheSourceTheLineAndTheColumn) {
  const absl::StatusOr<mpc_config::TaskFile> unknown = parseTaskFile("model_settings { robot_nam: \"g1\" }\n", /*source=*/"from the bus");
  EXPECT_EQ(unknown.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(unknown.status().message(), AllOf(HasSubstr("from the bus:1:"), HasSubstr("robot_nam")));
  const absl::StatusOr<mpc_config::TaskFile> syntax = parseTaskFile("model_settings {\n", /*source=*/"from the bus");
  EXPECT_EQ(syntax.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(syntax.status().message(), HasSubstr("from the bus:"));
}

TEST(ConfigFileText, AFileOfAnotherMessageIsRefusedByItsHeader) {
  // A reference file sent as a task file: its leading comment names its message, and the parser refuses it as the loader does.
  const std::string reference = "# proto-message: humanoid_mpc_config.ReferenceFile\ntarget_displacement_velocity: 0.5\n";
  const absl::StatusOr<mpc_config::TaskFile> asTask = parseTaskFile(reference, /*source=*/"saved task file");
  EXPECT_EQ(asTask.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(asTask.status().message(), HasSubstr("humanoid_mpc_config.ReferenceFile"));
  const absl::StatusOr<mpc_config::ReferenceFile> asReference = parseReferenceFile(reference, /*source=*/"saved reference file");
  ASSERT_TRUE(asReference.ok()) << asReference.status();
  EXPECT_EQ(asReference->target_displacement_velocity, 0.5);
}

TEST(ConfigFileText, APdGainsTextParses) {
  const absl::StatusOr<mpc_config::JointPdGainsFile> gains =
      parseJointPdGainsFile("default_gains { kp: 100.0 kd: 2.0 }\n", /*source=*/"saved gains");
  ASSERT_TRUE(gains.ok()) << gains.status();
  EXPECT_EQ(gains->default_gains.kp, 100.0);
  EXPECT_FALSE(parseJointPdGainsFile("default_gains { kq: 1.0 }\n", /*source=*/"saved gains").ok());
}

TEST(ConfigFileIdentity, IsThePathFromTheLastRobotModelsDirectoryOn) {
  constexpr char kIdentity[] = "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto";
  // The robot bundle, a checkout, a relative path and a runfiles tree whose root itself lies below a robot_models/.
  EXPECT_EQ(configFileIdentity("/opt/wb-humanoid-robot/robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto"), kIdentity);
  EXPECT_EQ(configFileIdentity("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto"), kIdentity);
  EXPECT_EQ(configFileIdentity("/x/robot_models/runfiles/_main/robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto"), kIdentity);
  // `.` and `..` are resolved before the directory is looked for.
  EXPECT_EQ(configFileIdentity("/w/robot_models/unitree_g1/g1_wb_mpc/config/command/../mpc/./task.textproto"), kIdentity);
  // Two configurations of one robot are two identities.
  EXPECT_NE(configFileIdentity("/w/robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto"), kIdentity);
}

TEST(ConfigFileIdentity, APathWithoutRobotModelsIsItsNormalForm) {
  EXPECT_EQ(configFileIdentity("/tmp/store/./mpc/task.textproto"), "/tmp/store/mpc/task.textproto");
  EXPECT_EQ(configFileIdentity("/tmp/robot_models"), "/tmp/robot_models");
}

}  // namespace
}  // namespace ocs2::humanoid
