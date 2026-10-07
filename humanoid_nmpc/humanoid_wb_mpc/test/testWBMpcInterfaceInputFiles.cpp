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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_wb_mpc/WBMpcInterface.h"

/**
 * The path forms of WBMpcInterface::Create() and CreateControllerModels() on input files that are missing, do not parse
 * or do not convert. Both check the three files for existence before anything reads them and return NotFound naming the
 * path; a task or reference file that does not parse is an InvalidArgument naming its line and column; and an error of
 * the typed files' conversion is prefixed with the file it is about. None of these builds a CppAD model: every case
 * returns before the problem is set up.
 */
namespace ocs2::humanoid {
namespace {

std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

std::string readFile(const std::string& path) {
  std::ifstream in(path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

/** `content` with `from` replaced by `to` exactly once; fails the test when `from` does not occur exactly once. */
std::string replacedOnce(const std::string& content, absl::string_view from, absl::string_view to) {
  const std::string::size_type position = content.find(from);
  EXPECT_NE(position, std::string::npos) << "'" << from << "' not found";
  if (position == std::string::npos) return content;
  EXPECT_EQ(content.find(from, position + 1), std::string::npos) << "'" << from << "' occurs more than once";
  std::string result = content;
  result.replace(position, from.size(), std::string(to));
  return result;
}

/** The 1-based line of `content` that `needle` starts on; 0 when it does not occur. */
int lineOf(const std::string& content, absl::string_view needle) {
  const std::string::size_type position = content.find(needle);
  if (position == std::string::npos) return 0;
  int line = 1;
  for (std::string::size_type i = 0; i < position; ++i) {
    if (content[i] == '\n') ++line;
  }
  return line;
}

class WBMpcInterfaceInputFilesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    taskFile_ = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto");
    urdfFile_ = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
    referenceFile_ = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto");
    ASSERT_FALSE(taskFile_.empty() || urdfFile_.empty() || referenceFile_.empty()) << "the G1 whole-body files are not in the runfiles";
    missing_ = (std::filesystem::path(testing::TempDir()) / "testWBMpcInterfaceInputFiles_missing.textproto").string();
    ASSERT_FALSE(std::filesystem::exists(missing_));
  }

  /** Create() must return, not throw, and return NotFound naming the missing path. */
  void expectNotFoundNamingTheMissingFile(const absl::StatusOr<std::unique_ptr<WBMpcInterface>>& created, absl::string_view what) const {
    EXPECT_EQ(created.status().code(), absl::StatusCode::kNotFound) << what << ": " << created.status();
    EXPECT_TRUE(absl::StrContains(created.status().message(), missing_)) << what << ": " << created.status();
    EXPECT_TRUE(absl::StrContains(created.status().message(), what)) << created.status();
  }

  /** `content`, written for the test as the file `name`. */
  static std::string written(absl::string_view name, const std::string& content) {
    const std::string path =
        (std::filesystem::path(testing::TempDir()) / absl::StrCat("testWBMpcInterfaceInputFiles_", name, ".textproto")).string();
    std::ofstream(path) << content;
    return path;
  }

  std::string taskFile_, urdfFile_, referenceFile_, missing_;
};

}  // namespace

TEST_F(WBMpcInterfaceInputFilesTest, AMissingTaskFileIsNotFoundNamingItsPath) {
  expectNotFoundNamingTheMissingFile(WBMpcInterface::Create(missing_, urdfFile_, referenceFile_), "task file");
}

TEST_F(WBMpcInterfaceInputFilesTest, AMissingUrdfIsNotFoundBeforeTheModelSettingsReadIt) {
  expectNotFoundNamingTheMissingFile(WBMpcInterface::Create(taskFile_, missing_, referenceFile_), "URDF file");
}

TEST_F(WBMpcInterfaceInputFilesTest, AMissingReferenceFileIsNotFoundNamingItsPath) {
  expectNotFoundNamingTheMissingFile(WBMpcInterface::Create(taskFile_, urdfFile_, missing_), "reference file");
}

TEST_F(WBMpcInterfaceInputFilesTest, ATaskFileThatDoesNotParseIsAnInvalidArgumentNamingItsLine) {
  // Positive control, without building the problem: with all three files present Create() gets past the check and reads
  // the task file, whose interface.verbose is made unreadable here - the refusal is then the parser's, not a NotFound.
  const std::string brokenTask = written("task", replacedOnce(readFile(taskFile_), "  verbose: false", "  verbose: sometimes"));
  const absl::StatusOr<std::unique_ptr<WBMpcInterface>> created = WBMpcInterface::Create(brokenTask, urdfFile_, referenceFile_);
  EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument) << created.status();
  EXPECT_TRUE(absl::StrContains(created.status().message(), absl::StrCat(brokenTask, ":"))) << created.status();
  EXPECT_TRUE(absl::StrContains(created.status().message(), "verbose")) << created.status();
}

TEST_F(WBMpcInterfaceInputFilesTest, ModelSettingsThatDoNotConvertAreAnInvalidArgumentPrefixedWithTheTaskFile) {
  // A fixed joint cannot swing: the model settings refuse it by its field, and the path form names the file.
  const std::string brokenTask = written(
      "model_settings",
      replacedOnce(readFile(taskFile_), "left_shoulder_y: \"left_shoulder_pitch_joint\"", "left_shoulder_y: \"left_wrist_roll_joint\""));
  const absl::StatusOr<std::unique_ptr<WBMpcInterface>> created = WBMpcInterface::Create(brokenTask, urdfFile_, referenceFile_);
  EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument) << created.status();
  EXPECT_TRUE(absl::StartsWith(created.status().message(), absl::StrCat(brokenTask, ": [ModelSettings]"))) << created.status();
  EXPECT_TRUE(absl::StrContains(created.status().message(), "model_settings.arm_joint_names.left_shoulder_y")) << created.status();
}

TEST_F(WBMpcInterfaceInputFilesTest, ASolverSettingThatDoesNotParseIsAnInvalidArgumentNotAnException) {
  // Both factories return the parser's error as a Status, naming the line of mpc.time_horizon; nothing is thrown.
  const std::string content = replacedOnce(readFile(taskFile_), "  time_horizon: 1.1", "  time_horizon: forever");
  const std::string brokenTask = written("solver_settings", content);
  const std::string place = absl::StrCat(brokenTask, ":", lineOf(content, "  time_horizon: forever"), ":");
  const absl::StatusOr<std::unique_ptr<WBMpcInterface>> created = WBMpcInterface::Create(brokenTask, urdfFile_, referenceFile_);
  EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument) << created.status();
  EXPECT_TRUE(absl::StrContains(created.status().message(), place)) << created.status();
  const absl::StatusOr<std::unique_ptr<WBMpcInterface>> models =
      WBMpcInterface::CreateControllerModels(brokenTask, urdfFile_, referenceFile_);
  EXPECT_EQ(models.status().code(), absl::StatusCode::kInvalidArgument) << models.status();
  EXPECT_TRUE(absl::StrContains(models.status().message(), place)) << models.status();
}

TEST_F(WBMpcInterfaceInputFilesTest, ASolverSettingThatDoesNotConvertIsPrefixedWithTheTaskFile) {
  // A name that is not an integrator parses as a string; the conversion refuses it by its field, in both factories.
  const std::string brokenTask =
      written("integrator", replacedOnce(readFile(taskFile_), "  integrator_type: \"RK4\"", "  integrator_type: \"RK5\""));
  for (const bool controllerModels : {false, true}) {
    SCOPED_TRACE(controllerModels);
    const absl::StatusOr<std::unique_ptr<WBMpcInterface>> created =
        controllerModels ? WBMpcInterface::CreateControllerModels(brokenTask, urdfFile_, referenceFile_)
                         : WBMpcInterface::Create(brokenTask, urdfFile_, referenceFile_);
    EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument) << created.status();
    EXPECT_TRUE(absl::StartsWith(created.status().message(), absl::StrCat(brokenTask, ": "))) << created.status();
    EXPECT_TRUE(absl::StrContains(created.status().message(), "multiple_shooting.integrator_type")) << created.status();
  }
}

TEST_F(WBMpcInterfaceInputFilesTest, AModeScheduleThatDoesNotConvertIsPrefixedWithTheReferenceFile) {
  // Two modes take one event time between them; the reference file's error names the reference file, not the task file.
  const std::string brokenReference =
      written("reference", replacedOnce(readFile(referenceFile_), "  event_times: 0.5\n", "  event_times: 0.5\n  event_times: 0.7\n"));
  const absl::StatusOr<std::unique_ptr<WBMpcInterface>> created =
      WBMpcInterface::CreateControllerModels(taskFile_, urdfFile_, brokenReference);
  EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument) << created.status();
  EXPECT_TRUE(absl::StartsWith(created.status().message(), absl::StrCat(brokenReference, ": "))) << created.status();
  EXPECT_TRUE(absl::StrContains(created.status().message(), "initial_mode_schedule")) << created.status();
}

}  // namespace ocs2::humanoid
