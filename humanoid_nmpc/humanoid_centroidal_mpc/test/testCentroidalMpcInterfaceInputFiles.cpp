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

#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "support/TypedConfigFiles.h"

/**
 * CentroidalMpcInterface::Create() on input files it cannot use. A missing file is NotFound naming its path, checked
 * before anything reads the files; a file that does not parse strictly is an InvalidArgument naming the file, the line
 * and the column; a YAML file is refused naming it; and a configuration that does not convert is an InvalidArgument naming
 * the field. None of these builds a CppAD model: every case returns before the problem is set up.
 */
namespace ocs2::humanoid {
namespace {

class CentroidalMpcInterfaceInputFilesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    files_ = atlasFiles();
    ASSERT_FALSE(files_.taskFile.empty() || files_.urdfFile.empty() || files_.referenceFile.empty())
        << "the DRC Atlas centroidal files are not in the runfiles";
    absl::StatusOr<CentroidalMpcConfig> config = loadConfigOf(files_);
    ASSERT_TRUE(config.ok()) << config.status();
    shipped_ = *std::move(config);
    missing_ = (std::filesystem::path(testing::TempDir()) / "testCentroidalMpcInterfaceInputFiles_missing.textproto").string();
    ASSERT_FALSE(std::filesystem::exists(missing_));
  }

  /** Create() must return, not throw, and return NotFound naming the missing path. */
  void expectNotFoundNamingTheMissingFile(const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>>& created,
                                          absl::string_view what) const {
    EXPECT_EQ(created.status().code(), absl::StatusCode::kNotFound) << what << ": " << created.status();
    EXPECT_TRUE(absl::StrContains(created.status().message(), missing_)) << what << ": " << created.status();
    EXPECT_TRUE(absl::StrContains(created.status().message(), what)) << created.status();
  }

  /** The robot's files in a directory of the test's own, with `taskText` as the task file. */
  CentroidalRobotFiles writtenWithTaskText(absl::string_view directory, absl::string_view taskText) const {
    absl::StatusOr<CentroidalRobotFiles> files = writeConfig(absl::StrCat(testing::TempDir(), "/", directory), shipped_, files_.urdfFile);
    EXPECT_TRUE(files.ok()) << files.status();
    if (!files.ok()) return CentroidalRobotFiles{};
    EXPECT_TRUE(writeTextFile(files->taskFile, taskText).ok());
    return *files;
  }

  CentroidalRobotFiles files_;
  CentroidalMpcConfig shipped_;
  std::string missing_;
};

}  // namespace

TEST_F(CentroidalMpcInterfaceInputFilesTest, AMissingTaskFileIsNotFoundNamingItsPath) {
  expectNotFoundNamingTheMissingFile(CentroidalMpcInterface::Create(missing_, files_.urdfFile, files_.referenceFile), "task file");
}

TEST_F(CentroidalMpcInterfaceInputFilesTest, AMissingUrdfIsNotFoundBeforeTheModelSettingsReadIt) {
  expectNotFoundNamingTheMissingFile(CentroidalMpcInterface::Create(files_.taskFile, missing_, files_.referenceFile), "URDF file");
  expectNotFoundNamingTheMissingFile(CentroidalMpcInterface::Create(shipped_, missing_), "URDF file");
}

TEST_F(CentroidalMpcInterfaceInputFilesTest, AMissingReferenceFileIsNotFoundNamingItsPath) {
  expectNotFoundNamingTheMissingFile(CentroidalMpcInterface::Create(files_.taskFile, files_.urdfFile, missing_), "reference file");
}

TEST_F(CentroidalMpcInterfaceInputFilesTest, ATaskFileThatDoesNotParseIsRefusedWithItsPosition) {
  // Positive control of the existence check: with all three files present Create() reads the task file, whose value is
  // made unreadable here - the refusal is then the parser's, naming the file, the line and the column.
  const std::string text = taskFileText(shipped_.task);
  const std::string broken = absl::StrReplaceAll(text, {{"verbose: true", "verbose: sometimes"}});
  ASSERT_NE(broken, text) << "the shipped task file no longer sets interface.verbose";
  const CentroidalRobotFiles files = writtenWithTaskText("input_files_unparsable", broken);
  for (const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>>& created :
       {CentroidalMpcInterface::Create(files.taskFile, files.urdfFile, files.referenceFile),
        CentroidalMpcInterface::CreateControllerModels(files.taskFile, files.urdfFile, files.referenceFile)}) {
    EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument) << created.status();
    EXPECT_TRUE(absl::StrContains(created.status().message(), "task.textproto:")) << created.status();
    EXPECT_TRUE(absl::StrContains(created.status().message(), "verbose")) << created.status();
  }
}

TEST_F(CentroidalMpcInterfaceInputFilesTest, AYamlTaskFileIsRefusedNamingIt) {
  // A task file in the YAML format the configuration was written in before the textproto migration: the MPC reads
  // textprotos only, and refuses it as a whole, naming the file, rather than reading a part of it.
  const std::string yaml = (std::filesystem::path(testing::TempDir()) / "input_files_task.yaml").string();
  ASSERT_TRUE(writeTextFile(yaml, "interface:\n  verbose: true\nmodel_settings:\n  robotName: drc_atlas\n").ok());
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created =
      CentroidalMpcInterface::Create(yaml, files_.urdfFile, files_.referenceFile);
  EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument) << created.status();
  EXPECT_TRUE(absl::StrContains(created.status().message(), yaml)) << created.status();
}

TEST_F(CentroidalMpcInterfaceInputFilesTest, ModelSettingsThatDoNotConvertAreAnInvalidArgumentNamingTheField) {
  CentroidalMpcConfig config = shipped_;
  config.task.model_settings.arm_joint_names.left_shoulder_y = "no_such_joint";
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created = CentroidalMpcInterface::Create(config, files_.urdfFile);
  EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument) << created.status();
  EXPECT_TRUE(absl::StrContains(created.status().message(), "model_settings.arm_joint_names")) << created.status();
}

TEST_F(CentroidalMpcInterfaceInputFilesTest, ASolverSettingThatDoesNotConvertIsAnInvalidArgumentNamingTheField) {
  CentroidalMpcConfig config = shipped_;
  config.task.multiple_shooting.integrator_type = "RK9";
  for (const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>>& created :
       {CentroidalMpcInterface::Create(config, files_.urdfFile), CentroidalMpcInterface::CreateControllerModels(config, files_.urdfFile)}) {
    EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument) << created.status();
    EXPECT_TRUE(absl::StrContains(created.status().message(), "multiple_shooting.integrator_type")) << created.status();
  }
}

}  // namespace ocs2::humanoid
