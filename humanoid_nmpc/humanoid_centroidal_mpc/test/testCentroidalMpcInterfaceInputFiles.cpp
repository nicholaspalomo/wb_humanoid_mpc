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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_common_mpc/common/ModelSettings.h"

/**
 * CentroidalMpcInterface::Create() on a missing input file. The constructor used to check the three files for existence and
 * throw std::invalid_argument, after its member initializer had already loaded the model settings - which read the URDF
 * as well as the task file - so a missing URDF surfaced as whatever the URDF parser threw, and a missing reference file
 * as an exception out of a function that returns a Status. Create() now checks all three before anything reads them and
 * returns NotFound naming the path. None of these builds a CppAD model: every case returns before the problem is set up.
 */
namespace ocs2::humanoid {
namespace {

std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

class CentroidalMpcInterfaceInputFilesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    taskFile_ = runfilePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml");
    urdfFile_ = runfilePath("robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf");
    referenceFile_ = runfilePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.yaml");
    ASSERT_FALSE(taskFile_.empty() || urdfFile_.empty() || referenceFile_.empty())
        << "the DRC Atlas centroidal files are not in the runfiles";
    missing_ = (std::filesystem::path(testing::TempDir()) / "testCentroidalMpcInterfaceInputFiles_missing.yaml").string();
    ASSERT_FALSE(std::filesystem::exists(missing_));
  }

  /** Create() must return, not throw, and return NotFound naming the missing path. */
  void expectNotFoundNamingTheMissingFile(const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>>& created,
                                          absl::string_view what) const {
    EXPECT_EQ(created.status().code(), absl::StatusCode::kNotFound) << what << ": " << created.status();
    EXPECT_TRUE(absl::StrContains(created.status().message(), missing_)) << what << ": " << created.status();
    EXPECT_TRUE(absl::StrContains(created.status().message(), what)) << created.status();
  }

  std::string taskFile_, urdfFile_, referenceFile_, missing_;
};

}  // namespace

TEST_F(CentroidalMpcInterfaceInputFilesTest, AMissingTaskFileIsNotFoundNamingItsPath) {
  expectNotFoundNamingTheMissingFile(CentroidalMpcInterface::Create(missing_, urdfFile_, referenceFile_), "task file");
}

TEST_F(CentroidalMpcInterfaceInputFilesTest, AMissingUrdfIsNotFoundBeforeTheModelSettingsReadIt) {
  expectNotFoundNamingTheMissingFile(CentroidalMpcInterface::Create(taskFile_, missing_, referenceFile_), "URDF file");
}

TEST_F(CentroidalMpcInterfaceInputFilesTest, AMissingReferenceFileIsNotFoundNamingItsPath) {
  expectNotFoundNamingTheMissingFile(CentroidalMpcInterface::Create(taskFile_, urdfFile_, missing_), "reference file");
}

TEST_F(CentroidalMpcInterfaceInputFilesTest, ExistingFilesPassTheCheckAndAreRead) {
  // Positive control, without building the problem: with all three files present Create() gets past the check and reads
  // the task file, whose interface.verbose is made unreadable here - the refusal is then that key's, not a NotFound.
  std::ifstream in(taskFile_);
  std::string task((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const std::string block = "\ninterface:\n  verbose: ";
  const size_t value = task.find(block);
  ASSERT_NE(value, std::string::npos) << "the shipped task file no longer carries interface.verbose";
  const size_t valueStart = value + block.size();
  task.replace(valueStart, task.find_first_of(" \n", valueStart) - valueStart, "sometimes");
  const std::string brokenTask = (std::filesystem::path(testing::TempDir()) / "testCentroidalMpcInterfaceInputFiles_task.yaml").string();
  std::ofstream(brokenTask) << task;

  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created =
      CentroidalMpcInterface::Create(brokenTask, urdfFile_, referenceFile_);
  EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument) << created.status();
  EXPECT_TRUE(absl::StrContains(created.status().message(), ModelSettings::kInterfaceVerboseKey)) << created.status();
}

}  // namespace ocs2::humanoid
