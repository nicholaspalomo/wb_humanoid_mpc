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
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/pinocchio_model/pinocchioUtils.h"
#include "humanoid_mpc_config/task_file.nproto.h"

/**
 * The MPC indexes its state, its joint limits and its joint weights by the order of ModelSettings::mpcModelJointNames,
 * and takes the Pinocchio model's actuated joints to be in that order. checkPinocchioJointNaming() is the check that
 * they are. It used to be an assert(), which the optimized build every test and every robot runs compiles out, so a
 * model built from another URDF or another fixed_joint_names than its ModelSettings went through unchecked.
 */
namespace ocs2::humanoid {
namespace {

constexpr absl::string_view kTaskFile = "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto";
constexpr absl::string_view kUrdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";
// An actuated joint of the shipped whole-body MPC that is neither an arm-swing joint nor a contact parent, so fixing
// or renaming it changes nothing but the joint list.
constexpr absl::string_view kJoint = "waist_roll_joint";

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

std::string writeFile(absl::string_view name, const std::string& content) {
  const std::string path = (std::filesystem::path(testing::TempDir()) / std::string(name)).string();
  std::ofstream out(path);
  out << content;
  return path;
}

/** `content` with every occurrence of `from` replaced by `to`; fails the test when there is none. */
std::string replacedAll(std::string content, absl::string_view from, absl::string_view to) {
  size_t position = content.find(from);
  EXPECT_NE(position, std::string::npos) << "'" << from << "' not found";
  while (position != std::string::npos) {
    content.replace(position, from.size(), std::string(to));
    position = content.find(from, position + to.size());
  }
  return content;
}

class PinocchioJointNamingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string taskFile = runfilePath(kTaskFile);
    urdfFile_ = runfilePath(kUrdfFile);
    ASSERT_FALSE(taskFile.empty() || urdfFile_.empty()) << "the G1 whole-body files are not in the runfiles";
    absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(taskFile);
    ASSERT_TRUE(task.ok()) << task.status();
    task_ = *std::move(task);
    settings_ = std::make_unique<ModelSettings>(ModelSettings::Create(task_, urdfFile_, "wb_mpc_", /*verbose=*/false).value());
    // The same task file with one more joint fixed: the ModelSettings of another joint list.
    moreFixedTask_ = task_;
    moreFixedTask_.model_settings.fixed_joint_names.emplace_back(kJoint);
    moreFixedSettings_ =
        std::make_unique<ModelSettings>(ModelSettings::Create(moreFixedTask_, urdfFile_, "wb_mpc_", /*verbose=*/false).value());
    ASSERT_EQ(moreFixedSettings_->mpcModelJointNames.size() + 1, settings_->mpcModelJointNames.size());
  }

  std::string urdfFile_;
  mpc_config::TaskFile task_;
  mpc_config::TaskFile moreFixedTask_;
  std::unique_ptr<ModelSettings> settings_;
  std::unique_ptr<ModelSettings> moreFixedSettings_;
};

void expectNamesJoint(const absl::Status& status, absl::string_view joint) {
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_TRUE(absl::StrContains(status.message(), absl::StrCat("'", joint, "'"))) << status << "\n does not name " << joint;
  EXPECT_TRUE(absl::StrContains(status.message(), "model_settings.fixed_joint_names")) << status;
}

}  // namespace

TEST_F(PinocchioJointNamingTest, aModelBuiltFromItsOwnSettingsPasses) {
  const absl::StatusOr<PinocchioInterface> pinocchioInterface = loadCustomPinocchioInterface(task_, urdfFile_, *settings_);
  ASSERT_TRUE(pinocchioInterface.ok()) << pinocchioInterface.status();
  EXPECT_TRUE(checkPinocchioJointNaming(*pinocchioInterface, *settings_).ok());
  const absl::StatusOr<PinocchioInterface> moreFixed = loadCustomPinocchioInterface(moreFixedTask_, urdfFile_, *moreFixedSettings_);
  ASSERT_TRUE(moreFixed.ok()) << moreFixed.status();
  EXPECT_TRUE(checkPinocchioJointNaming(*moreFixed, *moreFixedSettings_).ok());
}

TEST_F(PinocchioJointNamingTest, aModelOfAnotherJointListIsRefusedNamingTheFirstJointThatDiffers) {
  const absl::StatusOr<PinocchioInterface> full = loadCustomPinocchioInterface(task_, urdfFile_, *settings_);
  const absl::StatusOr<PinocchioInterface> moreFixed = loadCustomPinocchioInterface(moreFixedTask_, urdfFile_, *moreFixedSettings_);
  ASSERT_TRUE(full.ok() && moreFixed.ok());
  // The model has the joint the settings fixed: it is the first model joint without its settings counterpart.
  expectNamesJoint(checkPinocchioJointNaming(*full, *moreFixedSettings_), kJoint);
  // And the other way round: the settings list a joint the model has fixed.
  expectNamesJoint(checkPinocchioJointNaming(*moreFixed, *settings_), kJoint);
}

TEST_F(PinocchioJointNamingTest, aModelFromAnotherUrdfIsRefusedWhereItIsBuilt) {
  // The URDF of another revision, in which the joint was renamed: the settings were derived from the original.
  const std::string renamed = absl::StrCat(kJoint, "_renamed");
  const std::string urdf = writeFile("testPinocchioJointNaming_renamed.urdf",
                                     replacedAll(readFile(urdfFile_), absl::StrCat("\"", kJoint, "\""), absl::StrCat("\"", renamed, "\"")));
  const absl::StatusOr<PinocchioInterface> pinocchioInterface = loadCustomPinocchioInterface(task_, urdf, *settings_);
  ASSERT_FALSE(pinocchioInterface.ok()) << "a model without " << kJoint << " was accepted for settings that actuate it";
  expectNamesJoint(pinocchioInterface.status(), kJoint);
}

TEST_F(PinocchioJointNamingTest, anUnreadableUrdfIsAStatusNamingTheFile) {
  const std::string notUrdf = writeFile("testPinocchioJointNaming_notUrdf.urdf", "this is not a URDF");
  const absl::StatusOr<PinocchioInterface> pinocchioInterface = loadCustomPinocchioInterface(task_, notUrdf, *settings_);
  ASSERT_FALSE(pinocchioInterface.ok());
  EXPECT_EQ(pinocchioInterface.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(pinocchioInterface.status().message(), notUrdf)) << pinocchioInterface.status();
}

using PinocchioJointNamingDeathTest = PinocchioJointNamingTest;

TEST_F(PinocchioJointNamingDeathTest, theJointLimitsAreNotReadByPositionFromAMismatchingModel) {
  const absl::StatusOr<PinocchioInterface> full = loadCustomPinocchioInterface(task_, urdfFile_, *settings_);
  ASSERT_TRUE(full.ok()) << full.status();
  // Positive control: the matching settings read one limit per MPC joint.
  const std::pair<vector_t, vector_t> limits = readPinocchioJointLimits(*full, *settings_, /*verbose=*/false);
  EXPECT_EQ(limits.first.size(), static_cast<Eigen::Index>(settings_->mpcModelJointNames.size()));
  EXPECT_DEATH(readPinocchioJointLimits(*full, *moreFixedSettings_, /*verbose=*/false), kJoint.data());
}

}  // namespace ocs2::humanoid
