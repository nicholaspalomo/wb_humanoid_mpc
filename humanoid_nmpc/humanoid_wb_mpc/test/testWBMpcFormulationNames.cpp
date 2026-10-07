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
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/model/MpcFormulationFromConfig.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"

/**
 * The formulation choices the whole-body MPC does not implement, now that they are names rather than top-level booleans
 * (humanoid_nmpc/docs/README.md, sections 1 and 2): contact_schedule_source "contact_planner" and `dcm_terminal_cost`
 * are refused by name, the retired `useContactPlanning` and `useDcmTerminalCost` are refused by the strict parser naming
 * their replacement, and interface.verbose reaches the model settings. Every refusal happens before any term is built,
 * so these tests construct no CppAD model.
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

/** `names` with every `from` replaced by `to`; fails the test when `from` is not among them. */
std::vector<std::string> renamed(std::vector<std::string> names, absl::string_view from, absl::string_view to) {
  bool found = false;
  for (std::string& name : names) {
    if (name == from) {
      name = std::string(to);
      found = true;
    }
  }
  EXPECT_TRUE(found) << "'" << from << "' is not listed";
  return names;
}

class WBMpcFormulationNamesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    taskFile_ = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto");
    referenceFile_ = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto");
    urdfFile_ = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
    ASSERT_FALSE(taskFile_.empty() || referenceFile_.empty() || urdfFile_.empty()) << "the G1 whole-body files are not in the runfiles";
    shipped_ = readFile(taskFile_);
    absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(taskFile_);
    ASSERT_TRUE(task.ok()) << task.status();
    task_ = *std::move(task);
    absl::StatusOr<mpc_config::ReferenceFile> reference = loadReferenceFile(referenceFile_);
    ASSERT_TRUE(reference.ok()) << reference.status();
    reference_ = *std::move(reference);
  }

  /** The refusal of the interface of `task`. */
  absl::Status refusal(const mpc_config::TaskFile& task) const {
    const absl::StatusOr<std::unique_ptr<WBMpcInterface>> created = WBMpcInterface::Create(task, urdfFile_, reference_);
    return created.ok() ? absl::OkStatus() : created.status();
  }

  /** The refusal of the interface of the task file `content`, written for the test as `name`. */
  absl::Status refusalOfFile(absl::string_view name, const std::string& content) const {
    const std::string path =
        (std::filesystem::path(testing::TempDir()) / absl::StrCat("testWBMpcFormulationNames_", name, ".textproto")).string();
    std::ofstream(path) << content;
    const absl::StatusOr<std::unique_ptr<WBMpcInterface>> created = WBMpcInterface::Create(path, urdfFile_, referenceFile_);
    return created.ok() ? absl::OkStatus() : created.status();
  }

  /** The shipped task file, its contact schedule source the contact planner: refused after the model settings. */
  mpc_config::TaskFile withContactPlanner() const {
    mpc_config::TaskFile task = task_;
    task.contact_schedule_source = std::string(kContactPlannerContactScheduleSource);
    return task;
  }

  std::string taskFile_;
  std::string referenceFile_;
  std::string urdfFile_;
  std::string shipped_;
  mpc_config::TaskFile task_;
  mpc_config::ReferenceFile reference_;
};

}  // namespace

TEST_F(WBMpcFormulationNamesTest, theShippedFileNamesTheGaitScheduleAndTheQuadraticTerminalCost) {
  // Positive control for every refusal below: the shipped file selects what this interface builds.
  const absl::StatusOr<ContactScheduleSource> source = contactScheduleSourceFromConfig(task_);
  ASSERT_TRUE(source.ok()) << source.status();
  EXPECT_EQ(*source, ContactScheduleSource::kGaitSchedule);
  const absl::StatusOr<MpcFormulationTasks> tasks = mpcFormulationTasksFromConfig(task_, FormulationLogging::kQuiet);
  ASSERT_TRUE(tasks.ok()) << tasks.status();
  EXPECT_TRUE(tasks->hasCost(MpcCostType::kTerminalCost));
  EXPECT_FALSE(tasks->hasCost(MpcCostType::kDcmTerminalCost));
}

TEST_F(WBMpcFormulationNamesTest, theContactPlannerIsRefusedByName) {
  const absl::Status refused = refusal(withContactPlanner());
  ASSERT_FALSE(refused.ok()) << "the whole-body MPC accepted contact_schedule_source: \"contact_planner\"";
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), absl::StrCat("contact_schedule_source: \"", kContactPlannerContactScheduleSource, "\"")))
      << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "centroidal MPC only")) << refused;
}

TEST_F(WBMpcFormulationNamesTest, theRetiredContactPlanningBooleanIsRefusedNamingTheSource) {
  for (const char* absl_nonnull value : {"true", "false"}) {
    SCOPED_TRACE(value);
    const absl::Status refused =
        refusalOfFile(absl::StrCat("retiredPlanning_", value), absl::StrCat(shipped_, kRetiredContactPlanningKey, ": ", value, "\n"));
    ASSERT_FALSE(refused.ok()) << kRetiredContactPlanningKey << ": " << value << " was accepted";
    EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
    EXPECT_TRUE(absl::StrContains(refused.message(), absl::StrCat("'", kRetiredContactPlanningKey, "' is retired"))) << refused;
    EXPECT_TRUE(absl::StrContains(refused.message(), "contact_schedule_source")) << refused;
  }
}

TEST_F(WBMpcFormulationNamesTest, theDcmTerminalCostIsRefusedByName) {
  // This interface does not build DcmTerminalCost: accepting the name would end the horizon on no terminal cost at all.
  mpc_config::TaskFile task = task_;
  task.costs = renamed(task.costs, "terminal_cost", "dcm_terminal_cost");
  const absl::Status refused = refusal(task);
  ASSERT_FALSE(refused.ok()) << "the whole-body MPC accepted dcm_terminal_cost";
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "dcm_terminal_cost")) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "centroidal MPC only")) << refused;
}

TEST_F(WBMpcFormulationNamesTest, theRetiredDcmBooleanIsRefusedNamingTheCostThatReplacedIt) {
  for (const char* absl_nonnull value : {"true", "false"}) {
    SCOPED_TRACE(value);
    const absl::Status refused =
        refusalOfFile(absl::StrCat("retiredDcm_", value), absl::StrCat(shipped_, "useDcmTerminalCost: ", value, "\n"));
    ASSERT_FALSE(refused.ok()) << "useDcmTerminalCost: " << value << " was accepted";
    EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
    EXPECT_TRUE(absl::StrContains(refused.message(), "'useDcmTerminalCost' is retired")) << refused;
    EXPECT_TRUE(absl::StrContains(refused.message(), "dcm_terminal_cost")) << refused;
  }
}

/**
 * The interface used to hand its model settings the string literal "true" as their `verbose` flag, so every start-up
 * printed the settings banner whatever interface.verbose said - and the G1 whole-body file says false. The settings are
 * converted before the formulation is read, so a file refused for its contact schedule source has already shown
 * whether the banner followed the flag.
 */
TEST_F(WBMpcFormulationNamesTest, theModelSettingsBannerFollowsInterfaceVerbose) {
  ASSERT_FALSE(task_.interface.verbose) << "the shipped file no longer sets verbose: false";
  for (const bool verbose : {false, true}) {
    SCOPED_TRACE(verbose);
    mpc_config::TaskFile task = withContactPlanner();
    task.interface.verbose = verbose;
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, testing::_, testing::HasSubstr("Robot Model Settings")))
        .Times(verbose ? testing::AtLeast(1) : testing::Exactly(0));
    EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, testing::_, testing::HasSubstr("The solver settings of the task file")))
        .Times(verbose ? testing::Exactly(1) : testing::Exactly(0));
    log.StartCapturingLogs();
    const absl::Status refused = refusal(task);
    log.StopCapturingLogs();
    EXPECT_FALSE(refused.ok());
  }
  // And a value that is not a bool is refused by the parser, naming the field and its place in the file.
  const absl::Status notABool = refusalOfFile("verboseNotABool", replacedOnce(shipped_, "  verbose: false", "  verbose: loud"));
  ASSERT_FALSE(notABool.ok());
  EXPECT_EQ(notABool.code(), absl::StatusCode::kInvalidArgument) << notABool;
  EXPECT_TRUE(absl::StrContains(notABool.message(), "testWBMpcFormulationNames_verboseNotABool.textproto:")) << notABool;
  EXPECT_TRUE(absl::StrContains(notABool.message(), "verbose")) << notABool;
}

}  // namespace ocs2::humanoid
