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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"

/**
 * The formulation choices the whole-body MPC does not implement, now that they are names rather than top-level booleans
 * (humanoid_nmpc/docs/README.md, sections 1 and 2): `contactScheduleSource: contact_planner` and `dcm_terminal_cost`
 * are refused by name, the retired `useContactPlanning` and `useDcmTerminalCost` are refused naming their replacement,
 * and interface.verbose reaches the model settings. Every refusal happens before any term is built, so these tests
 * construct no CppAD model.
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

constexpr absl::string_view kShippedSource = "\ncontactScheduleSource: gait_schedule\n";

class WBMpcFormulationNamesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    taskFile_ = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml");
    referenceFile_ = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.yaml");
    urdfFile_ = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
    ASSERT_FALSE(taskFile_.empty() || referenceFile_.empty() || urdfFile_.empty()) << "the G1 whole-body files are not in the runfiles";
    shipped_ = readFile(taskFile_);
  }

  std::string writeVariant(absl::string_view name, const std::string& content) const {
    const std::string path =
        (std::filesystem::path(testing::TempDir()) / absl::StrCat("testWBMpcFormulationNames_", name, ".yaml")).string();
    std::ofstream out(path);
    out << content;
    return path;
  }

  /** Creates the interface of a variant that must be refused, and returns the refusal. */
  absl::Status refusal(absl::string_view name, const std::string& content) const {
    const absl::StatusOr<std::unique_ptr<WBMpcInterface>> created =
        WBMpcInterface::Create(writeVariant(name, content), urdfFile_, referenceFile_);
    if (created.ok()) return absl::OkStatus();
    return created.status();
  }

  std::string taskFile_;
  std::string referenceFile_;
  std::string urdfFile_;
  std::string shipped_;
};

}  // namespace

TEST_F(WBMpcFormulationNamesTest, theShippedFileNamesTheGaitScheduleAndTheQuadraticTerminalCost) {
  // Positive control for every refusal below: the shipped file selects what this interface builds.
  const absl::StatusOr<ContactScheduleSource> source = loadContactScheduleSource(taskFile_);
  ASSERT_TRUE(source.ok()) << source.status();
  EXPECT_EQ(*source, ContactScheduleSource::kGaitSchedule);
  ASSERT_NE(shipped_.find(kShippedSource), std::string::npos);
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(taskFile_);
  ASSERT_TRUE(tasks.ok()) << tasks.status();
  EXPECT_TRUE(tasks->hasCost(MpcCostType::TerminalCost));
  EXPECT_FALSE(tasks->hasCost(MpcCostType::DcmTerminalCost));
}

TEST_F(WBMpcFormulationNamesTest, theContactPlannerIsRefusedByName) {
  const absl::Status refused =
      refusal("contactPlanner", replacedOnce(shipped_, kShippedSource, "\ncontactScheduleSource: contact_planner\n"));
  ASSERT_FALSE(refused.ok()) << "the whole-body MPC accepted contactScheduleSource: contact_planner";
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), absl::StrCat(kContactScheduleSourceKey, ": ", kContactPlannerContactScheduleSource)))
      << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "centroidal MPC only")) << refused;
}

TEST_F(WBMpcFormulationNamesTest, theRetiredContactPlanningBooleanIsRefusedNamingTheSource) {
  for (const char* value : {"true", "false"}) {
    SCOPED_TRACE(value);
    const absl::Status refused =
        refusal(absl::StrCat("retiredPlanning_", value), absl::StrCat("useContactPlanning: ", value, "\n", shipped_));
    ASSERT_FALSE(refused.ok()) << "useContactPlanning: " << value << " was accepted";
    EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
    EXPECT_TRUE(absl::StrContains(refused.message(), kRetiredContactPlanningKey)) << refused;
    EXPECT_TRUE(absl::StrContains(refused.message(), kContactScheduleSourceKey)) << refused;
  }
}

TEST_F(WBMpcFormulationNamesTest, theDcmTerminalCostIsRefusedByName) {
  // This interface does not build DcmTerminalCost: accepting the name would end the horizon on no terminal cost at all.
  const absl::Status refused = refusal("dcmTerminalCost", replacedOnce(shipped_, "\n  - terminal_cost\n", "\n  - dcm_terminal_cost\n"));
  ASSERT_FALSE(refused.ok()) << "the whole-body MPC accepted dcm_terminal_cost";
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "dcm_terminal_cost")) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "centroidal MPC only")) << refused;
}

TEST_F(WBMpcFormulationNamesTest, theRetiredDcmBooleanIsRefusedNamingTheCostThatReplacedIt) {
  for (const char* value : {"true", "false"}) {
    SCOPED_TRACE(value);
    const absl::Status refused = refusal(absl::StrCat("retiredDcm_", value), absl::StrCat("useDcmTerminalCost: ", value, "\n", shipped_));
    ASSERT_FALSE(refused.ok()) << "useDcmTerminalCost: " << value << " was accepted";
    EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
    EXPECT_TRUE(absl::StrContains(refused.message(), "useDcmTerminalCost")) << refused;
    EXPECT_TRUE(absl::StrContains(refused.message(), "dcm_terminal_cost")) << refused;
  }
}

/**
 * The interface used to hand its model settings the string literal "true" as their `verbose` flag, so every start-up
 * printed the settings banner whatever interface.verbose said - and the G1 whole-body file says false. The settings are
 * loaded by the constructor, before the formulation is read, so a file refused for its contact schedule source has
 * already shown whether the banner followed the flag.
 */
TEST_F(WBMpcFormulationNamesTest, theModelSettingsBannerFollowsInterfaceVerbose) {
  ASSERT_NE(shipped_.find("\ninterface:\n  verbose: false"), std::string::npos) << "the shipped file no longer sets verbose: false";
  const std::string refusedSource = replacedOnce(shipped_, kShippedSource, "\ncontactScheduleSource: contact_planner\n");
  for (const bool verbose : {false, true}) {
    SCOPED_TRACE(verbose);
    const std::string content =
        verbose ? replacedOnce(refusedSource, "\ninterface:\n  verbose: false", "\ninterface:\n  verbose: true") : refusedSource;
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, testing::_, testing::HasSubstr("Robot Model Settings")))
        .Times(verbose ? testing::AtLeast(1) : testing::Exactly(0));
    log.StartCapturingLogs();
    const absl::Status refused = refusal(absl::StrCat("verbose_", verbose), content);
    log.StopCapturingLogs();
    EXPECT_FALSE(refused.ok());
  }
  // And a value that is not a bool is refused by Create(), naming the key.
  const absl::Status notABool =
      refusal("verboseNotABool", replacedOnce(shipped_, "\ninterface:\n  verbose: false", "\ninterface:\n  verbose: loud"));
  ASSERT_FALSE(notABool.ok());
  EXPECT_EQ(notABool.code(), absl::StatusCode::kInvalidArgument) << notABool;
  EXPECT_TRUE(absl::StrContains(notABool.message(), ModelSettings::kInterfaceVerboseKey)) << notABool;
}

}  // namespace ocs2::humanoid
