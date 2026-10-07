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

#include "humanoid_common_mpc/common/ContactInputParameterization.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/model/MpcFormulationFromConfig.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"

/**
 * The formulation choices the whole-body MPC refuses because it does not implement them, on the typed task file, and
 * the retired booleans that once selected them, which the strict parser refuses in a task file naming what replaced
 * them. Each refusal happens before any term is built, so these tests construct no CppAD model.
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

class WBMpcFormulationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    taskFile_ = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto");
    referenceFile_ = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto");
    urdfFile_ = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
    ASSERT_FALSE(taskFile_.empty() || referenceFile_.empty() || urdfFile_.empty()) << "the G1 whole-body files are not in the runfiles";
    absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(taskFile_);
    ASSERT_TRUE(task.ok()) << task.status();
    task_ = *std::move(task);
    absl::StatusOr<mpc_config::ReferenceFile> reference = loadReferenceFile(referenceFile_);
    ASSERT_TRUE(reference.ok()) << reference.status();
    reference_ = *std::move(reference);
  }

  /** The interface of the shipped task file with `line` appended, created from a file written for the test. */
  absl::Status createdWithLine(absl::string_view name, absl::string_view line) const {
    const std::string path =
        (std::filesystem::path(testing::TempDir()) / absl::StrCat("testWBMpcFormulation_", name, ".textproto")).string();
    std::ofstream(path) << readFile(taskFile_) << line << "\n";
    const absl::StatusOr<std::unique_ptr<WBMpcInterface>> created = WBMpcInterface::Create(path, urdfFile_, referenceFile_);
    return created.ok() ? absl::OkStatus() : created.status();
  }

  /** The refusal of the interface of `task`. */
  absl::Status refusal(const mpc_config::TaskFile& task) const {
    const absl::StatusOr<std::unique_ptr<WBMpcInterface>> created = WBMpcInterface::Create(task, urdfFile_, reference_);
    return created.ok() ? absl::OkStatus() : created.status();
  }

  std::string taskFile_;
  std::string referenceFile_;
  std::string urdfFile_;
  mpc_config::TaskFile task_;
  mpc_config::ReferenceFile reference_;
};

}  // namespace

/**
 * Findings A91/A103, scenario (b): CoM + ACoM tracking is implemented for the centroidal MPC only. When it was the
 * boolean `useComAndAcomTracking`, setting it in a whole-body task file built no ACoM cost, switched the arm swing off
 * and zeroed indices 6..11 of the state weights - the centroidal base pose, but here the first six joint angles, one
 * leg's weights. The whole-body MPC now refuses the cost by name.
 */
TEST_F(WBMpcFormulationTest, theCoMAndAcomTrackingCostIsRefusedByName) {
  // Positive control: the shipped file does not list the cost, so the refusal below is the variant's doing.
  const absl::StatusOr<MpcFormulationTasks> shippedTasks = mpcFormulationTasksFromConfig(task_, FormulationLogging::kQuiet);
  ASSERT_TRUE(shippedTasks.ok()) << shippedTasks.status();
  ASSERT_FALSE(shippedTasks->hasCost(MpcCostType::kComAndAcomTrackingCost));

  mpc_config::TaskFile listed = task_;
  listed.costs.push_back("com_and_acom_tracking_cost");
  ASSERT_TRUE(mpcFormulationTasksFromConfig(listed, FormulationLogging::kQuiet)->hasCost(MpcCostType::kComAndAcomTrackingCost));

  const absl::Status refused = refusal(listed);
  ASSERT_FALSE(refused.ok()) << "the whole-body MPC accepted com_and_acom_tracking_cost";
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.message(), "com_and_acom_tracking_cost")) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "centroidal MPC only")) << refused;
}

/** The retired boolean is refused by the strict parser, naming the cost that replaced it and the file. */
TEST_F(WBMpcFormulationTest, theRetiredBooleanIsRefusedNamingItsReplacement) {
  const absl::Status refused = createdWithLine("retired", "useComAndAcomTracking: true");
  ASSERT_FALSE(refused.ok());
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.message(), "'useComAndAcomTracking' is retired")) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "com_and_acom_tracking_cost")) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "testWBMpcFormulation_retired.textproto")) << refused;
}

/**
 * Finding AC6: the whole-body MPC implements neither the basis-vector contact inputs nor the online contact planner. It
 * used to read neither key and run the wrench parameterization on the periodic gait schedule while the file asked for
 * something else; it now refuses both by name, as it refuses the contact-implicit terms.
 */
TEST_F(WBMpcFormulationTest, theBasisVectorContactInputsAreRefusedByName) {
  // Positive control: the shipped file selects the wrench parameterization (by omission), so the refusal is the variant's.
  const absl::StatusOr<ContactInputParameterization> shippedInputs = contactInputParameterizationFromConfig(task_);
  ASSERT_TRUE(shippedInputs.ok()) << shippedInputs.status();
  ASSERT_EQ(*shippedInputs, ContactInputParameterization::kWrench);

  mpc_config::TaskFile basisInputs = task_;
  basisInputs.contact_input_parameterization = "basis_vectors";
  const absl::Status refused = refusal(basisInputs);
  ASSERT_FALSE(refused.ok()) << "the whole-body MPC accepted basis-vector contact inputs";
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.message(), "contact_input_parameterization: \"basis_vectors\"")) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "centroidal MPC only")) << refused;
}

TEST_F(WBMpcFormulationTest, theRetiredBasisBooleanIsRefusedNamingItsReplacement) {
  for (const std::string& value : {std::string("true"), std::string("false")}) {
    SCOPED_TRACE(value);
    const absl::Status refused =
        createdWithLine(absl::StrCat("retiredBasis_", value), absl::StrCat("useContactBasisVectorInputs: ", value));
    ASSERT_FALSE(refused.ok());
    EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(absl::StrContains(refused.message(), "'useContactBasisVectorInputs' is retired")) << refused;
    EXPECT_TRUE(absl::StrContains(refused.message(), "contact_input_parameterization")) << refused;
  }
}

TEST_F(WBMpcFormulationTest, theRetiredContactPlanningBooleanIsRefusedNamingItsReplacement) {
  // Positive control: the shipped whole-body file names the gait schedule, so the refusal is the retired key's.
  ASSERT_EQ(task_.contact_schedule_source, "gait_schedule");
  const absl::Status refused = createdWithLine("contactPlanning", "useContactPlanning: true");
  ASSERT_FALSE(refused.ok()) << "the whole-body MPC accepted useContactPlanning: true";
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.message(), "'useContactPlanning' is retired")) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "contact_schedule_source")) << refused;
}

}  // namespace ocs2::humanoid
