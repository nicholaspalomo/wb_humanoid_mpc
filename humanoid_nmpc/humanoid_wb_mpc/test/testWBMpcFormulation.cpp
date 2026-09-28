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
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/ContactInputParameterization.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"

/**
 * The formulation choices the whole-body MPC refuses because it does not implement them. Each refusal happens before
 * any term is built, so these tests construct no CppAD model.
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

class WBMpcFormulationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    taskFile_ = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml");
    referenceFile_ = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.yaml");
    urdfFile_ = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
    ASSERT_FALSE(taskFile_.empty() || referenceFile_.empty() || urdfFile_.empty()) << "the G1 whole-body files are not in the runfiles";
    shipped_ = readFile(taskFile_);
  }

  std::string writeVariant(absl::string_view name, const std::string& content) const {
    const std::string path = (std::filesystem::path(testing::TempDir()) / absl::StrCat("testWBMpcFormulation_", name, ".yaml")).string();
    std::ofstream out(path);
    out << content;
    return path;
  }

  std::string taskFile_;
  std::string referenceFile_;
  std::string urdfFile_;
  std::string shipped_;
};

}  // namespace

/**
 * Findings A91/A103, scenario (b): CoM + ACoM tracking is implemented for the centroidal MPC only. When it was the
 * boolean `useComAndAcomTracking`, setting it in a whole-body task file built no ACoM cost, switched the arm swing off
 * and zeroed indices 6..11 of Q and Q_final - the centroidal base pose, but here the first six joint angles, one leg's
 * weights. The whole-body MPC now refuses the cost by name.
 */
TEST_F(WBMpcFormulationTest, theCoMAndAcomTrackingCostIsRefusedByName) {
  // Positive control: the shipped file loads and does not list the cost, so the refusal below is the variant's doing.
  const absl::StatusOr<MpcFormulationTasks> shippedTasks = loadMpcFormulationTasks(taskFile_);
  ASSERT_TRUE(shippedTasks.ok()) << shippedTasks.status();
  ASSERT_FALSE(shippedTasks->hasCost(MpcCostType::ComAndAcomTrackingCost));
  const std::string::size_type costs = shipped_.find("\ncosts:\n");
  ASSERT_NE(costs, std::string::npos);

  std::string listed = shipped_;
  listed.insert(costs + std::string("\ncosts:\n").size(), "  - com_and_acom_tracking_cost\n");
  const std::string variant = writeVariant("acomListed", listed);
  ASSERT_TRUE(loadMpcFormulationTasks(variant)->hasCost(MpcCostType::ComAndAcomTrackingCost));

  const absl::StatusOr<std::unique_ptr<WBMpcInterface>> created = WBMpcInterface::Create(variant, urdfFile_, referenceFile_);
  ASSERT_FALSE(created.ok()) << "the whole-body MPC accepted com_and_acom_tracking_cost";
  EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(created.status().message(), "com_and_acom_tracking_cost")) << created.status();
  EXPECT_TRUE(absl::StrContains(created.status().message(), "centroidal MPC only")) << created.status();
}

/** The retired boolean is refused here too, by the formulation loader both interfaces share. */
TEST_F(WBMpcFormulationTest, theRetiredBooleanIsRefusedNamingItsReplacement) {
  const std::string variant = writeVariant("retired", absl::StrCat("useComAndAcomTracking: true\n", shipped_));
  const absl::StatusOr<std::unique_ptr<WBMpcInterface>> created = WBMpcInterface::Create(variant, urdfFile_, referenceFile_);
  ASSERT_FALSE(created.ok());
  EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(created.status().message(), "com_and_acom_tracking_cost")) << created.status();
}

/**
 * Finding AC6: the whole-body MPC implements neither the basis-vector contact inputs nor the online contact planner. It
 * used to read neither key and run the wrench parameterization on the periodic gait schedule while the file asked for
 * something else; it now refuses both by name, as it refuses the contact-implicit terms.
 */
TEST_F(WBMpcFormulationTest, theBasisVectorContactInputsAreRefusedByName) {
  // Positive control: the shipped file selects the wrench parameterization (by omission), so the refusal is the variant's.
  const absl::StatusOr<ContactInputParameterization> shippedInputs = loadContactInputParameterization(taskFile_);
  ASSERT_TRUE(shippedInputs.ok()) << shippedInputs.status();
  ASSERT_EQ(*shippedInputs, ContactInputParameterization::kWrench);

  const std::string variant = writeVariant("basisInputs", absl::StrCat("contactInputParameterization: basis_vectors\n", shipped_));
  const absl::StatusOr<std::unique_ptr<WBMpcInterface>> created = WBMpcInterface::Create(variant, urdfFile_, referenceFile_);
  ASSERT_FALSE(created.ok()) << "the whole-body MPC accepted basis-vector contact inputs";
  EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(created.status().message(), "contactInputParameterization: basis_vectors")) << created.status();
  EXPECT_TRUE(absl::StrContains(created.status().message(), "centroidal MPC only")) << created.status();
}

TEST_F(WBMpcFormulationTest, theRetiredBasisBooleanIsRefusedNamingItsReplacement) {
  for (const std::string& value : {std::string("true"), std::string("false")}) {
    SCOPED_TRACE(value);
    const std::string variant =
        writeVariant(absl::StrCat("retiredBasis_", value), absl::StrCat("useContactBasisVectorInputs: ", value, "\n", shipped_));
    const absl::StatusOr<std::unique_ptr<WBMpcInterface>> created = WBMpcInterface::Create(variant, urdfFile_, referenceFile_);
    ASSERT_FALSE(created.ok());
    EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(absl::StrContains(created.status().message(), "useContactBasisVectorInputs")) << created.status();
    EXPECT_TRUE(absl::StrContains(created.status().message(), "contactInputParameterization")) << created.status();
  }
}

TEST_F(WBMpcFormulationTest, theRetiredContactPlanningBooleanIsRefusedNamingItsReplacement) {
  // Positive control: the shipped whole-body file names the gait schedule, so the refusal is the retired key's.
  ASSERT_NE(shipped_.find("\ncontactScheduleSource: gait_schedule\n"), std::string::npos);
  const std::string variant = writeVariant("contactPlanning", absl::StrCat("useContactPlanning: true\n", shipped_));
  const absl::StatusOr<std::unique_ptr<WBMpcInterface>> created = WBMpcInterface::Create(variant, urdfFile_, referenceFile_);
  ASSERT_FALSE(created.ok()) << "the whole-body MPC accepted useContactPlanning: true";
  EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(created.status().message(), "useContactPlanning")) << created.status();
  EXPECT_TRUE(absl::StrContains(created.status().message(), "contactScheduleSource")) << created.status();
}

}  // namespace ocs2::humanoid
