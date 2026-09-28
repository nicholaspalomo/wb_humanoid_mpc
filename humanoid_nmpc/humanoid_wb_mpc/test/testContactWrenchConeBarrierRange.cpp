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
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/HumanoidCostConstraintFactory.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"

/**
 * HumanoidCostConstraintFactory::getContactWrenchConeConstraint() range-checks the barrier it wraps the cone in. `mu` and
 * `delta` of contacts.contactWrenchConeSoftConstraint used to be read with loadData::loadPtreeValue and used as they
 * came: a negative mu made the penalty a REWARD for leaving the cone, and a value that was not a number threw out of a
 * function that returns a Status. Both are now an InvalidArgument naming the key, as the ground's keys of the same block
 * are. Built on the G1 whole-body model, whose task file has no such block: each case writes one beside it. Builds no
 * CppAD model.
 */
namespace ocs2::humanoid {
namespace {

constexpr absl::string_view kMuKey = "contacts.contactWrenchConeSoftConstraint.mu";
constexpr absl::string_view kDeltaKey = "contacts.contactWrenchConeSoftConstraint.delta";

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

class ContactWrenchConeBarrierRangeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    shippedTaskFile_ = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml");
    urdfFile_ = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
    referenceFile_ = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.yaml");
    ASSERT_FALSE(shippedTaskFile_.empty() || urdfFile_.empty() || referenceFile_.empty())
        << "the G1 whole-body files are not in the runfiles";
    modelSettings_ = std::make_unique<ModelSettings>(shippedTaskFile_, urdfFile_, "wb_mpc_", /*verbose=*/false);
    absl::StatusOr<PinocchioInterface> pinocchioInterface = loadCustomPinocchioInterface(shippedTaskFile_, urdfFile_, *modelSettings_);
    ASSERT_TRUE(pinocchioInterface.ok()) << pinocchioInterface.status();
    pinocchioInterface_ = std::make_unique<PinocchioInterface>(*std::move(pinocchioInterface));
    model_ = std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*modelSettings_);
    modelAD_ = std::make_unique<WBAccelMpcRobotModel<ad_scalar_t>>(*modelSettings_);
    referenceManager_ = std::make_unique<SwitchedModelReferenceManager>(
        GaitSchedule::loadGaitSchedule(referenceFile_, *modelSettings_, /*verbose=*/false),
        std::make_unique<SwingTrajectoryPlanner>(
            loadSwingTrajectorySettings(shippedTaskFile_, "swing_trajectory_config", /*verbose=*/false), N_CONTACTS),
        *pinocchioInterface_, *model_);
  }

  /** The shipped G1 task file with a contact wrench cone block whose barrier lines are `barrierLines`. */
  std::string writeTaskFile(const std::string& name, const std::string& barrierLines) const {
    std::ifstream in(shippedTaskFile_);
    std::string task((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::string contacts = "\ncontacts:\n";
    EXPECT_NE(task.find(contacts), std::string::npos) << "the shipped task file no longer has a contacts block";
    EXPECT_EQ(task.find("contactWrenchConeSoftConstraint"), std::string::npos) << "the shipped task file now carries the block";
    task.insert(task.find(contacts) + contacts.size(),
                "  contactWrenchConeSoftConstraint:\n"
                "    frictionCoefficient: 0.5\n"
                "    torsionalFrictionCoefficient: 0.05\n"
                "    minNormalForce: 5\n"
                "    gripperForce: 0\n"
                "    numBasisVectors: 4\n" +
                    barrierLines);
    const std::string path =
        (std::filesystem::path(testing::TempDir()) / absl::StrCat("testContactWrenchConeBarrierRange_", name, ".yaml")).string();
    std::ofstream(path) << task;
    return path;
  }

  absl::Status build(const std::string& taskFile, bool scheduleGated) const {
    const HumanoidCostConstraintFactory factory(taskFile, referenceFile_, *referenceManager_, *pinocchioInterface_, *model_, *modelAD_,
                                                *modelSettings_, /*verbose=*/false, scheduleGated);
    return factory.getContactWrenchConeConstraint(CONTACT_LEFT_INDEX).status();
  }

  std::string shippedTaskFile_, urdfFile_, referenceFile_;
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  std::unique_ptr<WBAccelMpcRobotModel<scalar_t>> model_;
  std::unique_ptr<WBAccelMpcRobotModel<ad_scalar_t>> modelAD_;
  std::unique_ptr<SwitchedModelReferenceManager> referenceManager_;
};

}  // namespace

TEST_F(ContactWrenchConeBarrierRangeTest, APositiveBarrierBuildsTheCone) {
  // Positive control: the values the DRC Atlas and the SA01 ship, gated on the schedule and not. So is a block that
  // leaves both keys out, which keeps RelaxedBarrierPenalty's defaults.
  const std::string shipped = writeTaskFile("shipped", "    mu: 0.2\n    delta: 5\n");
  const std::string defaults = writeTaskFile("defaults", "");
  for (const bool scheduleGated : {true, false}) {
    EXPECT_EQ(build(shipped, scheduleGated), absl::OkStatus()) << "schedule gated " << scheduleGated;
    EXPECT_EQ(build(defaults, scheduleGated), absl::OkStatus()) << "schedule gated " << scheduleGated;
  }
}

TEST_F(ContactWrenchConeBarrierRangeTest, ANegativeOrZeroMuIsRefusedNamingItsKey) {
  for (const std::string& mu : {std::string("-0.2"), std::string("0")}) {
    const std::string taskFile = writeTaskFile(absl::StrCat("mu", mu), absl::StrCat("    mu: ", mu, "\n    delta: 5\n"));
    for (const bool scheduleGated : {true, false}) {
      const absl::Status status = build(taskFile, scheduleGated);
      EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << "mu " << mu << ": " << status;
      EXPECT_TRUE(absl::StrContains(status.message(), kMuKey)) << status;
    }
  }
}

TEST_F(ContactWrenchConeBarrierRangeTest, ANonPositiveDeltaIsRefusedNamingItsKey) {
  for (const std::string& delta : {std::string("-1"), std::string("0")}) {
    const std::string taskFile = writeTaskFile(absl::StrCat("delta", delta), absl::StrCat("    mu: 0.2\n    delta: ", delta, "\n"));
    const absl::Status status = build(taskFile, /*scheduleGated=*/true);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << "delta " << delta << ": " << status;
    EXPECT_TRUE(absl::StrContains(status.message(), kDeltaKey)) << status;
  }
}

TEST_F(ContactWrenchConeBarrierRangeTest, AValueThatIsNotANumberIsRefusedNamingItsKey) {
  const absl::Status mu = build(writeTaskFile("mu_word", "    mu: strong\n    delta: 5\n"), /*scheduleGated=*/true);
  EXPECT_EQ(mu.code(), absl::StatusCode::kInvalidArgument) << mu;
  EXPECT_TRUE(absl::StrContains(mu.message(), kMuKey)) << mu;
  const absl::Status delta = build(writeTaskFile("delta_word", "    mu: 0.2\n    delta: wide\n"), /*scheduleGated=*/true);
  EXPECT_EQ(delta.code(), absl::StatusCode::kInvalidArgument) << delta;
  EXPECT_TRUE(absl::StrContains(delta.message(), kDeltaKey)) << delta;
}

}  // namespace ocs2::humanoid
