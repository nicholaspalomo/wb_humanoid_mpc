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
#include <limits>
#include <memory>
#include <optional>
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

#include "humanoid_common_mpc/HumanoidCostConstraintFactory.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/swing/SwingTrajectoryFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/contacts_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"

/**
 * HumanoidCostConstraintFactory::makeContactWrenchConeConstraint() range-checks the barrier it wraps the cone in. `mu`
 * and `delta` of the task file's contacts.contact_wrench_cone_soft_constraint used to be read with
 * loadData::loadPtreeValue and used as they came: a negative mu made the penalty a REWARD for leaving the cone, and a value
 * that was not a number threw out of a function that returns a Status. A barrier that is not a finite positive number is
 * now an InvalidArgument naming its field, as the ground's fields of the same block are, and a word where the number
 * goes is refused by the strict parser with a Status. Built on the G1 whole-body model, whose task file has no such
 * block: each case gives it one. Builds no CppAD model.
 */
namespace ocs2::humanoid {
namespace {

constexpr char kTaskFile[] = "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto";
constexpr char kUrdfFile[] = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";
constexpr char kReferenceFile[] = "robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto";

constexpr absl::string_view kMuField = "contacts.contact_wrench_cone_soft_constraint.mu";
constexpr absl::string_view kDeltaField = "contacts.contact_wrench_cone_soft_constraint.delta";

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

class ContactWrenchConeBarrierRangeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string taskFile = runfilePath(kTaskFile);
    const std::string urdfFile = runfilePath(kUrdfFile);
    const std::string referenceFile = runfilePath(kReferenceFile);
    ASSERT_FALSE(taskFile.empty() || urdfFile.empty() || referenceFile.empty()) << "the G1 whole-body files are not in the runfiles";
    absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(taskFile);
    ASSERT_TRUE(task.ok()) << task.status();
    shippedTask_ = *std::move(task);
    ASSERT_FALSE(shippedTask_.contacts.contact_wrench_cone_soft_constraint.friction_coefficient.has_value())
        << "the shipped task file now carries the block";
    absl::StatusOr<ModelSettings> modelSettings = ModelSettings::Create(shippedTask_, urdfFile, "wb_mpc_", /*verbose=*/false);
    ASSERT_TRUE(modelSettings.ok()) << modelSettings.status();
    modelSettings_ = std::make_unique<ModelSettings>(*std::move(modelSettings));
    absl::StatusOr<PinocchioInterface> pinocchioInterface = loadCustomPinocchioInterface(shippedTask_, urdfFile, *modelSettings_);
    ASSERT_TRUE(pinocchioInterface.ok()) << pinocchioInterface.status();
    pinocchioInterface_ = std::make_unique<PinocchioInterface>(*std::move(pinocchioInterface));
    model_ = std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*modelSettings_);
    modelAD_ = std::make_unique<WBAccelMpcRobotModel<ad_scalar_t>>(*modelSettings_);
    absl::StatusOr<std::shared_ptr<GaitSchedule>> gaitSchedule = GaitSchedule::Create(referenceFile, *modelSettings_, /*verbose=*/false);
    ASSERT_TRUE(gaitSchedule.ok()) << gaitSchedule.status();
    absl::StatusOr<SwingTrajectoryPlanner::Config> swingConfig = swingTrajectorySettingsFromConfig(shippedTask_.swing_trajectory_config);
    ASSERT_TRUE(swingConfig.ok()) << swingConfig.status();
    referenceManager_ = std::make_unique<SwitchedModelReferenceManager>(
        *std::move(gaitSchedule), std::make_unique<SwingTrajectoryPlanner>(*swingConfig, kNumContacts), *pinocchioInterface_, *model_);
  }

  /** The shipped G1 task file with a contact wrench cone block whose barrier is `mu` and `delta` (nullopt: not given). */
  mpc_config::TaskFile taskFileWithBarrier(std::optional<double> mu, std::optional<double> delta) const {
    mpc_config::TaskFile task = shippedTask_;
    mpc_config::ContactsConfig::WrenchCone& cone = task.contacts.contact_wrench_cone_soft_constraint;
    cone.friction_coefficient = 0.5;
    cone.torsional_friction_coefficient = 0.05;
    cone.min_normal_force = 5.0;
    cone.gripper_force = 0.0;
    cone.num_basis_vectors = 4;
    cone.mu = mu;
    cone.delta = delta;
    return task;
  }

  /** The left foot's cone of the factory of `task`; `task` outlives the factory, which reads it. */
  absl::Status build(const mpc_config::TaskFile& task, bool scheduleGated) const {
    const HumanoidCostConstraintFactory factory(&task, StateInputLayout::Mpc::kWholeBody, *referenceManager_, *pinocchioInterface_, *model_,
                                                *modelAD_, *modelSettings_, /*verbose=*/false, scheduleGated);
    return factory.makeContactWrenchConeConstraint(kContactLeftIndex).status();
  }

  mpc_config::TaskFile shippedTask_;
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  std::unique_ptr<WBAccelMpcRobotModel<scalar_t>> model_;
  std::unique_ptr<WBAccelMpcRobotModel<ad_scalar_t>> modelAD_;
  std::unique_ptr<SwitchedModelReferenceManager> referenceManager_;
};

}  // namespace

TEST_F(ContactWrenchConeBarrierRangeTest, APositiveBarrierBuildsTheCone) {
  // Positive control: the values the DRC Atlas and the SA01 ship, gated on the schedule and not. So is a block that
  // leaves both fields out, which keeps RelaxedBarrierPenalty's defaults.
  const mpc_config::TaskFile shipped = taskFileWithBarrier(/*mu=*/0.2, /*delta=*/5.0);
  const mpc_config::TaskFile defaults = taskFileWithBarrier(/*mu=*/std::nullopt, /*delta=*/std::nullopt);
  for (const bool scheduleGated : {true, false}) {
    EXPECT_EQ(build(shipped, scheduleGated), absl::OkStatus()) << "schedule gated " << scheduleGated;
    EXPECT_EQ(build(defaults, scheduleGated), absl::OkStatus()) << "schedule gated " << scheduleGated;
  }
}

TEST_F(ContactWrenchConeBarrierRangeTest, ANegativeOrZeroMuIsRefusedNamingItsField) {
  for (const double mu : {-0.2, 0.0}) {
    const mpc_config::TaskFile task = taskFileWithBarrier(mu, /*delta=*/5.0);
    for (const bool scheduleGated : {true, false}) {
      const absl::Status status = build(task, scheduleGated);
      EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << "mu " << mu << ": " << status;
      EXPECT_TRUE(absl::StrContains(status.message(), kMuField)) << status;
    }
  }
}

TEST_F(ContactWrenchConeBarrierRangeTest, ANonPositiveDeltaIsRefusedNamingItsField) {
  for (const double delta : {-1.0, 0.0}) {
    const absl::Status status = build(taskFileWithBarrier(/*mu=*/0.2, delta), /*scheduleGated=*/true);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << "delta " << delta << ": " << status;
    EXPECT_TRUE(absl::StrContains(status.message(), kDeltaField)) << status;
  }
}

TEST_F(ContactWrenchConeBarrierRangeTest, AValueThatIsNotFiniteIsRefusedNamingItsField) {
  for (const double notFinite : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    const absl::Status mu = build(taskFileWithBarrier(notFinite, /*delta=*/5.0), /*scheduleGated=*/true);
    EXPECT_EQ(mu.code(), absl::StatusCode::kInvalidArgument) << mu;
    EXPECT_TRUE(absl::StrContains(mu.message(), kMuField)) << mu;
    const absl::Status delta = build(taskFileWithBarrier(/*mu=*/0.2, notFinite), /*scheduleGated=*/true);
    EXPECT_EQ(delta.code(), absl::StatusCode::kInvalidArgument) << delta;
    EXPECT_TRUE(absl::StrContains(delta.message(), kDeltaField)) << delta;
  }
}

TEST_F(ContactWrenchConeBarrierRangeTest, AWordWhereTheNumberGoesIsRefusedByTheParserNamingTheFile) {
  // The strict parser returns the refusal, naming the file (and the line and column).
  const std::string path = (std::filesystem::path(testing::TempDir()) / "testContactWrenchConeBarrierRange_mu_word.textproto").string();
  std::ofstream(path) << "contacts {\n  contact_wrench_cone_soft_constraint {\n    mu: strong\n  }\n}\n";
  const absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(path);
  EXPECT_EQ(task.status().code(), absl::StatusCode::kInvalidArgument) << task.status();
  EXPECT_TRUE(absl::StrContains(task.status().message(), path)) << task.status();
}

}  // namespace ocs2::humanoid
