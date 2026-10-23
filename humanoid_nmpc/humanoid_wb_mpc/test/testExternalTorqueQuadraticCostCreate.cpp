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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/swing/SwingTrajectoryFromConfig.h"
#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"

/**
 * ExternalTorqueQuadraticCostAD::Create() on the G1 whole-body model: the joints of the cost are resolved by name
 * before anything is taped, and a joint that is not an MPC joint, or a weight count that is not one per joint, is
 * refused with a Status. The joint index used to be looked up inside the taped cost function, by a lookup that threw
 * out of the constructor. Builds no CppAD model: every case is refused before the cost is constructed.
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

class ExternalTorqueQuadraticCostCreateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // LINT.IfChange(robot_files)
    const std::string taskFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto");
    const std::string urdfFile = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
    const std::string referenceFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto");
    // LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/BUILD.bazel:external_torque_test_data)
    ASSERT_FALSE(taskFile.empty() || urdfFile.empty() || referenceFile.empty()) << "the G1 whole-body files are not in the runfiles";
    const absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(taskFile);
    ASSERT_TRUE(task.ok()) << task.status();
    absl::StatusOr<ModelSettings> settings = ModelSettings::Create(*task, urdfFile, "wb_mpc_", /*verbose=*/false);
    ASSERT_TRUE(settings.ok()) << settings.status();
    modelSettings_ = std::make_unique<ModelSettings>(*std::move(settings));
    absl::StatusOr<PinocchioInterface> pinocchioInterface = loadCustomPinocchioInterface(*task, urdfFile, *modelSettings_);
    ASSERT_TRUE(pinocchioInterface.ok()) << pinocchioInterface.status();
    pinocchioInterface_ = std::make_unique<PinocchioInterface>(*std::move(pinocchioInterface));
    model_ = std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*modelSettings_);
    modelAd_ = std::make_unique<WBAccelMpcRobotModel<ad_scalar_t>>(*modelSettings_);
    absl::StatusOr<std::shared_ptr<GaitSchedule>> gaitSchedule = GaitSchedule::Create(referenceFile, *modelSettings_, /*verbose=*/false);
    ASSERT_TRUE(gaitSchedule.ok()) << gaitSchedule.status();
    absl::StatusOr<SwingTrajectoryPlanner::Config> swingConfig = swingTrajectorySettingsFromConfig(task->swing_trajectory_config);
    ASSERT_TRUE(swingConfig.ok()) << swingConfig.status();
    referenceManager_ = std::make_unique<SwitchedModelReferenceManager>(
        *std::move(gaitSchedule), std::make_unique<SwingTrajectoryPlanner>(*swingConfig, kNumContacts), *pinocchioInterface_, *model_);
  }

  absl::StatusOr<std::unique_ptr<ExternalTorqueQuadraticCostAD>> create(const ExternalTorqueQuadraticCostAD::Config& config) const {
    return ExternalTorqueQuadraticCostAD::Create(/*endEffectorIndex=*/0, config, *referenceManager_, *pinocchioInterface_, *modelAd_,
                                                 *modelSettings_);
  }

  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  std::unique_ptr<WBAccelMpcRobotModel<scalar_t>> model_;
  std::unique_ptr<WBAccelMpcRobotModel<ad_scalar_t>> modelAd_;
  std::unique_ptr<SwitchedModelReferenceManager> referenceManager_;
};

TEST_F(ExternalTorqueQuadraticCostCreateTest, AJointThatIsNotAnMpcJointIsNotFoundNamingIt) {
  ExternalTorqueQuadraticCostAD::Config config;
  config.activeJointNames = {modelSettings_->mpcModelJointNames.front(), "no_such_knee"};
  config.weights = vector_t::Ones(2);
  const absl::StatusOr<std::unique_ptr<ExternalTorqueQuadraticCostAD>> cost = create(config);
  EXPECT_EQ(cost.status().code(), absl::StatusCode::kNotFound);
  EXPECT_TRUE(absl::StrContains(cost.status().message(), "no_such_knee")) << cost.status();
}

TEST_F(ExternalTorqueQuadraticCostCreateTest, AWeightCountThatIsNotOnePerJointIsRefused) {
  ExternalTorqueQuadraticCostAD::Config config;
  config.activeJointNames = {modelSettings_->mpcModelJointNames[0], modelSettings_->mpcModelJointNames[1]};
  config.weights = vector_t::Ones(3);
  const absl::StatusOr<std::unique_ptr<ExternalTorqueQuadraticCostAD>> cost = create(config);
  EXPECT_EQ(cost.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(cost.status().message(), "one weight per joint")) << cost.status();
}

}  // namespace
}  // namespace ocs2::humanoid
