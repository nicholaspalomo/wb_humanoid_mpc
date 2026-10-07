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

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/costs/TaskSpaceCostFromConfig.h"
#include "humanoid_common_mpc/config/swing/SwingTrajectoryFromConfig.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "nproto/Textproto.h"
#include "support/TypedConfigFiles.h"

namespace ocs2::humanoid {

namespace {

/** The foot cost's settings of the task file `text` (the textproto parser, then taskSpaceFootCostFromConfig()). */
absl::StatusOr<TaskSpaceFootCostSettings> footCostOfText(absl::string_view text) {
  ASSIGN_OR_RETURN(const humanoid_mpc_config::TaskFile message,
                   nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(text, /*sourceName=*/"task file"));
  mpc_config::TaskFile task;
  RETURN_IF_ERROR(mpc_config::FromProto(message, &task));
  return taskSpaceFootCostFromConfig(task.task_space_foot_cost);
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: task_space_foot_cost.active_phases reaches the foot cost's settings, a swing foot only where a file leaves it
// out; the boolean it replaced is refused naming it
// ─────────────────────────────────────────────────────────────────────────────
TEST(ActiveInStanceTest, TaskFileActivePhasesReachTheFootCostSettings) {
  const absl::StatusOr<CentroidalMpcConfig> atlas = loadConfigOf(atlasFiles());
  ASSERT_TRUE(atlas.ok()) << atlas.status();
  const absl::StatusOr<TaskSpaceFootCostSettings> shipped = taskSpaceFootCostFromConfig(atlas->task.task_space_foot_cost);
  ASSERT_TRUE(shipped.ok()) << shipped.status();
  EXPECT_FALSE(shipped->activeInStance) << "the DRC Atlas task file must ship task_space_foot_cost.active_phases \"swing\"";

  const absl::StatusOr<TaskSpaceFootCostSettings> on = footCostOfText("task_space_foot_cost { active_phases: \"swing_and_stance\" }\n");
  ASSERT_TRUE(on.ok()) << on.status();
  EXPECT_TRUE(on->activeInStance);
  const absl::StatusOr<TaskSpaceFootCostSettings> off = footCostOfText("task_space_foot_cost { active_phases: \"swing\" }\n");
  ASSERT_TRUE(off.ok()) << off.status();
  EXPECT_FALSE(off->activeInStance);
  const absl::StatusOr<TaskSpaceFootCostSettings> absent = footCostOfText("task_space_foot_cost { }\n");
  ASSERT_TRUE(absent.ok()) << absent.status();
  EXPECT_FALSE(absent->activeInStance) << "a file without active_phases must leave the foot cost off in stance";
  const absl::StatusOr<TaskSpaceFootCostSettings> retired = footCostOfText("task_space_foot_cost { active_in_stance: true }\n");
  EXPECT_EQ(retired.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(retired.status().message().find("'active_in_stance' is retired"), absl::string_view::npos) << retired.status();
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Verify CentroidalMpcEndEffectorFootCost::isActive(time) logic
// ─────────────────────────────────────────────────────────────────────────────
TEST(ActiveInStanceTest, VerifyFootCostIsActiveBehaviorWithAtlasModel) {
  const CentroidalRobotFiles files = atlasFiles();
  const absl::StatusOr<CentroidalMpcConfig> config = loadConfigOf(files);
  ASSERT_TRUE(config.ok()) << config.status();

  ModelSettings modelSettings = ModelSettings::Create(config->task, files.urdfFile, "drc_atlas", /*verbose=*/false).value();
  PinocchioInterface pinocchioInterface(
      loadCustomPinocchioInterface(config->task, files.urdfFile, modelSettings, /*scaleTotalMass=*/false).value());

  CentroidalModelInfo centroidalModelInfo = centroidalModelInfoOf(*config, pinocchioInterface, modelSettings).value();

  CentroidalMpcRobotModel<scalar_t> mpcRobotModel(modelSettings, pinocchioInterface, centroidalModelInfo);
  CentroidalMpcRobotModel<ad_scalar_t> mpcRobotModelAD(modelSettings, pinocchioInterface.toCppAd(), centroidalModelInfo.toCppAd());

  std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPlanner(
      new SwingTrajectoryPlanner(swingTrajectorySettingsFromConfig(config->task.swing_trajectory_config).value(), /*numFeet=*/2));

  std::shared_ptr<GaitSchedule> gaitSchedule = GaitSchedule::Create(config->reference, modelSettings, /*verbose=*/false).value();
  ModeSchedule initialModeSchedule = gaitSchedule->getModeSchedule(0.0, 1.0);

  auto referenceManager = std::make_shared<SwitchedModelReferenceManager>(std::move(gaitSchedule), std::move(swingTrajectoryPlanner),
                                                                          pinocchioInterface, mpcRobotModel);
  referenceManager->setModeSchedule(initialModeSchedule);
  referenceManager->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, vector_t::Zero(centroidalModelInfo.stateDim), /*initMode=*/3);

  // Time t=0.0 is double stance for DRC Atlas
  const scalar_t stanceTime = 0.0;
  ASSERT_TRUE(referenceManager->isInContact(stanceTime, /*contactIndex=*/0)) << "Atlas left foot should be in contact at t=0";
  ASSERT_TRUE(referenceManager->isInContact(stanceTime, /*contactIndex=*/1)) << "Atlas right foot should be in contact at t=0";

  EndEffectorKinematicsWeights weights = taskSpaceFootCostFromConfig(config->task.task_space_foot_cost).value().weights;

  // 1. Check with activeInStance = false:
  CentroidalMpcEndEffectorFootCost footCostInactive(*referenceManager, weights, pinocchioInterface, mpcRobotModelAD, /*contactIndex=*/0,
                                                    "foot_l_contact_TaskSpaceKinematicsCost", modelSettings, /*activeInStance=*/false);

  EXPECT_FALSE(footCostInactive.getActiveInStance());
  EXPECT_FALSE(footCostInactive.isActive(stanceTime)) << "When activeInStance is false, foot cost MUST be inactive during stance";

  // 2. Check with activeInStance = true:
  CentroidalMpcEndEffectorFootCost footCostActive(*referenceManager, weights, pinocchioInterface, mpcRobotModelAD, /*contactIndex=*/0,
                                                  "foot_l_contact_TaskSpaceKinematicsCost", modelSettings, /*activeInStance=*/true);

  EXPECT_TRUE(footCostActive.getActiveInStance());
  EXPECT_TRUE(footCostActive.isActive(stanceTime)) << "When activeInStance is true, foot cost MUST BE ACTIVE during stance!";

  // 3. Test runtime dynamic toggling via setActiveInStance:
  footCostActive.setActiveInStance(false);
  EXPECT_FALSE(footCostActive.isActive(stanceTime));

  footCostActive.setActiveInStance(true);
  EXPECT_TRUE(footCostActive.isActive(stanceTime));

  // 4. Test the overall on/off switch setActive:
  footCostActive.setActive(false);
  EXPECT_FALSE(footCostActive.isActive(stanceTime));

  footCostActive.setActive(true);
  EXPECT_TRUE(footCostActive.isActive(stanceTime));

  // 5. Test clone preserves activeInStance:
  std::unique_ptr<CentroidalMpcEndEffectorFootCost> clonedCost(footCostActive.clone());
  EXPECT_TRUE(clonedCost->getActiveInStance());
  EXPECT_TRUE(clonedCost->isActive(stanceTime));
}

}  // namespace ocs2::humanoid
