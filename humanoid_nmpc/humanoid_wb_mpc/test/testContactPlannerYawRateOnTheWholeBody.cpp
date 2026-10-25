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
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/swing/SwingTrajectoryFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"

/**
 * The contact planner's reference manager on the whole-body model, which stores
 * the base velocity (v, w_B) where the centroidal model stores the normalized
 * momentum. The commanded and the measured yaw rate are read through the
 * layout-agnostic getTargetPlanarCommand(). They used to be read from entry
 * kTwistYawRateIndex of getBaseComVelocity(), which is L_z / m on the
 * centroidal layout but the body roll rate w_x on the whole body, so a
 * commanded turn of a level whole-body knot read as zero.
 */
namespace ocs2::humanoid {
namespace {

constexpr scalar_t kHorizon = 1.0;
// [rad/s] The commanded and measured turn about the world z axis.
constexpr scalar_t kYawRate = 0.5;

constexpr char kTaskFile[] =
    "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto";
constexpr char kUrdfFile[] =
    "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";
constexpr char kReferenceFile[] =
    "robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto";

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

TEST(ContactPlannerYawRateOnTheWholeBody,
     readsTheCommandedAndTheMeasuredTurnOfALevelKnot) {
  const std::string taskFile = runfilePath(kTaskFile);
  const std::string urdfFile = runfilePath(kUrdfFile);
  const std::string referenceFile = runfilePath(kReferenceFile);
  ASSERT_FALSE(taskFile.empty() || urdfFile.empty() || referenceFile.empty())
      << "the G1 whole-body files are not in the runfiles";
  const absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(taskFile);
  ASSERT_TRUE(task.ok()) << task.status();
  const absl::StatusOr<ModelSettings> modelSettings =
      ModelSettings::Create(*task, urdfFile, "wb_mpc_", /*verbose=*/false);
  ASSERT_TRUE(modelSettings.ok()) << modelSettings.status();
  const absl::StatusOr<PinocchioInterface> pinocchioInterface =
      loadCustomPinocchioInterface(*task, urdfFile, *modelSettings);
  ASSERT_TRUE(pinocchioInterface.ok()) << pinocchioInterface.status();
  const WBAccelMpcRobotModel<scalar_t> model(*modelSettings);
  absl::StatusOr<SwingTrajectoryPlanner::Config> swingConfig =
      swingTrajectorySettingsFromConfig(task->swing_trajectory_config);
  ASSERT_TRUE(swingConfig.ok()) << swingConfig.status();
  ContactPlanningConfig config;
  config.shared.comHeight = 0.85;  // no model derives the pendulum here (the
                                   // library default 0 means "from the model")
  config.setHeadingModel(/*on=*/true);
  ASSERT_EQ(config.validateStatus(), absl::OkStatus());
  const absl::StatusOr<std::shared_ptr<ContactPlanningReferenceManager>>
      created = ContactPlanningReferenceManager::Create(
          GaitSchedule::Create(referenceFile, *modelSettings, /*verbose=*/false)
              .value(),
          std::make_unique<SwingTrajectoryPlanner>(*std::move(swingConfig),
                                                   kNumContacts),
          *pinocchioInterface, model, config);
  ASSERT_TRUE(created.ok()) << created.status();
  ContactPlanningReferenceManager& referenceManager = **created;

  // A level knot turning at kYawRate about the world z axis, w_B = (0, 0,
  // kYawRate): its roll rate is zero.
  const absl::StatusOr<vector_t> tuningState = stateValuesFromConfig(
      task->initial_state,
      stateInputLayout(*modelSettings, StateInputLayout::Mpc::kWholeBody),
      "initial_state");
  ASSERT_TRUE(tuningState.ok()) << tuningState.status();
  absl::StatusOr<vector_t> initialState = model.stateFromTuning(*tuningState);
  ASSERT_TRUE(initialState.ok()) << initialState.status();
  vector_t turning = *std::move(initialState);
  model.setBaseAngularVelocityFromWorldYawRate(turning, /*yawRate=*/kYawRate);
  ASSERT_NEAR(
      model.getTargetPlanarCommand(turning, /*compositeYawInertia=*/0.0)(
          MpcRobotModelBase<scalar_t>::kPlanarCommandYawRateIndex),
      kYawRate, 1.0e-12);

  const vector_t zeroInput = vector_t::Zero(model.getInputDim());
  referenceManager.setTargetTrajectories(TargetTrajectories(
      {0.0, 2.0 * kHorizon}, {turning, turning}, {zeroInput, zeroInput}));
  referenceManager.preSolverRun(/*initTime=*/0.0, /*finalTime=*/kHorizon,
                                turning, ModeNumber::kStance);
  EXPECT_NEAR(referenceManager.commandedYawRate(), kYawRate, 1.0e-12);
  EXPECT_NEAR(referenceManager.getCommandedYawRate(/*time=*/0.0), kYawRate,
              1.0e-12);

  // The heading model's measured rate: the base's turn about the world z axis.
  const ContactPlannerInput input = referenceManager.makePlannerInput(
      /*initTime=*/0.0, turning, vector2_t::Zero());
  EXPECT_NEAR(input.headingRate, kYawRate, 1.0e-12);
  EXPECT_NEAR(input.headingRateCommand, kYawRate, 1.0e-12);
}

}  // namespace
}  // namespace ocs2::humanoid
