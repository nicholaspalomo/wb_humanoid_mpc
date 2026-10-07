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

#include "support/AtlasReferenceStack.h"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "ocs2_centroidal_model/AccessHelperFunctions.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"

#include "humanoid_centroidal_mpc/mrt/CentroidalMpcResetTarget.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/contact_planning/ContactPlanningFromConfig.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_common_mpc/config/solver/SolverSettingsFromConfig.h"
#include "humanoid_common_mpc/config/swing/SwingTrajectoryFromConfig.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningModelParameters.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/gait_file.nproto.h"
#include "support/TypedConfigFiles.h"

namespace ocs2::humanoid {

std::string atlasRunfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR")) roots.emplace_back(std::filesystem::path(srcDir) / "_main");
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

AtlasReferenceStack::AtlasReferenceStack(ScheduleSource scheduleSource, absl::string_view plannerType) {
  taskFile_ = atlasRunfilePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto");
  referenceFile_ = atlasRunfilePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.textproto");
  urdfFile_ = atlasRunfilePath("robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf");
  gaitFile_ = atlasRunfilePath("humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto");
  CHECK(!taskFile_.empty() && !referenceFile_.empty() && !urdfFile_.empty() && !gaitFile_.empty())
      << "[AtlasReferenceStack] the DRC Atlas files or the gait file are not in the runfiles";

  // The typed files, converted as CentroidalMpcInterface converts them.
  absl::StatusOr<CentroidalMpcConfig> config = loadCentroidalMpcConfig(taskFile_, referenceFile_);
  CHECK(config.ok()) << config.status();
  config_ = *std::move(config);
  const absl::StatusOr<mpc_config::GaitFile> gaits = loadGaitFile(gaitFile_);
  CHECK(gaits.ok()) << gaits.status();
  const absl::StatusOr<ReferenceSettings> referenceSettings = referenceSettingsFromConfig(config_.reference);
  CHECK(referenceSettings.ok()) << referenceSettings.status();

  mpcSettings_ = toMpcSettings(config_.task.mpc);
  modelSettings_ =
      std::make_unique<ModelSettings>(ModelSettings::Create(config_.task, urdfFile_, "centroidal_mpc_", /*verbose=*/false).value());
  pinocchioInterface_ = std::make_unique<PinocchioInterface>(
      loadCustomPinocchioInterface(config_.task, urdfFile_, *modelSettings_, /*scaleTotalMass=*/false).value());
  info_ = centroidalModelInfoOf(config_, *pinocchioInterface_, *modelSettings_).value();
  model_ = std::make_unique<CentroidalMpcRobotModel<scalar_t>>(*modelSettings_, *pinocchioInterface_, info_);
  initialState_ = initialStateOf(config_.task, *modelSettings_).value();

  std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPlanner(
      new SwingTrajectoryPlanner(swingTrajectorySettingsFromConfig(config_.task.swing_trajectory_config).value(), kNumContacts));
  if (scheduleSource == ScheduleSource::kGaitSchedule) {
    referenceManager_ =
        std::make_shared<SwitchedModelReferenceManager>(GaitSchedule::Create(config_.reference, *modelSettings_, /*verbose=*/false).value(),
                                                        std::move(swingTrajectoryPlanner), *pinocchioInterface_, *model_);
  } else {
    // As CentroidalMpcInterface::setupReferenceManager() builds it, synchronous so that every plan is deterministic.
    absl::StatusOr<ContactPlanningConfig> planningConfig =
        contactPlanningConfigFromOptionalFile(config_.contactPlanning.has_value() ? &*config_.contactPlanning : nullptr,
                                              ContactPlanningValidation::kDeferUntilModelParametersApplied);
    CHECK(planningConfig.ok()) << planningConfig.status();
    planningConfig->planner.runInBackgroundThread = false;
    if (!plannerType.empty() && plannerType != planningConfig->planner.type) {
      planningConfig->planner.type = std::string(plannerType);
      planningConfig->planner.dt = 0.1;
      planningConfig->planner.numNodes = 12;
      planningConfig->planner.commitTime = 0.3;
      planningConfig->shared.gaitLimits.minSwingDuration = 0.4;
      planningConfig->shared.gaitLimits.maxSwingDuration = 0.5;
      planningConfig->planner.maxSolveTime = 1.0e3;
      planningConfig->eventShiftLocalSearch.maxTime = 1.0e3;
    }
    const absl::StatusOr<ContactWrenchConeConstraint::Config> coneConfig = contactWrenchConeConfigFromConfig(config_.task.contacts);
    CHECK(coneConfig.ok()) << coneConfig.status();
    ContactPlanningGroundParameters ground;
    ground.frictionCoefficient = coneConfig->frictionCoefficient;
    ground.torsionalFrictionCoefficient = coneConfig->torsionalFrictionCoefficient;
    const ContactRectangle footprint = contactRectangleFromConfig(config_.task.contacts, *modelSettings_, /*contactIndex=*/0).value();
    ground.footprintHalfLengthX = 0.5 * (footprint.getBounds().x_max - footprint.getBounds().x_min);
    ground.footprintHalfWidthY = 0.5 * (footprint.getBounds().y_max - footprint.getBounds().y_min);
    const ContactPlanningModelParameters modelParameters =
        deriveContactPlanningModelParameters(*pinocchioInterface_, *model_, initialState_, modelSettings_->contactParentJointNames, ground,
                                             planningConfig->shared.gravity, planningConfig->stepWidth.nominalStepWidth);
    modelParameters.applyTo(*planningConfig);
    CHECK(planningConfig->validateStatus().ok()) << planningConfig->validateStatus();
    absl::StatusOr<std::shared_ptr<ContactPlanningReferenceManager>> manager =
        ContactPlanningReferenceManager::Create(GaitSchedule::Create(config_.reference, *modelSettings_, /*verbose=*/false).value(),
                                                std::move(swingTrajectoryPlanner), *pinocchioInterface_, *model_, *planningConfig);
    CHECK(manager.ok()) << manager.status();
    planningReferenceManager_ = *std::move(manager);
    absl::StatusOr<std::shared_ptr<ContactPlannerModule>> module =
        ContactPlannerModule::Create(planningReferenceManager_, *planningConfig, modelParameters);
    CHECK(module.ok()) << module.status();
    contactPlannerModule_ = *std::move(module);
    referenceManager_ = planningReferenceManager_;
  }

  targetCalculator_ =
      CentroidalMpcTargetTrajectoriesCalculator::Create(config_.reference, *model_, *pinocchioInterface_, info_, mpcSettings_.timeHorizon_)
          .value();
  targetCalculator_->setTerrainHeightSource(
      [referenceManager = referenceManager_]() { return referenceManager->getAppliedTerrainHeight(); });
  CentroidalMpcTargetTrajectoriesCalculator* absl_nonnull calculator = targetCalculator_.get();
  motionManager_ = ProceduralMpcMotionManager::Create(
                       *gaits, *referenceSettings, referenceManager_, *model_,
                       [calculator](const vector4_t& velocityTarget, scalar_t initTime, scalar_t /*finalTime*/, const vector_t& initState) {
                         return calculator->commandedVelocityToTargetTrajectories(velocityTarget, initTime, initState);
                       })
                       .value();
  motionManager_->setResetHook([calculator]() { calculator->reset(); });

  mpc_ = std::make_unique<mpc_test::ScriptedMpc>(mpcSettings_, model_->getInputDim());
  mpc_->getSolverPtr()->setReferenceManager(referenceManager_);
  mpc_->getSolverPtr()->addSynchronizedModule(motionManager_);
  if (contactPlannerModule_ != nullptr) mpc_->getSolverPtr()->addSynchronizedModule(contactPlannerModule_);
  command(0.0);
}

void AtlasReferenceStack::command(scalar_t forward, scalar_t lateral, scalar_t yawRate) {
  // No pelvis height in the command: every target stands defaultBaseHeight above the ground.
  motionManager_->setAndScaleVelocityCommand(WalkingVelocityCommand(forward, lateral, /*desired_pelvis_h=*/0.0, yawRate));
}

TargetTrajectories AtlasReferenceStack::resetTarget(scalar_t time, const vector_t& state) const {
  SystemObservation observation;
  observation.time = time;
  observation.state = state;
  return centroidalMpcResetTargetTrajectories(observation, info_, *model_, *pinocchioInterface_);
}

vector_t AtlasReferenceStack::standingAt(const vector_t& state, scalar_t x, scalar_t y, scalar_t yaw) const {
  vector_t moved = state;
  centroidal_model::getNormalizedMomentum(moved, info_).setZero();
  moved(6) = x;
  moved(7) = y;
  moved(9) = yaw;
  return moved;
}

}  // namespace ocs2::humanoid
