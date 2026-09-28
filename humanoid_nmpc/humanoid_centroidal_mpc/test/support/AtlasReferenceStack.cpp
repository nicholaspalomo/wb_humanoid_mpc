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

#include "support/AtlasReferenceStack.h"

#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/FactoryFunctions.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningModelParameters.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"

namespace ocs2::humanoid {

std::string atlasRunfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* srcDir = std::getenv("TEST_SRCDIR")) roots.emplace_back(std::filesystem::path(srcDir) / "_main");
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

AtlasReferenceStack::AtlasReferenceStack(ScheduleSource scheduleSource, absl::string_view plannerType) {
  taskFile_ = atlasRunfilePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml");
  referenceFile_ = atlasRunfilePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.yaml");
  urdfFile_ = atlasRunfilePath("robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf");
  gaitFile_ = atlasRunfilePath("humanoid_nmpc/humanoid_common_mpc/config/command/gait.yaml");
  CHECK(!taskFile_.empty() && !referenceFile_.empty() && !urdfFile_.empty() && !gaitFile_.empty())
      << "[AtlasReferenceStack] the DRC Atlas files or the gait file are not in the runfiles";

  mpcSettings_ = mpc::loadSettings(taskFile_, "mpc", /*verbose=*/false);
  modelSettings_ = std::make_unique<ModelSettings>(taskFile_, urdfFile_, "centroidal_mpc_", /*verbose=*/false);
  pinocchioInterface_ =
      std::make_unique<PinocchioInterface>(createCustomPinocchioInterface(taskFile_, urdfFile_, *modelSettings_, /*scaleTotalMass=*/false));
  info_ = centroidal_model::createCentroidalModelInfo(
      *pinocchioInterface_, centroidal_model::loadCentroidalType(taskFile_),
      centroidal_model::loadDefaultJointState(pinocchioInterface_->getModel().nq - 6, referenceFile_), modelSettings_->contactNames3DoF,
      modelSettings_->contactNames6DoF);
  model_ = std::make_unique<CentroidalMpcRobotModel<scalar_t>>(*modelSettings_, *pinocchioInterface_, info_);
  initialState_.setZero(info_.stateDim);
  loadData::loadEigenMatrix(taskFile_, "initialState", initialState_);

  std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPlanner(
      new SwingTrajectoryPlanner(loadSwingTrajectorySettings(taskFile_, "swing_trajectory_config", /*verbose=*/false), N_CONTACTS));
  if (scheduleSource == ScheduleSource::kGaitSchedule) {
    referenceManager_ =
        std::make_shared<SwitchedModelReferenceManager>(GaitSchedule::loadGaitSchedule(referenceFile_, *modelSettings_, /*verbose=*/false),
                                                        std::move(swingTrajectoryPlanner), *pinocchioInterface_, *model_);
  } else {
    // As CentroidalMpcInterface::setupReferenceManager() builds it, synchronous so that every plan is deterministic.
    absl::StatusOr<ContactPlanningConfig> config = loadContactPlanningConfigStatus(
        resolveContactPlanningConfigFile(taskFile_), "contact_planning.", /*verbose=*/false, /*validate=*/false);
    CHECK(config.ok()) << config.status();
    config->planner.runInBackgroundThread = false;
    if (!plannerType.empty() && plannerType != config->planner.type) {
      config->planner.type = std::string(plannerType);
      config->planner.dt = 0.1;
      config->planner.numNodes = 12;
      config->planner.commitTime = 0.3;
      config->shared.gaitLimits.minSwingDuration = 0.4;
      config->shared.gaitLimits.maxSwingDuration = 0.5;
      config->planner.maxSolveTime = 1.0e3;
      config->eventShiftLocalSearch.maxTime = 1.0e3;
    }
    const absl::StatusOr<ContactWrenchConeConstraint::Config> coneConfig =
        ContactWrenchConeConstraint::loadConfig(taskFile_, /*verbose=*/false);
    CHECK(coneConfig.ok()) << coneConfig.status();
    ContactPlanningGroundParameters ground;
    ground.frictionCoefficient = coneConfig->frictionCoefficient;
    ground.torsionalFrictionCoefficient = coneConfig->torsionalFrictionCoefficient;
    const ContactRectangle footprint =
        ContactRectangle::loadContactRectangle(taskFile_, *modelSettings_, /*contactIndex=*/0, /*verbose=*/false);
    ground.footprintHalfLengthX = 0.5 * (footprint.getBounds().x_max - footprint.getBounds().x_min);
    ground.footprintHalfWidthY = 0.5 * (footprint.getBounds().y_max - footprint.getBounds().y_min);
    const ContactPlanningModelParameters modelParameters =
        deriveContactPlanningModelParameters(*pinocchioInterface_, *model_, initialState_, modelSettings_->contactParentJointNames, ground,
                                             config->shared.gravity, config->stepWidth.nominalStepWidth);
    modelParameters.applyTo(*config);
    CHECK(config->validateStatus().ok()) << config->validateStatus();
    absl::StatusOr<std::shared_ptr<ContactPlanningReferenceManager>> manager =
        ContactPlanningReferenceManager::Create(GaitSchedule::loadGaitSchedule(referenceFile_, *modelSettings_, /*verbose=*/false),
                                                std::move(swingTrajectoryPlanner), *pinocchioInterface_, *model_, *config);
    CHECK(manager.ok()) << manager.status();
    planningReferenceManager_ = *std::move(manager);
    absl::StatusOr<std::shared_ptr<ContactPlannerModule>> module =
        ContactPlannerModule::Create(planningReferenceManager_, *config, modelParameters);
    CHECK(module.ok()) << module.status();
    contactPlannerModule_ = *std::move(module);
    referenceManager_ = planningReferenceManager_;
  }

  targetCalculator_ = std::make_unique<CentroidalMpcTargetTrajectoriesCalculator>(referenceFile_, *model_, *pinocchioInterface_, info_,
                                                                                  mpcSettings_.timeHorizon_);
  targetCalculator_->setTerrainHeightSource(
      [referenceManager = referenceManager_]() { return referenceManager->getAppliedTerrainHeight(); });
  CentroidalMpcTargetTrajectoriesCalculator* calculator = targetCalculator_.get();
  motionManager_ = std::make_shared<ProceduralMpcMotionManager>(
      gaitFile_, referenceFile_, referenceManager_, *model_,
      [calculator](const vector4_t& velocityTarget, scalar_t initTime, scalar_t /*finalTime*/, const vector_t& initState) {
        return calculator->commandedVelocityToTargetTrajectories(velocityTarget, initTime, initState);
      });
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
  vector_t target = state;
  centroidal_model::getNormalizedMomentum(target, info_).setZero();
  target(10) = 0.0;
  target(11) = 0.0;
  PinocchioInterface pinocchioInterface = *pinocchioInterface_;
  const vector_t input = weightCompensatingInput(pinocchioInterface, {true, true}, *model_, target);
  return TargetTrajectories({time, time + 2.0}, {target, target}, {input, input});
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
