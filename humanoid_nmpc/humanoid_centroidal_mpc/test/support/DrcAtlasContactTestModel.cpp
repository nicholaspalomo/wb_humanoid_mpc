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

#include "support/DrcAtlasContactTestModel.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ocs2_centroidal_model/AccessHelperFunctions.h"
#include "ocs2_centroidal_model/ModelHelperFunctions.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/config/swing/SwingTrajectoryFromConfig.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "support/TypedConfigFiles.h"

namespace ocs2::humanoid {

namespace {
/** [s] half-width of the stance brackets around the requested mode; see setContactFlags(). */
constexpr scalar_t kBracketHalfWidth = 10.0;
}  // namespace

DrcAtlasContactTestModel::DrcAtlasContactTestModel(const std::string& modelNamePrefix) : modelNamePrefix_(modelNamePrefix) {
  const CentroidalRobotFiles files = atlasFiles();
  taskFile_ = files.taskFile;
  referenceFile_ = files.referenceFile;
  urdfFile_ = files.urdfFile;
  absl::StatusOr<CentroidalMpcConfig> config = loadConfigOf(files);
  CHECK(config.ok()) << config.status();
  config_ = *std::move(config);

  modelSettings_ =
      std::make_unique<ModelSettings>(ModelSettings::Create(config_.task, urdfFile_, modelNamePrefix_, /*verbose=*/false).value());
  modelSettings_->recompileLibrariesCppAd = false;
  pinocchioInterface_ = std::make_unique<PinocchioInterface>(
      loadCustomPinocchioInterface(config_.task, urdfFile_, *modelSettings_, /*scaleTotalMass=*/false).value());
  info_ = centroidalModelInfoOf(config_, *pinocchioInterface_, *modelSettings_).value();
  wrenchModel_ = std::make_unique<CentroidalMpcRobotModel<scalar_t>>(*modelSettings_, *pinocchioInterface_, info_);
  adWrenchModel_ = std::make_unique<CentroidalMpcRobotModel<ad_scalar_t>>(*modelSettings_, pinocchioInterface_->toCppAd(), info_.toCppAd());

  // The basis-vector decorator, built through the conversion CentroidalMpcInterface builds it with, so that a test sees
  // the input parameterization - generator set and all - the shipped robot actually runs.
  const absl::StatusOr<feet_array_t<ContactWrenchConeBasisMatrix>> basisMatrices =
      contactWrenchConeBasesFromConfig(config_.task.contacts, *modelSettings_);
  CHECK(basisMatrices.ok()) << basisMatrices.status();
  basisModel_ = std::make_unique<BasisInputsModelDecorator<scalar_t>>(std::unique_ptr<MpcRobotModelBase<scalar_t>>(wrenchModel_->clone()),
                                                                      *basisMatrices, *pinocchioInterface_);

  const CentroidalModelInfoCppAd infoCppAd = info_.toCppAd();
  mappingCppAd_ = std::make_unique<CentroidalModelPinocchioMappingCppAd>(infoCppAd);

  std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPlanner(
      new SwingTrajectoryPlanner(swingTrajectorySettingsFromConfig(config_.task.swing_trajectory_config).value(), kNumContacts));
  referenceManager_ =
      std::make_unique<SwitchedModelReferenceManager>(GaitSchedule::Create(config_.reference, *modelSettings_, /*verbose=*/false).value(),
                                                      std::move(swingTrajectoryPlanner), *pinocchioInterface_, *wrenchModel_);
  // Loaded before the first setContactFlags(): that call drives preSolverRun(), which needs a valid state.
  nominalState_ = initialStateOf(config_.task, *modelSettings_).value();
  setStance();
}

DrcAtlasContactTestModel::~DrcAtlasContactTestModel() = default;

void DrcAtlasContactTestModel::setContactFlags(const contact_flag_t& contacts) {
  // The requested mode is bracketed by stance phases rather than standing alone, because the swing trajectory planner
  // that preSolverRun() drives needs every swing phase to be preceded and followed by a contact phase of the same
  // foot. The brackets are far outside the window the tests query, so getContactFlags() around kQueryTime returns
  // exactly what was asked for, and they already cover the horizon preSolverRun() asks the gait schedule for, so
  // nothing is tiled from the gait template on top of them.
  const ModeSchedule schedule({-kBracketHalfWidth, kBracketHalfWidth},
                              {ModeNumber::kStance, stanceLeg2ModeNumber(contacts), ModeNumber::kStance});
  // It has to go through the GAIT schedule, not through ReferenceManager::setModeSchedule(): that setter only fills a
  // buffer, and SwitchedModelReferenceManager::modifyReferences() overwrites the whole mode schedule from the gait
  // schedule on every preSolverRun() anyway. Setting the buffer alone leaves the terms under test reading whatever the
  // reference file's gait happened to be, so a test would pass or fail by coincidence.
  referenceManager_->getGaitSchedule()->updateModeSchedule(schedule);
  referenceManager_->preSolverRun(kQueryTime, kQueryTime + 1.0, nominalState_, stanceLeg2ModeNumber(contacts));
}

void DrcAtlasContactTestModel::setSwing(size_t swingFoot) {
  contact_flag_t contacts = makeFeetArray(true);
  contacts[swingFoot] = false;
  setContactFlags(contacts);
}

std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> DrcAtlasContactTestModel::makeEndEffectorKinematics(size_t contactIndex,
                                                                                                         size_t inputDim) const {
  const CentroidalModelInfoCppAd infoCppAd = info_.toCppAd();
  const PinocchioEndEffectorKinematicsCppAd::update_pinocchio_interface_callback velocityUpdateCallback =
      [infoCppAd](const ad_vector_t& state, PinocchioInterfaceCppAd& pinocchioInterfaceAd) {
        const ad_vector_t q = centroidal_model::getGeneralizedCoordinates(state, infoCppAd);
        updateCentroidalDynamics(pinocchioInterfaceAd, infoCppAd, q);
      };
  const std::string& footName = modelSettings_->contactNames[contactIndex];
  // The generated library is keyed to the input dimension: the wrench-space and the basis-vector parameterizations
  // have different input dimensions and must not share a cached model.
  const std::string modelName = absl::StrCat(modelNamePrefix_, footName, "_u", inputDim);
  return std::make_unique<PinocchioEndEffectorKinematicsCppAd>(
      *pinocchioInterface_, *mappingCppAd_, std::vector<std::string>{footName}, info_.stateDim, inputDim, velocityUpdateCallback, modelName,
      modelSettings_->modelFolderCppAd, /*recompileLibraries=*/false, /*verbose=*/false);
}

std::unique_ptr<FootprintCornerHeights> DrcAtlasContactTestModel::makeCornerHeights(size_t contactIndex,
                                                                                    const std::string& modelNameSuffix) const {
  const ContactRectangle footprint = contactRectangle(contactIndex);
  std::vector<std::string> cornerFrames;
  cornerFrames.reserve(footprint.getNumberOfContactPoints());
  for (size_t corner = 0; corner < footprint.getNumberOfContactPoints(); ++corner) {
    cornerFrames.push_back(footprint.getPolygonPointFrameName(static_cast<int>(corner)));
  }
  return std::make_unique<FootprintCornerHeights>(*pinocchioInterface_, *adWrenchModel_, std::move(cornerFrames),
                                                  absl::StrCat(modelNamePrefix_, modelNameSuffix), *modelSettings_);
}

Eigen::Index DrcAtlasContactTestModel::anklePitchStateIndex(size_t contactIndex) const {
  const std::string jointName = contactIndex == kContactLeftIndex ? "l_leg_aky" : "r_leg_aky";
  const std::vector<std::string>& jointNames = modelSettings_->mpcModelJointNames;
  const std::vector<std::string>::const_iterator found = std::find(jointNames.begin(), jointNames.end(), jointName);
  CHECK(found != jointNames.end()) << "[DrcAtlasContactTestModel] no joint named " << jointName << " in the MPC model";
  return static_cast<Eigen::Index>(wrenchModel_->getJointStartindex() + static_cast<size_t>(std::distance(jointNames.begin(), found)));
}

ContactRectangle DrcAtlasContactTestModel::contactRectangle(size_t contactIndex) const {
  return contactRectangleFromConfig(config_.task.contacts, *modelSettings_, static_cast<int>(contactIndex)).value();
}

vector_t DrcAtlasContactTestModel::makeInput(const MpcRobotModelBase<scalar_t>& model, size_t loadedFoot, scalar_t normalForce) const {
  vector_t input = vector_t::Zero(model.getInputDim());
  vector6_t wrench = vector6_t::Zero();
  wrench(kWrenchForceZIndex) = normalForce;
  model.setContactWrench(input, wrench, loadedFoot);
  for (size_t index = 0; index < model.getJointDim(); ++index) {
    input(model.getJointVelocitiesStartindex() + index) = 0.05 * static_cast<scalar_t>(index % 5) - 0.1;
  }
  return input;
}

}  // namespace ocs2::humanoid
