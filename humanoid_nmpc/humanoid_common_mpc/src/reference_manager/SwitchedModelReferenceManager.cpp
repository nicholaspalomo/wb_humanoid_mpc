/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

#include <algorithm>
#include <cmath>

#include <humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h>

namespace ocs2::humanoid {

namespace {

static constexpr scalar_t kHalf = 0.5;
static constexpr scalar_t kPhaseTimeOffset = 0.01;
static constexpr scalar_t kArmSwingLead = 0.15;
static constexpr scalar_t kArmSwingAmplitude = -0.15;

}  // namespace

SwitchedModelReferenceManager::SwitchedModelReferenceManager(std::shared_ptr<GaitSchedule> gaitSchedulePtr,
                                                             std::shared_ptr<SwingTrajectoryPlanner> swingTrajectoryPtr,
                                                             const PinocchioInterface& pinocchioInterface,
                                                             const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                             GaitOptimizationSettings gaitOptimizationSettings)
    : ReferenceManager(TargetTrajectories(), gaitSchedulePtr ? gaitSchedulePtr->getCurrentModeSchedule() : ModeSchedule()),
      gaitSchedulePtr_(std::move(gaitSchedulePtr)),
      swingTrajectoryPtr_(std::move(swingTrajectoryPtr)),
      pinocchioInterface_(pinocchioInterface),
      mpcRobotModelPtr_(&mpcRobotModel),
      gaitOptimizationSettings_(std::move(gaitOptimizationSettings)),
      gaitSwitchingTimeOptimizerPtr_(std::make_unique<GaitSwitchingTimeOptimizer>(gaitOptimizationSettings_)) {}

void SwitchedModelReferenceManager::setGaitOptimizationSettings(GaitOptimizationSettings settings) {
  gaitOptimizationSettings_ = std::move(settings);
  gaitSwitchingTimeOptimizerPtr_->setSettings(gaitOptimizationSettings_);
}

contact_flag_t SwitchedModelReferenceManager::getContactFlags(scalar_t time) const {
  return modeNumber2StanceLeg(this->getModeSchedule().modeAtTime(time));
}

scalar_t SwitchedModelReferenceManager::getPhaseVariable(scalar_t time) const {
  const std::vector<scalar_t>::const_iterator it = std::upper_bound(modeSchedule_.eventTimes.begin(), modeSchedule_.eventTimes.end(), time);
  const scalar_t nextEventTime = *it;
  const scalar_t prevEventTime = *(it - 1);

  if (modeSchedule_.modeAtTime(time) == LF) {
    return (kHalf * (time - prevEventTime) / (nextEventTime - prevEventTime));
  } else if (modeSchedule_.modeAtTime(time) == RF) {
    return (kHalf + kHalf * (time - prevEventTime) / (nextEventTime - prevEventTime));
  } else {
    if (modeSchedule_.modeAtTime(prevEventTime - kPhaseTimeOffset) == LF) {
      return kHalf;
    } else {
      return 0.0;
    }
  }
}

scalar_t SwitchedModelReferenceManager::adaptToCurrentGroundHeight(TargetTrajectories& targetTrajectories,
                                                                   const vector_t& initState,
                                                                   size_t initMode) {
  scalar_t terrainHeight = computeGroundHeightEstimate(pinocchioInterface_, *mpcRobotModelPtr_,
                                                       mpcRobotModelPtr_->getGeneralizedCoordinates(initState), initMode);

  terrainHeight = 0.0;

  for (size_t i = 0; i < targetTrajectories.stateTrajectory.size(); i++) {
    vector_t& targetState = targetTrajectories.stateTrajectory[i];
    const scalar_t heightDifference = terrainHeight - previousGroundHeightEstimate_;
    mpcRobotModelPtr_->adaptBasePoseHeight(targetState, heightDifference);
  }
  previousGroundHeightEstimate_ = terrainHeight;
  return terrainHeight;
}

vector_t SwitchedModelReferenceManager::getDesiredState(const TargetTrajectories& targetTrajectories,
                                                        const vector_t& state,
                                                        scalar_t time) const {
  vector_t xNominal = targetTrajectories.getDesiredState(time);

  if (armSwingReferenceActive_) {
    const scalar_t phaseVariable = this->getPhaseVariable(time);
    vector_t desiredJointAngles = mpcRobotModelPtr_->getJointAngles(xNominal);

    const vector3_t linVelCommand = mpcRobotModelPtr_->getBaseComLinearVelocity(xNominal);
    const scalar_t currentEulerZ = mpcRobotModelPtr_->getBasePose(state)[3];

    const scalar_t localVelXCommand = (std::cos(currentEulerZ) * linVelCommand[0] + std::sin(currentEulerZ) * linVelCommand[1]);

    const ModelSettings& modelSettings = mpcRobotModelPtr_->modelSettings;

    const scalar_t gaitCycleFactor = std::sin(2.0 * M_PI * (phaseVariable - kArmSwingLead)) * localVelXCommand;
    desiredJointAngles[modelSettings.j_l_shoulder_y_index] += kArmSwingAmplitude * gaitCycleFactor;
    desiredJointAngles[modelSettings.j_r_shoulder_y_index] += -kArmSwingAmplitude * gaitCycleFactor;
    desiredJointAngles[modelSettings.j_l_elbow_y_index] += kArmSwingAmplitude * gaitCycleFactor;
    desiredJointAngles[modelSettings.j_r_elbow_y_index] += -kArmSwingAmplitude * gaitCycleFactor;

    mpcRobotModelPtr_->setJointAngles(xNominal, desiredJointAngles);
  }
  return xNominal;
}

bool SwitchedModelReferenceManager::optimizeSwitchingTimes(const PrimalSolution& primalSolution) {
  if (!gaitOptimizationSettings_.enabled || !gaitSwitchingTimeOptimizerPtr_) {
    return false;
  }

  ModeSchedule schedule = gaitSchedulePtr_->getCurrentModeSchedule();
  const bool modified = gaitSwitchingTimeOptimizerPtr_->optimizeEventTimes(primalSolution, schedule);
  if (modified) {
    gaitSchedulePtr_->updateModeSchedule(schedule);
    modeSchedule_ = schedule;
  }
  return modified;
}

bool SwitchedModelReferenceManager::adaptFromContactFeedback(const contact_flag_t& measuredContactFlags, scalar_t currentTime) {
  if (!gaitOptimizationSettings_.enabled || !gaitSwitchingTimeOptimizerPtr_) {
    return false;
  }

  ModeSchedule schedule = gaitSchedulePtr_->getCurrentModeSchedule();
  const bool modified = gaitSwitchingTimeOptimizerPtr_->adaptFromContactFeedback(measuredContactFlags, currentTime, schedule);
  if (modified) {
    gaitSchedulePtr_->updateModeSchedule(schedule);
    modeSchedule_ = schedule;
  }
  return modified;
}

void SwitchedModelReferenceManager::modifyReferences(scalar_t initTime,
                                                     scalar_t finalTime,
                                                     const vector_t& initState,
                                                     size_t initMode,
                                                     TargetTrajectories& targetTrajectories,
                                                     ModeSchedule& modeSchedule) {
  const scalar_t timeHorizon = finalTime - initTime;
  modeSchedule = gaitSchedulePtr_->getModeSchedule(initTime - timeHorizon, finalTime + timeHorizon);

  const scalar_t terrainHeight = adaptToCurrentGroundHeight(targetTrajectories, initState, initMode);

  if (swingTrajectoryPtr_) {
    swingTrajectoryPtr_->update(modeSchedule, terrainHeight);
  }

  modeSchedule_ = modeSchedule;
}

}  // namespace ocs2::humanoid
