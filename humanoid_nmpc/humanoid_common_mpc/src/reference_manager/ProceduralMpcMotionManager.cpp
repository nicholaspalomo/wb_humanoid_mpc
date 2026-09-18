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

#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"

#include <stdexcept>

#include <ocs2_core/misc/LoadData.h>

#include <cmath>
#include "humanoid_common_mpc/gait/GaitScheduleUpdater.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ProceduralMpcMotionManager::ProceduralMpcMotionManager(const std::string& gaitFile,
                                                       const std::string& referenceFile,
                                                       std::shared_ptr<SwitchedModelReferenceManager> switchedModelReferenceManagerPtr,
                                                       const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                       VelocityTargetToTargetTrajectories velocityTargetToTargetTrajectories)
    : velocityTargetToTargetTrajectoriesFun_(std::move(velocityTargetToTargetTrajectories)),
      switchedModelReferenceManagerPtr_(switchedModelReferenceManagerPtr),
      gaitSchedulePtr_(switchedModelReferenceManagerPtr_->getGaitSchedule()),
      mpcRobotModelPtr_(&mpcRobotModel),
      velocityCommandFilter(5, vector4_t::Zero()) {
  reloadCommandLimits(referenceFile);

  gaitMap_ = getGaitMap(gaitFile);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void ProceduralMpcMotionManager::setVelocityCommandAccelerationLimits(scalar_t maxLinearAcceleration, scalar_t maxAngularAcceleration) {
  if (std::isnan(maxLinearAcceleration) || std::isnan(maxAngularAcceleration)) {
    throw std::invalid_argument("[ProceduralMpcMotionManager] the velocity command acceleration limits must be numbers");
  }
  maxLinearAcceleration_ = maxLinearAcceleration;
  maxAngularAcceleration_ = maxAngularAcceleration;
}

vector4_t ProceduralMpcMotionManager::rateLimitVelocityCommand(
    const vector4_t& target, const vector4_t& current, scalar_t dt, scalar_t maxLinearAcceleration, scalar_t maxAngularAcceleration) {
  vector4_t limited = target;
  if (dt <= 0.0) return current;
  if (maxLinearAcceleration > 0.0) {
    const vector2_t delta = target.head<2>() - current.head<2>();
    const scalar_t maxDelta = maxLinearAcceleration * dt;
    limited.head<2>() = delta.norm() > maxDelta ? vector2_t(current.head<2>() + delta * (maxDelta / delta.norm())) : target.head<2>();
  }
  if (maxAngularAcceleration > 0.0) {
    const scalar_t maxDelta = maxAngularAcceleration * dt;
    limited(3) = current(3) + std::clamp(target(3) - current(3), -maxDelta, maxDelta);
  }
  return limited;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void ProceduralMpcMotionManager::reloadCommandLimits(const std::string& referenceFile) {
  // See TargetTrajectoriesCalculatorBase::reloadCommandLimits: loadData writes through a reference, which an atomic
  // cannot give it, so each limit round-trips through a local seeded with its current value.
  const auto load = [&referenceFile](const std::string& key, std::atomic<scalar_t>& target) {
    scalar_t value = target.load();
    loadData::loadCppDataType(referenceFile, key, value);
    target.store(value);
  };
  load("maxDisplacementVelocityX", maxDisplacementVelocityX_);
  load("maxDisplacementVelocityY", maxDisplacementVelocityY_);
  load("maxDeltaPelvisHeight", maxDeltaPelvisHeight_);
  load("maxRotationVelocity", maxRotationVelocity_);
  // Optional: absent keys keep the ramps off (the historical behaviour, an unramped reference).
  boost::property_tree::ptree pt;
  loadData::readPropertyTree(referenceFile, pt);
  setVelocityCommandAccelerationLimits(pt.get<scalar_t>("maxLinearAcceleration", 0.0), pt.get<scalar_t>("maxAngularAcceleration", 0.0));
}

void ProceduralMpcMotionManager::setAndScaleVelocityCommand(const WalkingVelocityCommand& rawVelocityCommand) {
  velocityCommand_ = scaleWalkingVelocityCommand(rawVelocityCommand);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

WalkingVelocityCommand ProceduralMpcMotionManager::scaleWalkingVelocityCommand(const WalkingVelocityCommand& rawVelocityCommand) const {
  WalkingVelocityCommand scaledCommand = rawVelocityCommand;
  scaledCommand.linear_velocity_x *= maxDisplacementVelocityX_;
  scaledCommand.linear_velocity_y *= maxDisplacementVelocityY_;
  scaledCommand.angular_velocity_z *= maxRotationVelocity_;
  return scaledCommand;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

bool ProceduralMpcMotionManager::transitionToFasterGait(const vector4_t& velCommandVec,
                                                        const vector6_t& baseVelocity,
                                                        const GaitModeStateConfig& cfg) {
  bool fasterGaitRequested = (std::abs(velCommandVec(0)) > cfg.maxLinVelCmd || std::abs(velCommandVec(1)) > cfg.maxLinVelCmd ||
                              std::abs(velCommandVec(3)) > cfg.maxAngVelCmd);

  bool withinMaxSpeedErrorThreshold = (std::abs(baseVelocity(0)) > cfg.maxLinVelCmd - cfg.linVelErrorThresh ||
                                       std::abs(baseVelocity(1)) > cfg.maxLinVelCmd - cfg.linVelErrorThresh ||
                                       std::abs(baseVelocity(3)) > cfg.maxAngVelCmd - cfg.angVelErrorThresh);
  return fasterGaitRequested && withinMaxSpeedErrorThreshold;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

bool ProceduralMpcMotionManager::transitionToSlowerGait(const vector4_t& velCommandVec,
                                                        const vector6_t& baseVelocity,
                                                        const GaitModeStateConfig& cfg) {
  bool slowerGaitRequested = (std::abs(velCommandVec(0)) < cfg.minLinVelCmd && std::abs(velCommandVec(1)) < cfg.minLinVelCmd &&
                              std::abs(velCommandVec(3)) < cfg.minAngVelCmd);

  bool baseSpeedSlowEnough = (std::abs(baseVelocity(0)) < cfg.minLinVelCmd + cfg.linVelErrorThresh &&
                              std::abs(baseVelocity(1)) < cfg.minLinVelCmd + cfg.linVelErrorThresh &&
                              std::abs(velCommandVec(3)) < cfg.minAngVelCmd + cfg.angVelErrorThresh);

  return slowerGaitRequested && baseSpeedSlowEnough;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void ProceduralMpcMotionManager::preSolverRun(scalar_t initTime,
                                              scalar_t finalTime,
                                              const vector_t& initState,
                                              const ReferenceManagerInterface& referenceManager) {
  WalkingVelocityCommand incommingVelCommand = getScaledWalkingVelocityCommand();
  vector4_t filteredVelCommand = velocityCommandFilter.getFilteredVector(incommingVelCommand.toVector());
  // Acceleration limit on the reference (off by default). The first solve, and a solve after time ran backwards (a
  // reset), starts the ramp at the filtered command itself.
  if (!rampInitialised_ || initTime < lastRampTime_) {
    rampedVelocityCommand_ = filteredVelCommand;
    rampInitialised_ = true;
  } else {
    rampedVelocityCommand_ = rateLimitVelocityCommand(filteredVelCommand, rampedVelocityCommand_, initTime - lastRampTime_,
                                                      maxLinearAcceleration_, maxAngularAcceleration_);
  }
  lastRampTime_ = initTime;
  filteredVelCommand = rampedVelocityCommand_;

  // Update TargetTrajectories
  TargetTrajectories targetTrajectories = velocityTargetToTargetTrajectoriesFun_(filteredVelCommand, initTime, finalTime, initState);
  switchedModelReferenceManagerPtr_->setTargetTrajectories(targetTrajectories);

  static GaitModeStateConfig currentCfg = gaitModeStates_[currentGaitMode_];
  vector6_t baseVelocity = mpcRobotModelPtr_->getBaseComVelocity(initState);

  // Do not change the gait pattern for at least 0.5s
  if (initTime > lastGaitChangeTime_ + 0.2) {
    if (transitionToFasterGait(filteredVelCommand, baseVelocity, currentCfg)) {
      std::cout << "filteredVelCommand: " << filteredVelCommand.transpose() << std::endl;
      std::cout << "Linear limits: " << currentCfg.minLinVelCmd << ", " << currentCfg.maxLinVelCmd << std::endl;
      currentGaitMode_++;
      currentCfg = gaitModeStates_[currentGaitMode_];
      currentGaitCommand_ = currentCfg.gaitCommand;
      std::cout << "ProceduralMpcMotionManager: Increasing to gait:" << currentCfg.gaitCommand << std::endl;
      lastGaitChangeTime_ = initTime;
    } else if (transitionToSlowerGait(filteredVelCommand, baseVelocity, currentCfg)) {
      std::cout << "filteredVelCommand: " << filteredVelCommand.transpose() << std::endl;
      std::cout << "Linear limits: " << currentCfg.minLinVelCmd << ", " << currentCfg.maxLinVelCmd << std::endl;
      currentGaitMode_--;
      currentCfg = gaitModeStates_[currentGaitMode_];
      currentGaitCommand_ = currentCfg.gaitCommand;
      std::cout << "ProceduralMpcMotionManager: Decreasing to gait:" << currentCfg.gaitCommand << std::endl;
      lastGaitChangeTime_ = initTime;
    }
  }

  if (currentGaitCommand_ != lastGaitCommand_) {
    ModeSequenceTemplate modeSequenceTemplate = gaitMap_.at(currentGaitCommand_);

    GaitScheduleUpdater::updateGaitSchedule(gaitSchedulePtr_, modeSequenceTemplate, initTime, finalTime);
    lastGaitCommand_ = currentGaitCommand_;
  }
}

}  // namespace ocs2::humanoid
