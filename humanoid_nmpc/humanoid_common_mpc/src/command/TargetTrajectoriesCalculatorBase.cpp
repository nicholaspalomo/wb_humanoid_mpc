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

#include "humanoid_common_mpc/command/TargetTrajectoriesCalculatorBase.h"

#include <algorithm>  // For std::clamp
#include <cmath>
#include <functional>
#include <mutex>
#include <string>
#include <utility>

#include "absl/log/absl_check.h"

#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

namespace ocs2::humanoid {

TargetTrajectoriesCalculatorBase::TargetTrajectoriesCalculatorBase(const ReferenceSettings& referenceSettings,
                                                                   const vector_t& defaultJointState,
                                                                   const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                                   scalar_t mpcHorizon)
    : mpcRobotModelPtr_(mpcRobotModel.clone()), targetJointState_(defaultJointState), mpcHorizon_(mpcHorizon) {
  ABSL_CHECK_EQ(static_cast<size_t>(targetJointState_.size()), mpcRobotModel.getJointDim())
      << "TargetTrajectoriesCalculatorBase: the default joint state has one entry per MPC joint (defaultJointStateFromConfig())";
  applyCommandLimits(referenceSettings);
}

void TargetTrajectoriesCalculatorBase::applyCommandLimits(const ReferenceSettings& referenceSettings) {
  // LINT.IfChange(apply_command_limits)
  defaultBaseHeight_.store(referenceSettings.defaultBaseHeight);
  targetRotationVelocity_.store(referenceSettings.targetRotationVelocity);
  targetDisplacementVelocity_.store(referenceSettings.targetDisplacementVelocity);
  maxDisplacementVelocityX_.store(referenceSettings.maxDisplacementVelocityX);
  maxDisplacementVelocityY_.store(referenceSettings.maxDisplacementVelocityY);
  maxDeltaPelvisHeight_.store(referenceSettings.maxDeltaPelvisHeight);
  maxRotationVelocity_.store(referenceSettings.maxRotationVelocity);
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/config/reference/ReferenceFromConfig.cpp:hot_reference_file_fields)
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void TargetTrajectoriesCalculatorBase::reset() {
  filteredVelocityCommand_.setZero();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void TargetTrajectoriesCalculatorBase::setTerrainHeightSource(std::function<scalar_t()> terrainHeightSource) {
  std::lock_guard<std::mutex> lock(terrainHeightSourceMutex_);
  terrainHeightSource_ = std::move(terrainHeightSource);
}

scalar_t TargetTrajectoriesCalculatorBase::getTerrainHeight() const {
  std::lock_guard<std::mutex> lock(terrainHeightSourceMutex_);
  return terrainHeightSource_ ? terrainHeightSource_() : mpcRobotModelPtr_->modelSettings.terrainHeight;
}

scalar_t TargetTrajectoriesCalculatorBase::commandedBaseHeight(scalar_t commandedPelvisHeight) const {
  const scalar_t heightAboveGround = commandedPelvisHeight > kMinCommandedPelvisHeight ? commandedPelvisHeight : defaultBaseHeight_.load();
  return getTerrainHeight() + heightAboveGround;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector6_t TargetTrajectoriesCalculatorBase::getDeltaBaseTarget(const vector4_t& commadLinePoseTarget,
                                                               const vector6_t& currentPoseTarget) const {
  vector_t target(6);

  // X facing forward to the robot, Y to the left side in the baseFrame
  const scalar_t baseFrameDeltaX = commadLinePoseTarget(0);
  const scalar_t baseFrameDeltaY = commadLinePoseTarget(1);
  const scalar_t currentEulerZ = currentPoseTarget(3);

  const scalar_t globalFrameDeltaX = std::cos(currentEulerZ) * baseFrameDeltaX - std::sin(currentEulerZ) * baseFrameDeltaY;
  const scalar_t globalFrameDeltaY = std::sin(currentEulerZ) * baseFrameDeltaX + std::cos(currentEulerZ) * baseFrameDeltaY;

  // base p_x, p_y are relative to current state
  target(0) = currentPoseTarget(0) + globalFrameDeltaX;
  target(1) = currentPoseTarget(1) + globalFrameDeltaY;
  // base z relative to the default height above the ground
  const scalar_t maxDeltaPelvisHeight = maxDeltaPelvisHeight_;
  const scalar_t deltaPelvisHeight = std::clamp(commadLinePoseTarget(2), -maxDeltaPelvisHeight, maxDeltaPelvisHeight);
  target(2) = getTerrainHeight() + defaultBaseHeight_ + deltaPelvisHeight;
  // theta_z relative to current
  target(3) = currentPoseTarget(3) + commadLinePoseTarget(3) * M_PI / 180.0;
  target(4) = 0.0;
  target(5) = 0.0;

  return target;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector6_t TargetTrajectoriesCalculatorBase::getCurrentBasePoseTarget(const vector_t& state) const {
  vector_t currentPoseTarget = mpcRobotModelPtr_->getBasePose(state);
  // Zero out roll and pitch of the torso since target trajectories starts from current state
  currentPoseTarget(4) = 0.0;
  currentPoseTarget(5) = 0.0;

  return currentPoseTarget;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector4_t TargetTrajectoriesCalculatorBase::filterAndTransformVelCommandToLocal(const vector4_t& commandedVelLocal,
                                                                                scalar_t currentEulerZ,
                                                                                scalar_t filterAlpha) {
  filteredVelocityCommand_ = filteredVelocityCommand_ * filterAlpha + commandedVelLocal * (1 - filterAlpha);

  vector4_t globalTargetVel = filteredVelocityCommand_;

  globalTargetVel(0) = std::cos(currentEulerZ) * filteredVelocityCommand_[0] - std::sin(currentEulerZ) * filteredVelocityCommand_[1];
  globalTargetVel(1) = std::sin(currentEulerZ) * filteredVelocityCommand_[0] + std::cos(currentEulerZ) * filteredVelocityCommand_[1];

  return globalTargetVel;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector6_t TargetTrajectoriesCalculatorBase::integrateTargetBasePose(const vector6_t& currentPose,
                                                                    const vector3_t& averageVel,
                                                                    scalar_t baseHeight,
                                                                    scalar_t deltaT) const {
  vector6_t targetPose = currentPose;

  targetPose[0] += averageVel[0] * deltaT;
  targetPose[1] += averageVel[1] * deltaT;
  // A world height already, ground included (commandedBaseHeight()). The pelvis-height fallback is not applied to it a
  // second time, as it used to be: on ground below z = 0.1 m a correct world height would read as "no command".
  targetPose[2] = baseHeight;
  targetPose[3] += averageVel[2] * deltaT;
  targetPose[4] = 0.0;
  targetPose[5] = 0.0;
  return targetPose;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t TargetTrajectoriesCalculatorBase::estimateTimeToTarget(const vector_t& desiredBaseDisplacement) const {
  const scalar_t& dx = desiredBaseDisplacement(0);
  const scalar_t& dy = desiredBaseDisplacement(1);
  const scalar_t& dyaw = desiredBaseDisplacement(3);
  const scalar_t rotationTime = std::abs(dyaw) / targetRotationVelocity_;
  const scalar_t displacement = std::sqrt(dx * dx + dy * dy);
  const scalar_t displacementTime = displacement / targetDisplacementVelocity_;
  return std::max(rotationTime, displacementTime);
}

}  // namespace ocs2::humanoid
