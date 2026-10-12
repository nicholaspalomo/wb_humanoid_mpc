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

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"

#include "humanoid_common_mpc/command/BasePlanarMotion.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/reference/GaitFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_common_mpc/gait/GaitScheduleUpdater.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/reference_manager/StrideMotionHistory.h"
#include "humanoid_mpc_config/gait_file.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.h"

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
absl::StatusOr<std::unique_ptr<ProceduralMpcMotionManager>> ProceduralMpcMotionManager::Create(
    const mpc_config::GaitFile& gaitFile,
    const ReferenceSettings& referenceSettings,
    std::shared_ptr<SwitchedModelReferenceManager> switchedModelReferenceManagerPtr,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    VelocityTargetToTargetTrajectories velocityTargetToTargetTrajectories) {
  // The constructor is private, so std::make_unique cannot reach it.
  std::unique_ptr<ProceduralMpcMotionManager> manager = absl::WrapUnique(new ProceduralMpcMotionManager(
      std::move(switchedModelReferenceManagerPtr), mpcRobotModel, std::move(velocityTargetToTargetTrajectories)));
  RETURN_IF_ERROR(manager->applyCommandLimits(referenceSettings));
  ASSIGN_OR_RETURN(manager->gaitMap_, gaitMapFromConfig(gaitFile));
  RETURN_IF_ERROR(checkEveryGaitIsLoaded(manager->gaitMap_, manager->gaitModeStates_, "(gait_list and gaits)"));
  // The measured motion is averaged over the cycle of the gait the schedule runs, so the history keeps the longest
  // cycle of the gaits the command selects. gaitMapFromConfig() refuses a template whose switching times do not increase.
  scalar_t longestCycle = 0.0;
  for (const GaitModeStateConfig& gaitModeState : manager->gaitModeStates_) {
    longestCycle = std::max(longestCycle, manager->cycleDurationOf(gaitModeState.gaitCommand));
  }
  manager->strideMotion_ = StrideMotionHistory(longestCycle);
  manager->reset();
  return manager;
}

absl::StatusOr<std::unique_ptr<ProceduralMpcMotionManager>> ProceduralMpcMotionManager::Create(
    const std::string& gaitFile,
    const std::string& referenceFile,
    std::shared_ptr<SwitchedModelReferenceManager> switchedModelReferenceManagerPtr,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    VelocityTargetToTargetTrajectories velocityTargetToTargetTrajectories) {
  ASSIGN_OR_RETURN(const mpc_config::GaitFile gait, loadGaitFile(gaitFile));
  ASSIGN_OR_RETURN(const mpc_config::ReferenceFile reference, loadReferenceFile(referenceFile));
  absl::StatusOr<ReferenceSettings> referenceSettings = referenceSettingsFromConfig(reference);
  if (!referenceSettings.ok()) {
    return withConfigFile(referenceSettings.status(), referenceFile);
  }
  absl::StatusOr<std::unique_ptr<ProceduralMpcMotionManager>> manager = Create(
      gait, *referenceSettings, std::move(switchedModelReferenceManagerPtr), mpcRobotModel, std::move(velocityTargetToTargetTrajectories));
  if (!manager.ok()) {
    return withConfigFile(manager.status(), gaitFile);
  }
  return manager;
}

ProceduralMpcMotionManager::ProceduralMpcMotionManager(std::shared_ptr<SwitchedModelReferenceManager> switchedModelReferenceManagerPtr,
                                                       const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                       VelocityTargetToTargetTrajectories velocityTargetToTargetTrajectories)
    : switchedModelReferenceManagerPtr_(std::move(switchedModelReferenceManagerPtr)),
      gaitSchedulePtr_(switchedModelReferenceManagerPtr_->getGaitSchedule()),
      mpcRobotModelPtr_(&mpcRobotModel),
      velocityTargetToTargetTrajectoriesFun_(std::move(velocityTargetToTargetTrajectories)) {}

absl::Status ProceduralMpcMotionManager::checkEveryGaitIsLoaded(const std::map<std::string, ModeSequenceTemplate>& gaitMap,
                                                                absl::Span<const GaitModeStateConfig> gaitModeStates,
                                                                absl::string_view gaitFile) {
  // Every gait the velocity command can select has to be in the file: preSolverRun() switches to it by name.
  for (const GaitModeStateConfig& gaitModeState : gaitModeStates) {
    if (!gaitMap.contains(gaitModeState.gaitCommand)) {
      return absl::InvalidArgumentError(absl::StrCat("[ProceduralMpcMotionManager] the gait file ", gaitFile, " has no gait '",
                                                     gaitModeState.gaitCommand, "', which the walking velocity command selects."));
    }
  }
  return absl::OkStatus();
}

scalar_t ProceduralMpcMotionManager::cycleDurationOf(const std::string& gaitCommand) const {
  // Create() refused a gait file without every gait of gaitModeStates_, the only names this is asked for.
  const std::map<std::string, ModeSequenceTemplate>::const_iterator gait = gaitMap_.find(gaitCommand);
  ABSL_CHECK(gait != gaitMap_.end()) << "no gait " << gaitCommand;
  const std::vector<scalar_t>& switchingTimes = gait->second.switchingTimes;
  return switchingTimes.back() - switchingTimes.front();
}

scalar_t ProceduralMpcMotionManager::strideAt(scalar_t time) const {
  // The last gait started at or before `time`; the first one runs from the lowest time on.
  for (std::vector<ScheduledGait>::const_reverse_iterator gait = scheduledGaits_.rbegin(); gait != scheduledGaits_.rend(); ++gait) {
    if (gait->startTime <= time) return gait->cycleDuration;
  }
  return scheduledGaits_.front().cycleDuration;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::Status ProceduralMpcMotionManager::setVelocityCommandAccelerationLimits(scalar_t maxLinearAcceleration,
                                                                              scalar_t maxAngularAcceleration) {
  if (std::isnan(maxLinearAcceleration) || std::isnan(maxAngularAcceleration)) {
    return absl::InvalidArgumentError("[ProceduralMpcMotionManager] the velocity command acceleration limits must be numbers");
  }
  maxLinearAcceleration_ = maxLinearAcceleration;
  maxAngularAcceleration_ = maxAngularAcceleration;
  return absl::OkStatus();
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

absl::Status ProceduralMpcMotionManager::applyCommandLimits(const ReferenceSettings& settings) {
  // LINT.IfChange(apply_command_limits)
  maxDisplacementVelocityX_.store(settings.maxDisplacementVelocityX);
  maxDisplacementVelocityY_.store(settings.maxDisplacementVelocityY);
  maxDeltaPelvisHeight_.store(settings.maxDeltaPelvisHeight);
  maxRotationVelocity_.store(settings.maxRotationVelocity);
  RETURN_IF_ERROR(setVelocityCommandAccelerationLimits(settings.maxLinearAcceleration, settings.maxAngularAcceleration));
  return velocityCommandFilter_.setBreakFrequency(settings.velocityCommandFilterBreakFrequency);
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/config/reference/ReferenceFromConfig.cpp:hot_reference_file_fields)
}

void ProceduralMpcMotionManager::reset() {
  currentGaitMode_ = 0;
  currentGaitCommand_ = gaitModeStates_[currentGaitMode_].gaitCommand;
  lastGaitCommand_ = currentGaitCommand_;
  lastGaitChangeTime_ = std::numeric_limits<scalar_t>::lowest();
  rampedVelocityCommand_.setZero();
  lastRampTime_ = 0.0;
  rampInitialized_ = false;
  // The filter's output and clock are state of the previous run; its break frequency is configuration and stays.
  velocityCommandFilter_.reset();
  // The measured motion starts again, and the schedule runs the reference file's initial STANCE again (GaitSchedule::reset()).
  strideMotion_.reset();
  scheduledGaits_.assign(1, ScheduledGait{.startTime = std::numeric_limits<scalar_t>::lowest(),
                                          .cycleDuration = cycleDurationOf(gaitModeStates_[0].gaitCommand)});
  hasObservation_ = false;
  measuredMotion_ = BasePlanarMotion{};
  measuredMotionWindow_ = 0.0;
  measuredOverTheWholeWindow_ = false;
  if (resetHook_) resetHook_();
}

void ProceduralMpcMotionManager::observe(const SystemObservation& observation) {
  hasObservation_ = true;
  observedTime_ = observation.time;
  observedState_ = observation.state;
  observedInput_ = observation.input;
}

void ProceduralMpcMotionManager::setAndScaleVelocityCommand(const WalkingVelocityCommand& rawVelocityCommand) {
  // The limits are atomics, so the scaling needs no lock; only the command the solver thread reads does.
  const WalkingVelocityCommand scaledCommand = scaleWalkingVelocityCommand(rawVelocityCommand);
  absl::MutexLock lock(velocityCommandMutex_);
  velocityCommand_ = scaledCommand;
}

WalkingVelocityCommand ProceduralMpcMotionManager::getScaledWalkingVelocityCommand() {
  absl::MutexLock lock(velocityCommandMutex_);
  return velocityCommand_;
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
                                                        const BasePlanarMotion* absl_nullable strideMeanMotion,
                                                        const GaitModeStateConfig& cfg) {
  const bool fasterGaitRequested = (std::abs(velCommandVec(0)) > cfg.maxLinVelCmd || std::abs(velCommandVec(1)) > cfg.maxLinVelCmd ||
                                    std::abs(velCommandVec(3)) > cfg.maxAngVelCmd);
  const scalar_t linearThreshold = cfg.maxLinVelCmd - cfg.linVelErrorThresh;
  const scalar_t angularThreshold = cfg.maxAngVelCmd - cfg.angVelErrorThresh;
  // Without a measured window the switch reads no motion: it goes only where every motion would pass, i.e. where a
  // threshold is below zero (|m| > c holds for every m exactly when c < 0).
  if (strideMeanMotion == nullptr) return fasterGaitRequested && (linearThreshold < 0.0 || angularThreshold < 0.0);
  const vector2_t& velocity = strideMeanMotion->velocityInHeadingFrame;
  const bool withinMaxSpeedErrorThreshold = (std::abs(velocity(0)) > linearThreshold || std::abs(velocity(1)) > linearThreshold ||
                                             std::abs(strideMeanMotion->headingRate) > angularThreshold);
  return fasterGaitRequested && withinMaxSpeedErrorThreshold;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

bool ProceduralMpcMotionManager::transitionToSlowerGait(const vector4_t& velCommandVec,
                                                        const BasePlanarMotion* absl_nullable strideMeanMotion,
                                                        const GaitModeStateConfig& cfg) {
  const bool slowerGaitRequested = (std::abs(velCommandVec(0)) < cfg.minLinVelCmd && std::abs(velCommandVec(1)) < cfg.minLinVelCmd &&
                                    std::abs(velCommandVec(3)) < cfg.minAngVelCmd);
  // Without a measured window no motion is known to be slow enough: no c has |m| < c for every m.
  if (strideMeanMotion == nullptr) return false;
  // The measured motion, its turn included: the command's own yaw rate stood here, which slowerGaitRequested has
  // already bounded more tightly, so a robot still turning fast was let down into the slower gait (design defect D2);
  // and the velocity is the stride mean, not the instantaneous one, whose sway let a robot still walking down whenever
  // it dipped below the thresholds and held a stopped one up while it swayed past them (design README section 2.9).
  const vector2_t& velocity = strideMeanMotion->velocityInHeadingFrame;
  const scalar_t linearThreshold = cfg.minLinVelCmd + cfg.linVelErrorThresh;
  const bool baseSpeedSlowEnough = (std::abs(velocity(0)) < linearThreshold && std::abs(velocity(1)) < linearThreshold &&
                                    std::abs(strideMeanMotion->headingRate) < cfg.minAngVelCmd + cfg.angVelErrorThresh);
  return slowerGaitRequested && baseSpeedSlowEnough;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

BasePlanarMotion ProceduralMpcMotionManager::instantaneousMotionOf(scalar_t initTime, const vector_t& initState) const {
  const MpcRobotModelBase<scalar_t>& model = *mpcRobotModelPtr_;
  // The observation the solve starts from, exactly: the runner passes its time and state to the solver unchanged.
  const bool observed = hasObservation_ && observedTime_ == initTime && observedState_.size() == initState.size() &&
                        observedState_ == initState && static_cast<size_t>(observedInput_.size()) == model.getInputDim();
  if (!observed) {
    LOG_EVERY_N_SEC(WARNING, 5.0) << "[ProceduralMpcMotionManager] No observation of the solve at t = " << initTime
                                  << " was handed over (observe()): its measured motion starts at rest.";
    return BasePlanarMotion{};
  }
  const vector_t jointVelocities = model.getJointVelocities(initState, observedInput_);
  return instantaneousPlanarMotion(model.getBaseOrientation(initState), model.getBaseTwistInWorldFrame(initState, jointVelocities));
}

BasePlanarMotion ProceduralMpcMotionManager::updateMeasuredMotion(scalar_t initTime, const vector_t& initState) {
  const MpcRobotModelBase<scalar_t>& model = *mpcRobotModelPtr_;
  strideMotion_.addSample(initTime, vector2_t(model.getBasePosition(initState).head<2>()), model.getBaseHeading(initState),
                          [this, initTime, &initState]() { return instantaneousMotionOf(initTime, initState); });
  // The gaits that no longer run: every one before the one running now.
  while (scheduledGaits_.size() >= 2 && scheduledGaits_[1].startTime <= initTime) scheduledGaits_.erase(scheduledGaits_.begin());
  measuredMotionWindow_ = strideAt(initTime);
  measuredMotion_ = strideMotion_.mean(measuredMotionWindow_);
  // No seed and no stall in the window (the class comment).
  measuredOverTheWholeWindow_ = strideMotion_.isMeasuredOver(measuredMotionWindow_);
  return measuredMotion_;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void ProceduralMpcMotionManager::preSolverRun(scalar_t initTime,
                                              scalar_t finalTime,
                                              const vector_t& initState,
                                              const ReferenceManagerInterface& /*referenceManager*/) {
  WalkingVelocityCommand incommingVelCommand = getScaledWalkingVelocityCommand();
  // The command filter runs on the solver time (off unless reference.textproto sets its break frequency).
  vector4_t filteredVelCommand = velocityCommandFilter_.update(initTime, incommingVelCommand.toVector());
  // Acceleration limit on the reference (off by default). The first solve, the first after a reset, and a solve after
  // time ran backwards start the ramp at the filtered command itself.
  if (!rampInitialized_ || initTime < lastRampTime_) {
    rampedVelocityCommand_ = filteredVelCommand;
    rampInitialized_ = true;
  } else {
    rampedVelocityCommand_ = rateLimitVelocityCommand(filteredVelCommand, rampedVelocityCommand_, initTime - lastRampTime_,
                                                      maxLinearAcceleration_, maxAngularAcceleration_);
  }
  lastRampTime_ = initTime;
  filteredVelCommand = rampedVelocityCommand_;

  // The measured motion over the last stride, which both the targets and the gait switches start from.
  const BasePlanarMotion measuredMotion = updateMeasuredMotion(initTime, initState);

  // Update TargetTrajectories
  TargetTrajectories targetTrajectories =
      velocityTargetToTargetTrajectoriesFun_(filteredVelCommand, initTime, finalTime, initState, measuredMotion);
  switchedModelReferenceManagerPtr_->setTargetTrajectories(targetTrajectories);

  // The thresholds of the gait the manager is in. This was a function-static copy, initialized once for the whole
  // process and shared by every instance: a second manager, or this one after a reset of currentGaitMode_, went on
  // switching with the thresholds of whatever gait the first one had last reached.
  const GaitModeStateConfig& currentCfg = gaitModeStates_[currentGaitMode_];

  // Do not change the gait pattern for kGaitChangeHoldOff after the last change. A last change later than now was made on
  // a clock that has since run backwards; it must not hold the gait until the clock is back there.
  if (initTime < lastGaitChangeTime_) lastGaitChangeTime_ = initTime - kGaitChangeHoldOff;
  if (initTime > lastGaitChangeTime_ + kGaitChangeHoldOff) {
    // The switches read the measured motion once its window is measured throughout (the class comment).
    const BasePlanarMotion* absl_nullable strideMeanMotion = measuredOverTheWholeWindow_ ? &measuredMotion : nullptr;
    if (currentGaitMode_ + 1 < gaitModeStates_.size() && transitionToFasterGait(filteredVelCommand, strideMeanMotion, currentCfg)) {
      LOG(INFO) << "filteredVelCommand: " << filteredVelCommand.transpose();
      LOG(INFO) << "Linear limits: " << currentCfg.minLinVelCmd << ", " << currentCfg.maxLinVelCmd;
      ++currentGaitMode_;
      currentGaitCommand_ = gaitModeStates_[currentGaitMode_].gaitCommand;
      LOG(INFO) << "ProceduralMpcMotionManager: Increasing to gait:" << currentGaitCommand_;
      lastGaitChangeTime_ = initTime;
    } else if (currentGaitMode_ > 0 && transitionToSlowerGait(filteredVelCommand, strideMeanMotion, currentCfg)) {
      LOG(INFO) << "filteredVelCommand: " << filteredVelCommand.transpose();
      LOG(INFO) << "Linear limits: " << currentCfg.minLinVelCmd << ", " << currentCfg.maxLinVelCmd;
      --currentGaitMode_;
      currentGaitCommand_ = gaitModeStates_[currentGaitMode_].gaitCommand;
      LOG(INFO) << "ProceduralMpcMotionManager: Decreasing to gait:" << currentGaitCommand_;
      lastGaitChangeTime_ = initTime;
    }
  }

  if (currentGaitCommand_ != lastGaitCommand_) {
    // loadGaits() refused a gait file without every gait of gaitModeStates_, which currentGaitCommand_ is one of.
    const std::map<std::string, ModeSequenceTemplate>::const_iterator gait = gaitMap_.find(currentGaitCommand_);
    ABSL_CHECK(gait != gaitMap_.end()) << "no gait " << currentGaitCommand_;

    const scalar_t gaitStart = updateGaitSchedule(*gaitSchedulePtr_, gait->second, initTime, finalTime);
    // The schedule runs the new gait from gaitStart on, in place of whatever it had from there.
    while (scheduledGaits_.size() >= 2 && scheduledGaits_.back().startTime >= gaitStart) scheduledGaits_.pop_back();
    scheduledGaits_.push_back(ScheduledGait{.startTime = gaitStart, .cycleDuration = cycleDurationOf(currentGaitCommand_)});
    lastGaitCommand_ = currentGaitCommand_;
  }
}

}  // namespace ocs2::humanoid
