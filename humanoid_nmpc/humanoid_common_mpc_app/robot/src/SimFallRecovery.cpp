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

#include "humanoid_common_mpc_app/robot/SimFallRecovery.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "humanoid_common_mpc/mrt/ControlMode.h"

namespace ocs2::humanoid {

SimFallRecovery::SimFallRecovery(const Config& config,
                                 const robot::mujoco_sim_interface::MujocoSimInterface& robotInterface,
                                 std::vector<size_t> restJointIndices,
                                 RealtimeEventLog* eventLog)
    : config_(config),
      restJointIndices_(std::move(restJointIndices)),
      eventLog_(eventLog),
      lastResetEpoch_(robotInterface.resetEpoch()),
      lastGantryLocked_(robotInterface.isGantryLocked()) {}

scalar_t SimFallRecovery::baseTiltAngle(const quaternion_t& baseRotationLocalToWorld) {
  // The base's own vertical, expressed in the world: the third column of its rotation matrix. Its angle to the world
  // vertical is the arccosine of that column's z component, which is heading independent, so a robot that has turned
  // on the spot reads zero tilt exactly like one that has not. The clamp keeps a matrix entry that rounds just past
  // one from producing a NaN, which would compare false against the threshold and silently disable the recovery.
  const matrix3_t baseRotation = baseRotationLocalToWorld.toRotationMatrix();
  return std::acos(std::clamp(baseRotation(2, 2), scalar_t(-1.0), scalar_t(1.0)));
}

bool SimFallRecovery::isAtRest(const robot::model::RobotState& robotState,
                               const std::vector<scalar_t>& nominalJointPositions,
                               const std::vector<size_t>& jointIndices,
                               const Config& config) {
  if (baseTiltAngle(robotState.getRootRotationLocalToWorldFrame()) > config.settleTilt) return false;
  if (robotState.getRootLinearVelocityInLocalFrame().norm() > config.settleLinearSpeed) return false;
  if (robotState.getRootAngularVelocityInLocalFrame().norm() > config.settleAngularSpeed) return false;
  for (size_t joint : jointIndices) {
    if (joint >= nominalJointPositions.size()) continue;
    scalar_t position = 0.0;
    try {
      position = robotState.getJointPosition(joint);
    } catch (const std::runtime_error&) {
      continue;  // a joint the simulator does not report has no say in whether the robot is at rest
    }
    if (std::abs(position - nominalJointPositions[joint]) > config.settleJointError) return false;
  }
  return true;
}

std::string SimFallRecovery::describeDiscontinuity(const Cycle& cycle) const {
  return discontinuityReason(cycle.cause, cycle.resetEpoch, cycle.tilt, config_.maxBaseTiltAngle);
}

void SimFallRecovery::report(
    RealtimeEventCode code, std::int32_t detail, absl::string_view text, double value0, double value1, double value2, std::uint64_t count) {
  if (eventLog_ != nullptr) {
    eventLog_->post(code, detail, text, value0, value1, value2, count);
  }
}

bool SimFallRecovery::restedFor(const robot::model::RobotState& robotState, const std::vector<scalar_t>& nominalJointPositions) {
  if (!isAtRest(robotState, nominalJointPositions, restJointIndices_, config_)) {
    restingSince_.reset();
    return false;
  }
  if (!restingSince_.has_value()) restingSince_ = robotState.getTime();
  return robotState.getTime() - *restingSince_ >= config_.settleHoldTime;
}

void SimFallRecovery::enterPhase(Phase phase, scalar_t time) {
  phase_ = phase;
  phaseStartTime_ = time;
  restingSince_.reset();
}

SimFallRecovery::Cycle SimFallRecovery::update(const robot::model::RobotState& robotState,
                                               const std::vector<scalar_t>& nominalJointPositions,
                                               robot::mujoco_sim_interface::MujocoSimInterface& robotInterface,
                                               std::string& currentModeName) {
  Cycle cycle;
  // The lock is read before the epoch. The simulator counts a reset in the epoch before it locks the gantry
  // (MujocoSimInterface::resetAndCatch()), so a reset that lands between the two reads is always seen as the reset it is:
  // read the other way round, it was seen as a lock without a fall, and as a second discontinuity on the next cycle.
  const bool gantryLocked = robotInterface.isGantryLocked();
  const uint64_t resetEpoch = robotInterface.resetEpoch();

  // One discontinuity per event, compared with the previous cycle. The simulator's own resets lock the gantry as well
  // as moving the epoch; they are counted once, as the reset they are.
  bool caughtAfterFall = false;
  if (resetEpoch != lastResetEpoch_) {
    cycle.discontinuity = true;
    cycle.cause = DiscontinuityCause::kSimulatorReset;
    cycle.resetEpoch = resetEpoch;
    caughtAfterFall = true;
  } else if (gantryLocked && !lastGantryLocked_) {
    cycle.discontinuity = true;
    cycle.cause = DiscontinuityCause::kGantryLocked;
  } else if (!gantryLocked && config_.maxBaseTiltAngle > 0.0) {
    const scalar_t tilt = baseTiltAngle(robotState.getRootRotationLocalToWorldFrame());
    if (tilt > config_.maxBaseTiltAngle) {
      robotInterface.lockGantry();
      cycle.discontinuity = true;
      cycle.cause = DiscontinuityCause::kTiltCaught;
      cycle.tilt = tilt;
      caughtAfterFall = true;
    }
  }
  if (!cycle.discontinuity && !gantryLocked && lastGantryLocked_) {
    cycle.gantryUnlocked = true;
    if (phase_ != Phase::kIdle) {
      report(RealtimeEventCode::kSettleEndedByUnlock, static_cast<std::int32_t>(phase_));
      enterPhase(Phase::kIdle, robotState.getTime());
    }
  }

  if (cycle.discontinuity) {
    // JOINT_PD is the only mode that is safe on the gantry: the whole-body MPC is overconstrained against a pinned base.
    if (robotInterface.isZeroTorqueMode()) robotInterface.enableTorques();
    currentModeName.assign(control_mode::kJointPd.data(), control_mode::kJointPd.size());
    cycle.modeChanged = true;
    refusalReported_ = false;
    if (caughtAfterFall && config_.catchLift > 0.0) {
      // A catch during a sequence (the simulator reset the robot while it hung lifted) keeps the height the sequence
      // returns to: the gantry is not at the standing height then.
      if (phase_ == Phase::kIdle) standingGantryHeight_ = robotInterface.getGantryHeight();
      rampStartHeight_ = std::max(standingGantryHeight_, robotInterface.getGantryHeight());
      enterPhase(Phase::kLifting, robotState.getTime());
      report(RealtimeEventCode::kCaughtAndSettling, static_cast<std::int32_t>(cycle.cause), /*text=*/{}, cycle.tilt,
             config_.maxBaseTiltAngle, config_.catchLift, cycle.resetEpoch);
    } else {
      // A new discontinuity ends a sequence that was running: the robot is where the new event left it.
      if (!caughtAfterFall) enterPhase(Phase::kIdle, robotState.getTime());
      report(RealtimeEventCode::kDiscontinuity, static_cast<std::int32_t>(cycle.cause), /*text=*/{}, cycle.tilt, config_.maxBaseTiltAngle,
             config_.catchLift, cycle.resetEpoch);
    }
  }

  if (phase_ != Phase::kIdle) {
    stepSettleSequence(robotState, nominalJointPositions, robotInterface);
    if (phase_ != Phase::kIdle && control_mode::isMpc(currentModeName)) {
      if (!refusalReported_) {
        report(RealtimeEventCode::kMpcModeRefusedWhileSettling, static_cast<std::int32_t>(phase_), currentModeName);
        refusalReported_ = true;
      }
      currentModeName.assign(control_mode::kJointPd.data(), control_mode::kJointPd.size());
      cycle.modeChanged = true;
    }
  }

  lastResetEpoch_ = resetEpoch;
  lastGantryLocked_ = robotInterface.isGantryLocked();
  return cycle;
}

void SimFallRecovery::stepSettleSequence(const robot::model::RobotState& robotState,
                                         const std::vector<scalar_t>& nominalJointPositions,
                                         robot::mujoco_sim_interface::MujocoSimInterface& robotInterface) {
  const scalar_t time = robotState.getTime();
  const scalar_t elapsed = std::max(scalar_t(0.0), time - phaseStartTime_);
  const scalar_t liftedHeight = standingGantryHeight_ + config_.catchLift;
  const bool timedOut = elapsed > config_.settleTimeout;
  switch (phase_) {
    case Phase::kLifting: {
      const scalar_t height = std::min(liftedHeight, rampStartHeight_ + config_.gantryHeightRate * elapsed);
      robotInterface.setGantryHeight(height);
      if (height >= liftedHeight) enterPhase(Phase::kSettlingLifted, time);
      break;
    }
    case Phase::kSettlingLifted:
      robotInterface.setGantryHeight(liftedHeight);
      if (restedFor(robotState, nominalJointPositions) || timedOut) {
        if (timedOut) {
          report(RealtimeEventCode::kLiftedSettleTimedOut, /*detail=*/0, /*text=*/{}, config_.settleTimeout);
        }
        rampStartHeight_ = liftedHeight;
        enterPhase(Phase::kLowering, time);
      }
      break;
    case Phase::kLowering: {
      const scalar_t height = std::max(standingGantryHeight_, rampStartHeight_ - config_.gantryHeightRate * elapsed);
      robotInterface.setGantryHeight(height);
      if (height <= standingGantryHeight_) enterPhase(Phase::kSettlingOnFeet, time);
      break;
    }
    case Phase::kSettlingOnFeet:
      robotInterface.setGantryHeight(standingGantryHeight_);
      if (restedFor(robotState, nominalJointPositions) || timedOut) {
        if (timedOut) {
          report(RealtimeEventCode::kOnFeetSettleTimedOut, /*detail=*/0, /*text=*/{}, config_.settleTimeout);
        } else {
          report(RealtimeEventCode::kSettled);
        }
        enterPhase(Phase::kIdle, time);
      }
      break;
    case Phase::kIdle:
    default:
      break;
  }
}

}  // namespace ocs2::humanoid
