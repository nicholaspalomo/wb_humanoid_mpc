/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <pinocchio/algorithm/center-of-mass.hpp>

#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

namespace ocs2::humanoid {

namespace {
constexpr scalar_t kLongAgo = 10.0;  // [s] elapsed phase time reported when the applied schedule has no earlier event
}  // namespace

ContactPlanningReferenceManager::ContactPlanningReferenceManager(std::shared_ptr<GaitSchedule> gaitSchedulePtr,
                                                                 std::shared_ptr<SwingTrajectoryPlanner> swingTrajectoryPtr,
                                                                 const PinocchioInterface& pinocchioInterface,
                                                                 const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                                 ContactPlanningConfig config)
    : SwitchedModelReferenceManager(std::move(gaitSchedulePtr), std::move(swingTrajectoryPtr), pinocchioInterface, mpcRobotModel),
      config_(std::move(config)) {
  config_.validate();
}

void ContactPlanningReferenceManager::setContactPlan(const ContactPlan& plan) {
  std::lock_guard<std::mutex> lock(planMutex_);
  pendingPlan_ = plan;
}

void ContactPlanningReferenceManager::setConfig(const ContactPlanningConfig& config) {
  config.validate();
  std::lock_guard<std::mutex> lock(configMutex_);
  config_ = config;
}

ContactPlanningConfig ContactPlanningReferenceManager::getConfig() const {
  std::lock_guard<std::mutex> lock(configMutex_);
  return config_;
}

feet_array_t<vector3_t> ContactPlanningReferenceManager::computeFootPositions(const vector_t& state) {
  const vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  const std::vector<vector3_t> positions = computeContactPositions<scalar_t>(q, pinocchioInterface_, *mpcRobotModelPtr_);
  feet_array_t<vector3_t> feet;
  for (size_t i = 0; i < N_CONTACTS; ++i) {
    feet[i] = positions[i];
  }
  return feet;
}

void ContactPlanningReferenceManager::updateFootBookkeeping(scalar_t initTime, const vector_t& initState) {
  footPositions_ = computeFootPositions(initState);
  const contact_flag_t contacts = modeNumber2StanceLeg(appliedSchedule_.modeAtTime(initTime));
  for (size_t i = 0; i < N_CONTACTS; ++i) {
    if (contacts[i] || !footBookkeepingInitialized_) {
      liftOffPositions_[i] = footPositions_[i];
    }
  }
  footBookkeepingInitialized_ = true;
}

void ContactPlanningReferenceManager::modifyReferences(scalar_t initTime,
                                                       scalar_t finalTime,
                                                       const vector_t& initState,
                                                       size_t initMode,
                                                       TargetTrajectories& targetTrajectories,
                                                       ModeSchedule& modeSchedule) {
  const scalar_t timeHorizon = finalTime - initTime;
  const scalar_t lowerBoundTime = initTime - timeHorizon;
  const scalar_t upperBoundTime = finalTime + timeHorizon;
  const ContactPlanningConfig config = getConfig();

  {
    std::lock_guard<std::mutex> lock(planMutex_);
    if (pendingPlan_.has_value()) {
      activePlan_ = std::move(*pendingPlan_);
      pendingPlan_.reset();
    }
  }

  const scalar_t commitTime = initTime + config.commitTime;
  const bool planUsable = hasActivePlan() && activePlan_->endTime() > commitTime + config.dt && activePlan_->startTime <= commitTime;

  ModeSchedule schedule;
  if (planUsable) {
    const ModeSchedule planSchedule = activePlan_->toModeSchedule();
    const ModeSchedule& applied =
        hasAppliedSchedule_ ? appliedSchedule_ : gaitSchedulePtr_->getModeSchedule(lowerBoundTime, upperBoundTime);
    schedule = mergeModeSchedules(applied, planSchedule, commitTime, lowerBoundTime, upperBoundTime);
  } else if (hasAppliedSchedule_ && activePlan_.has_value()) {
    // The plan ran out (planner stalled): keep executing the applied schedule, which ends in STANCE.
    schedule = mergeModeSchedules(appliedSchedule_, ModeSchedule({}, {ModeNumber::STANCE}), upperBoundTime, lowerBoundTime, upperBoundTime);
  } else {
    schedule = gaitSchedulePtr_->getModeSchedule(lowerBoundTime, upperBoundTime);
  }

  const scalar_t terrainHeight = adaptToCurrentGroundHeight(targetTrajectories, initState, initMode);
  swingTrajectoryPtr_->update(schedule, terrainHeight);
  modeSchedule = schedule;
  modeSchedule_ = schedule;
  appliedSchedule_ = schedule;
  hasAppliedSchedule_ = true;
  updateFootBookkeeping(initTime, initState);
}

std::optional<std::pair<scalar_t, scalar_t>> ContactPlanningReferenceManager::swingPhase(size_t contactIndex, scalar_t time) const {
  const auto& eventTimes = appliedSchedule_.eventTimes;
  const auto& modeSequence = appliedSchedule_.modeSequence;
  if (modeSequence.empty()) return std::nullopt;
  const auto inContact = [&](size_t modeIndex) { return modeNumber2StanceLeg(modeSequence[modeIndex])[contactIndex]; };
  // Mode index containing `time` (events at exactly `time` count as already passed, like ModeSchedule::modeAtTime).
  size_t index = static_cast<size_t>(std::upper_bound(eventTimes.begin(), eventTimes.end(), time) - eventTimes.begin());
  if (index >= modeSequence.size()) index = modeSequence.size() - 1;
  if (inContact(index)) return std::nullopt;
  size_t first = index;
  while (first > 0 && !inContact(first - 1)) --first;
  size_t last = index;
  while (last + 1 < modeSequence.size() && !inContact(last + 1)) ++last;
  if (first == 0 || last + 1 >= modeSequence.size()) return std::nullopt;  // no lift-off or touch-down event
  return std::make_pair(eventTimes[first - 1], eventTimes[last]);
}

std::optional<SwingFootReference> ContactPlanningReferenceManager::getSwingFootReference(size_t contactIndex, scalar_t time) const {
  if (!hasActivePlan() || !footBookkeepingInitialized_) return std::nullopt;
  const auto phase = swingPhase(contactIndex, time);
  if (!phase.has_value()) return std::nullopt;
  const auto [liftOffTime, touchDownTime] = *phase;
  const scalar_t duration = touchDownTime - liftOffTime;
  if (duration <= 1e-6) return std::nullopt;
  const std::optional<vector2_t> landing = activePlan_->footholdAtTime(contactIndex, touchDownTime);
  if (!landing.has_value()) return std::nullopt;

  const vector2_t start = liftOffPositions_[contactIndex].head<2>();
  const scalar_t tau = std::clamp((time - liftOffTime) / duration, 0.0, 1.0);
  const scalar_t blend = tau * tau * (3.0 - 2.0 * tau);
  const scalar_t blendRate = 6.0 * tau * (1.0 - tau) / duration;
  const vector2_t delta = *landing - start;

  SwingFootReference reference;
  reference.position.head<2>() = start + blend * delta;
  reference.position(2) = swingTrajectoryPtr_->getZpositionConstraint(contactIndex, time);
  reference.linearVelocity.head<2>() = blendRate * delta;
  reference.linearVelocity(2) = swingTrajectoryPtr_->getZvelocityConstraint(contactIndex, time);
  return reference;
}

ContactPlannerInput ContactPlanningReferenceManager::makePlannerInput(scalar_t initTime,
                                                                      const vector_t& initState,
                                                                      const vector2_t& velocityCommand) {
  ContactPlannerInput input;
  input.time = initTime;
  input.velocityCommand = velocityCommand;

  const auto& model = pinocchioInterface_.getModel();
  auto& data = pinocchioInterface_.getData();
  const vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(initState);
  pinocchio::centerOfMass(model, data, q, false);
  input.comPosition = data.com[0].head<2>();
  input.comVelocity = mpcRobotModelPtr_->getBaseComLinearVelocity(initState).head<2>();
  input.yaw = mpcRobotModelPtr_->getBaseOrientationEulerZYX(initState)(0);

  const feet_array_t<vector3_t> feet = computeFootPositions(initState);
  for (size_t i = 0; i < N_CONTACTS; ++i) {
    input.footPositions[i] = feet[i].head<2>();
  }

  const ModeSchedule& schedule = hasAppliedSchedule_ ? appliedSchedule_ : this->getModeSchedule();
  input.contacts = modeNumber2StanceLeg(schedule.modeAtTime(initTime));
  for (size_t i = 0; i < N_CONTACTS; ++i) {
    // Walk back through the events while the foot keeps the same contact state.
    scalar_t phaseStart = initTime - kLongAgo;
    const auto& eventTimes = schedule.eventTimes;
    const auto& modeSequence = schedule.modeSequence;
    size_t index = static_cast<size_t>(std::upper_bound(eventTimes.begin(), eventTimes.end(), initTime) - eventTimes.begin());
    if (index >= modeSequence.size()) index = modeSequence.size() - 1;
    while (index > 0 && modeNumber2StanceLeg(modeSequence[index - 1])[i] == input.contacts[i]) {
      --index;
    }
    if (index > 0) {
      phaseStart = eventTimes[index - 1];
    }
    input.phaseElapsedTime[i] = std::max(0.0, initTime - phaseStart);
  }

  // Most recently swung foot: the foot whose contact flag most recently went from contact to swing.
  input.lastSwungFoot = -1;
  {
    const auto& eventTimes = schedule.eventTimes;
    const auto& modeSequence = schedule.modeSequence;
    size_t index = static_cast<size_t>(std::upper_bound(eventTimes.begin(), eventTimes.end(), initTime) - eventTimes.begin());
    if (index >= modeSequence.size()) index = modeSequence.size() - 1;
    for (size_t i = index; i > 0 && input.lastSwungFoot < 0; --i) {
      const contact_flag_t before = modeNumber2StanceLeg(modeSequence[i - 1]);
      const contact_flag_t after = modeNumber2StanceLeg(modeSequence[i]);
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        if (before[foot] && !after[foot]) input.lastSwungFoot = static_cast<int>(foot);
      }
    }
  }

  const ContactPlanningConfig config = getConfig();
  const int numCommitted = config.commitNodes();
  input.committedContacts.clear();
  for (int k = 0; k < numCommitted; ++k) {
    const scalar_t t = initTime + (static_cast<scalar_t>(k) + 0.5) * config.dt;
    input.committedContacts.push_back(modeNumber2StanceLeg(schedule.modeAtTime(t)));
  }
  return input;
}

}  // namespace ocs2::humanoid
