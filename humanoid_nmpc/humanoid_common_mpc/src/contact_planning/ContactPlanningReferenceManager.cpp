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
#include <limits>

#include <pinocchio/algorithm/center-of-mass.hpp>

#include <ocs2_core/misc/LinearInterpolation.h>

#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

namespace ocs2::humanoid {

namespace {
constexpr scalar_t kLongAgo = 10.0;                  // [s] elapsed phase time reported when the applied schedule has no earlier event
constexpr scalar_t kShiftLogAge = 2.0;               // [s] schedule shifts older than this cannot concern a pending plan any more
constexpr scalar_t kSameSwingTolerance = 1e-6;       // [s] lift-off times closer than this identify the same swing
constexpr scalar_t kPredictionTimeTolerance = 1e-6;  // [s] slack when checking that a prediction covers the solver time
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

std::pair<vector2_t, vector2_t> ContactPlanningReferenceManager::computeComState(const vector_t& state) {
  const auto& model = pinocchioInterface_.getModel();
  auto& data = pinocchioInterface_.getData();
  const vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  pinocchio::centerOfMass(model, data, q, false);
  const vector2_t com = data.com[0].head<2>();
  const vector2_t comVelocity = mpcRobotModelPtr_->getBaseComLinearVelocity(state).head<2>();
  return {com, comVelocity};
}

void ContactPlanningReferenceManager::updateFootBookkeeping(scalar_t initTime, const vector_t& initState) {
  footPositions_ = computeFootPositions(initState);
  const contact_flag_t contacts = contactFlagsAtTime(appliedSchedule_, initTime);
  for (size_t i = 0; i < N_CONTACTS; ++i) {
    if (contacts[i] || !footBookkeepingInitialized_) {
      liftOffPositions_[i] = footPositions_[i];
    }
  }
  footBookkeepingInitialized_ = true;
}

void ContactPlanningReferenceManager::activatePendingPlan(scalar_t initTime) {
  bool activated = false;
  {
    std::lock_guard<std::mutex> lock(planMutex_);
    if (pendingPlan_.has_value()) {
      activePlan_ = std::move(*pendingPlan_);
      pendingPlan_.reset();
      activated = true;
    }
  }
  // The planner snapshot is taken after this manager ran in the same cycle, so the plan has seen every shift logged at or
  // before its start time; shifts logged later re-timed the schedule it was built on and are applied to the plan too.
  if (activated && activePlan_->valid) {
    for (const auto& [time, shift] : scheduleShiftLog_) {
      if (time > activePlan_->startTime) activePlan_->shiftInTime(shift);
    }
  }
  while (!scheduleShiftLog_.empty() && scheduleShiftLog_.front().first < initTime - kShiftLogAge) {
    scheduleShiftLog_.pop_front();
  }
}

void ContactPlanningReferenceManager::setPredictedTrajectory(const scalar_array_t& times, const vector_array_t& states) {
  std::lock_guard<std::mutex> lock(predictionMutex_);
  if (times.size() != states.size()) {
    predictedTimes_.clear();
    predictedStates_.clear();
    return;
  }
  predictedTimes_ = times;
  predictedStates_ = states;
}

void ContactPlanningReferenceManager::updatePredictedComState(scalar_t initTime) {
  hasPredictedComState_ = false;
  vector_t predictedState;
  {
    std::lock_guard<std::mutex> lock(predictionMutex_);
    if (predictedTimes_.size() < 2 || initTime < predictedTimes_.front() - kPredictionTimeTolerance ||
        initTime > predictedTimes_.back() + kPredictionTimeTolerance) {
      return;
    }
    predictedState = LinearInterpolation::interpolate(initTime, predictedTimes_, predictedStates_);
  }
  // A diverged solve hands over a non-finite trajectory; measuring against it would poison the foot reference.
  if (predictedState.size() != static_cast<Eigen::Index>(mpcRobotModelPtr_->getStateDim()) || !predictedState.allFinite()) return;
  std::tie(predictedComState_[0], predictedComState_[1]) = computeComState(predictedState);
  hasPredictedComState_ = predictedComState_[0].allFinite() && predictedComState_[1].allFinite();
}

feet_array_t<scalar_t> ContactPlanningReferenceManager::computeCadenceTouchDownShifts(scalar_t initTime,
                                                                                      const ContactPlanningConfig& config) {
  feet_array_t<scalar_t> shifts = makeFeetArray(0.0);
  if (!config.enableEnergyCadenceModulation || !hasActivePlan() || !hasPredictedComState_) return shifts;
  const scalar_t omega = config.omega();
  // The planned support point (ZMP) is the reference point of the orbital energy; the energy itself is compared between
  // the measured state and what the whole-body controller predicted for now.
  const std::optional<LipState> reference = lipReferenceState(*activePlan_, omega, initTime);
  if (!reference.has_value()) return shifts;

  // Orbital energy along the heading of the plan: a CoM that carries more energy than the controller expected passes
  // over the support earlier and the step is brought forward, less energy delays it.
  const scalar_t mass = pinocchio::computeTotalMass(pinocchioInterface_.getModel());
  const vector2_t heading(std::cos(activePlan_->yaw), std::sin(activePlan_->yaw));
  const scalar_t measured = lipOrbitalEnergy(heading.dot(comState_[0] - reference->zmp), heading.dot(comState_[1]), omega, mass);
  const scalar_t predicted =
      lipOrbitalEnergy(heading.dot(predictedComState_[0] - reference->zmp), heading.dot(predictedComState_[1]), omega, mass);
  shifts.fill(-config.energyCadenceGain * (measured - predicted));
  return shifts;
}

void ContactPlanningReferenceManager::handleContactEvents(scalar_t initTime, size_t initMode, const ContactPlanningConfig& config) {
  lastContactEvents_.fill(ContactEventReport{});
  cadenceTouchDownShift_.fill(0.0);
  if (!hasAppliedSchedule_ || !activePlan_.has_value()) return;  // only the schedule of the planning path is adapted

  const contact_flag_t measuredContact = modeNumber2StanceLeg(initMode);
  cadenceTouchDownShift_ = computeCadenceTouchDownShifts(initTime, config);
  lastContactEvents_ =
      adaptScheduleToContactEvents(appliedSchedule_, initTime, measuredContact, cadenceTouchDownShift_, config, swingLatches_);

  scalar_t totalShift = 0.0;
  bool replan = false;
  for (const ContactEventReport& report : lastContactEvents_) {
    switch (report.type) {
      case ContactEventReport::Type::EARLY_TOUCH_DOWN:
        replan = true;
        break;
      case ContactEventReport::Type::LATE_TOUCH_DOWN:
        replan = true;
        totalShift += report.timeShift;
        break;
      case ContactEventReport::Type::CADENCE_SHIFT:
        totalShift += report.timeShift;
        break;
      case ContactEventReport::Type::NONE:
        break;
    }
  }
  if (std::abs(totalShift) > 0.0) {
    // Later events moved: keep the plan aligned with the executed schedule so that the merge does not cut phases short.
    if (activePlan_->valid) activePlan_->shiftInTime(totalShift);
    scheduleShiftLog_.emplace_back(initTime, totalShift);
  }
  if (replan) replanRequested_.store(true);
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

  activatePendingPlan(initTime);
  // Only the closed-form LIP corrections need the centre of mass; skip the kinematics when they are disabled so that
  // the default configuration does exactly the work it did before they existed.
  if (config.enableDcmStepAdjustment || config.enableEnergyCadenceModulation) {
    std::tie(comState_[0], comState_[1]) = computeComState(initState);
    updatePredictedComState(initTime);
  } else {
    hasPredictedComState_ = false;
  }

  // Adapt the schedule executed so far to the measured contact state before it is merged with the plan.
  handleContactEvents(initTime, initMode, config);

  // The plan honoured the applied schedule up to its own commit boundary. Merge from there (it is in the future when
  // the plan is fresh), never earlier than the boundary that protects swings in flight or imminent in the schedule
  // executed right now, and never in the past.
  const scalar_t boundaryNow = hasAppliedSchedule_ ? commitBoundary(initTime) : initTime + config.commitTime;
  scalar_t commitTime = boundaryNow;
  if (hasActivePlan()) {
    commitTime = std::max({initTime, activePlan_->committedUntil, boundaryNow});
  }
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
  updateSwingTrajectories(schedule, initTime, terrainHeight, config);

  modeSchedule = schedule;
  modeSchedule_ = schedule;
  appliedSchedule_ = schedule;
  hasAppliedSchedule_ = true;
  updateFootBookkeeping(initTime, initState);
  updateDcmStepAdjustment(initTime, config);
}

void ContactPlanningReferenceManager::updateSwingTrajectories(const ModeSchedule& schedule,
                                                              scalar_t initTime,
                                                              scalar_t terrainHeight,
                                                              const ContactPlanningConfig& config) {
  const SwingTrajectoryPlanner::Config& swingConfig = swingTrajectoryPtr_->getConfig();
  const size_t numPhases = schedule.modeSequence.size();
  feet_array_t<scalar_array_t> liftOffHeights = makeFeetArray(scalar_array_t(numPhases, terrainHeight));
  feet_array_t<scalar_array_t> touchDownHeights =
      makeFeetArray(scalar_array_t(numPhases, terrainHeight + swingConfig.touchDownHeightOffset));

  // A foot that missed the ground at its planned touch-down searches for it: while its swing is extended the height target
  // at touch-down descends at the search velocity, so that the foot keeps moving down instead of hovering.
  if (config.enablePhaseResetting && config.lateTouchdownSearchVelocity > 0.0) {
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const SwingTimingLatch& latch = swingLatches_[foot];
      if (!latch.active || latch.lateExtension <= 0.0) continue;
      const auto range = swingPhaseIndexRange(schedule, foot, initTime);
      if (!range.has_value() || range->first == 0) continue;
      if (std::abs(schedule.eventTimes[range->first - 1] - latch.liftOffTime) > kSameSwingTolerance) continue;
      for (size_t p = range->first; p <= range->second; ++p) {
        touchDownHeights[foot][p] -= config.lateTouchdownSearchVelocity * latch.lateExtension;
      }
    }
  }
  swingTrajectoryPtr_->update(schedule, liftOffHeights, touchDownHeights);
}

void ContactPlanningReferenceManager::updateDcmStepAdjustment(scalar_t initTime, const ContactPlanningConfig& config) {
  dcmStepAdjustment_.fill(vector2_t::Zero());
  if (!config.enableDcmStepAdjustment || !hasActivePlan() || !hasPredictedComState_) return;
  const scalar_t omega = config.omega();
  // Deviation from what the whole-body controller itself predicted for now. It is zero while the robot does what the
  // NMPC expects, however far the planner's reduced model has drifted, and non-zero only under a real disturbance.
  const vector2_t dcmError =
      computeDcm(comState_[0], comState_[1], omega) - computeDcm(predictedComState_[0], predictedComState_[1], omega);

  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    const auto phase = swingPhase(foot, initTime);
    if (!phase.has_value()) continue;
    const scalar_t touchDownTime = phase->second;
    const std::optional<vector2_t> landing = activePlan_->footholdAtTime(foot, touchDownTime);
    if (!landing.has_value()) continue;
    const vector2_t adjustment =
        dcmStepAdjustment(dcmError, omega, touchDownTime - initTime, config.dcmAdjustmentGain, config.dcmAdjustmentMaxOffset);
    const std::optional<LipState> atTouchDown = lipReferenceState(*activePlan_, omega, touchDownTime);
    const vector2_t comAtTouchDown = atTouchDown.has_value() ? atTouchDown->com : activePlan_->comPosition.back();
    const vector2_t clipped = clipFootholdToReach(*landing + adjustment, *landing, comAtTouchDown, activePlan_->yaw, config);
    dcmStepAdjustment_[foot] = clipped - *landing;
  }
}

scalar_t ContactPlanningReferenceManager::commitBoundary(scalar_t time) const {
  const ContactPlanningConfig config = getConfig();
  scalar_t boundary = time + config.commitTime;
  const auto& eventTimes = appliedSchedule_.eventTimes;
  const auto& modeSequence = appliedSchedule_.modeSequence;
  if (modeSequence.empty()) return boundary;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    // Walk the phases that overlap [time, boundary]; a swing phase among them extends the boundary to its touch-down.
    for (size_t i = modeIndexAtTime(appliedSchedule_, time); i < modeSequence.size(); ++i) {
      const scalar_t phaseStart = (i == 0) ? -std::numeric_limits<scalar_t>::infinity() : eventTimes[i - 1];
      if (phaseStart >= boundary) break;
      const bool inContact = modeNumber2StanceLeg(modeSequence[i])[foot];
      if (!inContact && i < eventTimes.size()) {
        boundary = std::max(boundary, eventTimes[i]);  // touch-down of this swing
      }
    }
  }
  return boundary;
}

std::optional<std::pair<scalar_t, scalar_t>> ContactPlanningReferenceManager::swingPhase(size_t contactIndex, scalar_t time) const {
  return swingPhaseAtTime(appliedSchedule_, contactIndex, time);
}

std::optional<SwingFootReference> ContactPlanningReferenceManager::getSwingFootReference(size_t contactIndex, scalar_t time) const {
  if (!hasActivePlan() || !footBookkeepingInitialized_) return std::nullopt;
  const auto phase = swingPhase(contactIndex, time);
  if (!phase.has_value()) return std::nullopt;
  const auto [liftOffTime, touchDownTime] = *phase;
  const std::optional<vector2_t> landing = activePlan_->footholdAtTime(contactIndex, touchDownTime);
  if (!landing.has_value()) return std::nullopt;

  // The xy motion is timed on the swing without its late touch-down extension: a foot searching for the ground holds its
  // landing target instead of drifting back along the step.
  scalar_t xyTouchDownTime = touchDownTime;
  const SwingTimingLatch& latch = swingLatches_[contactIndex];
  if (latch.active && std::abs(latch.liftOffTime - liftOffTime) <= kSameSwingTolerance && latch.lateExtension > 0.0) {
    xyTouchDownTime = touchDownTime - latch.lateExtension;
  }
  const scalar_t duration = xyTouchDownTime - liftOffTime;
  if (duration <= 1e-6) return std::nullopt;

  const vector2_t start = liftOffPositions_[contactIndex].head<2>();
  const scalar_t tau = std::clamp((time - liftOffTime) / duration, 0.0, 1.0);

  // Use a cubic spline with p'(0) = 1 and p'(1) = 0 to command an initial velocity matching the step velocity.
  // This prevents the swing foot from kicking backward relative to the moving body at lift-off.
  const scalar_t tau2 = tau * tau;
  const scalar_t blend = -tau2 * tau + tau2 + tau;
  const scalar_t blendRate = (-3.0 * tau2 + 2.0 * tau + 1.0) / duration;

  // The DCM step adjustment (zero when disabled) is blended in with the same profile, so the target moves smoothly.
  const vector2_t delta = *landing + dcmStepAdjustment_[contactIndex] - start;

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

  std::tie(input.comPosition, input.comVelocity) = computeComState(initState);
  input.yaw = mpcRobotModelPtr_->getBaseOrientationEulerZYX(initState)(0);

  const feet_array_t<vector3_t> feet = computeFootPositions(initState);
  for (size_t i = 0; i < N_CONTACTS; ++i) {
    input.footPositions[i] = feet[i].head<2>();
  }

  // Events at exactly `initTime` count as passed (a touch-down placed at the current time by the phase resetting is a
  // contact for the planner), consistently with every other schedule query of this manager.
  const ModeSchedule& schedule = hasAppliedSchedule_ ? appliedSchedule_ : this->getModeSchedule();
  input.contacts = contactFlagsAtTime(schedule, initTime);
  const auto& eventTimes = schedule.eventTimes;
  const auto& modeSequence = schedule.modeSequence;
  const size_t modeIndex = modeIndexAtTime(schedule, initTime);
  for (size_t i = 0; i < N_CONTACTS; ++i) {
    // Walk back through the events while the foot keeps the same contact state.
    scalar_t phaseStart = initTime - kLongAgo;
    size_t index = modeIndex;
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
  for (size_t i = modeIndex; i > 0 && input.lastSwungFoot < 0; --i) {
    const contact_flag_t before = modeNumber2StanceLeg(modeSequence[i - 1]);
    const contact_flag_t after = modeNumber2StanceLeg(modeSequence[i]);
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (before[foot] && !after[foot]) input.lastSwungFoot = static_cast<int>(foot);
    }
  }

  const ContactPlanningConfig config = getConfig();
  input.committedUntil = hasAppliedSchedule_ ? commitBoundary(initTime) : initTime + config.commitTime;
  // Every node whose midpoint lies before the boundary is fixed to the applied schedule; at least one node stays free.
  const int maxCommitted = std::max(0, config.numNodes - 1);
  input.committedContacts.clear();
  for (int k = 0; k < maxCommitted; ++k) {
    const scalar_t t = initTime + (static_cast<scalar_t>(k) + 0.5) * config.dt;
    if (t >= input.committedUntil) break;
    input.committedContacts.push_back(contactFlagsAtTime(schedule, t));
  }
  return input;
}

}  // namespace ocs2::humanoid
