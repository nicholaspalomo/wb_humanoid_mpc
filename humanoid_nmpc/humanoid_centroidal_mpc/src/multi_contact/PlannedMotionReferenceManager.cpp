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

#include "humanoid_centroidal_mpc/multi_contact/PlannedMotionReferenceManager.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "ocs2_core/Dimensions.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {
namespace {

/** [s] How far a swing's window may reach past the horizon and still be
 * built: one swing beyond either end, so the horizon's first and last nodes
 * read a whole swing. */
constexpr scalar_t kSwingWindowMargin = 1.0;
// [s] Plan times closer than this are one.
constexpr scalar_t kPhaseTimeTolerance = 1.0e-6;

}  // namespace

absl::StatusOr<std::shared_ptr<PlannedMotionReferenceManager>>
PlannedMotionReferenceManager::Create(
    std::shared_ptr<GaitSchedule> gaitSchedulePtr,
    std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPtr,
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    MultiContactPlanSource source,
    const PlannedMotionSettings& settings) {
  ASSIGN_OR_RETURN(
      const PlanContactOrder order,
      PlanContactOrder::Create(source.scenario.contactSet,
                               mpcRobotModel.modelSettings.contactNames));
  const scalar_t robotMass =
      pinocchio::computeTotalMass(pinocchioInterface.getModel());
  // Forward and reverse: one period of the cycle, played periodically.
  PlanRepetition repetition = PlanRepetition::kOnce;
  if (settings.playback == PlanPlayback::kForwardAndReverse) {
    ASSIGN_OR_RETURN(source.plan, forwardAndReverseCycle(
                                      source.plan, settings.playbackDwell));
    repetition = PlanRepetition::kPeriodic;
  }
  ASSIGN_OR_RETURN(PlannedMotion motion,
                   PlannedMotion::Create(source.plan, order, mpcRobotModel,
                                         robotMass, repetition));
  return std::shared_ptr<PlannedMotionReferenceManager>(
      absl::WrapUnique(new PlannedMotionReferenceManager(
          std::move(gaitSchedulePtr), std::move(swingTrajectoryPtr),
          pinocchioInterface, mpcRobotModel, std::move(motion),
          std::move(source.scenario), settings)));
}

PlannedMotionReferenceManager::PlannedMotionReferenceManager(
    std::shared_ptr<GaitSchedule> gaitSchedulePtr,
    std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPtr,
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    PlannedMotion motion,
    multi_contact::LoadedScenario scenario,
    const PlannedMotionSettings& settings)
    : SwitchedModelReferenceManager(std::move(gaitSchedulePtr),
                                    std::move(swingTrajectoryPtr),
                                    pinocchioInterface,
                                    mpcRobotModel),
      motion_(std::move(motion)),
      scenario_(std::move(scenario)),
      settings_(settings),
      clock_(motion_.eventTimes(),
             settings.clockHoldLead,
             motion_.startTime(),
             motion_.period()),
      swingPlanner_(numContacts_),
      measuredContactPositions_(numContacts_, vector3_t::Zero()),
      liftOffPositions_(numContacts_) {
  const pinocchio::Model& model = pinocchioInterface_.getModel();
  for (const std::string& name : mpcRobotModel.modelSettings.contactNames) {
    contactFrames_.push_back(model.getFrameId(name));
  }
}

std::optional<multi_contact::ContactCandidate>
PlannedMotionReferenceManager::candidateOf(size_t contactIndex) const {
  const std::optional<size_t> planIndex =
      motion_.order().planIndexOf(contactIndex);
  if (!planIndex.has_value()) return std::nullopt;
  return scenario_.contactSet.candidate(static_cast<int>(*planIndex));
}

void PlannedMotionReferenceManager::resetRuntimeState() {
  SwitchedModelReferenceManager::resetRuntimeState();
  std::fill(liftOffPositions_.begin(), liftOffPositions_.end(), std::nullopt);
  swingPlanner_.clear();
}

ContactPatchFrame PlannedMotionReferenceManager::getContactPatchFrame(
    size_t contactIndex, scalar_t time) const {
  const scalar_t planTime = planTimeAt(time);
  std::optional<PlannedContactPhase> phase =
      motion_.phaseAt(contactIndex, planTime);
  if (!phase.has_value()) phase = motion_.nextPhase(contactIndex, planTime);
  if (!phase.has_value()) phase = motion_.previousPhase(contactIndex, planTime);
  if (!phase.has_value()) {
    return SwitchedModelReferenceManager::getContactPatchFrame(contactIndex,
                                                               time);
  }
  return phase->patch;
}

void PlannedMotionReferenceManager::measure(const vector_t& state) {
  const pinocchio::Model& model = pinocchioInterface_.getModel();
  pinocchio::Data& data = pinocchioInterface_.getData();
  const vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  pinocchio::forwardKinematics(model, data, q);
  for (size_t i = 0; i < contactFrames_.size(); ++i) {
    measuredContactPositions_[i] =
        pinocchio::updateFramePlacement(model, data, contactFrames_[i])
            .translation();
  }
  measuredCenterOfMass_ = pinocchio::centerOfMass(model, data, q);
}

vector3_t PlannedMotionReferenceManager::centerOfMassOf(const vector_t& state) {
  return pinocchio::centerOfMass(
      pinocchioInterface_.getModel(), pinocchioInterface_.getData(),
      vector_t(mpcRobotModelPtr_->getGeneralizedCoordinates(state)));
}

bool PlannedMotionReferenceManager::stanceReached(size_t event,
                                                  scalar_t holdTime) {
  const size_t stance = motion_.modeBeforeEvent(event);
  for (size_t i = 0; i < numContacts_; ++i) {
    if (!contactIsActiveInMode(stance, i, numContacts_)) continue;
    const std::optional<PlannedContactPhase> phase =
        motion_.phaseAt(i, holdTime);
    if (!phase.has_value()) continue;
    const scalar_t distance =
        std::abs(phase->normal.dot(measuredContactPositions_[i]) -
                 phase->patch.normalOffset);
    if (distance > settings_.contactTolerance) return false;
  }
  const vector3_t plannedCom =
      centerOfMassOf(mpcRobotModelPtr_->getDesiredState(
          motion_.targets(), motion_.cycleTime(holdTime)));
  const scalar_t comError = (plannedCom - measuredCenterOfMass_)
                                .head<kHorizontalPositionDim>()
                                .norm();
  return comError <= settings_.comTolerance;
}

ModeSchedule PlannedMotionReferenceManager::modeScheduleAtOffset(
    scalar_t initTime, scalar_t finalTime) const {
  const scalar_t offset = clock_.offset();
  // A periodic plan's schedule over the horizon and the swings' margin.
  ModeSchedule plan =
      motion_.modeScheduleOver(initTime - offset - kSwingWindowMargin,
                               finalTime - offset + kSwingWindowMargin);
  std::vector<scalar_t> eventTimes;
  eventTimes.reserve(plan.eventTimes.size());
  for (const scalar_t planTime : plan.eventTimes) {
    eventTimes.push_back(planTime + offset);
  }
  return ModeSchedule(std::move(eventTimes), std::move(plan.modeSequence));
}

TargetTrajectories PlannedMotionReferenceManager::targetsAtOffset(
    scalar_t initTime, scalar_t finalTime) const {
  const TargetTrajectories& plan = motion_.targets();
  const scalar_t offset = clock_.offset();
  // The knots over the horizon, with the one before and the one after it, so
  // that every node interpolates between two of the plan's knots.
  TargetTrajectories targets =
      motion_.targetsOver(initTime - offset, finalTime - offset);
  for (scalar_t& time : targets.timeTrajectory) time += offset;
  if (targets.empty()) {
    // Past the plan's end: it holds its last knot.
    targets.timeTrajectory.push_back(initTime);
    targets.stateTrajectory.push_back(plan.stateTrajectory.back());
    targets.inputTrajectory.push_back(plan.inputTrajectory.back());
  }
  return targets;
}

vector3_t PlannedMotionReferenceManager::plannedContactPosition(
    size_t contactIndex, scalar_t planTime) {
  const pinocchio::Model& model = pinocchioInterface_.getModel();
  pinocchio::Data& data = pinocchioInterface_.getData();
  const vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(
      mpcRobotModelPtr_->getDesiredState(motion_.targets(),
                                         motion_.cycleTime(planTime)));
  pinocchio::forwardKinematics(model, data, q);
  return pinocchio::updateFramePlacement(model, data,
                                         contactFrames_[contactIndex])
      .translation();
}

void PlannedMotionReferenceManager::plannedPathSwings(
    size_t contactIndex,
    const PlannedSwingWindow& window,
    scalar_t offset,
    scalar_t from,
    scalar_t to,
    std::vector<ContactSwingTrajectory>& swings) {
  // A segment is a straight line: its arc's inner control points lie this far
  // [m] along it, it keeps no clearance of its own (the plan's path does), and
  // a segment that does not move bulges by as much.
  constexpr scalar_t kSegmentControlDistance = 1.0e-3;
  constexpr scalar_t kStill = 1.0e-6;  // [m]
  const ContactSwingSettings segmentSettings{
      .liftHeight = kSegmentControlDistance, .clearance = 0.0};
  // The plan's knots inside the window, at the window's period.
  const scalar_t shift =
      window.touchDownTime - motion_.cycleTime(window.touchDownTime);
  std::vector<scalar_t> times = {window.liftOffTime};
  for (const scalar_t knot : motion_.targets().timeTrajectory) {
    const scalar_t time = knot + shift;
    if (time > window.liftOffTime && time < window.touchDownTime) {
      times.push_back(time);
    }
  }
  times.push_back(window.touchDownTime);
  const multi_contact::TerrainModel& terrain = scenario_.terrain;
  for (size_t k = 0; k + 1 < times.size(); ++k) {
    if (times[k + 1] + offset < from || times[k] + offset > to) continue;
    const vector3_t start = plannedContactPosition(contactIndex, times[k]);
    const vector3_t end = plannedContactPosition(contactIndex, times[k + 1]);
    const vector3_t way = end - start;
    const vector3_t direction =
        way.norm() > kStill ? vector3_t(way.normalized()) : vector3_t::UnitZ();
    absl::StatusOr<ContactSwingTrajectory> segment =
        ContactSwingTrajectory::Create(
            times[k] + offset, times[k + 1] + offset,
            ContactSwingEndpoint{.position = start, .normal = direction},
            ContactSwingEndpoint{.position = end,
                                 .normal = way.norm() > kStill
                                               ? vector3_t(-direction)
                                               : direction},
            segmentSettings, [&terrain](const vector3_t& point) {
              return terrain.signedDistance<scalar_t>(point);
            });
    if (!segment.ok()) {
      LOG(WARNING)
          << "[PlannedMotionReferenceManager] no path segment of "
          << mpcRobotModelPtr_->modelSettings.contactNames[contactIndex] << ": "
          << segment.status();
      continue;
    }
    swings.push_back(*std::move(segment));
  }
}

std::vector<ContactSwingTrajectory> PlannedMotionReferenceManager::swingsOf(
    size_t contactIndex, scalar_t initTime, scalar_t finalTime) {
  std::vector<ContactSwingTrajectory> swings;
  const scalar_t offset = clock_.offset();
  const multi_contact::TerrainModel& terrain = scenario_.terrain;
  const bool inContactNow = contactIsActiveInMode(
      modeSchedule_.modeAtTime(initTime), contactIndex, numContacts_);
  for (const PlannedSwingWindow& window : motion_.swingWindowsOver(
           contactIndex, initTime - offset - kSwingWindowMargin,
           finalTime - offset + kSwingWindowMargin)) {
    // A swing longer than the limit follows the plan's pad.
    if (settings_.arcSwingDurationLimit > 0.0 &&
        window.touchDownTime - window.liftOffTime >
            settings_.arcSwingDurationLimit) {
      plannedPathSwings(contactIndex, window, offset,
                        initTime - kSwingWindowMargin,
                        finalTime + kSwingWindowMargin, swings);
      continue;
    }
    const scalar_t liftOffTime = window.liftOffTime + offset;
    const scalar_t touchDownTime = window.touchDownTime + offset;
    if (touchDownTime < initTime - kSwingWindowMargin ||
        liftOffTime > finalTime + kSwingWindowMargin) {
      continue;
    }
    // The pad it touches down on: the phase there, or the next one.
    std::optional<PlannedContactPhase> target =
        motion_.phaseAt(contactIndex, window.touchDownTime);
    if (!target.has_value()) {
      target = motion_.nextPhase(contactIndex, window.touchDownTime);
    }
    if (!target.has_value()) continue;
    ContactSwingEndpoint liftOff;
    if (window.fromStart) {
      // In the air from the plan's start: from where it was at the first
      // solve, departing along its target's normal.
      if (!liftOffPositions_[contactIndex].has_value()) continue;
      liftOff.position = *liftOffPositions_[contactIndex];
      liftOff.normal = target->normal;
    } else {
      std::optional<PlannedContactPhase> leaving =
          motion_.phaseAt(contactIndex, window.liftOffTime);
      if (!leaving.has_value()) {
        leaving = motion_.previousPhase(contactIndex, window.liftOffTime);
      }
      if (!leaving.has_value()) continue;
      // Already swinging: from where it last was in contact; in contact now
      // and lifting off later: from where it is; otherwise from the pad it
      // will lift off from.
      if (liftOffTime <= initTime) {
        liftOff.position =
            liftOffPositions_[contactIndex].value_or(leaving->targetPosition);
      } else if (inContactNow && leaving->startTime <=
                                     initTime - offset + kPhaseTimeTolerance) {
        // The phase it rests in now: a later one, after a touch-down within
        // the horizon, starts at its own pad.
        liftOff.position = measuredContactPositions_[contactIndex];
      } else {
        liftOff.position = leaving->targetPosition;
      }
      liftOff.normal = leaving->normal;
    }
    const ContactSwingEndpoint touchDown{.position = target->targetPosition,
                                         .normal = target->normal};
    absl::StatusOr<ContactSwingTrajectory> swing =
        ContactSwingTrajectory::Create(
            liftOffTime, touchDownTime, liftOff, touchDown, settings_.swing,
            [&terrain](const vector3_t& point) {
              return terrain.signedDistance<scalar_t>(point);
            });
    const std::string& name =
        mpcRobotModelPtr_->modelSettings.contactNames[contactIndex];
    if (!swing.ok()) {
      LOG(WARNING) << "[PlannedMotionReferenceManager] no swing of " << name
                   << ": " << swing.status();
      continue;
    }
    if (!swing->clearanceSatisfied()) {
      LOG_EVERY_N_SEC(WARNING, 1.0)
          << "[PlannedMotionReferenceManager] the swing of " << name
          << " to its pad at plan time " << window.touchDownTime
          << " s misses the clearance by " << -swing->clearanceMargin()
          << " m.";
    }
    swings.push_back(*std::move(swing));
  }
  return swings;
}

void PlannedMotionReferenceManager::updateSwings(scalar_t initTime,
                                                 scalar_t finalTime) {
  for (size_t i = 0; i < numContacts_; ++i) {
    swingPlanner_.setSwings(i, swingsOf(i, initTime, finalTime));
  }
}

void PlannedMotionReferenceManager::modifyReferences(
    scalar_t initTime,
    scalar_t finalTime,
    const vector_t& initState,
    size_t /*initMode*/,
    TargetTrajectories& targetTrajectories,
    ModeSchedule& modeSchedule) {
  measure(initState);
  clock_.advance(initTime, [this](size_t event, scalar_t holdTime) {
    return stanceReached(event, holdTime);
  });
  if (clock_.isHolding()) {
    LOG_EVERY_N_SEC(INFO, 1.0)
        << "[PlannedMotionReferenceManager] holding the plan at "
        << planTimeAt(initTime) << " s until its stance is reached.";
  }
  modeSchedule = modeScheduleAtOffset(initTime, finalTime);
  modeSchedule_ = modeSchedule;
  // Latch where every contact in contact is, and where every contact is at
  // the first solve: where the swings start from.
  for (size_t i = 0; i < numContacts_; ++i) {
    if (contactIsActiveInMode(modeSchedule.modeAtTime(initTime), i,
                              numContacts_) ||
        !liftOffPositions_[i].has_value()) {
      liftOffPositions_[i] = measuredContactPositions_[i];
    }
  }
  targetTrajectories = targetsAtOffset(initTime, finalTime);
  updateSwings(initTime, finalTime);
}

vector_t PlannedMotionReferenceManager::getDesiredInput(
    const TargetTrajectories& targetTrajectories,
    const vector_t& state,
    scalar_t time) const {
  vector_t input = SwitchedModelReferenceManager::getDesiredInput(
      targetTrajectories, state, time);
  if (settings_.forceReference != ForceReference::kPlannedWrenches) {
    return input;
  }
  const vector_t& planned = motion_.plannedWrenchesAt(planTimeAt(time));
  for (size_t i = 0; i < numContacts_; ++i) {
    const size_t start = mpcRobotModelPtr_->getContactWrenchStartIndices(i);
    const size_t dim = mpcRobotModelPtr_->getContactInputDim(i);
    if (isInContact(time, i)) {
      input.segment(start, dim) = planned.segment(start, dim);
    } else {
      input.segment(start, dim).setZero();
    }
  }
  return input;
}

}  // namespace ocs2::humanoid
