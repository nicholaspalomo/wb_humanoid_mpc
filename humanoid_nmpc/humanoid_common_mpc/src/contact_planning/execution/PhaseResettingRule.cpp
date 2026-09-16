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

#include "humanoid_common_mpc/contact_planning/execution/PhaseResettingRule.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace ocs2::humanoid {

namespace {
constexpr scalar_t kSameSwingTolerance = 1e-6;  // [s] lift-off times closer than this identify the same swing
constexpr scalar_t kMinTimeShift = 1e-6;        // [s] smaller event shifts are not applied
}  // namespace

std::string PhaseResettingRule::describe() const {
  std::ostringstream out;
  out << "early touch-down after " << params_.earlyTouchdownMinSwingRatio << " of the swing, debounced "
      << params_.earlyTouchdownMinContactDuration << " s, ends the swing in place; late touch-down extends the swing in steps of "
      << params_.lateTouchdownExtensionStep << " s up to " << params_.maxLateTouchdownExtension << " s, descending at "
      << params_.lateTouchdownSearchVelocity << " m/s";
  return out.str();
}

void PhaseResettingRule::configure(const ContactPlanningConfig& config) {
  const PhaseResettingParameters& p = config.phaseResetting;
  if (p.earlyTouchdownMinSwingRatio < 0.0 || p.earlyTouchdownMinSwingRatio > 1.0) {
    throw std::invalid_argument("[phase_resetting] earlyTouchdownMinSwingRatio must be in [0, 1]");
  }
  if (p.earlyTouchdownMinContactDuration < 0.0)
    throw std::invalid_argument("[phase_resetting] earlyTouchdownMinContactDuration must be >= 0");
  if (p.maxLateTouchdownExtension < 0.0) throw std::invalid_argument("[phase_resetting] maxLateTouchdownExtension must be >= 0");
  if (p.lateTouchdownExtensionStep <= 0.0) throw std::invalid_argument("[phase_resetting] lateTouchdownExtensionStep must be positive");
  if (p.lateTouchdownSearchVelocity < 0.0) throw std::invalid_argument("[phase_resetting] lateTouchdownSearchVelocity must be >= 0");
  params_ = p;
}

bool PhaseResettingRule::adaptSwingingFoot(const ExecutionContext& ctx,
                                           size_t foot,
                                           scalar_t liftOff,
                                           scalar_t /*touchDown*/,
                                           ModeSchedule& schedule,
                                           SwingTimingLatch& latch,
                                           ContactEventReport& report) const {
  const scalar_t time = ctx.time;
  // Early touch-down: contact measured after the initial fraction of the nominal swing that is ignored as scuffing, and
  // persisting for the debounce duration so that a single chattering sample does not end the swing.
  const scalar_t nominalDuration = latch.nominalTouchDownTime - liftOff;
  const bool pastScuffingWindow = nominalDuration > 0.0 && (time - liftOff) >= params_.earlyTouchdownMinSwingRatio * nominalDuration;
  if (ctx.measuredContact[foot] && pastScuffingWindow) {
    if (!latch.contactObserved) {
      latch.contactObserved = true;
      latch.contactObservedSince = time;
    }
    if (time - latch.contactObservedSince >= params_.earlyTouchdownMinContactDuration - kMinTimeShift &&
        truncateSwingPhase(schedule, foot, time).has_value()) {
      report.type = ContactEventReport::Type::EARLY_TOUCH_DOWN;
      report.touchDownTime = time;
      report.timeShift = 0.0;
      latch = SwingTimingLatch{};
      return true;
    }
  } else {
    latch.contactObserved = false;
  }
  return false;
}

bool PhaseResettingRule::adaptContactFoot(
    const ExecutionContext& ctx, size_t /*foot*/, ModeSchedule& schedule, SwingTimingLatch& latch, ContactEventReport& report) const {
  const scalar_t time = ctx.time;
  // Late touch-down: extend the swing in small steps up to the total extension budget, measured from the planned touch-down.
  const size_t index = modeIndexAtTime(schedule, time);
  const scalar_t touchDown = schedule.eventTimes[index - 1];
  const scalar_t plannedTouchDown = latch.plannedTouchDownTime();
  const scalar_t latestTouchDown = plannedTouchDown + params_.maxLateTouchdownExtension;
  const scalar_t target = std::min(time + params_.lateTouchdownExtensionStep, latestTouchDown);
  if (target <= time + kMinTimeShift) return false;  // budget used up: the contact phase proceeds as scheduled
  const scalar_t shift = target - touchDown;
  if (shift > kMinTimeShift && shiftEventsFrom(schedule, index - 1, shift)) {
    latch.lateExtension = target - plannedTouchDown;
    report.type = ContactEventReport::Type::LATE_TOUCH_DOWN;
    report.touchDownTime = target;
    report.timeShift = shift;
    return true;
  }
  return false;
}

std::optional<GroundSearchRequest> PhaseResettingRule::groundSearch(const ExecutionContext& ctx,
                                                                    size_t foot,
                                                                    const ModeSchedule& schedule,
                                                                    const SwingTimingLatch& latch) const {
  // A foot that missed the ground at its planned touch-down searches for it: the height reference of the extended swing
  // is the planned swing's, continued past the planned touch-down as a straight descent at the search velocity, so that
  // the foot keeps moving down instead of hovering, and the reference never rises when the swing is extended again.
  if (!latch.active || latch.lateExtension <= 0.0) return std::nullopt;
  const auto range = swingPhaseIndexRange(schedule, foot, ctx.time);
  if (!range.has_value() || range->first == 0) return std::nullopt;
  if (std::abs(schedule.eventTimes[range->first - 1] - latch.liftOffTime) > kSameSwingTolerance) return std::nullopt;
  return GroundSearchRequest{latch.liftOffTime, latch.plannedTouchDownTime(), params_.lateTouchdownSearchVelocity};
}

}  // namespace ocs2::humanoid
