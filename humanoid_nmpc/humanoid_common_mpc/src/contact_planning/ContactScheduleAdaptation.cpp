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

#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {

namespace {
constexpr scalar_t kSameSwingTolerance = 1e-6;  // [s] lift-off times closer than this identify the same swing
constexpr scalar_t kMinTimeShift = 1e-6;        // [s] smaller event shifts are not applied
constexpr scalar_t kMinRemainingSwing = 0.02;   // [s] a re-timed touch-down stays at least this far in the future

bool footInContact(const ModeSchedule& schedule, size_t phaseIndex, size_t foot) {
  return modeNumber2StanceLeg(schedule.modeSequence[phaseIndex])[foot];
}

/** First phase of the swing of `foot` that contains phase `index` (the foot is out of contact in `index`). */
size_t firstSwingPhase(const ModeSchedule& schedule, size_t foot, size_t index) {
  while (index > 0 && !footInContact(schedule, index - 1, foot)) --index;
  return index;
}

/** Last phase of the swing of `foot` that contains phase `index`. */
size_t lastSwingPhase(const ModeSchedule& schedule, size_t foot, size_t index) {
  while (index + 1 < schedule.modeSequence.size() && !footInContact(schedule, index + 1, foot)) ++index;
  return index;
}
}  // namespace

/*============================================ schedule queries ============================================*/

size_t modeIndexAtTime(const ModeSchedule& schedule, scalar_t time) {
  const auto& eventTimes = schedule.eventTimes;
  size_t index = static_cast<size_t>(std::upper_bound(eventTimes.begin(), eventTimes.end(), time) - eventTimes.begin());
  if (!schedule.modeSequence.empty() && index >= schedule.modeSequence.size()) index = schedule.modeSequence.size() - 1;
  return index;
}

contact_flag_t contactFlagsAtTime(const ModeSchedule& schedule, scalar_t time) {
  if (schedule.modeSequence.empty()) return makeFeetArray(true);
  return modeNumber2StanceLeg(schedule.modeSequence[modeIndexAtTime(schedule, time)]);
}

std::optional<std::pair<size_t, size_t>> swingPhaseIndexRange(const ModeSchedule& schedule, size_t foot, scalar_t time) {
  if (schedule.modeSequence.empty()) return std::nullopt;
  const size_t index = modeIndexAtTime(schedule, time);
  if (footInContact(schedule, index, foot)) return std::nullopt;
  return std::make_pair(firstSwingPhase(schedule, foot, index), lastSwingPhase(schedule, foot, index));
}

std::optional<std::pair<scalar_t, scalar_t>> swingPhaseAtTime(const ModeSchedule& schedule, size_t foot, scalar_t time) {
  const auto range = swingPhaseIndexRange(schedule, foot, time);
  if (!range.has_value()) return std::nullopt;
  const auto [first, last] = *range;
  if (first == 0 || last + 1 >= schedule.modeSequence.size()) return std::nullopt;  // no lift-off or no touch-down event
  return std::make_pair(schedule.eventTimes[first - 1], schedule.eventTimes[last]);
}

std::optional<size_t> touchDownEventIndex(const ModeSchedule& schedule, size_t foot, scalar_t time) {
  const auto range = swingPhaseIndexRange(schedule, foot, time);
  if (!range.has_value() || range->second + 1 >= schedule.modeSequence.size()) return std::nullopt;
  return range->second;
}

std::optional<scalar_t> currentOrNextLiftOffTime(const ModeSchedule& schedule, size_t foot, scalar_t time) {
  if (schedule.modeSequence.empty()) return std::nullopt;
  size_t index = modeIndexAtTime(schedule, time);
  if (!footInContact(schedule, index, foot)) {
    const size_t first = firstSwingPhase(schedule, foot, index);
    if (first == 0) return std::nullopt;
    return schedule.eventTimes[first - 1];
  }
  for (size_t p = index + 1; p < schedule.modeSequence.size(); ++p) {
    if (!footInContact(schedule, p, foot)) return schedule.eventTimes[p - 1];
  }
  return std::nullopt;
}

scalar_t commitBoundaryForSchedule(const ModeSchedule& schedule, scalar_t time, scalar_t commitTime) {
  scalar_t boundary = time + commitTime;
  const auto& eventTimes = schedule.eventTimes;
  const auto& modeSequence = schedule.modeSequence;
  if (modeSequence.empty()) return boundary;
  // Walk the phases that overlap [time, boundary] in time order, all feet at once; a swing phase among them extends the
  // boundary to its touch-down, and the walk then covers the phases that overlap the extended boundary as well. (One
  // walk per foot missed a swing of an earlier foot that started on the boundary a later foot had just extended.)
  for (size_t i = modeIndexAtTime(schedule, time); i < modeSequence.size(); ++i) {
    const scalar_t phaseStart = (i == 0) ? -std::numeric_limits<scalar_t>::infinity() : eventTimes[i - 1];
    // Closed at the far end: a swing that starts exactly on the boundary is executed too. Leaving it out let a plan
    // re-decide a lift-off that the previous plan had aligned onto this very boundary, and with the planning period
    // equal to the plan's age the lift-off receded by one period per plan and the robot never stepped.
    if (phaseStart > boundary + 1e-9) break;
    if (i >= eventTimes.size()) break;  // the last phase has no touch-down to extend to
    const contact_flag_t contacts = modeNumber2StanceLeg(modeSequence[i]);
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (!contacts[foot]) boundary = std::max(boundary, eventTimes[i]);  // touch-down of this swing
    }
  }
  return boundary;
}

bool planAgreesWithSwingsInFlight(const ModeSchedule& applied, const ContactPlan& plan, scalar_t time) {
  if (!plan.valid || plan.contacts.empty()) return true;
  const contact_flag_t planned = plan.contactsAtTime(time + kMinTimeShift);
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    const auto phase = swingPhaseAtTime(applied, foot, time);
    // Not swinging, or a lift-off at `time` itself: nothing is in flight that the plan could contradict.
    if (!phase.has_value() || phase->first >= time - kMinTimeShift) continue;
    if (planned[foot]) return false;
  }
  return true;
}

std::vector<scalar_t> committedSampleTimes(scalar_t startTime, scalar_t dt, int maxNodes, scalar_t committedUntil) {
  std::vector<scalar_t> sampleTimes;
  for (int k = 0; k < maxNodes; ++k) {
    const scalar_t nodeStart = startTime + static_cast<scalar_t>(k) * dt;
    if (nodeStart >= committedUntil - kMinTimeShift) break;
    const scalar_t nodeEnd = nodeStart + dt;
    const bool inside = nodeEnd <= committedUntil + kMinTimeShift;
    sampleTimes.push_back(inside ? nodeStart + 0.5 * dt : committedUntil);
  }
  return sampleTimes;
}

std::vector<contact_flag_t> committedContactsForPlanner(
    const ModeSchedule& schedule, scalar_t startTime, scalar_t dt, int maxNodes, scalar_t committedUntil) {
  std::vector<contact_flag_t> committed;
  for (const scalar_t sampleTime : committedSampleTimes(startTime, dt, maxNodes, committedUntil)) {
    committed.push_back(contactFlagsAtTime(schedule, sampleTime));
  }
  return committed;
}

feet_array_t<scalar_t> contactPhaseStartTimes(const ModeSchedule& schedule, scalar_t time) {
  feet_array_t<scalar_t> starts = makeFeetArray(-std::numeric_limits<scalar_t>::infinity());
  if (schedule.modeSequence.empty()) return starts;
  const size_t modeIndex = modeIndexAtTime(schedule, time);
  const contact_flag_t contacts = modeNumber2StanceLeg(schedule.modeSequence[modeIndex]);
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    // Walk back through the events while the foot keeps the same contact state.
    size_t index = modeIndex;
    while (index > 0 && footInContact(schedule, index - 1, foot) == contacts[foot]) --index;
    if (index > 0) starts[foot] = schedule.eventTimes[index - 1];
  }
  return starts;
}

std::vector<feet_array_t<scalar_t>> committedPhaseStartsForPlanner(
    const ModeSchedule& schedule, scalar_t startTime, scalar_t dt, int maxNodes, scalar_t committedUntil) {
  std::vector<feet_array_t<scalar_t>> starts;
  for (const scalar_t sampleTime : committedSampleTimes(startTime, dt, maxNodes, committedUntil)) {
    starts.push_back(contactPhaseStartTimes(schedule, sampleTime));
  }
  return starts;
}

scalar_t applyScheduleShiftsToPlan(ContactPlan& plan, const std::deque<std::pair<scalar_t, scalar_t>>& shiftLog) {
  const scalar_t snapshotTime = plan.startTime;  // the snapshot's identity, read before any shift moves it
  scalar_t total = 0.0;
  for (const auto& [time, shift] : shiftLog) {
    if (time > snapshotTime) total += shift;
  }
  if (total != 0.0) plan.shiftInTime(total);
  return total;
}

/*============================================ schedule edits ==============================================*/

void removeRedundantEvents(ModeSchedule& schedule) {
  auto& eventTimes = schedule.eventTimes;
  auto& modeSequence = schedule.modeSequence;
  size_t i = 0;
  while (i + 1 < modeSequence.size()) {
    if (modeSequence[i] == modeSequence[i + 1]) {
      modeSequence.erase(modeSequence.begin() + static_cast<std::ptrdiff_t>(i) + 1);
      eventTimes.erase(eventTimes.begin() + static_cast<std::ptrdiff_t>(i));
    } else {
      ++i;
    }
  }
}

std::optional<scalar_t> truncateSwingPhase(ModeSchedule& schedule, size_t foot, scalar_t time) {
  auto& eventTimes = schedule.eventTimes;
  auto& modeSequence = schedule.modeSequence;
  const auto range = swingPhaseIndexRange(schedule, foot, time);
  if (!range.has_value()) return std::nullopt;
  size_t index = modeIndexAtTime(schedule, time);
  size_t last = range->second;
  if (last + 1 >= modeSequence.size()) return std::nullopt;  // the swing never touches down within the schedule
  const scalar_t touchDown = eventTimes[last];

  // Split the phase containing `time` unless `time` already is the start of that phase.
  if (index == 0 || eventTimes[index - 1] < time) {
    eventTimes.insert(eventTimes.begin() + static_cast<std::ptrdiff_t>(index), time);
    modeSequence.insert(modeSequence.begin() + static_cast<std::ptrdiff_t>(index) + 1, modeSequence[index]);
    ++index;
    ++last;
  }
  for (size_t p = index; p <= last; ++p) {
    contact_flag_t flags = modeNumber2StanceLeg(modeSequence[p]);
    flags[foot] = true;
    modeSequence[p] = stanceLeg2ModeNumber(flags);
  }
  removeRedundantEvents(schedule);
  return touchDown;
}

bool shiftEventsFrom(ModeSchedule& schedule, size_t firstEventIndex, scalar_t shift) {
  auto& eventTimes = schedule.eventTimes;
  if (firstEventIndex >= eventTimes.size()) return false;
  if (firstEventIndex > 0 && eventTimes[firstEventIndex] + shift <= eventTimes[firstEventIndex - 1]) return false;
  for (size_t j = firstEventIndex; j < eventTimes.size(); ++j) {
    eventTimes[j] += shift;
  }
  return true;
}

/*============================================ contact events ==============================================*/

feet_array_t<ContactEventReport> adaptScheduleToContactEvents(ModeSchedule& schedule,
                                                              scalar_t time,
                                                              const contact_flag_t& measuredContact,
                                                              const feet_array_t<scalar_t>& cadenceTouchDownShift,
                                                              const ContactPlanningConfig& config,
                                                              feet_array_t<SwingTimingLatch>& latches) {
  feet_array_t<ContactEventReport> reports = makeFeetArray(ContactEventReport{});
  if (schedule.modeSequence.empty()) return reports;

  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    SwingTimingLatch& latch = latches[foot];
    ContactEventReport& report = reports[foot];

    const auto phase = swingPhaseAtTime(schedule, foot, time);
    if (phase.has_value()) {
      // ---- the foot is scheduled to swing at `time` ----
      const auto [liftOff, touchDown] = *phase;
      if (!latch.active || std::abs(latch.liftOffTime - liftOff) > kSameSwingTolerance) {
        latch = SwingTimingLatch{};
        latch.active = true;
        latch.liftOffTime = liftOff;
        latch.nominalTouchDownTime = touchDown;
      }

      // Early touch-down: contact measured after the initial fraction of the nominal swing that is ignored as scuffing,
      // and persisting for the debounce duration so that a single chattering sample does not end the swing.
      const scalar_t nominalDuration = latch.nominalTouchDownTime - liftOff;
      const bool pastScuffingWindow = nominalDuration > 0.0 && (time - liftOff) >= config.earlyTouchdownMinSwingRatio * nominalDuration;
      if (config.enablePhaseResetting && measuredContact[foot] && pastScuffingWindow) {
        if (!latch.contactObserved) {
          latch.contactObserved = true;
          latch.contactObservedSince = time;
        }
        if (time - latch.contactObservedSince >= config.earlyTouchdownMinContactDuration - kMinTimeShift &&
            truncateSwingPhase(schedule, foot, time).has_value()) {
          report.type = ContactEventReport::Type::EARLY_TOUCH_DOWN;
          report.touchDownTime = time;
          report.timeShift = 0.0;
          latch = SwingTimingLatch{};
          continue;
        }
      } else {
        latch.contactObserved = false;
      }

      // Cadence modulation of the touch-down, relative to the nominal touch-down and within the swing duration limits.
      // Not applied while the swing is being extended past its planned touch-down (late touch-down search).
      if (config.enableEnergyCadenceModulation && latch.lateExtension <= 0.0) {
        const auto tdIndex = touchDownEventIndex(schedule, foot, time);
        if (tdIndex.has_value()) {
          scalar_t target = latch.nominalTouchDownTime + cadenceTouchDownShift[foot];
          target = std::clamp(target, liftOff + config.minSwingDuration, liftOff + config.maxSwingDuration);
          const scalar_t previousEvent = (*tdIndex > 0) ? schedule.eventTimes[*tdIndex - 1] : time;
          const scalar_t earliest = std::max(time, previousEvent) + kMinRemainingSwing;
          // A request earlier than what is still feasible brings the touch-down forward as far as allowed, but never
          // pushes a touch-down that is already imminent further out: flooring at "now + margin" on every cycle would
          // drag the touch-down along with the clock and the foot would never land.
          if (target < earliest) target = std::min(earliest, touchDown);
          const scalar_t shift = target - touchDown;
          if (std::abs(shift) > kMinTimeShift && shiftEventsFrom(schedule, *tdIndex, shift)) {
            latch.cadenceShift = target - latch.nominalTouchDownTime;
            report.type = ContactEventReport::Type::CADENCE_SHIFT;
            report.touchDownTime = target;
            report.timeShift = shift;
          }
        }
      }
      continue;
    }

    // ---- the foot is scheduled to be in contact at `time` ----
    if (!latch.active) continue;
    // The latched swing must be the one that ended at the last event before `time`; otherwise the latch is stale.
    const size_t index = modeIndexAtTime(schedule, time);
    const bool justLanded = index > 0 && !footInContact(schedule, index - 1, foot);
    if (!justLanded) {
      latch = SwingTimingLatch{};
      continue;
    }
    const size_t first = firstSwingPhase(schedule, foot, index - 1);
    const scalar_t liftOff = (first > 0) ? schedule.eventTimes[first - 1] : -std::numeric_limits<scalar_t>::infinity();
    if (std::abs(liftOff - latch.liftOffTime) > kSameSwingTolerance) {
      latch = SwingTimingLatch{};
      continue;
    }
    if (measuredContact[foot] || !config.enablePhaseResetting) {
      latch = SwingTimingLatch{};  // landed as scheduled (possibly within an extension), nothing to adapt
      continue;
    }

    // Late touch-down: extend the swing in small steps up to the total extension budget, measured from the planned touch-down.
    const scalar_t touchDown = schedule.eventTimes[index - 1];
    const scalar_t plannedTouchDown = latch.plannedTouchDownTime();
    const scalar_t latestTouchDown = plannedTouchDown + config.maxLateTouchdownExtension;
    const scalar_t target = std::min(time + config.lateTouchdownExtensionStep, latestTouchDown);
    if (target <= time + kMinTimeShift) {
      latch = SwingTimingLatch{};  // budget used up: the contact phase proceeds as scheduled
      continue;
    }
    const scalar_t shift = target - touchDown;
    if (shift > kMinTimeShift && shiftEventsFrom(schedule, index - 1, shift)) {
      latch.lateExtension = target - plannedTouchDown;
      report.type = ContactEventReport::Type::LATE_TOUCH_DOWN;
      report.touchDownTime = target;
      report.timeShift = shift;
    }
  }
  return reports;
}

/*============================================ LIP helpers =================================================*/

std::optional<LipState> lipReferenceState(const ContactPlan& plan, scalar_t omega, scalar_t time) {
  const size_t numIntervals = plan.contacts.size();
  const bool consistent = plan.valid && numIntervals > 0 && plan.zmp.size() == numIntervals &&
                          plan.comPosition.size() == numIntervals + 1 && plan.comVelocity.size() == numIntervals + 1;
  if (!consistent || omega <= 0.0 || plan.dt <= 0.0) return std::nullopt;
  if (time < plan.startTime - 1e-9 || time > plan.endTime() + 1e-9) return std::nullopt;

  const int k = plan.intervalIndex(time);
  const scalar_t tau = std::clamp(time - (plan.startTime + plan.dt * static_cast<scalar_t>(k)), 0.0, plan.dt);
  const scalar_t ch = std::cosh(omega * tau);
  const scalar_t sh = std::sinh(omega * tau);
  LipState state;
  state.zmp = plan.zmp[k];
  const vector2_t offset = plan.comPosition[k] - state.zmp;
  state.com = state.zmp + offset * ch + plan.comVelocity[k] * (sh / omega);
  state.comVelocity = offset * (omega * sh) + plan.comVelocity[k] * ch;
  return state;
}

vector2_t dcmStepAdjustment(const vector2_t& dcmError, scalar_t omega, scalar_t timeToTouchDown, scalar_t gain, scalar_t maxOffset) {
  vector2_t adjustment = gain * std::exp(omega * std::max(timeToTouchDown, 0.0)) * dcmError;
  const scalar_t norm = adjustment.norm();
  if (maxOffset >= 0.0 && norm > maxOffset) {
    adjustment *= (norm > 0.0) ? maxOffset / norm : 0.0;
  }
  return adjustment;
}

vector2_t clipFootholdToReach(const vector2_t& foothold,
                              const vector2_t& nominalFoothold,
                              const vector2_t& comAtTouchDown,
                              scalar_t yaw,
                              const ContactPlanningConfig& config) {
  const vector2_t ex(std::cos(yaw), std::sin(yaw));
  const vector2_t ey(-std::sin(yaw), std::cos(yaw));
  const vector2_t relative = foothold - comAtTouchDown;
  const scalar_t side = (ey.dot(nominalFoothold - comAtTouchDown) >= 0.0) ? 1.0 : -1.0;
  const scalar_t x = std::clamp(ex.dot(relative), -config.reachX, config.reachX);
  const scalar_t y = side * std::clamp(side * ey.dot(relative), config.reachYInner, config.reachYOuter);
  return comAtTouchDown + x * ex + y * ey;
}

}  // namespace ocs2::humanoid
