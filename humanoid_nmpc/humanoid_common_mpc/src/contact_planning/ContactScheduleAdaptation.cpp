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

#include "humanoid_common_mpc/contact_planning/ContactPlanningTermFactory.h"
#include "humanoid_common_mpc/contact_planning/execution/ExecutionContext.h"
#include "humanoid_common_mpc/contact_planning/execution/ScheduleAdaptationPipeline.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {

namespace {
constexpr scalar_t kSameSwingTolerance = 1e-6;  // [s] lift-off times closer than this identify the same swing
constexpr scalar_t kMinTimeShift = 1e-6;        // [s] smaller event shifts are not applied

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

scalar_t stanceDutyFactor(const ModeSchedule& schedule, size_t foot, scalar_t time, scalar_t window) {
  // 1 rather than 0 is the right answer to "no schedule": it is the always-in-contact case, and it is the value at
  // which the impulse-scaling correction is identically zero, so an unknown schedule leaves the contact-force
  // reference exactly as it was instead of inventing a scaling from nothing.
  //
  // A schedule with NO EVENT TIMES counts as no schedule, and that case is not hypothetical: `ModeSchedule()` is
  // defined as ModeSchedule({}, {0}), i.e. one FLY mode and no events, and that is what SwitchedModelReferenceManager
  // holds until its first modifyReferences(). Walking it would report a duty factor of zero - no foot ever in contact
  // - which the caller clamps to minimumDutyFactor and turns into a fivefold contact-force reference on the strength
  // of a placeholder. A duty factor is a fraction of a CYCLE, and a schedule with no events describes no cycle.
  if (schedule.modeSequence.empty() || schedule.eventTimes.empty() || !(window > 0.0)) return 1.0;

  const scalar_t windowEnd = time + window;
  scalar_t contactDuration = 0.0;
  scalar_t coveredDuration = 0.0;
  // A well-formed schedule has N modes and N - 1 events, but nothing in ModeSchedule enforces that and its default
  // constructor produces one mode and no events at all. Bounding the walk by BOTH sizes is what keeps a malformed or
  // half-filled schedule from being read past the end of its event array.
  const size_t numPhases = std::min(schedule.modeSequence.size(), schedule.eventTimes.size() + 1);
  for (size_t phase = 0; phase < numPhases; ++phase) {
    // Phase `p` runs from eventTimes[p - 1] to eventTimes[p]; the first and last phases are unbounded, and are clipped
    // to the window rather than skipped so that a window lying entirely inside one of them still measures something.
    const scalar_t phaseStart = phase == 0 ? time : schedule.eventTimes[phase - 1];
    const scalar_t phaseEnd = phase < schedule.eventTimes.size() ? schedule.eventTimes[phase] : windowEnd;
    const scalar_t overlapStart = std::max(phaseStart, time);
    const scalar_t overlapEnd = std::min(phaseEnd, windowEnd);
    if (overlapEnd <= overlapStart) continue;
    const scalar_t overlap = overlapEnd - overlapStart;
    coveredDuration += overlap;
    if (footInContact(schedule, phase, foot)) contactDuration += overlap;
  }
  // Measured against what the schedule COVERS rather than against the requested window. A schedule that ends inside
  // the window would otherwise report the missing tail as swing, which would scale the force up on the strength of
  // events that were never scheduled.
  if (!(coveredDuration > 0.0)) return 1.0;
  return std::clamp(contactDuration / coveredDuration, 0.0, 1.0);
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

scalar_t commitBoundaryForSchedule(const ModeSchedule& schedule, scalar_t time, scalar_t commitTime, scalar_t maxCommitExtension) {
  scalar_t boundary = time + commitTime;
  // Cap on the extension below. Without one the walk chains through back-to-back swings, and a gait that exchanges
  // support in a single instant has nothing but back-to-back swings: the boundary then reaches the end of the stepping
  // region and no plan ever reaches past it again.
  const scalar_t limit = maxCommitExtension > 0.0 ? boundary + maxCommitExtension : std::numeric_limits<scalar_t>::infinity();
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
    if (boundary >= limit) break;  // capped: stop chaining into the swings the extension has just reached
  }
  // Clamping can leave the boundary inside a swing that is already in flight, which hands a later plan the authority
  // to re-time it. That is the price of keeping the boundary finite; the cap is off by default for that reason.
  return std::min(boundary, limit);
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
    // The last committed node is sampled at the boundary whether it straddles the boundary or ends on it: an event
    // inside it (a touch-down after its midpoint) has been executed by the boundary, and the plan's first free node has
    // to continue from the state the executed schedule hands over there, not from the state at the node's midpoint.
    const bool last = nodeEnd >= committedUntil - kMinTimeShift;
    sampleTimes.push_back(last ? committedUntil : nodeStart + 0.5 * dt);
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
  // The schedule rules of the configuration's execution list, in list order (the heading override and the foothold
  // rules do not touch the schedule); the cadence shifts are given, so the cadence rule's own computation is skipped.
  TermCollection<ExecutionRule> rules;
  for (const std::string& name : config.formulation.execution) {
    const std::string canonical = canonicalTermName(TermKind::EXECUTION_RULE, name);
    if (canonical == term::kPhaseResetting || canonical == term::kEnergyCadenceModulation) {
      rules.add(canonical, ContactPlanningTermFactory::makeExecutionRule(canonical));
      rules.get(canonical).configure(config);
    }
  }
  ExecutionContext ctx;
  ctx.time = time;
  ctx.measuredContact = measuredContact;
  ctx.config = &config;
  ctx.cadenceTouchDownShift = cadenceTouchDownShift;
  return adaptScheduleWithRules(schedule, ctx, rules, latches);
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

vector2_t clipFootholdToReach(
    const vector2_t& foothold, size_t contactIndex, const vector2_t& comAtTouchDown, scalar_t yaw, const ContactPlanningConfig& config) {
  const vector2_t ex(std::cos(yaw), std::sin(yaw));
  const vector2_t ey(-std::sin(yaw), std::cos(yaw));
  const vector2_t relative = foothold - comAtTouchDown;
  // The planner's reachability rows put the left foot (index 0) on the +y side of the CoM and the right foot on the -y
  // side. Reading the side off the nominal foothold instead flipped it whenever the planned foothold sat on the other
  // side of the planned CoM (a soft-row violation, or the CoM ahead of the feet in a tight turn) and clipped the foot
  // into the other foot's region.
  const scalar_t side = (contactIndex == 0) ? 1.0 : -1.0;
  const ReachabilityParameters& reach = config.reachability;
  const scalar_t x = std::clamp(ex.dot(relative), -reach.reachX, reach.reachX);
  const scalar_t y = side * std::clamp(side * ey.dot(relative), reach.reachYInner, reach.reachYOuter);
  return comAtTouchDown + x * ex + y * ey;
}

}  // namespace ocs2::humanoid
