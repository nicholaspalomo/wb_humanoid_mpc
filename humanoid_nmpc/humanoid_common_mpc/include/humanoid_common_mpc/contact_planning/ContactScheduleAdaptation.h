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

#pragma once

#include <deque>
#include <optional>
#include <utility>
#include <vector>

#include <ocs2_core/reference/ModeSchedule.h>

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"

/**
 * Adaptive execution of a planned contact schedule, generic in the number of feet (N_CONTACTS).
 *
 * The mixed-integer planner decides the contact sequence on a coarse grid and re-plans at a few Hz. Between plans the
 * executed schedule is adapted to what the robot actually does, using only the measured contact state and the
 * closed-form Linear Inverted Pendulum (LIP):
 *
 *  - early touch-down:  a swing foot whose measured contact persists for earlyTouchdownMinContactDuration (after the
 *                       initial scuffing window) is switched to contact at once, in place: later events keep their
 *                       timing and the planner re-times the rest at its next run;
 *  - late touch-down:   a foot that misses the ground at its scheduled touch-down keeps swinging in small steps, up to a
 *                       bounded total extension, and every later event is delayed by the same amount;
 *  - cadence modulation: the touch-down of the swing in flight is moved by a caller-provided shift (from the LIP orbital
 *                       energy error), bounded by the swing duration limits; later events move with it;
 *  - DCM step adjustment: the planned landing spot is shifted by the DCM (capture point) error propagated to touch-down and
 *                       clipped to the reachable region.
 *
 * All schedule queries treat an event at exactly the query time as already passed, which matches the SQP: after a
 * post-event node the first interval starts an epsilon after the event (ocs2 TimeDiscretization::getIntervalStart). Note
 * that ocs2::ModeSchedule::modeAtTime uses the opposite convention (lower_bound), do not mix the two.
 */
namespace ocs2::humanoid {

/*============================================ schedule queries ============================================*/

/** Index into modeSequence of the phase active at `time` (events at exactly `time` count as passed). */
size_t modeIndexAtTime(const ModeSchedule& schedule, scalar_t time);

/** Contact flags of the phase active at `time` (same convention as modeIndexAtTime). */
contact_flag_t contactFlagsAtTime(const ModeSchedule& schedule, scalar_t time);

/**
 * Range [first, last] of consecutive phases in which `foot` is out of contact around `time`, empty if the foot is in
 * contact at `time`. Several phases belong to the same swing when other feet switch while this foot is in the air.
 */
std::optional<std::pair<size_t, size_t>> swingPhaseIndexRange(const ModeSchedule& schedule, size_t foot, scalar_t time);

/**
 * Swing phase [liftOff, touchDown] of `foot` around `time`, empty if the foot is in contact or if the lift-off or the
 * touch-down event is not part of the schedule.
 */
std::optional<std::pair<scalar_t, scalar_t>> swingPhaseAtTime(const ModeSchedule& schedule, size_t foot, scalar_t time);

/** Index into eventTimes of the touch-down event of the swing of `foot` around `time`, empty if there is none. */
std::optional<size_t> touchDownEventIndex(const ModeSchedule& schedule, size_t foot, scalar_t time);

/**
 * Lift-off time of the swing of `foot` that is in flight at `time`, or of its next swing if the foot is in contact at
 * `time`. Empty if the schedule has no such lift-off event.
 */
std::optional<scalar_t> currentOrNextLiftOffTime(const ModeSchedule& schedule, size_t foot, scalar_t time);

/**
 * End of the window in which the executed `schedule` stays fixed: `time + commitTime`, extended to the touch-down of
 * every swing that overlaps the window, so that a swing in flight, or one that starts inside the window, is executed
 * to its end and never re-timed by a later plan.
 */
scalar_t commitBoundaryForSchedule(const ModeSchedule& schedule, scalar_t time, scalar_t commitTime);

/**
 * Aligns a freshly activated plan with the schedule executed right now. The plan may only change the schedule after
 * `plan.committedUntil`, the boundary it honoured when it was made; the executed schedule may only change after
 * `boundary`, its own commit boundary now, which lies later by the plan's age and by any swing that started meanwhile.
 * Merging the plan at `boundary` cut its first lift-off short by that difference (the merged swing began at the
 * boundary but kept the plan's touch-down). The plan is shifted forward by the difference instead, whole, so that its
 * first free decision lands on the boundary. Returns the shift applied (0 when the plan's boundary is not behind).
 */
scalar_t alignPlanToCommitBoundary(ContactPlan& plan, scalar_t boundary);

/**
 * Contacts the planner has to keep fixed on its first nodes, taken from the executed schedule: every node that starts
 * before `committedUntil`, at most `maxNodes` of them. A node that lies entirely inside the window is sampled at its
 * midpoint; the node that straddles the boundary is sampled just after the boundary, i.e. with the contact state the
 * executed schedule hands over to the plan there. Sampling that node at its midpoint let the planner contradict the
 * executed schedule inside it, which the merge then turned into a delayed touch-down or a phantom re-lift.
 */
std::vector<contact_flag_t> committedContactsForPlanner(
    const ModeSchedule& schedule, scalar_t startTime, scalar_t dt, int maxNodes, scalar_t committedUntil);

/**
 * Applies to `plan` the re-timings of the executed schedule that happened after the plan's snapshot was taken. The log
 * holds (time of the shift, shift) pairs; every entry younger than the snapshot is summed and applied once, so that
 * shifts of opposite sign cancel instead of the first one moving the comparison key for the second. Returns the total.
 */
scalar_t applyScheduleShiftsToPlan(ContactPlan& plan, const std::deque<std::pair<scalar_t, scalar_t>>& shiftLog);

/*============================================ schedule edits ==============================================*/

/** Removes every event between two consecutive identical modes. */
void removeRedundantEvents(ModeSchedule& schedule);

/**
 * Ends the swing of `foot` at `time`: the phase containing `time` is split there and the foot is in contact from `time`
 * until its scheduled touch-down. Other feet and all later events keep their timing. Returns the scheduled touch-down that
 * was cut, or empty (schedule untouched) if the foot is not swinging at `time` or the swing has no touch-down event.
 */
std::optional<scalar_t> truncateSwingPhase(ModeSchedule& schedule, size_t foot, scalar_t time);

/**
 * Adds `shift` to eventTimes[firstEventIndex] and every later event (the durations of the later phases are preserved).
 * Returns false and leaves the schedule untouched if the index is out of range or the shifted event would not stay
 * strictly after its predecessor.
 */
bool shiftEventsFrom(ModeSchedule& schedule, size_t firstEventIndex, scalar_t shift);

/*============================================ contact events ==============================================*/

/** Per-foot memory of the swing currently in flight, kept by the caller between cycles. */
struct SwingTimingLatch {
  bool active = false;
  scalar_t liftOffTime = 0.0;           // identifies the swing
  scalar_t nominalTouchDownTime = 0.0;  // touch-down as planned when the swing was first seen in flight
  scalar_t cadenceShift = 0.0;          // [s] touch-down shift applied by the cadence modulation (relative to nominal)
  scalar_t lateExtension = 0.0;         // [s] time the swing has been extended past the (cadence-shifted) touch-down
  bool contactObserved = false;         // contact has been measured past the scuffing window (debounce in progress)
  scalar_t contactObservedSince = 0.0;  // [s] first cycle at which that contact was measured
  scalar_t plannedTouchDownTime() const { return nominalTouchDownTime + cadenceShift; }
};

struct ContactEventReport {
  enum class Type { NONE, EARLY_TOUCH_DOWN, LATE_TOUCH_DOWN, CADENCE_SHIFT };
  Type type = Type::NONE;
  scalar_t touchDownTime = 0.0;  // touch-down after the update
  scalar_t timeShift = 0.0;      // shift applied to the touch-down and every later event (0 for an early touch-down)
};

/**
 * Adapts `schedule` to the measured contact state at `time` and applies the cadence shifts. One call per control cycle.
 *
 * @param schedule              the schedule the controller executes, modified in place
 * @param time                  current time
 * @param measuredContact       measured contact flags
 * @param cadenceTouchDownShift [s] per foot, desired touch-down shift relative to the nominal touch-down of the swing in
 *                              flight (ignored unless config.enableEnergyCadenceModulation)
 * @param config                thresholds and limits (enablePhaseResetting, earlyTouchdownMinSwingRatio,
 *                              earlyTouchdownMinContactDuration, maxLateTouchdownExtension,
 *                              lateTouchdownExtensionStep, min/maxSwingDuration)
 * @param latches               per-foot swing memory, owned by the caller
 * @return what happened to every foot
 */
feet_array_t<ContactEventReport> adaptScheduleToContactEvents(ModeSchedule& schedule,
                                                              scalar_t time,
                                                              const contact_flag_t& measuredContact,
                                                              const feet_array_t<scalar_t>& cadenceTouchDownShift,
                                                              const ContactPlanningConfig& config,
                                                              feet_array_t<SwingTimingLatch>& latches);

/*============================================ LIP helpers =================================================*/

struct LipState {
  vector2_t com = vector2_t::Zero();
  vector2_t comVelocity = vector2_t::Zero();
  vector2_t zmp = vector2_t::Zero();
};

/** Divergent component of motion (instantaneous capture point) of a LIP with natural frequency `omega`. */
inline vector2_t computeDcm(const vector2_t& com, const vector2_t& comVelocity, scalar_t omega) {
  return com + comVelocity / omega;
}

/**
 * Reference LIP state of the plan at `time`: the node state propagated in closed form through the interval containing
 * `time` with the planned (constant) ZMP of that interval. Empty if the plan is invalid or `time` is outside the plan.
 */
std::optional<LipState> lipReferenceState(const ContactPlan& plan, scalar_t omega, scalar_t time);

/**
 * Closed-form step adjustment for a DCM error measured `timeToTouchDown` before touch-down: on the LIP the error grows by
 * exp(omega * timeToTouchDown) until touch-down, and moving the foothold by that amount restores the planned DCM offset
 * with respect to the new support. `gain` scales the correction (1 = exact LIP compensation), the result is limited to
 * `maxOffset` in norm.
 */
vector2_t dcmStepAdjustment(const vector2_t& dcmError, scalar_t omega, scalar_t timeToTouchDown, scalar_t gain, scalar_t maxOffset);

/**
 * Clips `foothold` to the planner's reachable region around `comAtTouchDown` in the yaw frame: |x| <= reachX and
 * reachYInner <= side * y <= reachYOuter, where the side of the foot (left / right of the CoM) is taken from
 * `nominalFoothold` so that the clipping never moves a foot across the body.
 */
vector2_t clipFootholdToReach(const vector2_t& foothold,
                              const vector2_t& nominalFoothold,
                              const vector2_t& comAtTouchDown,
                              scalar_t yaw,
                              const ContactPlanningConfig& config);

/** Orbital energy of a 1D LIP, E = m (v^2 - omega^2 x^2) / 2, with x the CoM position relative to the ZMP. */
inline scalar_t lipOrbitalEnergy(scalar_t position, scalar_t velocity, scalar_t omega, scalar_t mass) {
  return 0.5 * mass * (velocity * velocity - omega * omega * position * position);
}

}  // namespace ocs2::humanoid
