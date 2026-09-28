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
#include <limits>
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
 * Stance duty factor of `foot` at `time`: the fraction of ITS OWN STRIDE that the foot spends on the ground, in (0, 1].
 *
 * Bledt's beta (Appendix C, the impulse-scaling heuristic): a foot down for a fraction beta of its cycle must push
 * m*g/(F beta) while it is down for the feet between them to carry the weight over the cycle. The stride is measured
 * lift-off to lift-off - a swing and the stance that follows it - and the stride containing `time` is used. On any
 * periodic gait that is the gait's own duty factor at EVERY node, whichever foot and wherever in the cycle, which is
 * what makes the scaled reference average to exactly m*g. (It used to be the contact fraction over a window as long as
 * the MPC horizon. A horizon is not a whole number of strides in general - a 1.4 s walk stride against a 1.0 s horizon
 * - so that beta swung with the phase, the two feet saw different values at the same node, and the reference averaged
 * to 17% more than the weight.)
 *
 * Outside every complete stride of this foot - before its first lift-off, or after its last complete stride - it is
 * the nearest complete stride's value while a gait is under way, and 1 once the robot is standing: 1 before any foot
 * has lifted off, and after the last event of the whole schedule. 1 is "always down", the value at which the impulse
 * correction vanishes, and it is also what an empty schedule, or the default ModeSchedule() - one FLY mode and no
 * events - returns, so a caller never has to guard against a divide by zero.
 */
scalar_t stanceDutyFactor(const ModeSchedule& schedule, size_t foot, scalar_t time);

/**
 * [s] Duration of the stance `foot` begins by touching down at `touchDownTime`: from there to its next lift-off.
 *
 * If the schedule does not reach that lift-off, the duration of the foot's most recent complete stance before it; 0 if
 * the foot has none. The Raibert-style foothold heuristics scale with it, so that one coefficient means the same thing
 * on every gait the gait scheduler moves between.
 */
scalar_t upcomingStanceDuration(const ModeSchedule& schedule, size_t foot, scalar_t touchDownTime);

/**
 * Touch-down time of the most recent touch-down of `foot` at or before `time`, empty if the schedule has none. The
 * nominal foothold measures a step from the stance foot's own landing, which is a fixed instant, rather than from the
 * last solve, which moves.
 */
std::optional<scalar_t> previousTouchDownTime(const ModeSchedule& schedule, size_t foot, scalar_t time);

/**
 * Lift-off time of the swing of `foot` that is in flight at `time`, or of its next swing if the foot is in contact at
 * `time`. Empty if the schedule has no such lift-off event.
 */
std::optional<scalar_t> currentOrNextLiftOffTime(const ModeSchedule& schedule, size_t foot, scalar_t time);

/**
 * End of the window in which the executed `schedule` stays fixed: `time + commitTime`, extended to the touch-down of
 * every swing that overlaps the window, so that a swing in flight, or one that starts inside the window, is executed
 * to its end and never re-timed by a later plan.
 *
 * `maxCommitExtension` (<= 0: no cap) bounds that extension at `time + commitTime + maxCommitExtension`. The extension
 * walks the phases overlapping the window, and each swing it reaches pushes the boundary to its touch-down, which
 * brings the following phases into the walk. In a gait that exchanges support in a single instant every swing begins
 * exactly where the previous one ends, so the walk never terminates early: the boundary reaches the end of the
 * stepping region, no plan reaches past it, and the planner stops being able to publish (see PlannerSettings::
 * maxCommitExtension). A double support of any length breaks the chain, which is why an uncapped extension only fails
 * once the double supports are gone.
 */
scalar_t commitBoundaryForSchedule(const ModeSchedule& schedule, scalar_t time, scalar_t commitTime, scalar_t maxCommitExtension = 0.0);

/**
 * True if `plan` can be merged into the executed schedule `applied` at `time` without contradicting a swing that is
 * already in flight there: every foot that `applied` has in the air at `time` (lift-off strictly before `time`) is in
 * the air in the plan as well. A plan is made from a snapshot of the applied schedule and takes over the swings that
 * are committed up to its boundary; a swing activated between that snapshot and the plan's own activation is unknown to
 * the plan, and merging the plan over it would land the foot at the merge point and lift it again (a phantom micro
 * swing). A lift-off exactly at `time` is not executing yet and may be overruled by the plan.
 */
bool planAgreesWithSwingsInFlight(const ModeSchedule& applied, const ContactPlan& plan, scalar_t time);

/**
 * Contacts the planner has to keep fixed on its first nodes, taken from the executed schedule: every node that starts
 * before `committedUntil`, at most `maxNodes` of them. A node that lies entirely inside the window is sampled at its
 * midpoint; the last committed node, whether it straddles the boundary or ends exactly on it, is sampled at the boundary,
 * i.e. with the contact state the executed schedule hands over to the plan there. Sampling that node at its midpoint let
 * the planner contradict the executed schedule inside it (a touch-down after the midpoint read as "still swinging"),
 * which the merge then turned into a delayed touch-down or a phantom re-lift of the foot that had just landed.
 */
std::vector<contact_flag_t> committedContactsForPlanner(
    const ModeSchedule& schedule, scalar_t startTime, scalar_t dt, int maxNodes, scalar_t committedUntil);

/** Times at which committedContactsForPlanner() samples the executed schedule, one per committed node. */
std::vector<scalar_t> committedSampleTimes(scalar_t startTime, scalar_t dt, int maxNodes, scalar_t committedUntil);

/**
 * Start time of the contact phase every foot is in at `time`: the last event at or before `time` that changed the
 * foot's contact state, or -infinity when the schedule has no such event (the phase started before the schedule).
 */
feet_array_t<scalar_t> contactPhaseStartTimes(const ModeSchedule& schedule, scalar_t time);

/**
 * For every committed node (same sampling as committedContactsForPlanner) and foot, the time at which the foot entered
 * the contact state the node reports, taken from the executed schedule. The planner's grid is not aligned with the
 * executed events, so a phase that begins inside a committed node has lasted less than a whole node when the node ends
 * (a touch-down at 0.97 s inside the node [0.9, 1.0) has lasted 0.03 s at 1.0 s, not 0.1 s). Counting such a phase
 * from the node start credited it with up to a node it never had, and every minimum-duration rule that followed
 * (minimum double support, minimum contact and swing durations) was satisfied up to a node too early: the merged
 * schedule contained double supports and contact phases a small fraction of the configured minimum long.
 */
std::vector<feet_array_t<scalar_t>> committedPhaseStartsForPlanner(
    const ModeSchedule& schedule, scalar_t startTime, scalar_t dt, int maxNodes, scalar_t committedUntil);

/**
 * The last lift-off of every foot, remembered across executed schedules.
 *
 * The reference manager keeps the executed schedule only from one horizon before the solver time on (the merge's
 * lowerBoundTime), so about a horizon after the robot comes to stand the schedule no longer holds the robot's last
 * lift-off at all. The last swung foot used to be read off that schedule alone, and it then became -1 (unknown): the
 * H-LIP blend lost the zero-command orbit it measures the lateral velocity against in double support
 * (HlipContactPlanner::zeroCommandOrbitLateralVelocity), so the sway of settling counted in full towards walking, and
 * the first step out of the stand was chosen from the phase timing instead of alternating. The history is kept by
 * whoever builds planner inputs, independent of that window.
 */
struct LiftOffHistory {
  /** [s] the latest lift-off seen of each foot; -infinity until one has been seen. */
  feet_array_t<scalar_t> lastLiftOffTimes = makeFeetArray(-std::numeric_limits<scalar_t>::infinity());

  /**
   * Records every lift-off of `schedule` at or before `time` (an event at `time` itself has passed, as in
   * modeIndexAtTime). A recorded lift-off later than `time` is forgotten first: time ran backwards (a reset), and it
   * has not happened in the new run.
   */
  void record(const ModeSchedule& schedule, scalar_t time);

  /** The foot whose lift-off is the latest recorded (the right one when both lifted at once), or -1 when none is. */
  int lastSwungFoot() const;
};

/**
 * Fills the part of a planner input that is read off the executed schedule at `input.time`, the one definition of it
 * (ContactPlanningReferenceManager::makePlannerInput builds every input with it, the closed-loop planner tests theirs):
 * the contact state, how long each foot has been in it (kPhaseElapsedTimeBeforeTheSchedule when the schedule does not
 * reach back to its start), the last swung foot (from `liftOffHistory`, which it updates first), and the committed
 * window of at most `maxCommittedNodes` nodes of `dt` up to `input.committedUntil`, which the caller sets first.
 */
void fillPlannerInputFromSchedule(
    const ModeSchedule& schedule, scalar_t dt, int maxCommittedNodes, LiftOffHistory& liftOffHistory, ContactPlannerInput& input);

/** [s] the elapsed phase time fillPlannerInputFromSchedule() reports for a phase that began before the schedule. */
inline constexpr scalar_t kPhaseElapsedTimeBeforeTheSchedule = 10.0;

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
 * Adapts `schedule` to the measured contact state at `time` and applies the cadence shifts, one call per control cycle:
 * the schedule rules the configuration lists in `execution` (phase_resetting, energy_cadence_modulation), run through
 * the pipeline of execution/ScheduleAdaptationPipeline.h with the given cadence shifts. The reference manager runs the
 * same pipeline with the rules it owns; this entry point exists for the tests of the rules.
 *
 * @param schedule              the schedule the controller executes, modified in place
 * @param time                  current time
 * @param measuredContact       measured contact flags
 * @param cadenceTouchDownShift [s] per foot, desired touch-down shift relative to the nominal touch-down of the swing in
 *                              flight (used by the energy_cadence_modulation rule when listed)
 * @param config                the configuration: the execution list and the rules' parameter blocks
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
 * Clips `foothold` to the planner's reachable region of foot `contactIndex` around `comAtTouchDown` in the yaw frame:
 * |x| <= reachX and reachYInner <= side * y <= reachYOuter (the `reachability` term's block of the configuration), with
 * side +1 for the left foot (index 0) and -1 for the right foot, as in the planner's reachability rows, so that the
 * clipping never moves a foot across the body. `yaw` is the planned heading at touch-down (the frame those rows were
 * written in).
 */
vector2_t clipFootholdToReach(
    const vector2_t& foothold, size_t contactIndex, const vector2_t& comAtTouchDown, scalar_t yaw, const ContactPlanningConfig& config);

/** Orbital energy of a 1D LIP, E = m (v^2 - omega^2 x^2) / 2, with x the CoM position relative to the ZMP. */
inline scalar_t lipOrbitalEnergy(scalar_t position, scalar_t velocity, scalar_t omega, scalar_t mass) {
  return 0.5 * mass * (velocity * velocity - omega * omega * position * position);
}

}  // namespace ocs2::humanoid
