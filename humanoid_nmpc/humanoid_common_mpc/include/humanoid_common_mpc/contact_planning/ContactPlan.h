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

#include <optional>
#include <string>
#include <vector>

#include <ocs2_core/Types.h>
#include <ocs2_core/reference/ModeSchedule.h>

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/** Snapshot of the robot and command state the contact planner plans from. All quantities are in the world frame. */
struct ContactPlannerInput {
  scalar_t time = 0.0;
  vector2_t comPosition = vector2_t::Zero();
  vector2_t comVelocity = vector2_t::Zero();
  scalar_t yaw = 0.0;  // base yaw, defines the planning frame
  feet_array_t<vector2_t> footPositions = makeFeetArray(vector2_t(vector2_t::Zero()));
  contact_flag_t contacts = makeFeetArray(true);                 // contact state of the phase active at `time`
  feet_array_t<scalar_t> phaseElapsedTime = makeFeetArray(0.0);  // [s] time already spent in that phase, per foot
  vector2_t velocityCommand = vector2_t::Zero();                 // commanded CoM velocity
  std::vector<contact_flag_t> committedContacts;                 // contacts imposed on the first nodes (commit window)
  // [s] per committed node and foot: the time the foot entered the contact state it has at that node, from the executed
  // schedule (committedPhaseStartsForPlanner). A phase that begins inside a committed node is then counted from its real
  // start instead of from the node start. Empty, or shorter than committedContacts: the remaining nodes count from
  // their node start.
  std::vector<feet_array_t<scalar_t>> committedPhaseStartTimes;
  int lastSwungFoot = -1;         // foot that swung most recently (-1 unknown), for alternation
  scalar_t committedUntil = 0.0;  // [s] the applied schedule is treated as fixed up to this time

  // Heading model (ContactPlanningConfig::useAcomDynamics). `yaw` then equals `heading`.
  scalar_t heading = 0.0;             // [rad] whole-body heading at planning time (ACoM yaw, or base yaw)
  scalar_t headingRate = 0.0;         // [rad/s] its rate: angular momentum about the vertical / yaw inertia
  scalar_t headingRateCommand = 0.0;  // [rad/s] commanded yaw rate
  // [kg m^2] the whole-body inertia about the vertical, filled from the robot model at every plan
  // (ContactPlanningReferenceManager::makePlannerInput reads I_zz out of pinocchio::ccrba, which is positive definite
  // for any physical model in any configuration). It must be strictly positive whenever the heading model is on:
  // LipContactPlanner::yawInertia throws on a non-positive value, plan() catches that and degrades to an invalid plan,
  // and the reference manager then keeps the previous schedule for that frame.
  //
  // This comment used to read "<= 0: configured value", promising a fallback that has never existed - there is no yaw
  // inertia anywhere in ContactPlanningConfig and none in any shipped contact_planning.yaml - and it contradicted the
  // two neighbouring contracts that state the real one, LipContactPlanner.h ("from the input (the robot model). Throws
  // if it is not positive") and ContactPlanningModelParameters.h ("taken from the model at every plan and is not part
  // of this"). A caller who believed it would leave the field at its default and silently get no plan at all.
  scalar_t yawInertia = 0.0;
  feet_array_t<scalar_t> footYaws = makeFeetArray(0.0);  // [rad] foot yaws at planning time, unwrapped near `heading`
};

/** Result of the contact planner: contact sequence, footholds and the reduced-model trajectories. */
struct ContactPlan {
  bool valid = false;
  scalar_t startTime = 0.0;
  scalar_t dt = 0.1;
  scalar_t committedUntil = 0.0;                   // [s] the plan honoured the applied schedule up to this time
  scalar_t yaw = 0.0;                              // [rad] base yaw at planning time; the geometry was planned in this frame
  std::vector<contact_flag_t> contacts;            // per interval k = 0..N-1
  std::vector<feet_array_t<vector2_t>> footholds;  // per node k = 0..N, planned foot xy (landing spot while swinging)
  std::vector<vector2_t> comPosition;              // per node
  std::vector<vector2_t> comVelocity;              // per node
  std::vector<vector2_t> zmp;                      // per interval
  // Heading model only (empty otherwise): per node k = 0..N.
  std::vector<scalar_t> heading;                 // [rad] whole-body heading
  std::vector<scalar_t> headingRate;             // [rad/s]
  std::vector<feet_array_t<scalar_t>> footYaws;  // [rad] planned foot yaw (the landing yaw while the foot swings)

  // Steps the planner had to cut to the reachable region. A clipped step is no longer the deadbeat step, so the
  // lateral error it was meant to cancel survives into the next step; when this is persistently non-zero the cadence
  // or the commanded velocity is asking for more than the legs can deliver, and the gait will diverge.
  int numClippedSteps = 0;

  // Solver statistics
  scalar_t objective = 0.0;
  int numBranchAndBoundNodes = 0;
  scalar_t solveTime = 0.0;
  bool optimal = false;
  bool nodeLimitHit = false;
  bool timeLimitHit = false;

  int numIntervals() const { return static_cast<int>(contacts.size()); }
  scalar_t endTime() const { return startTime + dt * static_cast<scalar_t>(contacts.size()); }

  /** Interval index containing `time`, clamped to the plan. */
  int intervalIndex(scalar_t time) const;

  /** Contact flags of the interval containing `time` (clamped). */
  contact_flag_t contactsAtTime(scalar_t time) const;

  /** Planned foot position of the node nearest to `time` (clamped). Empty when the plan is not valid. */
  std::optional<vector2_t> footholdAtTime(size_t contactIndex, scalar_t time) const;

  /**
   * Planned foot position of the node BEFORE the one nearest `time` (clamped) - the stance a foot leaves when it lifts
   * off at `time`.
   *
   * This exists because "the node before the lift-off" cannot be expressed by nudging the argument of footholdAtTime().
   * That function rounds to the nearest node, so `liftOffTime - 0.5 * dt` rounds back UP to the lift-off node itself
   * (`lround(k - 0.5) == k`), and any smaller nudge is a guess about where the lift-off sits between two nodes.
   *
   * Reading the lift-off node instead of the one before it is not a harmless rounding difference, because the two
   * planners disagree about what the lift-off node holds. Under `lip_miqp` the foothold state has not jumped yet, so
   * node k is still the stance. Under `hlip` - the shipped default - HlipContactPlanner computes the landing spot of a
   * swing BEFORE stamping `plan.footholds` over every node of that single-support phase, so node k already carries the
   * LANDING position. Reading it there makes a swing's start equal its target, which collapses the whole swing
   * reference to a constant with zero commanded velocity.
   *
   * Node k-1 is a contact node for the foot under BOTH planners, so this one accessor is correct for both. The nearest
   * node is resolved with the same rounding rule HlipContactPlanner uses for its phase boundaries, so a lift-off that
   * does not sit exactly on the node grid still resolves to a node the foot is standing on.
   */
  std::optional<vector2_t> footholdBeforeTime(size_t contactIndex, scalar_t time) const;

  /**
   * One line for the log (planner.logPlans): validity, search statistics (objective, relaxations, solve time, whether
   * the node or time limit cut the search), the CoM velocity at the start and end of the horizon, and per foot the
   * phase sequence with its durations and the length of every step (foothold displacement over a swing, planning frame).
   *
   * A step is measured from the last node the foot was still standing on to its touch-down node, so it is the same
   * quantity under both planners even though they disagree about what the lift-off node itself holds (see
   * footholdBeforeTime above). A swing that is already in flight at the start of the plan has no such node, so it
   * prints its duration and no displacement rather than a zero that would look like a step in place.
   */
  std::string describe() const;
  bool hasHeading() const { return valid && !heading.empty(); }
  /** Planned heading / heading rate at `time`, linearly interpolated between nodes and clamped to the plan; empty without the heading
   * model. */
  std::optional<scalar_t> headingAtTime(scalar_t time) const;
  std::optional<scalar_t> headingRateAtTime(scalar_t time) const;
  /** Planned foot yaw at the node nearest to `time` (the landing yaw while the foot swings); empty without the heading model. */
  std::optional<scalar_t> footYawAtTime(size_t contactIndex, scalar_t time) const;

  /** Planned foot yaw of the node BEFORE the one nearest `time`; the yaw counterpart of footholdBeforeTime(). */
  std::optional<scalar_t> footYawBeforeTime(size_t contactIndex, scalar_t time) const;

  /**
   * The reduced model's centre of mass at `time`, linearly interpolated between nodes and clamped to the plan.
   *
   * This is the trajectory the footholds were planned for: with the H-LIP planner it is the orbit the deadbeat step
   * regulates to, whose lateral part deliberately falls towards the swing foot. The whole-body MPC has to be asked for
   * that motion, or its own centre-of-mass reference and the planner's footholds pull in opposite directions
   * (planned_com_override, humanoid_nmpc/docs/hlip_contact_planner/README.md).
   */
  std::optional<vector2_t> comPositionAtTime(scalar_t time) const;
  std::optional<vector2_t> comVelocityAtTime(scalar_t time) const;

  /**
   * Moves the whole plan by `shift` seconds (start time and commit boundary). Used when the executed schedule is re-timed
   * after the plan was made (late touch-down, cadence modulation) so that the plan's later events keep their timing
   * relative to the re-timed switch.
   */
  void shiftInTime(scalar_t shift);

  /**
   * Converts the plan to a mode schedule. Event times are placed at the node times where the contact set changes. After the
   * planning horizon the schedule continues with the last planned mode, or with STANCE if the last mode is not double support.
   */
  ModeSchedule toModeSchedule() const;
};

/**
 * Merges the committed part of the currently applied schedule with a new plan.
 *
 * Modes strictly before `commitTime` are taken from `applied`, modes from `commitTime` on from `plan`. The result covers at
 * least [lowerBoundTime, upperBoundTime], always starts and ends with a STANCE mode and always has at least one event, so
 * that the swing trajectory planner finds a lift-off and a touch-down for every swing phase.
 */
ModeSchedule mergeModeSchedules(
    const ModeSchedule& applied, const ModeSchedule& plan, scalar_t commitTime, scalar_t lowerBoundTime, scalar_t upperBoundTime);

}  // namespace ocs2::humanoid
