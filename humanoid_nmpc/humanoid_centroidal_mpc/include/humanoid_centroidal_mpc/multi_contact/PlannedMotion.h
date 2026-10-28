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

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_centroidal_mpc/multi_contact/PlanContactOrder.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact/ContactPatchFrame.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"

namespace ocs2::humanoid {

/** One contact phase of a plan for one MPC contact: when it rests, and on
 * which patch, in plan time. Passive data. */
struct PlannedContactPhase {
  scalar_t startTime = 0.0;  // [s] plan time
  scalar_t endTime = 0.0;    // [s] plan time, >= startTime
  // [m] The target pad center, in the world frame.
  vector3_t targetPosition = vector3_t::Zero();
  // The patch's outward unit normal, in the world frame.
  vector3_t normal = vector3_t::UnitZ();
  // The patch frame the contact terms are written in (getContactPatchFrame()):
  // world-aligned exactly when the normal is the world z axis.
  ContactPatchFrame patch;
};

/** An interval of plan time in which one contact is not in contact and at
 * whose end it touches down. Passive data. */
struct PlannedSwingWindow {
  scalar_t liftOffTime = 0.0;    // [s] plan time
  scalar_t touchDownTime = 0.0;  // [s] plan time, > liftOffTime
  // Whether the contact is in the air from the plan's start: it lifts off from
  // no patch.
  bool fromStart = false;
};

/** Whether a converted plan plays once or repeats as a period of its own
 * length (PlannedMotion::Create()). */
enum class PlanRepetition {
  // Plan time runs from the first knot to the last, and the plan holds there.
  kOnce,
  // The plan is one period, from its first knot to its last, which is the
  // first again (forwardAndReverseCycle(), PlanPlayback.h): plan time t is the
  // plan's at start + (t - start) mod period, every lookup shifted by the
  // whole periods.
  kPeriodic,
};

/**
 * A multi-contact plan (multi_contact_plan.proto) converted into the
 * centroidal MPC's terms, once, at load: its knots as states of the MPC's
 * layout (the normalized centroidal momentum, the base position and unit
 * quaternion, and the MPC's joints by name), its contact modes as MPC mode
 * numbers by name (PlanContactOrder), the plan times at which the stance
 * changes, and the contact phases of every MPC contact with their patch
 * frames. Times are the plan's own (its first knot's time is its start).
 * A periodic motion (PlanRepetition::kPeriodic) answers every lookup of plan
 * time - its modes, phases, swings, targets and wrenches - at any plan time,
 * before its start or periods past its end, through the period.
 *
 * A value type, immutable after Create(): every method is const and safe to
 * call concurrently.
 */
class PlannedMotion {
 public:
  /**
   * The motion of `plan` for the MPC whose state layout and contacts are
   * `model`'s, with the robot mass `robotMass` [kg] that normalizes the plan's
   * momenta, the contacts mapped by `order` (copied).
   *
   * @return InvalidArgument for a plan without knots, with times that do not
   * increase, with vectors of the wrong size, a quaternion that is not unit
   * (within 1e-6), a joint that is not one of the URDF, a joint the MPC fixes
   * at a position other than 0, an active contact the MPC does not have
   * (PlanContactOrder::modeOf()), a phase of an unknown contact or with a
   * target pad pose of the wrong size or a normal that is not unit (within
   * 1e-3), or an active contact at a knot that no phase of it covers; and,
   * under PlanRepetition::kPeriodic, for a plan of one knot or one whose last
   * knot's state or mode is not its first's.
   */
  static absl::StatusOr<PlannedMotion> Create(
      const multi_contact::config::MultiContactPlan& plan,
      const PlanContactOrder& order,
      const MpcRobotModelBase<scalar_t>& model,
      scalar_t robotMass,
      PlanRepetition repetition = PlanRepetition::kOnce);

  PlannedMotion(const PlannedMotion&) = default;
  PlannedMotion& operator=(const PlannedMotion&) = default;
  PlannedMotion(PlannedMotion&&) noexcept = default;
  PlannedMotion& operator=(PlannedMotion&&) noexcept = default;
  ~PlannedMotion() = default;

  /** [s] The plan times of the first and the last knot: of one period, for a
   * periodic motion. */
  scalar_t startTime() const { return targets_.timeTrajectory.front(); }
  scalar_t endTime() const { return targets_.timeTrajectory.back(); }
  /** [s] endTime() - startTime() for a periodic motion; std::nullopt for one
   * that plays once. */
  std::optional<scalar_t> period() const { return period_; }
  /** [s] The plan time within one period of `planTime`: `planTime` itself
   * for a motion that plays once, otherwise startTime() + (planTime -
   * startTime()) mod period(), in [startTime(), endTime()). */
  scalar_t cycleTime(scalar_t planTime) const;
  /** The number of whole periods from startTime() to `planTime`, rounded
   * down (negative before the start); 0 for a motion that plays once. */
  int64_t cycleIndex(scalar_t planTime) const;

  /** The knots in plan time, with zero inputs: the states of the MPC's layout
   * (of one period, for a periodic motion: interpolate them at cycleTime()).
   */
  const TargetTrajectories& targets() const { return targets_; }
  /**
   * The knots over the plan times [from, to], with the one at or before
   * `from` and the first at or after `to`, so that every time of the
   * interval interpolates between two of them; for a motion that plays once,
   * none before its start and only its last past its end. Linear in the knots
   * it returns, and logarithmic in the plan's.
   */
  TargetTrajectories targetsOver(scalar_t from, scalar_t to) const;
  /**
   * The plan's contact wrenches at plan time `planTime`, those of the last
   * knot at or before it, in the MPC's input layout (forces and moments in the
   * world frame, every other entry zero): the force reference of
   * multi_contact_plan.force_reference "planned_wrenches".
   */
  const vector_t& plannedWrenchesAt(scalar_t planTime) const;
  /** The MPC state of the first knot: the state the MPC and its dummy
   * simulator start from. */
  const vector_t& initialState() const { return targets_.stateTrajectory[0]; }

  /** [s] The plan times at which the stance changes, increasing: the knots
   * whose mode differs from the knot's before. */
  const std::vector<scalar_t>& eventTimes() const { return eventTimes_; }
  /** The MPC mode of each phase between the events, eventTimes().size() + 1 of
   * them: modeSequence()[k] before eventTimes()[k]. A periodic motion's first
   * and last are one. */
  const std::vector<size_t>& modeSequence() const { return modeSequence_; }
  /** The MPC mode at plan time `planTime`. */
  size_t modeAt(scalar_t planTime) const;
  /** The MPC mode before the `event`-th change of the plan's stance counted
   * from its start (PlanClock's events): modeSequence()[event], or for a
   * periodic motion modeSequence()[event mod eventTimes().size()]. */
  size_t modeBeforeEvent(size_t event) const;
  /** The mode schedule in plan time: the plan's for a motion that plays once;
   * for a periodic one, that of every period that overlaps [from, to]. */
  ModeSchedule modeScheduleOver(scalar_t from, scalar_t to) const;

  /** The phases of MPC contact `mpcIndex`, in time order; empty for a contact
   * the plan never puts down. */
  const std::vector<PlannedContactPhase>& phases(size_t mpcIndex) const {
    return phases_[mpcIndex];
  }
  /** The phase of MPC contact `mpcIndex` whose interval holds `planTime`. */
  std::optional<PlannedContactPhase> phaseAt(size_t mpcIndex,
                                             scalar_t planTime) const;
  /** The first phase of `mpcIndex` that starts after `planTime`. */
  std::optional<PlannedContactPhase> nextPhase(size_t mpcIndex,
                                               scalar_t planTime) const;
  /** The last phase of `mpcIndex` that ends at or before `planTime`. */
  std::optional<PlannedContactPhase> previousPhase(size_t mpcIndex,
                                                   scalar_t planTime) const;

  /** The swings of MPC contact `mpcIndex` by the mode schedule: every
   * interval it is out of contact and that ends in a touch-down, in time
   * order (of one period, for a periodic motion, where a swing across the
   * start lifts off at the previous period's last lift-off and none is
   * fromStart). */
  const std::vector<PlannedSwingWindow>& swingWindows(size_t mpcIndex) const {
    return swingWindows_[mpcIndex];
  }
  /** swingWindows() for a motion that plays once; for a periodic one, the
   * swings of every period that touch down at or after `from` and lift off at
   * or before `to`, in time order. */
  std::vector<PlannedSwingWindow> swingWindowsOver(size_t mpcIndex,
                                                   scalar_t from,
                                                   scalar_t to) const;

  const PlanContactOrder& order() const { return order_; }

  /** The patch frame of the patch with outward unit normal `normal` through
   * `point`: z the normal, x the world x axis made orthogonal to it (the world
   * y axis where the normal is near x), and the offset n . point. */
  static ContactPatchFrame patchFrameOf(const vector3_t& point,
                                        const vector3_t& normal);

 private:
  explicit PlannedMotion(PlanContactOrder order) : order_(std::move(order)) {}

  /** The swings of every contact from the mode schedule. */
  void buildSwingWindows();
  /** The phase of the cycle of `mpcIndex` holding the cycle time
   * `cycleTime`, unshifted. */
  std::optional<PlannedContactPhase> cyclePhaseAt(size_t mpcIndex,
                                                  scalar_t cycleTime) const;
  /** `phase` shifted by `cycles` periods. */
  PlannedContactPhase shifted(PlannedContactPhase phase, int64_t cycles) const;

  PlanContactOrder order_;
  // [s] Positive, the plan's duration, for a periodic motion.
  std::optional<scalar_t> period_;
  TargetTrajectories targets_;  // at least one knot, times increasing
  // Per knot of targets_, the plan's wrenches in the MPC's input layout.
  vector_array_t plannedWrenches_;
  std::vector<scalar_t> eventTimes_;
  std::vector<size_t> modeSequence_;  // eventTimes_.size() + 1 entries
  // One list per MPC contact, each in time order.
  std::vector<std::vector<PlannedContactPhase>> phases_;
  // One list per MPC contact, each in time order.
  std::vector<std::vector<PlannedSwingWindow>> swingWindows_;
};

}  // namespace ocs2::humanoid
