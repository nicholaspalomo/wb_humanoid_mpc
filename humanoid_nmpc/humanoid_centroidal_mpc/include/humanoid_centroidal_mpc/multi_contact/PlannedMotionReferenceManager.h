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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_centroidal_mpc/multi_contact/MultiContactPlanSource.h"
#include "humanoid_centroidal_mpc/multi_contact/PlanClock.h"
#include "humanoid_centroidal_mpc/multi_contact/PlannedMotion.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/contact/ContactPatchFrame.h"
#include "humanoid_common_mpc/contact_swing/ContactSwingPlanner.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_multi_contact/contact_set/ContactSet.h"

namespace ocs2::humanoid {

/**
 * The reference manager of the centroidal MPC tracking an offline
 * multi-contact plan (contact_schedule_source: "multi_contact_plan"; design
 * section 7.2, humanoid_nmpc/docs/multi_contact_planning/README.md). A plan
 * played "forward_and_reverse" (section 7.3) is tracked as the periodic
 * motion of its cycle (forwardAndReverseCycle()): every lookup below maps the
 * plan time through the period. Before every solve it
 *  - advances the plan's clock (PlanClock): the plan starts at the first solve
 *    and is held before a contact change until the robot has reached the stance
 *    it is on - every contact of it on its patch within the contact tolerance,
 *    and the center of mass within the CoM tolerance of the plan's - so that no
 *    contact change is ever skipped;
 *  - provides the mode schedule of the plan's stances, by contact name in the
 *    MPC's contact order (PlanContactOrder), shifted to the solve's time;
 *  - provides the target trajectories of the plan's states, with the
 *    quaternion base, shifted the same way;
 *  - rebuilds the swings of every contact that leaves one patch for another
 *    (ContactSwingPlanner): from where it lifted off, along the patch normal it
 *    leaves, clear of the terrain, to its target pad along the target patch's
 *    normal; a contact that starts the plan in the air reaches its first
 *    target along that target's normal;
 *  - and answers the patch every contact rests on, or swings to, from the
 *    plan's phases (getContactPatchFrame()).
 * The velocity command and the gait schedule are not read: the plan is the
 * reference. Not thread-safe for its non-const methods, as its base.
 */
class PlannedMotionReferenceManager final
    : public SwitchedModelReferenceManager {
 public:
  /**
   * The manager tracking `source`'s plan for the MPC of `mpcRobotModel`
   * (retained; it must outlive the manager), with a copy of
   * `pinocchioInterface`, the MPC's model. `gaitSchedulePtr` and
   * `swingTrajectoryPtr` are the base's (SwitchedModelReferenceManager), which
   * this manager does not read.
   *
   * A std::shared_ptr because OCS2's solver holds its reference manager by
   * one (SolverBase::setReferenceManager()).
   *
   * @return PlannedMotion::Create()'s errors (an active contact the MPC does
   * not have, a joint the MPC fixes, a malformed knot or phase) and
   * InvalidArgument for a contact set of the MPC that cannot be mapped.
   */
  static absl::StatusOr<std::shared_ptr<PlannedMotionReferenceManager>> Create(
      std::shared_ptr<GaitSchedule> gaitSchedulePtr,
      std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPtr,
      const PinocchioInterface& pinocchioInterface,
      const MpcRobotModelBase<scalar_t>& mpcRobotModel,
      MultiContactPlanSource source,
      const PlannedMotionSettings& settings);

  ~PlannedMotionReferenceManager() override = default;
  PlannedMotionReferenceManager(const PlannedMotionReferenceManager&) = delete;
  PlannedMotionReferenceManager& operator=(
      const PlannedMotionReferenceManager&) = delete;
  PlannedMotionReferenceManager(PlannedMotionReferenceManager&&) = delete;
  PlannedMotionReferenceManager& operator=(PlannedMotionReferenceManager&&) =
      delete;

  /**
   * The patch of the plan's phase of contact `contactIndex` at `time`: the
   * phase it rests in, or the next one it swings to, or the last one it left;
   * the base's world-aligned patch for a contact the plan never puts down.
   * Thread-safe during a solve.
   */
  ContactPatchFrame getContactPatchFrame(size_t contactIndex,
                                         scalar_t time) const override;

  /**
   * The base's weight share, or under multi_contact_plan.force_reference
   * "planned_wrenches" the plan's wrench of every contact in contact at
   * `time`, at the plan time of the last solve's offset
   * (PlannedMotion::plannedWrenchesAt()); the joint velocities' entries are
   * the base's. Thread-safe during a solve.
   */
  vector_t getDesiredInput(const TargetTrajectories& targetTrajectories,
                           const vector_t& state,
                           scalar_t time) const override;

  const ContactSwingPlanner* absl_nullable getContactSwingPlanner()
      const override {
    return &swingPlanner_;
  }

  /** The plan, converted. */
  const PlannedMotion& plannedMotion() const { return motion_; }
  /** The scenario: its contact set and terrain. */
  const multi_contact::LoadedScenario& scenario() const { return scenario_; }
  const PlannedMotionSettings& settings() const { return settings_; }
  /** The plan's clock as of the last solve. */
  const PlanClock& clock() const { return clock_; }
  /** [s] The plan time of `time` at the last solve's offset. */
  scalar_t planTimeAt(scalar_t time) const { return time - clock_.offset(); }
  /** The candidate of the plan's contact set that MPC contact `contactIndex`
   * is, or std::nullopt when it is none. */
  std::optional<multi_contact::ContactCandidate> candidateOf(
      size_t contactIndex) const;

  /** [m] The world positions of the MPC's contact frames and the center of
   * mass at the last solve's state. */
  const std::vector<vector3_t>& measuredContactPositions() const {
    return measuredContactPositions_;
  }

 protected:
  void modifyReferences(scalar_t initTime,
                        scalar_t finalTime,
                        const vector_t& initState,
                        size_t initMode,
                        TargetTrajectories& targetTrajectories,
                        ModeSchedule& modeSchedule) override;
  /** The base's, and the lift-off positions; the plan's clock keeps running. */
  void resetRuntimeState() override;

 private:
  PlannedMotionReferenceManager(
      std::shared_ptr<GaitSchedule> gaitSchedulePtr,
      std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPtr,
      const PinocchioInterface& pinocchioInterface,
      const MpcRobotModelBase<scalar_t>& mpcRobotModel,
      PlannedMotion motion,
      multi_contact::LoadedScenario scenario,
      const PlannedMotionSettings& settings);

  /** Measures the contact frames and the center of mass at `state`. */
  void measure(const vector_t& state);
  /** [m] The center of mass of the MPC state `state`. */
  vector3_t centerOfMassOf(const vector_t& state);
  /** Whether the stance before the plan's change `event` is reached at the
   * measured state, against the plan at `holdTime`. */
  bool stanceReached(size_t event, scalar_t holdTime);
  /** The plan's mode schedule at the clock's offset: all of it, or a periodic
   * plan's over [initTime, finalTime] and the swings' margin. */
  ModeSchedule modeScheduleAtOffset(scalar_t initTime,
                                    scalar_t finalTime) const;
  /** The plan's knots over [initTime, finalTime] at the clock's offset. */
  TargetTrajectories targetsAtOffset(scalar_t initTime,
                                     scalar_t finalTime) const;
  /** Rebuilds the swings of every contact over [initTime, finalTime]. */
  void updateSwings(scalar_t initTime, scalar_t finalTime);
  /** The swings of MPC contact `contactIndex` over the window: an arc each, or,
   * longer than PlannedMotionSettings::arcSwingDurationLimit when that is
   * positive, the plan's pad from knot to knot (plannedPathSwings()). */
  std::vector<ContactSwingTrajectory> swingsOf(size_t contactIndex,
                                               scalar_t initTime,
                                               scalar_t finalTime);
  /** [m] Where the plan has MPC contact `contactIndex`'s frame at plan time
   * `planTime`; uses the Pinocchio data. */
  vector3_t plannedContactPosition(size_t contactIndex, scalar_t planTime);
  /** Appends to `swings` the straight segments of MPC contact `contactIndex`'s
   * pad, where the plan has it, between the plan's knots in its swing `window`
   * (plan times, the clock at `offset`) that overlap [from, to] (MPC
   * times). */
  void plannedPathSwings(size_t contactIndex,
                         const PlannedSwingWindow& window,
                         scalar_t offset,
                         scalar_t from,
                         scalar_t to,
                         std::vector<ContactSwingTrajectory>& swings);

  const PlannedMotion motion_;
  const multi_contact::LoadedScenario scenario_;
  const PlannedMotionSettings settings_;
  PlanClock clock_;
  ContactSwingPlanner swingPlanner_;
  // The frames of the MPC's contacts in the Pinocchio model.
  std::vector<pinocchio::FrameIndex> contactFrames_;
  // At the last solve's state: the contact frames' and the CoM's positions.
  std::vector<vector3_t> measuredContactPositions_;
  vector3_t measuredCenterOfMass_ = vector3_t::Zero();
  // Where each contact was the last time it was in contact (or, for one that
  // starts the plan in the air, where it was at the first solve); empty
  // before.
  std::vector<std::optional<vector3_t>> liftOffPositions_;
};

}  // namespace ocs2::humanoid
