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

#include <mutex>
#include <optional>

#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * Reference manager that takes its mode schedule from an online contact plan instead of the periodic gait schedule.
 *
 * A plan is handed over from the planner thread with setContactPlan(). Before every solver run the newest plan is merged
 * with the schedule that was applied so far: modes within the commit window keep the applied schedule (so that phases
 * the MPC is already executing are not rewritten under its feet), later modes follow the plan, and the swing trajectory
 * planner is updated with the merged schedule. While no valid plan is available the gait schedule is used as before.
 *
 * The planned footholds are exposed to the foot tracking cost through getSwingFootReference(): during a swing phase the
 * xy reference interpolates smoothly from the lift-off position to the planned landing spot, the height follows the
 * swing trajectory planner.
 */
class ContactPlanningReferenceManager final : public SwitchedModelReferenceManager {
 public:
  ContactPlanningReferenceManager(std::shared_ptr<GaitSchedule> gaitSchedulePtr,
                                  std::shared_ptr<SwingTrajectoryPlanner> swingTrajectoryPtr,
                                  const PinocchioInterface& pinocchioInterface,
                                  const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                  ContactPlanningConfig config);
  ~ContactPlanningReferenceManager() override = default;

  bool usesContactPlanning() const override { return true; }
  std::optional<SwingFootReference> getSwingFootReference(size_t contactIndex, scalar_t time) const override;

  /** Hands a new plan over (thread-safe). It becomes active at the next solver run. */
  void setContactPlan(const ContactPlan& plan);

  /** The plan the current mode schedule was built from. Only meaningful on the solver thread. */
  const std::optional<ContactPlan>& getActiveContactPlan() const { return activePlan_; }
  bool hasActivePlan() const { return activePlan_.has_value() && activePlan_->valid; }

  /** The mode schedule most recently produced by modifyReferences(). */
  const ModeSchedule& getAppliedModeSchedule() const { return appliedSchedule_; }

  /**
   * Builds the planner input from the current state: CoM position / velocity, foot positions, contacts and phase timing
   * from the applied schedule, and the committed contacts of the commit window. Solver thread only.
   */
  ContactPlannerInput makePlannerInput(scalar_t initTime, const vector_t& initState, const vector2_t& velocityCommand);

  void setConfig(const ContactPlanningConfig& config);
  ContactPlanningConfig getConfig() const;

  /**
   * Time up to which the applied schedule is treated as fixed when planning from / merging at `time`: at least
   * `time + commitTime`, extended to the touch-down of any swing phase that has started or starts within that window, so
   * that a swing in flight or about to start is never re-timed or cut short by a later plan.
   */
  scalar_t commitBoundary(scalar_t time) const;

 protected:
  void modifyReferences(scalar_t initTime,
                        scalar_t finalTime,
                        const vector_t& initState,
                        size_t initMode,
                        TargetTrajectories& targetTrajectories,
                        ModeSchedule& modeSchedule) override;

 private:
  /** Foot positions from the state; latches the lift-off position of every foot while it is in contact. */
  void updateFootBookkeeping(scalar_t initTime, const vector_t& initState);
  feet_array_t<vector3_t> computeFootPositions(const vector_t& state);

  /** Swing phase [liftOff, touchDown] of a foot around `time` in the applied schedule, empty if the foot is in contact. */
  std::optional<std::pair<scalar_t, scalar_t>> swingPhase(size_t contactIndex, scalar_t time) const;

  mutable std::mutex configMutex_;
  ContactPlanningConfig config_;

  std::mutex planMutex_;
  std::optional<ContactPlan> pendingPlan_;  // written by the planner thread
  std::optional<ContactPlan> activePlan_;   // solver thread copy

  ModeSchedule appliedSchedule_;
  bool hasAppliedSchedule_ = false;

  feet_array_t<vector3_t> footPositions_{vector3_t::Zero(), vector3_t::Zero()};
  feet_array_t<vector3_t> liftOffPositions_{vector3_t::Zero(), vector3_t::Zero()};
  bool footBookkeepingInitialized_ = false;
};

}  // namespace ocs2::humanoid
