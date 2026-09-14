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

#include <atomic>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>

#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
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
 * Between plans the applied schedule is adapted to the measured contact state (ContactScheduleAdaptation.h): a swing foot
 * that touches down early is switched to contact at once, a foot that misses the ground keeps searching for it for a
 * bounded time, and (optionally) the touch-down of the swing in flight is re-timed from the LIP orbital energy error.
 * Whenever such an adaptation moves later events, the active plan is shifted by the same amount so that the merge stays
 * consistent, and an immediate re-plan is requested from the planner module.
 *
 * The planned footholds are exposed to the foot tracking cost through getSwingFootReference(): during a swing phase the
 * xy reference interpolates smoothly from the lift-off position to the planned landing spot, corrected by the DCM error
 * propagated to touch-down (closed-form capture point step adjustment, clipped to the reachable region); the height
 * follows the swing trajectory planner.
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

  /**
   * Hands the NMPC's latest predicted trajectory over (thread-safe). The closed-form corrections (DCM step adjustment,
   * cadence modulation) measure the deviation of the measured centre of mass from this prediction, interpolated at the
   * start of the next solve, rather than from the planner's reduced model. Empty arrays clear the prediction.
   */
  void setPredictedTrajectory(const scalar_array_t& times, const vector_array_t& states);

  /** Predicted CoM position / velocity at the current solver time, when a prediction covering it was available. */
  bool hasPredictedComState() const { return hasPredictedComState_; }
  const vector2_t& getPredictedComPosition() const { return predictedComState_[0]; }
  const vector2_t& getPredictedComVelocity() const { return predictedComState_[1]; }

  /** True once after a contact event re-timed the schedule (thread-safe); the planner module then plans immediately. */
  bool consumeReplanRequest() { return replanRequested_.exchange(false); }

  // Introspection of the adaptive execution (solver thread only; for tests and telemetry).
  const feet_array_t<ContactEventReport>& getLastContactEvents() const { return lastContactEvents_; }
  const feet_array_t<SwingTimingLatch>& getSwingTimingLatches() const { return swingLatches_; }
  const feet_array_t<vector2_t>& getDcmStepAdjustment() const { return dcmStepAdjustment_; }
  const feet_array_t<scalar_t>& getCadenceTouchDownShift() const { return cadenceTouchDownShift_; }

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

  /** CoM position and velocity (xy) of the full model at `state`. */
  std::pair<vector2_t, vector2_t> computeComState(const vector_t& state);

  /** Swing phase [liftOff, touchDown] of a foot around `time` in the applied schedule, empty if the foot is in contact. */
  std::optional<std::pair<scalar_t, scalar_t>> swingPhase(size_t contactIndex, scalar_t time) const;

  /** Swaps a pending plan in, re-timed by the schedule shifts it has not seen. */
  void activatePendingPlan(scalar_t initTime);

  /** Interpolates the NMPC prediction at `initTime` and evaluates its CoM state; clears the flag if none covers it. */
  void updatePredictedComState(scalar_t initTime);

  /** Per-foot touch-down shift from the LIP orbital energy error (zero unless enabled and a plan is active). */
  feet_array_t<scalar_t> computeCadenceTouchDownShifts(scalar_t initTime, const ContactPlanningConfig& config);

  /** Adapts the applied schedule to the measured contact state; shifts the plan and requests a re-plan as needed. */
  void handleContactEvents(scalar_t initTime, size_t initMode, const ContactPlanningConfig& config);

  /** Updates the swing trajectory planner; a foot searching for the ground gets a descending touch-down height. */
  void updateSwingTrajectories(const ModeSchedule& schedule,
                               scalar_t initTime,
                               scalar_t terrainHeight,
                               const ContactPlanningConfig& config);

  /** DCM step adjustment of every swing foot from the DCM error with respect to the plan (zero unless enabled). */
  void updateDcmStepAdjustment(scalar_t initTime, const ContactPlanningConfig& config);

  mutable std::mutex configMutex_;
  ContactPlanningConfig config_;

  std::mutex planMutex_;
  std::optional<ContactPlan> pendingPlan_;  // written by the planner thread
  std::optional<ContactPlan> activePlan_;   // solver thread copy

  ModeSchedule appliedSchedule_;
  bool hasAppliedSchedule_ = false;
  scalar_t lastSolveTime_ = std::numeric_limits<scalar_t>::lowest();  // initTime of the last modifyReferences()
  size_t stalePlanCount_ = 0;                                         // solves that found the active plan stale (rate-limits the warning)

  feet_array_t<vector3_t> footPositions_ = makeFeetArray(vector3_t(vector3_t::Zero()));
  feet_array_t<vector3_t> liftOffPositions_ = makeFeetArray(vector3_t(vector3_t::Zero()));
  bool footBookkeepingInitialized_ = false;

  // Adaptive execution state (solver thread).
  feet_array_t<SwingTimingLatch> swingLatches_ = makeFeetArray(SwingTimingLatch{});
  feet_array_t<ContactEventReport> lastContactEvents_ = makeFeetArray(ContactEventReport{});
  feet_array_t<scalar_t> cadenceTouchDownShift_ = makeFeetArray(0.0);
  feet_array_t<vector2_t> dcmStepAdjustment_ = makeFeetArray(vector2_t(vector2_t::Zero()));
  vector2_t comState_[2] = {vector2_t::Zero(), vector2_t::Zero()};  // CoM position / velocity at the current solver run
  std::mutex predictionMutex_;
  scalar_array_t predictedTimes_;  // NMPC prediction handed over after the previous solve (written by the module)
  vector_array_t predictedStates_;
  bool hasPredictedComState_ = false;
  vector2_t predictedComState_[2] = {vector2_t::Zero(), vector2_t::Zero()};  // predicted CoM at the current solver run
  std::deque<std::pair<scalar_t, scalar_t>> scheduleShiftLog_;               // (time, shift) of every re-timing of later events
  std::atomic<bool> replanRequested_{false};
};

}  // namespace ocs2::humanoid
