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
#include <string>
#include <utility>

#include <ocs2_core/thread_support/BufferedValue.h>

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/contact_planning/TargetContactPose.h"
#include "humanoid_common_mpc/contact_planning/execution/ExecutionContext.h"
#include "humanoid_common_mpc/contact_planning/execution/ExecutionRule.h"
#include "humanoid_common_mpc/contact_planning/problem/TermCollection.h"
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
 * Between plans the heuristics listed in the configuration's `execution` list are applied, as ExecutionRule terms built
 * by ContactPlanningTermFactory (phase_resetting, energy_cadence_modulation, dcm_step_adjustment) plus the
 * planned_heading_override this manager provides itself: they adapt the applied schedule to the measured contact state,
 * correct the landing targets and rewrite the target trajectory. Whenever an adaptation moves later events, the active
 * plan is shifted by the same amount so that the merge stays consistent, and an immediate re-plan is requested. The
 * core of this manager (activation, merge, swing trajectories, references, target poses) is not a rule.
 *
 * The planned footholds are exposed to the foot tracking cost through getSwingFootReference(): during a swing phase the
 * xy reference interpolates smoothly from the lift-off position to the planned landing spot, corrected by the listed
 * foothold rules; the height follows the swing trajectory planner.
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

  /**
   * The operator's target is kept as it arrives, before any execution rule rewrites it.
   *
   * planned_com_override replaces the momentum channel of the live target with the plan's own centre-of-mass velocity,
   * and that rewrite survives into the next solver run because a target is only replaced when a new one is published.
   * A command read back off the live target would therefore be the planner's own output one cycle later: at rest it
   * reads as zero however far the operator pushes the stick, the standing blend never crosses its half point, and the
   * robot never starts walking. These overrides keep an untouched copy for commandedVelocity() / commandedYawRate().
   */
  void setTargetTrajectories(const TargetTrajectories& targetTrajectories) override;
  void setTargetTrajectories(TargetTrajectories&& targetTrajectories) override;

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
  /**
   * Heading model: the whole-body heading handed to the planner is the angular centre of mass yaw when an evaluator is
   * given here, the base yaw otherwise. Solver thread only (set once at construction of the interface).
   */
  void setAngularCenterOfMass(std::shared_ptr<AngularCenterOfMass> acom) { acom_ = std::move(acom); }
  bool hasAngularCenterOfMass() const { return acom_ != nullptr; }
  /** Whole-body heading at `state`: the ACoM yaw with an evaluator, the base yaw otherwise. */
  scalar_t computeHeading(const vector_t& state) const;
  /**
   * The operator's commanded yaw rate as of the last modifyReferences(), read off the momentum channel of the target
   * trajectory: the target carries the command as the angular momentum of a rigid turn about the vertical,
   * h_z = I_zz omega / m (CentroidalMpcTargetTrajectoriesCalculator), which the heading override never touches.
   *
   * The target's base yaw is not used. Its first stretch integrates the average of the measured and the commanded yaw
   * rate, so differentiating it (as an earlier version did) commanded half the rate at the start of a turn and fed the
   * measured rate back in; and once the override has rewritten it, it carries the previous plan's rate, so any
   * shortfall of the plan against the command was re-issued as the next command and the turn decayed. Solver thread only.
   */
  scalar_t commandedYawRate() const { return commandedYawRate_; }

  /**
   * The operator's commanded CoM velocity as of the last modifyReferences(), read off the same momentum channel
   * before the execution rules run. The planner module must use this and not the target trajectory it could read
   * itself: planned_com_override replaces that channel with the plan's own CoM velocity, so reading it afterwards
   * feeds the planner its own output, which at rest reads as a zero command and stops the robot ever starting to
   * walk. Solver thread only.
   */
  const vector2_t& commandedVelocity() const { return commandedVelocity_; }
  /** The same command, for the terms that ask through the base class. Solver thread only. */
  vector2_t getCommandedVelocity(scalar_t /*time*/) const override { return commandedVelocity_; }
  /** The plan's own DCM, so that the terminal capturability cost aims where the footholds are going. Solver thread only. */
  std::optional<vector2_t> getPlannedDcm(scalar_t time, scalar_t omega) const override;

  /** True while a plan handed over by setContactPlan() has not been activated by the solver thread yet (thread-safe). */
  bool hasPendingPlan() const;

  /**
   * Plans dropped at activation instead of applied (thread-safe): stale ones, whose commit boundary had already passed
   * (the planner latency exceeded commitTime), and inconsistent ones, made before a swing that is in flight at their
   * merge point was committed. While plans keep being dropped the executed schedule runs out and the robot stops
   * walking; these counters are the only sign of it besides a rate-limited warning.
   */
  size_t numStalePlansDropped() const { return stalePlanCount_.load(); }
  size_t numInconsistentPlansDropped() const { return inconsistentPlanCount_.load(); }

  /**
   * Replaces the configuration (validated) and re-assembles the execution rules from its `execution` list. Called on
   * the solver thread (the parameter updater's pre-solve hook) or before the solver runs.
   */
  void setConfig(const ContactPlanningConfig& config);
  ContactPlanningConfig getConfig() const;

  /** The listed execution rules, in order (solver thread only). */
  const TermCollection<ExecutionRule>& getExecutionRules() const { return executionRules_; }
  /** One line per rule with its description, for the start-up print. */
  std::string executionSummary() const;

  /**
   * Time up to which the applied schedule is treated as fixed when planning from / merging at `time`: at least
   * `time + commitTime`, extended to the touch-down of any swing phase that has started or starts within that window, so
   * that a swing in flight or about to start is never re-timed or cut short by a later plan.
   */
  scalar_t commitBoundary(scalar_t time) const;

  /**
   * Hands the NMPC's latest predicted trajectory over (thread-safe). The rules that compare the measured centre of mass
   * with the prediction (cadence modulation, DCM step adjustment) read it, interpolated at the start of the next solve.
   * Empty arrays clear the prediction.
   */
  void setPredictedTrajectory(const scalar_array_t& times, const vector_array_t& states);

  /** Predicted CoM position / velocity at the current solver time, when a prediction covering it was available. */
  bool hasPredictedComState() const { return hasPredictedComState_; }
  const vector2_t& getPredictedComPosition() const { return predictedComState_[0]; }
  const vector2_t& getPredictedComVelocity() const { return predictedComState_[1]; }

  /** True once after a contact event re-timed the schedule (thread-safe); the planner module then plans immediately. */
  bool consumeReplanRequest() { return replanRequested_.exchange(false); }

  /**
   * Target contact pose of every foot as of the last solver run (thread-safe; for the MuJoCo viewer, see
   * TargetContactPose.h): the landing pose of the foot's swing in flight or of its next swing, or its placement. Every
   * pose is invalid while no plan is active.
   */
  feet_array_t<TargetContactPose> getTargetContactPoses() const;

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
  /** Builds the execution rules of the configuration's list (the heading override with this manager's model). */
  void rebuildExecutionRules(const ContactPlanningConfig& config);
  bool rulesNeedPredictedTrajectory() const;
  /** True when a listed rule reads the measured centre of mass of the cycle (which the prediction rules also do). */
  bool rulesNeedComState() const;

  /** Foot positions from the state; latches the lift-off position of every foot while it is in contact. */
  void updateFootBookkeeping(scalar_t initTime, const vector_t& initState);
  feet_array_t<vector3_t> computeFootPositions(const vector_t& state);
  /** Foot yaws from the frame placements computed by the last computeFootPositions() call. */
  feet_array_t<scalar_t> readFootYaws() const;
  /** Whole-body inertia about the vertical through the centre of mass at `state`. */
  scalar_t computeYawInertia(const vector_t& state);
  /**
   * Refreshes commandedVelocity_ and commandedYawRate_ from the momentum channel of the target at `initTime` (the yaw
   * rate with the yaw inertia at `initState`), before the execution rules are allowed to rewrite that channel.
   */
  void captureOperatorCommand(scalar_t initTime, const vector_t& initState);

  /** CoM position and velocity (xy) of the full model at `state`. */
  std::pair<vector2_t, vector2_t> computeComState(const vector_t& state);

  /** Swing phase [liftOff, touchDown] of a foot around `time` in the applied schedule, empty if the foot is in contact. */
  std::optional<std::pair<scalar_t, scalar_t>> swingPhase(size_t contactIndex, scalar_t time) const;

  /**
   * Takes a pending plan over, re-timed by the schedule shifts it has not seen. A plan whose commit boundary has already
   * passed, or that contradicts a swing in flight at its merge point, is dropped and the previous plan stays active.
   */
  void activatePendingPlan(scalar_t initTime);

  /** Interpolates the NMPC prediction at `initTime` and evaluates its CoM state; clears the flag if none covers it. */
  void updatePredictedComState(scalar_t initTime);

  /** Runs the schedule rules on the applied schedule; shifts the plan and requests a re-plan as needed. */
  void handleContactEvents(ExecutionContext& ctx);

  /** Updates the swing trajectory planner; a foot searching for the ground (a rule's ground search) gets a descending target. */
  void updateSwingTrajectories(const ModeSchedule& schedule, const ExecutionContext& ctx, scalar_t terrainHeight);

  /** Landing target offsets of the swings in flight from the foothold rules (zero without them). */
  void updateDcmStepAdjustment(const ExecutionContext& ctx);

  /** Snapshot of the target contact poses for getTargetContactPoses(), from the state of the current solver run. */
  void updateTargetContactPoses(scalar_t initTime, scalar_t terrainHeight);

  mutable std::mutex configMutex_;
  ContactPlanningConfig config_;
  TermCollection<ExecutionRule> executionRules_;  // solver thread

  mutable std::mutex planMutex_;
  std::optional<ContactPlan> pendingPlan_;  // written by the planner thread
  std::optional<ContactPlan> activePlan_;   // solver thread copy

  ModeSchedule appliedSchedule_;
  bool hasAppliedSchedule_ = false;
  scalar_t lastSolveTime_ = std::numeric_limits<scalar_t>::lowest();  // initTime of the last modifyReferences()
  std::atomic<size_t> stalePlanCount_{0};         // plans dropped because their commit boundary had passed (rate-limits the warning)
  std::atomic<size_t> inconsistentPlanCount_{0};  // plans dropped because a swing they did not know about was in flight (rate-limited)

  // Heading model.
  std::shared_ptr<AngularCenterOfMass> acom_;
  scalar_t totalMass_ = 0.0;
  scalar_t commandedYawRate_ = 0.0;                  // [rad/s] operator command, from the target's momentum channel
  vector2_t commandedVelocity_ = vector2_t::Zero();  // [m/s] operator command, from the same channel
  // The target as published, before any rule rewrote it. BufferedValue has no default constructor: it is seeded with
  // an empty target, which captureOperatorCommand() reads as "no command yet".
  BufferedValue<TargetTrajectories> operatorTarget_{TargetTrajectories()};
  feet_array_t<scalar_t> footYaws_ = makeFeetArray(0.0);
  feet_array_t<scalar_t> liftOffYaws_ = makeFeetArray(0.0);

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

  mutable std::mutex targetPoseMutex_;
  feet_array_t<TargetContactPose> targetContactPoses_ = makeFeetArray(TargetContactPose{});  // read by the control thread
};

}  // namespace ocs2::humanoid
