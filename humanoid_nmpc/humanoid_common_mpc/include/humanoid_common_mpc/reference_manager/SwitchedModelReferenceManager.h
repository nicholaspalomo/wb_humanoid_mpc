/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include <atomic>
#include <optional>

#include <ocs2_core/thread_support/Synchronized.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"

namespace ocs2::humanoid {

/** Task-space reference of a swing foot, provided by a contact planner. */
struct SwingFootReference {
  vector3_t position = vector3_t::Zero();
  vector3_t linearVelocity = vector3_t::Zero();
  std::optional<scalar_t> yaw;  // [rad] planned foot yaw, present with the contact planner's heading model
};

/**
 * Manages the ModeSchedule and the TargetTrajectories for switched model.
 */
class SwitchedModelReferenceManager : public ReferenceManager {
 public:
  SwitchedModelReferenceManager(std::shared_ptr<GaitSchedule> gaitSchedulePtr,
                                std::shared_ptr<SwingTrajectoryPlanner> swingTrajectoryPtr,
                                const PinocchioInterface& pinocchioInterface,
                                const MpcRobotModelBase<scalar_t>& mpcRobotModel);

  ~SwitchedModelReferenceManager() override = default;

  /** Disable copy / move */
  SwitchedModelReferenceManager& operator=(const SwitchedModelReferenceManager&) = delete;
  SwitchedModelReferenceManager(const SwitchedModelReferenceManager&) = delete;
  SwitchedModelReferenceManager& operator=(SwitchedModelReferenceManager&&) = delete;
  SwitchedModelReferenceManager(SwitchedModelReferenceManager&&) = delete;

  /**
   * Returns the manager to its state after construction, keeping its configuration (the ground, the locomotion
   * heuristics, the arm swing): the gait schedule restarts from its initial STANCE schedule (GaitSchedule::reset()),
   * and every measurement latched at an earlier solve - the lift-off positions of the feet, the measured base, CoM and
   * yaw inertia - is forgotten, so that nothing measured before a reset is used after it. The swing trajectories are
   * rebuilt from the schedule at every solve and carry nothing over. See ReferenceManagerInterface::reset().
   */
  void reset() override;

  contact_flag_t getContactFlags(scalar_t time) const;

  bool isInStancePhase(scalar_t time) const { return (getContactFlags(time)[0] && getContactFlags(time)[1]); }

  bool isInContact(scalar_t time, size_t contactIndex) const { return getContactFlags(time)[contactIndex]; };

  /**
   * Switches the procedural arm swing of getDesiredState() on or off. Off by default. The MPC interfaces switch it on
   * for a robot that names model_settings.armJointNames, except the centroidal MPC under com_and_acom_tracking_cost,
   * where the arm motion emerges from the ACoM cost and a generator driving the same joints would fight it.
   */
  void setArmSwingReferenceActive(bool armSwingReferenceActive) { armSwingReferenceActive_ = armSwingReferenceActive; }
  bool isArmSwingReferenceActive() const { return armSwingReferenceActive_; }

  /**
   * [m] Moves the ground: the height the swing trajectories lift off from and touch down on, the landing targets stand
   * on (ContactPlanningReferenceManager) and the commanded base height stands on. This manager owns the ground; it
   * starts at ModelSettings::terrainHeight, the task file's top-level `terrainHeight`, and the parameter updater moves
   * it here on a hot reload. The base-height reference follows it: see adaptToCurrentGroundHeight().
   *
   * Takes effect at the next preSolverRun(), which rebuilds the references on it; until then getAppliedTerrainHeight()
   * still returns the ground the references in use were built on. Thread-safe.
   */
  void setTerrainHeight(scalar_t terrainHeight) { terrainHeight_.store(terrainHeight); }

  /** [m] The ground as last set: what the next preSolverRun() builds the references on. */
  scalar_t getTerrainHeight() const { return terrainHeight_.load(); }

  /**
   * [m] The ground the references of the last preSolverRun() were built on - the swing trajectories and the landing
   * targets the solver is tracking right now - or the configured one before the first run. The
   * contact-implicit terms follow this value rather than getTerrainHeight(), so that they and the references the
   * solver tracks never describe two different grounds, not even for the one solve after a hot reload. Thread-safe.
   */
  scalar_t getAppliedTerrainHeight() const { return appliedTerrainHeight_.load(); }

  const std::shared_ptr<GaitSchedule>& getGaitSchedule() const { return gaitSchedulePtr_; }

  const std::shared_ptr<SwingTrajectoryPlanner>& getSwingTrajectoryPlanner() const { return swingTrajectoryPtr_; }

  scalar_t getPhaseVariable(scalar_t time) const;

  /** True when the mode schedule is produced by an online contact planner instead of the gait schedule. */
  virtual bool usesContactPlanning() const { return false; }

  /**
   * Task-space reference of a foot in swing at `time`, or empty when nothing has an opinion about where it lands: a
   * cubic sweep from where the foot actually lifted off to nominalFoothold() at its touch-down.
   *
   * Present when `model_settings.nominal_foothold.stepWidth` is positive, or when any foothold heuristic is listed;
   * empty otherwise - the default, which leaves the foot cost's xy position weights switched off exactly as before. A
   * contact planner overrides this with its planned footholds (ContactPlanningReferenceManager), and never calls
   * nominalFoothold(), which is why the foothold heuristics are rejected under one.
   */
  virtual std::optional<SwingFootReference> getSwingFootReference(size_t contactIndex, scalar_t time) const;

  /**
   * Where a foot touching down at `time` should land, or empty when nothing has an opinion. A LANDMARK: it does not
   * move while the foot swings, however many times the solver re-evaluates it - which it does at every solve, with
   * the latest measurement - because a target that slides through the swing is a target the foot never reaches.
   *
   * With no foothold heuristic listed, the nominal step: a whole step width to this foot's side of the **other,
   * stance** foot, carried forward at the commanded velocity over one step, from the stance foot's own touch-down to
   * this one. Measured from the stance foot because it is the one landmark that holds still while this foot swings;
   * over one step because that is how far the body travels between the two landings, which makes this the steady-state
   * Raibert target without a coefficient to tune.
   *
   * With foothold heuristics listed, Bledt's H_r: the hip at touch-down, plus the heuristics' offsets. The anchor is
   * the base PREDICTED at touch-down - measured at the solve and carried forward by the command - at the heading
   * predicted for then. Along that heading it is where the base will be; across it, the stance foot plus a step width,
   * so that the feet keep their separation (a stepWidth of zero is rejected at start-up unless hip_centered_stepping is
   * listed, which supplies the separation from the hips instead). With hip_centered_stepping listed the anchor is the
   * predicted base itself. Either way the heuristics' leads - translational_stepping's a1 v above all - are measured
   * from the hip at the moment of landing, as in the dissertation, and mean the same thing on both paths.
   */
  std::optional<vector2_t> nominalFoothold(size_t contactIndex, scalar_t time) const;

  /**
   * `xNominal` with only the locomotion heuristics' base-pose offsets applied: roll, pitch and height, at `time`.
   *
   * The single definition of the shaped base pose, used by every cost that regularizes the base pose - the running
   * state costs through getDesiredState(), the terminal cost and the torso task-space cost - so that none of them pulls
   * towards the unshaped pose while the others pull towards the shaped one. Returns `xNominal` untouched, bit for bit,
   * when no base-pose heuristic is listed.
   */
  vector_t shapeBasePose(scalar_t time, const vector_t& xNominal) const;

  /**
   * Task-space velocity reference for a foot that is in swing at `time`. The default reference manager returns the
   * commanded CoM velocity as a heuristic to prevent the swing foot from dragging behind the robot during locomotion.
   */
  virtual std::optional<vector2_t> getSwingFootVelocityReference(size_t contactIndex, scalar_t time) const;

  /**
   * World-frame normal of the plane the foot orientation cost is tracked against. Flat ground (0, 0, 1) for a foot in
   * contact and whenever swing_trajectory_config.swingPitchAngle is zero, which is the default; otherwise the ground
   * normal tilted back along the heading by the swing pitch, so that tracking it pitches the swing foot toe-up. The
   * heading is the planned foot yaw where a contact planner provides one, and the commanded base yaw otherwise.
   */
  vector3_t getSwingFootPlaneNormal(size_t contactIndex, scalar_t time) const;

  vector_t getDesiredState(const TargetTrajectories& targetTrajectories, const vector_t& state, scalar_t time) const;

  /**
   * The contact-force reference the input costs regularize against at `time`: weight compensation, shaped by any
   * listed wrench heuristic.
   *
   * It exists because the input channel of `targetTrajectories` is not a reference at all - it is constructed all-zero
   * and commented "they are not used", and both InputQuadraticCost and StateInputQuadraticCost ignore it and build
   * their own nominal input from weightCompensatingInput() instead. That nominal input is contact-flag dependent and
   * the stance set changes several times inside one horizon, which is why it cannot live in a three-knot trajectory
   * and why this manager - the only object that knows the contact flags at an arbitrary node time - is where it
   * belongs.
   *
   * With no wrench heuristic listed this returns exactly the vector those two costs built for themselves before,
   * computed by the same call, so routing them through here is a no-op on every robot shipped here.
   */
  vector_t getDesiredInput(const TargetTrajectories& targetTrajectories, const vector_t& state, scalar_t time) const;

  /**
   * Installs the reference-shaping layer of Bledt's Regularized Predictive Control heuristics
   * (humanoid_nmpc/docs/locomotion_heuristics/README.md).
   *
   * Called once by the MPC interface after the task file has been read, because the layer needs model constants that
   * are derived from the Pinocchio model this class is handed a copy of. Until it is called - and on every robot whose
   * task file lists no heuristic - the layer is empty and every seam below short-circuits, so the three references are
   * bit for bit what they were before the layer existed.
   *
   * The layer is shared rather than owned because the parameter updater needs to reach it to hot-reload the
   * coefficients, and because the reference manager is itself held by shared_ptr.
   */
  void setLocomotionHeuristicLayer(std::shared_ptr<LocomotionHeuristicLayer> layer);

  const std::shared_ptr<LocomotionHeuristicLayer>& getLocomotionHeuristicLayer() const { return heuristicLayerPtr_; }

  /**
   * The operator's commanded CoM velocity at `time`, for the terms that want the command itself rather than whatever
   * the reference currently asks for.
   *
   * Both are the linear part of the target's momentum channel here, which is why this is a single line. They part
   * company under online contact planning: the planned_com_override execution rule replaces that channel with the
   * reduced model's own CoM velocity, which swings from side to side with the gait, and a term that read the channel
   * directly would take that oscillation for an operator command. ContactPlanningReferenceManager therefore answers
   * this from the untouched copy of what the operator published.
   */
  virtual vector2_t getCommandedVelocity(scalar_t time) const;

  /**
   * [rad/s] The operator's commanded yaw rate at `time`, recovered from the angular channel of the target's momentum.
   *
   * The counterpart of getCommandedVelocity(), and needed for the same reason: two of Bledt's foothold heuristics are
   * functions of the turn rate and one of his force heuristics is a function of the turn rate and the speed together.
   *
   * The recovery is an inversion rather than a read, because the channel does not carry the rate. The target
   * calculator writes the momentum of the whole body turning about the vertical through its center of mass,
   * h_z = I_zz * psidot / m, so the rate comes back out as m * h_z / I_zz with the same composite inertia - which is
   * why the yaw inertia is latched once per solve alongside the other measurements rather than being a constant.
   *
   * The target's base YAW is not usable for this. It carries the commanded heading, which the reference blends from
   * the measured one over the first stretch of the horizon and which the planned-heading override rewrites outright,
   * so differentiating it would recover the reference's own smoothing rather than the operator's command.
   */
  virtual scalar_t getCommandedYawRate(scalar_t time) const;

  /** The divergent component of motion of a reduced-order plan, and the pendulum it belongs to. */
  struct PlannedDcm {
    vector2_t dcm = vector2_t::Zero();  // [m] xi = c + v / omega of the plan's center of mass
    scalar_t omega = 0.0;               // [1/s] the natural frequency of the pendulum the plan was made on
  };

  /**
   * The divergent component of motion the reduced-order plan asks for at `time`, when a plan is active, with the omega
   * of the pendulum the plan was made on.
   *
   * The terminal DCM cost otherwise references the center of the terminal support, i.e. "come to rest over the feet".
   * That is the right reference for a gait whose footholds are decided elsewhere, and the wrong one under a
   * reduced-order plan: the H-LIP's lateral orbit puts the DCM *beyond* the stance foot, towards the next foothold,
   * and a cost pulling it back onto the foot fights the very footholds the planner is placing. Empty when no plan is
   * active, and the support-center reference then stands as before.
   *
   * The omega travels with the DCM so that the cost can measure the robot's DCM on the same pendulum: a DCM of the
   * plan's center of mass taken with any other omega is not the plan's DCM (DcmTerminalCost::getParameters).
   */
  virtual std::optional<PlannedDcm> getPlannedDcm(scalar_t /*time*/) const { return std::nullopt; }

 protected:
  virtual void modifyReferences(scalar_t initTime,
                                scalar_t finalTime,
                                const vector_t& initState,
                                size_t initMode,
                                TargetTrajectories& targetTrajectories,
                                ModeSchedule& modeSchedule) override;

  /**
   * Reads the ground (getTerrainHeight()), records it as the applied ground (getAppliedTerrainHeight()) and returns it:
   * the height the swing trajectories and the landing targets of this preSolverRun() are built on.
   *
   * It also moves the base height of the target trajectories in use by the CHANGE of the ground since the last run, so
   * a hot reload of the ground moves the base reference once. Targets are world poses: a base height taken from the
   * measured state is a world height, and a commanded one is written above the ground by the target calculator
   * (TargetTrajectoriesCalculatorBase::commandedBaseHeight), which in the MPC reads getAppliedTerrainHeight() - the
   * ground this run applied - so a target built after the change already stands on the new ground and the next run
   * does not move it again.
   */
  scalar_t adaptToCurrentGroundHeight(TargetTrajectories& targetTrajectories, const vector_t& initState, size_t initMode);

  /**
   * Drops everything this manager carries from one solve to the next - the gait schedule, the latched measurements - but
   * not its configuration and not the references buffered in ReferenceManager: the part of reset() a derived manager
   * with state of its own extends (it overrides this and calls it).
   */
  virtual void resetRuntimeState();
  /// [m] The ground the target in use stands on: the task file's until the first run, the applied ground after it. It
  /// used to start at 0 whatever the task file said, so the first run lifted a target that was already a world pose -
  /// the initial and the reset targets, taken from the measured state - by the whole terrain height.
  scalar_t targetTerrainHeight_{0.0};
  /// [m] The ground, as set (setTerrainHeight) and as applied by the last preSolverRun(); both start at the task file's.
  std::atomic<scalar_t> terrainHeight_{0.0};
  std::atomic<scalar_t> appliedTerrainHeight_{0.0};

  PinocchioInterface pinocchioInterface_;
  const MpcRobotModelBase<scalar_t>* mpcRobotModelPtr_;
  ModeSchedule modeSchedule_;

  /** Records the measured base pose and, per foot in contact, where it currently stands (its lift-off position). */
  void captureMeasuredState(scalar_t initTime, const vector_t& initState);

  // Measured state of the last solver run, for the nominal foothold. Filled only while it is enabled, so a
  // configuration without it does exactly the work it did before.
  bool hasMeasuredState_{false};
  scalar_t lastSolveTime_{0.0};
  vector2_t measuredBasePosition_{vector2_t::Zero()};
  scalar_t measuredBaseYaw_{0.0};
  /// Where each foot was the last time it was measured in contact, i.e. where it lifted off from.
  feet_array_t<vector2_t> liftOffPositions_{makeFeetArray(vector2_t(vector2_t::Zero()))};
  /// [m/s] measured CoM linear velocity, and [m] measured CoM height above the mean foot height, at the last solve.
  /// Latched for the foothold heuristics, which are feedback laws on where the robot actually is rather than on the
  /// plan; filled only while a foothold heuristic is listed.
  vector2_t measuredComVelocity_{vector2_t::Zero()};
  scalar_t measuredComHeight_{0.0};
  /// [kg] the robot's mass, and [kg m^2] the yaw component of its composite inertia about the center of mass at the
  /// last solve. Together they invert the target's angular-momentum channel back into a commanded yaw rate.
  scalar_t totalMass_{0.0};
  scalar_t yawInertia_{0.0};
  /** True while anything downstream needs captureMeasuredState() to do its work; see that function. */
  bool needsMeasuredState() const;
  /**
   * The context the foothold heuristics are evaluated with for a touch-down at `time`: the base predicted for then
   * from the last latched measurement, the commands at `time`, and the stance the foot is about to begin.
   */
  FootholdHeuristicContext footholdContext(size_t contactIndex, scalar_t time) const;
  /** [m], [rad] The base position and yaw predicted at `time`: the last measurement carried forward by the command. */
  std::pair<vector2_t, scalar_t> predictedBaseAt(scalar_t time) const;

  /**
   * The reference-shaping layer of Bledt's heuristics. Empty - and therefore an exact no-op - until
   * setLocomotionHeuristicLayer() is called, and on every robot that lists none.
   */
  std::shared_ptr<LocomotionHeuristicLayer> heuristicLayerPtr_{std::make_shared<LocomotionHeuristicLayer>()};

  bool armSwingReferenceActive_{false};

  std::shared_ptr<GaitSchedule> gaitSchedulePtr_;
  std::shared_ptr<SwingTrajectoryPlanner> swingTrajectoryPtr_;
};

}  // namespace ocs2::humanoid
