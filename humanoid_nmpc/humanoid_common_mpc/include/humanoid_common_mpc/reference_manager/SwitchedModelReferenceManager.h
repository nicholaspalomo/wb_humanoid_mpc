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

  contact_flag_t getContactFlags(scalar_t time) const;

  bool isInStancePhase(scalar_t time) const { return (getContactFlags(time)[0] && getContactFlags(time)[1]); }

  bool isInContact(scalar_t time, size_t contactIndex) const { return getContactFlags(time)[contactIndex]; };

  void setArmSwingReferenceActive(bool armSwingReferenceActive) { armSwingReferenceActive_ = armSwingReferenceActive; }

  const std::shared_ptr<GaitSchedule>& getGaitSchedule() const { return gaitSchedulePtr_; }

  const std::shared_ptr<SwingTrajectoryPlanner>& getSwingTrajectoryPlanner() const { return swingTrajectoryPtr_; }

  scalar_t getPhaseVariable(scalar_t time) const;

  /** True when the mode schedule is produced by an online contact planner instead of the gait schedule. */
  virtual bool usesContactPlanning() const { return false; }

  /**
   * Task-space reference for a foot that is in swing at `time`, when a contact planner provides one. The default (gait
   * schedule based) reference manager has no foothold targets and returns an empty optional.
   */
  /**
   * Task-space reference of a foot in swing at `time`, or empty when nothing has an opinion about where it lands.
   *
   * The default is the nominal foothold of `model_settings.nominal_foothold.stepWidth`: the foot placed that far to
   * its own side of the reference base pose, swept from where that puts it at lift-off to where it puts it at
   * touch-down, so forward placement follows the commanded motion and only the lateral offset is stated. It is empty
   * when that width is zero, which is the default and which leaves the foot cost's xy position weights switched off
   * exactly as before.
   *
   * It exists for one configuration: the contact-implicit formulation without a contact planner. That formulation
   * removes the stance constraint that used to pin each foot where it landed, and without a planner nothing else
   * places the feet horizontally, so they drift together. A contact planner overrides this with its planned footholds
   * (ContactPlanningReferenceManager), which is the arrangement that needs no heuristic at all.
   */
  virtual std::optional<SwingFootReference> getSwingFootReference(size_t contactIndex, scalar_t time) const;

  /**
   * The nominal foothold of a foot at `time`: a whole step width to its own side of the **other, stance** foot,
   * carried forward at the operator's commanded velocity.
   *
   * Measured from the stance foot, which is the only landmark that holds still while this foot swings. Not from the
   * reference base, which is where the operator asked the robot to be rather than where it is - an offset between the
   * two walks the robot after its own foot targets, and a yaw error between them turns the lateral offset into a
   * longitudinal one, which is a turn. And not from the measured base either: in single support that sits roughly over
   * the stance foot, so offsets taken from it give half the intended separation and the feet converge.
   */
  std::optional<vector2_t> nominalFoothold(size_t contactIndex, scalar_t time) const;

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
   * calculator writes the momentum of the whole body turning about the vertical through its centre of mass,
   * h_z = I_zz * psidot / m, so the rate comes back out as m * h_z / I_zz with the same composite inertia - which is
   * why the yaw inertia is latched once per solve alongside the other measurements rather than being a constant.
   *
   * The target's base YAW is not usable for this. It carries the commanded heading, which the reference blends from
   * the measured one over the first stretch of the horizon and which the planned-heading override rewrites outright,
   * so differentiating it would recover the reference's own smoothing rather than the operator's command.
   */
  virtual scalar_t getCommandedYawRate(scalar_t time) const;

  /**
   * The divergent component of motion the reduced-order plan asks for at `time`, when a plan is active.
   *
   * The terminal DCM cost otherwise references the centre of the terminal support, i.e. "come to rest over the feet".
   * That is the right reference for a gait whose footholds are decided elsewhere, and the wrong one under a
   * reduced-order plan: the H-LIP's lateral orbit puts the DCM *beyond* the stance foot, towards the next foothold,
   * and a cost pulling it back onto the foot fights the very footholds the planner is placing. Empty when no plan is
   * active, and the support-centre reference then stands as before.
   */
  virtual std::optional<vector2_t> getPlannedDcm(scalar_t /*time*/, scalar_t /*omega*/) const { return std::nullopt; }

 protected:
  virtual void modifyReferences(scalar_t initTime,
                                scalar_t finalTime,
                                const vector_t& initState,
                                size_t initMode,
                                TargetTrajectories& targetTrajectories,
                                ModeSchedule& modeSchedule) override;

  // Adjusts the height of the target trajectories to current terrain height and returns that height.
  scalar_t adaptToCurrentGroundHeight(TargetTrajectories& targetTrajectories, const vector_t& initState, size_t initMode);
  scalar_t previousGroundHeightEstimate_{0.0};

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
  /// [kg] the robot's mass, and [kg m^2] the yaw component of its composite inertia about the centre of mass at the
  /// last solve. Together they invert the target's angular-momentum channel back into a commanded yaw rate.
  scalar_t totalMass_{0.0};
  scalar_t yawInertia_{0.0};
  /// [s] The window every per-node stance duty factor is measured over: the solver's own time horizon, latched in
  /// modifyReferences(). Fixed rather than "whatever is left until the last scheduled event", so that beta is a
  /// property of the gait and not of where in the horizon it is asked. 1 s until the first solve says otherwise.
  scalar_t dutyFactorWindow_{1.0};

  /** True while anything downstream needs captureMeasuredState() to do its work; see that function. */
  bool needsMeasuredState() const;
  /** The context the foothold heuristics are evaluated with, built from the last latched measurement. */
  FootholdHeuristicContext footholdContext(size_t contactIndex, scalar_t time) const;

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
