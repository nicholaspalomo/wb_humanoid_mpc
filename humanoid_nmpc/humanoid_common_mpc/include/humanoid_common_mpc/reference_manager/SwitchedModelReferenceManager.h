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
  virtual std::optional<SwingFootReference> getSwingFootReference(size_t /*contactIndex*/, scalar_t /*time*/) const { return std::nullopt; }

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

  bool armSwingReferenceActive_{false};

  std::shared_ptr<GaitSchedule> gaitSchedulePtr_;
  std::shared_ptr<SwingTrajectoryPlanner> swingTrajectoryPtr_;
};

}  // namespace ocs2::humanoid
