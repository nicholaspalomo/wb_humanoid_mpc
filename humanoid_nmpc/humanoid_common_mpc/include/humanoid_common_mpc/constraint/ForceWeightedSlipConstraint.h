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

#include <memory>

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The no-slip and no-pivot condition of one foot, weighted by how hard the foot presses:
 * g(x, u) = f_n(u) * [v_x, v_y, omega_z](x, u).
 *
 * This is the contact-implicit replacement of the mode-scheduled stance constraint (the zero-velocity constraint that
 * is switched on for whichever foot the schedule calls a stance foot). A loaded foot must not move, an unloaded foot
 * may move freely, and nothing has to know in advance which of the two it is - which is what lets the whole-body MPC
 * put a foot down early, keep it down late, or leave it down altogether, as the reduced-order planner's nominal gait
 * is only a reference (see humanoid_nmpc/docs/contact_implicit_mpc/README.md).
 *
 * Three of the six components of the foot's twist are constrained: the two tangential linear velocities, and the
 * angular velocity about the contact normal. Those are the three a foot resting on the ground cannot have without
 * sliding or pivoting, and leaving the pivot out is not an option - the schedule-gated `zero_velocity` constraint this
 * term replaces was a full six-row twist constraint with `constrainOrientation`, and a loaded foot free to yaw walks
 * the robot sideways out from under itself.
 *
 * The other three are deliberately free. The normal velocity is left to ContactComplementarityConstraint, which
 * already forbids force at a height, and to GroundPenetrationConstraint, which forbids the foot going under the
 * terrain; constraining it here as well would forbid lift-off under load and reintroduce exactly the scheduling this
 * formulation removes. The two rocking rates are free for the same reason and for a second one: rolling the foot about
 * its heel and toe edges under load is how a heel-to-toe strike happens, which is one of the behaviours the
 * contact-implicit formulation exists to allow (arXiv:2502.15630, Fig. 3).
 *
 * Like the complementarity term this is bilinear in the force and the kinematics, so the linear approximation is
 * exact in closed form and needs no automatic differentiation.
 */
class ForceWeightedSlipConstraint final : public StateInputConstraint {
 public:
  /**
   * @param [in] endEffectorKinematics : kinematics of this foot's contact frame.
   * @param [in] mpcRobotModel : the robot model, for the input block that carries this contact's force.
   * @param [in] contactPointIndex : the contact this term belongs to.
   */
  ForceWeightedSlipConstraint(const EndEffectorKinematics<scalar_t>& endEffectorKinematics,
                              const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                              size_t contactPointIndex);

  ~ForceWeightedSlipConstraint() override = default;
  ForceWeightedSlipConstraint* clone() const override { return new ForceWeightedSlipConstraint(*this); }

  size_t getNumConstraints(scalar_t time) const override { return 3; }
  vector_t getValue(scalar_t time, const vector_t& state, const vector_t& input, const PreComputation& preComp) const override;
  VectorFunctionLinearApproximation getLinearApproximation(scalar_t time,
                                                           const vector_t& state,
                                                           const vector_t& input,
                                                           const PreComputation& preComp) const override;

 private:
  ForceWeightedSlipConstraint(const ForceWeightedSlipConstraint& rhs);

  std::unique_ptr<EndEffectorKinematics<scalar_t>> endEffectorKinematicsPtr_;
  size_t contactPointIndex_;
  vector_t normalForceRow_;
};

}  // namespace ocs2::humanoid
