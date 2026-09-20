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
 * WHERE AND IN WHICH FRAME the twist is measured, since both have been raised as defects and the answers differ.
 *
 * The FRAME is the world's, and that is correct rather than an oversight. The contact whose slip this prices is
 * between the foot and the GROUND, and every layer of this stack assumes the ground is flat - `terrainHeight` is a
 * single configured scalar, and the reduced-order planner assumes it most of all. So the contact normal is the world
 * vertical no matter how the foot is oriented, `velocity.head<2>()` really is the tangential velocity in the contact
 * plane, and `angularVelocity(2)` really is the spin about the contact normal. Rotating the twist into the FOOT's
 * frame, as one might expect from the name "contact frame", would be the bug: it would price the foot's own pitch
 * and roll as tangential slip. On sloped terrain this reasoning stops holding and the twist would have to be rotated
 * into the terrain frame - but so would the wrench cone, the penetration hinge and the swing planner, and none of
 * them support a slope either.
 *
 * The POINT is the contact frame, which `contact_frame_translation` places on the sole. That is an approximation,
 * and unlike the frame question it is a real one: when the foot rocks about an edge, the sole centre moves even
 * though the contact line does not, and the term charges for slip that is not happening. Because the contact frame
 * sits ON the sole, the lever arm is a * sin(theta) rather than a fixed offset, so the coupling vanishes at a flat
 * foot and grows with the tilt: v_x = omega_y * a * sin(theta) with a = 0.12 m on this robot. At the shipped 0.08 rad
 * of swing pitch and 1 rad/s that is 9.6 mm/s, a residual of 0.032 and a cost of 0.077 at full load - negligible. At
 * an aggressive 0.3 rad heel-to-toe roll at 2 rad/s it is 71 mm/s, a residual of 0.24 and a cost of 4.2, which is no
 * longer negligible and does resist the roll. Measuring the twist at the softmin-weighted contact point, the way
 * FootprintCornerHeights already measures the gap, would remove it; that is a change to the closed-loop behaviour and
 * belongs in its own validated step rather than folded into a tuning pass. testRelaxedContactConstraints pins the
 * magnitude so the trade-off stays visible.
 *
 * Like the complementarity term this is bilinear in the force and the kinematics, so the linear approximation is
 * exact in closed form and needs no automatic differentiation.
 *
 * The residual is normalised, (f_n / f_ref) (v / v_ref), for the reason given at length on
 * ContactComplementarityConstraint: the penalty is quadratic in the product, so the curvature it puts on the foot
 * velocity is w f_n^2 / f_ref^2 / v_ref^2, and unnormalised that spans nine orders of magnitude between a foot in
 * flight and a foot carrying the robot, which no single weight can span.
 *
 * Normalising also repairs a second problem the three rows had in common. Two of them are linear velocities in m/s
 * and the third is a yaw rate in rad/s, and a quadratic penalty over the un-normalised vector adds their squares
 * together - so the weight silently declared one radian per second to be as bad as one metre per second, a ratio that
 * has no physical justification and changes meaning with the size of the foot. Each row is now divided by a reference
 * in its own units before the squares are summed, so the weight applies to three comparable dimensionless numbers.
 */
class ForceWeightedSlipConstraint final : public StateInputConstraint {
 public:
  /**
   * @param [in] endEffectorKinematics : kinematics of this foot's contact frame.
   * @param [in] mpcRobotModel : the robot model, for the input block that carries this contact's force.
   * @param [in] contactPointIndex : the contact this term belongs to.
   * @param [in] forceReference : [N] the normal force the residual is measured in, normally the robot's weight.
   * @param [in] velocityReference : [m/s] the sliding speed the two tangential rows are measured in.
   * @param [in] angularVelocityReference : [rad/s] the pivot rate the yaw row is measured in.
   */
  ForceWeightedSlipConstraint(const EndEffectorKinematics<scalar_t>& endEffectorKinematics,
                              const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                              size_t contactPointIndex,
                              scalar_t forceReference = 1.0,
                              scalar_t velocityReference = 1.0,
                              scalar_t angularVelocityReference = 1.0);

  ~ForceWeightedSlipConstraint() override = default;
  ForceWeightedSlipConstraint* clone() const override { return new ForceWeightedSlipConstraint(*this); }

  size_t getNumConstraints(scalar_t time) const override { return 3; }
  vector_t getValue(scalar_t time, const vector_t& state, const vector_t& input, const PreComputation& preComp) const override;
  VectorFunctionLinearApproximation getLinearApproximation(scalar_t time,
                                                           const vector_t& state,
                                                           const vector_t& input,
                                                           const PreComputation& preComp) const override;

  scalar_t getForceReference() const { return 1.0 / inverseForceReference_; }
  /** The per-row reciprocal scales, in the row order [v_x, v_y, omega_z]; exposed for the tests. */
  const vector3_t& getInverseTwistReference() const { return inverseTwistReference_; }
  /** Retunes the two twist references, so that the keys are live in the tuning dashboard like the weight beside them. */
  void setTwistReferences(scalar_t velocityReference, scalar_t angularVelocityReference) {
    inverseTwistReference_ << 1.0 / velocityReference, 1.0 / velocityReference, 1.0 / angularVelocityReference;
  }

 private:
  ForceWeightedSlipConstraint(const ForceWeightedSlipConstraint& rhs);

  std::unique_ptr<EndEffectorKinematics<scalar_t>> endEffectorKinematicsPtr_;
  size_t contactPointIndex_;
  scalar_t inverseForceReference_;
  vector3_t inverseTwistReference_;
  vector_t normalForceRow_;
};

}  // namespace ocs2::humanoid
