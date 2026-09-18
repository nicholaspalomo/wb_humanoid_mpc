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
 * The relaxed contact complementarity of one foot: g(x, u) = f_n(u) * h(x), the normal contact force times the height
 * of the foot above the terrain.
 *
 * This is what makes the whole-body MPC contact-implicit in the sense of "Reduced-Order Model Guided Contact-Implicit
 * Model Predictive Control for Humanoid Locomotion" (arXiv:2502.15630). The paper's CI-MPC decides contact implicitly
 * because its inverse-dynamics formulation makes the contact force a smooth function of the configuration under a
 * compliant contact model. Here the contact wrenches are decision variables instead, so the same freedom is obtained
 * by dropping the mode-scheduled hard constraints (ZeroWrenchConstraint on a swinging foot, the stance zero-velocity
 * constraint) and asking the optimizer for the complementarity condition of rigid contact directly:
 *
 *   f_n >= 0,   h >= 0,   f_n h = 0,
 *
 * of which the first is already enforced (the friction cone, or the non-negativity of the basis scalings), the second
 * is GroundPenetrationConstraint and the third is this term, penalised rather than imposed. A foot may then carry load
 * only where it touches the ground, and where it touches the ground it may carry load whatever the nominal gait says.
 * The contact schedule from the reduced-order planner survives only as a reference for the swing-foot cost, which is
 * exactly the role the paper gives it.
 *
 * The term is bilinear - the force enters linearly through the model's contact parameterization and the height enters
 * through the foot kinematics - so its linear approximation is assembled in closed form from the end-effector
 * kinematics and the constant row that maps the input to the normal force. No automatic differentiation is needed.
 *
 * The normal direction is the sole's, i.e. the third component of the model's contact force. For the flat terrain the
 * reduced-order planner assumes, that is the world vertical; on a tilted foot it is the physically correct normal.
 */
class ContactComplementarityConstraint final : public StateInputConstraint {
 public:
  /**
   * @param [in] endEffectorKinematics : kinematics of this foot's contact frame.
   * @param [in] mpcRobotModel : the robot model, for the input block that carries this contact's force.
   * @param [in] contactPointIndex : the contact this term belongs to.
   * @param [in] terrainHeight : [m] height of the ground under the foot.
   */
  ContactComplementarityConstraint(const EndEffectorKinematics<scalar_t>& endEffectorKinematics,
                                   const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                   size_t contactPointIndex,
                                   scalar_t terrainHeight = 0.0);

  ~ContactComplementarityConstraint() override = default;
  ContactComplementarityConstraint* clone() const override { return new ContactComplementarityConstraint(*this); }

  size_t getNumConstraints(scalar_t time) const override { return 1; }
  vector_t getValue(scalar_t time, const vector_t& state, const vector_t& input, const PreComputation& preComp) const override;
  VectorFunctionLinearApproximation getLinearApproximation(scalar_t time,
                                                           const vector_t& state,
                                                           const vector_t& input,
                                                           const PreComputation& preComp) const override;

  void setTerrainHeight(scalar_t terrainHeight) { terrainHeight_ = terrainHeight; }
  scalar_t getTerrainHeight() const { return terrainHeight_; }
  /** The constant row with f_n = normalForceRow . u; exposed for the tests. */
  const vector_t& getNormalForceRow() const { return normalForceRow_; }

 private:
  ContactComplementarityConstraint(const ContactComplementarityConstraint& rhs);

  std::unique_ptr<EndEffectorKinematics<scalar_t>> endEffectorKinematicsPtr_;
  size_t contactPointIndex_;
  scalar_t terrainHeight_;
  vector_t normalForceRow_;
};

/** The row that maps the input to the normal force of `contactPointIndex`, probed from the model's linear accessor. */
vector_t normalContactForceRow(const MpcRobotModelBase<scalar_t>& mpcRobotModel, size_t contactPointIndex);

}  // namespace ocs2::humanoid
