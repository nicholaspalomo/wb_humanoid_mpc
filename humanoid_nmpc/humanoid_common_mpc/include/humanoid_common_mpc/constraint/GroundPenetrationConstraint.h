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

#include <ocs2_core/constraint/StateConstraint.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The unilateral side of the contact condition: g(x) = h(x) - terrainHeight >= 0, the foot may not go through the
 * ground.
 *
 * With the mode-scheduled stance constraint gone (see ForceWeightedSlipConstraint), nothing else holds a foot above
 * the terrain: the complementarity term only forbids force at a height, and a foot pushed below the ground would be a
 * free lunch of contact force. Together the three terms are the relaxed complementarity conditions of rigid contact,
 * and this is the one that is a genuine inequality; it is wrapped in a relaxed barrier penalty, so a small numerical
 * penetration is expensive but not fatal, as it must be for a solver that linearises.
 *
 * The height is that of the contact frame, so the terrain height configured for it is the height of the ground plus
 * whatever offset the contact frame has from the sole.
 */
class GroundPenetrationConstraint final : public StateConstraint {
 public:
  /**
   * @param [in] endEffectorKinematics : kinematics of this foot's contact frame.
   * @param [in] terrainHeight : [m] height of the ground under the foot.
   */
  GroundPenetrationConstraint(const EndEffectorKinematics<scalar_t>& endEffectorKinematics, scalar_t terrainHeight = 0.0);

  ~GroundPenetrationConstraint() override = default;
  GroundPenetrationConstraint* clone() const override { return new GroundPenetrationConstraint(*this); }

  size_t getNumConstraints(scalar_t time) const override { return 1; }
  vector_t getValue(scalar_t time, const vector_t& state, const PreComputation& preComp) const override;
  VectorFunctionLinearApproximation getLinearApproximation(scalar_t time,
                                                           const vector_t& state,
                                                           const PreComputation& preComp) const override;

  void setTerrainHeight(scalar_t terrainHeight) { terrainHeight_ = terrainHeight; }
  scalar_t getTerrainHeight() const { return terrainHeight_; }

 private:
  GroundPenetrationConstraint(const GroundPenetrationConstraint& rhs);

  std::unique_ptr<EndEffectorKinematics<scalar_t>> endEffectorKinematicsPtr_;
  scalar_t terrainHeight_;
};

}  // namespace ocs2::humanoid
