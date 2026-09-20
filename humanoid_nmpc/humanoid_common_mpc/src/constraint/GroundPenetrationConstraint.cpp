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

#include "humanoid_common_mpc/constraint/GroundPenetrationConstraint.h"

#include "absl/log/check.h"

namespace ocs2::humanoid {

GroundPenetrationConstraint::GroundPenetrationConstraint(const FootprintCornerHeights& cornerHeights, scalar_t terrainHeight)
    : StateConstraint(ConstraintOrder::Linear),
      cornerHeightsPtr_(cornerHeights.clone()),
      numPoints_(cornerHeights.numCorners()),
      terrainHeight_(terrainHeight) {
  CHECK_GT(numPoints_, 0U) << "[GroundPenetrationConstraint] needs at least one point of the foot to keep above the ground";
}

GroundPenetrationConstraint::GroundPenetrationConstraint(const GroundPenetrationConstraint& rhs)
    : StateConstraint(rhs),
      cornerHeightsPtr_(rhs.cornerHeightsPtr_->clone()),
      numPoints_(rhs.numPoints_),
      terrainHeight_(rhs.terrainHeight_) {}

vector_t GroundPenetrationConstraint::getValue(scalar_t time, const vector_t& state, const PreComputation& preComp) const {
  return cornerHeightsPtr_->getHeights(state).array() - terrainHeight_;
}

VectorFunctionLinearApproximation GroundPenetrationConstraint::getLinearApproximation(scalar_t time,
                                                                                      const vector_t& state,
                                                                                      const PreComputation& preComp) const {
  VectorFunctionLinearApproximation approximation;
  approximation.f = cornerHeightsPtr_->getHeights(state).array() - terrainHeight_;
  approximation.dfdx = cornerHeightsPtr_->getHeightsJacobian(state);
  return approximation;
}

}  // namespace ocs2::humanoid
