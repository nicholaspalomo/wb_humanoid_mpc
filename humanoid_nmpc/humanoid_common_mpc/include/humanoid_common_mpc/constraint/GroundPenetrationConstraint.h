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

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact/FootprintCornerHeights.h"

namespace ocs2::humanoid {

/**
 * The unilateral side of the contact condition: g(x) = h(x) - terrainHeight >= 0, the foot may not go through the
 * ground. One row per point of the foot it is given.
 *
 * With the mode-scheduled stance constraint gone (see ForceWeightedSlipConstraint), nothing else holds a foot above
 * the terrain: the complementarity term only forbids force at a height, and a foot pushed below the ground would be a
 * free lunch of contact force. Together the three terms are the relaxed complementarity conditions of rigid contact,
 * and this is the one that is a genuine inequality; it is wrapped in a one-sided squared hinge, which is zero in value
 * AND gradient on the ground and quadratic below it, so it says nothing at all about a foot that is merely resting.
 *
 * WHICH points matter. Given the contact frame alone this term constrains the CENTRE of the sole, and that is not
 * enough: the contact-implicit formulation deliberately leaves the foot's rocking rates free, because rolling over the
 * heel and the toe under load is how a heel-to-toe strike happens. A foot free to pitch about a sole centre held at
 * ground level has its toe and heel below ground for nothing. On the DRC Atlas the corners sit 0.12 m fore and aft, so
 * the shipped 0.08 rad of swing-foot pitch alone buries the toe by about 10 mm. The FootprintCornerHeights this term
 * is built with therefore carries the footprint's CORNER frames - which createPinocchioModel() already adds, one per
 * point of the contact polygon - and the corner heights are exact kinematics rather than a small-angle correction of
 * the centre's.
 *
 * It shares that object with ContactComplementarityConstraint of the same foot, so the two terms cannot end up
 * disagreeing about where the foot is; see the class comment there.
 */
class GroundPenetrationConstraint final : public StateConstraint {
 public:
  /**
   * @param [in] cornerHeights : the points of this foot that may not go below the ground, normally the footprint's
   *        corner frames; the contact frame alone leaves the toe and the heel unconstrained.
   * @param [in] terrainHeight : [m] height of the ground under the foot.
   */
  explicit GroundPenetrationConstraint(const FootprintCornerHeights& cornerHeights, scalar_t terrainHeight = 0.0);

  ~GroundPenetrationConstraint() override = default;
  GroundPenetrationConstraint* clone() const override { return new GroundPenetrationConstraint(*this); }

  size_t getNumConstraints(scalar_t time) const override { return numPoints_; }

  /** The number of points of the foot this term keeps above the ground. */
  size_t getNumPoints() const { return numPoints_; }
  vector_t getValue(scalar_t time, const vector_t& state, const PreComputation& preComp) const override;
  VectorFunctionLinearApproximation getLinearApproximation(scalar_t time,
                                                           const vector_t& state,
                                                           const PreComputation& preComp) const override;

  void setTerrainHeight(scalar_t terrainHeight) { terrainHeight_ = terrainHeight; }
  scalar_t getTerrainHeight() const { return terrainHeight_; }

 private:
  GroundPenetrationConstraint(const GroundPenetrationConstraint& rhs);

  std::unique_ptr<FootprintCornerHeights> cornerHeightsPtr_;
  size_t numPoints_;
  scalar_t terrainHeight_;
};

}  // namespace ocs2::humanoid
