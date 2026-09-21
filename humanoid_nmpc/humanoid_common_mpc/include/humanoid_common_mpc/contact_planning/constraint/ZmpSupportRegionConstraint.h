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

#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_common_mpc/contact_planning/constraint/LipConstraintBase.h"

namespace ocs2::humanoid {

/**
 * `zmp_support_region` (soft, running nodes): single support, the box of half-widths (r_x, r_y) around the supporting
 * foot, each foot's box relaxed unless that foot is the only contact; double support, along the heading the box of
 * half-width r_x around the midpoint of the feet, laterally the strip between the right foot minus and the left foot
 * plus r_y. Big-M disjunctions on the contact binaries.
 *
 * WHAT THE DOUBLE-SUPPORT REGION ACTUALLY IS, because this used to claim the opposite. The two rows above are each a
 * valid PROJECTION of the support hull onto one axis, but the set the solver sees is their INTERSECTION, and the
 * intersection of the projections of a convex set is that set's bounding box, not the set. The two agree only while
 * the feet are level along the heading - which is where the argument for calling this "a conservative inner
 * approximation" was made, and it does not survive a foot being in front of the other. In the transition double
 * support of every step the feet are a step apart along the heading, and then the admitted region strictly CONTAINS
 * the hull: it is an OUTER approximation, and the planned ZMP can sit outside the true support polygon.
 *
 * Concretely, with p_L = (0.2, 0.1), p_R = (0, -0.1), r_x = 0.08, r_y = 0.04, the point (0.18, -0.14) satisfies both
 * rows, yet the hull's support in the direction (1, -1)/sqrt(2) is 0.22 while that point reaches 0.32. The corner is
 * over-admitted by about half the heading offset between the feet, i.e. by up to half a step length.
 *
 * This is not repaired here, and the reason is worth stating rather than leaving as a TODO. Cutting the two missing
 * hull edges needs the row n'(zmp - p) <= r with n normal to (p_L - p_R); the foot positions are DECISION VARIABLES,
 * so that row is bilinear and cannot be expressed by LipConstraintBase, which emits affine rows only. It would have
 * to be linearised about a nominal foot separation, which is a modelling choice that changes the feasible set of a
 * SOFT constraint the whole-body MPC re-solves anyway against the true wrench cone. That makes it a conservatism
 * decision to be validated in simulation, not a repair to be folded into a correctness pass - so it belongs behind
 * its own configuration key, defaulting to the present behaviour, rather than being switched on here.
 */
class ZmpSupportRegionConstraint final : public LipConstraintBase {
 public:
  std::string describe() const override;
  void configure(const ContactPlanningConfig& config) override;
  NodeSet nodeSet() const override { return NodeSet::RUNNING; }
  Softness softness() const override { return Softness::SOFT; }
  void addRows(const ContactPlanningContext& ctx, int node, RowBuilder& rows) const override;

 private:
  scalar_t halfWidthX_ = 0.0;
  scalar_t halfWidthY_ = 0.0;
};

}  // namespace ocs2::humanoid
