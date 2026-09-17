/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

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

#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_common_mpc/contact_planning/constraint/LipConstraintBase.h"

namespace ocs2::humanoid {

/**
 * `contact_height` (soft): |z - z_nom| <= tol + M_z (1 - c_i) per foot on the running nodes. A foot on the ground puts
 * the centre of mass near the pendulum height z_nom, so a landing is where the ballistic arc returns to it and a
 * take-off starts from it. The terminal node has no contact binaries of its own and carries no row; `flight_durations`
 * keeps the plan from ending in flight instead, so the horizon never closes in free fall.
 */
class ContactHeightConstraint final : public LipConstraintBase {
 public:
  std::string describe() const override;
  std::vector<std::string> requiredBlocks() const override { return {term::kVerticalDoubleIntegrator}; }
  void configure(const ContactPlanningConfig& config) override;
  NodeSet nodeSet() const override { return NodeSet::RUNNING; }
  Softness softness() const override { return Softness::SOFT; }
  void addRows(const ContactPlanningContext& ctx, int node, RowBuilder& rows) const override;

  static constexpr scalar_t kHeightBigM = 1.0;  // [m] more than any flight apex above the pendulum height

 private:
  scalar_t tolerance_ = 0.0;
  scalar_t nominalHeight_ = 0.0;
};

}  // namespace ocs2::humanoid
