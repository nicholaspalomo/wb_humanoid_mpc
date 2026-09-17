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
 * `vertical_thrust_limit` (hard, running nodes): az + g <= a_max (c_L + c_R). With no foot down the centre of mass falls
 * at g (ballistic flight, since az >= -g is an input bound of the vertical block); every stance foot adds the thrust
 * a_max = F_max / m it can push with. One row on the contact sum, no extra binary.
 */
class VerticalThrustLimitConstraint final : public LipConstraintBase {
 public:
  std::string describe() const override;
  std::vector<std::string> requiredBlocks() const override { return {term::kVerticalDoubleIntegrator}; }
  void configure(const ContactPlanningConfig& config) override;
  NodeSet nodeSet() const override { return NodeSet::RUNNING; }
  Softness softness() const override { return Softness::HARD; }
  void addRows(const ContactPlanningContext& ctx, int node, RowBuilder& rows) const override;

 private:
  scalar_t maxContactAcceleration_ = 0.0;
  scalar_t gravity_ = 9.81;
};

}  // namespace ocs2::humanoid
