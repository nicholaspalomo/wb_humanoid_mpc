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

#include "humanoid_common_mpc/contact_planning/problem/AssignmentCost.h"

namespace ocs2::humanoid {

/**
 * `double_support_penalty`: a price per decided node at which both feet are in contact, which drives the planner to
 * exchange support in a single node (the touch-down of one foot at the very node the other lifts) instead of paying
 * for a double support with the freedom it buys the ZMP. It is the incentive that `minDoubleSupportDuration: 0` only
 * permits: with the duration at zero and this term absent the planner still keeps a double support at walking speed,
 * because the ZMP terms are cheaper there.
 *
 * The term prices *every* double-support node, and cannot tell the weight transfer of a step apart from standing on
 * two feet, so it prices standing still as well. Above roughly 0.2 with the Atlas gait limits, stepping in place
 * becomes cheaper than standing (a foot lifted and put back costs only the switch cost, while standing pays this term
 * at every node of the horizon) and the robot marches at a zero velocity command. See the regression test
 * DoubleSupportPenaltyLeavesStandingAlone.
 */
class DoubleSupportPenaltyCost final : public AssignmentCost {
 public:
  std::string describe() const override;
  void configure(const ContactPlanningConfig& config) override;
  scalar_t cost(const ContactLogicState& s, const MiqpAssignment& a) const override;

 private:
  scalar_t cost_ = 0.0;
};

}  // namespace ocs2::humanoid
