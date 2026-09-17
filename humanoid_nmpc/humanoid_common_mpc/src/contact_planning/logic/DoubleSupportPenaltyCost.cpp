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

#include "humanoid_common_mpc/contact_planning/logic/DoubleSupportPenaltyCost.h"

#include <sstream>
#include <stdexcept>

namespace ocs2::humanoid {

std::string DoubleSupportPenaltyCost::describe() const {
  std::ostringstream out;
  out << cost_ << " per decided node with every foot in contact (prices standing as well as the weight transfer)";
  return out.str();
}

void DoubleSupportPenaltyCost::configure(const ContactPlanningConfig& config) {
  if (config.doubleSupportPenalty.cost < 0.0) throw std::invalid_argument("[double_support_penalty] cost must be non-negative");
  cost_ = config.doubleSupportPenalty.cost;
}

scalar_t DoubleSupportPenaltyCost::cost(const ContactLogicState& s, const MiqpAssignment& a) const {
  using S = ContactLogicState;
  int numDoubleSupportNodes = 0;
  for (int k = 0; k < s.numNodes; ++k) {
    // Only nodes whose feet are both decided are charged. A node that is still free may or may not become a double
    // support, and the branch-and-bound prunes on this value: counting it would overestimate the partial assignment
    // and could discard the optimum. Skipping it keeps the cost a lower bound, exact once the assignment is complete.
    bool allInContact = true;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      allInContact = allInContact && a[static_cast<size_t>(S::contactBinaryIndex(k, foot))] == 1;
    }
    if (allInContact) ++numDoubleSupportNodes;
  }
  return cost_ * static_cast<scalar_t>(numDoubleSupportNodes);
}

}  // namespace ocs2::humanoid
