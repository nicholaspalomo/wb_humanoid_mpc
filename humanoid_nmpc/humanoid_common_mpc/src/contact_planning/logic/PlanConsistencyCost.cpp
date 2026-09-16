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

#include "humanoid_common_mpc/contact_planning/logic/PlanConsistencyCost.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace ocs2::humanoid {

std::string PlanConsistencyCost::describe() const {
  std::ostringstream out;
  out << cost_ << " per decided node whose contact differs from the previous plan";
  return out.str();
}

void PlanConsistencyCost::configure(const ContactPlanningConfig& config) {
  if (config.planConsistency.cost < 0.0) throw std::invalid_argument("[plan_consistency] cost must be non-negative");
  cost_ = config.planConsistency.cost;
}

scalar_t PlanConsistencyCost::cost(const ContactLogicState& s, const MiqpAssignment& a) const {
  using S = ContactLogicState;
  int numInconsistent = 0;
  if (s.previousPlanShift >= 0 && s.previousAssignment != nullptr && cost_ > 0.0) {
    for (int k = 0; k < s.numNodes; ++k) {
      const int source = std::min(k + s.previousPlanShift, s.numNodes - 1);
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        const std::int8_t value = a[static_cast<size_t>(S::contactBinaryIndex(k, foot))];
        if (value != kMiqpFree && value != (*s.previousAssignment)[static_cast<size_t>(S::contactBinaryIndex(source, foot))]) {
          ++numInconsistent;
        }
      }
    }
  }
  return cost_ * static_cast<scalar_t>(numInconsistent);
}

}  // namespace ocs2::humanoid
