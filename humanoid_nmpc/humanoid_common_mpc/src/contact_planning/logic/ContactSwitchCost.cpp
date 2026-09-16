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

#include "humanoid_common_mpc/contact_planning/logic/ContactSwitchCost.h"

#include <sstream>
#include <stdexcept>

namespace ocs2::humanoid {

std::string ContactSwitchCost::describe() const {
  std::ostringstream out;
  out << cost_ << " per lift-off / touch-down event";
  return out.str();
}

void ContactSwitchCost::configure(const ContactPlanningConfig& config) {
  if (config.contactSwitch.cost < 0.0) throw std::invalid_argument("[contact_switch] cost must be non-negative");
  cost_ = config.contactSwitch.cost;
}

scalar_t ContactSwitchCost::cost(const ContactLogicState& s, const MiqpAssignment& a) const {
  int numSwitches = 0;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    int previous = s.input->contacts[foot] ? 1 : 0;
    for (int k = 0; k < s.numNodes; ++k) {
      const std::int8_t value = a[static_cast<size_t>(ContactLogicState::contactBinaryIndex(k, foot))];
      if (value == kMiqpFree) continue;
      if (previous >= 0 && value != previous) ++numSwitches;
      previous = value;
    }
  }
  return cost_ * static_cast<scalar_t>(numSwitches);
}

}  // namespace ocs2::humanoid
