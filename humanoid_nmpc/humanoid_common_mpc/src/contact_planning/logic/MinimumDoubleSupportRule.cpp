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

#include "humanoid_common_mpc/contact_planning/logic/MinimumDoubleSupportRule.h"

#include <algorithm>
#include <sstream>

#include "humanoid_common_mpc/contact_planning/logic/ContactLogicHelpers.h"

namespace ocs2::humanoid {

static_assert(N_CONTACTS == 2, "the double support rule is written for a biped");

std::string MinimumDoubleSupportRule::describe() const {
  std::ostringstream out;
  out << "after a touch-down the other foot stays down >= " << minDoubleSupportDuration_ << " s (the touch-down node included)";
  return out.str();
}

void MinimumDoubleSupportRule::configure(const ContactPlanningConfig& config) {
  minDoubleSupportDuration_ = config.shared.gaitLimits.minDoubleSupportDuration;
}

bool MinimumDoubleSupportRule::propagate(const ContactLogicState& s,
                                         const ContactLogicScan& /*scan*/,
                                         MiqpAssignment& a,
                                         bool& changed) const {
  using S = ContactLogicState;
  if (s.nDoubleSupportHold <= 0) return true;
  std::array<std::int8_t, N_CONTACTS> kappa{s.input->contacts[0] ? std::int8_t(1) : std::int8_t(0),
                                            s.input->contacts[1] ? std::int8_t(1) : std::int8_t(0)};
  scalar_t lastTouchDown = s.initialTouchDownNode;
  for (int k = 0; k < s.numNodes; ++k) {
    // A touch-down decided at this very node already binds the other foot at this node: lifting it here would turn
    // single support on one foot into single support on the other with no double support at all.
    scalar_t latestTouchDown = lastTouchDown;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (kappa[foot] == 0 && a[static_cast<size_t>(S::contactBinaryIndex(k, foot))] == 1) {
        latestTouchDown = std::max(latestTouchDown, s.switchNode(k, foot));
      }
    }
    bool allFixed = true;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const int index = S::contactBinaryIndex(k, foot);
      if (kappa[foot] == 1 && s.heldAfterTouchDown(k, latestTouchDown) && k >= s.numCommitted) {
        if (a[static_cast<size_t>(index)] == 0) return false;
        fixBinary(a, index, 1, changed);
      }
      allFixed = allFixed && a[static_cast<size_t>(index)] != kMiqpFree;
    }
    if (!allFixed) break;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const std::int8_t value = a[static_cast<size_t>(S::contactBinaryIndex(k, foot))];
      if (kappa[foot] == 0 && value == 1) lastTouchDown = std::max(lastTouchDown, s.switchNode(k, foot));
      kappa[foot] = value;
    }
  }
  return true;
}

}  // namespace ocs2::humanoid
