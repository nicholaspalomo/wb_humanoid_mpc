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

#include "humanoid_common_mpc/contact_planning/logic/HopOnRequestRule.h"

#include <sstream>

#include "humanoid_common_mpc/contact_planning/logic/ContactLogicHelpers.h"

namespace ocs2::humanoid {

static_assert(N_CONTACTS == 2, "the hop rule is written for a biped");

std::string HopOnRequestRule::describe() const {
  std::ostringstream out;
  out << "commanded base height > " << triggerBaseHeight_ << " m: every double support of the minimum contact duration ends in a flight of "
      << flightDuration_ << " s";
  return out.str();
}

void HopOnRequestRule::configure(const ContactPlanningConfig& config) {
  triggerBaseHeight_ = config.hopOnRequest.triggerBaseHeight;
  flightDuration_ = config.hopOnRequest.flightDuration;
}

bool HopOnRequestRule::propagate(const ContactLogicState& s, const ContactLogicScan& /*scan*/, MiqpAssignment& a, bool& changed) const {
  using S = ContactLogicState;
  if (!s.input->hopRequested || s.nFlightMax <= 0) return true;
  const int N = s.numNodes;
  // Both feet down for at least the minimum contact duration (rounded up, like a maximum-duration rule) and the flight
  // able to end before the horizon: lift both.
  for (int k = std::max(0, s.numCommitted); k < N; ++k) {
    if (k + s.nFlightMin > N) break;
    const int ageL = s.contactAge(a, 0, k);
    const int ageR = s.contactAge(a, 1, k);
    if (ageL < 0 || ageR < 0) break;       // the prefix is not fixed yet
    if (ageL == 0 || ageR == 0) continue;  // not a double support
    if (std::min(ageL, ageR) < s.nContactMin) continue;
    const int iL = S::contactBinaryIndex(k, 0), iR = S::contactBinaryIndex(k, 1);
    if (a[static_cast<size_t>(iL)] == 1 || a[static_cast<size_t>(iR)] == 1) return false;  // it stayed down: not a hop
    if (!fixBinary(a, iL, 0, changed) || !fixBinary(a, iR, 0, changed)) return false;
  }
  return true;
}

}  // namespace ocs2::humanoid
