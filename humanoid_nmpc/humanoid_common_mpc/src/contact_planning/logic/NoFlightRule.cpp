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

#include "humanoid_common_mpc/contact_planning/logic/NoFlightRule.h"

#include "humanoid_common_mpc/contact_planning/logic/ContactLogicHelpers.h"

namespace ocs2::humanoid {

static_assert(N_CONTACTS == 2, "the no-flight rule is written for a biped");

std::string NoFlightRule::describe() const {
  return "c_L + c_R >= 1 at every node";
}

bool NoFlightRule::propagate(const ContactLogicState& s, const ContactLogicScan& /*scan*/, MiqpAssignment& a, bool& changed) const {
  using S = ContactLogicState;
  for (int k = 0; k < s.numNodes; ++k) {
    const int cL = S::contactBinaryIndex(k, 0), cR = S::contactBinaryIndex(k, 1);
    if (a[static_cast<size_t>(cL)] == 0 && a[static_cast<size_t>(cR)] == 0) return false;
    if (a[static_cast<size_t>(cL)] == 0 && !fixBinary(a, cR, 1, changed)) return false;
    if (a[static_cast<size_t>(cR)] == 0 && !fixBinary(a, cL, 1, changed)) return false;
  }
  return true;
}

}  // namespace ocs2::humanoid
