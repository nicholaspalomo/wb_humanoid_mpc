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

#include "humanoid_common_mpc/contact_planning/logic/AlternatingFeetRule.h"

#include "humanoid_common_mpc/contact_planning/logic/ContactLogicHelpers.h"

namespace ocs2::humanoid {

static_assert(N_CONTACTS == 2, "the alternation rule is written for a biped");

std::string AlternatingFeetRule::describe() const {
  return "a foot may not swing twice without the other foot swinging in between";
}

bool AlternatingFeetRule::propagate(const ContactLogicState& s, const ContactLogicScan& /*scan*/, MiqpAssignment& a, bool& changed) const {
  using S = ContactLogicState;
  int lastSwung = s.input->lastSwungFoot;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    if (!s.input->contacts[foot]) lastSwung = static_cast<int>(foot);
  }
  std::array<std::int8_t, N_CONTACTS> kappa{s.input->contacts[0] ? std::int8_t(1) : std::int8_t(0),
                                            s.input->contacts[1] ? std::int8_t(1) : std::int8_t(0)};
  for (int k = 0; k < s.numNodes; ++k) {
    bool allFixed = true;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const int index = S::contactBinaryIndex(k, foot);
      const std::int8_t value = a[static_cast<size_t>(index)];
      if (kappa[foot] == 1) {
        if (value == 0) {
          // The rule constrains the free nodes: along the committed prefix the executed schedule is what it is (a
          // repeated lift-off there came from an earlier plan or a re-timed event), and refusing it made every plan
          // infeasible, silently, until the node left the window.
          if (lastSwung == static_cast<int>(foot) && k >= s.numCommitted) return false;
          lastSwung = static_cast<int>(foot);
        } else if (value == kMiqpFree && lastSwung == static_cast<int>(foot)) {
          fixBinary(a, index, 1, changed);
        }
      }
      allFixed = allFixed && a[static_cast<size_t>(index)] != kMiqpFree;
    }
    if (!allFixed) break;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      kappa[foot] = a[static_cast<size_t>(S::contactBinaryIndex(k, foot))];
    }
  }
  return true;
}

}  // namespace ocs2::humanoid
