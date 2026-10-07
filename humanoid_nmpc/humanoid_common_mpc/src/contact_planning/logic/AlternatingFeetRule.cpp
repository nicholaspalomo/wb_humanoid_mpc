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

#include "humanoid_common_mpc/contact_planning/logic/AlternatingFeetRule.h"

#include <array>
#include <string>

#include "humanoid_common_mpc/contact_planning/logic/ContactLogicHelpers.h"

namespace ocs2::humanoid {

static_assert(kNumContacts == 2, "the alternation rule is written for a biped");

std::string AlternatingFeetRule::describe() const {
  return "a foot may not swing twice without the other foot swinging in between";
}

bool AlternatingFeetRule::propagate(const ContactLogicState& s, MiqpAssignment& a, bool& changed) const {
  int lastSwung = s.input->lastSwungFoot;
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    if (!s.input->contacts[foot]) lastSwung = static_cast<int>(foot);
  }
  std::array<int8_t, kNumContacts> kappa{static_cast<int8_t>(s.input->contacts[0] ? 1 : 0),
                                         static_cast<int8_t>(s.input->contacts[1] ? 1 : 0)};
  for (int k = 0; k < s.numNodes; ++k) {
    bool allFixed = true;
    // Whether a foot handled earlier at this node may still lift here. The feet of one node are read in index order,
    // so on a complete assignment a lift-off of foot 0 at this node already counts as the last swing when foot 1 is
    // read. While foot 0's binary here is free that is undecided, and so is everything this node says about foot 1.
    bool earlierFootMayLift = false;
    for (size_t foot = 0; foot < kNumContacts; ++foot) {
      const int index = ContactLogicState::contactBinaryIndex(k, foot);
      const int8_t value = a[static_cast<size_t>(index)];
      if (kappa[foot] == 1) {
        // The rule constrains the free nodes: along the committed prefix the executed schedule is what it is (a
        // repeated lift-off there came from an earlier plan or a re-timed event), and refusing it made every plan
        // infeasible, silently, until the node left the window. For the same reason nothing is FIXED there either:
        // a fixing has to hold in every completion that satisfies the rule, and on those nodes any value does.
        const bool decided = k >= s.numCommitted && !earlierFootMayLift;
        if (value == 0) {
          if (lastSwung == static_cast<int>(foot) && decided) return false;
          lastSwung = static_cast<int>(foot);
        } else if (value == kMiqpFree) {
          if (lastSwung == static_cast<int>(foot) && decided) {
            fixBinary(a, index, /*value=*/1, changed);
          } else {
            earlierFootMayLift = true;
          }
        }
      }
      allFixed = allFixed && a[static_cast<size_t>(index)] != kMiqpFree;
    }
    if (!allFixed) break;
    for (size_t foot = 0; foot < kNumContacts; ++foot) {
      kappa[foot] = a[static_cast<size_t>(ContactLogicState::contactBinaryIndex(k, foot))];
    }
  }
  return true;
}

}  // namespace ocs2::humanoid
