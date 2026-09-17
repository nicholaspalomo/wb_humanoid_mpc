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
  // The foot that swung most recently. A single foot in the air at the planning instant is that foot; with both in the
  // air (a flight, once the flight model is listed) neither is, and the input's own value, which the reference manager
  // reads off the executed schedule, stands. Taking the last airborne index there named the same foot after every
  // flight and rejected the other foot's next swing as a repeat, which made replanning in flight infeasible.
  int lastSwung = s.input->lastSwungFoot;
  int feetInAir = 0;
  int footInAir = -1;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    if (!s.input->contacts[foot]) {
      ++feetInAir;
      footInAir = static_cast<int>(foot);
    }
  }
  if (feetInAir == 1) lastSwung = footInAir;
  std::array<std::int8_t, N_CONTACTS> kappa{s.input->contacts[0] ? std::int8_t(1) : std::int8_t(0),
                                            s.input->contacts[1] ? std::int8_t(1) : std::int8_t(0)};
  for (int k = 0; k < s.numNodes; ++k) {
    // A take-off that lifts every foot at once is a hop: no foot swings past another, the alternation has nothing to
    // say about it, and it leaves no single foot as the one that swung last. Without this a hop right after a step was
    // rejected as that foot swinging twice, and the jump button could never produce a plan.
    int feetDown = 0;
    int liftingFeet = 0;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const std::int8_t value = a[static_cast<size_t>(S::contactBinaryIndex(k, foot))];
      if (value == 1) ++feetDown;
      if (kappa[foot] == 1 && value == 0) ++liftingFeet;
    }
    const bool hopTakeOff = feetDown == 0 && liftingFeet == static_cast<int>(N_CONTACTS);
    if (hopTakeOff) lastSwung = -1;

    bool allFixed = true;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const int index = S::contactBinaryIndex(k, foot);
      const std::int8_t value = a[static_cast<size_t>(index)];
      if (kappa[foot] == 1 && !hopTakeOff) {
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
