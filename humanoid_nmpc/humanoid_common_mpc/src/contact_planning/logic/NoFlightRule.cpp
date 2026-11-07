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

#include "humanoid_common_mpc/contact_planning/logic/NoFlightRule.h"

#include <array>
#include <cstddef>
#include <string>

#include "humanoid_common_mpc/common/ContactMask.h"
#include "humanoid_common_mpc/contact_planning/MixedIntegerOcpQp.h"
#include "humanoid_common_mpc/contact_planning/logic/ContactLogicHelpers.h"
#include "humanoid_common_mpc/contact_planning/logic/ContactLogicState.h"

namespace ocs2::humanoid {

namespace {
// At least one foot in contact at every node.
constexpr size_t kMinContactCount = 1;
}  // namespace

std::string NoFlightRule::describe() const {
  return "sum_i c_i >= 1 at every node";
}

bool NoFlightRule::propagate(const ContactLogicState& s,
                             MiqpAssignment& a,
                             bool& changed) const {
  const ContactMask everyFoot = ContactMask::allActive(s.numFeet());
  std::array<int, ContactLogicState::kNumFeet> nodeBinaries{};
  for (int k = 0; k < s.numNodes; ++k) {
    for (size_t foot = 0; foot < s.numFeet(); ++foot) {
      nodeBinaries[foot] = ContactLogicState::contactBinaryIndex(k, foot);
    }
    if (!propagateMinimumContactCount(nodeBinaries, everyFoot, kMinContactCount,
                                      a, changed)) {
      return false;
    }
  }
  return true;
}

}  // namespace ocs2::humanoid
