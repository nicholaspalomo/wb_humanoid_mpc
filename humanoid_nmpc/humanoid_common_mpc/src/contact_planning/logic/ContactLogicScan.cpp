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

#include "humanoid_common_mpc/contact_planning/logic/ContactLogicScan.h"

#include <algorithm>

namespace ocs2::humanoid {

ContactLogicScan ContactLogicScan::compute(const ContactLogicState& s, const MiqpAssignment& a) {
  ContactLogicScan scan;
  for (auto& hold : scan.heldByDoubleSupport) hold.assign(static_cast<size_t>(s.numNodes), false);
  if (s.nDoubleSupportHold <= 0) return scan;
  std::array<std::int8_t, N_CONTACTS> kappa;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) kappa[foot] = s.input->contacts[foot] ? 1 : 0;
  scalar_t lastTouchDown = s.initialTouchDownNode;
  for (int k = 0; k < s.numNodes; ++k) {
    scalar_t latestTouchDown = lastTouchDown;
    bool allFixed = true;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const std::int8_t value = a[static_cast<size_t>(ContactLogicState::contactBinaryIndex(k, foot))];
      allFixed = allFixed && value != kMiqpFree;
      if (kappa[foot] == 0 && value == 1) latestTouchDown = std::max(latestTouchDown, s.switchNode(k, foot));
    }
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      scan.heldByDoubleSupport[foot][static_cast<size_t>(k)] = kappa[foot] == 1 && s.heldAfterTouchDown(k, latestTouchDown);
    }
    if (!allFixed) break;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const std::int8_t value = a[static_cast<size_t>(ContactLogicState::contactBinaryIndex(k, foot))];
      if (kappa[foot] == 0 && value == 1) lastTouchDown = std::max(lastTouchDown, s.switchNode(k, foot));
      kappa[foot] = value;
    }
  }
  return scan;
}

}  // namespace ocs2::humanoid
