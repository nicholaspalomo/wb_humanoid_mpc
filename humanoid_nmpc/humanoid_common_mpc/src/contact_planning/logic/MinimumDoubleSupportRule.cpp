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

#include "humanoid_common_mpc/contact_planning/logic/MinimumDoubleSupportRule.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/ContactMask.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/MixedIntegerOcpQp.h"
#include "humanoid_common_mpc/contact_planning/logic/ContactLogicHelpers.h"
#include "humanoid_common_mpc/contact_planning/logic/ContactLogicState.h"

namespace ocs2::humanoid {

std::string MinimumDoubleSupportRule::describe() const {
  if (!minSupportCount_.has_value()) {
    return absl::StrCat(
        "after a touch-down every foot in contact stays down >= ",
        minDoubleSupportDuration_, " s (the touch-down node included)");
  }
  return absl::StrCat("after a touch-down a foot in contact lifts only while ",
                      *minSupportCount_,
                      " feet stay down, for >= ", minDoubleSupportDuration_,
                      " s (the touch-down node included)");
}

void MinimumDoubleSupportRule::configure(const ContactPlanningConfig& config) {
  minDoubleSupportDuration_ = config.shared.gaitLimits.minDoubleSupportDuration;
}

bool MinimumDoubleSupportRule::propagate(const ContactLogicState& s,
                                         MiqpAssignment& a,
                                         bool& changed) const {
  using S = ContactLogicState;
  // Sound on a partial assignment: the walk stops at the first node that is not
  // fixed for every foot, and at the node it stops on only a touch-down that is
  // already fixed is counted. A binary still free there can only add a
  // touch-down, which can only lengthen the hold, so every hold this pass
  // enforces holds in every completion.
  if (s.nDoubleSupportHold <= 0) return true;
  const size_t numFeet = s.numFeet();
  const size_t minSupportCount = minSupportCount_.value_or(numFeet);
  std::array<int8_t, S::kNumFeet> kappa{};
  for (size_t foot = 0; foot < numFeet; ++foot) {
    kappa[foot] = static_cast<int8_t>(s.input->contacts[foot] ? 1 : 0);
  }
  std::array<int, S::kNumFeet> nodeBinaries{};
  scalar_t lastTouchDown = s.initialTouchDownNode;
  for (int k = 0; k < s.numNodes; ++k) {
    for (size_t foot = 0; foot < numFeet; ++foot) {
      nodeBinaries[foot] = S::contactBinaryIndex(k, foot);
    }
    // A touch-down decided at this very node already binds the other feet at
    // this node: lifting one here would move the weight from one support to
    // another with no shared support at all.
    scalar_t latestTouchDown = lastTouchDown;
    for (size_t foot = 0; foot < numFeet; ++foot) {
      if (kappa[foot] == 0 && a[static_cast<size_t>(nodeBinaries[foot])] == 1) {
        latestTouchDown = std::max(latestTouchDown, s.switchNode(k, foot));
      }
    }
    // The feet the hold binds at this node: those in contact before it, while
    // the hold after the latest touch-down lasts, off the committed prefix.
    ContactMask heldFeet(numFeet);
    if (s.heldAfterTouchDown(k, latestTouchDown) && k >= s.numCommitted) {
      for (size_t foot = 0; foot < numFeet; ++foot) {
        heldFeet.set(foot, /*active=*/kappa[foot] == 1);
      }
    }
    if (!propagateMinimumContactCount(nodeBinaries, heldFeet, minSupportCount,
                                      a, changed)) {
      return false;
    }
    bool allFixed = true;
    for (size_t foot = 0; foot < numFeet; ++foot) {
      allFixed =
          allFixed && a[static_cast<size_t>(nodeBinaries[foot])] != kMiqpFree;
    }
    if (!allFixed) break;
    for (size_t foot = 0; foot < numFeet; ++foot) {
      const int8_t value = a[static_cast<size_t>(nodeBinaries[foot])];
      if (kappa[foot] == 0 && value == 1) {
        lastTouchDown = std::max(lastTouchDown, s.switchNode(k, foot));
      }
      kappa[foot] = value;
    }
  }
  return true;
}

}  // namespace ocs2::humanoid
