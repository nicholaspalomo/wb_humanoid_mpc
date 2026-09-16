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

#include "humanoid_common_mpc/contact_planning/search/WarmStartPreviousPlanStage.h"

#include <algorithm>

#include "humanoid_common_mpc/contact_planning/logic/ContactLogicState.h"

namespace ocs2::humanoid {

std::string WarmStartPreviousPlanStage::describe() const {
  return "the previous plan's contacts, shifted by the elapsed time, are tried first as the incumbent";
}

void WarmStartPreviousPlanStage::beforeSearch(SearchSetup& setup) const {
  if (setup.previousPlanShift < 0 || setup.previousAssignment == nullptr) return;
  constexpr int kBinariesPerNode = ContactLogicState::kBinariesPerNode;
  MiqpAssignment warm(static_cast<size_t>(kBinariesPerNode * setup.numNodes), kMiqpFree);
  for (int k = 0; k < setup.numNodes; ++k) {
    const int source = std::min(k + setup.previousPlanShift, setup.numNodes - 1);
    for (int j = 0; j < kBinariesPerNode; ++j) {
      warm[static_cast<size_t>(kBinariesPerNode * k + j)] = (*setup.previousAssignment)[static_cast<size_t>(kBinariesPerNode * source + j)];
    }
  }
  setup.warmStart = std::move(warm);
}

}  // namespace ocs2::humanoid
