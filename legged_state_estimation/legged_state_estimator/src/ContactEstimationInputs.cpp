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

#include "legged_state_estimator/ContactEstimationInputs.h"

#include <algorithm>
#include <cstddef>
#include <limits>

#include "ocs2_core/Types.h"

namespace ocs2::legged::estimation {

ContactEstimationInputs::ContactEstimationInputs(size_t numContacts)
    : nomPosContactsRtWorldEwrtWorldZ_(numContacts, /*value=*/0.0),
      hasNomPosContactsRtWorldEwrtWorldZ_(numContacts, /*value=*/true),
      plannedTimesSinceSwitch_(numContacts,
                               std::numeric_limits<scalar_t>::infinity()),
      plannedTimesToSwitch_(numContacts,
                            std::numeric_limits<scalar_t>::infinity()) {}

void ContactEstimationInputs::setAllNomPosContactsRtWorldEwrtWorldZ(
    scalar_t height) {
  std::fill(nomPosContactsRtWorldEwrtWorldZ_.begin(),
            nomPosContactsRtWorldEwrtWorldZ_.end(), height);
  std::fill(hasNomPosContactsRtWorldEwrtWorldZ_.begin(),
            hasNomPosContactsRtWorldEwrtWorldZ_.end(), true);
}

void ContactEstimationInputs::clearPlannedContactSwitchTimes() {
  std::fill(plannedTimesSinceSwitch_.begin(), plannedTimesSinceSwitch_.end(),
            std::numeric_limits<scalar_t>::infinity());
  std::fill(plannedTimesToSwitch_.begin(), plannedTimesToSwitch_.end(),
            std::numeric_limits<scalar_t>::infinity());
}

}  // namespace ocs2::legged::estimation
