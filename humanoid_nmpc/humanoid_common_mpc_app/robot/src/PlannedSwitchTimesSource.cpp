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

#include "humanoid_common_mpc_app/robot/PlannedSwitchTimesSource.h"

#include <algorithm>
#include <cstddef>
#include <limits>

#include "absl/types/span.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc_app/robot/RobotController.h"
#include "humanoid_state_estimator/ContactEstimationInputs.h"

namespace ocs2::humanoid {

PlannedSwitchTimesSource::PlannedSwitchTimesSource(size_t numContacts)
    : timesSinceSwitch_(numContacts, std::numeric_limits<scalar_t>::infinity()),
      timesToSwitch_(numContacts, std::numeric_limits<scalar_t>::infinity()) {}

void PlannedSwitchTimesSource::fill(
    const RobotController& controller,
    estimation::ContactEstimationInputs& inputs) {
  std::fill(timesSinceSwitch_.begin(), timesSinceSwitch_.end(),
            std::numeric_limits<scalar_t>::infinity());
  std::fill(timesToSwitch_.begin(), timesToSwitch_.end(),
            std::numeric_limits<scalar_t>::infinity());
  inputs.clearPlannedContactSwitchTimes();
  if (!controller.plannedContactSwitchTimes(absl::MakeSpan(timesSinceSwitch_),
                                            absl::MakeSpan(timesToSwitch_))) {
    return;
  }
  const size_t count = std::min(timesSinceSwitch_.size(), inputs.numContacts());
  for (size_t contact = 0; contact < count; ++contact) {
    inputs.setPlannedContactSwitchTimes(contact, timesSinceSwitch_[contact],
                                        timesToSwitch_[contact]);
  }
}

}  // namespace ocs2::humanoid
