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

#include "legged_common_mpc/gait/ModeScheduleSwitchTimes.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "absl/types/span.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "legged_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::legged {

void plannedContactSwitchTimes(const ModeSchedule& schedule,
                               scalar_t time,
                               size_t numContacts,
                               absl::Span<scalar_t> timeSinceSwitch,
                               absl::Span<scalar_t> timeToSwitch) {
  constexpr scalar_t kUnknown = std::numeric_limits<scalar_t>::infinity();
  const size_t count =
      std::min({numContacts, timeSinceSwitch.size(), timeToSwitch.size()});
  const std::vector<scalar_t>& eventTimes = schedule.eventTimes;
  const std::vector<size_t>& modes = schedule.modeSequence;
  // Event j separates modes j and j + 1; the mode at `time` is modes[now], as
  // ModeSchedule::modeAtTime() finds it.
  const size_t now = static_cast<size_t>(
      std::lower_bound(eventTimes.begin(), eventTimes.end(), time) -
      eventTimes.begin());
  for (size_t contact = 0; contact < count; ++contact) {
    timeSinceSwitch[contact] = kUnknown;
    timeToSwitch[contact] = kUnknown;
    if (now >= modes.size()) continue;  // a malformed schedule
    const bool active = contactIsActiveInMode(modes[now], contact, numContacts);
    for (size_t event = now;
         event < eventTimes.size() && event + 1 < modes.size(); ++event) {
      if (contactIsActiveInMode(modes[event + 1], contact, numContacts) !=
          active) {
        timeToSwitch[contact] = eventTimes[event] - time;
        break;
      }
    }
    for (size_t event = now; event > 0; --event) {
      if (contactIsActiveInMode(modes[event - 1], contact, numContacts) !=
          active) {
        timeSinceSwitch[contact] = time - eventTimes[event - 1];
        break;
      }
    }
  }
}

}  // namespace ocs2::legged
