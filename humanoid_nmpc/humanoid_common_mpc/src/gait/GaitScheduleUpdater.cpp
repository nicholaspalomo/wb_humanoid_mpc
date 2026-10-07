/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include "humanoid_common_mpc/gait/GaitScheduleUpdater.h"

#include <algorithm>
#include <vector>

#include "absl/log/log.h"

#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {

void updateGaitSchedule(GaitSchedule& gaitSchedule, const ModeSequenceTemplate& updatedGait, scalar_t initTime, scalar_t finalTime) {
  LOG(INFO) << updatedGait;
  const scalar_t timeHorizon = finalTime - initTime;
  const scalar_t earliestSwitchingTime = (0.7 * finalTime + 0.3 * initTime);  // This is a heuristic
  LOG(INFO) << "[updateGaitSchedule]: Setting new gait after time " << earliestSwitchingTime << "\n";
  // Find the first time that is greater than current_time
  const ModeSchedule modeSchedule = gaitSchedule.getModeSchedule(initTime, finalTime + timeHorizon);

  const std::vector<scalar_t>::const_iterator it =
      std::upper_bound(modeSchedule.eventTimes.begin(), modeSchedule.eventTimes.end(), earliestSwitchingTime);
  scalar_t nextEventTime = finalTime;
  if (it == modeSchedule.eventTimes.end()) {
    nextEventTime = finalTime;
  } else if (modeSchedule.modeAtTime(*it) == ModeNumber::kLf && it != modeSchedule.eventTimes.begin()) {
    // The phase that ends at this event is a left-foot swing (modeAtTime of an event time is the phase before it), so
    // the new gait starts where that swing starts, at the event before. When the swing is the first phase of the
    // schedule there is no event before it - the first event already lies past the switching time, as after a clock
    // that ran backwards - and reading one was undefined: the gait then starts where the swing ends.
    nextEventTime = *(it - 1);
  } else {
    nextEventTime = *it;
  }

  // insertModeSequenceTemplate tiles the new gait up to an ABSOLUTE time, 1.5 horizons ahead of now. This passed
  // 1.5 * timeHorizon, a duration: as soon as the new gait started later than that on the clock - about a second into
  // any run - it was not tiled at all and the schedule stopped in stance where the gait was to begin. The next
  // getModeSchedule() tiles from its last event again, so every solve still saw the new gait; only a reader of the
  // schedule in between did not (testGaitScheduleUpdaterHorizon).
  gaitSchedule.insertModeSequenceTemplate(updatedGait, nextEventTime, /*finalTime=*/initTime + 1.5 * timeHorizon);
}

}  // namespace ocs2::humanoid
