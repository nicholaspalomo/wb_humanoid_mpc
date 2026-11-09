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

#pragma once

#include <cstddef>

#include "absl/types/span.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

namespace ocs2::humanoid {

/**
 * Writes the planned contact switches of each of the `numContacts` contacts
 * around `time` in `schedule`: timeSinceSwitch[i] is the time [s] since contact
 * i's flag (contactIsActiveInMode()) last changed at an event of the schedule,
 * and timeToSwitch[i] the time [s] until it next changes; +infinity where the
 * schedule shows no such change. The mode at `time` is the one
 * ModeSchedule::modeAtTime() gives, so at an event time the event is still
 * ahead (a time to switch of 0). Writes the first min(numContacts, sizes)
 * entries of each span; the spans are the caller's preallocated storage.
 * Allocates nothing: for the realtime thread.
 */
void plannedContactSwitchTimes(const ModeSchedule& schedule,
                               scalar_t time,
                               size_t numContacts,
                               absl::Span<scalar_t> timeSinceSwitch,
                               absl::Span<scalar_t> timeToSwitch);

}  // namespace ocs2::humanoid
