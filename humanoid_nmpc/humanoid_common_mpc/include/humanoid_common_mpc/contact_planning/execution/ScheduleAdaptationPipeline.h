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

#pragma once

#include "humanoid_common_mpc/contact_planning/execution/ExecutionRule.h"
#include "humanoid_common_mpc/contact_planning/problem/TermCollection.h"

namespace ocs2::humanoid {

/**
 * Adapts `schedule` to the measured contact state at ctx.time with the listed rules, one call per control cycle: for
 * every foot the swing latch is tracked (a new swing is latched, a stale latch cleared, a foot that landed as scheduled
 * releases its latch), then the rules run in order until one reports an event. Returns what happened to every foot.
 */
feet_array_t<ContactEventReport> adaptScheduleWithRules(ModeSchedule& schedule,
                                                        const ExecutionContext& ctx,
                                                        const TermCollection<ExecutionRule>& rules,
                                                        feet_array_t<SwingTimingLatch>& latches);

}  // namespace ocs2::humanoid
