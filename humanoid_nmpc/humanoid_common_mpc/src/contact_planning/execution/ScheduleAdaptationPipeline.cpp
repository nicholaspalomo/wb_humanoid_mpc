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

#include "humanoid_common_mpc/contact_planning/execution/ScheduleAdaptationPipeline.h"

#include <cmath>
#include <limits>

#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {

namespace {
constexpr scalar_t kSameSwingTolerance = 1e-6;  // [s] lift-off times closer than this identify the same swing

bool footInContact(const ModeSchedule& schedule, size_t phaseIndex, size_t foot) {
  return modeNumber2StanceLeg(schedule.modeSequence[phaseIndex])[foot];
}

/** First phase of the swing of `foot` that contains phase `index` (the foot is out of contact in `index`). */
size_t firstSwingPhase(const ModeSchedule& schedule, size_t foot, size_t index) {
  while (index > 0 && !footInContact(schedule, index - 1, foot)) --index;
  return index;
}
}  // namespace

feet_array_t<ContactEventReport> adaptScheduleWithRules(ModeSchedule& schedule,
                                                        const ExecutionContext& ctx,
                                                        const TermCollection<ExecutionRule>& rules,
                                                        feet_array_t<SwingTimingLatch>& latches) {
  feet_array_t<ContactEventReport> reports = makeFeetArray(ContactEventReport{});
  if (schedule.modeSequence.empty()) return reports;
  const scalar_t time = ctx.time;

  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    SwingTimingLatch& latch = latches[foot];
    ContactEventReport& report = reports[foot];

    const auto phase = swingPhaseAtTime(schedule, foot, time);
    if (phase.has_value()) {
      // ---- the foot is scheduled to swing at `time`: latch the swing, then let the rules act on it ----
      const auto [liftOff, touchDown] = *phase;
      if (!latch.active || std::abs(latch.liftOffTime - liftOff) > kSameSwingTolerance) {
        latch = SwingTimingLatch{};
        latch.active = true;
        latch.liftOffTime = liftOff;
        latch.nominalTouchDownTime = touchDown;
      }
      for (const auto& rule : rules) {
        if (rule->adaptSwingingFoot(ctx, foot, liftOff, touchDown, schedule, latch, report)) break;
      }
      continue;
    }

    // ---- the foot is scheduled to be in contact at `time` ----
    if (!latch.active) continue;
    // The latched swing must be the one that ended at the last event before `time`; otherwise the latch is stale.
    const size_t index = modeIndexAtTime(schedule, time);
    const bool justLanded = index > 0 && !footInContact(schedule, index - 1, foot);
    if (!justLanded) {
      latch = SwingTimingLatch{};
      continue;
    }
    const size_t first = firstSwingPhase(schedule, foot, index - 1);
    const scalar_t liftOff = (first > 0) ? schedule.eventTimes[first - 1] : -std::numeric_limits<scalar_t>::infinity();
    if (std::abs(liftOff - latch.liftOffTime) > kSameSwingTolerance) {
      latch = SwingTimingLatch{};
      continue;
    }
    if (ctx.measuredContact[foot]) {
      latch = SwingTimingLatch{};  // landed as scheduled (possibly within an extension), nothing to adapt
      continue;
    }
    bool handled = false;
    for (const auto& rule : rules) {
      if (rule->adaptContactFoot(ctx, foot, schedule, latch, report)) {
        handled = true;
        break;
      }
    }
    // No rule extended the swing (none is listed, or the extension budget is used up): the contact phase proceeds.
    if (!handled) latch = SwingTimingLatch{};
  }
  return reports;
}

}  // namespace ocs2::humanoid
