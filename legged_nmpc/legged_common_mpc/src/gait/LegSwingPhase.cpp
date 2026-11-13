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

#include "legged_common_mpc/gait/LegSwingPhase.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <vector>

#include "absl/types/span.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "legged_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::legged {
namespace {

/** The number of phases of `schedule` that have a mode: phase p runs from
 * eventTimes[p - 1] to eventTimes[p]. Bounded by both sizes, because nothing
 * enforces N modes for N - 1 events. */
size_t numPhasesOf(const ModeSchedule& schedule) {
  return std::min(schedule.modeSequence.size(), schedule.eventTimes.size() + 1);
}

bool legIsInSwingInPhase(const ModeSchedule& schedule,
                         size_t phase,
                         size_t numContacts,
                         absl::Span<const size_t> legContactIndices) {
  return legIsInSwingInMode(schedule.modeSequence[phase], numContacts,
                            legContactIndices);
}

}  // namespace

bool legIsInSwingInMode(size_t modeNumber,
                        size_t numContacts,
                        absl::Span<const size_t> legContactIndices) {
  if (legContactIndices.empty()) return false;
  for (const size_t contact : legContactIndices) {
    if (contact < numContacts &&
        contactIsActiveInMode(modeNumber, contact, numContacts)) {
      return false;
    }
  }
  return true;
}

std::optional<LegSwingWindow> legSwingWindowAtTime(
    const ModeSchedule& schedule,
    size_t numContacts,
    absl::Span<const size_t> legContactIndices,
    scalar_t time) {
  const size_t numPhases = numPhasesOf(schedule);
  if (numPhases == 0) return std::nullopt;
  // The phase ModeSchedule::modeAtTime() reads at `time`.
  const std::vector<scalar_t>& eventTimes = schedule.eventTimes;
  const size_t now =
      std::min(static_cast<size_t>(std::lower_bound(eventTimes.begin(),
                                                    eventTimes.end(), time) -
                                   eventTimes.begin()),
               numPhases - 1);
  if (!legIsInSwingInPhase(schedule, now, numContacts, legContactIndices)) {
    return std::nullopt;
  }
  // Several phases belong to one swing when other legs switch while this one
  // is in the air.
  size_t first = now;
  while (first > 0 && legIsInSwingInPhase(schedule, first - 1, numContacts,
                                          legContactIndices)) {
    --first;
  }
  size_t last = now;
  while (
      last + 1 < numPhases &&
      legIsInSwingInPhase(schedule, last + 1, numContacts, legContactIndices)) {
    ++last;
  }
  if (first == 0 || last + 1 >= numPhases) return std::nullopt;
  return LegSwingWindow{.liftOff = eventTimes[first - 1],
                        .touchDown = eventTimes[last]};
}

scalar_t legSwingPhaseVariable(const ModeSchedule& schedule,
                               size_t numContacts,
                               absl::Span<const size_t> legContactIndices,
                               scalar_t time) {
  const std::optional<LegSwingWindow> swing =
      legSwingWindowAtTime(schedule, numContacts, legContactIndices, time);
  if (!swing.has_value() || !(swing->touchDown > swing->liftOff)) return 0.0;
  return std::clamp(
      (time - swing->liftOff) / (swing->touchDown - swing->liftOff), 0.0, 1.0);
}

scalar_t legSwingPhaseVariableRate(const ModeSchedule& schedule,
                                   size_t numContacts,
                                   absl::Span<const size_t> legContactIndices,
                                   scalar_t time) {
  const std::optional<LegSwingWindow> swing =
      legSwingWindowAtTime(schedule, numContacts, legContactIndices, time);
  if (!swing.has_value() || !(swing->touchDown > swing->liftOff)) return 0.0;
  return 1.0 / (swing->touchDown - swing->liftOff);
}

scalar_t legUpcomingStanceDuration(const ModeSchedule& schedule,
                                   size_t numContacts,
                                   absl::Span<const size_t> legContactIndices,
                                   scalar_t touchDownTime) {
  const size_t numPhases = numPhasesOf(schedule);
  // The transition between phase p and p + 1 is eventTimes[p].
  bool hadLiftOff = false;
  std::optional<scalar_t> stanceStart;
  std::optional<scalar_t> recentStanceDuration;
  for (size_t phase = 0; phase + 1 < numPhases; ++phase) {
    const bool swingBefore =
        legIsInSwingInPhase(schedule, phase, numContacts, legContactIndices);
    const bool swingAfter = legIsInSwingInPhase(schedule, phase + 1,
                                                numContacts, legContactIndices);
    const scalar_t eventTime = schedule.eventTimes[phase];
    if (!swingBefore && swingAfter) {
      if (eventTime > touchDownTime) return eventTime - touchDownTime;
      // A complete stance: a touch-down after a lift-off, and this lift-off.
      if (stanceStart.has_value()) {
        recentStanceDuration = eventTime - *stanceStart;
      }
      hadLiftOff = true;
      stanceStart.reset();
    } else if (swingBefore && !swingAfter && hadLiftOff) {
      stanceStart = eventTime;
    }
  }
  // The schedule ends before this stance does: the most recent complete
  // stance before it stands in.
  return recentStanceDuration.value_or(0.0);
}

std::optional<scalar_t> legPreviousTouchDownTime(
    const ModeSchedule& schedule,
    size_t numContacts,
    absl::Span<const size_t> legContactIndices,
    scalar_t time) {
  const size_t numPhases = numPhasesOf(schedule);
  std::optional<scalar_t> latest;
  for (size_t phase = 0; phase + 1 < numPhases; ++phase) {
    const scalar_t eventTime = schedule.eventTimes[phase];
    if (eventTime > time) break;
    if (legIsInSwingInPhase(schedule, phase, numContacts, legContactIndices) &&
        !legIsInSwingInPhase(schedule, phase + 1, numContacts,
                             legContactIndices)) {
      latest = eventTime;
    }
  }
  return latest;
}

}  // namespace ocs2::legged
