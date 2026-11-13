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
#include <optional>

#include "absl/types/span.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

namespace ocs2::legged {

/**
 * The schedule queries of one leg of any contact set
 * (ModelSettings::legs): a leg is in swing in a mode when none of its contacts
 * is active in it (contactIsActiveInMode()), and in stance when any is. The
 * mode at a time is the one ModeSchedule::modeAtTime() gives, so at an event
 * time the event is still ahead: a leg is in stance at the instant it lifts off
 * and in swing at the instant it touches down.
 *
 * Each function takes the leg as the indices of its contacts into a set of
 * `numContacts` contacts. An index at or above `numContacts` is never active.
 */

/** One swing of a leg in continuous time: [liftOff, touchDown]. */
struct LegSwingWindow {
  scalar_t liftOff = 0.0;    // [s]
  scalar_t touchDown = 0.0;  // [s]
};

/** Returns whether every contact of the leg is inactive in `modeNumber`. A leg
 * of no contacts is never in swing. */
bool legIsInSwingInMode(size_t modeNumber,
                        size_t numContacts,
                        absl::Span<const size_t> legContactIndices);

/**
 * Returns the swing of the leg around `time`: from the event at which it last
 * lifted off to the event at which it next touches down. Empty while the leg is
 * in stance, and when the schedule holds no lift-off or no touch-down event for
 * that swing. Allocates nothing.
 */
std::optional<LegSwingWindow> legSwingWindowAtTime(
    const ModeSchedule& schedule,
    size_t numContacts,
    absl::Span<const size_t> legContactIndices,
    scalar_t time);

/**
 * Returns the phase of the leg's swing at `time`: (time - liftOff) /
 * (touchDown - liftOff), in [0, 1]. 0 in stance (the lift-off instant
 * included), and 0 where legSwingWindowAtTime() is empty or the window has no
 * duration.
 */
scalar_t legSwingPhaseVariable(const ModeSchedule& schedule,
                               size_t numContacts,
                               absl::Span<const size_t> legContactIndices,
                               scalar_t time);

/** Returns d legSwingPhaseVariable() / dt [1/s]: 1 / (touchDown - liftOff) in
 * swing, 0 wherever the phase is constant. */
scalar_t legSwingPhaseVariableRate(const ModeSchedule& schedule,
                                   size_t numContacts,
                                   absl::Span<const size_t> legContactIndices,
                                   scalar_t time);

/**
 * Returns the duration [s] of the stance the leg begins by touching down at
 * `touchDownTime`: from there to its next lift-off. If the schedule does not
 * reach that lift-off, the duration of the leg's most recent complete stance
 * that ends at or before `touchDownTime`; 0 if it has none. The leg version of
 * upcomingStanceDuration() (ContactScheduleAdaptation.h), which reads the
 * biped's two feet. Allocates nothing.
 */
scalar_t legUpcomingStanceDuration(const ModeSchedule& schedule,
                                   size_t numContacts,
                                   absl::Span<const size_t> legContactIndices,
                                   scalar_t touchDownTime);

/**
 * Returns the most recent touch-down [s] of the leg at or before `time`, empty
 * if the schedule has none: the leg version of previousTouchDownTime()
 * (ContactScheduleAdaptation.h). Allocates nothing.
 */
std::optional<scalar_t> legPreviousTouchDownTime(
    const ModeSchedule& schedule,
    size_t numContacts,
    absl::Span<const size_t> legContactIndices,
    scalar_t time);

}  // namespace ocs2::legged
