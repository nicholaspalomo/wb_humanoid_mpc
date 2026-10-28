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
#include <vector>

#include "absl/functional/function_ref.h"

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The clock of a tracked multi-contact plan (design section 7.2,
 * humanoid_nmpc/docs/multi_contact_planning/README.md): which plan time the
 * MPC tracks at each solve. The plan starts at the first solve and runs at the
 * MPC's rate, until it reaches `lead` [s] before its next contact change. There
 * the stance the robot is on is judged: when the robot has reached it (the
 * caller's readiness, at the plan time of the hold), the change is released
 * and the clock runs on; otherwise the clock holds at that plan time, solve
 * after solve, until it has. A contact change is never skipped: at most one is
 * released per solve, and the plan time never passes the hold point of a
 * change that is not released. A periodic plan's changes repeat every period,
 * forever: change k is change k mod n of the period k / n (n the changes of a
 * period), and is held and judged like any other, across the wrap too.
 *
 * Not thread-safe: advance() runs on the solver thread before each solve; the
 * const methods read what it latched.
 */
class PlanClock {
 public:
  /**
   * The clock of a plan whose stance changes at the increasing plan times
   * `eventTimes` [s], held `lead` [s] (not negative) before each, the plan
   * starting at `startTime` [s]; with a `period` [s] (positive, longer than
   * the last event's time minus the start), the changes repeat every period.
   */
  PlanClock(std::vector<scalar_t> eventTimes,
            scalar_t lead,
            scalar_t startTime,
            std::optional<scalar_t> period = std::nullopt);

  /**
   * The plan time to track at the solve of time `time` [s] (not decreasing
   * from call to call), `isReady(event, holdTime)` judging whether the stance
   * before the change `event` (an index of the event times) has been reached
   * when the clock comes to its hold point, the plan time `holdTime`. The
   * first call starts the plan at `time`.
   */
  scalar_t advance(
      scalar_t time,
      absl::FunctionRef<bool(size_t event, scalar_t holdTime)> isReady);

  /** Starts the plan anew at the next advance(). */
  void reset();

  /** [s] time - plan time at the last advance(): the plan time t' - offset()
   * belongs to the time t'. */
  scalar_t offset() const { return offset_; }
  /** Whether the last advance() held the clock. */
  bool isHolding() const { return holding_; }
  /** The number of changes released so far: the index of the next one. */
  size_t releasedEvents() const { return nextEvent_; }
  /** [s] The plan time of the hold point of change `event`. */
  scalar_t holdTime(size_t event) const;
  /** [s] The plan time of change `event`, counted from the start over every
   * period. */
  scalar_t eventTime(size_t event) const;

 private:
  /** Whether the plan has a change `event`. */
  bool hasEvent(size_t event) const;

  std::vector<scalar_t> eventTimes_;  // increasing
  std::optional<scalar_t> period_;    // [s] positive, for a periodic plan
  scalar_t lead_ = 0.0;               // [s] not negative
  scalar_t startTime_ = 0.0;          // [s] the plan's first knot
  bool started_ = false;
  scalar_t offset_ = 0.0;
  size_t nextEvent_ = 0;  // the events before it are released
  bool holding_ = false;
};

}  // namespace ocs2::humanoid
