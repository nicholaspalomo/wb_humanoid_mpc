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

#include "humanoid_centroidal_mpc/multi_contact/PlanClock.h"

#include <algorithm>
#include <optional>
#include <utility>

namespace ocs2::humanoid {

PlanClock::PlanClock(std::vector<scalar_t> eventTimes,
                     scalar_t lead,
                     scalar_t startTime,
                     std::optional<scalar_t> period)
    : eventTimes_(std::move(eventTimes)),
      period_(period),
      lead_(std::max(lead, 0.0)),
      startTime_(startTime) {}

void PlanClock::reset() {
  started_ = false;
  offset_ = 0.0;
  nextEvent_ = 0;
  holding_ = false;
}

bool PlanClock::hasEvent(size_t event) const {
  if (period_.has_value()) return !eventTimes_.empty();
  return event < eventTimes_.size();
}

scalar_t PlanClock::eventTime(size_t event) const {
  if (!period_.has_value()) return eventTimes_[event];
  const size_t perPeriod = eventTimes_.size();
  return eventTimes_[event % perPeriod] +
         static_cast<scalar_t>(event / perPeriod) * *period_;
}

scalar_t PlanClock::holdTime(size_t event) const {
  // Never before the plan's start: a change within `lead` of it is judged at
  // the start.
  return std::max(eventTime(event) - lead_, startTime_);
}

scalar_t PlanClock::advance(
    scalar_t time,
    absl::FunctionRef<bool(size_t event, scalar_t holdTime)> isReady) {
  if (!started_) {
    offset_ = time - startTime_;
    started_ = true;
  }
  scalar_t planTime = time - offset_;
  holding_ = false;
  if (hasEvent(nextEvent_)) {
    const scalar_t hold = holdTime(nextEvent_);
    if (planTime >= hold) {
      if (isReady(nextEvent_, hold)) {
        ++nextEvent_;
        // One change per solve: the clock stops at the next hold point, to be
        // judged at the next solve.
        if (hasEvent(nextEvent_)) {
          planTime = std::min(planTime, holdTime(nextEvent_));
        }
      } else {
        planTime = hold;
        holding_ = true;
      }
    }
  }
  offset_ = time - planTime;
  return planTime;
}

}  // namespace ocs2::humanoid
