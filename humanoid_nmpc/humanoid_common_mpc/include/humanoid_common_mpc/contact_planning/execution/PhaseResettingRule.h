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

namespace ocs2::humanoid {

/**
 * `phase_resetting`: early touch-down, a swing foot whose measured contact persists for the debounce duration (after
 * the scuffing window) is switched to contact at once, in place; late touch-down, a foot that misses the ground at its
 * scheduled touch-down keeps swinging in small steps, up to a bounded total extension, every later event delayed by
 * the same amount, its height reference descending at the search velocity.
 */
class PhaseResettingRule final : public ExecutionRule {
 public:
  std::string describe() const override;
  void configure(const ContactPlanningConfig& config) override;
  bool adaptSwingingFoot(const ExecutionContext& ctx,
                         size_t foot,
                         scalar_t liftOff,
                         scalar_t touchDown,
                         ModeSchedule& schedule,
                         SwingTimingLatch& latch,
                         ContactEventReport& report) const override;
  bool adaptContactFoot(
      const ExecutionContext& ctx, size_t foot, ModeSchedule& schedule, SwingTimingLatch& latch, ContactEventReport& report) const override;
  std::optional<GroundSearchRequest> groundSearch(const ExecutionContext& ctx,
                                                  size_t foot,
                                                  const ModeSchedule& schedule,
                                                  const SwingTimingLatch& latch) const override;

 private:
  PhaseResettingParameters params_;
};

}  // namespace ocs2::humanoid
