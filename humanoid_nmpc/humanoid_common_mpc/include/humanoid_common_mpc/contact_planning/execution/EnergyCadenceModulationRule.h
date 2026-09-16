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
 * `energy_cadence_modulation`: the touch-down of the swing in flight is moved by the LIP orbital energy error between
 * the measured centre of mass and the NMPC's prediction, relative to its nominal touch-down and within the swing
 * duration limits; later events move with it. Must be listed after `phase_resetting` when both are on: an early
 * touch-down ends a swing before the cadence rule may re-time it.
 */
class EnergyCadenceModulationRule final : public ExecutionRule {
 public:
  std::string describe() const override;
  void configure(const ContactPlanningConfig& config) override;
  bool needsPredictedTrajectory() const override { return true; }
  void beginCycle(ExecutionContext& ctx) const override;
  bool adaptSwingingFoot(const ExecutionContext& ctx,
                         size_t foot,
                         scalar_t liftOff,
                         scalar_t touchDown,
                         ModeSchedule& schedule,
                         SwingTimingLatch& latch,
                         ContactEventReport& report) const override;

 private:
  EnergyCadenceModulationParameters params_;
  GaitLimits limits_;
};

}  // namespace ocs2::humanoid
