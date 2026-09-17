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

#include "humanoid_common_mpc/contact_planning/problem/ContactLogicRule.h"

namespace ocs2::humanoid {

/**
 * `flight_durations`: a flight (no foot in contact) lasts between minFlightDuration and maxFlightDuration, counted in
 * nodes like the other phases, and never starts so late that it cannot end before the horizon does. With
 * maxFlightDuration at 0 no flight is allowed and the rule is `no_flight`. A hop request (ContactPlannerInput) raises
 * the minimum to the requested flight. Replaces `no_flight` in the logic list when the flight model is on; the QP row
 * of `no_flight` must then be dropped as well.
 *
 * Flight also has to be worth its search: below `allowedAboveSpeed` the rule keeps a foot on the ground at every node,
 * so a robot that lists the flight model walks exactly as it does without it and only turns into a runner once the
 * command asks for a speed its cadence cannot reach on the ground. A hop request opens the gate at any speed.
 */
class FlightDurationsRule final : public ContactLogicRule {
 public:
  std::string describe() const override;
  void configure(const ContactPlanningConfig& config) override;
  bool propagate(const ContactLogicState& state, const ContactLogicScan& scan, MiqpAssignment& a, bool& changed) const override;

 private:
  scalar_t minFlightDuration_ = 0.0;
  scalar_t maxFlightDuration_ = 0.0;
  scalar_t allowedAboveSpeed_ = 0.0;
};

}  // namespace ocs2::humanoid
