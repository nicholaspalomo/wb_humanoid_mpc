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
 * `hop_on_request`: while a hop is requested (ContactPlannerInput::hopRequested, set by the planner module when the
 * commanded base height exceeds hop_on_request.triggerBaseHeight), both feet lift together as soon as they may: a
 * double support that has lasted the minimum contact duration after the commit window ends in a flight. The flight's
 * length is the requested one (flight_durations enforces it, with maxFlightDuration as the cap). Without a request the
 * rule does nothing, so it can stay listed.
 */
class HopOnRequestRule final : public ContactLogicRule {
 public:
  std::string describe() const override;
  void configure(const ContactPlanningConfig& config) override;
  bool propagate(const ContactLogicState& state, const ContactLogicScan& scan, MiqpAssignment& a, bool& changed) const override;

 private:
  scalar_t triggerBaseHeight_ = 0.0;
  scalar_t flightDuration_ = 0.0;
};

}  // namespace ocs2::humanoid
