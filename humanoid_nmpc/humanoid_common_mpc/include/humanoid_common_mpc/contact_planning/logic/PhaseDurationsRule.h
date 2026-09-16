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

#include "humanoid_common_mpc/contact_planning/problem/ContactLogicRule.h"

namespace ocs2::humanoid {

/**
 * `phase_durations`: minimum and maximum swing duration, minimum (and optionally maximum) contact duration, counted in
 * nodes from the start of the phase including the time spent before the planning instant; a foot may not lift in the
 * last minSwingNodes - 1 nodes of the horizon (a swing that cannot reach its minimum before the horizon ends is
 * deferred to the next plan); and the maximum contact duration yields to the rules that can make lifting impossible
 * (no flight, the double-support hold, the other foot being overdue too). Written for a biped.
 */
class PhaseDurationsRule final : public ContactLogicRule {
 public:
  std::string describe() const override;
  void configure(const ContactPlanningConfig& config) override;
  bool propagate(const ContactLogicState& state, const ContactLogicScan& scan, MiqpAssignment& a, bool& changed) const override;

 private:
  GaitLimits limits_;
};

}  // namespace ocs2::humanoid
