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

#include "humanoid_common_mpc/contact_planning/logic/FlightDurationsRule.h"

#include <sstream>

#include "humanoid_common_mpc/contact_planning/logic/ContactLogicHelpers.h"

namespace ocs2::humanoid {

static_assert(N_CONTACTS == 2, "the flight rule is written for a biped");

std::string FlightDurationsRule::describe() const {
  std::ostringstream out;
  if (maxFlightDuration_ <= 0.0) return "no flight (maxFlightDuration is 0): c_L + c_R >= 1 at every node";
  out << "flight (c_L = c_R = 0) lasts " << minFlightDuration_ << " to " << maxFlightDuration_
      << " s, ends before the horizon; a hop request raises the minimum to the requested flight";
  if (allowedAboveSpeed_ > 0.0) out << "; on the ground below " << allowedAboveSpeed_ << " m/s unless a hop is requested";
  return out.str();
}

void FlightDurationsRule::configure(const ContactPlanningConfig& config) {
  minFlightDuration_ = config.shared.gaitLimits.minFlightDuration;
  maxFlightDuration_ = config.shared.gaitLimits.maxFlightDuration;
  allowedAboveSpeed_ = config.flightDurations.allowedAboveSpeed;
}

bool FlightDurationsRule::propagate(const ContactLogicState& s, const ContactLogicScan& /*scan*/, MiqpAssignment& a, bool& changed) const {
  using S = ContactLogicState;
  const int N = s.numNodes;
  const int nMin = s.nFlightMin;
  // Flight the plan may use at all: none below the speed that makes it worth the search, unless a hop was asked for.
  const bool gateOpen = s.input->hopRequested || s.input->velocityCommand.norm() >= allowedAboveSpeed_;
  const int nMax = gateOpen ? s.nFlightMax : 0;
  // A flight in progress at planning time started at the later lift-off: count it like the other initial phases.
  int runMin = 0;
  int runMax = 0;
  if (!s.input->contacts[0] && !s.input->contacts[1]) {
    runMin = std::min(s.initialPhaseNodes(0, false), s.initialPhaseNodes(1, false));
    runMax = std::min(s.initialPhaseNodes(0, true), s.initialPhaseNodes(1, true));
  }
  for (int k = 0; k < N; ++k) {
    const int iL = S::contactBinaryIndex(k, 0), iR = S::contactBinaryIndex(k, 1);
    std::int8_t vL = a[static_cast<size_t>(iL)], vR = a[static_cast<size_t>(iR)];
    const bool free_ = k >= s.numCommitted;
    if (nMax <= 0 || k == N - 1) {
      // No flight at all, or not on the last interval: a flight closed by the horizon would land on the terminal node,
      // which carries no contact-height row, and the plan could end in free fall.
      if (vL == 0 && vR == 0) return false;
      if (vL == 0 && !fixBinary(a, iR, 1, changed)) return false;
      if (vR == 0 && !fixBinary(a, iL, 1, changed)) return false;
      if (nMax <= 0) continue;  // every node is checked, whether the prefix is fixed or not (as no_flight does)
    }
    const bool inFlight = runMax > 0 || runMin > 0;
    if (inFlight && free_) {
      // The flight must go on until it has lasted the minimum, and must end once it has lasted the maximum.
      if (runMin < nMin) {
        if (vL == 1 || vR == 1) return false;
        if (!fixBinary(a, iL, 0, changed) || !fixBinary(a, iR, 0, changed)) return false;
      } else if (runMax >= nMax) {
        if (vL == 0 && vR == 0) return false;
        if (vL == 0 && !fixBinary(a, iR, 1, changed)) return false;
        if (vR == 0 && !fixBinary(a, iL, 1, changed)) return false;
      }
    } else if (!inFlight && free_) {
      // A flight starting here must be able to reach its minimum and land before the horizon ends.
      if (k + nMin > N - 1) {
        if (vL == 0 && vR == 0) return false;
        if (vL == 0 && !fixBinary(a, iR, 1, changed)) return false;
        if (vR == 0 && !fixBinary(a, iL, 1, changed)) return false;
      }
    }
    vL = a[static_cast<size_t>(iL)];
    vR = a[static_cast<size_t>(iR)];
    if (vL == kMiqpFree || vR == kMiqpFree) break;  // the rest of the horizon is not fixed yet
    if (vL == 0 && vR == 0) {
      if (inFlight) {
        ++runMin;
        ++runMax;
      } else {
        runMin = runMax = 1;
      }
    } else {
      runMin = runMax = 0;
    }
  }
  return true;
}

}  // namespace ocs2::humanoid
