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

#include "humanoid_common_mpc/contact_planning/logic/PhaseDurationsRule.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "humanoid_common_mpc/contact_planning/logic/ContactLogicHelpers.h"

namespace ocs2::humanoid {

static_assert(N_CONTACTS == 2, "the phase duration rule is written for a biped (the other foot is 1 - foot)");

std::string PhaseDurationsRule::describe() const {
  std::ostringstream out;
  out << "swing in [" << limits_.minSwingDuration << ", " << limits_.maxSwingDuration << "] s, contact >= " << limits_.minContactDuration
      << " s";
  if (limits_.maxContactDuration > 0.0) out << " and <= " << limits_.maxContactDuration << " s (yields to no flight / double support)";
  out << ", no lift-off that cannot swing the minimum before the horizon ends";
  return out.str();
}

void PhaseDurationsRule::configure(const ContactPlanningConfig& config) {
  limits_ = config.shared.gaitLimits;
}

bool PhaseDurationsRule::propagate(const ContactLogicState& s, const ContactLogicScan& scan, MiqpAssignment& a, bool& changed) const {
  using S = ContactLogicState;
  const int N = s.numNodes;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    std::int8_t kappa = s.input->contacts[foot] ? 1 : 0;
    // The phase active at planning time has lasted a non-integer number of nodes: count it rounded down against the
    // minimum duration and rounded up against the maximum, so that neither limit is violated by the grid. Once the
    // phase switched inside the horizon both counts are exact and coincide.
    int tauMin = s.initialPhaseNodes(foot, false);
    int tauMax = s.initialPhaseNodes(foot, true);
    for (int k = 0; k < N; ++k) {
      const int index = S::contactBinaryIndex(k, foot);
      if (k >= s.numCommitted) {
        bool mustStay = (kappa == 1 && tauMin < s.nContactMin) || (kappa == 0 && tauMin < s.nSwingMin);
        bool mustSwitch = (kappa == 0 && tauMax >= s.nSwingMax) || (kappa == 1 && s.nContactMax > 0 && tauMax >= s.nContactMax);
        // A swing that starts this late cannot reach its minimum duration before the horizon ends. Letting it start
        // leaves the plan ending mid-swing, and ContactPlan::toModeSchedule() then closes the schedule with a
        // touch-down at endTime(), handing the controller a swing far shorter than minSwingDuration. Keep the foot
        // down instead and let the next plan start the step; the maximum contact duration yields to this.
        if (kappa == 1 && k + s.nSwingMin > N) {
          mustStay = true;
          mustSwitch = false;
        }
        if (mustStay && mustSwitch) {
          // The grid cannot honour both limits for this phase (they are less than a node apart); decide by the nearest
          // node instead of declaring the whole horizon infeasible.
          const int tauNearest = static_cast<int>(std::lround(std::max(0.0, s.input->phaseElapsedTime[foot]) / s.dt)) +
                                 (tauMin - s.initialPhaseNodes(foot, false));
          mustStay = (kappa == 1 && tauNearest < s.nContactMin) || (kappa == 0 && tauNearest < s.nSwingMin);
          mustSwitch = (kappa == 0 && tauNearest >= s.nSwingMax) || (kappa == 1 && s.nContactMax > 0 && tauNearest >= s.nContactMax);
          if (mustStay && mustSwitch) return false;
        }
        if (mustSwitch && kappa == 1) {
          // Overdue to lift. Yield to the rules that can make lifting impossible right now, and when both feet are
          // overdue at once let the one that has stood longer (or that did not swing last) go first.
          const size_t other = 1 - foot;
          const std::int8_t otherValue = a[static_cast<size_t>(S::contactBinaryIndex(k, other))];
          // The other foot has to be known to support: in contact at this node, or in contact before it with its value
          // here still free (it can then only stay, since this foot lifting forbids it to lift too). If it is in the air
          // and free it might land right here, and lifting now would violate the double support.
          const bool otherSupports = otherValue == 1 || (otherValue == kMiqpFree && s.stateBefore(a, other, k) == 1);
          const bool held = scan.heldByDoubleSupport[foot][static_cast<size_t>(k)];
          const int otherAge = s.contactAge(a, other, k);
          const bool otherOverdue = otherValue != 0 && s.nContactMax > 0 && otherAge >= s.nContactMax;
          bool thisGoesFirst = true;
          if (otherOverdue) {
            if (s.input->lastSwungFoot >= 0) {
              thisGoesFirst = static_cast<size_t>(s.input->lastSwungFoot) != foot;
            } else {
              const int thisAge = s.contactAge(a, foot, k);
              thisGoesFirst = thisAge > otherAge || (thisAge == otherAge && foot < other);
            }
          }
          if (!otherSupports || held || !thisGoesFirst) mustSwitch = false;
        }
        if (mustStay && !fixBinary(a, index, kappa, changed)) return false;
        if (mustSwitch && !fixBinary(a, index, static_cast<std::int8_t>(1 - kappa), changed)) return false;
      }
      const std::int8_t value = a[static_cast<size_t>(index)];
      if (value == kMiqpFree) break;
      if (value == kappa) {
        ++tauMin;
        ++tauMax;
      } else {
        tauMin = s.switchedPhaseNodes(k, foot, false);
        tauMax = s.switchedPhaseNodes(k, foot, true);
        kappa = value;
      }
    }
  }
  return true;
}

}  // namespace ocs2::humanoid
