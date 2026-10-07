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

#include "humanoid_common_mpc/contact_planning/logic/PhaseDurationsRule.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/contact_planning/logic/ContactLogicHelpers.h"

namespace ocs2::humanoid {

static_assert(kNumContacts == 2, "the phase duration rule is written for a biped (the other foot is 1 - foot)");

std::string PhaseDurationsRule::describe() const {
  std::string out = absl::StrCat("swing in [", limits_.minSwingDuration, ", ", limits_.maxSwingDuration,
                                 "] s, contact >= ", limits_.minContactDuration, " s");
  if (limits_.maxContactDuration > 0.0) {
    absl::StrAppend(&out, " and <= ", limits_.maxContactDuration, " s (yields to no flight / double support)");
  }
  absl::StrAppend(&out, ", no lift-off that cannot swing the minimum before the horizon ends");
  return out;
}

void PhaseDurationsRule::configure(const ContactPlanningConfig& config) {
  limits_ = config.shared.gaitLimits;
}

bool PhaseDurationsRule::propagate(const ContactLogicState& s, MiqpAssignment& a, bool& changed) const {
  const int N = s.numNodes;
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    int8_t kappa = s.input->contacts[foot] ? 1 : 0;
    // The phase active at planning time has lasted a non-integer number of nodes: count it rounded down against the
    // minimum duration and rounded up against the maximum, so that neither limit is violated by the grid. Once the
    // phase switched inside the horizon both counts are exact and coincide.
    int tauMin = s.initialPhaseNodes(foot, /*roundUp=*/false);
    int tauMax = s.initialPhaseNodes(foot, /*roundUp=*/true);
    // The (fractional, possibly negative) node index at which the CURRENT phase began. The tie break below needs the
    // phase's age rounded to the NEAREST node, which is neither of the two counts above, and it cannot be
    // reconstructed from them: it used to be recovered as `lround(phaseElapsedTime / dt) + (tauMin - initialPhaseNodes)`,
    // an identity that holds only while the phase at node k is still the phase that was active at planning time. Once
    // the phase switched inside the horizon, tauMin is reseeded from switchedPhaseNodes() and that difference stops
    // meaning "nodes since the phase began": the integer part then came from the current phase while the rounding bit
    // still came from a phase that had already finished. Tracking the start directly is correct on both sides of a
    // switch and needs no clamp.
    scalar_t phaseStartNode = -std::max(0.0, s.input->phaseElapsedTime[foot]) / s.dt;
    // The loop walks this foot's fixed prefix: every node before k is fixed for this foot, and it stops at the first
    // node whose binary is free.
    for (int k = 0; k < N; ++k) {
      const int index = ContactLogicState::contactBinaryIndex(k, foot);
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
          // The grid cannot honor both limits for this phase (they are less than a node apart); decide by the nearest
          // node instead of declaring the whole horizon infeasible.
          const int tauNearest = static_cast<int>(std::lround(static_cast<scalar_t>(k) - phaseStartNode));
          mustStay = (kappa == 1 && tauNearest < s.nContactMin) || (kappa == 0 && tauNearest < s.nSwingMin);
          mustSwitch = (kappa == 0 && tauNearest >= s.nSwingMax) || (kappa == 1 && s.nContactMax > 0 && tauNearest >= s.nContactMax);
          if (mustStay && mustSwitch) return false;
        }
        if (mustSwitch && kappa == 1) {
          // Overdue to lift. On a complete assignment the foot has to lift at this node unless one of three things
          // excuses it: the other foot is not supporting (it is in the air here), the minimum double support after
          // the latest touch-down of either foot still holds this foot down, or the other foot is overdue too and
          // the tie break lets it go first. All three are functions of the OTHER foot's binaries up to and including
          // this node (and of this foot's, which are fixed before this node here).
          //
          // On a partial assignment the lift may therefore be FIXED only once the other foot is fixed through this
          // node and supports here: then all three are decided exactly as they will be in every completion, and the
          // lift is implied. Before that it is not. This rule used to decide them early - it took the other foot as
          // supporting while its binary here was still free (on a complete assignment the other foot lifting at this
          // node excuses this one), it treated a history it could not read as "no touch-down" (skipping the hold
          // after one), and it read the hold from a per-pass snapshot of the prefix that missed a touch-down this
          // very pass had fixed. Each of those fixed binaries that some feasible completions disagree with, so the
          // branch-and-bound pruned feasible subtrees - the other foot stepping first after a push, for one - and
          // still reported the plan optimal. See MiqpPropagateFn and ContactLogicRule for the contract, and
          // testContactPlanningRegression for the enumeration over the partial assignments the search reaches.
          const size_t other = 1 - foot;
          const int8_t otherValue = a[static_cast<size_t>(ContactLogicState::contactBinaryIndex(k, other))];
          const bool otherDecided = otherValue != kMiqpFree && s.stateBefore(a, other, k) != -1;
          if (!otherDecided || otherValue == 0) {
            mustSwitch = false;
          } else {
            const bool held = s.nDoubleSupportHold > 0 && s.heldAfterTouchDown(k, s.latestTouchDownNode(a, k));
            const int otherAge = s.contactAge(a, other, k);
            const bool otherOverdue = s.nContactMax > 0 && otherAge >= s.nContactMax;
            bool thisGoesFirst = true;
            if (otherOverdue) {
              // Both feet are overdue and both support here: one of them has to lift, and the tie break - who swung
              // last, else who has stood longer, else the lower foot index - decides which. On a complete assignment
              // that is the rule itself, not a preference, since exactly one order satisfies it; the other foot's
              // own pass forces its lift when it wins.
              if (s.input->lastSwungFoot >= 0) {
                thisGoesFirst = static_cast<size_t>(s.input->lastSwungFoot) != foot;
              } else {
                const int thisAge = s.contactAge(a, foot, k);
                thisGoesFirst = thisAge > otherAge || (thisAge == otherAge && foot < other);
              }
            }
            if (held || !thisGoesFirst) mustSwitch = false;
          }
        }
        if (mustStay && !fixBinary(a, index, kappa, changed)) return false;
        if (mustSwitch && !fixBinary(a, index, static_cast<int8_t>(1 - kappa), changed)) return false;
      }
      const int8_t value = a[static_cast<size_t>(index)];
      if (value == kMiqpFree) break;
      if (value == kappa) {
        ++tauMin;
        ++tauMax;
      } else {
        tauMin = s.switchedPhaseNodes(k, foot, /*roundUp=*/false);
        tauMax = s.switchedPhaseNodes(k, foot, /*roundUp=*/true);
        // The new phase begins at the switch, which is where switchedPhaseNodes() measures its two counts from; the
        // nearest-node count the tie break uses has to be measured from the same place.
        phaseStartNode = s.switchNode(k, foot);
        kappa = value;
      }
    }
  }
  return true;
}

}  // namespace ocs2::humanoid
