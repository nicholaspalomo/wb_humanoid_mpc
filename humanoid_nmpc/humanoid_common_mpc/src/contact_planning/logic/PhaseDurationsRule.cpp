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
    // The (fractional, possibly negative) node index at which the CURRENT phase began. The tie break below needs the
    // phase's age rounded to the NEAREST node, which is neither of the two counts above, and it cannot be
    // reconstructed from them: it used to be recovered as `lround(phaseElapsedTime / dt) + (tauMin - initialPhaseNodes)`,
    // an identity that holds only while the phase at node k is still the phase that was active at planning time. Once
    // the phase switched inside the horizon, tauMin is reseeded from switchedPhaseNodes() and that difference stops
    // meaning "nodes since the phase began": the integer part then came from the current phase while the rounding bit
    // still came from a phase that had already finished. Tracking the start directly is correct on both sides of a
    // switch and needs no clamp.
    scalar_t phaseStartNode = -std::max(0.0, s.input->phaseElapsedTime[foot]) / s.dt;
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
          const int tauNearest = static_cast<int>(std::lround(static_cast<scalar_t>(k) - phaseStartNode));
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
          // Whether this foot is pinned down by the minimum double support, read LIVE and not only from the per-pass
          // scan. Every other unknown in this block - otherValue, stateBefore, contactAge - is read from the live
          // assignment, exactly as ContactPlanningProblem::propagate documents ("within a pass a rule reads the live
          // assignment"); this one read was the exception, and it is the one that decides whether the foot may lift.
          //
          // Two things make the snapshot too weak. It is computed once per pass, before any rule runs, so a touch-down
          // that THIS rule has just fixed for the other foot is invisible - and since the rule iterates feet
          // outermost, foot 0's whole node loop finishes before foot 1's begins. Worse, ContactLogicScan::compute
          // stops at the first node whose binaries are not all fixed, so beyond the fixed prefix the snapshot is
          // uniformly false. The yield that exists precisely for "the other foot lands right here" was therefore
          // bypassed, this foot was fixed to lift at the same node, and MinimumDoubleSupportRule - which recomputes
          // the touch-down live - then returned false and the node was pruned as infeasible when it was not.
          scalar_t latestTouchDownNode = s.initialTouchDownNode;
          if (otherValue == 1 && s.stateBefore(a, other, k) == 0) {
            latestTouchDownNode = std::max(latestTouchDownNode, s.switchNode(k, other));
          }
          const bool held = scan.heldByDoubleSupport[foot][static_cast<size_t>(k)] ||
                            (s.nDoubleSupportHold > 0 && s.heldAfterTouchDown(k, latestTouchDownNode));
          const int otherAge = s.contactAge(a, other, k);
          const bool otherOverdue = otherValue != 0 && s.nContactMax > 0 && otherAge >= s.nContactMax;
          bool thisGoesFirst = true;
          if (otherOverdue) {
            if (otherValue == kMiqpFree) {
              // BOTH FEET ARE OVERDUE AND THE ORDER IS STILL OPEN, so this rule must not pick one.
              //
              // The tie break below is a PREFERENCE - who swung last, else who has stood longer, else the lower foot
              // index - and a propagator may only fix what is IMPLIED. MiqpPropagateFn documents its false return as
              // "provably infeasible", and by the same token a fixing has to hold in every feasible completion. When
              // both feet are overdue and the other foot's binary at this node is still free, lifting either one
              // first is feasible; forcing this one closed the subtree that lifts the other first, and the
              // branch-and-bound never looked at it. The preference belongs in the objective (ContactSwitchCost and
              // the assignment costs already price the choice), not in propagation.
              //
              // Once the other foot's value at this node IS known - which is always so for a complete assignment -
              // the tie break resolves a real conflict rather than a free choice, so the behaviour there is unchanged
              // and the recorded logic fixture is untouched.
              thisGoesFirst = false;
            } else if (s.input->lastSwungFoot >= 0) {
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
