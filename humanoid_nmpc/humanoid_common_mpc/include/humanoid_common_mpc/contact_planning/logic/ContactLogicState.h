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

#include <array>
#include <vector>

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/MixedIntegerOcpQp.h"

namespace ocs2::humanoid {

/**
 * Pre-computation of the contact logic layer, built once per plan from the input and the grid: the node clocks of the
 * phases (with the conservative rounding of a phase that started off-grid), the duration limits in nodes, the committed
 * prefix, and the previous plan for the assignment costs. The rules and the assignment costs read from it; the helpers
 * on a live assignment (stateBefore, contactAge) read the assignment as it is at the time of the call, which is what the
 * coupling between the rules relies on (an overdue foot yields to what the other rules have already fixed in this pass).
 *
 * The binaries are laid out node-major: index N_CONTACTS * node + foot (contactBinaryIndex).
 */
struct ContactLogicState {
  const ContactPlannerInput* input = nullptr;
  int numNodes = 0;
  scalar_t dt = 0.1;
  int nSwingMin = 1, nSwingMax = 1, nContactMin = 1, nContactMax = 0, nDoubleSupportHold = 0;
  int nFlightMin = 1, nFlightMax = 0;    // flight limits in nodes; nFlightMax 0: no flight (a hop request raises nFlightMin)
  scalar_t minDoubleSupportNodes = 0.0;  // fractional, minDoubleSupportDuration / dt
  int numCommitted = 0;
  scalar_t initialTouchDownNode = -1000.0;  // touch-down node of a double support already in progress at planning time
  // Previous plan, for the plan-consistency cost: -1 / null when there is no usable one.
  int previousPlanShift = -1;
  const MiqpAssignment* previousAssignment = nullptr;

  static constexpr int kBinariesPerNode = static_cast<int>(N_CONTACTS);
  static int contactBinaryIndex(int node, size_t foot) { return kBinariesPerNode * node + static_cast<int>(foot); }
  int numBinaries() const { return kBinariesPerNode * numNodes; }

  static ContactLogicState make(const ContactPlannerInput& input,
                                const ContactPlanningConfig& config,
                                int previousPlanShift,
                                const MiqpAssignment* previousAssignment);

  /**
   * Nodes already spent in the phase active at planning time. Rounded down by default, which is conservative for a
   * minimum-duration rule; `roundUp` rounds up, which is conservative for a maximum-duration rule.
   */
  int initialPhaseNodes(size_t foot, bool roundUp) const;
  /**
   * Time at which a phase that switches at node k begins: the node start, except along the committed prefix, where the
   * executed schedule knows the real event time (input.committedPhaseStartTimes).
   */
  scalar_t switchTime(int k, size_t foot) const;
  /** Switch time of node k in node units from the grid start (k itself for a switch at the node start). */
  scalar_t switchNode(int k, size_t foot) const;
  /** Nodes spent in the phase that switched at node k, at the start of node k + 1 (rounded like initialPhaseNodes). */
  int switchedPhaseNodes(int k, size_t foot, bool roundUp) const;
  /** Whether a foot in contact at node k is held down by the minimum double support after a touch-down at `latestTouchDownNode`. */
  bool heldAfterTouchDown(int k, scalar_t latestTouchDownNode) const;
  /** Contact state of a foot just before `node` along the fixed prefix: 1 in contact, 0 in the air, -1 unknown. */
  int stateBefore(const MiqpAssignment& a, size_t foot, int node) const;
  /** Elapsed contact nodes (rounded up) of a foot along the fixed prefix at `node`; -1 while the prefix is not fixed. */
  int contactAge(const MiqpAssignment& a, size_t foot, int node) const;
};

}  // namespace ocs2::humanoid
