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

#include "humanoid_common_mpc/contact_planning/logic/ContactLogicState.h"

#include <algorithm>
#include <cmath>

namespace ocs2::humanoid {

ContactLogicState ContactLogicState::make(const ContactPlannerInput& input,
                                          const ContactPlanningConfig& config,
                                          int previousPlanShift,
                                          const MiqpAssignment* previousAssignment) {
  ContactLogicState s;
  s.input = &input;
  s.numNodes = config.planner.numNodes;
  s.dt = config.planner.dt;
  s.nSwingMin = config.minSwingNodes();
  s.nSwingMax = config.maxSwingNodes();
  s.nContactMin = config.minContactNodes();
  // The walking cap on a stance replaces the standing one while a velocity is commanded.
  s.nContactMax = input.velocityCommand.norm() > config.shared.gaitLimits.walkingSpeedThreshold && config.maxContactNodesWalking() > 0
                      ? config.maxContactNodesWalking()
                      : config.maxContactNodes();
  s.nFlightMin = config.minFlightNodes();
  s.nFlightMax = config.maxFlightNodes();
  if (input.hopRequested && s.nFlightMax > 0) {
    const int hop = static_cast<int>(std::ceil(input.hopFlightDuration / config.planner.dt - 1e-9));
    s.nFlightMin = std::clamp(hop, s.nFlightMin, s.nFlightMax);
  }
  s.nDoubleSupportHold = config.minDoubleSupportNodes();
  // Minimum double support, in (fractional) nodes: after a touch-down at node time t_td the other foot may not lift at a
  // node that starts before t_td + minDoubleSupportDuration.
  s.minDoubleSupportNodes = config.shared.gaitLimits.minDoubleSupportDuration / config.planner.dt;
  s.numCommitted = std::min(static_cast<int>(input.committedContacts.size()), s.numNodes);
  // A double support already in progress at planning time counts from its touch-down, the start of the shorter contact.
  s.initialTouchDownNode = -1000.0;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    if (input.contacts[foot] && input.contacts[1 - foot]) {
      s.initialTouchDownNode = std::max(s.initialTouchDownNode, -std::max(0.0, input.phaseElapsedTime[foot]) / config.planner.dt);
    }
  }
  s.previousPlanShift = previousPlanShift;
  s.previousAssignment = previousAssignment;
  return s;
}

int ContactLogicState::initialPhaseNodes(size_t foot, bool roundUp) const {
  const int cap = 2 * numNodes;
  const scalar_t nodes = std::max(0.0, input->phaseElapsedTime[foot]) / dt;
  // Rounding to the nearest node let both limits be violated by up to half a node: a swing 0.16 s old counted as two
  // nodes and could end one node later at 0.26 s against a 0.3 s minimum.
  const int elapsed = static_cast<int>(roundUp ? std::ceil(nodes - 1e-9) : std::floor(nodes + 1e-9));
  return std::clamp(elapsed, 0, cap);
}

scalar_t ContactLogicState::switchTime(int k, size_t foot) const {
  // A phase that switches at node k begins at the node start, except along the committed prefix, where the executed
  // schedule knows the real event time: a touch-down at 0.97 s inside the node [0.9, 1.0) has lasted 0.03 s when the
  // node ends, not a whole node. Counting it from the node start let every minimum-duration rule after it be satisfied
  // up to a node too early.
  const scalar_t nodeStart = input->time + static_cast<scalar_t>(k) * dt;
  if (k < static_cast<int>(input->committedPhaseStartTimes.size())) {
    const scalar_t start = input->committedPhaseStartTimes[static_cast<size_t>(k)][foot];
    // The switch is detected between two samples; an event outside that window is not the switch we are looking at.
    // The window is closed at the node's end with a tolerance: the last committed node is sampled at the commit
    // boundary, which lies exactly on its end whenever the boundary is on the grid, and the node times are sums of dt
    // that land a few ulp off the event time.
    if (std::isfinite(start) && start > nodeStart - dt && start <= nodeStart + dt + 1e-9) return start;
  }
  return nodeStart;
}

scalar_t ContactLogicState::switchNode(int k, size_t foot) const {
  return (switchTime(k, foot) - input->time) / dt;
}

int ContactLogicState::switchedPhaseNodes(int k, size_t foot, bool roundUp) const {
  // Rounded down against a minimum, up against a maximum, like the elapsed time of the phase active at planning time.
  const scalar_t nodes = static_cast<scalar_t>(k + 1) - switchNode(k, foot);
  const int rounded = static_cast<int>(roundUp ? std::ceil(nodes - 1e-9) : std::floor(nodes + 1e-9));
  return std::max(roundUp ? 1 : 0, rounded);
}

bool ContactLogicState::heldAfterTouchDown(int k, scalar_t latestTouchDownNode) const {
  return static_cast<scalar_t>(k) < latestTouchDownNode + minDoubleSupportNodes - 1e-9;
}

int ContactLogicState::stateBefore(const MiqpAssignment& a, size_t foot, int node) const {
  std::int8_t kappa = input->contacts[foot] ? 1 : 0;
  for (int k = 0; k < node; ++k) {
    const std::int8_t value = a[static_cast<size_t>(contactBinaryIndex(k, foot))];
    if (value == kMiqpFree) return -1;
    kappa = value;
  }
  return static_cast<int>(kappa);
}

int ContactLogicState::contactAge(const MiqpAssignment& a, size_t foot, int node) const {
  std::int8_t kappa = input->contacts[foot] ? 1 : 0;
  int tau = initialPhaseNodes(foot, true);
  for (int k = 0; k < node; ++k) {
    const std::int8_t value = a[static_cast<size_t>(contactBinaryIndex(k, foot))];
    if (value == kMiqpFree) return -1;
    if (value == kappa) {
      ++tau;
    } else {
      tau = switchedPhaseNodes(k, foot, true);
      kappa = value;
    }
  }
  return kappa == 1 ? tau : 0;
}

}  // namespace ocs2::humanoid
