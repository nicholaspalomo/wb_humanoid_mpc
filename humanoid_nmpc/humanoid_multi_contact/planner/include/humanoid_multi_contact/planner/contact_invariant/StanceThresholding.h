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

#include <vector>

#include "absl/status/statusor.h"

namespace ocs2::humanoid::multi_contact {

/** The defaults of ThresholdingOptions (design section 4.2). */
inline constexpr double kDefaultActivationThreshold = 0.5;
inline constexpr double kDefaultMinimumDwell = 0.3;
inline constexpr double kDefaultMinimumChangeSeparation = 0.1;

/** How the contact variables become a stance sequence. Passive data. */
struct ThresholdingOptions {
  // A candidate is in contact at a node where c >= threshold.
  double threshold = kDefaultActivationThreshold;
  // [s] The shortest contact phase or gap between two phases of one
  // candidate inside the motion; shorter ones are merged into their
  // neighbors.
  double minimumDwell = kDefaultMinimumDwell;
  // [s] The least time between two contact changes, of any candidates:
  // consecutive stances differ by one contact (design section 2, condition
  // 3).
  double minimumChangeSeparation = kDefaultMinimumChangeSeparation;
  // The fewest candidates in contact at every node: with 1, no flight phase
  // (a contact that would leave the robot in the air is kept until another
  // has touched down).
  int minimumContacts = 0;
  // [s] When positive, every change is delayed to the first node at a
  // multiple of it: the plan's stances then change at times the tracking
  // MPC's period divides (its certifier judges the momentum rate at the
  // MPC's samples, across the change when it falls between two).
  double changeQuantum = 0.0;
};

/**
 * The stance of every node: per node, per candidate, whether it is in
 * contact. Passive data.
 */
struct StanceSchedule {
  std::vector<std::vector<bool>> active;  // [node][candidate]
};

/**
 * Thresholds the contact variables `activations` ([node][candidate], at the
 * increasing node `times`) into a stance sequence (design section 4.2):
 *  1. a candidate is in contact where c >= threshold; at the first node
 *     exactly the `initial` candidates are, at the last exactly the `final`
 *     ones;
 *  2. a contact phase or a gap shorter than the minimum dwell, other than the
 *     first and the last of a candidate, is merged into its neighbors,
 *     shortest first;
 *  3. a node with fewer than the minimum contacts keeps the contact that lifted
 *     off last until another has touched down and the minimum separation has
 *     passed;
 *  4. contact changes closer than the minimum separation to the change
 *     before them (of any candidate) are delayed to it; a phase or gap a
 *     delay empties disappears;
 *  5. with a change quantum, every change is delayed to the next node at a
 *     multiple of it, and 3 and 4 run again.
 * InvalidArgument for inconsistent sizes or times that do not increase.
 */
absl::StatusOr<StanceSchedule> thresholdActivations(
    const std::vector<double>& times,
    const std::vector<std::vector<double>>& activations,
    const std::vector<bool>& initial,
    const std::vector<bool>& final,
    const ThresholdingOptions& options);

}  // namespace ocs2::humanoid::multi_contact
