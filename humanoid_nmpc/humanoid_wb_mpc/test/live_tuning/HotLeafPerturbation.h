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

#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"

#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid::live_tuning_test {

/** How a leaf is moved. */
enum class LeafStep {
  // A floating-point value times a factor in [1.05, 1.2] of its own, 0 kept; an integer plus 1. Keeps every value valid
  // (a weight positive, g_min below g_max) and every zero a zero.
  kScaled,
  // A floating-point value times 1.1, a 0 made 0.1; an integer plus 1: a change of every leaf, for the liveness checks.
  kMoved,
};

/**
 * Returns the paths of the numeric leaves of `task` that the whole-body updater applies (below a name of
 * wholeBodyHotFieldNames()) and the file sets, in the order of the schema, each element of a list with its index
 * ("state_weights.joint_positions[3].value"), except terrain_height, whose live change is pinned on its own
 * (testBaseHeightFollowsTerrain): a live ground change lifts the target in use once, which a fresh start does not.
 */
std::vector<std::string> appliedNumericLeaves(const mpc_config::TaskFile& task);

/**
 * Returns `task` with the leaves of appliedNumericLeaves() whose path starts with `prefix` moved by `step`; `prefix`
 * empty moves them all. `onlyLeaf` set moves that one leaf alone.
 */
mpc_config::TaskFile withMovedLeaves(const mpc_config::TaskFile& task,
                                     absl::string_view prefix,
                                     LeafStep step,
                                     std::optional<std::string> onlyLeaf = std::nullopt);

}  // namespace ocs2::humanoid::live_tuning_test
