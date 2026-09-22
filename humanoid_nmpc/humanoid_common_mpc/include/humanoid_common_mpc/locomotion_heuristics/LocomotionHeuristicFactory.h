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

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/locomotion_heuristics/BasePoseHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/FootholdHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicFormulation.h"
#include "humanoid_common_mpc/locomotion_heuristics/WrenchHeuristic.h"

namespace ocs2::humanoid {

/**
 * Turns a canonical heuristic name into an instance of the class that implements it.
 *
 * The single place a new heuristic is added: one line here, one name in knownHeuristicNames(), one parameter struct in
 * LocomotionHeuristicConfig, and one block in each robot's task.yaml. The four are tied together by the
 * heuristic_factory / known_heuristic_names IFTTT directives so that adding the name without the line, or the line
 * without the block, is caught by the linter rather than at run time.
 */
class LocomotionHeuristicFactory {
 public:
  static absl::StatusOr<std::unique_ptr<BasePoseHeuristic>> makeBasePoseHeuristic(absl::string_view name);
  static absl::StatusOr<std::unique_ptr<FootholdHeuristic>> makeFootholdHeuristic(absl::string_view name);
  static absl::StatusOr<std::unique_ptr<WrenchHeuristic>> makeWrenchHeuristic(absl::string_view name);
};

}  // namespace ocs2::humanoid
