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

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicFactory.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

#include "humanoid_common_mpc/locomotion_heuristics/base_pose/HeightCompensationHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/base_pose/OrientationCompensationHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/base_pose/PeriodicOrientationHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/foothold/CapturePointHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/foothold/HighSpeedTurningHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/foothold/HipCenteredSteppingHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/foothold/InPlaceTurningHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/foothold/TranslationalSteppingHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/wrench/CentripetalAccelerationHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/wrench/ImpulseScalingHeuristic.h"

namespace ocs2::humanoid {

namespace {

absl::Status unknownHeuristic(HeuristicKind kind, absl::string_view name) {
  return absl::InvalidArgumentError(absl::StrCat("[LocomotionHeuristicFactory] unknown ", heuristicKindName(kind), " heuristic '", name,
                                                 "'; supported: ", absl::StrJoin(knownHeuristicNames(kind), ", "), "."));
}

}  // namespace

// Every heuristic is built here and nowhere else. Adding one is four edits - this switch, knownHeuristicNames(), a
// parameter struct in LocomotionHeuristicConfig, and the block in each robot's task.yaml - and the IFTTT directives
// tie the first, the second and the fourth together so the linter catches three of the four ways of doing it halfway.
// LINT.IfChange(heuristic_factory)
absl::StatusOr<std::unique_ptr<BasePoseHeuristic>> LocomotionHeuristicFactory::makeBasePoseHeuristic(absl::string_view name) {
  const std::string canonical = canonicalHeuristicName(HeuristicKind::BASE_POSE, name);
  if (canonical == heuristic::kOrientationCompensation) return std::make_unique<OrientationCompensationHeuristic>();
  if (canonical == heuristic::kPeriodicOrientation) return std::make_unique<PeriodicOrientationHeuristic>();
  if (canonical == heuristic::kHeightCompensation) return std::make_unique<HeightCompensationHeuristic>();
  return unknownHeuristic(HeuristicKind::BASE_POSE, name);
}

absl::StatusOr<std::unique_ptr<FootholdHeuristic>> LocomotionHeuristicFactory::makeFootholdHeuristic(absl::string_view name) {
  const std::string canonical = canonicalHeuristicName(HeuristicKind::FOOTHOLD, name);
  if (canonical == heuristic::kHipCenteredStepping) return std::make_unique<HipCenteredSteppingHeuristic>();
  if (canonical == heuristic::kCapturePoint) return std::make_unique<CapturePointHeuristic>();
  if (canonical == heuristic::kTranslationalStepping) return std::make_unique<TranslationalSteppingHeuristic>();
  if (canonical == heuristic::kInPlaceTurning) return std::make_unique<InPlaceTurningHeuristic>();
  if (canonical == heuristic::kHighSpeedTurning) return std::make_unique<HighSpeedTurningHeuristic>();
  return unknownHeuristic(HeuristicKind::FOOTHOLD, name);
}

absl::StatusOr<std::unique_ptr<WrenchHeuristic>> LocomotionHeuristicFactory::makeWrenchHeuristic(absl::string_view name) {
  const std::string canonical = canonicalHeuristicName(HeuristicKind::WRENCH, name);
  if (canonical == heuristic::kImpulseScaling) return std::make_unique<ImpulseScalingHeuristic>();
  if (canonical == heuristic::kCentripetalAcceleration) return std::make_unique<CentripetalAccelerationHeuristic>();
  return unknownHeuristic(HeuristicKind::WRENCH, name);
}
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/locomotion_heuristics/LocomotionHeuristicFormulation.cpp:known_heuristic_names)
// clang-format on

}  // namespace ocs2::humanoid
