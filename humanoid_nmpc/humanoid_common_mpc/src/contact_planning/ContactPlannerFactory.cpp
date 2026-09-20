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

#include "humanoid_common_mpc/contact_planning/ContactPlannerFactory.h"

#include <algorithm>
#include <cctype>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "humanoid_common_mpc/contact_planning/LipContactPlanner.h"
#include "humanoid_common_mpc/contact_planning/hlip/HlipContactPlanner.h"

namespace ocs2::humanoid {
namespace {

/** Lower case, without the separators, so that `lip_miqp`, `lipMiqp` and `LIP-MIQP` all name the same planner. */
std::string normalize(absl::string_view name) {
  std::string normalized;
  normalized.reserve(name.size());
  for (const char character : name) {
    if (character == '_' || character == '-' || character == ' ') continue;
    normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
  }
  return normalized;
}

absl::Status unknownPlanner(absl::string_view name) {
  return absl::InvalidArgumentError(
      absl::StrCat("[ContactPlannerFactory] unknown planner.type '", name, "'; supported: ", absl::StrJoin(knownPlannerNames(), ", ")));
}

}  // namespace

const std::vector<std::string>& knownPlannerNames() {
  // LINT.IfChange(known_planner_names)
  static const std::vector<std::string> names{planner::kHlip, planner::kLipMiqp};
  // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/contact_planning.yaml:contact_planning_config)
  return names;
}

std::string canonicalPlannerName(absl::string_view name) {
  const std::string normalized = normalize(name);
  const std::vector<std::string>::const_iterator match =
      std::find_if(knownPlannerNames().begin(), knownPlannerNames().end(),
                   [&normalized](const std::string& known) { return normalize(known) == normalized; });
  return match == knownPlannerNames().end() ? std::string() : *match;
}

absl::StatusOr<std::unique_ptr<ContactPlannerInterface>> makeContactPlanner(const ContactPlanningConfig& config) {
  const std::string name = canonicalPlannerName(config.planner.type);
  if (name == planner::kHlip) return std::make_unique<HlipContactPlanner>(config);
  if (name == planner::kLipMiqp) return std::make_unique<LipContactPlanner>(config);
  return unknownPlanner(config.planner.type);
}

absl::StatusOr<std::string> contactPlannerSummary(const ContactPlanningConfig& config) {
  const std::string name = canonicalPlannerName(config.planner.type);
  if (name == planner::kHlip) return HlipContactPlanner::formulationSummary(config);
  if (name == planner::kLipMiqp) return LipContactPlanner::formulationSummary(config);
  return unknownPlanner(config.planner.type);
}

}  // namespace ocs2::humanoid
