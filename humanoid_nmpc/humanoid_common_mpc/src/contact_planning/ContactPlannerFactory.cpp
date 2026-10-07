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

#include "humanoid_common_mpc/contact_planning/ContactPlannerFactory.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
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
  static const std::vector<std::string>& kNames = *new std::vector<std::string>{planner::kHlip, planner::kLipMiqp};
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/contact_planning/ContactPlannerFactory.h:planner_names, //humanoid_nmpc/humanoid_mpc_config/contact_planning_file.proto:planner_type)
  // clang-format on
  return kNames;
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
  if (name.empty()) return unknownPlanner(config.planner.type);
  // An invalid configuration is returned as the Status of ContactPlanningConfig::validateStatus(), which names the key.
  // This used to construct both planners directly: LipContactPlanner's constructor validated by throwing, so the
  // rejection escaped this StatusOr function as an exception, and HlipContactPlanner does not validate at all and was
  // built from the invalid configuration silently - the same bad key threw for one planner and passed for the other.
  if (name == planner::kLipMiqp) {
    ASSIGN_OR_RETURN(std::unique_ptr<LipContactPlanner> lip, LipContactPlanner::Create(config));
    return std::unique_ptr<ContactPlannerInterface>(std::move(lip));
  }
  RETURN_IF_ERROR(config.validateStatus());
  return std::make_unique<HlipContactPlanner>(config);
}

absl::StatusOr<std::string> contactPlannerSummary(const ContactPlanningConfig& config) {
  const std::string name = canonicalPlannerName(config.planner.type);
  if (name.empty()) return unknownPlanner(config.planner.type);
  // As in makeContactPlanner: the H-LIP summary is computed from the configuration as it is, so it is validated first
  // (the mixed-integer summary assembles a planner through LipContactPlanner::Create(), which validates it too).
  RETURN_IF_ERROR(config.validateStatus());
  if (name == planner::kHlip) return HlipContactPlanner::formulationSummary(config);
  return LipContactPlanner::formulationSummary(config);
}

}  // namespace ocs2::humanoid
