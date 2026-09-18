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

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerInterface.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"

namespace ocs2::humanoid {

/** The contact planner implementations, selected by `planner.type` in `contact_planning.yaml`. */
namespace planner {
/** The closed-form H-LIP stepper of arXiv:2502.15630 (HlipContactPlanner). */
inline constexpr const char* kHlip = "hlip";
/** The mixed-integer program on a LIP model (LipContactPlanner). */
inline constexpr const char* kLipMiqp = "lip_miqp";
}  // namespace planner

/** The planner names that can be selected, in the order they are offered. */
const std::vector<std::string>& knownPlannerNames();

/** The stored spelling of a planner name, or an empty string when it is not a known planner (case insensitive). */
std::string canonicalPlannerName(absl::string_view name);

/**
 * Builds the planner `config.planner.type` names. An unknown name is rejected with a message that lists the names
 * that exist, so a typo in the task file names the fix.
 */
absl::StatusOr<std::unique_ptr<ContactPlannerInterface>> makeContactPlanner(const ContactPlanningConfig& config);

/** The formulation summary of the planner a configuration selects, without building one. */
absl::StatusOr<std::string> contactPlannerSummary(const ContactPlanningConfig& config);

}  // namespace ocs2::humanoid
