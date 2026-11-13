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

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/contact_planning/ContactPlannerInterface.h"
#include "legged_common_mpc/contact_planning/ContactPlanningConfig.h"

namespace ocs2::legged {

/** The contact planner implementations, selected by `planner.type` in
 * `contact_planning.textproto`. */
namespace planner {
// The names planner.type takes, as a file spells them.
// LINT.IfChange(planner_names)
/** The closed-form H-LIP stepper of arXiv:2502.15630 (HlipContactPlanner). */
inline constexpr char kHlip[] = "hlip";
/** The mixed-integer program on a LIP model (LipContactPlanner). */
inline constexpr char kLipMiqp[] = "lip_miqp";
// clang-format off
// LINT.ThenChange(//legged_nmpc/legged_common_mpc/src/contact_planning/ContactPlannerFactory.cpp:known_planner_names, //legged_nmpc/legged_mpc_config/contact_planning_file.proto:planner_type)
// clang-format on
}  // namespace planner

/**
 * The contact set a planner is built for, as much of it as the factory reads:
 * how many contacts, and whether they are the biped's two 6-DoF feet
 * (ModelSettings::isBipedContactSet()). Passive data; the planning core holds
 * no robot model, so its callers fill it from their ModelSettings.
 */
struct ContactSetSummary {
  size_t numContacts = 0;
  bool isBiped = false;
};

/** The biped's two 6-DoF feet, the only contact set the planners run on. */
inline constexpr ContactSetSummary kBipedContactSetSummary = {
    .numContacts = static_cast<size_t>(kBipedFootCount), .isBiped = true};

/** The planner names that can be selected, in the order they are offered. */
const std::vector<std::string>& knownPlannerNames();

/** The stored spelling of a planner name, or an empty string when it is not a
 * known planner (case insensitive). */
std::string canonicalPlannerName(absl::string_view name);

/**
 * Builds the planner `config.planner.type` names for the contact set
 * `contactSet`. An unknown name is rejected with a message that lists the names
 * that exist, so a typo in the task file names the fix, and an invalid
 * configuration with the message of ContactPlanningConfig::validateStatus(),
 * which names the key to change. Both planners, the H-LIP stepper and the
 * mixed-integer LIP program, are written for a left and a right foot: their
 * layout (LipIndices, ContactPlannerInput, ContactLogicState::kNumFeet), their
 * step widths (hlip.nom_pos_lfoot_rt_rfoot_ewrt_heading_y,
 * step_width.nom_pos_lfoot_rt_rfoot_ewrt_heading_y) and their two-foot terms
 * (FootSeparationConstraint, ZmpSupportRegionConstraint, the contact logic
 * rules), so any other contact set is an InvalidArgument naming planner.type
 * and the contact count. Neither throws.
 */
absl::StatusOr<std::unique_ptr<ContactPlannerInterface>> makeContactPlanner(
    const ContactPlanningConfig& config, ContactSetSummary contactSet);

/** Returns the formulation summary of the planner a configuration selects for
 * `contactSet`, rejecting what makeContactPlanner rejects. */
absl::StatusOr<std::string> contactPlannerSummary(
    const ContactPlanningConfig& config, ContactSetSummary contactSet);

}  // namespace ocs2::legged
