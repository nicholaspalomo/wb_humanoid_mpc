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

#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_mpc_config/contact_planning_file.nproto.h"

namespace ocs2::humanoid {

// The names of planner.threading: where the contact planner runs (PlannerSettings::runInBackgroundThread).
// LINT.IfChange(planner_threading_names)
/** A worker thread plans the latest snapshot, at most at planner.planning_frequency (runInBackgroundThread true). */
inline constexpr absl::string_view kBackgroundThreadPlannerThreading = "background_thread";
/** The MPC's pre-solve hook plans synchronously, once per solve (runInBackgroundThread false). */
inline constexpr absl::string_view kPreSolveHookPlannerThreading = "pre_solve_hook";
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/contact_planning_file.proto:planner_threading)

/** Every name planner.threading accepts. */
std::vector<std::string> plannerThreadingNames();

// The names of terminal_dcm.target: what the planner's terminal DCM is drawn to (TerminalDcmParameters::trackCommandedVelocity).
// LINT.IfChange(terminal_dcm_target_names)
/** Onto the last ZMP, so that the plan comes to rest (trackCommandedVelocity false). */
inline constexpr absl::string_view kRestTerminalDcmTarget = "rest";
/** The last ZMP plus v_cmd / omega, so that the plan keeps walking (trackCommandedVelocity true). */
inline constexpr absl::string_view kCommandedVelocityTerminalDcmTarget = "commanded_velocity";
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/contact_planning_file.proto:terminal_dcm_target)

/** Every name terminal_dcm.target accepts. */
std::vector<std::string> terminalDcmTargetNames();

/** Whether a conversion of a contact planner's file validates the configuration it makes. */
enum class ContactPlanningValidation {
  /** ContactPlanningConfig::validateStatus() refuses an inconsistent configuration. */
  kValidate,
  /**
   * Not yet: the values that come from the model (shared.com_height when the file leaves it out, ZMP half widths of 0)
   * are filled in afterwards, as the MPC's start-up and its parameter updater do, which validate after
   * ContactPlanningModelParameters::applyTo().
   */
  kDeferUntilModelParametersApplied,
};

/**
 * The contact planner's configuration from its typed file, config/mpc/contact_planning.textproto
 * (humanoid_mpc_config/contact_planning_file.proto).
 *
 * Every block maps onto its parameter struct of ContactPlanningConfig.h, and every value the file omits is that struct's
 * default, which the schema's defaults equal. Two things are not a copy:
 *  - A soft constraint's `slack` block that is present gives the term its own penalty, whose omitted half is the one of
 *    `shared.slack_penalty`; a term without the block keeps no penalty of its own (std::nullopt) and uses the shared one.
 *  - A term list the file omits is empty (an absent list is empty), so a ContactPlanningFile{} converts to no formulation at all and is
 *    refused by validation. A robot WITHOUT a contact-planning file runs ContactPlanningConfig{}, the library defaults,
 *    not the conversion of an empty file.
 *  - planner.threading and terminal_dcm.target are names (plannerThreadingNames(), terminalDcmTargetNames()) of the two
 *    settings runInBackgroundThread and trackCommandedVelocity; their booleans of the old file are retired fields.
 *  - shared.com_height is optional: absent, it is the model's pendulum, which the configuration leaves unset for
 *    ContactPlanningModelParameters::applyTo() to fill in; a height the file gives must be positive.
 * The model-derived parameters (hip_yaw_range bounds, yaw_torque_budget) are no file keys and stay unset, for
 * ContactPlanningModelParameters::applyTo().
 *
 * @param file The parsed file (nproto::LoadTextprotoFile()).
 * @param validation Whether to run ContactPlanningConfig::validateStatus().
 * @return The configuration, or InvalidArgument for an empty name in a term list (naming the list and the index), for
 *         a threading or a target that is none of its names, for a shared.com_height that is given but not positive
 *         and, with kValidate, every rejection of validateStatus().
 */
absl::StatusOr<ContactPlanningConfig> contactPlanningConfigFromConfig(const mpc_config::ContactPlanningFile& file,
                                                                      ContactPlanningValidation validation);

/**
 * The contact planner's configuration of a robot whose file may be absent (loadContactPlanningFileBeside()): the
 * conversion of `file`, or for a robot without one (null) the library defaults ContactPlanningConfig{}, never the
 * conversion of an empty file, whose term lists are empty. Validated as contactPlanningConfigFromConfig() with
 * kValidate.
 */
absl::StatusOr<ContactPlanningConfig> contactPlanningConfigFromOptionalFile(const mpc_config::ContactPlanningFile* absl_nullable file,
                                                                            ContactPlanningValidation validation);

}  // namespace ocs2::humanoid
