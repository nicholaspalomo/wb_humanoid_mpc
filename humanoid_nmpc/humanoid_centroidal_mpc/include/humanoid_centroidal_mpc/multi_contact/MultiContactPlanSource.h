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

#include <array>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_centroidal_mpc/multi_contact/PlanPlayback.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact_swing/ContactSwingTrajectory.h"
#include "humanoid_mpc_config/multi_contact_plan_config.nproto.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"

namespace ocs2::humanoid {

/** What the centroidal MPC tracks under contact_schedule_source:
 * "multi_contact_plan": a scenario, with its contact set and terrain, and a
 * plan made for them. Passive data. */
struct MultiContactPlanSource {
  multi_contact::LoadedScenario scenario;
  multi_contact::config::MultiContactPlan plan;
};

/**
 * Reads the scenario at `scenarioFile` (multi_contact::loadScenario(): its
 * contact set, terrain and URDF) and the plan at `planFile`, a binary .binpb
 * or a textproto, refused unless it was made for that scenario's contact set
 * and terrain (multi_contact::loadPlanFile()). A path is read as given, from
 * the working directory, or else from the binary's runfiles
 * (robot::resolveResourcePath()), as a repository-relative path.
 *
 * @return InvalidArgument naming the task file's field when a path is empty;
 * NotFound naming a file that is in neither place; FailedPrecondition, with
 * both hashes, for a plan made for another contact set or terrain; the
 * loaders' errors otherwise.
 */
absl::StatusOr<MultiContactPlanSource> loadMultiContactPlanSource(
    absl::string_view scenarioFile, absl::string_view planFile);

/** The contact-force reference of the input costs under a tracked plan
 * (multi_contact_plan.force_reference). */
enum class ForceReference {
  kWeightShare,      // the weight shared among the active contacts
  kPlannedWrenches,  // the plan's wrenches at the plan time
};

/** The names of ForceReference, in its order. */
// LINT.IfChange(force_reference_names)
inline constexpr std::array<absl::string_view, 2> kForceReferenceNames = {
    "weight_share", "planned_wrenches"};
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/multi_contact_plan_config.proto:force_reference_names)

/** The tracking of a plan (the task file's multi_contact_plan block, its
 * tuning). Passive data. */
struct PlannedMotionSettings {
  // [s] The hold point of the plan's clock before a contact change.
  scalar_t clockHoldLead = 0.0;
  // [m] How far along its patch normal a contact of the stance may be from
  // its patch, and how far horizontally the center of mass from the plan's,
  // for the stance to count as reached.
  scalar_t contactTolerance = 0.0;
  scalar_t comTolerance = 0.0;
  ContactSwingSettings swing;
  // The soft constraint contact_swing: its quadratic penalty's weight and its
  // position gain [1/s].
  scalar_t contactSwingWeight = 0.0;
  scalar_t contactSwingPositionGain = 0.0;
  // The landing alignment of contact_swing's 6-DoF contacts: its quadratic
  // penalty's weight, 0 for none, and the window [s] before a touch-down in
  // which it is active.
  scalar_t landingAlignmentWeight = 0.0;
  scalar_t landingWindow = 0.0;
  // The soft constraint terrain_non_penetration: its relaxed barrier and the
  // margin [m] the collision spheres keep.
  scalar_t nonPenetrationMu = 0.0;
  scalar_t nonPenetrationDelta = 0.0;
  scalar_t nonPenetrationMargin = 0.0;
  ForceReference forceReference = ForceReference::kWeightShare;
  // How the plan plays, and [s] how long the robot stands at either end of it
  // under PlanPlayback::kForwardAndReverse.
  PlanPlayback playback = PlanPlayback::kOnce;
  scalar_t playbackDwell = 0.0;
  // [s] The longest swing on an arc; a longer one follows the plan's pad from
  // knot to knot. 0: every swing on an arc.
  scalar_t arcSwingDurationLimit = 0.0;
};

/**
 * The settings of `config`. InvalidArgument naming the field
 * (multi_contact_plan.<block>.<field>) of a value that is not finite, or not
 * positive where the field must be (the lead, the tolerances, the lift
 * height, the contact_swing weight and landing window, the barrier's mu and
 * delta), or negative (the clearance, the position gain, the landing
 * alignment weight, the margin), for a force reference that is not one of
 * kForceReferenceNames, for a playback that is not one of
 * kPlanPlaybackNames, and under "forward_and_reverse" for a playback_dwell
 * that is not finite and positive.
 */
absl::StatusOr<PlannedMotionSettings> plannedMotionSettingsFromConfig(
    const mpc_config::MultiContactPlanConfig& config);

}  // namespace ocs2::humanoid
