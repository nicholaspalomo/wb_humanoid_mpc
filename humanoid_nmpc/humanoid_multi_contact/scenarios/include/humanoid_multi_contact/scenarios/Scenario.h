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
#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_multi_contact/collision/CollisionModel.h"
#include "humanoid_multi_contact/collision/collision_model.nproto.h"
#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/contact_set/contact_set.nproto.h"
#include "humanoid_multi_contact/scenarios/scenario.nproto.h"
#include "humanoid_multi_contact/stability/JointLimits.h"
#include "humanoid_multi_contact/stability/RobotStanceModel.h"
#include "humanoid_multi_contact/stability/StabilityCertifier.h"
#include "humanoid_multi_contact/terrain/TerrainModel.h"

namespace ocs2::humanoid::multi_contact {

/** The planners a scenario may name (design section 5.3). */
// LINT.IfChange(planner_names)
inline constexpr std::array<absl::string_view, 3> kPlannerNames = {
    "contact_invariant", "stance_graph", "stance_graph_then_contact_invariant"};
// LINT.ThenChange(//humanoid_nmpc/humanoid_multi_contact/scenarios/scenario.proto:planner_names, //humanoid_nmpc/docs/multi_contact_planning/README.md:planner_names)

/** When the simulation's tilt catch and gantry catch arm (SimulationStart). */
// LINT.IfChange(arming_names)
inline constexpr std::array<absl::string_view, 2> kArmingNames = {
    "at_start", "when_standing"};
// LINT.ThenChange(//humanoid_nmpc/humanoid_multi_contact/scenarios/scenario.proto:arming_names)

/** Reads the scenario textproto at `path` (its header names
 * humanoid_multi_contact.Scenario). */
absl::StatusOr<config::Scenario> loadScenarioFile(absl::string_view path);

/**
 * Checks `scenario` against its contact set and terrain. InvalidArgument names
 * the field: no name or robot, an unknown planner, arming or joint limits
 * name, a negative or NaN static margin, a torque margin outside (0, 1], a
 * joint limit relaxation outside (0, kMaxJointLimitRelaxation] under
 * "relaxed_position" or not 0 under the others, a
 * duration that is not
 * positive, a standing height that is not positive under "when_standing", an
 * allowed, initial, goal or waypoint contact that is not a candidate of
 * `contactSet` or (initial, goal and waypoint) not allowed, a goal or waypoint
 * primitive that is not in `terrain`, an empty goal or waypoint stance, or a
 * malformed pose or base region.
 */
absl::Status validateScenario(const config::Scenario& scenario,
                              const ContactSet& contactSet,
                              const TerrainModel& terrain);

/** The configuration of a scenario pose; InvalidArgument for a base pose of the
 * wrong size. */
absl::StatusOr<RobotConfiguration> configurationOf(
    const config::Scenario::Pose& pose);

/** The contact mode of a scenario pose's active contacts. */
absl::StatusOr<ContactMask> contactsOf(const config::Scenario::Pose& pose,
                                       const ContactSet& contactSet);

/** The certifier settings of `scenario`: its static margin, joint limits and
 * torque margin; InvalidArgument for an unknown joint limits name. */
absl::StatusOr<CertifierOptions> certifierOptionsOf(
    const config::Scenario& scenario);

/** A validated scenario with its contact set and terrain. Passive data. */
struct LoadedScenario {
  config::Scenario scenario;
  config::ContactSet contactSetConfig;
  ContactSet contactSet;
  TerrainModel terrain;
  // The absolute path of the scenario's URDF.
  std::string urdfPath;
  // The robot's collision model (collision_model_file); std::nullopt when the
  // scenario names none.
  std::optional<config::CollisionModel> collisionModelConfig;
};

/**
 * Reads the scenario at `path`, its contact set, URDF and collision model
 * (contact_set_file, urdf_file and collision_model_file, found in the binary's
 * runfiles by robot::resolveResourcePath()), builds its terrain, and validates
 * it (validateScenario()).
 */
absl::StatusOr<LoadedScenario> loadScenario(absl::string_view path);

/** A certifier of plans for `loaded`: its robot, terrain, margins and
 * collision model. */
absl::StatusOr<std::unique_ptr<StabilityCertifier>> makeCertifier(
    const LoadedScenario& loaded);

}  // namespace ocs2::humanoid::multi_contact
