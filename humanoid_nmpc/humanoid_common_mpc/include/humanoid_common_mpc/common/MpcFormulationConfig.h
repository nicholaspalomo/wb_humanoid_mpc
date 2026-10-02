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

#include <ocs2_core/misc/PropertyTree.h>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/ModelSettings.h"

namespace ocs2::humanoid {

/**
 * @brief Enumeration of available MPC cost terms.
 */
// LINT.IfChange(mpc_cost_type_enum)
enum class MpcCostType {
  StateInputQuadraticCost,
  StateQuadraticCost,
  InputQuadraticCost,
  TerminalCost,
  IcpCost,
  TaskSpaceFootCost,
  TaskSpaceTorsoCost,
  ExternalTorqueCost,
  JointTorqueCost,
  // Ends the horizon on the Divergent Component of Motion (capture point) viability cost instead of the quadratic
  // Q_final cost of `terminal_cost`; the two are alternatives, and the loader refuses a list that names both. Centroidal
  // MPC only. It replaced the `useDcmTerminalCost` boolean, which loadMpcFormulationTasks() now refuses
  // (humanoid_nmpc/docs/README.md, section 1).
  DcmTerminalCost,
  // Regulates the whole-body center of mass and the Angular Center of Mass instead of the base pose: listing it builds
  // ComAndAcomTrackingCost and zeroes the base-pose block of every quadratic state weight (Q, and Q_final, whose
  // terminal node then gets a terminal ComAndAcomTrackingCost of its own). Centroidal MPC only. It replaced the
  // `useComAndAcomTracking` boolean, which loadMpcFormulationTasks() now refuses (humanoid_learning/acom/README.md).
  ComAndAcomTrackingCost,
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/common/MpcFormulationConfig.cpp:mpc_cost_registry)

/**
 * @brief Enumeration of available MPC soft constraint terms (barriers / penalties).
 */
enum class MpcSoftConstraintType {
  JointLimits,
  FootCollision,
  FrictionForceCone,
  ContactMomentXY,
  ContactWrenchCone,
  ZeroVelocity,
  // The swing-foot vertical servo of the hard `normal_velocity` constraint, priced instead of imposed. The hard form
  // fixes the whole height profile of a scheduled swing, so the solver can neither land early nor late - fatal under
  // the contact-implicit formulation, which exists to give it exactly that freedom. As a cost it shapes the swing and
  // is overruled whenever anything else pays more, which is what a reduced-order plan's guidance should be.
  NormalVelocity,
  // The relaxed complementarity conditions of rigid contact, which replace the schedule-gated zero_wrench and
  // zero_velocity constraints and let the MPC decide contact itself
  // (humanoid_nmpc/docs/contact_implicit_mpc/README.md).
  ContactComplementarity,
  ForceWeightedSlip,
  GroundPenetration,
};

/**
 * @brief Enumeration of available MPC hard constraint terms (equality / inequality constraints).
 */
enum class MpcHardConstraintType {
  ZeroWrench,
  ZeroVelocity,
  NormalVelocity,
  KneeJointMimic,
};

/**
 * @brief Container holding the active MPC formulation tasks parsed from YAML.
 */
struct MpcFormulationTasks {
  absl::flat_hash_set<MpcCostType> costs;
  absl::flat_hash_set<MpcSoftConstraintType> softConstraints;
  absl::flat_hash_set<MpcHardConstraintType> hardConstraints;

  bool hasCost(MpcCostType type) const { return costs.contains(type); }

  bool hasSoftConstraint(MpcSoftConstraintType type) const { return softConstraints.contains(type); }

  bool hasHardConstraint(MpcHardConstraintType type) const { return hardConstraints.contains(type); }
};

/**
 * @brief Whether the contact constraints may be gated on the mode schedule's contact flags.
 *
 * The friction cone, the wrench cone, the center-of-pressure limits and the non-negativity of the basis scalings were
 * all written to switch themselves off while the schedule calls a foot a swing foot. That was only ever sound because
 * the hard `zero_wrench` constraint pinned the swinging foot's wrench to zero, so there was nothing left for a cone to
 * bound. The contact-implicit formulation removes `zero_wrench` - loadMpcFormulationTasks() insists on it - and with
 * it the reason those terms could be switched off. Left gated, a foot the schedule calls a swing foot would carry an
 * unbounded wrench: adhesion, unlimited friction, a center of pressure anywhere.
 *
 * So the gate follows `zero_wrench` rather than the contact-implicit terms themselves. That is the narrower and the
 * stronger condition: it is also correct for a task file that drops `zero_wrench` without listing the
 * contact-implicit terms, which the loader permits and which keying off those terms would leave unguarded.
 *
 * See humanoid_nmpc/docs/contact_implicit_mpc/README.md.
 */
bool contactConstraintsAreScheduleGated(const MpcFormulationTasks& formulationTasks);

/**
 * @brief Whether the task set selects the contact-implicit formulation, i.e. lists any of its three terms.
 *
 * On a task set that came out of loadMpcFormulationTasks() this is a single question about the formulation rather than
 * about one term, because the loader refuses any of `contact_complementarity`, `force_weighted_slip` and
 * `ground_penetration` without the other two. On a set assembled by hand it errs towards "yes": any one of the three is
 * enough, so a caller that refuses the formulation (the whole-body MPC does) refuses a partial one as well.
 */
bool usesContactImplicitFormulation(const MpcFormulationTasks& formulationTasks);

/**
 * @brief Checks the values of the `contact_implicit` block of a task file, as ModelSettings loaded it.
 *
 * The contact-implicit terms divide by the three references and by the smoothing length, and wrap the three weights in
 * quadratic penalties, so a value out of range is not a tuning mistake the solver can live with: a zero reference or
 * smoothing is a division by zero, and a negative weight turns the penalty into a reward - a negative
 * `penetrationWeight` pays a foot to go through the floor. The term constructors CHECK their own arguments as a last
 * line of defense, but a CHECK aborts the process from inside CentroidalMpcInterface::Create(), which returns a Status,
 * so this is the check that has to run first: CentroidalMpcInterface runs it before it builds any term of the
 * formulation, and MpcParameterUpdaterModule on every hot reload of the block.
 *
 * Which key is a weight and which a divisor is ModelSettings::contactImplicitKeys()'s to say.
 *
 * @return OkStatus when every weight is finite and non-negative and every reference and the smoothing length are finite
 *         and positive; otherwise InvalidArgument naming the offending `contact_implicit.<key>`.
 */
absl::Status validateContactImplicitConfig(const ModelSettings::ContactImplicitConfig& config);

/**
 * @brief Checks that every key the task file's `contact_implicit` block carries is one of
 * ModelSettings::contactImplicitKeys().
 *
 * Every reader of the block looks its keys up by name and keeps its current value for a key that is absent, so a key
 * renamed in the task file but not in the code - or misspelled - used to be skipped without a word: the term ran on its
 * default from start-up, and the tuning slider the GUI renders for the key reached nothing on a hot reload.
 *
 * @param taskTree The parsed task file.
 * @return OkStatus when the file has no `contact_implicit` block or every key of it is known; otherwise InvalidArgument
 *         naming the unknown `contact_implicit.<key>` and listing the keys the block may carry.
 */
absl::Status checkContactImplicitBlockKeys(const PropertyTree& taskTree);

// String to Enum conversions (supports snake_case and camelCase, case-insensitive). An unknown name is an
// InvalidArgument whose message lists the canonical name of every entry of the registry, generated from the registry
// itself, so that a term added to it is offered without anyone having to remember a second list.
absl::StatusOr<MpcCostType> stringToMpcCostType(absl::string_view name);
absl::StatusOr<std::string> mpcCostTypeToString(MpcCostType type);

absl::StatusOr<MpcSoftConstraintType> stringToMpcSoftConstraintType(absl::string_view name);
absl::StatusOr<std::string> mpcSoftConstraintTypeToString(MpcSoftConstraintType type);

absl::StatusOr<MpcHardConstraintType> stringToMpcHardConstraintType(absl::string_view name);
absl::StatusOr<std::string> mpcHardConstraintTypeToString(MpcHardConstraintType type);

/**
 * @brief Loads the active MPC formulation tasks from the specified YAML configuration file.
 *
 * Reads from "hard_constraints", "soft_constraints", and "costs" keys (or nested under "tasks" / "mpc_tasks").
 *
 * It refuses the combinations that are not a formulation at all, each with a message naming the list entry to change:
 * `zero_velocity` or `normal_velocity` both hard and soft; any of the three contact-implicit terms without the other
 * two; `contact_complementarity` beside the hard `zero_wrench`; a missing `zero_wrench` with neither cone listed;
 * `force_weighted_slip` beside either `zero_velocity`; and the contact-implicit formulation beside the hard
 * `normal_velocity` or without the soft one. See humanoid_nmpc/docs/contact_implicit_mpc/README.md, section 3.
 *
 * It also refuses a file that still carries a formulation switch which has become a list entry, naming the entry that
 * replaced it, so that a stale file cannot silently run a different formulation: `useComAndAcomTracking` is now the
 * cost `com_and_acom_tracking_cost`, and `useDcmTerminalCost` the cost `dcm_terminal_cost`. And it refuses a `costs`
 * list that names both terminal costs, `terminal_cost` and `dcm_terminal_cost`, which are alternative ends of the
 * horizon (humanoid_nmpc/docs/README.md, section 1).
 *
 * @param taskFile Path to the task.yaml configuration file.
 * @param verbose If true, logs the loaded tasks via LOG(INFO).
 * @return absl::StatusOr<MpcFormulationTasks> The parsed task configuration, or error status.
 */
absl::StatusOr<MpcFormulationTasks> loadMpcFormulationTasks(absl::string_view taskFile, bool verbose = false);

/**
 * Where the mode schedule and the swing feet's landing targets come from, selected by name with the top-level task-file
 * key `contactScheduleSource` (kContactScheduleSourceKey).
 *
 *   gait_schedule    (default, and what every robot ships) the clock-driven gait schedule of the reference files,
 *                    with the nominal footholds (`nominal_foothold`, the locomotion heuristics' foothold channel).
 *   contact_planner  the online contact planner: the contact sequence, the switching times and the footholds are
 *                    planned on a reduced model and replace the gait schedule. WHICH planner runs is not chosen here
 *                    but by `planner.type` in the robot's contact_planning.yaml, the one source of truth for it, next
 *                    to the blocks that tune it. Centroidal MPC only: the whole-body MPC refuses it.
 *
 * The choice decides which reference manager the problem is built on, so it is structural: it takes effect at start-up
 * only. See humanoid_nmpc/docs/README.md, section 2.
 */
enum class ContactScheduleSource { kGaitSchedule, kContactPlanner };

// LINT.IfChange(contact_schedule_source_names)
/** The top-level task-file key that names the contact schedule source. */
inline constexpr absl::string_view kContactScheduleSourceKey = "contactScheduleSource";
inline constexpr absl::string_view kGaitScheduleContactScheduleSource = "gait_schedule";
inline constexpr absl::string_view kContactPlannerContactScheduleSource = "contact_planner";
/**
 * The boolean the key replaced (true meant contact_planner). A task file that still carries it is refused at start-up
 * by both MPCs whatever its value, so that a stale file cannot silently run the other contact schedule.
 */
inline constexpr absl::string_view kRetiredContactPlanningKey = "useContactPlanning";
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/common/MpcFormulationConfig.cpp:contact_schedule_source_registry, //robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:contact_schedule_source_config, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:contact_schedule_source_config, //robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml:contact_schedule_source_config, //robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml:contact_schedule_source_config, //robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.yaml:contact_schedule_source_config, //tools/locomotion_heuristics/derive_parameters.py:contact_schedule_source_name)
// clang-format on

/** The contact schedule source of a task file that does not name one. It is what every robot ran before the key existed. */
inline constexpr ContactScheduleSource kDefaultContactScheduleSource = ContactScheduleSource::kGaitSchedule;

/** Every registered contact schedule source name, in registration order. */
std::vector<std::string> contactScheduleSourceNames();

/** The task-file name of a contact schedule source. */
absl::string_view contactScheduleSourceName(ContactScheduleSource source);

/**
 * Resolves a name to its contact schedule source. An unknown name is an InvalidArgumentError that names
 * kContactScheduleSourceKey and lists every valid name.
 */
absl::StatusOr<ContactScheduleSource> contactScheduleSourceFromName(absl::string_view name);

/**
 * The contact schedule source a task file selects: kDefaultContactScheduleSource when the key is absent, the named one
 * otherwise.
 *
 * @return InvalidArgument naming kContactScheduleSourceKey for an unknown name or a value that is not a name, or naming
 *         kRetiredContactPlanningKey and its replacement when the file still carries the retired boolean; NotFound when
 *         the file cannot be read.
 */
absl::StatusOr<ContactScheduleSource> loadContactScheduleSource(absl::string_view taskFile);

}  // namespace ocs2::humanoid
