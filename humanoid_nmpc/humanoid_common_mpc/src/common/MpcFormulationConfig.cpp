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

#include "humanoid_common_mpc/common/MpcFormulationConfig.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

namespace {

std::string normalizeString(absl::string_view input) {
  std::string output;
  output.reserve(input.size());
  for (char c : input) {
    if (c != '_' && c != '-' && c != ' ') {
      output.push_back(absl::ascii_tolower(c));
    }
  }
  return output;
}

// The cost registry: the names, normalized, that a task file's costs list may use. The unknown-name message is generated
// from it, so an enumerator added here is offered at once; the enum is walked by testMpcFormulationConfig to catch one
// that has a canonical name but was never registered.
// LINT.IfChange(mpc_cost_registry)
const absl::NoDestructor<absl::flat_hash_map<std::string, MpcCostType>> kCostMap({
    {"stateinputquadraticcost", MpcCostType::kStateInputQuadraticCost},
    {"stateinputcost", MpcCostType::kStateInputQuadraticCost},
    {"statequadraticcost", MpcCostType::kStateQuadraticCost},
    {"statecost", MpcCostType::kStateQuadraticCost},
    {"statetrackingcost", MpcCostType::kStateQuadraticCost},
    {"inputquadraticcost", MpcCostType::kInputQuadraticCost},
    {"inputcost", MpcCostType::kInputQuadraticCost},
    {"inputeffortcost", MpcCostType::kInputQuadraticCost},
    {"terminalcost", MpcCostType::kTerminalCost},
    {"icpcost", MpcCostType::kIcpCost},
    {"taskspacefootcost", MpcCostType::kTaskSpaceFootCost},
    {"taskspacefoottrackingcost", MpcCostType::kTaskSpaceFootCost},
    {"taskspacetorsocost", MpcCostType::kTaskSpaceTorsoCost},
    {"taskspacekinematicscost", MpcCostType::kTaskSpaceTorsoCost},
    {"externaltorquecost", MpcCostType::kExternalTorqueCost},
    {"legtorquecost", MpcCostType::kExternalTorqueCost},
    {"jointtorquecost", MpcCostType::kJointTorqueCost},
    {"dcmterminalcost", MpcCostType::kDcmTerminalCost},
    {"terminaldcmcost", MpcCostType::kDcmTerminalCost},
    {"dcmviabilityterminalcost", MpcCostType::kDcmTerminalCost},
    {"comandacomtrackingcost", MpcCostType::kComAndAcomTrackingCost},
});
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/MpcFormulationConfig.h:mpc_cost_type_enum, //humanoid_nmpc/humanoid_mpc_config/task_file.proto:costs)
// clang-format on

/**
 * The canonical names of every entry of a registry, generated from the registry itself: each enumerator it resolves to,
 * once, in the order of the enum, spelled by `toString`. This is the list an unknown-name message offers, so a term
 * added to a registry is offered with it rather than waiting for someone to update a second, hand-written list.
 */
template <typename Type>
std::vector<std::string> registeredNames(const absl::flat_hash_map<std::string, Type>& registry,
                                         absl::StatusOr<std::string> (*absl_nonnull toString)(Type)) {
  std::vector<Type> types;
  for (const std::pair<const std::string, Type>& entry : registry) {
    if (std::find(types.begin(), types.end(), entry.second) == types.end()) {
      types.push_back(entry.second);
    }
  }
  std::sort(types.begin(), types.end(), [](Type lhs, Type rhs) { return static_cast<int>(lhs) < static_cast<int>(rhs); });
  std::vector<std::string> names;
  names.reserve(types.size());
  for (const Type type : types) {
    const absl::StatusOr<std::string> name = toString(type);
    // A registered enumerator without a canonical name is a bug of this file; the message still says which one it is.
    names.push_back(name.ok() ? *name : absl::StrCat("<enumerator ", static_cast<int>(type), " has no name>"));
  }
  return names;
}

// The soft-constraint registry: the names, normalized, that a task file's soft_constraints list may use. The robots' task
// files and README section 5 list the contact-implicit entries by these names, so a name dropped or renamed here leaves
// the documented switch-on lists naming something the loader refuses (see also soft_constraint_names below).
// LINT.IfChange(soft_constraint_registry)
const absl::NoDestructor<absl::flat_hash_map<std::string, MpcSoftConstraintType>> kSoftConstraintMap({
    {"jointlimits", MpcSoftConstraintType::kJointLimits},
    {"jointlimitssoftconstraint", MpcSoftConstraintType::kJointLimits},
    {"footcollision", MpcSoftConstraintType::kFootCollision},
    {"footcollisionsoftconstraint", MpcSoftConstraintType::kFootCollision},
    {"frictionforcecone", MpcSoftConstraintType::kFrictionForceCone},
    {"frictionforceconesoftconstraint", MpcSoftConstraintType::kFrictionForceCone},
    {"contactmomentxy", MpcSoftConstraintType::kContactMomentXy},
    {"contactmomentxyconstraint", MpcSoftConstraintType::kContactMomentXy},
    {"contactwrenchcone", MpcSoftConstraintType::kContactWrenchCone},
    {"contactwrenchconesoftconstraint", MpcSoftConstraintType::kContactWrenchCone},
    {"zerovelocity", MpcSoftConstraintType::kZeroVelocity},
    {"zerovelocitysoftconstraint", MpcSoftConstraintType::kZeroVelocity},
    {"normalvelocity", MpcSoftConstraintType::kNormalVelocity},
    {"normalvelocitysoftconstraint", MpcSoftConstraintType::kNormalVelocity},
    {"contactcomplementarity", MpcSoftConstraintType::kContactComplementarity},
    {"contactcomplementarityconstraint", MpcSoftConstraintType::kContactComplementarity},
    {"forceweightedslip", MpcSoftConstraintType::kForceWeightedSlip},
    {"forceweightedslipconstraint", MpcSoftConstraintType::kForceWeightedSlip},
    {"groundpenetration", MpcSoftConstraintType::kGroundPenetration},
    {"groundpenetrationconstraint", MpcSoftConstraintType::kGroundPenetration},
});
// clang-format off
// LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto:contact_implicit_soft_constraints, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto:contact_implicit_soft_constraints, //humanoid_nmpc/docs/contact_implicit_mpc/README.md:contact_implicit_switch_on_lists, //humanoid_nmpc/humanoid_mpc_config/task_file.proto:soft_constraints)
// clang-format on

// The hard-constraint registry: the names, normalized, that a task file's hard_constraints list may use.
// LINT.IfChange(hard_constraint_registry)
const absl::NoDestructor<absl::flat_hash_map<std::string, MpcHardConstraintType>> kHardConstraintMap({
    {"zerowrench", MpcHardConstraintType::kZeroWrench},
    {"zerowrenchconstraint", MpcHardConstraintType::kZeroWrench},
    {"zerovelocity", MpcHardConstraintType::kZeroVelocity},
    {"zerovelocityconstraint", MpcHardConstraintType::kZeroVelocity},
    {"normalvelocity", MpcHardConstraintType::kNormalVelocity},
    {"normalvelocityconstraint", MpcHardConstraintType::kNormalVelocity},
    {"kneejointmimic", MpcHardConstraintType::kKneeJointMimic},
    {"kneejointmimicconstraint", MpcHardConstraintType::kKneeJointMimic},
    {"mimicjoints", MpcHardConstraintType::kKneeJointMimic},
});
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto:hard_constraints)

}  // namespace

absl::StatusOr<MpcCostType> stringToMpcCostType(absl::string_view name) {
  const std::string normalized = normalizeString(name);
  const absl::flat_hash_map<std::string, MpcCostType>::const_iterator it = kCostMap->find(normalized);
  if (it != kCostMap->end()) {
    return it->second;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown MPC cost type: '", name, "'. Supported costs are: ", absl::StrJoin(mpcCostNames(), ", "), "."));
}

absl::StatusOr<std::string> mpcCostTypeToString(MpcCostType type) {
  switch (type) {
    case MpcCostType::kStateInputQuadraticCost:
      return "state_input_quadratic_cost";
    case MpcCostType::kStateQuadraticCost:
      return "state_quadratic_cost";
    case MpcCostType::kInputQuadraticCost:
      return "input_quadratic_cost";
    case MpcCostType::kTerminalCost:
      return "terminal_cost";
    case MpcCostType::kIcpCost:
      return "icp_cost";
    case MpcCostType::kTaskSpaceFootCost:
      return "task_space_foot_cost";
    case MpcCostType::kTaskSpaceTorsoCost:
      return "task_space_torso_cost";
    case MpcCostType::kExternalTorqueCost:
      return "external_torque_cost";
    case MpcCostType::kJointTorqueCost:
      return "joint_torque_cost";
    case MpcCostType::kDcmTerminalCost:
      return "dcm_terminal_cost";
    // LINT.IfChange(com_and_acom_tracking_cost_name)
    case MpcCostType::kComAndAcomTrackingCost:
      return "com_and_acom_tracking_cost";
      // LINT.ThenChange(//tools/locomotion_heuristics/derive_parameters.py:com_and_acom_tracking_cost_name)
  }
  // Not a default: label, so that -Wswitch reports an enumerator added without a name (ToTW #147).
  return absl::InvalidArgumentError(absl::StrCat("Unknown MpcCostType: ", static_cast<int>(type)));
}

absl::StatusOr<MpcSoftConstraintType> stringToMpcSoftConstraintType(absl::string_view name) {
  const std::string normalized = normalizeString(name);
  const absl::flat_hash_map<std::string, MpcSoftConstraintType>::const_iterator it = kSoftConstraintMap->find(normalized);
  if (it != kSoftConstraintMap->end()) {
    return it->second;
  }
  return absl::InvalidArgumentError(absl::StrCat("Unknown MPC soft constraint type: '", name, "'. Supported soft constraints are: ",
                                                 absl::StrJoin(mpcSoftConstraintNames(), ", "), "."));
}

// The canonical soft-constraint names, the spelling a task file lists. The robots' task files name the contact-implicit
// entries in comments and README section 5 in its switch-on snippet; a rename here has to reach both, or the documented
// switch-on lists name something this registry refuses.
// LINT.IfChange(soft_constraint_names)
absl::StatusOr<std::string> mpcSoftConstraintTypeToString(MpcSoftConstraintType type) {
  switch (type) {
    case MpcSoftConstraintType::kJointLimits:
      return "joint_limits";
    case MpcSoftConstraintType::kFootCollision:
      return "foot_collision";
    case MpcSoftConstraintType::kFrictionForceCone:
      return "friction_force_cone";
    case MpcSoftConstraintType::kContactMomentXy:
      return "contact_moment_xy";
    case MpcSoftConstraintType::kContactWrenchCone:
      return "contact_wrench_cone";
    case MpcSoftConstraintType::kZeroVelocity:
      return "zero_velocity";
    case MpcSoftConstraintType::kNormalVelocity:
      return "normal_velocity";
    case MpcSoftConstraintType::kContactComplementarity:
      return "contact_complementarity";
    case MpcSoftConstraintType::kForceWeightedSlip:
      return "force_weighted_slip";
    case MpcSoftConstraintType::kGroundPenetration:
      return "ground_penetration";
  }
  // Not a default: label, so that -Wswitch reports an enumerator added without a name (ToTW #147).
  return absl::InvalidArgumentError(absl::StrCat("Unknown MpcSoftConstraintType: ", static_cast<int>(type)));
}
// clang-format off
// LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto:contact_implicit_soft_constraints, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto:contact_implicit_soft_constraints, //humanoid_nmpc/docs/contact_implicit_mpc/README.md:contact_implicit_switch_on_lists, //humanoid_nmpc/humanoid_mpc_config/task_file.proto:soft_constraints)
// clang-format on

absl::StatusOr<MpcHardConstraintType> stringToMpcHardConstraintType(absl::string_view name) {
  const std::string normalized = normalizeString(name);
  const absl::flat_hash_map<std::string, MpcHardConstraintType>::const_iterator it = kHardConstraintMap->find(normalized);
  if (it != kHardConstraintMap->end()) {
    return it->second;
  }
  return absl::InvalidArgumentError(absl::StrCat("Unknown MPC hard constraint type: '", name, "'. Supported hard constraints are: ",
                                                 absl::StrJoin(mpcHardConstraintNames(), ", "), "."));
}

absl::StatusOr<std::string> mpcHardConstraintTypeToString(MpcHardConstraintType type) {
  switch (type) {
    case MpcHardConstraintType::kZeroWrench:
      return "zero_wrench";
    case MpcHardConstraintType::kZeroVelocity:
      return "zero_velocity";
    case MpcHardConstraintType::kNormalVelocity:
      return "normal_velocity";
    case MpcHardConstraintType::kKneeJointMimic:
      return "knee_joint_mimic";
  }
  // Not a default: label, so that -Wswitch reports an enumerator added without a name (ToTW #147).
  return absl::InvalidArgumentError(absl::StrCat("Unknown MpcHardConstraintType: ", static_cast<int>(type)));
}

std::vector<std::string> mpcCostNames() {
  return registeredNames(*kCostMap, &mpcCostTypeToString);
}

std::vector<std::string> mpcSoftConstraintNames() {
  return registeredNames(*kSoftConstraintMap, &mpcSoftConstraintTypeToString);
}

std::vector<std::string> mpcHardConstraintNames() {
  return registeredNames(*kHardConstraintMap, &mpcHardConstraintTypeToString);
}

bool contactConstraintsAreScheduleGated(const MpcFormulationTasks& formulationTasks) {
  return formulationTasks.hasHardConstraint(MpcHardConstraintType::kZeroWrench);
}

bool usesContactImplicitFormulation(const MpcFormulationTasks& formulationTasks) {
  return formulationTasks.hasSoftConstraint(MpcSoftConstraintType::kContactComplementarity) ||
         formulationTasks.hasSoftConstraint(MpcSoftConstraintType::kForceWeightedSlip) ||
         formulationTasks.hasSoftConstraint(MpcSoftConstraintType::kGroundPenetration);
}

absl::Status validateContactImplicitConfig(const ModelSettings::ContactImplicitConfig& config) {
  // A weight may be zero - that switches its term off without removing it from the problem - but not negative, which
  // turns the penalty into a reward. A reference or the smoothing length is a divisor, so it has to be positive. The
  // fields, and which of the two each one is, come from ModelSettings::contactImplicitKeys().
  for (const ModelSettings::ContactImplicitKey& key : ModelSettings::contactImplicitKeys()) {
    const scalar_t value = config.*key.field;
    const bool inRange = std::isfinite(value) && (key.isWeight ? value >= 0.0 : value > 0.0);
    if (!inRange) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[validateContactImplicitConfig] ", ModelSettings::kContactImplicitBlock, ".", key.fieldName, " is ", value, " but must be ",
          key.isWeight ? "finite and non-negative: it is the weight of a quadratic penalty, and a negative weight rewards the "
                         "violation it is meant to price"
                       : "finite and positive: the contact-implicit terms divide by it",
          " (humanoid_nmpc/docs/contact_implicit_mpc/README.md, section 4)."));
    }
  }
  return absl::OkStatus();
}

namespace {

struct ContactScheduleSourceEntry {
  // NOLINTNEXTLINE(totw-view-member): every entry is a string literal of a constexpr registry, alive for the whole program.
  absl::string_view name;
  ContactScheduleSource source;
};

// The single place a contact schedule source is added.
// LINT.IfChange(contact_schedule_source_registry)
constexpr std::array<ContactScheduleSourceEntry, 2> kContactScheduleSourceRegistry = {{
    {.name = kGaitScheduleContactScheduleSource, .source = ContactScheduleSource::kGaitSchedule},
    {.name = kContactPlannerContactScheduleSource, .source = ContactScheduleSource::kContactPlanner},
}};
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/MpcFormulationConfig.h:contact_schedule_source_names)
// clang-format on

}  // namespace

std::vector<std::string> contactScheduleSourceNames() {
  std::vector<std::string> names;
  names.reserve(kContactScheduleSourceRegistry.size());
  for (const ContactScheduleSourceEntry& entry : kContactScheduleSourceRegistry) {
    names.emplace_back(entry.name);
  }
  return names;
}

absl::string_view contactScheduleSourceName(ContactScheduleSource source) {
  for (const ContactScheduleSourceEntry& entry : kContactScheduleSourceRegistry) {
    if (entry.source == source) {
      return entry.name;
    }
  }
  // Unreachable while every enumerator is registered; the registry test walks the enum to keep it so.
  return "unregistered";
}

absl::StatusOr<ContactScheduleSource> contactScheduleSourceFromName(absl::string_view name) {
  for (const ContactScheduleSourceEntry& entry : kContactScheduleSourceRegistry) {
    if (entry.name == name) {
      return entry.source;
    }
  }
  return absl::InvalidArgumentError(absl::StrCat("[ContactScheduleSource] unknown ", kContactScheduleSourceKey, " '", name,
                                                 "'; valid names are: ", absl::StrJoin(contactScheduleSourceNames(), ", "),
                                                 " (humanoid_nmpc/docs/README.md, section 2)."));
}

}  // namespace ocs2::humanoid
