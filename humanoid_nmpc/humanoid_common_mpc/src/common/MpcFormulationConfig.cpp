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

#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/optional.hpp>
#include <boost/property_tree/ptree.hpp>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "humanoid_common_mpc/common/StatusMacros.h"

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
const absl::flat_hash_map<std::string, MpcCostType> kCostMap = {
    {"stateinputquadraticcost", MpcCostType::StateInputQuadraticCost},
    {"stateinputcost", MpcCostType::StateInputQuadraticCost},
    {"statequadraticcost", MpcCostType::StateQuadraticCost},
    {"statecost", MpcCostType::StateQuadraticCost},
    {"statetrackingcost", MpcCostType::StateQuadraticCost},
    {"inputquadraticcost", MpcCostType::InputQuadraticCost},
    {"inputcost", MpcCostType::InputQuadraticCost},
    {"inputeffortcost", MpcCostType::InputQuadraticCost},
    {"terminalcost", MpcCostType::TerminalCost},
    {"icpcost", MpcCostType::IcpCost},
    {"taskspacefootcost", MpcCostType::TaskSpaceFootCost},
    {"taskspacefoottrackingcost", MpcCostType::TaskSpaceFootCost},
    {"taskspacetorsocost", MpcCostType::TaskSpaceTorsoCost},
    {"taskspacekinematicscost", MpcCostType::TaskSpaceTorsoCost},
    {"externaltorquecost", MpcCostType::ExternalTorqueCost},
    {"legtorquecost", MpcCostType::ExternalTorqueCost},
    {"jointtorquecost", MpcCostType::JointTorqueCost},
    {"dcmterminalcost", MpcCostType::DcmTerminalCost},
    {"terminaldcmcost", MpcCostType::DcmTerminalCost},
    {"dcmviabilityterminalcost", MpcCostType::DcmTerminalCost},
    {"comandacomtrackingcost", MpcCostType::ComAndAcomTrackingCost},
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/MpcFormulationConfig.h:mpc_cost_type_enum)

/**
 * The canonical names of every entry of a registry, generated from the registry itself: each enumerator it resolves to,
 * once, in the order of the enum, spelled by `toString`. This is the list an unknown-name message offers, so a term
 * added to a registry is offered with it rather than waiting for someone to update a second, hand-written list.
 */
template <typename Type>
std::string registeredNames(const absl::flat_hash_map<std::string, Type>& registry, absl::StatusOr<std::string> (*toString)(Type)) {
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
  return absl::StrJoin(names, ", ");
}

/** A formulation switch that used to be a top-level boolean of the task file, and the list entry that replaced it. */
struct RetiredFormulationKey {
  const char* key;
  const char* list;
  const char* replacement;
  // What else a file that set the key to true has to change, appended to the message after the replacement.
  const char* alsoChange;
  // Where the replacement is documented.
  const char* documentation;
};

// LINT.IfChange(retired_formulation_keys)
constexpr RetiredFormulationKey kRetiredFormulationKeys[] = {
    {"useComAndAcomTracking", "costs", "com_and_acom_tracking_cost", "", "humanoid_learning/acom/README.md, section 4.1"},
    {"useDcmTerminalCost", "costs", "dcm_terminal_cost",
     ", in place of 'terminal_cost': the two are alternative ends of the horizon and listing both is refused",
     "humanoid_nmpc/docs/README.md, section 1"},
};
// clang-format off
// LINT.ThenChange(//tools/locomotion_heuristics/derive_parameters.py:retired_formulation_keys)
// clang-format on

/**
 * Refuses a task file that still carries a retired formulation switch. Ignoring the key instead would let a stale file
 * silently run a different formulation from the one it asks for - in the case of `useComAndAcomTracking: true`, one that
 * regulates the base pose where the file wanted the center of mass and the ACoM, and in the case of
 * `useDcmTerminalCost: true`, one that ends the horizon on Q_final where the file wanted the capture point - so its
 * presence is an error whatever its value, and the message names the entry that replaced it.
 */
absl::Status checkNoRetiredFormulationKeys(const YAML::Node& root) {
  for (const RetiredFormulationKey& retired : kRetiredFormulationKeys) {
    if (root[retired.key]) {
      return absl::InvalidArgumentError(
          absl::StrCat("[loadMpcFormulationTasks] '", retired.key, "' is no longer read: it was replaced by the entry '",
                       retired.replacement, "' of the '", retired.list, "' list. Delete the key and, where it was true, list '",
                       retired.replacement, "' under ", retired.list, " instead", retired.alsoChange, " (", retired.documentation, ")."));
    }
  }
  return absl::OkStatus();
}

// The soft-constraint registry: the names, normalized, that a task file's soft_constraints list may use. The robots' task
// files and README section 5 list the contact-implicit entries by these names, so a name dropped or renamed here leaves
// the documented switch-on lists naming something the loader refuses (see also soft_constraint_names below).
// LINT.IfChange(soft_constraint_registry)
const absl::flat_hash_map<std::string, MpcSoftConstraintType> kSoftConstraintMap = {
    {"jointlimits", MpcSoftConstraintType::JointLimits},
    {"jointlimitssoftconstraint", MpcSoftConstraintType::JointLimits},
    {"footcollision", MpcSoftConstraintType::FootCollision},
    {"footcollisionsoftconstraint", MpcSoftConstraintType::FootCollision},
    {"frictionforcecone", MpcSoftConstraintType::FrictionForceCone},
    {"frictionforceconesoftconstraint", MpcSoftConstraintType::FrictionForceCone},
    {"contactmomentxy", MpcSoftConstraintType::ContactMomentXY},
    {"contactmomentxyconstraint", MpcSoftConstraintType::ContactMomentXY},
    {"contactwrenchcone", MpcSoftConstraintType::ContactWrenchCone},
    {"contactwrenchconesoftconstraint", MpcSoftConstraintType::ContactWrenchCone},
    {"zerovelocity", MpcSoftConstraintType::ZeroVelocity},
    {"zerovelocitysoftconstraint", MpcSoftConstraintType::ZeroVelocity},
    {"normalvelocity", MpcSoftConstraintType::NormalVelocity},
    {"normalvelocitysoftconstraint", MpcSoftConstraintType::NormalVelocity},
    {"contactcomplementarity", MpcSoftConstraintType::ContactComplementarity},
    {"contactcomplementarityconstraint", MpcSoftConstraintType::ContactComplementarity},
    {"forceweightedslip", MpcSoftConstraintType::ForceWeightedSlip},
    {"forceweightedslipconstraint", MpcSoftConstraintType::ForceWeightedSlip},
    {"groundpenetration", MpcSoftConstraintType::GroundPenetration},
    {"groundpenetrationconstraint", MpcSoftConstraintType::GroundPenetration},
};
// clang-format off
// LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:contact_implicit_soft_constraints, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:contact_implicit_soft_constraints, //humanoid_nmpc/docs/contact_implicit_mpc/README.md:contact_implicit_switch_on_lists)
// clang-format on

// The hard-constraint registry.
const absl::flat_hash_map<std::string, MpcHardConstraintType> kHardConstraintMap = {
    {"zerowrench", MpcHardConstraintType::ZeroWrench},         {"zerowrenchconstraint", MpcHardConstraintType::ZeroWrench},
    {"zerovelocity", MpcHardConstraintType::ZeroVelocity},     {"zerovelocityconstraint", MpcHardConstraintType::ZeroVelocity},
    {"normalvelocity", MpcHardConstraintType::NormalVelocity}, {"normalvelocityconstraint", MpcHardConstraintType::NormalVelocity},
    {"kneejointmimic", MpcHardConstraintType::KneeJointMimic}, {"kneejointmimicconstraint", MpcHardConstraintType::KneeJointMimic},
    {"mimicjoints", MpcHardConstraintType::KneeJointMimic},
};

/** One of the three contact-implicit terms, and what goes wrong without it. */
struct ContactImplicitTerm {
  MpcSoftConstraintType type;
  const char* withoutIt;
};

constexpr ContactImplicitTerm kContactImplicitTerms[] = {
    {MpcSoftConstraintType::ContactComplementarity,
     "nothing forbids load on a foot above the ground, which is the condition that lets the solver decide contact at all"},
    {MpcSoftConstraintType::ForceWeightedSlip,
     "nothing holds a loaded foot still - the other two terms are positional, so a foot carrying full body weight may slide "
     "freely; it is the term that replaces the schedule-gated 'zero_velocity'"},
    {MpcSoftConstraintType::GroundPenetration,
     "nothing stops a foot being pushed through the ground, where the complementarity product is satisfied by a negative "
     "height"},
};

/**
 * Refuses a task set that lists some, but not all, of the three contact-implicit terms. The message names every
 * missing term with the hole it leaves, so the operator can tell a forgotten line from a misunderstanding.
 */
absl::Status checkContactImplicitTermsListedTogether(const MpcFormulationTasks& formulationTasks) {
  std::vector<std::string> listed;
  std::vector<std::string> missing;
  for (const ContactImplicitTerm& term : kContactImplicitTerms) {
    ASSIGN_OR_RETURN(const std::string name, mpcSoftConstraintTypeToString(term.type));
    if (formulationTasks.hasSoftConstraint(term.type)) {
      listed.push_back(absl::StrCat("'", name, "'"));
    } else {
      missing.push_back(absl::StrCat("'", name, "' (without it ", term.withoutIt, ")"));
    }
  }
  if (listed.empty() || missing.empty()) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "[loadMpcFormulationTasks] 'contact_complementarity', 'force_weighted_slip' and 'ground_penetration' are the three halves of "
      "the contact-implicit formulation and are listed in soft_constraints together or not at all. soft_constraints lists ",
      absl::StrJoin(listed, " and "), " but not ", absl::StrJoin(missing, ", nor "),
      ". Either add the missing terms and switch the formulation on as a whole, or remove ", absl::StrJoin(listed, " and "),
      " (humanoid_nmpc/docs/contact_implicit_mpc/README.md, section 5)."));
}

}  // namespace

absl::StatusOr<MpcCostType> stringToMpcCostType(absl::string_view name) {
  const std::string normalized = normalizeString(name);
  const absl::flat_hash_map<std::string, MpcCostType>::const_iterator it = kCostMap.find(normalized);
  if (it != kCostMap.end()) {
    return it->second;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown MPC cost type: '", name, "'. Supported costs are: ", registeredNames(kCostMap, &mpcCostTypeToString), "."));
}

absl::StatusOr<std::string> mpcCostTypeToString(MpcCostType type) {
  switch (type) {
    case MpcCostType::StateInputQuadraticCost:
      return "state_input_quadratic_cost";
    case MpcCostType::StateQuadraticCost:
      return "state_quadratic_cost";
    case MpcCostType::InputQuadraticCost:
      return "input_quadratic_cost";
    case MpcCostType::TerminalCost:
      return "terminal_cost";
    case MpcCostType::IcpCost:
      return "icp_cost";
    case MpcCostType::TaskSpaceFootCost:
      return "task_space_foot_cost";
    case MpcCostType::TaskSpaceTorsoCost:
      return "task_space_torso_cost";
    case MpcCostType::ExternalTorqueCost:
      return "external_torque_cost";
    case MpcCostType::JointTorqueCost:
      return "joint_torque_cost";
    case MpcCostType::DcmTerminalCost:
      return "dcm_terminal_cost";
    // LINT.IfChange(com_and_acom_tracking_cost_name)
    case MpcCostType::ComAndAcomTrackingCost:
      return "com_and_acom_tracking_cost";
      // LINT.ThenChange(//tools/locomotion_heuristics/derive_parameters.py:com_and_acom_tracking_cost_name)
    default:
      return absl::InvalidArgumentError(absl::StrCat("Unknown MpcCostType: ", static_cast<int>(type)));
  }
}

absl::StatusOr<MpcSoftConstraintType> stringToMpcSoftConstraintType(absl::string_view name) {
  const std::string normalized = normalizeString(name);
  const absl::flat_hash_map<std::string, MpcSoftConstraintType>::const_iterator it = kSoftConstraintMap.find(normalized);
  if (it != kSoftConstraintMap.end()) {
    return it->second;
  }
  return absl::InvalidArgumentError(absl::StrCat("Unknown MPC soft constraint type: '", name, "'. Supported soft constraints are: ",
                                                 registeredNames(kSoftConstraintMap, &mpcSoftConstraintTypeToString), "."));
}

// The canonical soft-constraint names, the spelling a task file lists. The robots' task files name the contact-implicit
// entries in comments and README section 5 in its switch-on snippet; a rename here has to reach both, or the documented
// switch-on lists name something this registry refuses.
// LINT.IfChange(soft_constraint_names)
absl::StatusOr<std::string> mpcSoftConstraintTypeToString(MpcSoftConstraintType type) {
  switch (type) {
    case MpcSoftConstraintType::JointLimits:
      return "joint_limits";
    case MpcSoftConstraintType::FootCollision:
      return "foot_collision";
    case MpcSoftConstraintType::FrictionForceCone:
      return "friction_force_cone";
    case MpcSoftConstraintType::ContactMomentXY:
      return "contact_moment_xy";
    case MpcSoftConstraintType::ContactWrenchCone:
      return "contact_wrench_cone";
    case MpcSoftConstraintType::ZeroVelocity:
      return "zero_velocity";
    case MpcSoftConstraintType::NormalVelocity:
      return "normal_velocity";
    case MpcSoftConstraintType::ContactComplementarity:
      return "contact_complementarity";
    case MpcSoftConstraintType::ForceWeightedSlip:
      return "force_weighted_slip";
    case MpcSoftConstraintType::GroundPenetration:
      return "ground_penetration";
    default:
      return absl::InvalidArgumentError(absl::StrCat("Unknown MpcSoftConstraintType: ", static_cast<int>(type)));
  }
}
// clang-format off
// LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:contact_implicit_soft_constraints, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:contact_implicit_soft_constraints, //humanoid_nmpc/docs/contact_implicit_mpc/README.md:contact_implicit_switch_on_lists)
// clang-format on

absl::StatusOr<MpcHardConstraintType> stringToMpcHardConstraintType(absl::string_view name) {
  const std::string normalized = normalizeString(name);
  const absl::flat_hash_map<std::string, MpcHardConstraintType>::const_iterator it = kHardConstraintMap.find(normalized);
  if (it != kHardConstraintMap.end()) {
    return it->second;
  }
  return absl::InvalidArgumentError(absl::StrCat("Unknown MPC hard constraint type: '", name, "'. Supported hard constraints are: ",
                                                 registeredNames(kHardConstraintMap, &mpcHardConstraintTypeToString), "."));
}

absl::StatusOr<std::string> mpcHardConstraintTypeToString(MpcHardConstraintType type) {
  switch (type) {
    case MpcHardConstraintType::ZeroWrench:
      return "zero_wrench";
    case MpcHardConstraintType::ZeroVelocity:
      return "zero_velocity";
    case MpcHardConstraintType::NormalVelocity:
      return "normal_velocity";
    case MpcHardConstraintType::KneeJointMimic:
      return "knee_joint_mimic";
    default:
      return absl::InvalidArgumentError(absl::StrCat("Unknown MpcHardConstraintType: ", static_cast<int>(type)));
  }
}

bool contactConstraintsAreScheduleGated(const MpcFormulationTasks& formulationTasks) {
  return formulationTasks.hasHardConstraint(MpcHardConstraintType::ZeroWrench);
}

bool usesContactImplicitFormulation(const MpcFormulationTasks& formulationTasks) {
  return formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ContactComplementarity) ||
         formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ForceWeightedSlip) ||
         formulationTasks.hasSoftConstraint(MpcSoftConstraintType::GroundPenetration);
}

absl::Status validateContactImplicitConfig(const ModelSettings::ContactImplicitConfig& config) {
  // A weight may be zero - that switches its term off without removing it from the problem - but not negative, which
  // turns the penalty into a reward. A reference or the smoothing length is a divisor, so it has to be positive. The
  // keys, and which of the two each one is, come from the list ModelSettings loads the block with.
  for (const ModelSettings::ContactImplicitKey& key : ModelSettings::contactImplicitKeys()) {
    const scalar_t value = config.*key.field;
    const bool inRange = std::isfinite(value) && (key.isWeight ? value >= 0.0 : value > 0.0);
    if (!inRange) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[validateContactImplicitConfig] ", ModelSettings::kContactImplicitBlock, ".", key.name, " is ", value, " but must be ",
          key.isWeight ? "finite and non-negative: it is the weight of a quadratic penalty, and a negative weight rewards the "
                         "violation it is meant to price"
                       : "finite and positive: the contact-implicit terms divide by it",
          " (humanoid_nmpc/docs/contact_implicit_mpc/README.md, section 4)."));
    }
  }
  return absl::OkStatus();
}

absl::Status checkContactImplicitBlockKeys(const boost::property_tree::ptree& taskTree) {
  const boost::optional<const boost::property_tree::ptree&> block =
      taskTree.get_child_optional(std::string(ModelSettings::kContactImplicitBlock));
  if (!block) {
    return absl::OkStatus();
  }
  std::vector<std::string> knownKeys;
  for (const ModelSettings::ContactImplicitKey& key : ModelSettings::contactImplicitKeys()) {
    knownKeys.emplace_back(key.name);
  }
  for (const boost::property_tree::ptree::value_type& entry : *block) {
    const std::string& name = entry.first;
    if (std::find(knownKeys.begin(), knownKeys.end(), name) != knownKeys.end()) {
      continue;
    }
    // The one key the block used to carry: the ground moved out of it, so that the terms and the swing trajectories
    // could not be configured against two different grounds.
    const std::string retiredHint =
        name == "terrainHeight" ? " The ground is no longer configured here: it is the top-level `terrainHeight` of the task file." : "";
    return absl::InvalidArgumentError(
        absl::StrCat("[checkContactImplicitBlockKeys] ", ModelSettings::kContactImplicitBlock, ".", name,
                     " is not a key of the contact_implicit block, so nothing would read it - a renamed or misspelled key, whose "
                     "term would run on its default and whose tuning slider would reach nothing. The block's keys are ",
                     absl::StrJoin(knownKeys, ", "), ".", retiredHint));
  }
  return absl::OkStatus();
}

absl::StatusOr<MpcFormulationTasks> loadMpcFormulationTasks(absl::string_view taskFile, bool verbose) {
  MpcFormulationTasks formulationTasks;
  const std::string taskFilePath(taskFile);
  YAML::Node root;
  try {
    root = YAML::LoadFile(taskFilePath);
  } catch (const std::exception& e) {
    return absl::NotFoundError(absl::StrCat("[loadMpcFormulationTasks] Failed to load YAML file '", taskFilePath, "': ", e.what()));
  }
  // A key lookup on a scalar throws rather than finding nothing, and would escape this Status-returning loader; an empty
  // file is an empty map, and lists nothing.
  if (!root.IsMap() && !root.IsNull()) {
    return absl::InvalidArgumentError(absl::StrCat("[loadMpcFormulationTasks] the task file '", taskFilePath,
                                                   "' is not a map of keys, so it cannot carry the costs, soft_constraints and "
                                                   "hard_constraints lists."));
  }
  RETURN_IF_ERROR(checkNoRetiredFormulationKeys(root));

  // Find the node containing the task lists: check root, mpc_tasks, or tasks
  YAML::Node context = root;
  if (root["mpc_tasks"] && root["mpc_tasks"].IsMap()) {
    context = root["mpc_tasks"];
  } else if (root["tasks"] && root["tasks"].IsMap()) {
    context = root["tasks"];
  }

  // 1. Costs
  YAML::Node costsNode;
  if (context["costs"] && context["costs"].IsSequence()) {
    costsNode = context["costs"];
  } else if (root["costs"] && root["costs"].IsSequence()) {
    costsNode = root["costs"];
  }

  if (costsNode) {
    for (const YAML::Node& item : costsNode) {
      if (item.IsScalar()) {
        ASSIGN_OR_RETURN(MpcCostType cost, stringToMpcCostType(item.as<std::string>()));
        formulationTasks.costs.insert(cost);
      }
    }
  }

  // 2. Soft Constraints
  YAML::Node softNode;
  if (context["soft_constraints"] && context["soft_constraints"].IsSequence()) {
    softNode = context["soft_constraints"];
  } else if (root["soft_constraints"] && root["soft_constraints"].IsSequence()) {
    softNode = root["soft_constraints"];
  }

  if (softNode) {
    for (const YAML::Node& item : softNode) {
      if (item.IsScalar()) {
        ASSIGN_OR_RETURN(MpcSoftConstraintType sc, stringToMpcSoftConstraintType(item.as<std::string>()));
        formulationTasks.softConstraints.insert(sc);
      }
    }
  }

  // 3. Hard Constraints
  YAML::Node hardNode;
  if (context["hard_constraints"] && context["hard_constraints"].IsSequence()) {
    hardNode = context["hard_constraints"];
  } else if (root["hard_constraints"] && root["hard_constraints"].IsSequence()) {
    hardNode = root["hard_constraints"];
  }

  if (hardNode) {
    for (const YAML::Node& item : hardNode) {
      if (item.IsScalar()) {
        ASSIGN_OR_RETURN(MpcHardConstraintType hc, stringToMpcHardConstraintType(item.as<std::string>()));
        formulationTasks.hardConstraints.insert(hc);
      }
    }
  }

  // The two terminal costs are alternative ends of the horizon, not a pair: the quadratic Q_final cost regulates the whole
  // state towards its reference, the DCM cost keeps the capture point over the support. When `useDcmTerminalCost` was a
  // boolean, a list naming `terminal_cost` beside it had the entry silently ignored; now that the list is the switch, a
  // list naming both is ambiguous, and it is refused rather than resolved by a precedence nobody wrote down.
  if (formulationTasks.hasCost(MpcCostType::TerminalCost) && formulationTasks.hasCost(MpcCostType::DcmTerminalCost)) {
    return absl::InvalidArgumentError(
        "[loadMpcFormulationTasks] costs lists both 'terminal_cost' and 'dcm_terminal_cost', but the two are alternative "
        "ends of the horizon - the quadratic Q_final cost and the DCM (capture point) viability cost - and the problem "
        "carries exactly one of them. Remove 'terminal_cost' to end the horizon on the DCM cost, or 'dcm_terminal_cost' to "
        "end it on Q_final (humanoid_nmpc/docs/README.md, section 1).");
  }

  if (formulationTasks.hasHardConstraint(MpcHardConstraintType::ZeroVelocity) &&
      formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ZeroVelocity)) {
    return absl::InvalidArgumentError(
        "[loadMpcFormulationTasks] 'zero_velocity' cannot be configured as both a hard constraint and a soft constraint simultaneously.");
  }

  if (formulationTasks.hasHardConstraint(MpcHardConstraintType::NormalVelocity) &&
      formulationTasks.hasSoftConstraint(MpcSoftConstraintType::NormalVelocity)) {
    return absl::InvalidArgumentError(
        "[loadMpcFormulationTasks] 'normal_velocity' cannot be configured as both a hard constraint and a soft constraint "
        "simultaneously.");
  }

  // THE THREE CONTACT-IMPLICIT TERMS ARE ONE SWITCH. They are the relaxed complementarity conditions of rigid contact -
  // no load above the ground, no foot below it, no loaded foot sliding - and none of them is a smaller version of the
  // formulation on its own: each one that is missing leaves a hole of its own (see kContactImplicitTerms), and the
  // callers that ask usesContactImplicitFormulation() - the whole-body MPC's refusal, the checks below - treat any one
  // of them as the formulation. So a partial list is refused here, with a message naming what is missing and why,
  // rather than reaching a later check whose explanation does not fit it.
  RETURN_IF_ERROR(checkContactImplicitTermsListedTogether(formulationTasks));

  // The contact-implicit terms and the schedule-gated contact constraints contradict each other: the first let the
  // optimizer decide where a foot carries load, the second decide it from the mode schedule before the solve.
  if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ContactComplementarity) &&
      formulationTasks.hasHardConstraint(MpcHardConstraintType::ZeroWrench)) {
    return absl::InvalidArgumentError(
        "[loadMpcFormulationTasks] 'contact_complementarity' and the hard 'zero_wrench' constraint are mutually exclusive: the first "
        "lets the solver decide when a foot carries load, the second forces the swing foot's wrench to zero from the mode schedule. "
        "Remove 'zero_wrench' from hard_constraints to run the contact-implicit formulation "
        "(humanoid_nmpc/docs/contact_implicit_mpc/README.md).");
  }
  // WITHOUT `zero_wrench` SOMETHING STILL HAS TO BOUND THE CONTACT WRENCH, and only a cone does.
  //
  // `zero_wrench` is what let every contact cone gate itself on the mode schedule: the swinging foot's wrench was
  // already pinned to zero, so there was nothing left for a cone to bound. contactConstraintsAreScheduleGated() now
  // keys the gate off that constraint, so dropping it un-gates the cones - but un-gating a cone that is not there
  // enforces nothing. The relaxed complementarity conditions this formulation is built on are
  //
  //     f_n >= 0,   h >= 0,   f_n h = 0,
  //
  // and `contact_complementarity` and `ground_penetration` supply only the second and the third. The first is the
  // cone's, and without it a foot may pull on the ground: at h = 0 the complementarity product is zero for ANY f_n,
  // including a negative one, so nothing in the formulation objects to adhesion on a foot that is on the floor.
  //
  // Either cone will do here. `contact_wrench_cone` carries the friction, center-of-pressure and torsional rows
  // together; `friction_force_cone` un-gated carries Fz >= 0 and |F_xy| <= mu*Fz as two rows, the first of which is
  // exactly the missing condition. Under the basis-vector parameterization neither list entry is what bounds the
  // individual scalings: CentroidalMpcInterface builds the lambda >= 0 barrier for every contact whatever this list says.
  if (!contactConstraintsAreScheduleGated(formulationTasks) &&
      !formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ContactWrenchCone) &&
      !formulationTasks.hasSoftConstraint(MpcSoftConstraintType::FrictionForceCone)) {
    return absl::InvalidArgumentError(
        "[loadMpcFormulationTasks] with the hard 'zero_wrench' constraint removed, 'contact_wrench_cone' or "
        "'friction_force_cone' must be listed in soft_constraints: 'zero_wrench' is what pinned the swinging foot's wrench "
        "to zero, and it is also what let the contact cones gate themselves on the mode schedule, so without it and "
        "without a cone nothing bounds any foot's wrench at all - adhesion, unlimited friction, a center of pressure "
        "anywhere. f_n >= 0 is the first of the three conditions the contact-implicit formulation rests on, and neither "
        "'contact_complementarity' nor 'ground_penetration' supplies it "
        "(humanoid_nmpc/docs/contact_implicit_mpc/README.md).");
  }
  if (formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ForceWeightedSlip) &&
      (formulationTasks.hasHardConstraint(MpcHardConstraintType::ZeroVelocity) ||
       formulationTasks.hasSoftConstraint(MpcSoftConstraintType::ZeroVelocity))) {
    return absl::InvalidArgumentError(
        "[loadMpcFormulationTasks] 'force_weighted_slip' replaces 'zero_velocity': the two hold the same foot still, one from the "
        "measured load and one from the mode schedule. Remove 'zero_velocity' from the constraint lists "
        "(humanoid_nmpc/docs/contact_implicit_mpc/README.md).");
  }
  // The hard normal-velocity constraint is the last place the schedule still decides contact. It pins the contact
  // frame's vertical velocity to the swing trajectory's reference over the whole scheduled swing, and since the foot's
  // height at the start of that swing is given, a hard equality on the vertical velocity fixes the height profile.
  // The solver can then neither land early, nor land late, nor keep a foot down - the three freedoms the
  // contact-implicit formulation exists to provide. Worse, where the solver does want load early the only variable it
  // has left is the normal force, so the complementarity product is driven to zero by removing the force rather than
  // by closing the gap, which is the formulation backwards.
  if (usesContactImplicitFormulation(formulationTasks) && formulationTasks.hasHardConstraint(MpcHardConstraintType::NormalVelocity)) {
    return absl::InvalidArgumentError(
        "[loadMpcFormulationTasks] the contact-implicit terms and the hard 'normal_velocity' constraint are mutually exclusive: "
        "'normal_velocity' forces the swing foot's vertical velocity onto the swing trajectory's reference from the mode schedule, "
        "which fixes the whole height profile of the swing and leaves the solver unable to move a touch-down it is being asked to "
        "choose. Remove 'normal_velocity' from hard_constraints and list it in soft_constraints instead "
        "(humanoid_nmpc/docs/contact_implicit_mpc/README.md).");
  }
  // ...and it has to come back as a COST. Refused as a hard equality above, the swing-foot vertical servo is the one
  // term left with the authority to lift a foot: the only other vertical term is task_space_foot_cost_weights.pos_z,
  // against the leg-joint entries of Q whose reference posture is the foot ON THE FLOOR, a fight of about 100:1 that
  // the leg - eighty times more compliant horizontally than vertically at the standing crouch - resolves by sliding the
  // foot instead of lifting it. Deleting the row outright was tried, and the robot shuffled.
  if (usesContactImplicitFormulation(formulationTasks) && !formulationTasks.hasSoftConstraint(MpcSoftConstraintType::NormalVelocity)) {
    return absl::InvalidArgumentError(
        "[loadMpcFormulationTasks] the contact-implicit formulation needs 'normal_velocity' in soft_constraints: with the hard "
        "'normal_velocity' removed, as it has to be, the soft one is the only term with the authority to lift a swing foot, and "
        "without it the robot shuffles its feet along the ground instead of stepping. Add 'normal_velocity' to soft_constraints; "
        "its weight is model_settings.foot_constraint.normalVelocitySoftConstraintWeight "
        "(humanoid_nmpc/docs/contact_implicit_mpc/README.md, section 3).");
  }

  if (verbose) {
    LOG(INFO) << "MPC Formulation Tasks Loaded from: " << taskFile;
    LOG(INFO) << "Costs (" << formulationTasks.costs.size() << "):";
    for (MpcCostType cost : formulationTasks.costs) {
      ASSIGN_OR_RETURN(const std::string costStr, mpcCostTypeToString(cost));
      LOG(INFO) << "   - " << costStr;
    }
    LOG(INFO) << "Soft Constraints (" << formulationTasks.softConstraints.size() << "):";
    for (MpcSoftConstraintType sc : formulationTasks.softConstraints) {
      ASSIGN_OR_RETURN(const std::string scStr, mpcSoftConstraintTypeToString(sc));
      LOG(INFO) << "   - " << scStr;
    }
    LOG(INFO) << "Hard Constraints (" << formulationTasks.hardConstraints.size() << "):";
    for (MpcHardConstraintType hc : formulationTasks.hardConstraints) {
      ASSIGN_OR_RETURN(const std::string hcStr, mpcHardConstraintTypeToString(hc));
      LOG(INFO) << "   - " << hcStr;
    }
  }

  return formulationTasks;
}

namespace {

struct ContactScheduleSourceEntry {
  absl::string_view name;
  ContactScheduleSource source;
};

// The single place a contact schedule source is added.
// LINT.IfChange(contact_schedule_source_registry)
constexpr std::array<ContactScheduleSourceEntry, 2> kContactScheduleSourceRegistry = {{
    {kGaitScheduleContactScheduleSource, ContactScheduleSource::kGaitSchedule},
    {kContactPlannerContactScheduleSource, ContactScheduleSource::kContactPlanner},
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

absl::StatusOr<ContactScheduleSource> loadContactScheduleSource(absl::string_view taskFile) {
  const std::string taskFilePath(taskFile);
  YAML::Node root;
  try {
    root = YAML::LoadFile(taskFilePath);
  } catch (const std::exception& error) {
    return absl::NotFoundError(absl::StrCat("[ContactScheduleSource] failed to read the task file '", taskFilePath, "': ", error.what()));
  }
  // A key lookup on a scalar or a sequence throws rather than finding nothing; an empty file is an empty map.
  if (!root.IsMap() && !root.IsNull()) {
    return absl::InvalidArgumentError(absl::StrCat("[ContactScheduleSource] the task file '", taskFilePath,
                                                   "' is not a map of keys, so it cannot name a ", kContactScheduleSourceKey, "."));
  }

  // Present at all, whatever its value: `false` meant the gait schedule, which is also the default, but a file that
  // still spells the switch the old way was written against the old loader and has to be looked at.
  if (root[std::string(kRetiredContactPlanningKey)]) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[ContactScheduleSource] '", kRetiredContactPlanningKey, "' in ", taskFilePath,
        " is no longer read: where the contact schedule comes from is selected by name. Delete the key and write '",
        kContactScheduleSourceKey, ": ", kContactPlannerContactScheduleSource,
        "' where it was true (the online contact planner, centroidal MPC only; which planner runs stays planner.type in "
        "contact_planning.yaml), '",
        kContactScheduleSourceKey, ": ", kGaitScheduleContactScheduleSource, "' (the default) where it was false. Valid names: ",
        absl::StrJoin(contactScheduleSourceNames(), ", "), " (humanoid_nmpc/docs/README.md, section 2)."));
  }

  const YAML::Node node = root[std::string(kContactScheduleSourceKey)];
  if (!node) {
    return kDefaultContactScheduleSource;
  }
  if (!node.IsScalar()) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ContactScheduleSource] ", kContactScheduleSourceKey, " in ", taskFilePath,
                     " must be one name; valid names are: ", absl::StrJoin(contactScheduleSourceNames(), ", "), "."));
  }
  return contactScheduleSourceFromName(node.Scalar());
}

}  // namespace ocs2::humanoid
