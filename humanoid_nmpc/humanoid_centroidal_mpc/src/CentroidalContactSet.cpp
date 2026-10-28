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

#include "humanoid_centroidal_mpc/CentroidalContactSet.h"

#include <array>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/ContactInputParameterization.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/model/MpcFormulationFromConfig.h"
#include "humanoid_common_mpc/config/swing/LocomotionHeuristicsFromConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"

namespace ocs2::humanoid {
namespace {

// LINT.IfChange(biped_only_terms)
/** The costs written for the biped's two feet: the swing-foot tracking (on the
 * biped swing planner), the leg torques and the DCM of a two-foot support. */
constexpr std::array<MpcCostType, 3> kBipedOnlyCosts = {
    MpcCostType::kTaskSpaceFootCost,
    MpcCostType::kExternalTorqueCost,
    MpcCostType::kDcmTerminalCost,
};
/** The soft constraints written for the feet: the leg collision spheres, the
 * swing servo of the biped swing planner and the footprint corners and 6-D
 * slip of the contact-implicit formulation. */
constexpr std::array<MpcSoftConstraintType, 6> kBipedOnlySoftConstraints = {
    MpcSoftConstraintType::kFootCollision,
    MpcSoftConstraintType::kFootCollisionCbf,
    MpcSoftConstraintType::kNormalVelocity,
    MpcSoftConstraintType::kContactComplementarity,
    MpcSoftConstraintType::kForceWeightedSlip,
    MpcSoftConstraintType::kGroundPenetration,
};
/** The hard constraints written for the feet: the swing servo, and the knee
 * mimic joints of one leg per foot. */
constexpr std::array<MpcHardConstraintType, 2> kBipedOnlyHardConstraints = {
    MpcHardConstraintType::kNormalVelocity,
    MpcHardConstraintType::kKneeJointMimic,
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc/test/testMixedContactSet.cpp:biped_only_terms, //humanoid_nmpc/docs/multi_contact_planning/README.md:biped_only_terms)

/** "name (3-DoF), ..." of the contacts of `modelSettings`, in input order. */
std::string describeContactSet(const ModelSettings& modelSettings) {
  std::vector<std::string> contacts;
  for (size_t i = 0; i < modelSettings.contactNames.size(); ++i) {
    contacts.push_back(absl::StrCat(
        modelSettings.contactNames[i],
        modelSettings.isPointContact(i) ? " (3-DoF)" : " (6-DoF)"));
  }
  return absl::StrJoin(contacts, ", ");
}

/** Appends `<list> '<name>'` for every listed term of `types`. */
template <typename Type, size_t kSize>
absl::Status appendListed(
    const std::array<Type, kSize>& types,
    absl::string_view list,
    absl::FunctionRef<bool(Type)> listed,
    absl::FunctionRef<absl::StatusOr<std::string>(Type)> name,
    std::vector<std::string>& refused) {
  for (const Type type : types) {
    if (!listed(type)) continue;
    ASSIGN_OR_RETURN(const std::string termName, name(type));
    refused.push_back(absl::StrCat(list, " '", termName, "'"));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status checkCentroidalContactSet(const ModelSettings& modelSettings,
                                       const mpc_config::TaskFile& task) {
  if (modelSettings.isBipedContactSet()) return absl::OkStatus();

  ASSIGN_OR_RETURN(
      const MpcFormulationTasks tasks,
      mpcFormulationTasksFromConfig(task, FormulationLogging::kQuiet));
  std::vector<std::string> refused;
  RETURN_IF_ERROR(appendListed<MpcCostType>(
      kBipedOnlyCosts, "costs",
      [&tasks](MpcCostType type) { return tasks.hasCost(type); },
      &mpcCostTypeToString, refused));
  RETURN_IF_ERROR(appendListed<MpcSoftConstraintType>(
      kBipedOnlySoftConstraints, "soft_constraints",
      [&tasks](MpcSoftConstraintType type) {
        return tasks.hasSoftConstraint(type);
      },
      &mpcSoftConstraintTypeToString, refused));
  RETURN_IF_ERROR(appendListed<MpcHardConstraintType>(
      kBipedOnlyHardConstraints, "hard_constraints",
      [&tasks](MpcHardConstraintType type) {
        return tasks.hasHardConstraint(type);
      },
      &mpcHardConstraintTypeToString, refused));

  // The online contact planner: its H-LIP and MIQP planners, its eight
  // contact-planning rules and its foot separation, ZMP support region,
  // no-flight and yaw-torque-budget constraints are all written for two feet.
  // The gait schedule and a tracked multi-contact plan run on any set.
  ASSIGN_OR_RETURN(const ContactScheduleSource contactScheduleSource,
                   contactScheduleSourceFromConfig(task));
  if (contactScheduleSource == ContactScheduleSource::kContactPlanner) {
    refused.push_back(
        absl::StrCat("contact_schedule_source '",
                     contactScheduleSourceName(contactScheduleSource), "'"));
  }
  // The basis-vector parameterization: one wrench cone basis per foot.
  ASSIGN_OR_RETURN(const ContactInputParameterization contactInputs,
                   contactInputParameterizationFromConfig(task));
  if (contactInputs != ContactInputParameterization::kWrench) {
    refused.push_back(
        absl::StrCat("contact_input_parameterization '",
                     contactInputParameterizationName(contactInputs), "'"));
  }
  // The locomotion heuristics: footholds, a base pose over the feet and
  // per-foot wrench offsets.
  ASSIGN_OR_RETURN(
      const LocomotionHeuristicConfig heuristics,
      locomotionHeuristicConfigFromConfig(task.locomotion_heuristics));
  if (!heuristics.formulation.empty()) {
    refused.push_back("locomotion_heuristics (any listed heuristic)");
  }

  if (refused.empty()) return absl::OkStatus();
  return absl::InvalidArgumentError(absl::StrCat(
      "[CentroidalMpcInterface] model_settings lists the contact set ",
      describeContactSet(modelSettings),
      ", which is not the biped's two 6-DoF feet, and the task file selects ",
      absl::StrJoin(refused, ", "),
      ", which are written for exactly two 6-DoF feet. Remove them, or list "
      "the two feet alone (model_settings.contact_names_6dof, no "
      "contacts_3dof) (humanoid_nmpc/docs/multi_contact_planning/README.md, "
      "section 7.2)."));
}

}  // namespace ocs2::humanoid
