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

#include "legged_centroidal_mpc/CentroidalContactSet.h"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

#include "legged_common_mpc/common/ContactInputParameterization.h"
#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/MpcFormulationConfig.h"
#include "legged_common_mpc/common/StatusMacros.h"
#include "legged_common_mpc/config/model/ModelSettingsFromConfig.h"
#include "legged_common_mpc/config/model/MpcFormulationFromConfig.h"
#include "legged_common_mpc/config/swing/LocomotionHeuristicsFromConfig.h"
#include "legged_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "legged_common_mpc/locomotion_heuristics/LocomotionHeuristicFormulation.h"
#include "legged_common_mpc/reference_manager/LegJointReference.h"
#include "legged_mpc_config/task_file.nproto.h"

namespace ocs2::legged {
namespace {

// LINT.IfChange(biped_only_terms)
/** The costs written for the biped's two feet: the leg torques and the DCM of
 * a two-foot support. */
constexpr std::array<MpcCostType, 2> kBipedOnlyCosts = {
    MpcCostType::kExternalTorqueCost,
    MpcCostType::kDcmTerminalCost,
};
/** The soft constraints written for the feet: the leg collision spheres and
 * the footprint corners and 6-D slip of the contact-implicit formulation. */
constexpr std::array<MpcSoftConstraintType, 5> kBipedOnlySoftConstraints = {
    MpcSoftConstraintType::kFootCollision,
    MpcSoftConstraintType::kFootCollisionCbf,
    MpcSoftConstraintType::kContactComplementarity,
    MpcSoftConstraintType::kForceWeightedSlip,
    MpcSoftConstraintType::kGroundPenetration,
};
/** The hard constraints written for the feet: the knee mimic joints of one leg
 * per foot. */
constexpr std::array<MpcHardConstraintType, 1> kBipedOnlyHardConstraints = {
    MpcHardConstraintType::kKneeJointMimic,
};
/** The terms that track the swing trajectory planner, which plans every
 * contact: the swing-foot cost and the normal-velocity servo. They run on the
 * biped and on a set its legs describe (ModelSettings::isLeggedContactSet()),
 * a 3-DoF contact's foot cost without its orientation weights; a set without
 * legs swings its contacts with contact_swing instead. */
constexpr std::array<MpcCostType, 1> kLeggedOnlyCosts = {
    MpcCostType::kTaskSpaceFootCost,
};
constexpr std::array<MpcSoftConstraintType, 1> kLeggedOnlySoftConstraints = {
    MpcSoftConstraintType::kNormalVelocity,
};
constexpr std::array<MpcHardConstraintType, 1> kLeggedOnlyHardConstraints = {
    MpcHardConstraintType::kNormalVelocity,
};
// Beside these, checkCentroidalContactSet() refuses on any non-biped set the
// online contact planner, the basis vectors on a set with a 3-DoF contact, the
// base-pose and wrench heuristics, every heuristic on a set without legs, and
// the step_width foothold anchor with a positive step width.
// LINT.ThenChange(//legged_nmpc/legged_centroidal_mpc/test/testMixedContactSet.cpp:biped_only_terms, //legged_nmpc/legged_centroidal_mpc/test/testPointContactQuadrupedLikeSet.cpp:biped_only_terms, //legged_nmpc/docs/multi_contact_planning/README.md:biped_only_terms)

// LINT.IfChange(rolling_refused_terms)
/** The soft constraints refused on a set with a rolling contact: the
 * contact-implicit terms (footprint corners and the slip of a point fixed on
 * its link), the leg collision spheres of the biped's feet, the terms of a
 * tracked multi-contact plan (the swing servo, the landing alignment it
 * builds with it, and the collision sphere of a fixed contact geometry) and
 * the center of pressure of a 6-DoF sole. */
constexpr std::array<MpcSoftConstraintType, 8> kRollingRefusedSoftConstraints =
    {
        MpcSoftConstraintType::kContactComplementarity,
        MpcSoftConstraintType::kForceWeightedSlip,
        MpcSoftConstraintType::kGroundPenetration,
        MpcSoftConstraintType::kFootCollision,
        MpcSoftConstraintType::kFootCollisionCbf,
        MpcSoftConstraintType::kContactSwing,
        MpcSoftConstraintType::kTerrainNonPenetration,
        MpcSoftConstraintType::kContactMomentXy,
};
// Beside these, checkRollingContactSet() refuses the basis-vector contact
// inputs, the contact_planner and multi_contact_plan schedule sources and the
// foot_motion leg joint reference.
// LINT.ThenChange(//legged_nmpc/legged_centroidal_mpc/test/testRollingContactSet.cpp:rolling_refused_terms, //legged_nmpc/docs/rolling_contact/README.md:rolling_refused_terms)

/** Returns the label of the kind of contact `contactIndex` of `modelSettings`
 * for describeContactSet(): " (3-DoF)", " (rolling)" or " (6-DoF)". */
absl::string_view contactKindLabel(const ModelSettings& modelSettings,
                                   size_t contactIndex) {
  switch (modelSettings.contactKind(contactIndex)) {
    case ModelSettings::ContactKind::kPoint:
      return " (3-DoF)";
    case ModelSettings::ContactKind::kRolling:
      return " (rolling)";
    case ModelSettings::ContactKind::kSurface:
      return " (6-DoF)";
  }
  return " (unknown)";
}

/** "name (3-DoF), ..." of the contacts of `modelSettings`, in input order. */
std::string describeContactSet(const ModelSettings& modelSettings) {
  std::vector<std::string> contacts;
  for (size_t i = 0; i < modelSettings.contactNames.size(); ++i) {
    contacts.push_back(absl::StrCat(modelSettings.contactNames[i],
                                    contactKindLabel(modelSettings, i)));
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

/** Appends `<list> '<name>'` for every term of the three lists that `tasks`
 * selects. */
template <size_t kNumCosts, size_t kNumSoft, size_t kNumHard>
absl::Status appendListedTerms(
    const MpcFormulationTasks& tasks,
    const std::array<MpcCostType, kNumCosts>& costs,
    const std::array<MpcSoftConstraintType, kNumSoft>& softConstraints,
    const std::array<MpcHardConstraintType, kNumHard>& hardConstraints,
    std::vector<std::string>& refused) {
  RETURN_IF_ERROR(appendListed<MpcCostType>(
      costs, "costs",
      [&tasks](MpcCostType type) { return tasks.hasCost(type); },
      &mpcCostTypeToString, refused));
  RETURN_IF_ERROR(appendListed<MpcSoftConstraintType>(
      softConstraints, "soft_constraints",
      [&tasks](MpcSoftConstraintType type) {
        return tasks.hasSoftConstraint(type);
      },
      &mpcSoftConstraintTypeToString, refused));
  return appendListed<MpcHardConstraintType>(
      hardConstraints, "hard_constraints",
      [&tasks](MpcHardConstraintType type) {
        return tasks.hasHardConstraint(type);
      },
      &mpcHardConstraintTypeToString, refused);
}

/**
 * Appends the online contact planner, which `task` selects as its contact
 * schedule source, and the basis-vector parameterization on a set with a 3-DoF
 * contact.
 */
absl::Status appendRefusedPlannerAndInputs(const ModelSettings& modelSettings,
                                           const mpc_config::TaskFile& task,
                                           std::vector<std::string>& refused) {
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
  // The basis-vector parameterization: one wrench cone basis per 6-DoF
  // contact, for any number of them. Its generator sets span a 6-DoF contact's
  // wrench cone over its footprint; none exists for a 3-DoF contact's force
  // cone, so a set with a point contact is refused.
  ASSIGN_OR_RETURN(const ContactInputParameterization contactInputs,
                   contactInputParameterizationFromConfig(task));
  if (contactInputs != ContactInputParameterization::kWrench &&
      !modelSettings.contactNames3DoF.empty()) {
    refused.push_back(
        absl::StrCat("contact_input_parameterization '",
                     contactInputParameterizationName(contactInputs),
                     "' (its generator sets are for 6-DoF contacts; the set "
                     "has 3-DoF contacts)"));
  }
  return absl::OkStatus();
}

/**
 * Appends the locomotion heuristics of `formulation` that `modelSettings`
 * cannot run. The foothold family places each leg's contacts
 * (FootholdHeuristicContext: the leg's side, its hip, its nominal stance) and
 * runs on a set described by its legs; the base-pose family reads the biped's
 * gait phase and the wrench family a left and a right foot, so they stay the
 * biped's. A set without legs has no leg to place a foothold for.
 */
void appendRefusedHeuristics(const ModelSettings& modelSettings,
                             const LocomotionHeuristicFormulation& formulation,
                             std::vector<std::string>& refused) {
  if (!modelSettings.isLeggedContactSet()) {
    if (!formulation.empty()) {
      refused.push_back(
          "locomotion_heuristics (any listed heuristic; the set has no legs)");
    }
    return;
  }
  for (const std::string& name : formulation.basePoseHeuristics) {
    refused.push_back(absl::StrCat(
        "locomotion_heuristics.base_pose_heuristics '", name, "'"));
  }
  for (const std::string& name : formulation.wrenchHeuristics) {
    refused.push_back(
        absl::StrCat("locomotion_heuristics.wrench_heuristics '", name, "'"));
  }
}

/**
 * Appends the step_width anchor of `nominalFoothold` with a positive step
 * width, which measures the width from the other, stance foot of two. The
 * nominal_stance anchor places any legged set's contacts
 * (ModelSettings::Create() refuses it on a set without legs).
 */
void appendRefusedFootholdAnchor(
    const ModelSettings::NominalFootholdConfig& nominalFoothold,
    std::vector<std::string>& refused) {
  if (nominalFoothold.anchor ==
          ModelSettings::NominalFootholdAnchor::kStepWidth &&
      nominalFoothold.nomPosLfootRtRfootEwrtHeadingY > 0.0) {
    refused.push_back(absl::StrCat(
        "nominal_foothold.anchor '",
        nominalFootholdAnchorName(nominalFoothold.anchor),
        "' with nom_pos_lfoot_rt_rfoot_ewrt_heading_y ",
        nominalFoothold.nomPosLfootRtRfootEwrtHeadingY,
        " m (set it to 0, or the anchor to 'nominal_stance' on a legged set)"));
  }
}

/** Returns the names of the rolling contacts of `modelSettings`,
 * comma-separated, in input order. */
std::string describeRollingContacts(const ModelSettings& modelSettings) {
  std::vector<std::string> contacts;
  for (size_t i = 0; i < modelSettings.contactNames.size(); ++i) {
    if (modelSettings.isRollingContact(i)) {
      contacts.push_back(modelSettings.contactNames[i]);
    }
  }
  return absl::StrJoin(contacts, ", ");
}

/**
 * Appends the selections of `task` beside its soft constraints that a rolling
 * contact cannot run: the basis-vector contact inputs (no generator set spans
 * a point force cone), the schedule sources that place a contact on a patch
 * of its own (the online contact planner and a tracked multi-contact plan,
 * whose patches may tilt; a rolling contact assumes flat world-z ground), and
 * the foot_motion leg joint reference, which moves a point fixed on its link.
 */
absl::Status appendRefusedRollingSelections(const mpc_config::TaskFile& task,
                                            std::vector<std::string>& refused) {
  ASSIGN_OR_RETURN(const ContactInputParameterization contactInputs,
                   contactInputParameterizationFromConfig(task));
  if (contactInputs != ContactInputParameterization::kWrench) {
    refused.push_back(
        absl::StrCat("contact_input_parameterization '",
                     contactInputParameterizationName(contactInputs), "'"));
  }
  ASSIGN_OR_RETURN(const ContactScheduleSource contactScheduleSource,
                   contactScheduleSourceFromConfig(task));
  if (contactScheduleSource != ContactScheduleSource::kGaitSchedule) {
    refused.push_back(
        absl::StrCat("contact_schedule_source '",
                     contactScheduleSourceName(contactScheduleSource), "'"));
  }
  ASSIGN_OR_RETURN(const LegJointReference legJointReference,
                   legJointReferenceFromName(task.leg_joint_reference));
  if (legJointReference == LegJointReference::kFootMotion) {
    refused.push_back(absl::StrCat("leg_joint_reference '",
                                   legJointReferenceName(legJointReference),
                                   "'"));
  }
  return absl::OkStatus();
}

/**
 * Returns OkStatus, or InvalidArgument naming the rolling contacts of
 * `modelSettings` and every selection of `task` a rolling contact cannot run
 * (kRollingRefusedSoftConstraints, appendRefusedRollingSelections()).
 * Precondition: modelSettings.hasRollingContacts().
 */
absl::Status checkRollingContactSet(const ModelSettings& modelSettings,
                                    const mpc_config::TaskFile& task,
                                    const MpcFormulationTasks& tasks) {
  std::vector<std::string> refused;
  RETURN_IF_ERROR(appendListed<MpcSoftConstraintType>(
      kRollingRefusedSoftConstraints, "soft_constraints",
      [&tasks](MpcSoftConstraintType type) {
        return tasks.hasSoftConstraint(type);
      },
      &mpcSoftConstraintTypeToString, refused));
  RETURN_IF_ERROR(appendRefusedRollingSelections(task, refused));
  if (refused.empty()) return absl::OkStatus();
  return absl::InvalidArgumentError(absl::StrCat(
      "[CentroidalMpcInterface] model_settings lists the rolling contacts ",
      describeRollingContacts(modelSettings),
      " (model_settings.contacts_rolling), and the task file selects ",
      absl::StrJoin(refused, ", "),
      ", which hold a contact point fixed on its link or place a contact on a "
      "tilted patch; a rolling contact's point moves over its wheel's rim on "
      "flat ground. Remove them "
      "(legged_nmpc/docs/rolling_contact/README.md, section \"Refused "
      "terms\")."));
}

}  // namespace

absl::Status checkCentroidalContactSet(const ModelSettings& modelSettings,
                                       const mpc_config::TaskFile& task) {
  if (modelSettings.isBipedContactSet()) return absl::OkStatus();

  ASSIGN_OR_RETURN(
      const MpcFormulationTasks tasks,
      mpcFormulationTasksFromConfig(task, FormulationLogging::kQuiet));
  // A rolling contact's terms first, with their own reason; the set's other
  // checks follow as on any non-biped set.
  if (modelSettings.hasRollingContacts()) {
    RETURN_IF_ERROR(checkRollingContactSet(modelSettings, task, tasks));
  }
  std::vector<std::string> refused;
  RETURN_IF_ERROR(appendListedTerms(tasks, kBipedOnlyCosts,
                                    kBipedOnlySoftConstraints,
                                    kBipedOnlyHardConstraints, refused));
  if (!modelSettings.isLeggedContactSet()) {
    RETURN_IF_ERROR(appendListedTerms(tasks, kLeggedOnlyCosts,
                                      kLeggedOnlySoftConstraints,
                                      kLeggedOnlyHardConstraints, refused));
  }
  RETURN_IF_ERROR(appendRefusedPlannerAndInputs(modelSettings, task, refused));
  ASSIGN_OR_RETURN(
      const LocomotionHeuristicConfig heuristics,
      locomotionHeuristicConfigFromConfig(task.locomotion_heuristics));
  appendRefusedHeuristics(modelSettings, heuristics.formulation, refused);
  appendRefusedFootholdAnchor(modelSettings.nominalFootholdConfig, refused);

  if (refused.empty()) return absl::OkStatus();
  return absl::InvalidArgumentError(absl::StrCat(
      "[CentroidalMpcInterface] model_settings lists a contact set of ",
      modelSettings.contactNames.size(), " contacts, ",
      describeContactSet(modelSettings),
      ", which is not the biped's two 6-DoF feet, and the task file selects ",
      absl::StrJoin(refused, ", "),
      ", which are written for exactly two 6-DoF feet",
      modelSettings.isLeggedContactSet()
          ? ""
          : " (task_space_foot_cost, normal_velocity and the foothold "
            "heuristics also run on a set that model_settings.legs "
            "describes)",
      ". Remove them, or list the two feet alone "
      "(model_settings.contact_names_6dof, no contacts_3dof) "
      "(legged_nmpc/docs/multi_contact_planning/README.md, section 7.2)."));
}

}  // namespace ocs2::legged
