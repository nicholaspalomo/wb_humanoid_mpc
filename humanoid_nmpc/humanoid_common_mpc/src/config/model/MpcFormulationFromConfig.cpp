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

#include "humanoid_common_mpc/config/model/MpcFormulationFromConfig.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"

#include "humanoid_common_mpc/common/ContactInputParameterization.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {
namespace {

constexpr absl::string_view kPrefix = "[mpcFormulationTasksFromConfig] ";
// The prefix of checkMpcFormulationTasks()'s refusals, which the conversion returns as they are.
constexpr absl::string_view kCheckPrefix = "[checkMpcFormulationTasks] ";

/**
 * Resolves every name of the list `field` with `fromName` and inserts the terms into `terms`; an unknown name is
 * InvalidArgument naming its entry, `costs[2]`, with the registry's own message.
 */
template <typename Type, typename Set>
absl::Status insertTerms(absl::string_view field,
                         const std::vector<std::string>& names,
                         absl::StatusOr<Type> (*absl_nonnull fromName)(absl::string_view),
                         Set* absl_nonnull terms) {
  for (size_t i = 0; i < names.size(); ++i) {
    const absl::StatusOr<Type> term = fromName(names[i]);
    if (!term.ok()) {
      return absl::InvalidArgumentError(absl::StrCat(kPrefix, field, "[", i, "]: ", term.status().message()));
    }
    terms->insert(*term);
  }
  return absl::OkStatus();
}

/** One of the three contact-implicit terms, and what goes wrong without it. */
struct ContactImplicitTerm {
  MpcSoftConstraintType type;
  const char* absl_nonnull withoutIt;
};

constexpr ContactImplicitTerm kContactImplicitTerms[] = {
    {.type = MpcSoftConstraintType::kContactComplementarity,
     .withoutIt = "nothing forbids load on a foot above the ground, which is the condition that lets the solver decide contact at all"},
    {.type = MpcSoftConstraintType::kForceWeightedSlip,
     .withoutIt = "nothing holds a loaded foot still - the other two terms are positional, so a foot carrying full body weight may slide "
                  "freely; it is the term that replaces the schedule-gated 'zero_velocity'"},
    {.type = MpcSoftConstraintType::kGroundPenetration,
     .withoutIt = "nothing stops a foot being pushed through the ground, where the complementarity product is satisfied by a negative "
                  "height"},
};

/** Refuses a term set that lists some, but not all, of the three contact-implicit terms, naming each missing one. */
absl::Status checkContactImplicitTermsListedTogether(const MpcFormulationTasks& tasks) {
  std::vector<std::string> listed;
  std::vector<std::string> missing;
  for (const ContactImplicitTerm& term : kContactImplicitTerms) {
    ASSIGN_OR_RETURN(const std::string name, mpcSoftConstraintTypeToString(term.type));
    if (tasks.hasSoftConstraint(term.type)) {
      listed.push_back(absl::StrCat("'", name, "'"));
    } else {
      missing.push_back(absl::StrCat("'", name, "' (without it ", term.withoutIt, ")"));
    }
  }
  if (listed.empty() || missing.empty()) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrCat(kCheckPrefix,
                   "'contact_complementarity', 'force_weighted_slip' and 'ground_penetration' are the three halves of the contact-implicit "
                   "formulation and are listed in soft_constraints together or not at all. soft_constraints lists ",
                   absl::StrJoin(listed, " and "), " but not ", absl::StrJoin(missing, ", nor "),
                   ". Either add the missing terms and switch the formulation on as a whole, or remove ", absl::StrJoin(listed, " and "),
                   " (humanoid_nmpc/docs/contact_implicit_mpc/README.md, section 5)."));
}

/** Logs the terms of `tasks`, list by list. */
absl::Status logTasks(const MpcFormulationTasks& tasks) {
  LOG(INFO) << "[mpcFormulationTasksFromConfig] costs (" << tasks.costs.size() << "):";
  for (const MpcCostType cost : tasks.costs) {
    ASSIGN_OR_RETURN(const std::string name, mpcCostTypeToString(cost));
    LOG(INFO) << "   - " << name;
  }
  LOG(INFO) << "[mpcFormulationTasksFromConfig] soft_constraints (" << tasks.softConstraints.size() << "):";
  for (const MpcSoftConstraintType constraint : tasks.softConstraints) {
    ASSIGN_OR_RETURN(const std::string name, mpcSoftConstraintTypeToString(constraint));
    LOG(INFO) << "   - " << name;
  }
  LOG(INFO) << "[mpcFormulationTasksFromConfig] hard_constraints (" << tasks.hardConstraints.size() << "):";
  for (const MpcHardConstraintType constraint : tasks.hardConstraints) {
    ASSIGN_OR_RETURN(const std::string name, mpcHardConstraintTypeToString(constraint));
    LOG(INFO) << "   - " << name;
  }
  return absl::OkStatus();
}

/** Whether `names` holds `name`. */
bool isOneOf(const std::vector<std::string>& names, absl::string_view name) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

}  // namespace

absl::Status checkMpcFormulationTasks(const MpcFormulationTasks& tasks) {
  // The two terminal costs are alternative ends of the horizon, not a pair: the quadratic final_state_weights cost
  // regulates the whole state towards its reference, the DCM cost keeps the capture point over the support. When
  // `useDcmTerminalCost` was a boolean, a list naming `terminal_cost` beside it had the entry silently ignored; now that
  // the list is the switch, a list naming both is ambiguous, and it is refused rather than resolved by a precedence
  // nobody wrote down.
  if (tasks.hasCost(MpcCostType::kTerminalCost) && tasks.hasCost(MpcCostType::kDcmTerminalCost)) {
    return absl::InvalidArgumentError(absl::StrCat(
        kCheckPrefix,
        "costs lists both 'terminal_cost' and 'dcm_terminal_cost', but the two are alternative ends of the horizon - the "
        "quadratic final_state_weights cost and the DCM (capture point) viability cost - and the problem carries exactly one of "
        "them. Remove 'terminal_cost' to end the horizon on the DCM cost, or 'dcm_terminal_cost' to end it on final_state_weights "
        "(humanoid_nmpc/docs/README.md, section 1)."));
  }
  if (tasks.hasHardConstraint(MpcHardConstraintType::kZeroVelocity) && tasks.hasSoftConstraint(MpcSoftConstraintType::kZeroVelocity)) {
    return absl::InvalidArgumentError(
        absl::StrCat(kCheckPrefix, "'zero_velocity' is listed in both hard_constraints and soft_constraints; list it in one of them."));
  }
  if (tasks.hasHardConstraint(MpcHardConstraintType::kNormalVelocity) && tasks.hasSoftConstraint(MpcSoftConstraintType::kNormalVelocity)) {
    return absl::InvalidArgumentError(
        absl::StrCat(kCheckPrefix, "'normal_velocity' is listed in both hard_constraints and soft_constraints; list it in one of them."));
  }
  // THE THREE CONTACT-IMPLICIT TERMS ARE ONE SWITCH. They are the relaxed complementarity conditions of rigid contact -
  // no load above the ground, no foot below it, no loaded foot sliding - and none of them is a smaller version of the
  // formulation on its own: each one that is missing leaves a hole of its own (see kContactImplicitTerms), and the
  // callers that ask usesContactImplicitFormulation() - the whole-body MPC's refusal, the checks below - treat any one
  // of them as the formulation. So a partial list is refused here, with a message naming what is missing and why,
  // rather than reaching a later check whose explanation does not fit it.
  RETURN_IF_ERROR(checkContactImplicitTermsListedTogether(tasks));
  // The contact-implicit terms and the schedule-gated contact constraints contradict each other: the first let the
  // optimizer decide where a foot carries load, the second decide it from the mode schedule before the solve.
  if (tasks.hasSoftConstraint(MpcSoftConstraintType::kContactComplementarity) &&
      tasks.hasHardConstraint(MpcHardConstraintType::kZeroWrench)) {
    return absl::InvalidArgumentError(absl::StrCat(
        kCheckPrefix,
        "'contact_complementarity' and the hard 'zero_wrench' constraint are mutually exclusive: the first lets the solver decide "
        "when a foot carries load, the second forces the swing foot's wrench to zero from the mode schedule. Remove "
        "'zero_wrench' from hard_constraints to run the contact-implicit formulation "
        "(humanoid_nmpc/docs/contact_implicit_mpc/README.md)."));
  }
  // WITHOUT `zero_wrench` SOMETHING STILL HAS TO BOUND THE CONTACT WRENCH, and only a cone does.
  //
  // `zero_wrench` is what let every contact cone gate itself on the mode schedule: the swinging foot's wrench was
  // already pinned to zero, so there was nothing left for a cone to bound. contactConstraintsAreScheduleGated() keys
  // the gate off that constraint, so dropping it un-gates the cones - but un-gating a cone that is not there enforces
  // nothing. The relaxed complementarity conditions this formulation is built on are
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
  if (!contactConstraintsAreScheduleGated(tasks) && !tasks.hasSoftConstraint(MpcSoftConstraintType::kContactWrenchCone) &&
      !tasks.hasSoftConstraint(MpcSoftConstraintType::kFrictionForceCone)) {
    return absl::InvalidArgumentError(absl::StrCat(
        kCheckPrefix,
        "with the hard 'zero_wrench' constraint removed, 'contact_wrench_cone' or 'friction_force_cone' must be listed in "
        "soft_constraints: 'zero_wrench' is what pinned the swinging foot's wrench to zero, and it is also what let the contact "
        "cones gate themselves on the mode schedule, so without it and without a cone nothing bounds any foot's wrench at all - "
        "adhesion, unlimited friction, a center of pressure anywhere. f_n >= 0 is the first of the three conditions the "
        "contact-implicit formulation rests on, and neither 'contact_complementarity' nor 'ground_penetration' supplies it "
        "(humanoid_nmpc/docs/contact_implicit_mpc/README.md)."));
  }
  if (tasks.hasSoftConstraint(MpcSoftConstraintType::kForceWeightedSlip) &&
      (tasks.hasHardConstraint(MpcHardConstraintType::kZeroVelocity) || tasks.hasSoftConstraint(MpcSoftConstraintType::kZeroVelocity))) {
    return absl::InvalidArgumentError(
        absl::StrCat(kCheckPrefix,
                     "'force_weighted_slip' replaces 'zero_velocity': the two hold the same foot still, one from the measured load and one "
                     "from the mode schedule. Remove 'zero_velocity' from the constraint lists "
                     "(humanoid_nmpc/docs/contact_implicit_mpc/README.md)."));
  }
  // The hard normal-velocity constraint is the last place the schedule still decides contact. It pins the contact
  // frame's vertical velocity to the swing trajectory's reference over the whole scheduled swing, and since the foot's
  // height at the start of that swing is given, a hard equality on the vertical velocity fixes the height profile.
  // The solver can then neither land early, nor land late, nor keep a foot down - the three freedoms the
  // contact-implicit formulation exists to provide. Worse, where the solver does want load early the only variable it
  // has left is the normal force, so the complementarity product is driven to zero by removing the force rather than
  // by closing the gap, which is the formulation backwards.
  if (usesContactImplicitFormulation(tasks) && tasks.hasHardConstraint(MpcHardConstraintType::kNormalVelocity)) {
    return absl::InvalidArgumentError(absl::StrCat(
        kCheckPrefix,
        "the contact-implicit terms and the hard 'normal_velocity' constraint are mutually exclusive: 'normal_velocity' forces "
        "the swing foot's vertical velocity onto the swing trajectory's reference from the mode schedule, which fixes the whole "
        "height profile of the swing and leaves the solver unable to move a touch-down it is being asked to choose. Remove "
        "'normal_velocity' from hard_constraints and list it in soft_constraints instead "
        "(humanoid_nmpc/docs/contact_implicit_mpc/README.md)."));
  }
  // ...and it has to come back as a COST. Refused as a hard equality above, the swing-foot vertical servo is the one
  // term left with the authority to lift a foot: the only other vertical term is the foot cost's pos_z, against the
  // leg-joint entries of state_weights whose reference posture is the foot ON THE FLOOR, a fight of about 100:1 that
  // the leg - eighty times more compliant horizontally than vertically at the standing crouch - resolves by sliding the
  // foot instead of lifting it. Deleting the row outright was tried, and the robot shuffled.
  if (usesContactImplicitFormulation(tasks) && !tasks.hasSoftConstraint(MpcSoftConstraintType::kNormalVelocity)) {
    return absl::InvalidArgumentError(absl::StrCat(
        kCheckPrefix,
        "the contact-implicit formulation needs 'normal_velocity' in soft_constraints: with the hard 'normal_velocity' removed, "
        "as it has to be, the soft one is the only term with the authority to lift a swing foot, and without it the robot "
        "shuffles its feet along the ground instead of stepping. Add 'normal_velocity' to soft_constraints; its weight is "
        "model_settings.foot_constraint.normal_velocity_soft_constraint_weight "
        "(humanoid_nmpc/docs/contact_implicit_mpc/README.md, section 3)."));
  }
  return absl::OkStatus();
}

absl::StatusOr<MpcFormulationTasks> mpcFormulationTasksFromConfig(const mpc_config::TaskFile& taskFile, FormulationLogging logging) {
  MpcFormulationTasks tasks;
  RETURN_IF_ERROR(insertTerms<MpcCostType>("costs", taskFile.costs, &stringToMpcCostType, &tasks.costs));
  RETURN_IF_ERROR(insertTerms<MpcSoftConstraintType>("soft_constraints", taskFile.soft_constraints, &stringToMpcSoftConstraintType,
                                                     &tasks.softConstraints));
  RETURN_IF_ERROR(insertTerms<MpcHardConstraintType>("hard_constraints", taskFile.hard_constraints, &stringToMpcHardConstraintType,
                                                     &tasks.hardConstraints));
  RETURN_IF_ERROR(checkMpcFormulationTasks(tasks));
  if (logging == FormulationLogging::kLogSummary) {
    RETURN_IF_ERROR(logTasks(tasks));
  }
  return tasks;
}

absl::StatusOr<ContactScheduleSource> contactScheduleSourceFromConfig(const mpc_config::TaskFile& taskFile) {
  const std::vector<std::string> names = contactScheduleSourceNames();
  if (!isOneOf(names, taskFile.contact_schedule_source)) {
    return absl::InvalidArgumentError(
        absl::StrCat("[contactScheduleSourceFromConfig] contact_schedule_source is '", taskFile.contact_schedule_source,
                     "', which is not a contact schedule source; valid names are: ", absl::StrJoin(names, ", "),
                     " (humanoid_nmpc/docs/README.md, section 2)."));
  }
  return contactScheduleSourceFromName(taskFile.contact_schedule_source);
}

absl::StatusOr<ContactInputParameterization> contactInputParameterizationFromConfig(const mpc_config::TaskFile& taskFile) {
  const std::vector<std::string> names = contactInputParameterizationNames();
  if (!isOneOf(names, taskFile.contact_input_parameterization)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[contactInputParameterizationFromConfig] contact_input_parameterization is '", taskFile.contact_input_parameterization,
        "', which is not a contact input parameterization; valid names are: ", absl::StrJoin(names, ", "),
        " (humanoid_nmpc/docs/contact_basis_vectors/README.md)."));
  }
  return contactInputParameterizationFromName(taskFile.contact_input_parameterization);
}

std::vector<std::string> centroidalModelNames() {
  return {std::string(kFullCentroidalDynamicsModel), std::string(kSingleRigidBodyDynamicsModel)};
}

absl::string_view centroidalModelName(CentroidalModelType type) {
  switch (type) {
    case CentroidalModelType::FullCentroidalDynamics:
      return kFullCentroidalDynamicsModel;
    case CentroidalModelType::SingleRigidBodyDynamics:
      return kSingleRigidBodyDynamicsModel;
  }
  // Unreachable for an enumerator; -Werror=switch keeps the switch complete when OCS2 adds a model.
  return "unknown";
}

absl::StatusOr<CentroidalModelType> centroidalModelTypeFromName(absl::string_view name) {
  for (const CentroidalModelType type : {CentroidalModelType::FullCentroidalDynamics, CentroidalModelType::SingleRigidBodyDynamics}) {
    if (centroidalModelName(type) == name) return type;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("[centroidalModelTypeFromConfig] centroidal_model is '", name,
                   "', which is not a centroidal model; valid names are: ", absl::StrJoin(centroidalModelNames(), ", "), "."));
}

absl::StatusOr<CentroidalModelType> centroidalModelTypeFromConfig(const mpc_config::TaskFile& taskFile) {
  if (!taskFile.centroidal_model.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat("[centroidalModelTypeFromConfig] the task file sets no centroidal_model, which the ",
                                                   "centroidal MPC needs; valid names are: ", absl::StrJoin(centroidalModelNames(), ", "),
                                                   "."));
  }
  return centroidalModelTypeFromName(*taskFile.centroidal_model);
}

}  // namespace ocs2::humanoid
