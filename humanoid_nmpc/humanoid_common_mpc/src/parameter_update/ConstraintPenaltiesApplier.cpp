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

#include "humanoid_common_mpc/parameter_update/ConstraintPenaltiesApplier.h"

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h"
#include "ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h"
#include "ocs2_core/soft_constraint/StateInputSoftConstraint.h"
#include "ocs2_core/soft_constraint/StateSoftConstraint.h"

#include "humanoid_common_mpc/HumanoidCostConstraintFactory.h"
#include "humanoid_common_mpc/common/ContactTermNames.h"
#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/config/costs/CollisionConstraintFromConfig.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/config/costs/JointLimitsFromConfig.h"
#include "humanoid_common_mpc/constraint/ContactMomentXYConstraintCppAd.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/constraint/FrictionForceConeConstraint.h"
#include "humanoid_common_mpc/constraint/JointLimitsSoftConstraint.h"
#include "humanoid_common_mpc/parameter_update/OcpTermUpdates.h"

namespace ocs2::humanoid {

namespace {

// LINT.IfChange(constraint_penalties_fields)
constexpr std::array<absl::string_view, 8> kFields = {
    "contacts.contact_wrench_cone_soft_constraint.mu",
    "contacts.contact_wrench_cone_soft_constraint.delta",
    "contacts.friction_force_cone_soft_constraint.mu",
    "contacts.friction_force_cone_soft_constraint.delta",
    "contacts.contact_moment_xy_soft_constraint",
    "joint_limits",
    "collision_constraint.mu",
    "collision_constraint.delta",
};
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto, //humanoid_nmpc/humanoid_mpc_config/contacts_config.proto, //humanoid_nmpc/humanoid_mpc_config/collision_constraint_config.proto)
// clang-format on

/** Whether any foot of `contactNames` carries the soft constraint with `suffix` in `ocp`. */
bool carriesContactTerm(const OptimalControlProblem& ocp, const std::vector<std::string>& contactNames, absl::string_view suffix) {
  for (const std::string& footName : contactNames) {
    if (carriesTerm(*ocp.softConstraintPtr, contact_term::name(footName, suffix))) return true;
  }
  return false;
}

/** The penalties a reload writes; each is empty when its term is not carried or its block was refused (reported). */
struct Penalties {
  std::optional<RelaxedBarrierPenalty::Config> wrenchCone;
  std::optional<RelaxedBarrierPenalty::Config> frictionCone;
  std::optional<RelaxedBarrierPenalty::Config> contactMoment;
  std::optional<PieceWisePolynomialBarrierPenalty::Config> jointLimits;
  std::optional<PieceWisePolynomialBarrierPenalty::Config> collision;
};

/** The penalties of `target`'s file for the terms its running problem carries. */
Penalties penaltiesOf(HotUpdateTarget& target) {
  const mpc_config::TaskFile& task = target.task();
  const OptimalControlProblem& running = target.runningProblem();
  const std::vector<std::string>& contactNames = target.contactNames();
  Penalties penalties;
  if (carriesContactTerm(running, contactNames, contact_term::kContactWrenchCone)) {
    penalties.wrenchCone = convertedOrReported(contactWrenchConeBarrierFromConfig(task.contacts), target.source(),
                                               "the barrier of contacts.contact_wrench_cone_soft_constraint");
  }
  if (carriesContactTerm(running, contactNames, contact_term::kFrictionForceCone)) {
    penalties.frictionCone = convertedOrReported(frictionForceConeBarrierFromConfig(task.contacts), target.source(),
                                                 "the barrier of contacts.friction_force_cone_soft_constraint");
  }
  if (carriesContactTerm(running, contactNames, contact_term::kContactMomentXY)) {
    penalties.contactMoment =
        convertedOrReported(contactMomentXyBarrierFromConfig(task.contacts), target.source(), "contacts.contact_moment_xy_soft_constraint");
  }
  if (carriesTerm(*running.stateSoftConstraintPtr, kJointLimitsTerm)) {
    penalties.jointLimits = convertedOrReported(jointLimitsBarrierFromConfig(task.joint_limits), target.source(), "joint_limits");
  }
  if (carriesTerm(*running.stateSoftConstraintPtr, kFootCollisionTerm)) {
    penalties.collision = convertedOrReported(footCollisionBarrierFromConfig(task.collision_constraint), target.source(),
                                              "the barrier of collision_constraint");
  }
  return penalties;
}

/** Writes the cone barrier `barrier` into the soft constraint `<foot><suffix>` of every foot, as a Cone. */
template <typename Cone>
void setConePenalty(OptimalControlProblem& ocp,
                    const std::vector<std::string>& contactNames,
                    absl::string_view suffix,
                    const RelaxedBarrierPenalty::Config& barrier) {
  for (const std::string& footName : contactNames) {
    updateTermIfPresent<StateInputSoftConstraint>(
        *ocp.softConstraintPtr, contact_term::name(footName, suffix), [&](StateInputSoftConstraint& softCon) {
          setPenaltyParameters(softCon, contactConePenaltyParameters(barrier, softCon.get<Cone>().isScheduleGated()));
        });
  }
}

}  // namespace

absl::Span<const absl::string_view> ConstraintPenaltiesApplier::staticFields() {
  return absl::MakeConstSpan(kFields);
}

void ConstraintPenaltiesApplier::apply(HotUpdateTarget& target) {
  const Penalties penalties = penaltiesOf(target);
  const std::vector<std::string>& contactNames = target.contactNames();
  for (OptimalControlProblem& ocp : target.problems()) {
    if (penalties.wrenchCone.has_value()) {
      setConePenalty<ContactWrenchConeConstraint>(ocp, contactNames, contact_term::kContactWrenchCone, *penalties.wrenchCone);
    }
    if (penalties.frictionCone.has_value()) {
      setConePenalty<FrictionForceConeConstraint>(ocp, contactNames, contact_term::kFrictionForceCone, *penalties.frictionCone);
    }
    if (penalties.contactMoment.has_value()) {
      setConePenalty<ContactMomentXYConstraintCppAd>(ocp, contactNames, contact_term::kContactMomentXY, *penalties.contactMoment);
    }
    if (penalties.jointLimits.has_value()) {
      updateTermIfPresent<JointLimitsSoftConstraint>(*ocp.stateSoftConstraintPtr, kJointLimitsTerm, [&](JointLimitsSoftConstraint& term) {
        term.setGains(penalties.jointLimits->mu, penalties.jointLimits->delta);
      });
    }
    if (penalties.collision.has_value()) {
      const vector_t collisionParams = (vector_t(2) << penalties.collision->mu, penalties.collision->delta).finished();
      updateTermIfPresent<StateSoftConstraint>(*ocp.stateSoftConstraintPtr, kFootCollisionTerm,
                                               [&](StateSoftConstraint& softCon) { setPenaltyParameters(softCon, collisionParams); });
    }
  }
}

}  // namespace ocs2::humanoid
