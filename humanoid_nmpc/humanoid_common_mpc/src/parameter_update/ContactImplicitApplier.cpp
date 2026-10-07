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

#include "humanoid_common_mpc/parameter_update/ContactImplicitApplier.h"

#include <array>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_core/soft_constraint/StateInputSoftConstraint.h"
#include "ocs2_core/soft_constraint/StateSoftConstraint.h"

#include "humanoid_common_mpc/common/ContactTermNames.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/config/model/ModelSettingsFromConfig.h"
#include "humanoid_common_mpc/constraint/ContactComplementarityConstraint.h"
#include "humanoid_common_mpc/constraint/ForceWeightedSlipConstraint.h"
#include "humanoid_common_mpc/constraint/GroundPenetrationConstraint.h"
#include "humanoid_common_mpc/parameter_update/OcpTermUpdates.h"

namespace ocs2::humanoid {

namespace {

// LINT.IfChange(contact_implicit_fields)
constexpr std::array<absl::string_view, 1> kFields = {
    "contact_implicit",
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto)

/** Whether `ocp` carries any contact-implicit term of a foot of `contactNames`. */
bool carriesContactImplicitTerms(const OptimalControlProblem& ocp, const std::vector<std::string>& contactNames) {
  for (const std::string& footName : contactNames) {
    if (carriesTerm(*ocp.softConstraintPtr, contact_term::name(footName, contact_term::kContactComplementarity)) ||
        carriesTerm(*ocp.softConstraintPtr, contact_term::name(footName, contact_term::kForceWeightedSlip)) ||
        carriesTerm(*ocp.stateSoftConstraintPtr, contact_term::name(footName, contact_term::kGroundPenetration))) {
      return true;
    }
  }
  return false;
}

}  // namespace

absl::Span<const absl::string_view> ContactImplicitApplier::staticFields() {
  return absl::MakeConstSpan(kFields);
}

void ContactImplicitApplier::apply(HotUpdateTarget& target) {
  if (!carriesContactImplicitTerms(target.runningProblem(), target.contactNames())) return;
  // The residual references and the weights, retuned live; checked as at start-up, and refused as a whole, so that the
  // terms never run on half of an edit. The ground is not applied here: see beforeEverySolve().
  const ModelSettings::ContactImplicitConfig reloaded = contactImplicitFromConfig(target.task().contact_implicit);
  if (const absl::Status status = validateContactImplicitConfig(reloaded); !status.ok()) {
    target.reportNotApplied("contact_implicit", status);
    return;
  }
  const vector_t complementarityParams = (vector_t(1) << reloaded.complementarityWeight).finished();
  const vector_t slipParams = (vector_t(1) << reloaded.slipWeight).finished();
  // SquaredHingePenalty::setParameters takes (mu, delta); delta stays 0 so the hinge's zero stays on the ground.
  const vector_t penetrationParams = (vector_t(2) << reloaded.penetrationWeight, 0.0).finished();
  for (OptimalControlProblem& ocp : target.problems()) {
    for (const std::string& footName : target.contactNames()) {
      updateTermIfPresent<StateInputSoftConstraint>(
          *ocp.softConstraintPtr, contact_term::name(footName, contact_term::kContactComplementarity),
          [&](StateInputSoftConstraint& softCon) {
            setPenaltyParameters(softCon, complementarityParams);
            ContactComplementarityConstraint& constraint = softCon.get<ContactComplementarityConstraint>();
            constraint.setHeightReference(reloaded.heightReference);
            constraint.setGapSmoothing(reloaded.gapSmoothing);
          });
      updateTermIfPresent<StateInputSoftConstraint>(
          *ocp.softConstraintPtr, contact_term::name(footName, contact_term::kForceWeightedSlip), [&](StateInputSoftConstraint& softCon) {
            setPenaltyParameters(softCon, slipParams);
            softCon.get<ForceWeightedSlipConstraint>().setTwistReferences(reloaded.velocityReference, reloaded.angularVelocityReference);
          });
      updateTermIfPresent<StateSoftConstraint>(*ocp.stateSoftConstraintPtr, contact_term::name(footName, contact_term::kGroundPenetration),
                                               [&](StateSoftConstraint& softCon) { setPenaltyParameters(softCon, penetrationParams); });
    }
  }
}

void ContactImplicitApplier::beforeEverySolve(SqpSolver& solver, const SwitchedModelReferenceManager* absl_nullable referenceManager) {
  if (referenceManager == nullptr) return;
  const scalar_t appliedTerrainHeight = referenceManager->getAppliedTerrainHeight();
  if (termsTerrainHeight_.has_value() && *termsTerrainHeight_ == appliedTerrainHeight) return;
  setContactImplicitTermsTerrainHeight(solver.getOcpDefinitions(), contactNames_, appliedTerrainHeight);
  termsTerrainHeight_ = appliedTerrainHeight;
}

void setContactImplicitTermsTerrainHeight(std::vector<OptimalControlProblem>& problems,
                                          const std::vector<std::string>& contactNames,
                                          scalar_t terrainHeight) {
  for (OptimalControlProblem& ocp : problems) {
    for (const std::string& footName : contactNames) {
      updateTermIfPresent<StateInputSoftConstraint>(
          *ocp.softConstraintPtr, contact_term::name(footName, contact_term::kContactComplementarity),
          [&](StateInputSoftConstraint& softCon) { softCon.get<ContactComplementarityConstraint>().setTerrainHeight(terrainHeight); });
      updateTermIfPresent<StateSoftConstraint>(
          *ocp.stateSoftConstraintPtr, contact_term::name(footName, contact_term::kGroundPenetration),
          [&](StateSoftConstraint& softCon) { softCon.get<GroundPenetrationConstraint>().setTerrainHeight(terrainHeight); });
    }
  }
}

}  // namespace ocs2::humanoid
