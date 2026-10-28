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

#include "humanoid_multi_contact/planner/contact_invariant/ContactInvariantProblem.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/misc/Collection.h"
#include "ocs2_core/soft_constraint/StateInputSoftConstraint.h"
#include "ocs2_core/soft_constraint/StateSoftConstraint.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_multi_contact/planner/contact_invariant/ActivationBoundsConstraint.h"
#include "humanoid_multi_contact/planner/contact_invariant/ActivationRateCost.h"
#include "humanoid_multi_contact/planner/contact_invariant/AugmentedDynamics.h"
#include "humanoid_multi_contact/planner/contact_invariant/AugmentedPreComputation.h"
#include "humanoid_multi_contact/planner/contact_invariant/AugmentedStateCost.h"
#include "humanoid_multi_contact/planner/contact_invariant/AugmentedStateInputConstraint.h"
#include "humanoid_multi_contact/planner/contact_invariant/AugmentedStateInputCost.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureConstraint.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureRows.h"
#include "humanoid_multi_contact/planner/contact_invariant/MomentumRateCost.h"

namespace ocs2::humanoid::multi_contact {
namespace {

/** A term's name and its index in its collection. Passive data. */
struct IndexedName {
  size_t index = 0;
  std::string name;
};

/** The names of `collection`'s terms in the order they were added. */
template <typename Term>
std::vector<IndexedName> namesInOrder(const Collection<Term>& collection) {
  std::vector<IndexedName> names;
  for (const std::pair<const std::string, size_t>& entry :
       collection.getTermNameMap()) {
    names.push_back(IndexedName{.index = entry.second, .name = entry.first});
  }
  std::sort(names.begin(), names.end(),
            [](const IndexedName& a, const IndexedName& b) {
              return a.index < b.index;
            });
  return names;
}

/** Moves every term of `source` into `target`, wrapped in a `Wrapper` of
 * `layout`, in order. */
template <typename Wrapper, typename Term>
void wrapTerms(Collection<Term>& source,
               Collection<Term>& target,
               const ContactInvariantLayout& layout) {
  for (const IndexedName& entry : namesInOrder(source)) {
    target.add(entry.name,
               std::make_unique<Wrapper>(layout, source.extract(entry.name)));
  }
}

/** The feature rows of the point candidates' friction cones. */
std::vector<int> coneRows(const ContactInvariantLayout& layout) {
  std::vector<int> rows;
  for (int i = 0; i < layout.numCandidates(); ++i) {
    if (layout.candidates[i].wrenchDim == kForceDim) {
      rows.push_back(featureRow(i, kConeFeature));
    }
  }
  return rows;
}

/** The feature rows of every candidate's pad clearance. */
std::vector<int> clearanceRows(const ContactInvariantLayout& layout) {
  std::vector<int> rows;
  for (int i = 0; i < layout.numCandidates(); ++i) {
    rows.push_back(featureRow(i, kClearanceFeature));
  }
  return rows;
}

/** InvalidArgument unless `physical` has only the terms augmentProblem()
 * wraps and `layout` matches `features`. */
absl::Status checkWrappable(const OptimalControlProblem& physical,
                            const ContactInvariantLayout& layout,
                            const ContactFeatureFunction& features) {
  if (!physical.stateEqualityConstraintPtr->empty() ||
      !physical.preJumpEqualityConstraintPtr->empty() ||
      !physical.finalEqualityConstraintPtr->empty() ||
      !physical.stateInequalityConstraintPtr->empty() ||
      !physical.preJumpInequalityConstraintPtr->empty() ||
      !physical.finalInequalityConstraintPtr->empty()) {
    return absl::InvalidArgumentError(
        "the contact-invariant problem wraps no state-only constraint of the "
        "physical problem");
  }
  if (physical.dynamicsPtr == nullptr ||
      physical.preComputationPtr == nullptr) {
    return absl::InvalidArgumentError(
        "the physical problem has no dynamics or pre-computation");
  }
  if (features.numCandidates() != layout.numCandidates() ||
      features.stateDim() != layout.physicalStateDim ||
      features.inputDim() != layout.physicalInputDim) {
    return absl::InvalidArgumentError(absl::StrCat(
        "the contact features (", features.numCandidates(), " candidates, ",
        features.stateDim(), " states, ", features.inputDim(),
        " inputs) do not match the layout (", layout.numCandidates(), ", ",
        layout.physicalStateDim, ", ", layout.physicalInputDim, ")"));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<OptimalControlProblem> augmentProblem(
    const OptimalControlProblem& physical,
    const ContactInvariantLayout& layout,
    const ContactFeatureFunction& features,
    std::shared_ptr<const StateManifold> augmentedManifold,
    const ContactInvariantStageWeights& weights,
    const ContactInvariantBarriers& barriers) {
  RETURN_IF_ERROR(checkWrappable(physical, layout, features));
  OptimalControlProblem source(physical);
  OptimalControlProblem problem;
  wrapTerms<AugmentedStateInputCost>(*source.costPtr, *problem.costPtr, layout);
  wrapTerms<AugmentedStateCost>(*source.stateCostPtr, *problem.stateCostPtr,
                                layout);
  wrapTerms<AugmentedStateCost>(*source.preJumpCostPtr, *problem.preJumpCostPtr,
                                layout);
  wrapTerms<AugmentedStateCost>(*source.finalCostPtr, *problem.finalCostPtr,
                                layout);
  wrapTerms<AugmentedStateInputCost>(*source.softConstraintPtr,
                                     *problem.softConstraintPtr, layout);
  wrapTerms<AugmentedStateCost>(*source.stateSoftConstraintPtr,
                                *problem.stateSoftConstraintPtr, layout);
  wrapTerms<AugmentedStateCost>(*source.preJumpSoftConstraintPtr,
                                *problem.preJumpSoftConstraintPtr, layout);
  wrapTerms<AugmentedStateCost>(*source.finalSoftConstraintPtr,
                                *problem.finalSoftConstraintPtr, layout);
  wrapTerms<AugmentedStateInputConstraint>(
      *source.equalityConstraintPtr, *problem.equalityConstraintPtr, layout);
  wrapTerms<AugmentedStateInputConstraint>(*source.inequalityConstraintPtr,
                                           *problem.inequalityConstraintPtr,
                                           layout);
  problem.dynamicsPtr = std::make_unique<AugmentedDynamics>(
      layout, std::move(source.dynamicsPtr));
  problem.preComputationPtr = std::make_unique<AugmentedPreComputation>(
      layout, std::move(source.preComputationPtr),
      std::make_unique<ContactFeatureFunction>(features));
  problem.stateManifoldPtr = std::move(augmentedManifold);

  problem.costPtr->add(
      kContactInvariantTerm,
      std::make_unique<ContactInvariantCost>(layout, weights.contactInvariant));
  problem.costPtr->add(kForceGateTerm, std::make_unique<ContactForceGateCost>(
                                           layout, weights.forceGate));
  problem.costPtr->add(kActivationRateTerm,
                       std::make_unique<ActivationRateCost>(
                           layout, weights.activationRateWeight));
  problem.costPtr->add(kMomentumRateTerm,
                       std::make_unique<MomentumRateCost>(
                           layout, weights.linearMomentumRateWeight,
                           weights.angularMomentumRateWeight));
  problem.stateSoftConstraintPtr->add(
      kActivationBoundsTerm,
      std::make_unique<StateSoftConstraint>(
          std::make_unique<ActivationBoundsConstraint>(layout,
                                                       weights.minimumSupport),
          std::make_unique<SquaredHingePenalty>(barriers.activationBounds)));
  if (const std::vector<int> cones = coneRows(layout); !cones.empty()) {
    problem.softConstraintPtr->add(
        kFrictionConeTerm,
        std::make_unique<StateInputSoftConstraint>(
            std::make_unique<ContactFeatureConstraint>(cones),
            std::make_unique<RelaxedBarrierPenalty>(barriers.frictionCone)));
  }
  problem.softConstraintPtr->add(
      kPadClearanceTerm,
      std::make_unique<StateInputSoftConstraint>(
          std::make_unique<ContactFeatureConstraint>(clearanceRows(layout)),
          std::make_unique<RelaxedBarrierPenalty>(barriers.padClearance)));
  return problem;
}

}  // namespace ocs2::humanoid::multi_contact
