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

#include "humanoid_common_mpc/constraint/ContactSwingConstraint.h"

#include <memory>
#include <optional>
#include <utility>

#include "absl/base/nullability.h"
#include "ocs2_core/Dimensions.h"

#include "humanoid_common_mpc/HumanoidPreComputation.h"

namespace ocs2::humanoid {

ContactSwingConstraint::ContactSwingConstraint(
    const SwitchedModelReferenceManager& referenceManager,
    const EndEffectorKinematics<scalar_t>& endEffectorKinematics,
    size_t contactIndex,
    scalar_t positionGain)
    : StateInputConstraint(ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      eeLinearConstraintPtr_(
          std::make_unique<EndEffectorKinematicsLinearVelConstraint>(
              endEffectorKinematics, kNumConstraints)),
      contactIndex_(contactIndex),
      positionGain_(positionGain) {}

ContactSwingConstraint::ContactSwingConstraint(
    const ContactSwingConstraint& rhs)
    : StateInputConstraint(rhs),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      eeLinearConstraintPtr_(rhs.eeLinearConstraintPtr_->clone()),
      contactIndex_(rhs.contactIndex_),
      positionGain_(rhs.positionGain_),
      // Copied deliberately: the SQP solver clones the whole problem once per
      // worker thread.
      isActive_(rhs.isActive_) {}

bool ContactSwingConstraint::isActive(scalar_t time) const {
  if (!isActive_ || referenceManagerPtr_->isInContact(time, contactIndex_)) {
    return false;
  }
  const ContactSwingPlanner* absl_nullable planner =
      referenceManagerPtr_->getContactSwingPlanner();
  return planner != nullptr &&
         planner->reference(contactIndex_, time).has_value();
}

bool ContactSwingConstraint::configureForEvaluation(
    const PreComputation& preComp) const {
  const std::optional<ContactSwingReference> reference =
      cast<HumanoidPreComputation>(preComp).getContactSwingReference(
          contactIndex_);
  if (!reference.has_value()) return false;
  // v + k p - (v_ref + k p_ref) = 0.
  EndEffectorKinematicsLinearVelConstraint::Config config;
  config.Av = matrix_t::Identity(kPositionDim, kPositionDim);
  config.Ax = positionGain_ * matrix_t::Identity(kPositionDim, kPositionDim);
  config.b = -(reference->linvelContactRtWorldEwrtWorld +
               positionGain_ * reference->posContactRtWorldEwrtWorld);
  eeLinearConstraintPtr_->configure(std::move(config));
  return true;
}

vector_t ContactSwingConstraint::getValue(scalar_t time,
                                          const vector_t& state,
                                          const vector_t& input,
                                          const PreComputation& preComp) const {
  if (!configureForEvaluation(preComp)) {
    return vector_t::Zero(kNumConstraints);
  }
  return eeLinearConstraintPtr_->getValue(time, state, input, preComp);
}

VectorFunctionLinearApproximation
ContactSwingConstraint::getLinearApproximation(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& preComp) const {
  if (!configureForEvaluation(preComp)) {
    return VectorFunctionLinearApproximation::Zero(kNumConstraints,
                                                   state.size(), input.size());
  }
  return eeLinearConstraintPtr_->getLinearApproximation(time, state, input,
                                                        preComp);
}

}  // namespace ocs2::humanoid
