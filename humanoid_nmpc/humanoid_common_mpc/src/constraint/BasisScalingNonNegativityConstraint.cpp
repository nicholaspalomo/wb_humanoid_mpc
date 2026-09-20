/******************************************************************************
Copyright (c) 2025. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include "humanoid_common_mpc/constraint/BasisScalingNonNegativityConstraint.h"

#include <memory>
#include <utility>

#include <ocs2_core/penalties/penalties/SquaredHingePenalty.h>

namespace ocs2::humanoid {

namespace {
/**
 * The penalty that bounds λ from below.
 *
 * Gated, the configured barrier: the term is off during swing anyway, and its small negative gradient inside
 * [0, delta] cannot be reached where it would matter.
 *
 * Un-gated, a one-sided squared hinge with its zero exactly on λ = 0. The barrier's derivative at λ = 0 is
 * `-mu * delta / 2 < 0`, which pays the solver to lift λ off zero - a bribe to invent contact force on a foot in
 * flight. See the class documentation, and makeContactConePenalty() in HumanoidCostConstraintFactory.cpp, which makes
 * the identical substitution for the three un-gated cone terms.
 */
std::unique_ptr<PenaltyBase> makeLambdaPenalty(const PieceWisePolynomialBarrierPenalty::Config& barrierSettings, bool scheduleGated) {
  if (scheduleGated) {
    return std::unique_ptr<PenaltyBase>(new PieceWisePolynomialBarrierPenalty(barrierSettings));
  }
  return std::unique_ptr<PenaltyBase>(new SquaredHingePenalty(SquaredHingePenalty::Config(barrierSettings.mu, 0.0)));
}
}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

BasisScalingNonNegativityConstraint::BasisScalingNonNegativityConstraint(const SwitchedModelReferenceManager& referenceManager,
                                                                         size_t contactIndex,
                                                                         size_t lambdaStartIdx,
                                                                         size_t numBasis,
                                                                         PieceWisePolynomialBarrierPenalty::Config barrierSettings,
                                                                         bool scheduleGated)
    : referenceManagerPtr_(&referenceManager),
      contactIndex_(contactIndex),
      lambdaStartIdx_(lambdaStartIdx),
      numBasis_(numBasis),
      scheduleGated_(scheduleGated),
      barrierConfig_(barrierSettings),
      penaltyPtr_(makeLambdaPenalty(barrierSettings, scheduleGated)) {}

BasisScalingNonNegativityConstraint::BasisScalingNonNegativityConstraint(const BasisScalingNonNegativityConstraint& rhs)
    : referenceManagerPtr_(rhs.referenceManagerPtr_),
      contactIndex_(rhs.contactIndex_),
      lambdaStartIdx_(rhs.lambdaStartIdx_),
      numBasis_(rhs.numBasis_),
      scheduleGated_(rhs.scheduleGated_),
      barrierConfig_(rhs.barrierConfig_),
      penaltyPtr_(rhs.penaltyPtr_->clone()) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

bool BasisScalingNonNegativityConstraint::isActive(scalar_t time) const {
  // Under the contact-implicit formulation this is the only bound on λ, and the mode schedule must not be allowed to
  // switch it off for a foot the solver may choose to load; see contactConstraintsAreScheduleGated().
  if (!scheduleGated_) return true;
  // Otherwise only active when the foot is in contact — swing-phase λ are already zeroed by the ZeroWrench hard
  // equality constraint, which is what makes the gate sound.
  return referenceManagerPtr_->getContactFlags(time)[contactIndex_];
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t BasisScalingNonNegativityConstraint::getValue(scalar_t time,
                                                       const vector_t& state,
                                                       const vector_t& input,
                                                       const TargetTrajectories& targetTrajectories,
                                                       const PreComputation& preComp) const {
  // h_i = λ_i − 0 = λ_i.  Penalty grows as λ_i → 0⁺.
  const vector_t lambda = input.segment(lambdaStartIdx_, numBasis_);
  return lambda.unaryExpr([&](scalar_t h) { return penaltyPtr_->getValue(0.0, h); }).sum();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

ScalarFunctionQuadraticApproximation BasisScalingNonNegativityConstraint::getQuadraticApproximation(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const TargetTrajectories& targetTrajectories,
    const PreComputation& preComp) const {
  const size_t stateDim = state.size();
  const size_t inputDim = input.size();

  const vector_t lambda = input.segment(lambdaStartIdx_, numBasis_);

  ScalarFunctionQuadraticApproximation cost;

  // Value
  cost.f = lambda.unaryExpr([&](scalar_t h) { return penaltyPtr_->getValue(0.0, h); }).sum();

  // Gradients
  cost.dfdx = vector_t::Zero(stateDim);
  cost.dfdu = vector_t::Zero(inputDim);
  // d/d(λ_i) of penalty(λ_i) = penalty'(λ_i)
  cost.dfdu.segment(lambdaStartIdx_, numBasis_) = lambda.unaryExpr([&](scalar_t h) { return penaltyPtr_->getDerivative(0.0, h); });

  // Hessians (diagonal — each λ_i is independent)
  cost.dfdxx = matrix_t::Zero(stateDim, stateDim);
  cost.dfdux = matrix_t::Zero(inputDim, stateDim);
  cost.dfduu = matrix_t::Zero(inputDim, inputDim);
  cost.dfduu.diagonal().segment(lambdaStartIdx_, numBasis_) =
      lambda.unaryExpr([&](scalar_t h) { return penaltyPtr_->getSecondDerivative(0.0, h); });

  return cost;
}

void BasisScalingNonNegativityConstraint::setBarrierPenalty(const PieceWisePolynomialBarrierPenalty::Config& barrierSettings) {
  barrierConfig_ = barrierSettings;
  // Both penalties take (mu, delta). Un-gated the hinge's delta stays 0: it is the offset of the hinge's zero, and the
  // whole point of using a hinge for λ >= 0 is that its zero sits exactly on λ = 0. Writing the barrier's smoothing
  // width there would demand a positive λ from every generator and reinstate the bribe this term exists to remove.
  const vector_t parameters = (vector_t(2) << barrierSettings.mu, scheduleGated_ ? barrierSettings.delta : 0.0).finished();
  penaltyPtr_->setParameters(parameters);
}

}  // namespace ocs2::humanoid
