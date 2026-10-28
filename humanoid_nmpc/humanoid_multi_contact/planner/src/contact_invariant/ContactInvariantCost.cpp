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

#include "humanoid_multi_contact/planner/contact_invariant/ContactInvariantCost.h"

#include <algorithm>

#include "ocs2_core/PreComputation.h"

#include "humanoid_multi_contact/planner/contact_invariant/AugmentedPreComputation.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureRows.h"
#include "humanoid_multi_contact/planner/contact_invariant/PsdProjection.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// The variables a candidate's term couples: its contact variable, then e_i.
constexpr int kCoupledVariables = 1 + kResidualFeatures;
// The contact variable is the first of them.
constexpr int kActivationVariable = 0;

}  // namespace

ContactInvariantCost::ContactInvariantCost(
    const ContactInvariantLayout& layout,
    const ContactInvariantWeights& weights)
    : layout_(layout), weights_(weights) {}

vector_t ContactInvariantCost::residualWeights() const {
  vector_t w(kResidualFeatures);
  w(kDistanceFeature) = 1.0;
  w.segment<kPositionDim>(kSlipFeature).setConstant(weights_.slipWeight);
  w(kTiltFeature) = weights_.tiltWeight;
  return w;
}

scalar_t ContactInvariantCost::residual(const vector_t& features,
                                        int candidate) const {
  const vector_t e = features.segment(featureRow(candidate, kDistanceFeature),
                                      kResidualFeatures);
  return e.dot(residualWeights().cwiseProduct(e));
}

scalar_t ContactInvariantCost::getValue(
    scalar_t /*time*/,
    const vector_t& state,
    const vector_t& /*input*/,
    const TargetTrajectories& /*targetTrajectories*/,
    const PreComputation& preComp) const {
  const vector_t& features =
      cast<AugmentedPreComputation>(preComp).features().f;
  scalar_t value = 0.0;
  for (int i = 0; i < layout_.numCandidates(); ++i) {
    const scalar_t c = std::max(state(layout_.activationIndex(i)), 0.0);
    value += weights_.weight * c * residual(features, i);
  }
  return value;
}

ScalarFunctionQuadraticApproximation
ContactInvariantCost::getQuadraticApproximation(
    scalar_t /*time*/,
    const vector_t& state,
    const vector_t& /*input*/,
    const TargetTrajectories& /*targetTrajectories*/,
    const PreComputation& preComp) const {
  const VectorFunctionLinearApproximation& features =
      cast<AugmentedPreComputation>(preComp).features();
  const vector_t w = residualWeights();
  ScalarFunctionQuadraticApproximation result;
  result.setZero(layout_.stateDim(), layout_.inputDim());
  for (int i = 0; i < layout_.numCandidates(); ++i) {
    const scalar_t rawActivation = state(layout_.activationIndex(i));
    const bool claimed = rawActivation > 0.0;
    const scalar_t c = std::max(rawActivation, 0.0);
    const int row = featureRow(i, kDistanceFeature);
    const vector_t e = features.f.segment(row, kResidualFeatures);
    const vector_t we = w.cwiseProduct(e);
    const scalar_t r = e.dot(we);
    result.f += weights_.weight * c * r;

    // Gradient and Hessian in (c_i, e_i).
    vector_t gradient = vector_t::Zero(kCoupledVariables);
    matrix_t hessian = matrix_t::Zero(kCoupledVariables, kCoupledVariables);
    if (claimed) gradient(kActivationVariable) = weights_.weight * r;
    gradient.tail(kResidualFeatures) = 2.0 * weights_.weight * c * we;
    if (claimed) {
      hessian.block(1, 0, kResidualFeatures, 1) = 2.0 * weights_.weight * we;
      hessian.block(0, 1, 1, kResidualFeatures) =
          2.0 * weights_.weight * we.transpose();
    }
    hessian.bottomRightCorner(kResidualFeatures, kResidualFeatures) =
        (2.0 * weights_.weight * c * w).asDiagonal();
    const matrix_t projected = projectToPositiveSemidefinite(hessian);

    // The Jacobian of (c_i, e_i) in the augmented state and input.
    matrix_t jx = matrix_t::Zero(kCoupledVariables, layout_.stateDim());
    jx(kActivationVariable, layout_.activationIndex(i)) = 1.0;
    jx.bottomRows(kResidualFeatures) =
        features.dfdx.middleRows(row, kResidualFeatures);
    matrix_t ju = matrix_t::Zero(kCoupledVariables, layout_.inputDim());
    ju.bottomRows(kResidualFeatures) =
        features.dfdu.middleRows(row, kResidualFeatures);
    result.dfdx += jx.transpose() * gradient;
    result.dfdu += ju.transpose() * gradient;
    const matrix_t hx = projected * jx;
    result.dfdxx += jx.transpose() * hx;
    result.dfdux += ju.transpose() * hx;
    result.dfduu += ju.transpose() * projected * ju;
  }
  return result;
}

}  // namespace ocs2::humanoid::multi_contact
