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

#include "humanoid_multi_contact/planner/contact_invariant/MomentumRateCost.h"

#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PreComputation.h"

#include "humanoid_multi_contact/planner/contact_invariant/AugmentedPreComputation.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureRows.h"

namespace ocs2::humanoid::multi_contact {

MomentumRateCost::MomentumRateCost(const ContactInvariantLayout& layout,
                                   scalar_t linearWeight,
                                   scalar_t angularWeight)
    : layout_(layout), weights_(kMomentumRateFeatures) {
  weights_.head<kForceDim>().setConstant(linearWeight);
  weights_.tail<kTorqueDim>().setConstant(angularWeight);
}

scalar_t MomentumRateCost::getValue(
    scalar_t /*time*/,
    const vector_t& /*state*/,
    const vector_t& /*input*/,
    const TargetTrajectories& /*targetTrajectories*/,
    const PreComputation& preComp) const {
  const vector_t& features =
      cast<AugmentedPreComputation>(preComp).features().f;
  const vector_t rate = features.segment(
      momentumRateRow(layout_.numCandidates()), kMomentumRateFeatures);
  return rate.dot(weights_.cwiseProduct(rate));
}

ScalarFunctionQuadraticApproximation
MomentumRateCost::getQuadraticApproximation(
    scalar_t /*time*/,
    const vector_t& /*state*/,
    const vector_t& /*input*/,
    const TargetTrajectories& /*targetTrajectories*/,
    const PreComputation& preComp) const {
  const VectorFunctionLinearApproximation& features =
      cast<AugmentedPreComputation>(preComp).features();
  const int row = momentumRateRow(layout_.numCandidates());
  const vector_t rate = features.f.segment(row, kMomentumRateFeatures);
  const matrix_t jx = features.dfdx.middleRows(row, kMomentumRateFeatures);
  const matrix_t ju = features.dfdu.middleRows(row, kMomentumRateFeatures);
  const vector_t weightedRate = weights_.cwiseProduct(rate);
  ScalarFunctionQuadraticApproximation result;
  result.f = rate.dot(weightedRate);
  result.dfdx = 2.0 * jx.transpose() * weightedRate;
  result.dfdu = 2.0 * ju.transpose() * weightedRate;
  const matrix_t wjx = weights_.asDiagonal() * jx;
  result.dfdxx = 2.0 * jx.transpose() * wjx;
  result.dfdux = 2.0 * ju.transpose() * wjx;
  result.dfduu = 2.0 * ju.transpose() * weights_.asDiagonal() * ju;
  return result;
}

}  // namespace ocs2::humanoid::multi_contact
