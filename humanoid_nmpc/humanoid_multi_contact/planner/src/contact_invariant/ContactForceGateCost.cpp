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

#include "humanoid_multi_contact/planner/contact_invariant/ContactForceGateCost.h"

#include <algorithm>
#include <cmath>

#include "ocs2_core/Dimensions.h"

#include "humanoid_multi_contact/planner/contact_invariant/PsdProjection.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// The second derivative of k0 s / (c^2 + k1) in c is k0 s (6 c^2 - 2 k1) /
// (c^2 + k1)^3.
constexpr scalar_t kCurvatureSquareFactor = 6.0;
// The mixed derivative in (c, w) is -4 k0 c w / (c^2 + k1)^2.
constexpr scalar_t kMixedFactor = 4.0;

}  // namespace

ContactForceGateCost::ContactForceGateCost(const ContactInvariantLayout& layout,
                                           const ForceGateWeights& weights)
    : layout_(layout), weights_(weights) {}

vector_t ContactForceGateCost::wrenchScales(int candidate) const {
  const int dim = layout_.candidates[candidate].wrenchDim;
  vector_t scales = vector_t::Constant(dim, 1.0 / weights_.forceScale);
  if (dim > kForceDim) {
    scales.tail(dim - kForceDim)
        .setConstant(1.0 / (weights_.forceScale * weights_.momentArm));
  }
  return scales;
}

vector_t ContactForceGateCost::scaledWrench(const vector_t& input,
                                            int candidate) const {
  const ActivationCandidate& entry = layout_.candidates[candidate];
  return input.segment(entry.forceOffset, entry.wrenchDim)
      .cwiseProduct(wrenchScales(candidate));
}

scalar_t ContactForceGateCost::getValue(
    scalar_t /*time*/,
    const vector_t& state,
    const vector_t& input,
    const TargetTrajectories& /*targetTrajectories*/,
    const PreComputation& /*preComp*/) const {
  scalar_t value = 0.0;
  for (int i = 0; i < layout_.numCandidates(); ++i) {
    const scalar_t c = std::clamp(state(layout_.activationIndex(i)), 0.0, 1.0);
    value += weights_.k0 * scaledWrench(input, i).squaredNorm() /
             (c * c + weights_.k1);
  }
  return value;
}

ScalarFunctionQuadraticApproximation
ContactForceGateCost::getQuadraticApproximation(
    scalar_t /*time*/,
    const vector_t& state,
    const vector_t& input,
    const TargetTrajectories& /*targetTrajectories*/,
    const PreComputation& /*preComp*/) const {
  ScalarFunctionQuadraticApproximation result;
  result.setZero(layout_.stateDim(), layout_.inputDim());
  const scalar_t k0 = weights_.k0;
  for (int i = 0; i < layout_.numCandidates(); ++i) {
    const ActivationCandidate& entry = layout_.candidates[i];
    const int dim = entry.wrenchDim;
    const scalar_t rawActivation = state(layout_.activationIndex(i));
    const bool claimed = rawActivation > 0.0 && rawActivation < 1.0;
    const scalar_t c = std::clamp(rawActivation, 0.0, 1.0);
    const vector_t w = scaledWrench(input, i);
    const scalar_t s = w.squaredNorm();
    const scalar_t d = c * c + weights_.k1;
    result.f += k0 * s / d;

    // Gradient and Hessian in (c, w).
    vector_t gradient = vector_t::Zero(1 + dim);
    matrix_t hessian = matrix_t::Zero(1 + dim, 1 + dim);
    gradient.tail(dim) = 2.0 * k0 * w / d;
    hessian.bottomRightCorner(dim, dim).diagonal().setConstant(2.0 * k0 / d);
    if (claimed) {
      gradient(0) = -2.0 * k0 * c * s / (d * d);
      hessian(0, 0) = k0 * s *
                      (kCurvatureSquareFactor * c * c - 2.0 * weights_.k1) /
                      (d * d * d);
      const vector_t mixed = -kMixedFactor * k0 * c * w / (d * d);
      hessian.block(1, 0, dim, 1) = mixed;
      hessian.block(0, 1, 1, dim) = mixed.transpose();
    }
    const matrix_t projected = projectToPositiveSemidefinite(hessian);

    // w = S u_w: the chain to the input; c is a state.
    const vector_t scales = wrenchScales(i);
    const int cIndex = layout_.activationIndex(i);
    result.dfdx(cIndex) += gradient(0);
    result.dfdxx(cIndex, cIndex) += projected(0, 0);
    result.dfdu.segment(entry.forceOffset, dim) +=
        scales.cwiseProduct(gradient.tail(dim));
    result.dfduu.block(entry.forceOffset, entry.forceOffset, dim, dim) +=
        scales.asDiagonal() * projected.bottomRightCorner(dim, dim) *
        scales.asDiagonal();
    result.dfdux.block(entry.forceOffset, cIndex, dim, 1) +=
        scales.cwiseProduct(projected.block(1, 0, dim, 1));
  }
  return result;
}

}  // namespace ocs2::humanoid::multi_contact
