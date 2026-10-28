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

#include "humanoid_multi_contact/planner/contact_invariant/LandingAlignmentCost.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureRows.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// The tilt below which the Hessian's 1 / (2 t) is held: the Gauss-Newton
// curvature of a sole that is already flat.
constexpr scalar_t kMinimumTilt = 1.0e-4;

}  // namespace

LandingAlignmentCost::LandingAlignmentCost(
    const ContactFeatureFunction& features,
    std::vector<LandingWindow> windows,
    scalar_t weight)
    : features_(std::make_unique<ContactFeatureFunction>(features)),
      windows_(std::move(windows)),
      weight_(weight) {}

LandingAlignmentCost::LandingAlignmentCost(const LandingAlignmentCost& other)
    : StateInputCost(other),
      features_(std::make_unique<ContactFeatureFunction>(*other.features_)),
      windows_(other.windows_),
      weight_(other.weight_) {}

bool LandingAlignmentCost::isActive(scalar_t time) const {
  for (const LandingWindow& window : windows_) {
    if (time >= window.startTime && time < window.endTime) return true;
  }
  return false;
}

scalar_t LandingAlignmentCost::getValue(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const TargetTrajectories& /*targetTrajectories*/,
    const PreComputation& /*preComp*/) const {
  const vector_t features = features_->value(state, input);
  scalar_t value = 0.0;
  for (const LandingWindow& window : windows_) {
    if (time < window.startTime || time >= window.endTime) continue;
    value += weight_ * features(featureRow(window.candidate, kTiltFeature));
  }
  return value;
}

ScalarFunctionQuadraticApproximation
LandingAlignmentCost::getQuadraticApproximation(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const TargetTrajectories& /*targetTrajectories*/,
    const PreComputation& /*preComp*/) const {
  const VectorFunctionLinearApproximation features =
      features_->linearApproximation(state, input);
  ScalarFunctionQuadraticApproximation result;
  result.setZero(static_cast<int>(state.size()),
                 static_cast<int>(input.size()));
  for (const LandingWindow& window : windows_) {
    if (time < window.startTime || time >= window.endTime) continue;
    const int row = featureRow(window.candidate, kTiltFeature);
    const scalar_t tilt = features.f(row);
    const vector_t gx = features.dfdx.row(row).transpose();
    const vector_t gu = features.dfdu.row(row).transpose();
    const scalar_t curvature = weight_ / (2.0 * std::max(tilt, kMinimumTilt));
    result.f += weight_ * tilt;
    result.dfdx += weight_ * gx;
    result.dfdu += weight_ * gu;
    result.dfdxx += curvature * gx * gx.transpose();
    result.dfdux += curvature * gu * gx.transpose();
    result.dfduu += curvature * gu * gu.transpose();
  }
  return result;
}

}  // namespace ocs2::humanoid::multi_contact
