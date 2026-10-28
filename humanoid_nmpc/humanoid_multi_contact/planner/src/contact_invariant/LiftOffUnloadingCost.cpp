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

#include "humanoid_multi_contact/planner/contact_invariant/LiftOffUnloadingCost.h"

#include <utility>
#include <vector>

namespace ocs2::humanoid::multi_contact {

LiftOffUnloadingCost::LiftOffUnloadingCost(std::vector<UnloadingWindow> windows,
                                           scalar_t weight,
                                           scalar_t forceScale)
    : windows_(std::move(windows)), weight_(weight), forceScale_(forceScale) {}

bool LiftOffUnloadingCost::isActive(scalar_t time) const {
  for (const UnloadingWindow& window : windows_) {
    if (time >= window.startTime && time < window.endTime) return true;
  }
  return false;
}

scalar_t LiftOffUnloadingCost::getValue(
    scalar_t time,
    const vector_t& /*state*/,
    const vector_t& input,
    const TargetTrajectories& /*targetTrajectories*/,
    const PreComputation& /*preComp*/) const {
  scalar_t value = 0.0;
  for (const UnloadingWindow& window : windows_) {
    if (time < window.startTime || time >= window.endTime) continue;
    value += weight_ *
             input.segment(window.forceOffset, window.wrenchDim).squaredNorm() /
             (forceScale_ * forceScale_);
  }
  return value;
}

ScalarFunctionQuadraticApproximation
LiftOffUnloadingCost::getQuadraticApproximation(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const TargetTrajectories& targetTrajectories,
    const PreComputation& preComp) const {
  ScalarFunctionQuadraticApproximation result;
  result.setZero(static_cast<int>(state.size()),
                 static_cast<int>(input.size()));
  result.f = getValue(time, state, input, targetTrajectories, preComp);
  const scalar_t scale = 2.0 * weight_ / (forceScale_ * forceScale_);
  for (const UnloadingWindow& window : windows_) {
    if (time < window.startTime || time >= window.endTime) continue;
    result.dfdu.segment(window.forceOffset, window.wrenchDim) +=
        scale * input.segment(window.forceOffset, window.wrenchDim);
    result.dfduu
        .block(window.forceOffset, window.forceOffset, window.wrenchDim,
               window.wrenchDim)
        .diagonal()
        .array() += scale;
  }
  return result;
}

}  // namespace ocs2::humanoid::multi_contact
