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

#pragma once

#include <vector>

#include "absl/base/nullability.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/cost/StateInputCost.h"

namespace ocs2::humanoid::multi_contact {

/** A contact's wrench and the interval before its lift-off in which it is
 * unloaded. Passive data. */
struct UnloadingWindow {
  // The wrench's first index in the MPC's input and its size (3 or 6).
  int forceOffset = 0;
  int wrenchDim = 0;
  // [s] [startTime, endTime): the interval before the lift-off at endTime.
  double startTime = 0.0;
  double endTime = 0.0;
};

/**
 * weight |w / forceScale|^2 of every contact in its unloading window: a
 * contact is unloaded before it lifts off (planner A's solve 3). The
 * certifier judges the momentum rate of a knot by the central difference of
 * the plan's momenta, which at the first knot after a lift-off averages the
 * interval before it; unloaded there, the contact that lifts off is not
 * needed by that average, as a quasi-static robot that shifts its weight
 * before it lifts a foot does not need it. Quadratic in the input, so its
 * approximation is exact. A value type.
 */
class LiftOffUnloadingCost final : public StateInputCost {
 public:
  /** `windows` at `weight`, the wrenches over `forceScale` [N]. */
  LiftOffUnloadingCost(std::vector<UnloadingWindow> windows,
                       scalar_t weight,
                       scalar_t forceScale);
  ~LiftOffUnloadingCost() override = default;
  LiftOffUnloadingCost* absl_nonnull clone() const override {
    return new LiftOffUnloadingCost(*this);
  }
  LiftOffUnloadingCost& operator=(const LiftOffUnloadingCost&) = delete;
  LiftOffUnloadingCost(LiftOffUnloadingCost&&) = delete;
  LiftOffUnloadingCost& operator=(LiftOffUnloadingCost&&) = delete;

  bool isActive(scalar_t time) const override;
  scalar_t getValue(scalar_t time,
                    const vector_t& state,
                    const vector_t& input,
                    const TargetTrajectories& targetTrajectories,
                    const PreComputation& preComp) const override;
  ScalarFunctionQuadraticApproximation getQuadraticApproximation(
      scalar_t time,
      const vector_t& state,
      const vector_t& input,
      const TargetTrajectories& targetTrajectories,
      const PreComputation& preComp) const override;

 private:
  LiftOffUnloadingCost(const LiftOffUnloadingCost& other) = default;

  std::vector<UnloadingWindow> windows_;
  scalar_t weight_ = 0.0;
  scalar_t forceScale_ = 1.0;
};

}  // namespace ocs2::humanoid::multi_contact
