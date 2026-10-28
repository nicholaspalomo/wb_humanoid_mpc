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

#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/cost/StateInputCost.h"

#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureFunction.h"

namespace ocs2::humanoid::multi_contact {

/** A candidate of a feature function and the interval before its
 * touch-down in which it aligns with the surface. Passive data. */
struct LandingWindow {
  // The candidate's index in the feature function.
  int candidate = 0;
  // [s] [startTime, endTime): the interval before the touch-down at endTime.
  double startTime = 0.0;
  double endTime = 0.0;
};

/**
 * weight t of every candidate in its landing window, with t = 1 - n_pad . n
 * the tilt feature of a surface pad (ContactFeatureRows.h): a sole lands flat
 * (planner A's solve 3), so that its corners do not reach the surface before
 * its center does. t is 1 - cos(theta) = r^2 / 2 with r = sqrt(2 t), so the
 * cost is the least squares (weight / 2) r^2, whose Gauss-Newton Hessian is
 * weight grad(t) grad(t)' / (2 t), regularized at t = 0. Owns a copy of the
 * features; cloned per solver worker.
 */
class LandingAlignmentCost final : public StateInputCost {
 public:
  /** The windows `windows` of the candidates of `features` (copied). */
  LandingAlignmentCost(const ContactFeatureFunction& features,
                       std::vector<LandingWindow> windows,
                       scalar_t weight);
  ~LandingAlignmentCost() override = default;
  LandingAlignmentCost* absl_nonnull clone() const override {
    return new LandingAlignmentCost(*this);
  }
  LandingAlignmentCost& operator=(const LandingAlignmentCost&) = delete;
  LandingAlignmentCost(LandingAlignmentCost&&) = delete;
  LandingAlignmentCost& operator=(LandingAlignmentCost&&) = delete;

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
  LandingAlignmentCost(const LandingAlignmentCost& other);

  std::unique_ptr<ContactFeatureFunction> features_;
  std::vector<LandingWindow> windows_;
  scalar_t weight_ = 0.0;
};

}  // namespace ocs2::humanoid::multi_contact
