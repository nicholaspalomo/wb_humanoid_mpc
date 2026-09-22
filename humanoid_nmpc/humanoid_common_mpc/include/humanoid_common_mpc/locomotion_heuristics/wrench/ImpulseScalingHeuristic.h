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

#include <string>

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/WrenchHeuristic.h"

namespace ocs2::humanoid {

/**
 * `impulse_scaling`: H_f(Phi) = m g / (F beta) (Bledt, Appendix C, Table C.1).
 *
 * The vertical stance force reference scaled by the reciprocal of the foot's stance duty factor.
 *
 * The statement is about IMPULSE rather than force, which is where the name comes from: over one gait cycle the
 * vertical impulse the feet deliver has to equal the robot's weight times the cycle, so a foot that is down for a
 * fraction beta of the cycle must average m g / beta while it is down. Bledt credits the idea to the vertical impulse
 * scaling of the MIT Cheetah 2's bounding controller (section 4.1). It is an analytic heuristic: there is nothing to
 * fit, only a duty factor to read off the gait.
 *
 * The reference this shapes divides the weight over the feet in contact AT THIS INSTANT, i.e. it is the beta = 1
 * case, so the offset below is the difference between the two and an empty list changes nothing. What the correction
 * buys is anticipation: during double support it asks the feet to push a little harder than static equilibrium
 * requires, because a single-support phase is coming in which one of them will carry everything.
 *
 * Purely vertical, which is what keeps the cheap yaw-invariant input path valid for the whole layer whenever this is
 * the only wrench heuristic listed - see WrenchHeuristic::producesHorizontalForce().
 */
class ImpulseScalingHeuristic final : public WrenchHeuristic {
 public:
  ImpulseScalingHeuristic() = default;
  ~ImpulseScalingHeuristic() override = default;

  absl::string_view name() const override { return heuristic::kImpulseScaling; }
  absl::Status configure(const LocomotionHeuristicConfig& config, const LocomotionHeuristicModelParameters& model) override;
  std::string describe() const override;

  vector3_t forceOffset(const WrenchHeuristicContext& context, size_t contactIndex) const override;

 private:
  ImpulseScalingParameters parameters_;
};

}  // namespace ocs2::humanoid
