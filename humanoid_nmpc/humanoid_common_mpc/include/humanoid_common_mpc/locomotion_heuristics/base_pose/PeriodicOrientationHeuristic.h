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

#include "humanoid_common_mpc/locomotion_heuristics/BasePoseHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"

namespace ocs2::humanoid {

/**
 * `periodic_orientation`: H_Theta(Phi) = b1 sin(c1 Phi + d1) (Bledt, Appendix C, Table C.2).
 *
 * The orientation limit cycle a walking robot falls into on its own, stated as a sinusoid in the gait phase.
 *
 * This is the clearest illustration of what the extraction framework is for. The robot's steady-state pitch during a
 * trot is genuinely complicated, and Bledt makes the point that it is "currently too difficult to analyze without
 * data-driven methods" - yet a single sine at the step frequency fits it with R^2 = 0.84, and summed with the
 * velocity-dependent term of `orientation_compensation` it reproduces the measured pitch closely (section 4.2,
 * equation 4.13, figures 4-6 and 4-7). The value of the fit is not its accuracy but that its four numbers mean
 * something an engineer can turn: an amplitude, a frequency and a phase.
 *
 * Because it is a limit cycle at gait frequency it is also the heuristic that could not be expressed by shaping the
 * target trajectory, whose three knots over a one-second horizon would sample two or three whole periods and
 * interpolate linearly between the samples. It only works at the per-node seam.
 */
class PeriodicOrientationHeuristic final : public BasePoseHeuristic {
 public:
  PeriodicOrientationHeuristic() = default;
  ~PeriodicOrientationHeuristic() override = default;

  absl::string_view name() const override { return heuristic::kPeriodicOrientation; }
  absl::Status configure(const LocomotionHeuristicConfig& config, const LocomotionHeuristicModelParameters& model) override;
  std::string describe() const override;

  BasePoseOffset offset(const BasePoseHeuristicContext& context) const override;

 private:
  PeriodicOrientationParameters parameters_;
};

}  // namespace ocs2::humanoid
