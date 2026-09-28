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
 * `orientation_compensation`: H_Theta(pdot) = a1 pdot + a0 (Bledt, Appendix C, Table C.2).
 *
 * Roll from the commanded LATERAL velocity and pitch from the commanded FORWARD velocity, each an affine function of
 * one component of the command in the base's own yaw frame.
 *
 * Bledt did not design this one; he found it. Running the optimizer offline over a sweep of velocity commands and
 * regressing the orientation it chose against the command produced two relationships with R^2 of 0.98 and 0.95
 * (section 4.2, figure 4-5), and injecting them back as regularization both enlarged the viable operating region and
 * made the resulting motion smoother - partly because the simplified orientation dynamics the control model assumes
 * are a better approximation when the body is closer to level at speed (section 4.3, figure 4-12).
 *
 * On this controller it is shaping a reference that is currently a hard zero at every speed, so it is the first thing
 * in the layer worth sweeping.
 */
class OrientationCompensationHeuristic final : public BasePoseHeuristic {
 public:
  OrientationCompensationHeuristic() = default;
  ~OrientationCompensationHeuristic() override = default;

  absl::string_view name() const override { return heuristic::kOrientationCompensation; }
  absl::Status configure(const LocomotionHeuristicConfig& config, const LocomotionHeuristicModelParameters& model) override;
  std::string describe() const override;

  BasePoseOffset offset(const BasePoseHeuristicContext& context) const override;

 private:
  OrientationCompensationParameters parameters_;
};

}  // namespace ocs2::humanoid
