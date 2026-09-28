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

#include "humanoid_common_mpc/locomotion_heuristics/FootholdHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"

namespace ocs2::humanoid {

/**
 * `translational_stepping`: H_r(pdot) = a1 pdot + a0 (Bledt, Appendix C, Table C.2).
 *
 * Step in the direction of travel, affinely in the commanded velocity, in the base's own yaw frame.
 *
 * The first heuristic the extraction framework produced and the one with the largest effect on what the robot could
 * do: it roughly doubled the usable stance of each step and multiplied the achievable forward speed (section 4.3,
 * figure 4-9). The mechanism is simple enough to state - a foot placed under the hip at lift-off has fallen behind
 * the robot by touch-down, so the forces over the stance that follows are biased into tipping the robot in the
 * direction it is already going - and it is the reason a step is placed AHEAD of the hip by roughly half of what the
 * robot will travel during the stance.
 *
 * The lateral offset is applied to this foot's own side, so one number in the task file widens the stance rather than
 * shifting the whole robot sideways.
 */
class TranslationalSteppingHeuristic final : public FootholdHeuristic {
 public:
  TranslationalSteppingHeuristic() = default;
  ~TranslationalSteppingHeuristic() override = default;

  absl::string_view name() const override { return heuristic::kTranslationalStepping; }
  absl::Status configure(const LocomotionHeuristicConfig& config, const LocomotionHeuristicModelParameters& model) override;
  std::string describe() const override;

  vector2_t offset(const FootholdHeuristicContext& context) const override;

 private:
  TranslationalSteppingParameters parameters_;
};

}  // namespace ocs2::humanoid
