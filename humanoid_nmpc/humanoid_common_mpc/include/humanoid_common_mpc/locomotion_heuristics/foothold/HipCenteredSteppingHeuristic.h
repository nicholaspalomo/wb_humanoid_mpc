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
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/locomotion_heuristics/FootholdHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicFormulation.h"

namespace ocs2::humanoid {

/**
 * `hip_centered_stepping`: H_r(Theta) = PTP(R(Theta) r_hip) (Bledt, Appendix C,
 * Table C.1).
 *
 * The foot placed under its own hip: the hip's position relative to the base in
 * the heading frame, rotated into the world by the measured base yaw and
 * projected onto the ground.
 *
 * This is the base case of the dissertation's whole foot-placement story, the
 * "no heuristics" panel of figure 4-13 against which every later addition is
 * measured. On its own it is barely enough to walk with - Bledt's robot could
 * "stably stand and take a few steps forward" and then fell as soon as it had
 * any speed, because a foot placed under the hip at lift-off is well behind the
 * robot by touch-down. It earns its place as the term the velocity-dependent
 * heuristics are summed on top of, and as the only member of the family that
 * tells the left foot from the right at zero velocity. Each landing contact
 * goes under the hip of its own leg (FootholdHeuristicContext::legIndex), so
 * it serves any number of legs.
 *
 * It is also the only foothold heuristic that MOVES THE ANCHOR, from the stance
 * foot to the measured base, which is a change SwitchedModelReferenceManager's
 * own documentation argues against for good reasons. See FootholdHeuristic and
 * HipCenteredSteppingParameters; the layer warns when this is listed alone, and
 * warns again when the others are listed without it.
 *
 * Like every LocomotionHeuristic, its const methods are safe to call
 * concurrently; configure() is not, and must not run while they do.
 */
class HipCenteredSteppingHeuristic final : public FootholdHeuristic {
 public:
  HipCenteredSteppingHeuristic() = default;
  ~HipCenteredSteppingHeuristic() override = default;

  absl::string_view name() const override {
    return heuristic::kHipCenteredStepping;
  }
  absl::Status configure(
      const LocomotionHeuristicConfig& config,
      const LocomotionHeuristicModelParameters& model) override;
  std::string describe() const override;

  vector2_t computeOffsetPosFootRtWorldEwrtWorldXy(
      const FootholdHeuristicContext& context) const override;
  bool movesAnchor() const override { return true; }

 private:
  HipCenteredSteppingParameters parameters_;
  // One hip per leg (LocomotionHeuristicModelParameters).
  std::vector<vector2_t> posHipsRtBaseEwrtHeadingXy_;
};

}  // namespace ocs2::humanoid
