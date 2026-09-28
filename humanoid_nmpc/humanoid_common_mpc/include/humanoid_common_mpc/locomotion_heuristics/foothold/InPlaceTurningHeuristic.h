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
 * `in_place_turning`: H_r(psidot) = a1 psidot + a0 (Bledt, Appendix C, Table C.2).
 *
 * Step along the arc of the turn, affinely in the commanded yaw rate.
 *
 * The rotational counterpart of `translational_stepping`, found the same way and for the same reason: with only the
 * translational term the feet lagged the hips through a fast spin until they reached the end of their workspace,
 * exactly as they lagged the body through a fast walk (section 4.3, figure 4-10). Adding it let the feet LEAD the
 * hips into the turn and raised the achievable yaw rate substantially.
 *
 * `psidot` is the commanded yaw rate rather than the measured one, because the point is to step where the robot is
 * being asked to go. The forward term is signed per foot: on a turn the outer foot travels further along the heading
 * than the inner one, so one coefficient moves the two feet in opposite directions.
 *
 * Distinct from `high_speed_turning`, which is zero at zero speed however fast the robot spins; this one is the term
 * that survives a turn on the spot.
 *
 * Direction: a foot to the LEFT of a body turning counter-clockwise travels BACKWARDS (omega x r), so the fore-aft
 * displacements of the two feet are equal and opposite. The implementation carries the sign, so a positive
 * `forwardPerYawRate` means "lead the hips into the turn"; see the comment on offset().
 */
class InPlaceTurningHeuristic final : public FootholdHeuristic {
 public:
  InPlaceTurningHeuristic() = default;
  ~InPlaceTurningHeuristic() override = default;

  absl::string_view name() const override { return heuristic::kInPlaceTurning; }
  absl::Status configure(const LocomotionHeuristicConfig& config, const LocomotionHeuristicModelParameters& model) override;
  std::string describe() const override;

  vector2_t offset(const FootholdHeuristicContext& context) const override;

 private:
  InPlaceTurningParameters parameters_;
};

}  // namespace ocs2::humanoid
