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
 * `high_speed_turning`: H_r(pdot x omega) = a1 (pdot x omega) + a0 (Bledt, Appendix C, Table C.2).
 *
 * Throw the feet outwards along the turn radius when a turn is taken at speed.
 *
 * The result the dissertation is most pleased with, and the best argument for its extraction framework. With both
 * `translational_stepping` and `in_place_turning` in place the robot still fell when asked to do both at once: it
 * "tended to fall outwards along the turning radius as if it were an object slipping off a spinning plate", the outer
 * leading foot running out of workspace with no contact foot left able to stabilise the body. The framework surfaced
 * the cross term v x omega as statistically significant, and only afterwards did Bledt recognise what it was - the
 * foot placement that lines up with the resultant of gravity and the centripetal acceleration, which is what animals
 * visibly do when they corner (section 4.3, equation 4.31, figure 4-11).
 *
 * It only exists because the designer had thought to put `pdot x omega` into the candidate variable set: the framework
 * fits simple models between variables it is given and cannot invent a product it was never shown. That caveat is
 * Bledt's own (section 4.6) and is worth carrying into any attempt to extend this layer.
 *
 * With omega = psidot e_z and a horizontal velocity the cross product reduces to (v_y psidot, -v_x psidot), which are
 * the two scalars the coefficients below multiply.
 */
class HighSpeedTurningHeuristic final : public FootholdHeuristic {
 public:
  HighSpeedTurningHeuristic() = default;
  ~HighSpeedTurningHeuristic() override = default;

  absl::string_view name() const override { return heuristic::kHighSpeedTurning; }
  absl::Status configure(const LocomotionHeuristicConfig& config, const LocomotionHeuristicModelParameters& model) override;
  std::string describe() const override;

  vector2_t offset(const FootholdHeuristicContext& context) const override;

 private:
  HighSpeedTurningParameters parameters_;
};

}  // namespace ocs2::humanoid
