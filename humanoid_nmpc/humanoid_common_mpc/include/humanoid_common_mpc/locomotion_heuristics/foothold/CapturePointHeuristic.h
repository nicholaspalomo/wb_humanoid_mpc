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
 * `capture_point`: H_r(pdot, Phi) = gain sqrt(p_z / g) (pdot - pdot_d) (Bledt, Appendix C, Table C.1).
 *
 * Step into the VELOCITY ERROR by the capture-point distance of a linear inverted pendulum of the measured
 * centre-of-mass height.
 *
 * The one member of the family that is feedback on a disturbance rather than a function of the command: it is
 * identically zero whenever the robot is moving as fast as it was asked to, and grows only when a push, a slip or a
 * model error has opened a gap between the two. Bledt lists it among the ANALYTIC heuristics because it needs no
 * fitting - it is Pratt's capture point, sqrt(z/g) being the time constant of the pendulum, and this repository
 * already uses the same quantity in its DCM terminal cost and its ZMP and DCM viewer markers.
 *
 * Under this controller it is the heuristic with the clearest mandate. The foot cost's xy weights are the only thing
 * in the problem that can move a foot sideways in response to a push while the mode schedule holds the timing fixed,
 * and nothing is currently writing them a target.
 */
class CapturePointHeuristic final : public FootholdHeuristic {
 public:
  CapturePointHeuristic() = default;
  ~CapturePointHeuristic() override = default;

  absl::string_view name() const override { return heuristic::kCapturePoint; }
  absl::Status configure(const LocomotionHeuristicConfig& config, const LocomotionHeuristicModelParameters& model) override;
  std::string describe() const override;

  vector2_t offset(const FootholdHeuristicContext& context) const override;

 private:
  CapturePointParameters parameters_;
  scalar_t nominalComHeight_ = 0.0;
};

}  // namespace ocs2::humanoid
