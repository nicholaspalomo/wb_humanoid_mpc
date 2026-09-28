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
 * `height_compensation`: H_z(v) = a2 v^2 + a1 v + a0 (Bledt, Appendix C, Table C.2).
 *
 * The base height the robot wants as a function of how fast it is being asked to go, quadratic in the speed.
 *
 * Bledt reports this one as having made the motion "smoother and more consistent" and as having marginally reduced
 * the solve time without by itself enlarging the viable operating region (section 4.3, table 4.1) - which is a fair
 * description of what a well-chosen regularization does when it is not fixing a failure: it shapes the cost space so
 * that the optimizer converges sooner on something it was going to find anyway.
 *
 * On a humanoid it also buys something a quadruped needs less of: lowering the base shortens the leg and gives the
 * swing more vertical room within the same joint range, at the cost of knee torque headroom. Which way that trade
 * goes is a question for simulation, which is why `heightPerSpeed` ships at zero.
 *
 * The result is an OFFSET to the reference height and never an absolute height, because the same channel is what
 * adaptToCurrentGroundHeight() writes the terrain into.
 */
class HeightCompensationHeuristic final : public BasePoseHeuristic {
 public:
  HeightCompensationHeuristic() = default;
  ~HeightCompensationHeuristic() override = default;

  absl::string_view name() const override { return heuristic::kHeightCompensation; }
  absl::Status configure(const LocomotionHeuristicConfig& config, const LocomotionHeuristicModelParameters& model) override;
  std::string describe() const override;

  BasePoseOffset offset(const BasePoseHeuristicContext& context) const override;

 private:
  HeightCompensationParameters parameters_;
};

}  // namespace ocs2::humanoid
