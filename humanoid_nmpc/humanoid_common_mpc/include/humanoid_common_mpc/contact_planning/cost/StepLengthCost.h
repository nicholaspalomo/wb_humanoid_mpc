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

#include "humanoid_common_mpc/contact_planning/cost/LipWeightedCost.h"

namespace ocs2::humanoid {

/**
 * `step_length`: w sum_i ||dp_{i,k} - d_nom (1 - c_{i,k})||^2 on the running nodes, the cyclic step-length term.
 *
 * A swinging foot (c = 0) is drawn to advance by the per-node displacement d_nom a cyclic gait at the commanded velocity
 * needs, a standing foot (c = 1) to stay put. Over a stride T_stride each foot advances v_cmd T_stride, all of it during
 * its swing of duration T_swing, so d_nom = v_cmd dt T_stride / T_swing with the nominal cadence of the gait limits,
 * T_swing = minSwingDuration and T_stride = 2 (minSwingDuration + minDoubleSupportDuration). The residual is affine in
 * the QP variables (dp and the relaxed binary c), so the term stays a convex quadratic in every relaxation.
 *
 * It complements velocity_tracking, which alone is indifferent between a few long steps and many short ones at the
 * same average speed and, from rest, favours the short quick steps that accelerate the pendulum fastest: this term ties
 * the step length to the commanded speed, so a ramped command lengthens the steps progressively.
 */
class StepLengthCost final : public LipWeightedCost {
 public:
  std::string describe() const override;
  void configure(const ContactPlanningConfig& config) override;
  NodeSet nodeSet() const override { return NodeSet::RUNNING; }
  void addToStage(const ContactPlanningContext& ctx, int node, StageAccumulator& stage) const override;

  /** d_nom of a foot in swing: the per-node displacement along each axis for `velocityCommand` at the nominal cadence. */
  vector2_t nominalDisplacementPerNode(const vector2_t& velocityCommand, scalar_t dt) const;

  /** T_stride / T_swing of the nominal cadence (the factor between the CoM's and a swinging foot's advance per node). */
  scalar_t strideToSwingRatio() const { return strideToSwingRatio_; }

 private:
  scalar_t strideToSwingRatio_ = 1.0;
};

}  // namespace ocs2::humanoid
