/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include "humanoid_common_mpc/contact_planning/cost/StepWidthCost.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace ocs2::humanoid {

std::string StepWidthCost::describe() const {
  return weightLine("w (e_y . (p_L - p_R) - " + std::to_string(nominalStepWidth_) + ")^2 at every node (first-order in the heading)");
}

void StepWidthCost::configure(const ContactPlanningConfig& config) {
  checkWeight("step_width", config.stepWidth.weight);
  weight_ = config.stepWidth.weight;
  nominalStepWidth_ = config.stepWidth.nominalStepWidth;
}

void StepWidthCost::addToStage(const ContactPlanningContext& ctx, int node, StageAccumulator& stage) const {
  const vector2_t& ey = ctx.axesAt(node)[1];
  Coefficients xc;
  for (int axis = 0; axis < 2; ++axis) {
    xc.push_back({idx_.foot[0][axis], ey(axis)});
    xc.push_back({idx_.foot[1][axis], -ey(axis)});
  }
  const vector2_t dNominal =
      ctx.hasHeading() ? vector2_t(ctx.nominal->feet[static_cast<size_t>(node)][0] - ctx.nominal->feet[static_cast<size_t>(node)][1])
                       : vector2_t(vector2_t::Zero());
  const auto [g, offset] = ctx.frameTerm(node, 1, dNominal);
  if (ctx.hasHeading()) xc.push_back({idx_.heading, g});
  stage.addQuadraticResidual(xc, {}, -nominalStepWidth_ + offset, weight_);
}

}  // namespace ocs2::humanoid
