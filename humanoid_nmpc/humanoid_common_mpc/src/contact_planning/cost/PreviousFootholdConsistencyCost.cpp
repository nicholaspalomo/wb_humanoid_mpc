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

#include "humanoid_common_mpc/contact_planning/cost/PreviousFootholdConsistencyCost.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace ocs2::humanoid {

std::string PreviousFootholdConsistencyCost::describe() const {
  return weightLine("w sum_i ||p_{i,k} - p_{i,prev}(k + shift)||^2 at every node, when a previous plan is usable");
}

void PreviousFootholdConsistencyCost::configure(const ContactPlanningConfig& config) {
  checkWeight("previous_foothold_consistency", config.previousFootholdConsistency.weight);
  weight_ = config.previousFootholdConsistency.weight;
}

void PreviousFootholdConsistencyCost::addToStage(const ContactPlanningContext& ctx, int node, StageAccumulator& stage) const {
  if (ctx.previousPlanShift < 0 || ctx.previousPlan == nullptr || weight_ <= 0.0) return;
  const int source = std::min(node + ctx.previousPlanShift, static_cast<int>(ctx.previousPlan->footholds.size()) - 1);
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    for (int axis = 0; axis < 2; ++axis) {
      stage.addQuadraticResidual({{idx_.foot[foot][axis], 1.0}}, {}, -ctx.previousPlan->footholds[static_cast<size_t>(source)][foot](axis),
                                 weight_);
    }
  }
}

}  // namespace ocs2::humanoid
