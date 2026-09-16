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

#include "humanoid_common_mpc/contact_planning/cost/ZmpRegularizationCost.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace ocs2::humanoid {

std::string ZmpRegularizationCost::describe() const {
  return weightLine("w ||zmp_k - c_k||^2 on the running nodes");
}

void ZmpRegularizationCost::configure(const ContactPlanningConfig& config) {
  checkWeight("zmp_regularization", config.zmpRegularization.weight);
  weight_ = config.zmpRegularization.weight;
}

void ZmpRegularizationCost::addToStage(const ContactPlanningContext& /*ctx*/, int /*node*/, StageAccumulator& stage) const {
  for (int axis = 0; axis < 2; ++axis) {
    stage.addQuadraticResidual({{idx_.com[axis], -1.0}}, {{idx_.zmp[axis], 1.0}}, 0.0, weight_);
  }
}

}  // namespace ocs2::humanoid
