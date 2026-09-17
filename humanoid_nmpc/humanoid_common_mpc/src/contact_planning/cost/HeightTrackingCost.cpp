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

#include "humanoid_common_mpc/contact_planning/cost/HeightTrackingCost.h"

#include <sstream>

#include "humanoid_common_mpc/contact_planning/problem/StageAccumulator.h"

namespace ocs2::humanoid {

std::string HeightTrackingCost::describe() const {
  std::ostringstream out;
  out << weightLine("w (z_k - z_nom)^2 at every node") << ", z_nom = " << nominalHeight_ << " m";
  return out.str();
}

void HeightTrackingCost::configure(const ContactPlanningConfig& config) {
  checkWeight("height_tracking", config.heightTracking.weight);
  weight_ = config.heightTracking.weight;
  nominalHeight_ = config.shared.comHeight;
}

void HeightTrackingCost::addToStage(const ContactPlanningContext& /*ctx*/, int /*node*/, StageAccumulator& stage) const {
  stage.addQuadraticResidual({{idx_.height, 1.0}}, {}, -nominalHeight_, weight_);
}

}  // namespace ocs2::humanoid
