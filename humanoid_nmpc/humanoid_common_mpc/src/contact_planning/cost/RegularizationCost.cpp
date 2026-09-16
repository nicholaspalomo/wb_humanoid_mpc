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

#include "humanoid_common_mpc/contact_planning/cost/RegularizationCost.h"

#include <sstream>
#include <stdexcept>

namespace ocs2::humanoid {

std::string RegularizationCost::describe() const {
  std::ostringstream out;
  out << "Q += " << state_ << " I at every node, R += " << input_ << " I on the running nodes";
  return out.str();
}

void RegularizationCost::configure(const ContactPlanningConfig& config) {
  if (config.regularization.state < 0.0 || config.regularization.input < 0.0) {
    throw std::invalid_argument("[regularization] state and input must be >= 0");
  }
  state_ = config.regularization.state;
  input_ = config.regularization.input;
}

void RegularizationCost::addToStage(const ContactPlanningContext& /*ctx*/, int /*node*/, StageAccumulator& stage) const {
  stage.addStateRegularization(state_);
  stage.addInputRegularization(input_);
}

}  // namespace ocs2::humanoid
