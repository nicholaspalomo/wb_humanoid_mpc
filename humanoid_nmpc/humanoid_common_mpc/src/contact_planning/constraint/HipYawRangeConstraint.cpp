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

#include "humanoid_common_mpc/contact_planning/constraint/HipYawRangeConstraint.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace ocs2::humanoid {

std::string HipYawRangeConstraint::describe() const {
  std::ostringstream out;
  out << "psi_i - theta in";
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) out << " [" << params_.lower[foot] << ", " << params_.upper[foot] << "]";
  out << " rad (hip yaw limits from the model) at every node, " << penaltyText();
  return out.str();
}

void HipYawRangeConstraint::configure(const ContactPlanningConfig& config) {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    const bool unset = config.hipYawRange.lower[foot] == 0.0 && config.hipYawRange.upper[foot] == 0.0;
    if (!unset && !(config.hipYawRange.lower[foot] < 0.0 && config.hipYawRange.upper[foot] > 0.0)) {
      throw std::invalid_argument("[hip_yaw_range] foot yaw bounds must be lower < 0 < upper");
    }
  }
  params_ = config.hipYawRange;
  configurePenalty(config, config.hipYawRange.slack, "hip_yaw_range");
}

void HipYawRangeConstraint::addRows(const ContactPlanningContext& /*ctx*/, int /*node*/, RowBuilder& rows) const {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    rows.addSoft({{idx_.footYaw[foot], 1.0}, {idx_.heading, -1.0}}, {}, params_.lower[foot], params_.upper[foot], penalty_);
  }
}

}  // namespace ocs2::humanoid
