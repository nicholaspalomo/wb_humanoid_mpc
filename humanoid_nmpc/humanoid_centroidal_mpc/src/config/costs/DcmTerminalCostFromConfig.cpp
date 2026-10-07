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

#include "pinocchio/fwd.hpp"

#include "humanoid_centroidal_mpc/config/costs/DcmTerminalCostFromConfig.h"

#include <cmath>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_mpc_config/dcm_terminal_cost_config.nproto.h"

namespace ocs2::humanoid {

absl::StatusOr<DcmTerminalCost::Config> dcmTerminalCostConfigFromConfig(const mpc_config::DcmTerminalCostConfig& dcmTerminalCost) {
  DcmTerminalCost::Config config;
  if (dcmTerminalCost.com_height.has_value()) {
    const double comHeight = *dcmTerminalCost.com_height;
    if (!std::isfinite(comHeight) || comHeight <= 0.0) {
      return absl::InvalidArgumentError(absl::StrCat("[DcmTerminalCost] ", DcmTerminalCost::kConfigPrefix, "com_height is ", comHeight,
                                                     ", which is not a pendulum height: give a positive height, or leave ",
                                                     DcmTerminalCost::kConfigPrefix,
                                                     "com_height out for the model's center of mass above its feet at initial_state."));
    }
    config.comHeight = comHeight;
  }
  config.gravity = dcmTerminalCost.gravity;
  config.weights(0) = dcmTerminalCost.weight_x;
  config.weights(1) = dcmTerminalCost.weight_y;
  config.velocityOffsetFactor = dcmTerminalCost.velocity_offset_factor;
  config.supportBlendTime = dcmTerminalCost.support_blend_time;
  RETURN_IF_ERROR(config.validate());
  return config;
}

}  // namespace ocs2::humanoid
