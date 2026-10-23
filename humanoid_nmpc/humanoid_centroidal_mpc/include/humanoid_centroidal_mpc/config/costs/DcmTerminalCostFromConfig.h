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

#include "absl/status/statusor.h"

#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_mpc_config/dcm_terminal_cost_config.nproto.h"

namespace ocs2::humanoid {

/**
 * The configuration of the DCM terminal cost from the task file's dcm_terminal_cost block: each field becomes the member
 * of the same name (weight_x and weight_y the two weights), and the schema's defaults are DcmTerminalCost::Config's, so
 * an absent field gives the member's default.
 * com_height is optional: absent, it is the model's pendulum, which the configuration leaves unset (comHeight nullopt)
 * for DcmTerminalCost::resolveConfig() to resolve; a height the file gives must be positive.
 *
 * @return The configuration; InvalidArgument for a com_height that is given but not positive (0 included, which used
 *         to stand for the model's) and for every rejection of DcmTerminalCost::Config::validate(), which a value that
 *         is not finite is among.
 */
absl::StatusOr<DcmTerminalCost::Config> dcmTerminalCostConfigFromConfig(const mpc_config::DcmTerminalCostConfig& dcmTerminalCost);

}  // namespace ocs2::humanoid
