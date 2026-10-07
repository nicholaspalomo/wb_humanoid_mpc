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

#include "humanoid_common_mpc/config/costs/JointLimitsFromConfig.h"

#include <cmath>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h"

#include "humanoid_mpc_config/joint_limits_config.nproto.h"

namespace ocs2::humanoid {

absl::StatusOr<PieceWisePolynomialBarrierPenalty::Config> jointLimitsBarrierFromConfig(const mpc_config::JointLimitsConfig& jointLimits) {
  if (!std::isfinite(jointLimits.mu)) {
    return absl::InvalidArgumentError(absl::StrCat("joint_limits.mu is ", jointLimits.mu, ", but it must be finite."));
  }
  if (!std::isfinite(jointLimits.delta)) {
    return absl::InvalidArgumentError(absl::StrCat("joint_limits.delta is ", jointLimits.delta, ", but it must be finite."));
  }
  return PieceWisePolynomialBarrierPenalty::Config(jointLimits.mu, jointLimits.delta);
}

}  // namespace ocs2::humanoid
