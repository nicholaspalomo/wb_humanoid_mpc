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

#include "humanoid_common_mpc_app/teleop/KeyboardVelocityCommand.h"

#include <cmath>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc_app/teleop/config/reference/KeyboardCommandLimitsFromConfig.h"
#include "humanoid_mpc_config/reference_file.nproto.h"

namespace ocs2::humanoid::teleop {
absl::StatusOr<KeyboardCommandLimits> loadKeyboardCommandLimits(const std::string& referenceFile) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(referenceFile, error)) {
    return absl::NotFoundError(absl::StrCat("The reference file ", referenceFile, " does not exist"));
  }
  ASSIGN_OR_RETURN(const mpc_config::ReferenceFile reference, loadReferenceFile(referenceFile));
  absl::StatusOr<KeyboardCommandLimits> limits = keyboardCommandLimitsFromConfig(reference);
  if (!limits.ok()) {
    return withConfigFile(limits.status(), referenceFile);
  }
  return limits;
}

absl::StatusOr<vector4_t> parseKeyboardCommandLine(absl::string_view line) {
  const std::vector<absl::string_view> words = absl::StrSplit(line, absl::ByAnyChar(" \t\r\n"), absl::SkipEmpty());
  vector4_t command = vector4_t::Zero();
  for (size_t i = 0; i < words.size() && i < 4; ++i) {
    double value = 0.0;
    if (!absl::SimpleAtod(words[i], &value) || !std::isfinite(value)) {
      return absl::InvalidArgumentError(absl::StrCat("`", words[i], "` is not a number"));
    }
    command(static_cast<Eigen::Index>(i)) = value;
  }
  return command;
}

humanoid_mpc_msgs::WalkingVelocityCommand keyboardCommandToMessage(const vector4_t& command, const KeyboardCommandLimits& limits) {
  const vector4_t clamped = command.cwiseMin(limits.limits).cwiseMax(-limits.limits);
  humanoid_mpc_msgs::WalkingVelocityCommand message;
  message.set_linear_velocity_x(clamped(0) / limits.limits(0));
  message.set_linear_velocity_y(clamped(1) / limits.limits(1));
  message.set_desired_pelvis_height(limits.defaultBaseHeight + clamped(2));
  message.set_angular_velocity_z(clamped(3) / limits.limits(3));
  return message;
}

}  // namespace ocs2::humanoid::teleop
