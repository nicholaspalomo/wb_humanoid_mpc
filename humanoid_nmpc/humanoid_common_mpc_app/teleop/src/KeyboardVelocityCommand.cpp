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
#include <exception>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/misc/PropertyTree.h>

namespace ocs2::humanoid::teleop {
namespace {

absl::Status loadLimit(const PropertyTree& tree, const std::string& referenceFile, absl::string_view key, scalar_t& value) {
  const std::optional<scalar_t> loaded = tree.getOptional<scalar_t>(key);
  if (!loaded.has_value() || !std::isfinite(*loaded)) {
    return absl::InvalidArgumentError(
        absl::StrCat(referenceFile, ": `", key, "` is missing or not a number; the keyboard command needs it"));
  }
  value = *loaded;
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<KeyboardCommandLimits> loadKeyboardCommandLimits(const std::string& referenceFile) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(referenceFile, error)) {
    return absl::NotFoundError(absl::StrCat("The reference file ", referenceFile, " does not exist"));
  }
  PropertyTree tree;
  try {
    loadData::readPropertyTree(referenceFile, tree);
  } catch (const std::exception& exception) {
    return absl::InvalidArgumentError(absl::StrCat("The reference file ", referenceFile, " does not parse: ", exception.what()));
  }
  KeyboardCommandLimits limits;
  // The keys the MPC scales the command with (TargetTrajectoriesCalculatorBase::reloadCommandLimits()).
  // LINT.IfChange(keyboard_command_limits)
  const std::pair<absl::string_view, scalar_t*> keys[] = {
      {"maxDisplacementVelocityX", &limits.limits(0)},  {"maxDisplacementVelocityY", &limits.limits(1)},
      {"maxDeltaPelvisHeight", &limits.limits(2)},      {"maxRotationVelocity", &limits.limits(3)},
      {"defaultBaseHeight", &limits.defaultBaseHeight},
  };
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/command/TargetTrajectoriesCalculatorBase.cpp:command_limits)
  for (const std::pair<absl::string_view, scalar_t*>& key : keys) {
    const absl::Status loaded = loadLimit(tree, referenceFile, key.first, *key.second);
    if (!loaded.ok()) return loaded;
  }
  // The command is normalized by the limits: a limit of 0 would send 0 / 0.
  for (Eigen::Index i = 0; i < limits.limits.size(); ++i) {
    if (limits.limits(i) <= 0.0) {
      return absl::InvalidArgumentError(
          absl::StrCat(referenceFile, ": `", keys[i].first, ": ", limits.limits(i), "` must be positive; the command is normalized by it"));
    }
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
