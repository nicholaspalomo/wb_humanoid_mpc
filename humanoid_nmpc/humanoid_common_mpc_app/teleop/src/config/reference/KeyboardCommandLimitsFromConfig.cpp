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

#include "humanoid_common_mpc_app/teleop/config/reference/KeyboardCommandLimitsFromConfig.h"

#include <cmath>
#include <optional>
#include <utility>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc_app/teleop/KeyboardVelocityCommand.h"
#include "humanoid_mpc_config/reference_file.nproto.h"

namespace ocs2::humanoid::teleop {
namespace {

/** Writes the reference file's `field` into `target`; InvalidArgument naming it
 * when it is absent or not finite. */
absl::Status readRequired(absl::string_view field,
                          std::optional<double> value,
                          scalar_t* absl_nonnull target) {
  if (!value.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat(field, " is absent; the keyboard command needs it"));
  }
  if (!std::isfinite(*value)) {
    return absl::InvalidArgumentError(
        absl::StrCat(field, " is ", *value, "; it must be a finite number"));
  }
  *target = *value;
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<KeyboardCommandLimits> keyboardCommandLimitsFromConfig(
    const mpc_config::ReferenceFile& file) {
  KeyboardCommandLimits limits;
  // [v_x, v_y, pelvis height change, yaw rate], the order of
  // KeyboardCommandLimits::limits.
  const std::pair<absl::string_view, std::optional<double>> fields[] = {
      {"cmd_max_linvel_base_rt_world_ewrt_heading_x",
       file.cmd_max_linvel_base_rt_world_ewrt_heading_x},
      {"cmd_max_linvel_base_rt_world_ewrt_heading_y",
       file.cmd_max_linvel_base_rt_world_ewrt_heading_y},
      {"cmd_max_err_pos_base_rt_world_ewrt_world_z",
       file.cmd_max_err_pos_base_rt_world_ewrt_world_z},
      {"cmd_max_angvel_base_rt_world_ewrt_world_z",
       file.cmd_max_angvel_base_rt_world_ewrt_world_z},
  };
  for (Eigen::Index i = 0; i < limits.limits.size(); ++i) {
    const std::pair<absl::string_view, std::optional<double>>& field =
        fields[i];
    if (absl::Status read =
            readRequired(field.first, field.second, &limits.limits(i));
        !read.ok()) {
      return read;
    }
    if (limits.limits(i) <= 0.0) {
      return absl::InvalidArgumentError(
          absl::StrCat(field.first, " is ", limits.limits(i),
                       "; it must be positive, since the keyboard command is "
                       "normalized by it"));
    }
  }
  if (absl::Status read = readRequired("nom_pos_base_rt_world_ewrt_world_z",
                                       file.nom_pos_base_rt_world_ewrt_world_z,
                                       &limits.defaultBaseHeight);
      !read.ok()) {
    return read;
  }
  return limits;
}

}  // namespace ocs2::humanoid::teleop
