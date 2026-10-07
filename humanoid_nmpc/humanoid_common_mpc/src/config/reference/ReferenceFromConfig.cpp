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

#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"

#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_common_mpc/reference_manager/BreakFrequencyAlphaFilter.h"
#include "humanoid_mpc_config/joint_value.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.h"

namespace ocs2::humanoid {
namespace {

/** `value` of the reference file's `field`; InvalidArgument naming the field unless it is a finite number. */
absl::StatusOr<scalar_t> finite(absl::string_view field, double value) {
  if (!std::isfinite(value)) {
    return absl::InvalidArgumentError(absl::StrCat(field, " is ", value, "; it must be a finite number"));
  }
  return value;
}

/** The value of the reference file's `field`, which the MPC requires; InvalidArgument naming it when it is absent. */
absl::StatusOr<scalar_t> required(absl::string_view field, std::optional<double> value) {
  if (!value.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(field, " is absent; the reference file must give it"));
  }
  return finite(field, *value);
}

// What the reloaders of a reference file apply (hotReferenceFileFields()).
// LINT.IfChange(hot_reference_file_fields)
constexpr absl::string_view kHotReferenceFileFields[] = {
    "target_displacement_velocity", "target_rotation_velocity", "max_displacement_velocity_x",
    "max_displacement_velocity_y",  "max_delta_pelvis_height",  "max_rotation_velocity",
    "max_linear_acceleration",      "max_angular_acceleration", "velocity_command_filter_break_frequency",
    "default_base_height",
};
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/command/TargetTrajectoriesCalculatorBase.cpp:apply_command_limits, //humanoid_nmpc/humanoid_common_mpc/src/reference_manager/ProceduralMpcMotionManager.cpp:apply_command_limits, //humanoid_nmpc/humanoid_mpc_config/reference_file.proto)
// clang-format on

}  // namespace

absl::Span<const absl::string_view> hotReferenceFileFields() {
  return absl::MakeConstSpan(kHotReferenceFileFields);
}

absl::StatusOr<ReferenceSettings> referenceSettingsFromConfig(const mpc_config::ReferenceFile& file) {
  ReferenceSettings settings;
  ASSIGN_OR_RETURN(settings.targetDisplacementVelocity, required("target_displacement_velocity", file.target_displacement_velocity));
  ASSIGN_OR_RETURN(settings.targetRotationVelocity, required("target_rotation_velocity", file.target_rotation_velocity));
  ASSIGN_OR_RETURN(settings.maxDisplacementVelocityX, required("max_displacement_velocity_x", file.max_displacement_velocity_x));
  ASSIGN_OR_RETURN(settings.maxDisplacementVelocityY, required("max_displacement_velocity_y", file.max_displacement_velocity_y));
  ASSIGN_OR_RETURN(settings.maxDeltaPelvisHeight, required("max_delta_pelvis_height", file.max_delta_pelvis_height));
  ASSIGN_OR_RETURN(settings.maxRotationVelocity, required("max_rotation_velocity", file.max_rotation_velocity));
  ASSIGN_OR_RETURN(settings.maxLinearAcceleration, finite("max_linear_acceleration", file.max_linear_acceleration));
  ASSIGN_OR_RETURN(settings.maxAngularAcceleration, finite("max_angular_acceleration", file.max_angular_acceleration));
  settings.velocityCommandFilterBreakFrequency = file.velocity_command_filter_break_frequency;
  if (const absl::Status valid = BreakFrequencyAlphaFilter::validateBreakFrequency(settings.velocityCommandFilterBreakFrequency);
      !valid.ok()) {
    return absl::InvalidArgumentError(absl::StrCat("velocity_command_filter_break_frequency is invalid: ", valid.message(),
                                                   " Give the break frequency of the command filter in Hz, or 0 to switch it off."));
  }
  if (file.target_joint_state_interpolation_time_constant.has_value()) {
    ASSIGN_OR_RETURN(settings.targetJointStateInterpolationTimeConstant,
                     finite("target_joint_state_interpolation_time_constant", *file.target_joint_state_interpolation_time_constant));
  }
  ASSIGN_OR_RETURN(settings.defaultBaseHeight, required("default_base_height", file.default_base_height));
  return settings;
}

absl::StatusOr<vector_t> defaultJointStateFromConfig(const mpc_config::ReferenceFile& file,
                                                     absl::Span<const std::string> mpcJointNames,
                                                     absl::Span<const std::string> fixedJointNames) {
  absl::flat_hash_map<std::string, size_t> indices;
  for (size_t i = 0; i < mpcJointNames.size(); ++i) {
    indices.emplace(mpcJointNames[i], i);
  }
  const absl::flat_hash_set<std::string> fixed(fixedJointNames.begin(), fixedJointNames.end());

  vector_t state = vector_t::Zero(static_cast<Eigen::Index>(mpcJointNames.size()));
  std::vector<bool> given(mpcJointNames.size(), false);
  std::vector<std::string> twice;
  std::vector<std::string> fixedNamed;
  std::vector<std::string> unknown;
  for (const mpc_config::JointValue& entry : file.default_joint_state) {
    const absl::flat_hash_map<std::string, size_t>::const_iterator found = indices.find(entry.joint);
    if (found == indices.end()) {
      (fixed.contains(entry.joint) ? fixedNamed : unknown).push_back(entry.joint);
      continue;
    }
    if (given[found->second]) {
      twice.push_back(entry.joint);
      continue;
    }
    if (!std::isfinite(entry.value)) {
      return absl::InvalidArgumentError(
          absl::StrCat("default_joint_state of ", entry.joint, " is ", entry.value, "; it must be a finite number"));
    }
    state(static_cast<Eigen::Index>(found->second)) = entry.value;
    given[found->second] = true;
  }
  std::vector<std::string> missing;
  for (size_t i = 0; i < mpcJointNames.size(); ++i) {
    if (!given[i]) missing.push_back(mpcJointNames[i]);
  }

  std::vector<std::string> problems;
  if (!missing.empty()) problems.push_back(absl::StrCat("misses the MPC joints ", absl::StrJoin(missing, ", ")));
  if (!twice.empty()) problems.push_back(absl::StrCat("names more than once ", absl::StrJoin(twice, ", ")));
  if (!fixedNamed.empty()) {
    problems.push_back(absl::StrCat("names the joints ", absl::StrJoin(fixedNamed, ", "), ", which the task file fixes"));
  }
  if (!unknown.empty()) problems.push_back(absl::StrCat("names ", absl::StrJoin(unknown, ", "), ", which the MPC model does not have"));
  if (!problems.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("default_joint_state ", absl::StrJoin(problems, "; "),
                     ". It gives every joint of the MPC model exactly once: ", absl::StrJoin(mpcJointNames, ", "), "."));
  }
  return state;
}

}  // namespace ocs2::humanoid
