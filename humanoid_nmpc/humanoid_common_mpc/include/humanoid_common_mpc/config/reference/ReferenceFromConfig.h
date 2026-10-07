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

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_mpc_config/reference_file.nproto.h"

namespace ocs2::humanoid {

/**
 * The command limits and the command shaping of a robot's reference file, checked (ReferenceSettings).
 *
 * The file must give what the MPC cannot do without: target_displacement_velocity, target_rotation_velocity,
 * max_displacement_velocity_x, max_displacement_velocity_y, max_delta_pelvis_height, max_rotation_velocity and
 * default_base_height. target_joint_state_interpolation_time_constant stays absent when the file has none, and the
 * centroidal MPC, which requires it, refuses that. Every value is a finite number (inf and nan are refused), and the
 * command filter's break frequency is >= 0 (BreakFrequencyAlphaFilter::validateBreakFrequency()).
 *
 * @return The settings; InvalidArgument naming the field that is absent or out of range.
 */
absl::StatusOr<ReferenceSettings> referenceSettingsFromConfig(const mpc_config::ReferenceFile& file);

/**
 * The fields of the reference file that a running MPC applies when the file is reloaded, by their paths: the
 * ReferenceSettings members that TargetTrajectoriesCalculatorBase::applyCommandLimits() and
 * ProceduralMpcMotionManager::applyCommandLimits() apply. The schema marks exactly these RELOAD_HOT (testConfigReload).
 */
absl::Span<const absl::string_view> hotReferenceFileFields();

/**
 * The joint positions [rad] of the reference file's default posture, in the order of the MPC model's joints.
 *
 * default_joint_state names every joint of `mpcJointNames` exactly once, in any order: the index of a joint is the
 * model's, never the file's.
 *
 * @param file The reference file.
 * @param mpcJointNames The joints of the MPC model in state order (ModelSettings::mpcModelJointNames).
 * @param fixedJointNames The joints the task file fixes (ModelSettings::fixedJointNames), which the posture may not name.
 * @return The posture, one entry per MPC joint; InvalidArgument listing every joint that is missing, named twice, fixed
 *         or not a joint of the model, and naming a value that is not a finite number.
 */
absl::StatusOr<vector_t> defaultJointStateFromConfig(const mpc_config::ReferenceFile& file,
                                                     absl::Span<const std::string> mpcJointNames,
                                                     absl::Span<const std::string> fixedJointNames);

}  // namespace ocs2::humanoid
