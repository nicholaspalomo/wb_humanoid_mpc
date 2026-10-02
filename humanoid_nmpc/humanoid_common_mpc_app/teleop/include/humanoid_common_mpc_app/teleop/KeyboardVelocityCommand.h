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

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"

namespace ocs2::humanoid::teleop {

/**
 * The command limits of the robot's reference.yaml, as the MPC reads them (TargetTrajectoriesCalculatorBase): the
 * keyboard command is clamped to them and normalized by them, and the MPC scales the normalized command back by its own
 * copy, so the two must be the same numbers.
 */
struct KeyboardCommandLimits {
  /** [v_x m/s, v_y m/s, pelvis height change m, yaw rate rad/s]: maxDisplacementVelocityX, maxDisplacementVelocityY,
   *  maxDeltaPelvisHeight, maxRotationVelocity. */
  vector4_t limits = vector4_t(0.5, 0.3, 0.4, 0.5);
  /** [m] defaultBaseHeight: the pelvis height a zero height change commands. */
  scalar_t defaultBaseHeight = 0.7;
};

/**
 * The command limits of `referenceFile`. Every key is required, as the MPC requires it: InvalidArgument naming the
 * file and the key when one is missing, is not a number or, for the four limits, is not positive; NotFound when the
 * file does not exist.
 */
absl::StatusOr<KeyboardCommandLimits> loadKeyboardCommandLimits(const std::string& referenceFile);

/**
 * The command of one typed line, "v_x v_y delta_height yaw_rate" separated by white space, of the PELVIS: words left
 * out are 0 and words past the fourth are ignored, as the ROS keyboard node read them. InvalidArgument naming the word
 * that is not a finite number (the ROS node ended on one).
 */
absl::StatusOr<vector4_t> parseKeyboardCommandLine(absl::string_view line);

/**
 * The operator/walking_velocity_command of `command`: clamped to +-limits, the velocities normalized by their limits,
 * the pelvis height defaultBaseHeight plus the clamped change.
 */
humanoid_mpc_msgs::WalkingVelocityCommand keyboardCommandToMessage(const vector4_t& command, const KeyboardCommandLimits& limits);

}  // namespace ocs2::humanoid::teleop
