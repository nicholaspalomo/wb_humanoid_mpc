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

#include <optional>

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The command limits and the command shaping of a robot's reference file (config/command/reference.textproto, the
 * ReferenceFile of humanoid_mpc_config): what the target trajectories calculators, the procedural motion manager, the
 * keyboard teleoperation and the validation driver build the operator's commands with. referenceSettingsFromConfig()
 * (ReferenceFromConfig.h) makes it from the file and checks it. The default posture's joint positions, which need the
 * model's joint order, come from defaultJointStateFromConfig(), and the gait schedule's start from GaitFromConfig.h.
 *
 * Each member has the meaning and the unit of the reference file's field of the same name (reference_file.proto). The
 * ones the file must give are 0 here; the others hold the field's default.
 */
struct ReferenceSettings {
  /** [m/s] The speed a base pose command travels at. */
  scalar_t targetDisplacementVelocity = 0.0;
  /** [rad/s] The yaw rate a base pose command turns at. */
  scalar_t targetRotationVelocity = 0.0;
  /** [m/s] The forward velocity of a full stick deflection. */
  scalar_t maxDisplacementVelocityX = 0.0;
  /** [m/s] The sideways velocity of a full stick deflection. */
  scalar_t maxDisplacementVelocityY = 0.0;
  /** [m] The largest change of the base height a command may ask for, either way of defaultBaseHeight. */
  scalar_t maxDeltaPelvisHeight = 0.0;
  /** [rad/s] The yaw rate of a full stick deflection. */
  scalar_t maxRotationVelocity = 0.0;
  /** [m/s^2] The acceleration limit of the velocity reference in the (x, y) plane; 0: no ramp. */
  scalar_t maxLinearAcceleration = 0.0;
  /** [rad/s^2] The acceleration limit of the yaw rate reference; 0: no ramp. */
  scalar_t maxAngularAcceleration = 0.0;
  /** [Hz] The break frequency of the low-pass filter on the velocity command; 0: no filter. */
  scalar_t velocityCommandFilterBreakFrequency = 0.0;
  /**
   * [s] The time constant with which the centroidal MPC's joint reference approaches the target joint state, which
   * that MPC requires; absent when the file has none (the whole-body MPC does not read it).
   */
  std::optional<scalar_t> targetJointStateInterpolationTimeConstant;
  /** [m] The base height above the ground of the default posture: what a command without a pelvis height asks for. */
  scalar_t defaultBaseHeight = 0.0;
};

}  // namespace ocs2::humanoid
