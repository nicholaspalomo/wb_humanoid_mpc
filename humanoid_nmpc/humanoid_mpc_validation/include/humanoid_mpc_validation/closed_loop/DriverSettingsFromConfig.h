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

#include "Eigen/Core"
#include "absl/status/statusor.h"

#include "humanoid_mpc_config/reference_file.nproto.h"

namespace ocs2::humanoid::validation {

/**
 * What the base-controller GUI scales its velocity command with: the sticks it sends are the physical command over the
 * limits of a full deflection, and its pelvis height slider starts at the height of the default posture. The closed-loop
 * driver reads it from the reference file whose limits the MPC node's motion manager scales a received message back by
 * (guiCommandScalingFromConfig()). A value type.
 */
struct GuiCommandScaling {
  /**
   * [m/s, m/s, rad/s] max_displacement_velocity_x, max_displacement_velocity_y and max_rotation_velocity of the
   * reference file: the forward, sideways and yaw velocity of a full stick deflection. Positive.
   */
  Eigen::Vector3d commandLimits = Eigen::Vector3d::Ones();
  /** [m] default_base_height of the reference file: the pelvis height the GUI's slider sends. */
  double defaultPelvisHeight = 0.0;
};

/**
 * The GUI's command scaling of a robot's reference file, through referenceSettingsFromConfig(), which refuses a file
 * that leaves out a limit or the default base height or gives a value that is not finite.
 *
 * @param file The reference file (loadReferenceFile()).
 * @return The scaling; InvalidArgument naming the field that is absent, not finite or, for a command limit, not positive.
 */
absl::StatusOr<GuiCommandScaling> guiCommandScalingFromConfig(const mpc_config::ReferenceFile& file);

}  // namespace ocs2::humanoid::validation
