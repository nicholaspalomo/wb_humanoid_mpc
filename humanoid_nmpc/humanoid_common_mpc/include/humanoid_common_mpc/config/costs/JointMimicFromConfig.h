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

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_mpc_config/mimic_joints_config.nproto.h"

namespace ocs2::humanoid {

/**
 * A mimic joint of the task file's mimic_joints block: q_child = multiplier * q_parent, held with the gains. The
 * centroidal MPC's JointMimicKinematicConstraint takes all but the velocity gain, the whole-body MPC's
 * JointMimicDynamicsConstraint all of it.
 */
struct JointMimicSettings {
  std::string parentJointName;
  std::string childJointName;
  scalar_t multiplier = 0.0;
  scalar_t positionGain = 0.0;
  scalar_t velocityGain = 0.0;
};

/**
 * The knee mimic joints of both legs, in the order of the contacts (left_knee, then right_knee), as the two MPC
 * interfaces read the blocks mimicJoints.left_knee and mimicJoints.right_knee: a joint name the block leaves out is
 * empty and a number 0, which the constraints' Create() then refuses by name.
 *
 * @return The settings; InvalidArgument naming the field when a multiplier or gain is not finite (inf or nan).
 */
absl::StatusOr<feet_array_t<JointMimicSettings>> kneeMimicJointsFromConfig(const mpc_config::MimicJointsConfig& mimicJoints);

/**
 * kneeMimicJointsFromConfig() for the centroidal MPC, whose JointMimicKinematicConstraint holds the position alone and
 * has no use for a velocity gain.
 *
 * @return The settings, velocity gains 0; InvalidArgument as kneeMimicJointsFromConfig(), and naming a velocity_gain
 *         that is not 0.
 */
absl::StatusOr<feet_array_t<JointMimicSettings>> kneeMimicKinematicJointsFromConfig(const mpc_config::MimicJointsConfig& mimicJoints);

}  // namespace ocs2::humanoid
