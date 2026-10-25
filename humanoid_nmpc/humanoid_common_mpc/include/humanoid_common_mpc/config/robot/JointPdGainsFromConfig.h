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
#include <vector>

#include "absl/status/statusor.h"

#include "humanoid_common_mpc/mrt/JointPdGains.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"

namespace ocs2::humanoid {

/**
 * The joint PD gains of an MRT joint controller from a joint PD gains file
 * (config/controller/joint_pd_gains.textproto, or the GUI's live gains on
 * operator/pd_gains), every gain from the file - a controller has none of its
 * own:
 *
 *   - default_gains, which sets kp, kd and torque_limit;
 *   - an MPC joint (`mpcJointNames`, in the order of
 * ModelSettings::mpcModelJointNames) the file names gets its own gains, each
 * over default_gains; one it does not name gets default_gains;
 *   - another joint (`otherJointNames`, ModelSettings::fixedJointNames) the
 * file names gets its own gains; one it does not name gets
 * kOtherJointDefaultGainScale times the default kp and kd, and the default
 * torque limit;
 *   - the controller constants gravity_comp_damping_fraction,
 * max_policy_joint_error and diverged_policy_joint_target_bound, at the
 * schema's defaults when the file leaves them out.
 *
 * A file that names a joint of neither list is another robot's (or a typo): it
 * is refused, so that another robot's default_gains never reach this robot's
 * joints.
 *
 * Does no I/O and keeps no state; allocates, so it runs on the communication
 * thread, never the realtime one.
 *
 * @return InvalidArgument naming the field when default_gains leaves out kp, kd
 * or torque_limit, when a gain is not a finite non-negative number, when
 * gravity_comp_damping_fraction is not in [0, 1], max_policy_joint_error not a
 * finite positive number or diverged_policy_joint_target_bound not a finite
 * non-negative one, when an entry of joint_gains names no joint, when two
 * entries name the same joint (a joint is named at most once), or when entries
 * name joints this robot does not have (listing them, in the file's order).
 */
absl::StatusOr<JointPdGains> jointPdGainsFromConfig(
    const mpc_config::JointPdGainsFile& file,
    const std::vector<std::string>& mpcJointNames,
    const std::vector<std::string>& otherJointNames);

}  // namespace ocs2::humanoid
