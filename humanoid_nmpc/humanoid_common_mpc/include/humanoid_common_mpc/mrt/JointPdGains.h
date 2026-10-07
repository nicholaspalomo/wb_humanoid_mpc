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

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/** The gains a joint gets when its gains file (joint_pd_gains.textproto) does not set its own. */
struct JointPdGainsDefaults {
  scalar_t kp = 0.0;  ///< [N*m/rad]
  scalar_t kd = 0.0;  ///< [N*m*s/rad]
  /// [N*m] Empty for a controller that commands no torque limit (WBMpcMrtJointController): the file's
  /// `torque_limit` fields are then not read at all, and every joint's limit is +infinity.
  std::optional<scalar_t> torqueLimit = 0.0;
};

/**
 * The joint PD gains of an MRT joint controller: one entry per MPC joint (in the order of
 * ModelSettings::mpcModelJointNames) and one per other joint (ModelSettings::fixedJointNames), and the defaults the
 * document they came from resolved to.
 */
struct JointPdGains {
  vector_t mpcJointKp;
  vector_t mpcJointKd;
  vector_t mpcJointTorqueLimit;  ///< +infinity where JointPdGainsDefaults::torqueLimit is empty
  vector_t otherJointKp;
  vector_t otherJointKd;
  vector_t otherJointTorqueLimit;  ///< +infinity where JointPdGainsDefaults::torqueLimit is empty
  /// `default_gains` of the document, over the controller's own defaults.
  JointPdGainsDefaults defaults;

  /** Whether every vector has the size of `numMpcJoints` MPC joints and `numOtherJoints` other joints. */
  bool hasDimensions(size_t numMpcJoints, size_t numOtherJoints) const;
};

/** A joint outside the MPC model that the document does not name gets this fraction of the default kp and kd. */
inline constexpr scalar_t kOtherJointDefaultGainScale = 0.3;

/**
 * The gains when there is no document: `defaults` for every MPC joint, and for every other joint
 * kOtherJointDefaultGainScale times the default kp and kd with the default torque limit.
 */
JointPdGains defaultJointPdGains(const JointPdGainsDefaults& defaults,
                                 const std::vector<std::string>& mpcJointNames,
                                 const std::vector<std::string>& otherJointNames);

/**
 * The joint PD gains of the gains file `file` - a robot's config/controller/joint_pd_gains.textproto, a
 * humanoid_mpc_config.JointPdGainsFile read strictly - resolved over `defaults` as jointPdGainsFromConfig() resolves
 * them; nullopt when `file` is empty or names no file. Reads the file and allocates: for a thread other than the
 * realtime one.
 *
 * @return The error of reading or parsing the file, which names it, the line and the column, and InvalidArgument naming
 *         the file when jointPdGainsFromConfig() refuses its gains.
 */
absl::StatusOr<std::optional<JointPdGains>> loadJointPdGains(const std::string& file,
                                                             const JointPdGainsDefaults& defaults,
                                                             const std::vector<std::string>& mpcJointNames,
                                                             const std::vector<std::string>& otherJointNames);

/**
 * The modification time of the gains file `file`, for a watcher to compare
 * with later: the epoch when `file` is empty or names no file, file_time_type::min() when it exists but cannot be
 * stat'ed. A watcher records it BEFORE it reads the file, so that a save landing between the two is seen as a change by
 * its next check rather than taken as read.
 */
std::filesystem::file_time_type jointPdGainsFileWriteTime(const std::string& file);

}  // namespace ocs2::humanoid
