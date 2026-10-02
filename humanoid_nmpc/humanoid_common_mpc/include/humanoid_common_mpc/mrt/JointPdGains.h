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

/** The gains a joint gets when a joint_pd_gains.yaml document does not set its own. */
struct JointPdGainsDefaults {
  scalar_t kp = 0.0;  ///< [N*m/rad]
  scalar_t kd = 0.0;  ///< [N*m*s/rad]
  /// [N*m] Empty for a controller that commands no torque limit (WBMpcMrtJointController): the document's
  /// `torque_limit` keys are then not read at all, and every joint's limit is +infinity.
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
 * Parses a joint_pd_gains.yaml document (the file under a robot's config/controller/, or the same text the tuning GUI
 * publishes):
 *
 *   default_gains: {kp: 250.0, kd: 15.0, torque_limit: 500.0}   # each key optional, over `defaults`
 *   joint_gains:
 *     <joint name>: {kp: ..., kd: ..., torque_limit: ...}      # each key optional, over default_gains
 *
 * A joint the document names gets its own gains. An MPC joint it does not name gets default_gains; any other joint gets
 * kOtherJointDefaultGainScale times the default kp and kd, and the default torque limit. Entries for joints the
 * controller does not command are ignored, and so are the `torque_limit` keys when `defaults` has no torque limit (a
 * controller that commands none). Does no I/O and keeps no state, so it may run on any thread.
 *
 * @return InvalidArgument, naming the key where there is one, when the text is not YAML, holds no document, is not a
 *         map, or carries a `default_gains` or `joint_gains` section or a joint entry that is not a map, or a gain it
 *         reads that is not a finite non-negative number.
 */
absl::StatusOr<JointPdGains> parseJointPdGainsYaml(absl::string_view yamlText,
                                                   const JointPdGainsDefaults& defaults,
                                                   const std::vector<std::string>& mpcJointNames,
                                                   const std::vector<std::string>& otherJointNames);

/** The whole text of `file`; NotFound when it cannot be opened. For a caller that parses it with the function above. */
absl::StatusOr<std::string> readJointPdGainsFile(const std::string& file);

/**
 * The modification time of the gains file `file`, for a watcher to compare with later: the epoch when `file` is empty
 * or does not exist, file_time_type::min() when it exists but cannot be stat'ed. A watcher records it BEFORE it reads the
 * file, so that a save landing between the two is seen as a change by its next check rather than taken as read.
 */
std::filesystem::file_time_type jointPdGainsFileWriteTime(const std::string& file);

}  // namespace ocs2::humanoid
