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

/** The gains of one joint, or the `default_gains` of a gains file
 * (joint_pd_gains.textproto), which sets all three. */
struct JointPdGainsDefaults {
  scalar_t kp = 0.0;  ///< [N*m/rad]
  scalar_t kd = 0.0;  ///< [N*m*s/rad]
  /// [N*m] The bound of the feedforward torque the controller commands the
  /// joint, |ff| <= maxJointTorque, in every mode.
  scalar_t maxJointTorque = 0.0;
};

/**
 * The joint PD gains of an MRT joint controller, all from its gains file: one
 * entry per MPC joint (in the order of ModelSettings::mpcModelJointNames) and
 * one per other joint (ModelSettings::fixedJointNames), the file's
 * default_gains, and the controller constants the file carries with them
 * (JointPdGainsFile's fields 3 to 5).
 */
struct JointPdGains {
  vector_t mpcJointKp;
  vector_t mpcJointKd;
  vector_t mpcJointTorqueLimit;  ///< [N*m] the bound of each MPC joint's
                                 ///< feedforward torque
  vector_t otherJointKp;
  vector_t otherJointKd;
  vector_t otherJointTorqueLimit;  ///< [N*m] the bound of each other joint's
                                   ///< feedforward torque
  /// `default_gains` of the document.
  JointPdGainsDefaults defaults;
  /// [-] The fraction of its kd an MPC joint keeps in GRAVITY_COMP, in [0, 1]
  /// (gravity_comp_damping_fraction).
  scalar_t gravityCompDampingFraction = 0.0;
  /// [rad] Joint targets further than this from the measured joints mark a
  /// diverged policy (max_policy_joint_error).
  scalar_t maxPolicyJointError = 0.0;
  /// [rad] How far from the measured joints a diverged policy's targets are
  /// clamped (diverged_policy_joint_target_bound).
  scalar_t divergedPolicyJointTargetBound = 0.0;

  /** Whether every vector has the size of `numMpcJoints` MPC joints and
   * `numOtherJoints` other joints. */
  bool hasDimensions(size_t numMpcJoints, size_t numOtherJoints) const;
};

/** A joint outside the MPC model that the document does not name gets this
 * fraction of the default kp and kd. */
inline constexpr scalar_t kOtherJointDefaultGainScale = 0.3;

/** The joint groups of an MRT joint controller's gains and passive actions: the
 * MPC joints and the other joints. */
inline constexpr int kNumMrtJointGroups = 2;

/**
 * The gains of a document that names no joint: `defaults` for every MPC joint,
 * and for every other joint kOtherJointDefaultGainScale times the default kp
 * and kd with the default torque limit; the controller constants at the
 * schema's defaults (JointPdGainsFile::kDefault...).
 */
JointPdGains defaultJointPdGains(
    const JointPdGainsDefaults& defaults,
    const std::vector<std::string>& mpcJointNames,
    const std::vector<std::string>& otherJointNames);

/**
 * The joint PD gains of the gains file `file` - a robot's
 * config/controller/joint_pd_gains.textproto, a
 * humanoid_mpc_config.JointPdGainsFile read strictly - as
 * jointPdGainsFromConfig() resolves them; nullopt when `file` is empty or names
 * no file (for a watcher, whose editor may save by renaming; a controller
 * refuses to start on it). Reads the file and allocates: for a thread other
 * than the realtime one.
 *
 * @return The error of reading or parsing the file, which names it, the line
 * and the column, and InvalidArgument naming the file when
 * jointPdGainsFromConfig() refuses its gains.
 */
absl::StatusOr<std::optional<JointPdGains>> loadJointPdGains(
    absl::string_view file,
    const std::vector<std::string>& mpcJointNames,
    const std::vector<std::string>& otherJointNames);

/**
 * The gains a controller starts on: those of `file` (loadJointPdGains()).
 * InvalidArgument when `file` is empty and NotFound when it names no file - the
 * controllers have no gains of their own, and the hard-coded ones they used to
 * fall back to (500 N*m on every joint) were nobody's choice for any robot -
 * and loadJointPdGains()'s errors, each naming the file and prefixed with
 * `controller`.
 */
absl::StatusOr<JointPdGains> loadRequiredJointPdGains(
    absl::string_view file,
    const std::vector<std::string>& mpcJointNames,
    const std::vector<std::string>& otherJointNames,
    absl::string_view controller);

/**
 * The modification time of the gains file `file`, for a watcher to compare
 * with later: the epoch when `file` is empty or names no file,
 * file_time_type::min() when it exists but cannot be stat'ed. A watcher records
 * it BEFORE it reads the file, so that a save landing between the two is seen
 * as a change by its next check rather than taken as read.
 */
std::filesystem::file_time_type jointPdGainsFileWriteTime(
    absl::string_view file);

}  // namespace ocs2::humanoid
