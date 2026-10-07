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

#include "humanoid_common_mpc/config/robot/JointPdGainsFromConfig.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/mrt/JointPdGains.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"

namespace ocs2::humanoid {
namespace {

/** `value` when the file sets it, else `fallback`; InvalidArgument naming `path` for a value that is not a gain. */
absl::StatusOr<scalar_t> gainOrFallback(std::optional<double> value, absl::string_view path, scalar_t fallback) {
  if (!value.has_value()) {
    return fallback;
  }
  if (!std::isfinite(*value) || *value < 0.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("[JointPdGains] ", path, " is ", *value, "; a gain must be a finite non-negative number."));
  }
  return *value;
}

/**
 * The gains kp, kd and torque limit over `fallback`. The torque limit is read only when `fallback` has one: a controller
 * that commands no torque limit never reads it, so a value it would not use cannot make it refuse a file.
 */
absl::StatusOr<JointPdGainsDefaults> gainSet(std::optional<double> kp,
                                             std::optional<double> kd,
                                             std::optional<double> torqueLimit,
                                             absl::string_view path,
                                             const JointPdGainsDefaults& fallback) {
  JointPdGainsDefaults gains;
  ASSIGN_OR_RETURN(gains.kp, gainOrFallback(kp, absl::StrCat(path, ".kp"), fallback.kp));
  ASSIGN_OR_RETURN(gains.kd, gainOrFallback(kd, absl::StrCat(path, ".kd"), fallback.kd));
  gains.torqueLimit = std::nullopt;
  if (fallback.torqueLimit.has_value()) {
    ASSIGN_OR_RETURN(gains.torqueLimit, gainOrFallback(torqueLimit, absl::StrCat(path, ".torque_limit"), *fallback.torqueLimit));
  }
  return gains;
}

/** The torque limit of `gains` [N*m]: +infinity for a controller that commands none. */
scalar_t torqueLimitOf(const JointPdGainsDefaults& gains) {
  return gains.torqueLimit.value_or(std::numeric_limits<scalar_t>::infinity());
}

}  // namespace

absl::StatusOr<JointPdGains> jointPdGainsFromConfig(const mpc_config::JointPdGainsFile& file,
                                                    const JointPdGainsDefaults& defaults,
                                                    const std::vector<std::string>& mpcJointNames,
                                                    const std::vector<std::string>& otherJointNames) {
  const mpc_config::JointPdGainsFile::Gains& fileDefaults = file.default_gains;
  ASSIGN_OR_RETURN(const JointPdGainsDefaults resolvedDefaults,
                   gainSet(fileDefaults.kp, fileDefaults.kd, fileDefaults.torque_limit, /*path=*/"default_gains", defaults));

  absl::flat_hash_set<absl::string_view> robotJoints(mpcJointNames.begin(), mpcJointNames.end());
  robotJoints.insert(otherJointNames.begin(), otherJointNames.end());
  std::vector<std::string> notThisRobots;
  absl::flat_hash_map<std::string, JointPdGainsDefaults> named;
  for (size_t i = 0; i < file.joint_gains.size(); ++i) {
    const mpc_config::JointPdGainsFile::JointGains& entry = file.joint_gains[i];
    if (entry.joint.empty()) {
      return absl::InvalidArgumentError(absl::StrCat("[JointPdGains] joint_gains[", i, "] names no joint."));
    }
    const std::string path = absl::StrCat("joint_gains[joint=", entry.joint, "]");
    ASSIGN_OR_RETURN(const JointPdGainsDefaults gains, gainSet(entry.kp, entry.kd, entry.torque_limit, path, resolvedDefaults));
    if (!named.emplace(entry.joint, gains).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("[JointPdGains] joint_gains names the joint ", entry.joint, " twice; a joint is named at most once."));
    }
    if (!robotJoints.contains(entry.joint)) notThisRobots.push_back(entry.joint);
  }
  if (!notThisRobots.empty()) {
    return absl::InvalidArgumentError(absl::StrCat("[JointPdGains] joint_gains names ", absl::StrJoin(notThisRobots, ", "),
                                                   ", which this robot does not have: the gains file is another robot's"));
  }

  // The gains of every joint the file does not name, then the named ones over them.
  JointPdGains gains = defaultJointPdGains(resolvedDefaults, mpcJointNames, otherJointNames);
  for (size_t i = 0; i < mpcJointNames.size(); ++i) {
    const absl::flat_hash_map<std::string, JointPdGainsDefaults>::const_iterator it = named.find(mpcJointNames[i]);
    if (it == named.end()) continue;
    const Eigen::Index index = static_cast<Eigen::Index>(i);
    gains.mpcJointKp[index] = it->second.kp;
    gains.mpcJointKd[index] = it->second.kd;
    gains.mpcJointTorqueLimit[index] = torqueLimitOf(it->second);
  }
  for (size_t i = 0; i < otherJointNames.size(); ++i) {
    const absl::flat_hash_map<std::string, JointPdGainsDefaults>::const_iterator it = named.find(otherJointNames[i]);
    if (it == named.end()) continue;
    const Eigen::Index index = static_cast<Eigen::Index>(i);
    gains.otherJointKp[index] = it->second.kp;
    gains.otherJointKd[index] = it->second.kd;
    gains.otherJointTorqueLimit[index] = torqueLimitOf(it->second);
  }
  return gains;
}

}  // namespace ocs2::humanoid
