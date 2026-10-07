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

#include "humanoid_common_mpc/mrt/JointPdGains.h"

#include <cmath>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/robot/JointPdGainsFromConfig.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"

namespace ocs2::humanoid {

namespace {

/** The torque limit of `gains` [N*m]: +infinity for a controller that commands none. */
scalar_t torqueLimitOf(const JointPdGainsDefaults& gains) {
  return gains.torqueLimit.value_or(std::numeric_limits<scalar_t>::infinity());
}

/** The gains of every joint for the named gains `named` over `defaults` (see jointPdGainsFromConfig()). */
JointPdGains resolveJointPdGains(const JointPdGainsDefaults& defaults,
                                 const absl::flat_hash_map<std::string, JointPdGainsDefaults>& named,
                                 const std::vector<std::string>& mpcJointNames,
                                 const std::vector<std::string>& otherJointNames) {
  JointPdGains gains;
  gains.defaults = defaults;
  gains.mpcJointKp.resize(mpcJointNames.size());
  gains.mpcJointKd.resize(mpcJointNames.size());
  gains.mpcJointTorqueLimit.resize(mpcJointNames.size());
  gains.otherJointKp.resize(otherJointNames.size());
  gains.otherJointKd.resize(otherJointNames.size());
  gains.otherJointTorqueLimit.resize(otherJointNames.size());

  for (size_t i = 0; i < mpcJointNames.size(); ++i) {
    const absl::flat_hash_map<std::string, JointPdGainsDefaults>::const_iterator it = named.find(mpcJointNames[i]);
    const JointPdGainsDefaults& joint = it != named.end() ? it->second : defaults;
    gains.mpcJointKp[i] = joint.kp;
    gains.mpcJointKd[i] = joint.kd;
    gains.mpcJointTorqueLimit[i] = torqueLimitOf(joint);
  }
  for (size_t i = 0; i < otherJointNames.size(); ++i) {
    const absl::flat_hash_map<std::string, JointPdGainsDefaults>::const_iterator it = named.find(otherJointNames[i]);
    if (it != named.end()) {
      gains.otherJointKp[i] = it->second.kp;
      gains.otherJointKd[i] = it->second.kd;
      gains.otherJointTorqueLimit[i] = torqueLimitOf(it->second);
    } else {
      gains.otherJointKp[i] = defaults.kp * kOtherJointDefaultGainScale;
      gains.otherJointKd[i] = defaults.kd * kOtherJointDefaultGainScale;
      gains.otherJointTorqueLimit[i] = torqueLimitOf(defaults);
    }
  }
  return gains;
}

}  // namespace

bool JointPdGains::hasDimensions(size_t numMpcJoints, size_t numOtherJoints) const {
  const Eigen::Index mpc = static_cast<Eigen::Index>(numMpcJoints);
  const Eigen::Index other = static_cast<Eigen::Index>(numOtherJoints);
  return mpcJointKp.size() == mpc && mpcJointKd.size() == mpc && mpcJointTorqueLimit.size() == mpc && otherJointKp.size() == other &&
         otherJointKd.size() == other && otherJointTorqueLimit.size() == other;
}

JointPdGains defaultJointPdGains(const JointPdGainsDefaults& defaults,
                                 const std::vector<std::string>& mpcJointNames,
                                 const std::vector<std::string>& otherJointNames) {
  return resolveJointPdGains(defaults, /*named=*/{}, mpcJointNames, otherJointNames);
}

absl::StatusOr<std::optional<JointPdGains>> loadJointPdGains(const std::string& file,
                                                             const JointPdGainsDefaults& defaults,
                                                             const std::vector<std::string>& mpcJointNames,
                                                             const std::vector<std::string>& otherJointNames) {
  std::error_code error;
  if (file.empty() || !std::filesystem::exists(file, error)) {
    return std::nullopt;
  }
  // The parser's errors name the file, the line and the column already.
  ASSIGN_OR_RETURN(const mpc_config::JointPdGainsFile typed, loadJointPdGainsFile(file));
  absl::StatusOr<JointPdGains> gains = jointPdGainsFromConfig(typed, defaults, mpcJointNames, otherJointNames);
  if (!gains.ok()) {
    return absl::InvalidArgumentError(absl::StrCat(file, ": ", gains.status().message()));
  }
  return std::optional<JointPdGains>(*std::move(gains));
}

std::filesystem::file_time_type jointPdGainsFileWriteTime(const std::string& file) {
  // The std::error_code overloads, which do not throw (AGENTS.md, <filesystem>).
  std::error_code error;
  if (file.empty() || !std::filesystem::exists(file, error)) {
    return std::filesystem::file_time_type();
  }
  return std::filesystem::last_write_time(file, error);
}

}  // namespace ocs2::humanoid
