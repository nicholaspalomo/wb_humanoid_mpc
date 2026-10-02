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

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid {

namespace {

/** A section or an entry of the document that may be absent or empty, or a map, and nothing else. */
absl::Status checkMapOrEmpty(const YAML::Node& node, absl::string_view path) {
  if (!node || node.IsNull() || node.IsMap()) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat("[JointPdGains] ", path, " is not a map."));
}

/**
 * `gains[key]` as a finite non-negative number, or `fallback` when `gains` (a map, or absent or empty) does not carry
 * the key.
 */
absl::StatusOr<scalar_t> loadGain(const YAML::Node& gains, absl::string_view key, absl::string_view path, scalar_t fallback) {
  if (!gains || !gains.IsMap()) {
    return fallback;
  }
  const YAML::Node value = gains[std::string(key)];
  if (!value) {
    return fallback;
  }
  scalar_t parsed = fallback;
  try {
    parsed = value.as<scalar_t>();
  } catch (const YAML::Exception& e) {
    return absl::InvalidArgumentError(
        absl::StrCat("[JointPdGains] ", path, ".", key, " is '", value.IsScalar() ? value.Scalar() : "", "', which is not a number."));
  }
  if (!std::isfinite(parsed) || parsed < 0.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("[JointPdGains] ", path, ".", key, " is ", parsed, "; a gain must be a finite non-negative number."));
  }
  return parsed;
}

/**
 * The gains of one map over `fallback`. The torque limit is read only when `fallback` has one: a controller that
 * commands no torque limit never reads the key, so a value it would not use cannot make it refuse a document.
 */
absl::StatusOr<JointPdGainsDefaults> loadGainSet(const YAML::Node& gains, absl::string_view path, const JointPdGainsDefaults& fallback) {
  RETURN_IF_ERROR(checkMapOrEmpty(gains, path));
  JointPdGainsDefaults loaded;
  ASSIGN_OR_RETURN(loaded.kp, loadGain(gains, "kp", path, fallback.kp));
  ASSIGN_OR_RETURN(loaded.kd, loadGain(gains, "kd", path, fallback.kd));
  loaded.torqueLimit = std::nullopt;
  if (fallback.torqueLimit.has_value()) {
    ASSIGN_OR_RETURN(loaded.torqueLimit, loadGain(gains, "torque_limit", path, *fallback.torqueLimit));
  }
  return loaded;
}

/** The torque limit of `gains` [N*m]: +infinity for a controller that commands none. */
scalar_t torqueLimitOf(const JointPdGainsDefaults& gains) {
  return gains.torqueLimit.value_or(std::numeric_limits<scalar_t>::infinity());
}

/** The gains of every joint for the named gains `named` over `defaults` (see parseJointPdGainsYaml()). */
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

absl::StatusOr<JointPdGains> parseJointPdGainsYaml(absl::string_view yamlText,
                                                   const JointPdGainsDefaults& defaults,
                                                   const std::vector<std::string>& mpcJointNames,
                                                   const std::vector<std::string>& otherJointNames) {
  YAML::Node root;
  try {
    root = YAML::Load(std::string(yamlText));
  } catch (const YAML::Exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("[JointPdGains] the document is not YAML: ", e.what()));
  }
  if (!root || root.IsNull()) {
    return absl::InvalidArgumentError("[JointPdGains] the document is empty.");
  }
  if (!root.IsMap()) {
    return absl::InvalidArgumentError("[JointPdGains] the document is not a map of default_gains and joint_gains.");
  }

  // Read through a const node: the non-const operator[] of yaml-cpp inserts the keys it looks up.
  const YAML::Node& document = root;
  ASSIGN_OR_RETURN(const JointPdGainsDefaults resolvedDefaults, loadGainSet(document["default_gains"], "default_gains", defaults));

  absl::flat_hash_map<std::string, JointPdGainsDefaults> named;
  const YAML::Node jointGains = document["joint_gains"];
  RETURN_IF_ERROR(checkMapOrEmpty(jointGains, "joint_gains"));
  if (jointGains && jointGains.IsMap()) {
    for (YAML::const_iterator entry = jointGains.begin(); entry != jointGains.end(); ++entry) {
      std::string jointName;
      try {
        jointName = entry->first.as<std::string>();
      } catch (const YAML::Exception& e) {
        return absl::InvalidArgumentError(absl::StrCat("[JointPdGains] a key of joint_gains is not a joint name: ", e.what()));
      }
      ASSIGN_OR_RETURN(named[jointName], loadGainSet(entry->second, absl::StrCat("joint_gains.", jointName), resolvedDefaults));
    }
  }
  return resolveJointPdGains(resolvedDefaults, named, mpcJointNames, otherJointNames);
}

absl::StatusOr<std::string> readJointPdGainsFile(const std::string& file) {
  std::ifstream stream(file);
  if (!stream.is_open()) {
    return absl::NotFoundError(absl::StrCat("[JointPdGains] cannot open ", file, "."));
  }
  std::ostringstream text;
  text << stream.rdbuf();
  return text.str();
}

std::filesystem::file_time_type jointPdGainsFileWriteTime(const std::string& file) {
  if (file.empty() || !std::filesystem::exists(file)) {
    return std::filesystem::file_time_type();
  }
  std::error_code error;
  return std::filesystem::last_write_time(file, error);
}

}  // namespace ocs2::humanoid
