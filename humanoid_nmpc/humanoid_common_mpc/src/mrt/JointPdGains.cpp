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

#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/robot/JointPdGainsFromConfig.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"

namespace ocs2::humanoid {

bool JointPdGains::hasDimensions(size_t numMpcJoints,
                                 size_t numOtherJoints) const {
  const Eigen::Index mpc = static_cast<Eigen::Index>(numMpcJoints);
  const Eigen::Index other = static_cast<Eigen::Index>(numOtherJoints);
  return mpcJointKp.size() == mpc && mpcJointKd.size() == mpc &&
         mpcJointTorqueLimit.size() == mpc && otherJointKp.size() == other &&
         otherJointKd.size() == other && otherJointTorqueLimit.size() == other;
}

JointPdGains defaultJointPdGains(
    const JointPdGainsDefaults& defaults,
    const std::vector<std::string>& mpcJointNames,
    const std::vector<std::string>& otherJointNames) {
  JointPdGains gains;
  gains.defaults = defaults;
  gains.mpcJointKp = vector_t::Constant(
      static_cast<Eigen::Index>(mpcJointNames.size()), defaults.kp);
  gains.mpcJointKd = vector_t::Constant(
      static_cast<Eigen::Index>(mpcJointNames.size()), defaults.kd);
  gains.mpcJointTorqueLimit = vector_t::Constant(
      static_cast<Eigen::Index>(mpcJointNames.size()), defaults.torqueLimit);
  gains.otherJointKp =
      vector_t::Constant(static_cast<Eigen::Index>(otherJointNames.size()),
                         defaults.kp * kOtherJointDefaultGainScale);
  gains.otherJointKd =
      vector_t::Constant(static_cast<Eigen::Index>(otherJointNames.size()),
                         defaults.kd * kOtherJointDefaultGainScale);
  gains.otherJointTorqueLimit = vector_t::Constant(
      static_cast<Eigen::Index>(otherJointNames.size()), defaults.torqueLimit);
  gains.gravityCompDampingFraction =
      mpc_config::JointPdGainsFile::kDefaultGravityCompDampingFraction;
  gains.maxPolicyJointError =
      mpc_config::JointPdGainsFile::kDefaultMaxPolicyJointError;
  gains.divergedPolicyJointTargetBound =
      mpc_config::JointPdGainsFile::kDefaultDivergedPolicyJointTargetBound;
  return gains;
}

absl::StatusOr<std::optional<JointPdGains>> loadJointPdGains(
    absl::string_view file,
    const std::vector<std::string>& mpcJointNames,
    const std::vector<std::string>& otherJointNames) {
  std::error_code error;
  if (file.empty() ||
      !std::filesystem::exists(std::filesystem::path(file), error)) {
    return std::nullopt;
  }
  // The parser's errors name the file, the line and the column already.
  ASSIGN_OR_RETURN(const mpc_config::JointPdGainsFile typed,
                   loadJointPdGainsFile(file));
  absl::StatusOr<JointPdGains> gains =
      jointPdGainsFromConfig(typed, mpcJointNames, otherJointNames);
  if (!gains.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat(file, ": ", gains.status().message()));
  }
  return std::optional<JointPdGains>(*std::move(gains));
}

absl::StatusOr<JointPdGains> loadRequiredJointPdGains(
    absl::string_view file,
    const std::vector<std::string>& mpcJointNames,
    const std::vector<std::string>& otherJointNames,
    absl::string_view controller) {
  if (file.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("[", controller,
                     "] no joint PD gains file was given: the controller has "
                     "no gains of its own and does not start."));
  }
  absl::StatusOr<std::optional<JointPdGains>> loaded =
      loadJointPdGains(file, mpcJointNames, otherJointNames);
  if (!loaded.ok()) {
    return absl::Status(
        loaded.status().code(),
        absl::StrCat("[", controller, "] The joint PD gains file ", file,
                     " was refused, so the controller does not start: ",
                     loaded.status().message()));
  }
  if (!loaded->has_value()) {
    return absl::NotFoundError(
        absl::StrCat("[", controller, "] the joint PD gains file ", file,
                     " does not exist: the controller has no gains of its own "
                     "and does not start."));
  }
  return **std::move(loaded);
}

std::filesystem::file_time_type jointPdGainsFileWriteTime(
    absl::string_view file) {
  // The std::error_code overloads, which do not throw (AGENTS.md,
  // <filesystem>).
  std::error_code error;
  if (file.empty() ||
      !std::filesystem::exists(std::filesystem::path(file), error)) {
    return std::filesystem::file_time_type();
  }
  return std::filesystem::last_write_time(std::filesystem::path(file), error);
}

}  // namespace ocs2::humanoid
