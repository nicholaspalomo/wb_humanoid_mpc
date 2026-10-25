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

#include "humanoid_state_estimator/StateEstimatorRegistry.h"

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_state_estimator/ContactKinematics.h"
#include "humanoid_state_estimator/GroundTruthStateEstimator.h"
#include "humanoid_state_estimator/KalmanStateEstimator.h"

namespace ocs2::humanoid::estimation {
namespace {

// LINT.IfChange(state_estimator_names)
/** The names, in registration order. */
constexpr std::array<absl::string_view, 2> kNames = {
    kGroundTruthStateEstimatorName, kKalmanStateEstimatorName};
/** What each of kNames is, for availableNames(). */
constexpr std::array<absl::string_view, 2> kDescriptions = {
    "the backend's state",
    "the IMU and the leg kinematics through a Kalman filter"};
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto:state_estimator, //humanoid_nmpc/humanoid_mpc_config/config_registries.textproto)
// clang-format on

absl::StatusOr<std::unique_ptr<StateEstimator>> createKalman(
    const StateEstimatorContext& context) {
  std::vector<ContactPointSpec> contacts;
  for (size_t contact = 0; contact < context.contactFrameNames.size();
       ++contact) {
    ContactPointSpec spec;
    spec.frameNames.push_back(context.contactFrameNames[contact]);
    if (contact < context.contactParentJointNames.size())
      spec.frameNames.push_back(context.contactParentJointNames[contact]);
    contacts.push_back(std::move(spec));
  }
  ASSIGN_OR_RETURN(
      std::unique_ptr<ContactKinematics> kinematics,
      ContactKinematics::Create(context.urdfFile, contacts,
                                context.description->getJointNames()));
  ASSIGN_OR_RETURN(
      std::unique_ptr<KalmanBaseEstimator> base,
      KalmanBaseEstimator::Create(context.kalman, std::move(kinematics)));
  ASSIGN_OR_RETURN(
      std::unique_ptr<KalmanStateEstimator> estimator,
      KalmanStateEstimator::Create(*context.description, std::move(base)));
  return std::unique_ptr<StateEstimator>(std::move(estimator));
}

}  // namespace

std::vector<std::string> StateEstimatorRegistry::names() const {
  std::vector<std::string> names;
  for (const absl::string_view name : kNames) names.emplace_back(name);
  return names;
}

std::string StateEstimatorRegistry::availableNames() const {
  std::string names;
  for (size_t index = 0; index < kNames.size(); ++index) {
    absl::StrAppend(&names, names.empty() ? "" : ", ", kNames[index], " (",
                    kDescriptions[index], ")");
  }
  return names;
}

bool StateEstimatorRegistry::has(absl::string_view name) const {
  for (const absl::string_view known : kNames) {
    if (known == name) return true;
  }
  return false;
}

absl::StatusOr<std::unique_ptr<StateEstimator>> StateEstimatorRegistry::create(
    absl::string_view name, const StateEstimatorContext& context) const {
  if (name == kGroundTruthStateEstimatorName)
    return std::unique_ptr<StateEstimator>(
        std::make_unique<GroundTruthStateEstimator>());
  if (name == kKalmanStateEstimatorName) return createKalman(context);
  return absl::InvalidArgumentError(
      absl::StrCat("There is no state estimator '", name,
                   "' (state_estimator). Available: ", availableNames(), "."));
}

}  // namespace ocs2::humanoid::estimation
