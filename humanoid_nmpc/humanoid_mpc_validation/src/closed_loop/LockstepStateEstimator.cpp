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

#include "humanoid_mpc_validation/closed_loop/LockstepStateEstimator.h"

#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace ocs2::humanoid::validation {

// LINT.IfChange(robot_process_state_estimator)
estimation::StateEstimatorContext robotProcessStateEstimatorContext(
    const RobotProcessSettings& settings,
    const ModelSettings& modelSettings,
    const std::string& urdfFile,
    const robot::model::RobotDescription* absl_nonnull description) {
  return estimation::StateEstimatorContext{
      .urdfFile = urdfFile,
      .contactFrameNames = modelSettings.contactNames,
      .contactParentJointNames = modelSettings.contactParentJointNames,
      .description = description,
      .kalman = settings.kalmanStateEstimator};
}

absl::StatusOr<std::unique_ptr<estimation::StateEstimator>>
createRobotProcessStateEstimator(
    const RobotProcessSettings& settings,
    const ModelSettings& modelSettings,
    const std::string& urdfFile,
    absl::string_view taskFile,
    const robot::model::RobotDescription* absl_nonnull description) {
  absl::StatusOr<std::unique_ptr<estimation::StateEstimator>> estimator =
      estimation::StateEstimatorRegistry().create(
          settings.stateEstimator,
          robotProcessStateEstimatorContext(settings, modelSettings, urdfFile,
                                            description));
  if (!estimator.ok()) {
    return absl::Status(
        estimator.status().code(),
        absl::StrCat(taskFile.empty() ? "the task file" : taskFile,
                     ": state_estimator: ", estimator.status().message()));
  }
  return estimator;
}
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/src/RobotProcess.cpp:robot_process_state_estimator)
// clang-format on

}  // namespace ocs2::humanoid::validation
