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

#include "legged_mpc_validation/closed_loop/LockstepStateEstimator.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/StatusMacros.h"
#include "legged_common_mpc_app/robot/RobotProcessSettings.h"
#include "legged_common_mpc_app/robot/RobotSetUpSteps.h"
#include "legged_mpc_config/contacts_config.nproto.h"
#include "legged_state_estimator/ContactKinematics.h"
#include "legged_state_estimator/StateEstimator.h"
#include "legged_state_estimator/StateEstimatorRegistry.h"
#include "robot_model/RobotDescription.h"

namespace ocs2::legged::validation {

RobotProcessSettings withStateEstimator(RobotProcessSettings settings,
                                        absl::string_view stateEstimator) {
  if (!stateEstimator.empty()) {
    settings.stateEstimator = std::string(stateEstimator);
  }
  return settings;
}

// LINT.IfChange(robot_process_state_estimator)
absl::StatusOr<estimation::StateEstimatorContext>
robotProcessStateEstimatorContext(
    const RobotProcessSettings& settings,
    const ModelSettings& modelSettings,
    const mpc_config::ContactsConfig& contacts,
    absl::string_view urdfFile,
    const robot::model::RobotDescription* absl_nonnull description) {
  ASSIGN_OR_RETURN(std::vector<estimation::ContactPointSpec> contactPoints,
                   stateEstimatorContacts(modelSettings, contacts, urdfFile));
  return estimation::StateEstimatorContext{
      .urdfFile = std::string(urdfFile),
      .contacts = std::move(contactPoints),
      .description = description,
      .kalman = settings.kalmanStateEstimator};
}

absl::StatusOr<std::unique_ptr<estimation::StateEstimator>>
createRobotProcessStateEstimator(
    const RobotProcessSettings& settings,
    const ModelSettings& modelSettings,
    const mpc_config::ContactsConfig& contacts,
    absl::string_view urdfFile,
    absl::string_view taskFile,
    const robot::model::RobotDescription* absl_nonnull description) {
  const absl::string_view source =
      taskFile.empty() ? absl::string_view("the task file") : taskFile;
  const absl::StatusOr<estimation::StateEstimatorContext> context =
      robotProcessStateEstimatorContext(settings, modelSettings, contacts,
                                        urdfFile, description);
  if (!context.ok()) {
    return absl::Status(context.status().code(),
                        absl::StrCat(source, ": state_estimator: ",
                                     context.status().message()));
  }
  absl::StatusOr<std::unique_ptr<estimation::StateEstimator>> estimator =
      estimation::StateEstimatorRegistry().create(settings.stateEstimator,
                                                  *context);
  if (!estimator.ok()) {
    return absl::Status(estimator.status().code(),
                        absl::StrCat(source, ": state_estimator: ",
                                     estimator.status().message()));
  }
  return estimator;
}
// clang-format off
// LINT.ThenChange(//legged_nmpc/legged_common_mpc_app/robot/src/RobotProcess.cpp:robot_process_state_estimator)
// clang-format on

}  // namespace ocs2::legged::validation
