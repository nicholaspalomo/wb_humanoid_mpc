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

#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "humanoid_state_estimator/StateEstimator.h"
#include "humanoid_state_estimator/StateEstimatorRegistry.h"
#include "robot_model/RobotDescription.h"

namespace ocs2::humanoid::validation {

/**
 * Returns the context the robot process builds its state estimator from
 * (RobotProcess::Create()): the URDF `urdfFile`, the MPC's contact frames and
 * their parent joints of `modelSettings` (as the robot binaries hand them to
 * the backend options), the robot `description`, which is retained in the
 * result and must outlive every estimator built from it, and the `kalman` noise
 * model of `settings`.
 */
estimation::StateEstimatorContext robotProcessStateEstimatorContext(
    const RobotProcessSettings& settings,
    const ModelSettings& modelSettings,
    const std::string& urdfFile,
    const robot::model::RobotDescription* absl_nonnull description);

/**
 * Builds the state estimator the task file's `state_estimator` names
 * (settings.stateEstimator) with robotProcessStateEstimatorContext(), as
 * RobotProcess::Create() builds it: `ground_truth` hands the measured state on
 * itself, `kalman` estimates the base's position and linear velocity. An
 * unknown name or an estimator that cannot be built is the registry's error,
 * prefixed with "<taskFile>: state_estimator: " as the robot process reports
 * it. `description` must outlive the estimator.
 */
absl::StatusOr<std::unique_ptr<estimation::StateEstimator>>
createRobotProcessStateEstimator(
    const RobotProcessSettings& settings,
    const ModelSettings& modelSettings,
    const std::string& urdfFile,
    absl::string_view taskFile,
    const robot::model::RobotDescription* absl_nonnull description);

}  // namespace ocs2::humanoid::validation
