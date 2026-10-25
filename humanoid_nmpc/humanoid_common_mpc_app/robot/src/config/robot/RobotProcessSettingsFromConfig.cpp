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

#include "humanoid_common_mpc_app/robot/config/robot/RobotProcessSettingsFromConfig.h"

#include <cmath>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/robot/ControllerSideSettingsFromConfig.h"
#include "humanoid_common_mpc/contact/ContactWrenchGate.h"
#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {
namespace {

/** The feedforward wb_mpc_feedforward names; InvalidArgument listing the names
 * for any other. */
absl::StatusOr<WbMpcFeedforward> wbMpcFeedforwardFromName(
    absl::string_view name) {
  if (name == kInverseDynamicsFeedforwardName) {
    return WbMpcFeedforward::kInverseDynamics;
  }
  if (name == kGravityCompensationFeedforwardName) {
    return WbMpcFeedforward::kGravityCompensation;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("wb_mpc_feedforward is '", name, "'; it must be ",
                   kInverseDynamicsFeedforwardName, " or ",
                   kGravityCompensationFeedforwardName, "."));
}

/** 0 (off) for an absent `value` of the task file's `field`; the value when it
 * is finite and positive; InvalidArgument otherwise. */
absl::StatusOr<double> absentOrPositive(absl::string_view field,
                                        std::optional<double> value) {
  if (!value.has_value()) return 0.0;
  // Negated, so that NaN is refused too.
  if (!std::isfinite(*value) || !(*value > 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat(field, " is ", *value,
                     "; it must be positive, or absent to switch it off."));
  }
  return *value;
}

}  // namespace

estimation::KalmanBaseEstimator::Config kalmanBaseEstimatorConfigFromConfig(
    const mpc_config::KalmanStateEstimatorConfig& config) {
  // LINT.IfChange(kalman_state_estimator_fields)
  return estimation::KalmanBaseEstimator::Config{
      .imuAccelerationNoise = config.imu_acceleration_noise,
      .baseVelocityRandomWalk = config.base_velocity_random_walk,
      .basePositionRandomWalk = config.base_position_random_walk,
      .stanceContactRandomWalk = config.stance_contact_random_walk,
      .swingContactRandomWalk = config.swing_contact_random_walk,
      .kinematicPositionNoise = config.kinematic_position_noise,
      .kinematicVelocityNoise = config.kinematic_velocity_noise,
      .contactHeightNoise = config.contact_height_noise,
      .contactHeight = config.contact_height,
  };
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/kalman_state_estimator_config.proto:kalman_state_estimator_defaults)
}

absl::StatusOr<RobotProcessSettings> robotProcessSettingsFromConfig(
    const mpc_config::TaskFile& task) {
  RobotProcessSettings settings;
  // The controller-side settings through their one reader, which the mailbox
  // and the task file watcher use too; at start-up there is no running gate to
  // keep, so a block it refuses refuses the file.
  ControllerSideConfig controllerSide = controllerSideSettingsFromConfig(task);
  if (!controllerSide.problems.empty()) {
    return absl::InvalidArgumentError(
        absl::StrJoin(controllerSide.problems, "; "));
  }
  settings.contactEstimator = std::move(controllerSide.contactEstimator);
  settings.contactWrenchGate = controllerSide.contactWrenchGate;
  settings.simulator.contactForceThreshold = task.sim_contact_force_threshold;
  settings.simulator.contactTimelineWindow = task.sim_contact_timeline_window;
  settings.simulator.visualizations = task.sim_visualizations;
  settings.simulator.gantryHold = task.gantry_hold;
  settings.simulator.projectile = task.sim_projectile;
  settings.simulator.ragdollJointDamping = task.sim_ragdoll_joint_damping;
  settings.simulator.autoResetBaseHeight = task.sim_auto_reset_base_height;
  // The schema has no sentinel: an absent field is off, which the fall
  // recovery's config spells 0.
  ASSIGN_OR_RETURN(settings.fallRecovery.maxBaseTiltAngle,
                   absentOrPositive("sim_max_base_tilt_angle",
                                    task.sim_max_base_tilt_angle));
  ASSIGN_OR_RETURN(
      settings.fallRecovery.catchLift,
      absentOrPositive("sim_gantry_catch_lift", task.sim_gantry_catch_lift));
  settings.mpcEntryBlendTime = task.mpc_entry_blend_time;
  settings.safetyDecayTimeConstant = task.safety_decay_time_constant;
  const absl::StatusOr<WbMpcFeedforward> feedforward =
      wbMpcFeedforwardFromName(task.wb_mpc_feedforward);
  if (!feedforward.ok()) {
    return feedforward.status();
  }
  settings.wbMpcFeedforward = *feedforward;
  settings.telemetrySinks = task.telemetry_sinks;
  settings.telemetryFrequency = task.telemetry_frequency;
  settings.mpcLinkPolicyTimeout = task.mpc_link.policy_timeout;
  settings.stateEstimator = task.state_estimator;
  settings.kalmanStateEstimator =
      kalmanBaseEstimatorConfigFromConfig(task.kalman_state_estimator);

  if (settings.telemetryFrequency.has_value() &&
      !(*settings.telemetryFrequency > 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "telemetry_frequency must be positive [Hz], got ",
        *settings.telemetryFrequency,
        "; a file without telemetry_sinks turns the telemetry off."));
  }
  if (!(settings.mpcLinkPolicyTimeout > 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "mpc_link.policy_timeout must be a positive number of seconds, got ",
        settings.mpcLinkPolicyTimeout));
  }
  return settings;
}

}  // namespace ocs2::humanoid
