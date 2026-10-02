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

#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"

#include <exception>
#include <fstream>
#include <sstream>

#include <yaml-cpp/yaml.h>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid {
namespace {

/** `node[key]` into `value` when present; a value that is not a T is InvalidArgument naming the key. */
template <typename T>
absl::Status readOptional(const YAML::Node& node, absl::string_view source, const std::string& key, T& value) {
  const YAML::Node child = node[key];
  if (!child) {
    return absl::OkStatus();
  }
  try {
    value = child.as<T>();
  } catch (const std::exception& error) {
    return absl::InvalidArgumentError(absl::StrCat(source, ": ", key, " is not a value of the expected type: ", error.what()));
  }
  return absl::OkStatus();
}

template <typename T>
absl::Status readOptional(const YAML::Node& node, absl::string_view source, const std::string& key, std::optional<T>& value) {
  if (!node[key]) {
    return absl::OkStatus();
  }
  T parsed{};
  RETURN_IF_ERROR(readOptional(node, source, key, parsed));
  value = parsed;
  return absl::OkStatus();
}

absl::Status refuseRetiredKey(const YAML::Node& root, absl::string_view source, const std::string& key, absl::string_view replacement) {
  if (root[key]) {
    return absl::FailedPreconditionError(absl::StrCat(source, ": `", key, "` is retired; ", replacement));
  }
  return absl::OkStatus();
}

}  // namespace

absl::string_view wbMpcFeedforwardName(WbMpcFeedforward feedforward) {
  return feedforward == WbMpcFeedforward::kGravityCompensation ? kGravityCompensationFeedforwardName : kInverseDynamicsFeedforwardName;
}

absl::StatusOr<RobotProcessSettings> parseRobotProcessSettings(absl::string_view yamlText, absl::string_view source) {
  YAML::Node root;
  try {
    root = YAML::Load(std::string(yamlText));
  } catch (const std::exception& error) {
    return absl::InvalidArgumentError(absl::StrCat(source, " is not YAML: ", error.what()));
  }
  if (!root.IsMap()) {
    return absl::InvalidArgumentError(absl::StrCat(source, " is not a YAML map of task keys"));
  }

  // Booleans that switched a component on or off, replaced by names (the robot process refuses a stale file rather than
  // silently changing what it does).
  // LINT.IfChange(retired_robot_keys)
  RETURN_IF_ERROR(refuseRetiredKey(root, source, "enableTelemetry",
                                   "list the telemetry sinks by name instead: `telemetrySinks: [bus]` publishes robot/state, "
                                   "`telemetrySinks: []` turns the telemetry off."));
  RETURN_IF_ERROR(refuseRetiredKey(root, source, "enable_telemetry", "list the telemetry sinks by name instead: `telemetrySinks: [bus]`."));
  RETURN_IF_ERROR(refuseRetiredKey(root, source, "useGravityCompFeedforward",
                                   "name the feedforward of WB_MPC instead: `wbMpcFeedforward: inverse_dynamics` (the default) or "
                                   "`wbMpcFeedforward: gravity_compensation`."));
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/README.md:robot_task_keys)

  RobotProcessSettings settings;
  // LINT.IfChange(robot_task_keys)
  RETURN_IF_ERROR(readOptional(root, source, "contactEstimator", settings.contactEstimator));
  RETURN_IF_ERROR(readOptional(root, source, "simContactForceThreshold", settings.simulator.contactForceThreshold));
  RETURN_IF_ERROR(readOptional(root, source, "simContactTimelineWindow", settings.simulator.contactTimelineWindow));
  RETURN_IF_ERROR(readOptional(root, source, "simVisualizations", settings.simulator.visualizations));
  RETURN_IF_ERROR(readOptional(root, source, "gantryHold", settings.simulator.gantryHold));
  RETURN_IF_ERROR(readOptional(root, source, "simProjectile", settings.simulator.projectile));
  RETURN_IF_ERROR(readOptional(root, source, "mpcEntryBlendTime", settings.mpcEntryBlendTime));
  RETURN_IF_ERROR(readOptional(root, source, "safetyDecayTimeConstant", settings.safetyDecayTimeConstant));
  if (root["contact_wrench_gate"]) {
    ContactWrenchGate::Config gate;
    const YAML::Node block = root["contact_wrench_gate"];
    RETURN_IF_ERROR(readOptional(block, source, "debounceTime", gate.debounceTime));
    RETURN_IF_ERROR(readOptional(block, source, "rampTime", gate.rampTime));
    settings.contactWrenchGate = gate;
  }
  std::string feedforward(kInverseDynamicsFeedforwardName);
  RETURN_IF_ERROR(readOptional(root, source, "wbMpcFeedforward", feedforward));
  if (feedforward == kInverseDynamicsFeedforwardName) {
    settings.wbMpcFeedforward = WbMpcFeedforward::kInverseDynamics;
  } else if (feedforward == kGravityCompensationFeedforwardName) {
    settings.wbMpcFeedforward = WbMpcFeedforward::kGravityCompensation;
  } else {
    return absl::InvalidArgumentError(absl::StrCat(source, ": wbMpcFeedforward is '", feedforward, "'; it must be ",
                                                   kInverseDynamicsFeedforwardName, " or ", kGravityCompensationFeedforwardName, "."));
  }
  RETURN_IF_ERROR(readOptional(root, source, "telemetrySinks", settings.telemetrySinks));
  if (root["telemetryFrequency"]) {
    RETURN_IF_ERROR(readOptional(root, source, "telemetryFrequency", settings.telemetryFrequency));
  } else {
    RETURN_IF_ERROR(readOptional(root, source, "telemetry_frequency", settings.telemetryFrequency));
  }
  if (root["mpcLink"]) {
    RETURN_IF_ERROR(readOptional(root["mpcLink"], source, "policyTimeout", settings.mpcLinkPolicyTimeout));
  }
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/README.md:robot_task_keys)
  // clang-format on
  // LINT.IfChange(sim_fall_recovery_keys)
  RETURN_IF_ERROR(readOptional(root, source, "simMaxBaseTiltAngle", settings.fallRecovery.maxBaseTiltAngle));
  RETURN_IF_ERROR(readOptional(root, source, "simGantryCatchLift", settings.fallRecovery.catchLift));
  // clang-format off
  // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:sim_fall_recovery, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:sim_fall_recovery, //robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml:sim_fall_recovery, //robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml:sim_fall_recovery, //robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.yaml:sim_fall_recovery, //humanoid_nmpc/docs/mpc_reset/README.md:sim_fall_recovery_keys)
  // clang-format on

  if (settings.contactWrenchGate.has_value() &&
      (settings.contactWrenchGate->debounceTime < 0.0 || settings.contactWrenchGate->rampTime < 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat(source, ": contact_wrench_gate.debounceTime and rampTime must be non-negative"));
  }
  if (settings.telemetryFrequency.has_value() && !(*settings.telemetryFrequency > 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat(source, ": telemetryFrequency must be positive [Hz], got ", *settings.telemetryFrequency,
                                                   "; `telemetrySinks: []` turns the telemetry off."));
  }
  if (!(settings.mpcLinkPolicyTimeout > 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat(source, ": mpcLink.policyTimeout must be a positive number of seconds, got ", settings.mpcLinkPolicyTimeout));
  }
  return settings;
}

absl::StatusOr<RobotProcessSettings> loadRobotProcessSettings(const std::string& taskFile) {
  std::ifstream stream(taskFile);
  if (!stream.is_open()) {
    return absl::NotFoundError(absl::StrCat("cannot read the task file ", taskFile));
  }
  std::stringstream text;
  text << stream.rdbuf();
  return parseRobotProcessSettings(text.str(), taskFile);
}

}  // namespace ocs2::humanoid
