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

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include <humanoid_common_mpc/common/Types.h>
#include <mujoco_sim_interface/CheaterSimContactEstimator.h>

#include "humanoid_common_mpc/contact/ContactWrenchGate.h"
#include "humanoid_common_mpc_app/robot/RobotBackendRegistry.h"
#include "humanoid_common_mpc_app/robot/SimFallRecovery.h"

namespace ocs2::humanoid {

/** The feedforward torques of WB_MPC (`wbMpcFeedforward`), the name that replaced `useGravityCompFeedforward`. */
enum class WbMpcFeedforward {
  /** The inverse dynamics of the planned state, input and contact wrenches (the default). */
  kInverseDynamics,
  /** Pure gravity compensation, to isolate a problem of the inverse dynamics (centroidal controller only). */
  kGravityCompensation,
};

// LINT.IfChange(wb_mpc_feedforward_names)
inline constexpr absl::string_view kInverseDynamicsFeedforwardName = "inverse_dynamics";
inline constexpr absl::string_view kGravityCompensationFeedforwardName = "gravity_compensation";
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/README.md:robot_task_keys)

/**
 * What the robot process reads from the robot's task file (config/mpc/task.yaml) besides the MPC's own settings: the
 * keys the ROS sims (CentroidalMpcRobotSim, WBMpcRobotSim) read for their control loops. Absent keys keep the defaults
 * below, as in the sims; a key whose value is not of its type is an InvalidArgument naming it, where the sims logged a
 * warning and dropped the whole block.
 */
struct RobotProcessSettings {
  /** `contactEstimator`: the controller's measured contact state, by ContactEstimatorRegistry name. */
  std::string contactEstimator = robot::mujoco_sim_interface::kCheaterSimContactEstimatorName;
  /** `simContactForceThreshold`, `simContactTimelineWindow`, `simVisualizations`, `gantryHold`, `simProjectile`. */
  SimulatorSettings simulator;
  /** `simMaxBaseTiltAngle`, `simGantryCatchLift`. */
  SimFallRecovery::Config fallRecovery;
  /** `mpcEntryBlendTime` [s] (centroidal controller); nullopt: the controller's default. */
  std::optional<scalar_t> mpcEntryBlendTime;
  /** `safetyDecayTimeConstant` [s]; nullopt: the controller's default. */
  std::optional<scalar_t> safetyDecayTimeConstant;
  /** `contact_wrench_gate` (debounceTime, rampTime); nullopt: the instantaneous gate. */
  std::optional<ContactWrenchGate::Config> contactWrenchGate;
  /** `wbMpcFeedforward`. */
  WbMpcFeedforward wbMpcFeedforward = WbMpcFeedforward::kInverseDynamics;
  /** `telemetrySinks`: the TelemetrySinkRegistry names the telemetry goes to; empty: no telemetry. */
  std::vector<std::string> telemetrySinks{"bus"};
  /** `telemetryFrequency` [Hz] (or `telemetry_frequency`); nullopt: min(100 Hz, the control rate). */
  std::optional<double> telemetryFrequency;
  /**
   * `mpcLink.policyTimeout` [s, robot clock]: the remote MPC link's link-loss timeout (RemoteMpcLink::Config). Longer than
   * one TCP retransmission with margin: Linux retransmits a lost segment after at least 200 ms (TCP_RTO_MIN), so a 0.2 s
   * timeout read every single lost packet on the bus as a lost link and swapped the MPC's action for the JOINT_PD hold,
   * also under a free-standing robot. 0.5 s still leaves at least half of every shipped policy horizon (1.0-1.2 s).
   */
  // LINT.IfChange(policy_timeout_default)
  scalar_t mpcLinkPolicyTimeout = 0.5;
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_ipc/include/humanoid_mpc_ipc/RemoteMpcLink.h:policy_timeout_default, //humanoid_nmpc/humanoid_common_mpc_app/robot/README.md:robot_task_keys)
  // clang-format on
};

/**
 * The robot process's keys of `taskFile`. A retired key is a FailedPrecondition that names its replacement:
 * `enableTelemetry` / `enable_telemetry` (now `telemetrySinks`), `useGravityCompFeedforward` (now `wbMpcFeedforward`).
 */
absl::StatusOr<RobotProcessSettings> loadRobotProcessSettings(const std::string& taskFile);

/** The same keys of a YAML document (for tests). `source` names it in the errors. */
absl::StatusOr<RobotProcessSettings> parseRobotProcessSettings(absl::string_view yamlText, absl::string_view source);

/** The name of `feedforward`, as `wbMpcFeedforward` spells it. */
absl::string_view wbMpcFeedforwardName(WbMpcFeedforward feedforward);

}  // namespace ocs2::humanoid
