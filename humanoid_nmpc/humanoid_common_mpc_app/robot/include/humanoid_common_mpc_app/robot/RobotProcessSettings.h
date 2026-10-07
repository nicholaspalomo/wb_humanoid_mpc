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

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact/ContactWrenchGate.h"
#include "humanoid_common_mpc_app/robot/RobotBackendRegistry.h"
#include "humanoid_common_mpc_app/robot/SimFallRecovery.h"
#include "mujoco_sim_interface/CheaterSimContactEstimator.h"

namespace ocs2::humanoid {

/** The feedforward torques of WB_MPC (`wb_mpc_feedforward`), the name that replaced `use_gravity_comp_feedforward`. */
enum class WbMpcFeedforward {
  /** The inverse dynamics of the planned state, input and contact wrenches (the default). */
  kInverseDynamics,
  /** Pure gravity compensation, to isolate a problem of the inverse dynamics (centroidal controller only). */
  kGravityCompensation,
};

// LINT.IfChange(wb_mpc_feedforward_names)
inline constexpr absl::string_view kInverseDynamicsFeedforwardName = "inverse_dynamics";
inline constexpr absl::string_view kGravityCompensationFeedforwardName = "gravity_compensation";
/** The names `wb_mpc_feedforward` accepts, in the order of WbMpcFeedforward. */
inline std::vector<std::string> wbMpcFeedforwardNames() {
  return {std::string(kInverseDynamicsFeedforwardName), std::string(kGravityCompensationFeedforwardName)};
}
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/README.md:robot_task_keys, //humanoid_nmpc/humanoid_mpc_config/task_file.proto:wb_mpc_feedforward)
// clang-format on

/**
 * What the robot process reads from the robot's task file (config/mpc/task.textproto) besides the MPC's own settings: the
 * fields the ROS sims (CentroidalMpcRobotSim, WBMpcRobotSim) read for their control loops. An absent field keeps the
 * default below, as in the sims, except the lists (robotProcessSettingsFromConfig()); a value the robot process cannot
 * run with is an InvalidArgument naming its field, where the sims logged a warning and dropped the whole block.
 */
struct RobotProcessSettings {
  /** `contact_estimator`: the controller's measured contact state, by ContactEstimatorRegistry name. */
  std::string contactEstimator = robot::mujoco_sim_interface::kCheaterSimContactEstimatorName;
  /** `sim_contact_force_threshold`, `sim_contact_timeline_window`, `sim_visualizations`, `gantry_hold`, `sim_projectile`. */
  SimulatorSettings simulator;
  /** `sim_max_base_tilt_angle`, `sim_gantry_catch_lift`. */
  SimFallRecovery::Config fallRecovery;
  /** `mpc_entry_blend_time` [s] (centroidal controller); nullopt: the controller's default. */
  std::optional<scalar_t> mpcEntryBlendTime;
  /** `safety_decay_time_constant` [s]; nullopt: the controller's default. */
  std::optional<scalar_t> safetyDecayTimeConstant;
  /** `contact_wrench_gate` (debounce_time, ramp_time); nullopt: the instantaneous gate. */
  std::optional<ContactWrenchGate::Config> contactWrenchGate;
  /** `wb_mpc_feedforward`. */
  WbMpcFeedforward wbMpcFeedforward = WbMpcFeedforward::kInverseDynamics;
  /** `telemetry_sinks`: the TelemetrySinkRegistry names the telemetry goes to; empty: no telemetry. */
  std::vector<std::string> telemetrySinks{"bus"};
  /** `telemetry_frequency` [Hz]; nullopt: min(100 Hz, the control rate). */
  std::optional<double> telemetryFrequency;
  /**
   * `mpc_link.policy_timeout` [s, robot clock]: the remote MPC link's link-loss timeout (RemoteMpcLink::Config). Longer than
   * one TCP retransmission with margin: Linux retransmits a lost segment after at least 200 ms (TCP_RTO_MIN), so a 0.2 s
   * timeout read every single lost packet on the bus as a lost link and swapped the MPC's action for the JOINT_PD hold,
   * also under a free-standing robot. 0.5 s still leaves at least half of every shipped policy horizon (1.0-1.2 s).
   */
  // LINT.IfChange(policy_timeout_default)
  scalar_t mpcLinkPolicyTimeout = 0.5;
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_ipc/include/humanoid_mpc_ipc/RemoteMpcLink.h:policy_timeout_default, //humanoid_nmpc/humanoid_common_mpc_app/robot/README.md:robot_task_keys, //humanoid_nmpc/humanoid_mpc_config/mpc_link_config.proto:policy_timeout_default)
  // clang-format on
};

/**
 * The robot process's settings of the task file `taskFile` (a robot's config/mpc/task.textproto, read strictly:
 * loadTaskFile(), then robotProcessSettingsFromConfig()). A retired field is refused by the parser with its replacement
 * (`enable_telemetry`, now telemetry_sinks; `use_gravity_comp_feedforward`, now wb_mpc_feedforward).
 *
 * @return The errors of loadTaskFile(), which name the file, the line and the column, and those of
 *         robotProcessSettingsFromConfig() prefixed with the file.
 */
absl::StatusOr<RobotProcessSettings> loadRobotProcessSettings(const std::string& taskFile);

/** The name of `feedforward`, as `wb_mpc_feedforward` spells it. */
absl::string_view wbMpcFeedforwardName(WbMpcFeedforward feedforward);

}  // namespace ocs2::humanoid
