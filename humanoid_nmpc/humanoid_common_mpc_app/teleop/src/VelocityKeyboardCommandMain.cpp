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

#include <unistd.h>

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc_app/node/MpcFiles.h"
#include "humanoid_common_mpc_app/node/NodeBus.h"
#include "humanoid_common_mpc_app/node/ShutdownSignal.h"
#include "humanoid_common_mpc_app/teleop/KeyboardVelocityCommand.h"
#include "humanoid_common_mpc_app/teleop/LineReader.h"
#include "humanoid_common_mpc_app/teleop/VelocityCommandRepeater.h"
#include "robot_ipc/Bus.h"

// LINT.IfChange(teleop_flags)
ABSL_FLAG(std::string,
          reference_file,
          "",
          "The robot's reference file (config/command/reference.textproto): the command limits. Required.");
ABSL_FLAG(std::string,
          network_config,
          "",
          "The network file of the bus (config/ipc/network.textproto); empty: the shipped single-machine network.");
ABSL_FLAG(std::string, ipc_node, "teleop", "The bus node this process publishes as.");
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/teleop/README.md:teleop_flags)

/*
 * velocity_keyboard_command: the walking command of the pelvis typed as a line, "v_x v_y delta_height yaw_rate", and
 * published on operator/walking_velocity_command until the next line (the ROS-era velocity_keyboard_command_node on the
 * bus). See humanoid_nmpc/humanoid_common_mpc_app/teleop/README.md.
 */
int main(int argc, char* absl_nonnull* absl_nonnull argv) {
  absl::SetProgramUsageMessage(
      "Publishes the walking command typed in the terminal on operator/walking_velocity_command.\n"
      "  velocity_keyboard_command --reference_file=<robot>/config/command/reference.textproto [--network_config=...] [--ipc_node=teleop]");
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  ocs2::humanoid::node::installShutdownSignalHandlers();

  const std::string referenceFile = absl::GetFlag(FLAGS_reference_file);
  if (const absl::Status valid = ocs2::humanoid::node::validateFileFlag("--reference_file", referenceFile); !valid.ok()) {
    LOG(ERROR) << valid.message();
    return 2;
  }
  absl::StatusOr<ocs2::humanoid::teleop::KeyboardCommandLimits> limits = ocs2::humanoid::teleop::loadKeyboardCommandLimits(referenceFile);
  if (!limits.ok()) {
    LOG(ERROR) << limits.status();
    return 2;
  }

  absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> bus =
      ocs2::humanoid::node::createNodeBus(absl::GetFlag(FLAGS_network_config), absl::GetFlag(FLAGS_ipc_node));
  if (!bus.ok()) {
    LOG(ERROR) << "The bus: " << bus.status();
    return 2;
  }
  absl::StatusOr<std::unique_ptr<ocs2::humanoid::teleop::VelocityCommandRepeater>> repeater =
      ocs2::humanoid::teleop::VelocityCommandRepeater::Create(**bus);
  if (!repeater.ok()) {
    LOG(ERROR) << repeater.status();
    return 1;
  }
  if (const absl::Status started = (*bus)->start(); !started.ok()) {
    LOG(ERROR) << "Starting the bus: " << started;
    return 1;
  }

  ocs2::humanoid::teleop::LineReader reader(STDIN_FILENO);
  constexpr char kPrompt[] = "Enter v_x [m/s], v_y [m/s], delta_height [m], ang_vel_z [rad/s] of the PELVIS, separated by spaces: ";
  while (!ocs2::humanoid::node::shutdownRequested()) {
    std::cout << kPrompt << std::flush;
    const std::optional<std::string> line = reader.readLine([]() { return ocs2::humanoid::node::shutdownRequested(); });
    if (!line.has_value()) break;

    // Before every command, so that a limit the operator has just changed in reference.textproto - through the tuning
    // GUI or by hand - applies to the command about to be sent: the MPC scales it back by its own copy, which its parameter
    // updater reloads, and a stale copy here would make the robot walk at a speed nobody asked for.
    absl::StatusOr<ocs2::humanoid::teleop::KeyboardCommandLimits> reloaded =
        ocs2::humanoid::teleop::loadKeyboardCommandLimits(referenceFile);
    if (reloaded.ok()) {
      limits = *std::move(reloaded);
    } else {
      LOG(WARNING) << "Keeping the previous command limits: " << reloaded.status();
    }
    const absl::StatusOr<ocs2::vector4_t> command = ocs2::humanoid::teleop::parseKeyboardCommandLine(*line);
    if (!command.ok()) {
      LOG(WARNING) << command.status().message() << "; nothing is sent.";
      continue;
    }
    const ocs2::vector4_t clamped = command->cwiseMin(limits->limits).cwiseMax(-limits->limits);
    LOG(INFO) << "Publishing [" << absl::StrJoin(clamped.data(), clamped.data() + clamped.size(), ", ") << "]";
    (*repeater)->setCommand(ocs2::humanoid::teleop::keyboardCommandToMessage(*command, *limits));
  }
  (*bus)->stop();
  return 0;
}
