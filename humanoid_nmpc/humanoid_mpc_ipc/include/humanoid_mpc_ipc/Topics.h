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

#include <array>

#include "absl/strings/string_view.h"

/**
 * The topics of the IPC bus (humanoid_nmpc/docs/distributed_runtime/README.md, "Topics"). A topic is the first frame of
 * every bus message; the comment of each constant names the message it carries and who publishes it.
 *
 * ZeroMQ's SUB filter matches a prefix of the topic frame, so no topic may be a prefix of another
 * (test/testTopics.cpp checks it).
 */
namespace ocs2::humanoid::ipc::topics {

// LINT.IfChange(topics)
// MpcObservation, robot -> MPC, every control cycle.
inline constexpr absl::string_view kRobotMpcObservation = "robot/mpc_observation";
// RobotStateSample, robot -> MPC (visualization), every telemetry period.
inline constexpr absl::string_view kRobotState = "robot/state";
// FsmState, robot -> GUI, on change and at 2 Hz.
inline constexpr absl::string_view kRobotFsmState = "robot/fsm_state";
// LoopTiming, robot -> GUI and Rerun bridge, at 1 Hz.
inline constexpr absl::string_view kRobotLoopTiming = "robot/loop_timing";
// MpcPolicy, MPC -> robot and dummy sim, every solve.
inline constexpr absl::string_view kMpcPolicy = "mpc/policy";
// MpcStatus, MPC -> robot (solver health), GUI and Rerun bridge, every solve attempt.
inline constexpr absl::string_view kMpcStatus = "mpc/status";
// VisualizationScene, MPC (visualization) -> Rerun bridge.
inline constexpr absl::string_view kVizScene = "viz/scene";
// TelemetrySeries, MPC (visualization) -> Rerun bridge, one per robot/state sample.
inline constexpr absl::string_view kVizTelemetry = "viz/telemetry";
// WalkingVelocityCommand, GUI and teleop -> MPC and robot, at 25 Hz.
inline constexpr absl::string_view kOperatorWalkingVelocityCommand = "operator/walking_velocity_command";
// FsmCommand, GUI -> robot, on change.
inline constexpr absl::string_view kOperatorFsmCommand = "operator/fsm_command";
// YamlDocument, GUI -> MPC and robot, on edit.
inline constexpr absl::string_view kOperatorMpcParameters = "operator/mpc_parameters";
// YamlDocument, GUI -> robot, on edit.
inline constexpr absl::string_view kOperatorPdGains = "operator/pd_gains";
// JointTargets, GUI -> robot, on edit (JOINT_PD only).
inline constexpr absl::string_view kOperatorJointTargets = "operator/joint_targets";
// YamlDocument, GUI -> robot (simulation), on button press.
inline constexpr absl::string_view kOperatorDodgeballThrow = "operator/dodgeball_throw";

// Every topic above, for tools that list or check them.
inline constexpr std::array<absl::string_view, 14> kAllTopics = {
    kRobotMpcObservation,
    kRobotState,
    kRobotFsmState,
    kRobotLoopTiming,
    kMpcPolicy,
    kMpcStatus,
    kVizScene,
    kVizTelemetry,
    kOperatorWalkingVelocityCommand,
    kOperatorFsmCommand,
    kOperatorMpcParameters,
    kOperatorPdGains,
    kOperatorJointTargets,
    kOperatorDodgeballThrow,
};
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_ipc/python/humanoid_mpc_ipc/topics.py:topics, //humanoid_nmpc/docs/distributed_runtime/README.md:topic_table)
// clang-format on

}  // namespace ocs2::humanoid::ipc::topics
