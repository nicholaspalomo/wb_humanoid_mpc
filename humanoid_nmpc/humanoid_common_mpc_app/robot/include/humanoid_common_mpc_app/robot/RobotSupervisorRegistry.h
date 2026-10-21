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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc_app/robot/FallRecovery.h"
#include "humanoid_common_mpc_app/robot/FsmBridge.h"
#include "humanoid_common_mpc_app/robot/FsmStateMailbox.h"
#include "humanoid_common_mpc_app/robot/OperatorCommandMailbox.h"
#include "humanoid_common_mpc_app/robot/RealtimeEventLog.h"
#include "humanoid_common_mpc_app/robot/RobotBackend.h"
#include "humanoid_common_mpc_app/robot/SimFallRecovery.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid {

// LINT.IfChange(supervisor_names)
/** SimFsmBridge and SimFallRecovery: the simulator's torque switch, gantry, dodgeballs and resets. */
inline constexpr absl::string_view kSimSupervisorName = "sim";
/** HardwareFsmBridge and HardwareFallRecovery: the backend's torque switch only, no gantry, the tilt cut-off. */
inline constexpr absl::string_view kHardwareSupervisorName = "hardware";
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/README.md:supervisor_names)

/** The FSM bridge and the fall recovery that drive one backend, on the realtime thread of the robot process. */
struct RobotSupervisor {
  std::unique_ptr<FsmBridge> fsmBridge;
  std::unique_ptr<FallRecovery> fallRecovery;
};

/** What a supervisor is built from. Every pointer must outlive the supervisor; eventLog may be nullptr. */
struct RobotSupervisorContext {
  RobotBackend* absl_nonnull backend;
  const robot::model::RobotDescription* absl_nonnull description;
  /** The state the robot starts in, whose joints are the first nominal posture of JOINT_PD. */
  const robot::model::RobotState* absl_nonnull initialState;
  OperatorCommandMailbox* absl_nonnull commands;
  FsmStateMailbox* absl_nonnull fsmStates;
  RealtimeEventLog* absl_nullable eventLog = nullptr;
  /** The task file's fall recovery keys (`sim_max_base_tilt_angle`, `sim_gantry_catch_lift`). */
  SimFallRecovery::Config fallRecovery;
  /** The joints whose posture and speed decide whether a caught robot rests (SimFallRecovery). */
  std::vector<size_t> restJointIndices;
};

/**
 * The supervisors by name (RobotBackend::supervisorName()): the only place one is added. An unknown name, or a `sim`
 * supervisor for a backend that is no simulator, is refused with the reason and the list of the available ones.
 *
 * Not thread-safe: built and used on one thread during set-up.
 */
class RobotSupervisorRegistry {
 public:
  using Factory = std::function<absl::StatusOr<RobotSupervisor>(const RobotSupervisorContext& context)>;

  /** A registry with the built-in supervisors (`sim`, `hardware`). */
  RobotSupervisorRegistry();

  /** Adds the supervisor `name`. */
  void add(std::string name, std::string description, Factory factory);
  bool has(absl::string_view name) const;
  std::vector<std::string> names() const;
  /** "sim (the simulator's ...), hardware (...)". */
  std::string availableNames() const;

  /** The supervisor `name` names, built from `context`; InvalidArgument listing the available names for an unknown one. */
  absl::StatusOr<RobotSupervisor> create(absl::string_view name, const RobotSupervisorContext& context) const;

 private:
  struct Entry {
    std::string name;
    std::string description;
    Factory factory;
  };

  std::vector<Entry> entries_;
};

}  // namespace ocs2::humanoid
