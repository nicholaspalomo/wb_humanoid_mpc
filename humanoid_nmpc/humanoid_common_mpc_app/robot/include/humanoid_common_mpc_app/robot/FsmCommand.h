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
#include <cstdint>
#include <optional>

#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

/** What an operator/fsm_command asks the robot process to do, parsed off the realtime thread (parseFsmCommand()). */
enum class FsmCommandKind : std::uint8_t {
  kZeroTorque,    ///< ZERO_TORQUE, DISABLE_TORQUES: the torques off
  kJointPd,       ///< JOINT_PD
  kGravityComp,   ///< GRAVITY_COMP
  kWbMpc,         ///< WB_MPC, and its aliases MPC_ACTIVE and ENABLE_TORQUES
  kSafety,        ///< SAFETY
  kLockGantry,    ///< LOCK_GANTRY
  kUnlockGantry,  ///< UNLOCK_GANTRY
};

/** One FSM command as the realtime thread receives it: plain data, for an SPSC queue. */
struct FsmCommandEvent {
  FsmCommandKind kind = FsmCommandKind::kZeroTorque;
  /** The command as the operator sent it, NUL-terminated (for the log line of the realtime thread's report). */
  std::array<char, 24> name{};
};

/**
 * The command `command` names, or nullopt for a string that names none (the robot ignores it, as the ROS sims did). The
 * names are the remote control's (remote_control/humanoid_finite_state_machine.py) and those of the ROS sims' FSM
 * bridge: ENABLE_TORQUES and MPC_ACTIVE enter WB_MPC, DISABLE_TORQUES is ZERO_TORQUE.
 */
std::optional<FsmCommandKind> parseFsmCommand(absl::string_view command);

/** The control mode a mode command enters ("WB_MPC" for kWbMpc); empty for the gantry commands. */
absl::string_view fsmCommandModeName(FsmCommandKind kind);

/** The event for `kind`, with `command` as its name (cut to fit). */
FsmCommandEvent makeFsmCommandEvent(FsmCommandKind kind, absl::string_view command);

/** The name of `event`, as a view of its NUL-terminated buffer. */
absl::string_view fsmCommandEventName(const FsmCommandEvent& event);

}  // namespace ocs2::humanoid
