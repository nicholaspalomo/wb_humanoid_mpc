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

#include "absl/strings/string_view.h"

namespace ocs2::humanoid::control_mode {

/**
 * The names of the supervisory control modes, as the FSM commands and publishes them (the bus topics
 * `operator/fsm_command` and `robot/fsm_state`, humanoid_mpc_ipc/Topics.h), and the two families the MRT joint controllers and the sim fall
 * recovery tell apart: the passive modes, whose action is computed without the MPC, and the modes that execute the MPC policy.
 */
// LINT.IfChange(control_mode_names)
inline constexpr absl::string_view kZeroTorque = "ZERO_TORQUE";
inline constexpr absl::string_view kJointPd = "JOINT_PD";
inline constexpr absl::string_view kGravityComp = "GRAVITY_COMP";
inline constexpr absl::string_view kSafety = "SAFETY";
inline constexpr absl::string_view kWbMpc = "WB_MPC";
/// An older name of WB_MPC that the controllers still accept.
inline constexpr absl::string_view kMpcActive = "MPC_ACTIVE";
// LINT.ThenChange(//humanoid_nmpc/remote_control/remote_control/humanoid_finite_state_machine.py:control_mode_names)

// LINT.IfChange(control_mode_families)
/** ZERO_TORQUE, JOINT_PD, GRAVITY_COMP or SAFETY: an action computed without the MPC. */
inline bool isPassive(absl::string_view mode) {
  return mode == kZeroTorque || mode == kJointPd || mode == kGravityComp || mode == kSafety;
}

/** WB_MPC (or MPC_ACTIVE): the MPC policy is executed. */
inline bool isMpc(absl::string_view mode) {
  return mode == kWbMpc || mode == kMpcActive;
}
// LINT.ThenChange(//humanoid_nmpc/remote_control/remote_control/humanoid_finite_state_machine.py:control_mode_names)

}  // namespace ocs2::humanoid::control_mode
