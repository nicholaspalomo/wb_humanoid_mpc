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

#include "humanoid_common_mpc_app/robot/FsmCommand.h"

#include <algorithm>
#include <cstring>

#include "humanoid_common_mpc/mrt/ControlMode.h"

namespace ocs2::humanoid {

std::optional<FsmCommandKind> parseFsmCommand(absl::string_view command) {
  // LINT.IfChange(fsm_command_names)
  if (command == control_mode::kZeroTorque || command == "DISABLE_TORQUES") return FsmCommandKind::kZeroTorque;
  if (command == control_mode::kJointPd) return FsmCommandKind::kJointPd;
  if (command == control_mode::kGravityComp) return FsmCommandKind::kGravityComp;
  if (command == control_mode::kWbMpc || command == control_mode::kMpcActive || command == "ENABLE_TORQUES") return FsmCommandKind::kWbMpc;
  if (command == control_mode::kSafety) return FsmCommandKind::kSafety;
  if (command == "LOCK_GANTRY") return FsmCommandKind::kLockGantry;
  if (command == "UNLOCK_GANTRY") return FsmCommandKind::kUnlockGantry;
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/remote_control/remote_control/humanoid_finite_state_machine.py:control_mode_names, //humanoid_nmpc/remote_control/remote_control/base_velocity_controller_gui.py:gantry_commands)
  // clang-format on
  return std::nullopt;
}

absl::string_view fsmCommandModeName(FsmCommandKind kind) {
  switch (kind) {
    case FsmCommandKind::kZeroTorque:
      return control_mode::kZeroTorque;
    case FsmCommandKind::kJointPd:
      return control_mode::kJointPd;
    case FsmCommandKind::kGravityComp:
      return control_mode::kGravityComp;
    case FsmCommandKind::kWbMpc:
      return control_mode::kWbMpc;
    case FsmCommandKind::kSafety:
      return control_mode::kSafety;
    case FsmCommandKind::kLockGantry:
    case FsmCommandKind::kUnlockGantry:
      return {};
  }
  return {};  // not an enumerator (ToTW #147)
}

FsmCommandEvent makeFsmCommandEvent(FsmCommandKind kind, absl::string_view command) {
  FsmCommandEvent event;
  event.kind = kind;
  const size_t length = std::min(command.size(), event.name.size() - 1);
  std::memcpy(event.name.data(), command.data(), length);
  event.name[length] = '\0';
  return event;
}

absl::string_view fsmCommandEventName(const FsmCommandEvent& event) {
  return absl::string_view(event.name.data(), ::strnlen(event.name.data(), event.name.size()));
}

}  // namespace ocs2::humanoid
