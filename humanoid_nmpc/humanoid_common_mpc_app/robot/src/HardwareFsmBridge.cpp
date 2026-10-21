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

#include "humanoid_common_mpc_app/robot/HardwareFsmBridge.h"

#include <string>

#include "absl/base/nullability.h"

#include "humanoid_common_mpc/mrt/ControlMode.h"

namespace ocs2::humanoid {

HardwareFsmBridge::HardwareFsmBridge(const robot::model::RobotDescription& robotDescription,
                                     const robot::model::RobotState& initState,
                                     OperatorCommandMailbox* absl_nonnull commands,
                                     FsmStateMailbox* absl_nonnull fsmStates,
                                     RobotBackend* absl_nonnull backend,
                                     RealtimeEventLog* absl_nullable eventLog)
    : backend_(backend), eventLog_(eventLog), state_(robotDescription, initState, commands, fsmStates, /*gantryLocked=*/false) {}

bool HardwareFsmBridge::processCommands(std::string& currentModeName) {
  FsmCommandEvent command;
  if (!state_.commands().takeFsmCommand(command)) {
    return false;
  }
  const absl::string_view name = fsmCommandEventName(command);
  switch (command.kind) {
    case FsmCommandKind::kZeroTorque:
      if (backend_->acceptsJointAction()) {
        if (eventLog_ != nullptr) eventLog_->post(RealtimeEventCode::kTorquesDisabled, /*detail=*/0, name);
        backend_->enterSafeState();
      }
      currentModeName.assign(control_mode::kZeroTorque.data(), control_mode::kZeroTorque.size());
      publishFsmState(currentModeName);
      return true;
    case FsmCommandKind::kJointPd:
    case FsmCommandKind::kGravityComp:
    case FsmCommandKind::kWbMpc:
    case FsmCommandKind::kSafety: {
      if (!backend_->acceptsJointAction()) {
        if (eventLog_ != nullptr) eventLog_->post(RealtimeEventCode::kTorquesEnabled, /*detail=*/0, name);
        backend_->enableTorques();
      }
      const absl::string_view mode = fsmCommandModeName(command.kind);
      currentModeName.assign(mode.data(), mode.size());
      publishFsmState(currentModeName);
      return true;
    }
    case FsmCommandKind::kLockGantry:
    case FsmCommandKind::kUnlockGantry:
      if (eventLog_ != nullptr) eventLog_->post(RealtimeEventCode::kGantryCommandIgnored, /*detail=*/0, name);
      return false;
  }
  return false;
}

}  // namespace ocs2::humanoid
