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

#include "humanoid_common_mpc_app/robot/SimFsmBridge.h"

#include <optional>
#include <string>

#include "absl/base/nullability.h"

#include "humanoid_common_mpc/mrt/ControlMode.h"

namespace ocs2::humanoid {

SimFsmBridge::SimFsmBridge(const robot::model::RobotDescription& robotDescription,
                           const robot::model::RobotState& initState,
                           OperatorCommandMailbox* absl_nonnull commands,
                           FsmStateMailbox* absl_nonnull fsmStates,
                           robot::mujoco_sim_interface::MujocoSimInterface* absl_nonnull simulator,
                           RealtimeEventLog* absl_nullable eventLog)
    : simulator_(simulator),
      eventLog_(eventLog),
      state_(robotDescription, initState, commands, fsmStates, /*gantryLocked=*/simulator->isGantryLocked()) {}

bool SimFsmBridge::processCommands(std::string& currentModeName) {
  OperatorCommandMailbox& commands = state_.commands();
  // Handed over first and unconditionally: a throw is independent of the FSM, and the early return below fires on
  // every cycle in which no mode change is pending - which is almost all of them.
  robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow throwCommand;
  if (commands.takeDodgeballThrow(throwCommand)) {
    simulator_->throwDodgeball(throwCommand);
  }

  FsmCommandEvent command;
  if (!commands.takeFsmCommand(command)) {
    // No FSM command pending, but the gantry still follows the height slider while it is locked - once the GUI has
    // sent a height at all.
    if (simulator_->isGantryLocked()) {
      const std::optional<double> height = commands.desiredGantryHeight();
      if (height.has_value()) simulator_->setGantryHeight(*height);
    }
    return false;
  }

  const absl::string_view name = fsmCommandEventName(command);
  switch (command.kind) {
    case FsmCommandKind::kZeroTorque:
      // 1. Zero-torque mode commands
      if (!simulator_->isZeroTorqueMode()) {
        if (eventLog_ != nullptr) eventLog_->post(RealtimeEventCode::kTorquesDisabled, /*detail=*/0, name);
        simulator_->disableTorques();
      }
      currentModeName.assign(control_mode::kZeroTorque.data(), control_mode::kZeroTorque.size());
      publishFsmState(currentModeName);
      return true;
    case FsmCommandKind::kJointPd:
    case FsmCommandKind::kGravityComp:
    case FsmCommandKind::kWbMpc:
    case FsmCommandKind::kSafety: {
      // 2. Active torque modes (JOINT_PD, GRAVITY_COMP, WB_MPC, SAFETY, MPC_ACTIVE, ENABLE_TORQUES)
      // Every mode command is reported, also one between two active modes (JOINT_PD to WB_MPC), which switches nothing.
      if (simulator_->isZeroTorqueMode()) {
        if (eventLog_ != nullptr) eventLog_->post(RealtimeEventCode::kTorquesEnabled, /*detail=*/0, name);
        simulator_->enableTorques();
      } else if (eventLog_ != nullptr) {
        eventLog_->post(RealtimeEventCode::kModeCommanded, /*detail=*/0, name);
      }
      const absl::string_view mode = fsmCommandModeName(command.kind);
      currentModeName.assign(mode.data(), mode.size());
      publishFsmState(currentModeName);
      return true;
    }
    case FsmCommandKind::kLockGantry:
      // 3. Virtual gantry locking / unlocking commands
      if (simulator_->isGantryLocked()) return false;
      if (eventLog_ != nullptr) eventLog_->post(RealtimeEventCode::kGantryLockCommanded);
      simulator_->lockGantry();
      publishFsmState(currentModeName);
      return true;
    case FsmCommandKind::kUnlockGantry:
      if (!simulator_->isGantryLocked()) return false;
      if (eventLog_ != nullptr) eventLog_->post(RealtimeEventCode::kGantryUnlockCommanded);
      simulator_->unlockGantry();
      publishFsmState(currentModeName);
      return true;
  }
  return false;
}

}  // namespace ocs2::humanoid
