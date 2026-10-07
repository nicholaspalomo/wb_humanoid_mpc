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
                           OperatorCommandMailbox& commands,
                           FsmStateMailbox& fsmStates,
                           RealtimeEventLog* absl_nullable eventLog)
    : commands_(commands), fsmStates_(fsmStates), eventLog_(eventLog) {
  nominalJointPositions_.resize(robotDescription.getNumJoints(), 0.0);
  for (size_t i = 0; i < robotDescription.getNumJoints(); ++i) {
    nominalJointPositions_[i] = initState.getCheckedJointPosition(i);
  }
  // The initial state: zero torque, held by the gantry.
  publishFsmState(control_mode::kZeroTorque, /*gantryLocked=*/true);
}

void SimFsmBridge::publishFsmState(absl::string_view modeName, bool gantryLocked) {
  fsmStates_.write(modeName, gantryLocked, controllerResets_, mpcHealthy_);
}

void SimFsmBridge::publishControllerReset(absl::string_view modeName, bool gantryLocked) {
  ++controllerResets_;
  publishFsmState(modeName, gantryLocked);
}

void SimFsmBridge::setMpcHealthy(bool mpcHealthy, absl::string_view modeName, bool gantryLocked) {
  if (mpcHealthy == mpcHealthy_) {
    return;
  }
  mpcHealthy_ = mpcHealthy;
  publishFsmState(modeName, gantryLocked);
}

bool SimFsmBridge::processCommands(std::string& currentModeName, robot::mujoco_sim_interface::MujocoSimInterface& robotInterface) {
  // Handed over first and unconditionally: a throw is independent of the FSM, and the early return below fires on
  // every cycle in which no mode change is pending - which is almost all of them.
  robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow throwCommand;
  if (commands_.takeDodgeballThrow(throwCommand)) {
    robotInterface.throwDodgeball(throwCommand);
  }

  FsmCommandEvent command;
  if (!commands_.takeFsmCommand(command)) {
    // No FSM command pending, but the gantry still follows the height slider while it is locked - once the GUI has
    // sent a height at all.
    if (robotInterface.isGantryLocked()) {
      const std::optional<double> height = commands_.desiredGantryHeight();
      if (height.has_value()) robotInterface.setGantryHeight(*height);
    }
    return false;
  }

  const absl::string_view name = fsmCommandEventName(command);
  switch (command.kind) {
    case FsmCommandKind::kZeroTorque:
      // 1. Zero-torque mode commands
      if (!robotInterface.isZeroTorqueMode()) {
        if (eventLog_ != nullptr) eventLog_->post(RealtimeEventCode::kTorquesDisabled, /*detail=*/0, name);
        robotInterface.disableTorques();
      }
      currentModeName.assign(control_mode::kZeroTorque.data(), control_mode::kZeroTorque.size());
      publishFsmState(currentModeName, robotInterface.isGantryLocked());
      return true;
    case FsmCommandKind::kJointPd:
    case FsmCommandKind::kGravityComp:
    case FsmCommandKind::kWbMpc:
    case FsmCommandKind::kSafety: {
      // 2. Active torque modes (JOINT_PD, GRAVITY_COMP, WB_MPC, SAFETY, MPC_ACTIVE, ENABLE_TORQUES)
      if (robotInterface.isZeroTorqueMode()) {
        if (eventLog_ != nullptr) eventLog_->post(RealtimeEventCode::kTorquesEnabled, /*detail=*/0, name);
        robotInterface.enableTorques();
      }
      const absl::string_view mode = fsmCommandModeName(command.kind);
      currentModeName.assign(mode.data(), mode.size());
      publishFsmState(currentModeName, robotInterface.isGantryLocked());
      return true;
    }
    case FsmCommandKind::kLockGantry:
      // 3. Virtual gantry locking / unlocking commands
      if (robotInterface.isGantryLocked()) return false;
      if (eventLog_ != nullptr) eventLog_->post(RealtimeEventCode::kGantryLockCommanded);
      robotInterface.lockGantry();
      publishFsmState(currentModeName, robotInterface.isGantryLocked());
      return true;
    case FsmCommandKind::kUnlockGantry:
      if (!robotInterface.isGantryLocked()) return false;
      if (eventLog_ != nullptr) eventLog_->post(RealtimeEventCode::kGantryUnlockCommanded);
      robotInterface.unlockGantry();
      publishFsmState(currentModeName, robotInterface.isGantryLocked());
      return true;
  }
  return false;
}

void SimFsmBridge::applyJointTargetUpdates() {
  commands_.takeNominalPosture(nominalJointPositions_);
}

}  // namespace ocs2::humanoid
