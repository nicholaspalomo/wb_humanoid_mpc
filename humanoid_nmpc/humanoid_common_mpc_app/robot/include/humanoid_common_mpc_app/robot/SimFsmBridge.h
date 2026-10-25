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

#include <cstdint>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc_app/robot/FsmBridge.h"
#include "humanoid_common_mpc_app/robot/FsmBridgeState.h"
#include "humanoid_common_mpc_app/robot/FsmStateMailbox.h"
#include "humanoid_common_mpc_app/robot/OperatorCommandMailbox.h"
#include "humanoid_common_mpc_app/robot/RealtimeEventLog.h"
#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid {

/**
 * The FsmBridge of the simulated robot (the `sim` supervisor of
 * RobotSupervisorRegistry): the FSM commands switch the simulator's torques and
 * move its virtual gantry, the gantry follows the GUI's height slider while it
 * is locked, and a dodgeball throw from the GUI is handed to the simulator.
 *
 * The FSM commands: ZERO_TORQUE (or DISABLE_TORQUES) switches the torques off;
 * JOINT_PD, GRAVITY_COMP, WB_MPC (or MPC_ACTIVE, ENABLE_TORQUES) and SAFETY
 * switch them on and enter the mode; LOCK_GANTRY and UNLOCK_GANTRY move the
 * virtual gantry. The fall recovery - catching a fallen robot, noticing the
 * simulator's own resets, settling the caught robot - is SimFallRecovery. Every
 * call is for the realtime thread.
 */
class SimFsmBridge final : public FsmBridge {
 public:
  /**
   * Captures the nominal joint positions from `initState` and writes the
   * initial state (ZERO_TORQUE, the simulator's gantry) to `fsmStates`. The
   * mailboxes, the simulator and the log must outlive the bridge; `eventLog`
   * may be nullptr.
   */
  SimFsmBridge(
      const robot::model::RobotDescription& robotDescription,
      const robot::model::RobotState& initState,
      OperatorCommandMailbox* absl_nonnull commands,
      FsmStateMailbox* absl_nonnull fsmStates,
      robot::mujoco_sim_interface::MujocoSimInterface* absl_nonnull simulator,
      RealtimeEventLog* absl_nullable eventLog = nullptr);

  SimFsmBridge(const SimFsmBridge&) = delete;
  SimFsmBridge& operator=(const SimFsmBridge&) = delete;
  ~SimFsmBridge() override = default;

  void applyJointTargetUpdates() override { state_.applyJointTargetUpdates(); }
  const std::vector<scalar_t>& getNominalJointPositions() const override {
    return state_.nominalJointPositions();
  }

  /**
   * Hands the newest dodgeball throw to the simulator, then processes the
   * oldest FSM command, if any, applies its mode and gantry changes to the
   * simulator and publishes the updated state. Without a command, the gantry
   * follows the height slider while it is locked.
   */
  bool processCommands(std::string& currentModeName) override;

  bool isGantryLocked() const override { return simulator_->isGantryLocked(); }
  void publishFsmState(absl::string_view modeName) override {
    state_.publish(modeName, isGantryLocked());
  }
  void publishControllerReset(absl::string_view modeName) override {
    state_.publishControllerReset(modeName, isGantryLocked());
  }
  void setMpcHealthy(bool mpcHealthy, absl::string_view modeName) override {
    state_.setMpcHealthy(mpcHealthy, modeName, isGantryLocked());
  }
  uint64_t controllerResets() const override {
    return state_.controllerResets();
  }
  bool mpcHealthy() const override { return state_.mpcHealthy(); }

 private:
  robot::mujoco_sim_interface::MujocoSimInterface* absl_nonnull simulator_;
  RealtimeEventLog* absl_nullable const eventLog_;
  FsmBridgeState state_;
};

}  // namespace ocs2::humanoid
