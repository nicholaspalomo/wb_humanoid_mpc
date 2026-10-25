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
#include "humanoid_common_mpc_app/robot/RobotBackend.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid {

/**
 * The FsmBridge of a robot without a simulator (the `hardware` supervisor of
 * RobotSupervisorRegistry): the FSM commands switch the backend's torques -
 * ZERO_TORQUE (or DISABLE_TORQUES) puts it in its safe state
 * (enterSafeState()), JOINT_PD, GRAVITY_COMP, WB_MPC (or MPC_ACTIVE,
 * ENABLE_TORQUES) and SAFETY switch them on (enableTorques()) and enter the
 * mode. The process drives no gantry here: a gantry, if the robot hangs in one,
 * is the operator's, so LOCK_GANTRY and UNLOCK_GANTRY are reported and ignored,
 * the FSM state reports the gantry unlocked, the height slider moves nothing
 * and a dodgeball throw is not taken. Every call is for the realtime thread.
 */
class HardwareFsmBridge final : public FsmBridge {
 public:
  /**
   * Captures the nominal joint positions from `initState` and writes the
   * initial state (ZERO_TORQUE) to `fsmStates`. The mailboxes, the backend and
   * the log must outlive the bridge; `eventLog` may be nullptr.
   */
  HardwareFsmBridge(const robot::model::RobotDescription& robotDescription,
                    const robot::model::RobotState& initState,
                    OperatorCommandMailbox* absl_nonnull commands,
                    FsmStateMailbox* absl_nonnull fsmStates,
                    RobotBackend* absl_nonnull backend,
                    RealtimeEventLog* absl_nullable eventLog = nullptr);

  HardwareFsmBridge(const HardwareFsmBridge&) = delete;
  HardwareFsmBridge& operator=(const HardwareFsmBridge&) = delete;
  ~HardwareFsmBridge() override = default;

  void applyJointTargetUpdates() override { state_.applyJointTargetUpdates(); }
  const std::vector<scalar_t>& getNominalJointPositions() const override {
    return state_.nominalJointPositions();
  }
  bool processCommands(std::string& currentModeName) override;
  bool isGantryLocked() const override { return false; }
  void publishFsmState(absl::string_view modeName) override {
    state_.publish(modeName, /*gantryLocked=*/false);
  }
  void publishControllerReset(absl::string_view modeName) override {
    state_.publishControllerReset(modeName, /*gantryLocked=*/false);
  }
  void setMpcHealthy(bool mpcHealthy, absl::string_view modeName) override {
    state_.setMpcHealthy(mpcHealthy, modeName, /*gantryLocked=*/false);
  }
  uint64_t controllerResets() const override {
    return state_.controllerResets();
  }
  bool mpcHealthy() const override { return state_.mpcHealthy(); }

 private:
  RobotBackend* absl_nonnull backend_;
  RealtimeEventLog* absl_nullable const eventLog_;
  FsmBridgeState state_;
};

}  // namespace ocs2::humanoid
