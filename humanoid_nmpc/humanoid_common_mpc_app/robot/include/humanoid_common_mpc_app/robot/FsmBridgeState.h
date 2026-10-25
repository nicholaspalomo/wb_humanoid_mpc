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
#include <vector>

#include "absl/base/nullability.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc_app/robot/FsmStateMailbox.h"
#include "humanoid_common_mpc_app/robot/OperatorCommandMailbox.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid {

/**
 * What every FsmBridge keeps, whatever it switches: the nominal posture of
 * JOINT_PD with the operator's joint targets merged in, the controller resets
 * counted so far and the MPC's health, and the FSM state it writes to the
 * FsmStateMailbox. The bridges hold one each (composition). Realtime thread;
 * allocates nothing after construction.
 */
class FsmBridgeState {
 public:
  /**
   * Captures the nominal joint positions from `initState` and writes the
   * initial FSM state (ZERO_TORQUE, the gantry as `gantryLocked` says) to
   * `fsmStates`. The mailboxes must outlive the state.
   */
  FsmBridgeState(const robot::model::RobotDescription& robotDescription,
                 const robot::model::RobotState& initState,
                 OperatorCommandMailbox* absl_nonnull commands,
                 FsmStateMailbox* absl_nonnull fsmStates,
                 bool gantryLocked);

  FsmBridgeState(const FsmBridgeState&) = delete;
  FsmBridgeState& operator=(const FsmBridgeState&) = delete;
  ~FsmBridgeState() = default;

  OperatorCommandMailbox& commands() { return *commands_; }

  /** Takes the operator's newest joint targets into the nominal posture, if new
   * ones have arrived. */
  void applyJointTargetUpdates() {
    commands_->takeNominalPosture(nominalJointPositions_);
  }
  const std::vector<scalar_t>& nominalJointPositions() const {
    return nominalJointPositions_;
  }

  /** Writes the FSM state of `modeName` and the gantry. */
  void publish(absl::string_view modeName, bool gantryLocked);
  /** Counts one controller reset and writes the state. */
  void publishControllerReset(absl::string_view modeName, bool gantryLocked);
  /** Records the MPC's health and writes the state when it changed. */
  void setMpcHealthy(bool mpcHealthy,
                     absl::string_view modeName,
                     bool gantryLocked);

  uint64_t controllerResets() const { return controllerResets_; }
  bool mpcHealthy() const { return mpcHealthy_; }

 private:
  OperatorCommandMailbox* absl_nonnull commands_;
  FsmStateMailbox* absl_nonnull fsmStates_;
  std::vector<scalar_t> nominalJointPositions_;
  uint64_t controllerResets_ = 0;
  bool mpcHealthy_ = true;
};

}  // namespace ocs2::humanoid
