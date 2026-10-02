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

#include "absl/strings/string_view.h"

#include <humanoid_common_mpc/common/Types.h>
#include <mujoco_sim_interface/MujocoSimInterface.h>
#include <robot_model/RobotDescription.h>
#include <robot_model/RobotState.h>

#include "humanoid_common_mpc_app/robot/FsmStateMailbox.h"
#include "humanoid_common_mpc_app/robot/OperatorCommandMailbox.h"
#include "humanoid_common_mpc_app/robot/RealtimeEventLog.h"

namespace ocs2::humanoid {

/**
 * The supervisory FSM of the simulated robot, on the realtime thread of the robot process: it takes the operator's
 * commands from the OperatorCommandMailbox, switches the simulator's torques and gantry, and hands the FSM state to
 * the FsmStateMailbox, which the communication thread publishes on robot/fsm_state. It touches no bus, parses nothing
 * and logs nothing (its reports go to a RealtimeEventLog). Every call is for the realtime thread.
 *
 * Encapsulates:
 * 1. The nominal stance positions (the JOINT_PD posture, which the controllers compute the action of), with the
 *    operator's joint targets merged in (applyJointTargetUpdates()).
 * 2. The FSM commands: ZERO_TORQUE (or DISABLE_TORQUES) switches the torques off; JOINT_PD, GRAVITY_COMP, WB_MPC (or
 *    MPC_ACTIVE, ENABLE_TORQUES) and SAFETY switch them on and enter the mode; LOCK_GANTRY and UNLOCK_GANTRY move the
 *    virtual gantry. One command per cycle, in the order they were sent.
 * 3. The gantry height slider (the walking command's desired_pelvis_height), followed while the gantry is locked.
 * 4. A dodgeball throw from the GUI, handed to the simulator.
 * 5. The FSM state: the mode, the gantry, the number of controller resets so far and the MPC's health.
 * The fall recovery - catching a fallen robot, noticing the simulator's own resets, settling the caught robot - is
 * SimFallRecovery.
 */
class SimFsmBridge {
 public:
  /**
   * Captures the nominal joint positions and the gantry height from `initState`, and writes the initial state
   * (ZERO_TORQUE, gantry locked) to `fsmStates`. The mailboxes and the log must outlive the bridge; `eventLog` may be
   * nullptr.
   */
  SimFsmBridge(const robot::model::RobotDescription& robotDescription,
               const robot::model::RobotState& initState,
               OperatorCommandMailbox& commands,
               FsmStateMailbox& fsmStates,
               RealtimeEventLog* eventLog = nullptr);

  /** Writes the current FSM mode, the gantry state, the controller resets so far and the MPC's health. */
  void publishFsmState(absl::string_view modeName, bool gantryLocked);

  /**
   * Counts one controller reset and publishes the state with the new count.
   *
   * A controller reset is a discontinuity of the plant after which the controller starts again from where the robot is
   * (SimFallRecovery::Cycle::discontinuity): a catch, a LOCK_GANTRY, or a reset the simulator made on its own thread.
   * The remote control re-centers its joysticks on every change of the count, which is what releases them after a reset
   * that changes neither the mode nor the gantry - the simulator putting the robot back while the gantry was already
   * locked in JOINT_PD.
   */
  void publishControllerReset(absl::string_view modeName, bool gantryLocked);

  /** Records the MPC's health (false: the controller holds JOINT_PD), and publishes the state when it changed. */
  void setMpcHealthy(bool mpcHealthy, absl::string_view modeName, bool gantryLocked);

  /** The controller resets counted so far, the controller_resets of every state published. */
  uint64_t controllerResets() const { return controllerResets_; }
  bool mpcHealthy() const { return mpcHealthy_; }

  /**
   * Hands the newest dodgeball throw to the simulator, then processes the oldest FSM command, if any, applies its
   * mode and gantry changes to `robotInterface` and publishes the updated state. Without a command, the gantry follows
   * the height slider while it is locked.
   * @param currentModeName The current mode name (updated if changed).
   * @return true if a command was processed, false otherwise.
   */
  bool processCommands(std::string& currentModeName, robot::mujoco_sim_interface::MujocoSimInterface& robotInterface);

  /** The nominal joint positions, indexed like the robot's joints. */
  const std::vector<scalar_t>& getNominalJointPositions() const { return nominalJointPositions_; }

  /** Takes the operator's newest joint targets into the nominal posture, if new ones have arrived. */
  void applyJointTargetUpdates();

 private:
  OperatorCommandMailbox& commands_;
  FsmStateMailbox& fsmStates_;
  RealtimeEventLog* const eventLog_;
  std::vector<scalar_t> nominalJointPositions_;
  uint64_t controllerResets_ = 0;
  bool mpcHealthy_ = true;
};

}  // namespace ocs2::humanoid
