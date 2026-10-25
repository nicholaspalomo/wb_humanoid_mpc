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
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"
#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "robot_model/ContactEstimatorRegistry.h"
#include "robot_model/RobotHWInterfaceBase.h"

namespace ocs2::humanoid {

/**
 * The robot the robot process controls: a robot::model::RobotHWInterfaceBase,
 * which the realtime loop reads the state from and writes the joint action to,
 * and what the backend adds around it. Selected by name with --backend
 * (RobotBackendRegistry): `mujoco` is the MuJoCo simulator
 * (MujocoRobotBackend), so that simulation runs the hardware topology. A
 * hardware backend is registered next to it under its own name; it names the
 * supervisor that drives it (supervisorName(): the FSM bridge and the fall
 * recovery of RobotSupervisorRegistry): `sim` drives the simulator's gantry,
 * torque switch and resets (simulator()), `hardware` the backend's torque
 * switch only (enableTorques(), enterSafeState()). See
 * humanoid_nmpc/humanoid_common_mpc_app/robot/README.md, "A hardware backend".
 *
 * WHAT THE STATE CARRIES. hardware().getRobotState() is what the backend
 * measures: the joints, the contact flags, the base's orientation and angular
 * velocity (the IMU's on hardware), the IMU's linear acceleration
 * (RobotState::getImuLinearAccelerationInLocalFrame()), and the base's position
 * and linear velocity, which a simulator knows and a robot does not: a hardware
 * backend reports its best guess (the initial pose, or a motion capture's), and
 * the task file's state_estimator replaces them for the controller (`kalman`).
 *
 * THREADS. initialize() and start() run on the main thread before the realtime
 * loop; hardware(), acceptsJointAction() and readMeasuredContactForces() are
 * called by the realtime thread every cycle and must be realtime-safe: no lock
 * the backend's own threads hold, no allocation, no I/O. hardware()'s state and
 * action go through the lock-free buffers of RobotHWInterfaceBase.
 * enterSafeState() is called by the realtime thread when its cycle fails, and
 * by the main thread when the process stops.
 */
class RobotBackend {
 public:
  RobotBackend() = default;
  /** An interface: implementations are used through a pointer or reference,
   * never copied or moved. */
  RobotBackend(const RobotBackend&) = delete;
  RobotBackend& operator=(const RobotBackend&) = delete;
  virtual ~RobotBackend() = default;

  /** The registry name of this backend. */
  virtual absl::string_view name() const = 0;

  /** The robot's state and joint action. */
  virtual robot::model::RobotHWInterfaceBase& hardware() = 0;
  virtual const robot::model::RobotHWInterfaceBase& hardware() const = 0;

  /**
   * Brings the robot up before the loop starts (MuJoCo: takes the first physics
   * step and starts the viewer). A step of the process's lifecycle, not the
   * second half of construction: a backend is complete when its registry
   * factory returns it (hardware() and the realtime-thread methods are valid
   * from then on), and this connects it to what is outside the process.
   * RobotProcess::start() calls it once, on the main thread, before reading the
   * first state and before start(); a backend whose initialize() fails is
   * destroyed without start().
   */
  virtual absl::Status initialize() = 0;

  /**
   * Starts the backend's own threads (MuJoCo: the physics and the viewer),
   * pinned to `cores` (empty: not pinned). The robot starts with its torques
   * off (ZERO_TORQUE) and, in simulation, held by the gantry.
   */
  virtual absl::Status start(const std::vector<int>& cores) = 0;

  /** Whether the joint action of this cycle reaches the actuators: false while
   * the torques are off. Realtime thread. */
  virtual bool acceptsJointAction() const = 0;

  /**
   * Switches the torques on (leaves the safe state): the joint action applied
   * from now on reaches the actuators, and none applied before
   * (RobotHWInterfaceBase::discardAppliedJointAction()). The FSM bridge's
   * switch for an FSM command into an active mode. Realtime-safe; idempotent.
   */
  virtual void enableTorques() = 0;

  /**
   * The forces the feet measure [N], in contact order (left, right): the force
   * each foot carries in the world frame, +z up while it stands
   * (robot_state_sample.proto, measured_contact_wrenches); zero where the robot
   * has no sensor. Realtime thread.
   */
  virtual void readMeasuredContactForces(
      std::array<vector3_t, kNumContacts>& forces) = 0;

  /**
   * Takes the actuators to the backend's safe state, whatever action is
   * latched: the realtime loop's last act when a cycle fails (an exception) and
   * when the process stops, so that no action of a controller that is gone
   * stays in force (MuJoCo: zero torque, with the ragdoll damping).
   * Realtime-safe; idempotent. The robot leaves it through the FSM as from the
   * start-up state (an FSM command that switches the torques on).
   */
  virtual void enterSafeState() = 0;

  /** Adds the contact estimators only this backend can provide (MuJoCo:
   * cheater_sim) to `registry`. */
  virtual void registerContactEstimators(
      robot::model::ContactEstimatorRegistry& registry) const = 0;

  /** The RobotSupervisorRegistry name of the FSM bridge and fall recovery that
   * drive this backend (`sim`, `hardware`). */
  virtual absl::string_view supervisorName() const = 0;

  /**
   * The simulator, when this backend is one: its gantry, its torque switch, its
   * dodgeballs and its viewer, which the `sim` supervisor and the viewer
   * annotations drive. nullptr on hardware.
   */
  virtual robot::mujoco_sim_interface::MujocoSimInterface* absl_nullable
  simulator() {
    return nullptr;
  }
};

}  // namespace ocs2::humanoid
