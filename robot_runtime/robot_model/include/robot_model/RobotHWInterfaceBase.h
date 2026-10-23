/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

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

#include <atomic>
#include <cstdint>
#include <string>
#include <utility>

#include "robot_core/TripleBuffer.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"

namespace robot::model {

/**
 * How a controller and a robot hand the state and the joint action to each other.
 *
 * THE CONTROLLER'S SIDE, one thread (the robot process's realtime loop): updateInterfaceStateFromRobot() takes the
 * newest state the robot published into getRobotState(); the controller fills getRobotJointAction() and hands it over
 * with applyJointAction().
 *
 * THE ROBOT'S SIDE, one thread (a simulator's physics thread, a hardware driver): publishRobotState() hands a new state
 * over and takeJointAction() takes the newest action.
 *
 * Both directions go through a robot::TripleBuffer: lock-free, wait-free and, once the slots hold a state and an action
 * of the robot's size, free of allocation, so that neither side ever waits for the other. The controller's thread runs
 * SCHED_FIFO; the robot's may not, and a mutex shared between the two lets a preempted robot thread hold the control
 * loop up for a scheduler time slice (priority inversion: std::mutex has no priority inheritance).
 *
 * A STALE ACTION. discardAppliedJointAction() marks every action applied so far as not to be executed: until the next
 * applyJointAction(), takeJointAction() reports the action it copies as stale. A robot switches its torques back on
 * this way (MujocoSimInterface::enableTorques()), so that the action latched from before they went off - possibly long
 * before, in another mode - never reaches the actuators.
 */
class RobotHWInterfaceBase {
 public:
  /** The interface of the robot `robotDescription` describes (RobotDescription::Create()). */
  explicit RobotHWInterfaceBase(RobotDescription robotDescription)
      : robotDescription_(std::move(robotDescription)),
        robotState_(model::RobotState(robotDescription_)),
        robotJointAction_(model::RobotJointAction(robotDescription_)),
        stateBuffer_(robotState_),
        actionBuffer_(ActionSlot{.action = robotJointAction_, .generation = 0}) {}

  virtual ~RobotHWInterfaceBase() = default;

  RobotHWInterfaceBase(const RobotHWInterfaceBase&) = delete;
  RobotHWInterfaceBase& operator=(const RobotHWInterfaceBase&) = delete;

  const model::RobotDescription& getRobotDescription() const { return robotDescription_; }

  // ------------------------------------------------------------------ the controller's thread

  /** The state the last updateInterfaceStateFromRobot() took. */
  const RobotState& getRobotState() const { return robotState_; }

  /** Takes the newest state the robot has published; keeps the previous one when none is newer. */
  void updateInterfaceStateFromRobot() {
    stateBuffer_.acquireRead();
    robotState_ = stateBuffer_.readSlot();
  }

  /** The action to fill in before applyJointAction(). */
  RobotJointAction& getRobotJointAction() { return robotJointAction_; }

  /** Hands getRobotJointAction() to the robot. */
  void applyJointAction() {
    ActionSlot& slot = actionBuffer_.writeSlot();
    slot.action = robotJointAction_;
    slot.generation = actionGeneration_.load(std::memory_order_acquire);
    actionBuffer_.publishWrite();
  }

 protected:
  // ------------------------------------------------------------------ the robot's thread

  /**
   * Hands a new state to the controller. `state` is a state of getRobotDescription(), as every state and action this
   * interface copies: the copies are unchecked and allocation-free (IDMapBase::operator=).
   */
  void publishRobotState(const RobotState& state) {
    stateBuffer_.writeSlot() = state;
    stateBuffer_.publishWrite();
  }

  /**
   * Copies the newest action into `action`, an action of getRobotDescription(). False when it is stale: no action has
   * been applied since the last discardAppliedJointAction(), so the robot must not execute it.
   */
  bool takeJointAction(RobotJointAction& action) {
    actionBuffer_.acquireRead();
    const ActionSlot& slot = actionBuffer_.readSlot();
    action = slot.action;
    return slot.generation == actionGeneration_.load(std::memory_order_acquire);
  }

  /** Marks every action applied so far as stale (see the class comment). Any thread; lock-free. */
  void discardAppliedJointAction() { actionGeneration_.fetch_add(1, std::memory_order_acq_rel); }

 private:
  /** An applied action and the discardAppliedJointAction() generation it was applied in. */
  struct ActionSlot {
    RobotJointAction action;
    uint64_t generation = 0;
  };

  const RobotDescription robotDescription_;
  RobotState robotState_;
  RobotJointAction robotJointAction_;
  TripleBuffer<RobotState> stateBuffer_;
  TripleBuffer<ActionSlot> actionBuffer_;
  std::atomic<uint64_t> actionGeneration_{0};
};

}  // namespace robot::model
