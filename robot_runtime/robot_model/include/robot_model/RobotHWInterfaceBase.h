#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#include <robot_core/TripleBuffer.h>
#include <robot_model/RobotJointAction.h>
#include <robot_model/RobotState.h>

#include "robot_model/RobotDescription.h"

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
  explicit RobotHWInterfaceBase(const std::string& urdfPath)
      : robotDescription_(urdfPath),
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

  /** Hands a new state to the controller. */
  void publishRobotState(const RobotState& state) {
    stateBuffer_.writeSlot() = state;
    stateBuffer_.publishWrite();
  }

  /**
   * Copies the newest action into `action`. False when it is stale: no action has been applied since the last
   * discardAppliedJointAction(), so the robot must not execute it.
   */
  bool takeJointAction(RobotJointAction& action) {
    actionBuffer_.acquireRead();
    const ActionSlot& slot = actionBuffer_.readSlot();
    action = slot.action;
    return slot.generation == actionGeneration_.load(std::memory_order_acquire);
  }

  /** Marks every action applied so far as stale (see the class comment). Any thread; lock-free. */
  void discardAppliedJointAction() {
    actionGeneration_.fetch_add(1, std::memory_order_acq_rel);  // NOLINT(argument-comment): libstdc++ names the value __i.
  }

 private:
  /** An applied action and the discardAppliedJointAction() generation it was applied in. */
  struct ActionSlot {
    RobotJointAction action;
    std::uint64_t generation = 0;
  };

  const RobotDescription robotDescription_;
  RobotState robotState_;
  RobotJointAction robotJointAction_;
  TripleBuffer<RobotState> stateBuffer_;
  TripleBuffer<ActionSlot> actionBuffer_;
  std::atomic<std::uint64_t> actionGeneration_{0};
};

}  // namespace robot::model
