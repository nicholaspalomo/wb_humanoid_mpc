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
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc_app/robot/RobotBackend.h"
#include "robot_model/ContactEstimatorRegistry.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotHWInterfaceBase.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid::robot_test {

/** The registry name of FakeRobotBackend. */
inline constexpr absl::string_view kFakeBackendName = "fake";

/**
 * A robot backend that is no simulator, for the tests of the robot process on
 * its `hardware` supervisor: the way a hardware backend plugs in
 * (humanoid_nmpc/humanoid_common_mpc_app/robot/README.md, "A hardware
 * backend"), with a driver thread in place of a robot's drives. start()
 * launches the driver, which every `period` publishes the state the trajectory
 * gives for its time (by default the initial state, standing still, the IMU
 * reading gravity alone) and takes the newest joint action, counting the ones
 * it would execute: those applied while the torques are on and after they came
 * on. The torque switch is an atomic flag; there is no gantry, no contact force
 * sensor and no own contact estimator.
 *
 * Thread-safe as a backend is: the robot process's realtime thread calls the
 * RobotBackend methods, the driver thread publishes, and the test thread reads
 * the counters. setTrajectory() comes before start().
 */
class FakeRobotBackend final : public RobotBackend {
 public:
  /** The state at `time` [s], written into `state` (a state of the backend's
   * description, starting from the last one). */
  using Trajectory =
      std::function<void(double time, robot::model::RobotState& state)>;

  static constexpr std::chrono::microseconds kDefaultPeriod{1000};

  FakeRobotBackend(robot::model::RobotDescription description,
                   const robot::model::RobotState& initialState,
                   std::chrono::microseconds period = kDefaultPeriod);
  ~FakeRobotBackend() override;

  FakeRobotBackend(const FakeRobotBackend&) = delete;
  FakeRobotBackend& operator=(const FakeRobotBackend&) = delete;

  /** Replaces the standing-still trajectory. Before start(). */
  void setTrajectory(Trajectory trajectory) {
    trajectory_ = std::move(trajectory);
  }

  absl::string_view name() const override { return kFakeBackendName; }
  robot::model::RobotHWInterfaceBase& hardware() override { return hardware_; }
  const robot::model::RobotHWInterfaceBase& hardware() const override {
    return hardware_;
  }
  /** Publishes the trajectory's state at time 0. */
  absl::Status initialize() override;
  /** Launches the driver thread; the cores are ignored. */
  absl::Status start(const std::vector<int>& cores) override;
  bool acceptsJointAction() const override { return torquesEnabled_.load(); }
  void enableTorques() override;
  void readMeasuredContactForces(
      std::array<vector3_t, kNumContacts>& forces) override;
  void enterSafeState() override;
  void registerContactEstimators(
      robot::model::ContactEstimatorRegistry& /*registry*/) const override {}
  absl::string_view supervisorName() const override;

  /** Stops and joins the driver thread. Idempotent; the destructor calls it. */
  void stop();

  // ---- The test thread.
  /** The states the driver has published, and the joint actions it would have
   * executed. */
  uint64_t statesPublished() const { return statesPublished_.load(); }
  uint64_t actionsExecuted() const { return actionsExecuted_.load(); }
  /** How often the torques were switched on, and how often the backend was put
   * in its safe state. */
  uint64_t torqueEnables() const { return torqueEnables_.load(); }
  uint64_t safeStates() const { return safeStates_.load(); }

 private:
  /** RobotHWInterfaceBase's robot side, opened for the driver thread. */
  class Hardware final : public robot::model::RobotHWInterfaceBase {
   public:
    explicit Hardware(robot::model::RobotDescription description)
        : RobotHWInterfaceBase(std::move(description)) {}
    void publish(const robot::model::RobotState& state) {
      publishRobotState(state);
    }
    bool take(robot::model::RobotJointAction& action) {
      return takeJointAction(action);
    }
    void discardApplied() { discardAppliedJointAction(); }
  };

  void runDriver();

  Hardware hardware_;
  const std::chrono::microseconds period_;
  Trajectory trajectory_;
  // The driver thread's own.
  robot::model::RobotState driverState_;
  robot::model::RobotJointAction driverAction_;

  std::atomic<bool> torquesEnabled_{false};
  std::atomic<bool> stopRequested_{false};
  std::thread driver_;
  std::atomic<uint64_t> statesPublished_{0};
  std::atomic<uint64_t> actionsExecuted_{0};
  std::atomic<uint64_t> torqueEnables_{0};
  std::atomic<uint64_t> safeStates_{0};
};

/** The trajectory that holds `initialState` still: its pose and joints, the IMU
 * reading gravity alone. */
FakeRobotBackend::Trajectory standingStill(
    const robot::model::RobotState& initialState);

}  // namespace ocs2::humanoid::robot_test
