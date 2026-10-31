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

#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/FakeRobotBackend.h"

#include <array>
#include <chrono>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"

#include "humanoid_common_mpc_app/robot/RobotSupervisorRegistry.h"

namespace ocs2::humanoid::robot_test {
namespace {

/** [m/s^2] What an accelerometer at rest reads along the world vertical. */
constexpr double kGravity = 9.81;

}  // namespace

FakeRobotBackend::Trajectory standingStill(
    const robot::model::RobotState& initialState) {
  return [initialState](double time, robot::model::RobotState& state) {
    state = initialState;
    state.setTime(time);
    state.setLinvelBaseRtWorldEwrtBase(vector3_t::Zero());
    state.setAngvelBaseRtWorldEwrtBase(vector3_t::Zero());
    state.setSpecforceBaseRtWorldEwrtBase(
        initialState.getQuatBaseRtWorld().inverse() *
        vector3_t(0.0, 0.0, kGravity));
  };
}

FakeRobotBackend::FakeRobotBackend(robot::model::RobotDescription description,
                                   const robot::model::RobotState& initialState,
                                   std::chrono::microseconds period)
    : hardware_(std::move(description)),
      period_(period),
      trajectory_(standingStill(initialState)),
      driverState_(initialState),
      driverAction_(hardware_.getRobotDescription()) {}

FakeRobotBackend::~FakeRobotBackend() {
  stop();
}

absl::Status FakeRobotBackend::initialize() {
  trajectory_(/*time=*/0.0, driverState_);
  hardware_.publish(driverState_);
  return absl::OkStatus();
}

absl::Status FakeRobotBackend::start(const std::vector<int>& /*cores*/) {
  if (driver_.joinable()) return absl::OkStatus();
  stopRequested_.store(false);
  driver_ = std::thread([this]() { runDriver(); });
  return absl::OkStatus();
}

void FakeRobotBackend::stop() {
  stopRequested_.store(true);
  if (driver_.joinable()) driver_.join();
}

void FakeRobotBackend::enableTorques() {
  if (!torquesEnabled_.exchange(true)) {
    // No action applied while the torques were off is executed
    // (RobotHWInterfaceBase::discardAppliedJointAction()).
    hardware_.discardApplied();
    torqueEnables_.fetch_add(1);
  }
}

void FakeRobotBackend::enterSafeState() {
  torquesEnabled_.store(false);
  safeStates_.fetch_add(1);
}

void FakeRobotBackend::readMeasuredContactForces(
    std::array<vector3_t, kNumContacts>& forces) {
  forces.fill(vector3_t::Zero());
}

absl::string_view FakeRobotBackend::supervisorName() const {
  return kHardwareSupervisorName;
}

void FakeRobotBackend::runDriver() {
  const double periodSeconds = std::chrono::duration<double>(period_).count();
  uint64_t step = 0;
  std::chrono::steady_clock::time_point next = std::chrono::steady_clock::now();
  while (!stopRequested_.load()) {
    next += period_;
    std::this_thread::sleep_until(next);
    ++step;
    trajectory_(static_cast<double>(step) * periodSeconds, driverState_);
    hardware_.publish(driverState_);
    statesPublished_.fetch_add(1);
    if (hardware_.take(driverAction_) && torquesEnabled_.load())
      actionsExecuted_.fetch_add(1);
  }
}

}  // namespace ocs2::humanoid::robot_test
