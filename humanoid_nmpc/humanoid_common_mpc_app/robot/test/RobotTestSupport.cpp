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

#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/RobotTestSupport.h"

#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/mrt/ControlMode.h"

namespace ocs2::humanoid::robot_test {

robot::model::RobotDescription atlasDescription() {
  absl::StatusOr<robot::model::RobotDescription> description = robot::model::RobotDescription::Create(kAtlasUrdf);
  CHECK_OK(description.status());
  return *std::move(description);
}

std::unique_ptr<robot::mujoco_sim_interface::MujocoSimInterface> makeHeadlessAtlas(bool gantryLocked) {
  robot::mujoco_sim_interface::MujocoSimConfig config;
  config.scenePath = kAtlasScene;
  config.headless = true;
  config.isGantryLocked = gantryLocked;
  config.gantryHold = "weld_constraint";
  absl::StatusOr<std::unique_ptr<robot::mujoco_sim_interface::MujocoSimInterface>> sim =
      robot::mujoco_sim_interface::MujocoSimInterface::Create(config, kAtlasUrdf);
  CHECK_OK(sim.status());
  return *std::move(sim);
}

std::unique_ptr<robot::ipc::Bus> createLoopbackBus(const std::string& nodeName) {
  robot::ipc::BusOptions options;
  options.nodeName = nodeName;
  options.network.nodes = {
      robot::ipc::NodeEndpoint{.name = nodeName, .host = "127.0.0.1", .port = robot::ipc::kEphemeralPort, .bindHost = ""}};
  options.ioPollPeriod = absl::Milliseconds(5);
  absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> bus = robot::ipc::Bus::Create(std::move(options));
  CHECK_OK(bus.status());
  return *std::move(bus);
}

void connectBoth(robot::ipc::Bus& first, robot::ipc::Bus& second) {
  CHECK_OK(first.connect(second.boundEndpoint()));
  CHECK_OK(second.connect(first.boundEndpoint()));
}

bool waitFor(const std::function<bool()>& condition, absl::Duration timeout) {
  const absl::Time deadline = absl::Now() + timeout;
  while (!condition()) {
    if (absl::Now() > deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

void ScriptedRobotController::prepareCycle(absl::string_view controlMode, const std::vector<scalar_t>& nominalJointPositions) {
  absl::MutexLock lock(mutex_);
  mode_.assign(controlMode.data(), controlMode.size());
  nominal_ = nominalJointPositions;
}

void ScriptedRobotController::throwInNextCycle(const std::string& message) {
  throwMessage_ = message;
  throwInNextCycle_.store(true);
}

void ScriptedRobotController::computeJointControlAction(const robot::model::RobotState& robotState,
                                                        robot::model::RobotJointAction& jointAction) {
  if (throwInNextCycle_.exchange(false)) {
    throw std::runtime_error(throwMessage_);
  }
  {
    absl::MutexLock lock(mutex_);
    const bool zeroTorque = mode_ == control_mode::kZeroTorque;
    for (size_t joint = 0; joint < nominal_.size(); ++joint) {
      std::optional<robot::model::JointAction>& action = jointAction.at(joint);
      if (!action.has_value()) continue;
      action->q_des = nominal_[joint];
      action->qd_des = 0.0;
      action->kp = zeroTorque ? 0.0 : 100.0;
      action->kd = zeroTorque ? 0.0 : 5.0;
      action->feed_forward_effort = 0.0;
    }
  }
  if (contactEstimator_ != nullptr) {
    contactEstimator_->estimateContactFlags(robotState, estimatedFlags_);
    for (size_t contact = 0; contact < kNumContacts && contact < estimatedFlags_.size(); ++contact) {
      measuredContactFlags_[contact] = estimatedFlags_[contact];
    }
  }
  cycles_.fetch_add(1);
}

std::optional<contact_flag_t> ScriptedRobotController::plannedContactFlags() const {
  return std::nullopt;
}

void ScriptedRobotController::setContactEstimator(std::shared_ptr<robot::model::ContactEstimator> contactEstimator) {
  contactEstimator_ = std::move(contactEstimator);
  absl::MutexLock lock(mutex_);
  contactEstimatorName_ = contactEstimator_ != nullptr ? contactEstimator_->getName() : std::string();
}

void ScriptedRobotController::setContactWrenchGateConfig(const ContactWrenchGate::Config& config) {
  absl::MutexLock lock(mutex_);
  gate_ = config;
}

absl::Status ScriptedRobotController::setPdGains(const mpc_config::JointPdGainsFile& /*gains*/) {
  pdGainsDocuments_.fetch_add(1);
  return absl::OkStatus();
}

std::string ScriptedRobotController::mode() const {
  absl::MutexLock lock(mutex_);
  return mode_;
}

std::string ScriptedRobotController::contactEstimatorName() const {
  absl::MutexLock lock(mutex_);
  return contactEstimatorName_;
}

ContactWrenchGate::Config ScriptedRobotController::gate() const {
  absl::MutexLock lock(mutex_);
  return gate_;
}

std::vector<scalar_t> ScriptedRobotController::nominal() const {
  absl::MutexLock lock(mutex_);
  return nominal_;
}

}  // namespace ocs2::humanoid::robot_test
