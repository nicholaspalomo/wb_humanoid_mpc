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

#include "humanoid_common_mpc_app/robot/test_support/ScriptedOperator.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "humanoid_common_mpc_app/robot/test_support/LoopbackNetwork.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/fsm_command.pb.h"
#include "humanoid_mpc_msgs/visualization_scene.pb.h"
#include "robot_ipc/Delivery.h"

namespace ocs2::humanoid::test_support {

namespace topics = ::ocs2::humanoid::ipc::topics;

ScriptedOperator::ScriptedOperator(const robot::ipc::NetworkConfig& network) : bus_(createBus(network, "operator")) {
  CHECK_OK(bus_->subscribe<humanoid_mpc_msgs::FsmState>(topics::kRobotFsmState, robot::ipc::Delivery::kLatest,
                                                        [this](const humanoid_mpc_msgs::FsmState& state) {
                                                          absl::MutexLock lock(mutex_);
                                                          fsmState_ = state;
                                                        }));
  CHECK_OK(bus_->subscribe<humanoid_mpc_msgs::RobotStateSample>(topics::kRobotState, robot::ipc::Delivery::kAll,
                                                                [this](const humanoid_mpc_msgs::RobotStateSample& sample) {
                                                                  absl::MutexLock lock(mutex_);
                                                                  sample_ = sample;
                                                                  const double height = sample.base_position_world().z();
                                                                  baseHeightRange_.min = std::min(baseHeightRange_.min, height);
                                                                  baseHeightRange_.max = std::max(baseHeightRange_.max, height);
                                                                  ++baseHeightRange_.samples;
                                                                }));
  CHECK_OK(bus_->subscribe<humanoid_mpc_msgs::LoopTiming>(topics::kRobotLoopTiming, robot::ipc::Delivery::kLatest,
                                                          [this](const humanoid_mpc_msgs::LoopTiming& timing) {
                                                            absl::MutexLock lock(mutex_);
                                                            timing_ = timing;
                                                          }));
  CHECK_OK(bus_->subscribe<humanoid_mpc_msgs::MpcStatus>(topics::kMpcStatus, robot::ipc::Delivery::kLatest,
                                                         [this](const humanoid_mpc_msgs::MpcStatus& status) {
                                                           absl::MutexLock lock(mutex_);
                                                           mpcStatus_ = status;
                                                         }));
  CHECK_OK(bus_->subscribe<humanoid_mpc_msgs::VisualizationScene>(topics::kVizScene, robot::ipc::Delivery::kAll,
                                                                  [this](const humanoid_mpc_msgs::VisualizationScene& /*scene*/) {
                                                                    absl::MutexLock lock(mutex_);
                                                                    ++scenesReceived_;
                                                                  }));
}

ScriptedOperator::~ScriptedOperator() {
  stop();
}

void ScriptedOperator::start() {
  CHECK_OK(bus_->start());
}

void ScriptedOperator::stop() {
  bus_->stop();
}

std::optional<humanoid_mpc_msgs::FsmState> ScriptedOperator::fsmState() const {
  absl::MutexLock lock(mutex_);
  return fsmState_;
}

std::optional<humanoid_mpc_msgs::RobotStateSample> ScriptedOperator::sample() const {
  absl::MutexLock lock(mutex_);
  return sample_;
}

std::optional<humanoid_mpc_msgs::LoopTiming> ScriptedOperator::timing() const {
  absl::MutexLock lock(mutex_);
  return timing_;
}

std::optional<humanoid_mpc_msgs::MpcStatus> ScriptedOperator::mpcStatus() const {
  absl::MutexLock lock(mutex_);
  return mpcStatus_;
}

uint64_t ScriptedOperator::scenesReceived() const {
  absl::MutexLock lock(mutex_);
  return scenesReceived_;
}

bool ScriptedOperator::sendFsmCommand(const std::string& command,
                                      const std::function<bool(const humanoid_mpc_msgs::FsmState&)>& reached,
                                      absl::Duration timeout) {
  humanoid_mpc_msgs::FsmCommand message;
  message.set_command(command);
  message.set_sequence(++sequence_);
  return waitFor(
      [&]() {
        bus_->publish(topics::kOperatorFsmCommand, message).IgnoreError();
        absl::SleepFor(absl::Milliseconds(50));
        const std::optional<humanoid_mpc_msgs::FsmState> state = fsmState();
        return state.has_value() && reached(*state);
      },
      timeout);
}

bool ScriptedOperator::enterMode(const std::string& mode) {
  return sendFsmCommand(mode, [&mode](const humanoid_mpc_msgs::FsmState& state) { return state.mode() == mode; });
}

void ScriptedOperator::sendVelocityCommand(const humanoid_mpc_msgs::WalkingVelocityCommand& command, absl::Duration duration) {
  const absl::Time end = absl::Now() + duration;
  while (absl::Now() < end) {
    bus_->publish(topics::kOperatorWalkingVelocityCommand, command).IgnoreError();
    absl::SleepFor(absl::Milliseconds(40));
  }
}

void ScriptedOperator::resetBaseHeightRange() {
  absl::MutexLock lock(mutex_);
  baseHeightRange_ = BaseHeightRange();
}

BaseHeightRange ScriptedOperator::baseHeightRange() const {
  absl::MutexLock lock(mutex_);
  return baseHeightRange_;
}

}  // namespace ocs2::humanoid::test_support
