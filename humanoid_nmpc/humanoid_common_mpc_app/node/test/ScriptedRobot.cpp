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

#include "humanoid_nmpc/humanoid_common_mpc_app/node/test/ScriptedRobot.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_mpc_ipc/MpcMessageConversions.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "robot_ipc/BusOptions.h"
#include "robot_ipc/Delivery.h"
#include "robot_ipc/NodeEndpoint.h"

namespace ocs2::humanoid::node::test_support {
namespace {

namespace topics = ::ocs2::humanoid::ipc::topics;

std::unique_ptr<robot::ipc::Bus> createLoopbackBus(const std::string& name) {
  robot::ipc::BusOptions options;
  options.nodeName = name;
  options.network.nodes = {robot::ipc::NodeEndpoint{.name = name, .host = "127.0.0.1", .port = robot::ipc::kEphemeralPort, .bindHost = ""}};
  absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> bus = robot::ipc::Bus::Create(std::move(options));
  CHECK_OK(bus.status());
  return *std::move(bus);
}

}  // namespace

bool waitFor(const std::function<bool()>& condition, double timeoutSeconds) {
  const absl::Time deadline = absl::Now() + absl::Seconds(timeoutSeconds);
  while (!condition()) {
    if (absl::Now() > deadline) return false;
    absl::SleepFor(absl::Milliseconds(1));
  }
  return true;
}

ScriptedRobot::ScriptedRobot() : robot_(createLoopbackBus("robot")), operator_(createLoopbackBus("operator")) {}

ScriptedRobot::~ScriptedRobot() {
  robot_->stop();
  operator_->stop();
}

void ScriptedRobot::connect(robot::ipc::Bus& mpcBus) {
  CHECK_OK(mpcBus.connect(robot_->boundEndpoint()));
  CHECK_OK(mpcBus.connect(operator_->boundEndpoint()));
  CHECK_OK(robot_->connect(mpcBus.boundEndpoint()));
}

absl::Status ScriptedRobot::start() {
  const std::function<void(const humanoid_mpc_msgs::MpcPolicy&)> onPolicy = [this](const humanoid_mpc_msgs::MpcPolicy& policy) {
    absl::MutexLock lock(mutex_);
    policies_.push_back(policy);
  };
  RETURN_IF_ERROR(robot_->subscribe<humanoid_mpc_msgs::MpcPolicy>(topics::kMpcPolicy, robot::ipc::Delivery::kAll, onPolicy));
  RETURN_IF_ERROR(robot_->start());
  return operator_->start();
}

humanoid_mpc_msgs::MpcObservation ScriptedRobot::observationMessage(const SystemObservation& observation,
                                                                    uint64_t sequence,
                                                                    uint64_t requested,
                                                                    uint64_t fullRequested) {
  humanoid_mpc_msgs::MpcObservation message;
  ipc::toProto(observation, message.mutable_observation());
  message.mutable_resets()->set_requested(requested);
  message.mutable_resets()->set_full_requested(fullRequested);
  message.set_sequence(sequence);
  return message;
}

std::optional<humanoid_mpc_msgs::MpcPolicy> ScriptedRobot::solve(const humanoid_mpc_msgs::MpcObservation& observation,
                                                                 double timeoutSeconds) {
  const double time = observation.observation().time();
  absl::Time nextSend = absl::InfinitePast();
  const bool arrived = waitFor(
      [&]() {
        // Resent every 10 ms: the server skips a repeated sequence number, so a resend is never solved twice.
        if (absl::Now() >= nextSend) {
          robot_->publish(topics::kRobotMpcObservation, observation).IgnoreError();
          nextSend = absl::Now() + absl::Milliseconds(10);
        }
        return policySolvedAt(time).has_value();
      },
      timeoutSeconds);
  if (!arrived) return std::nullopt;
  return policySolvedAt(time);
}

bool ScriptedRobot::sendAsOperator(absl::string_view topic,
                                   const google::protobuf::Message& message,
                                   const std::function<bool()>& received) {
  absl::Time nextSend = absl::InfinitePast();
  return waitFor([&]() {
    if (absl::Now() >= nextSend) {
      operator_->publish(topic, message).IgnoreError();
      nextSend = absl::Now() + absl::Milliseconds(10);
    }
    return received();
  });
}

std::optional<humanoid_mpc_msgs::MpcPolicy> ScriptedRobot::policySolvedAt(double time) const {
  absl::MutexLock lock(mutex_);
  for (size_t index = policies_.size(); index-- > 0;) {
    if (policies_[index].init_observation().time() == time) return policies_[index];
  }
  return std::nullopt;
}

size_t ScriptedRobot::numPolicies() const {
  absl::MutexLock lock(mutex_);
  return policies_.size();
}

}  // namespace ocs2::humanoid::node::test_support
