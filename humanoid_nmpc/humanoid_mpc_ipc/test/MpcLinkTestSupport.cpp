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

#include "humanoid_nmpc/humanoid_mpc_ipc/test/MpcLinkTestSupport.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"

#include <ocs2_core/control/FeedforwardController.h>
#include <ocs2_mpc/CommandData.h>
#include <ocs2_oc/oc_data/PerformanceIndex.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>

#include "robot_ipc/BusOptions.h"
#include "robot_ipc/NodeEndpoint.h"

namespace ocs2::humanoid::ipc::test_support {

ModelDimensions modelDimensions() {
  return ModelDimensions{.stateDim = kStateDim, .inputDim = kInputDim, .numModes = kNumModes};
}

SystemObservation observationAt(scalar_t time, scalar_t value) {
  SystemObservation observation;
  observation.time = time;
  observation.state = vector_t::Constant(kStateDim, value);
  observation.input = vector_t::Zero(kInputDim);
  return observation;
}

TargetTrajectories resetTargetsFor(const SystemObservation& observation) {
  return TargetTrajectories({observation.time}, {observation.state}, {observation.input});
}

mpc::Settings mpcSettings(scalar_t solutionTimeWindow) {
  mpc::Settings settings;
  settings.timeHorizon_ = 1.0;
  settings.solutionTimeWindow_ = solutionTimeWindow;
  return settings;
}

std::unique_ptr<mpc_test::ScriptedMpc> makeScriptedMpc(const mpc::Settings& settings) {
  std::unique_ptr<mpc_test::ScriptedMpc> mpc = std::make_unique<mpc_test::ScriptedMpc>(settings, kInputDim);
  mpc->solver().setPlan([](scalar_t time, scalar_t initTime, const vector_t& initState) {
    return vector_t(initState + vector_t::Constant(initState.size(), time - initTime));
  });
  return mpc;
}

bool waitFor(const std::function<bool()>& condition, absl::Duration timeout) {
  const absl::Time deadline = absl::Now() + timeout;
  while (absl::Now() < deadline) {
    if (condition()) {
      return true;
    }
    absl::SleepFor(absl::Milliseconds(1));
  }
  return condition();
}

std::unique_ptr<robot::ipc::Bus> createNodeBus(const std::string& name) {
  robot::ipc::BusOptions options;
  options.nodeName = name;
  options.network.nodes = {robot::ipc::NodeEndpoint{.name = name, .host = "127.0.0.1", .port = robot::ipc::kEphemeralPort}};
  absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> bus = robot::ipc::Bus::Create(std::move(options));
  CHECK_OK(bus.status());
  return std::move(*bus);
}

std::unique_ptr<robot::ipc::Bus> createSubscriberBus(const std::string& nodeName, int port) {
  robot::ipc::BusOptions options;
  options.network.nodes = {robot::ipc::NodeEndpoint{.name = nodeName, .host = "127.0.0.1", .port = port}};
  absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> bus = robot::ipc::Bus::Create(std::move(options));
  CHECK_OK(bus.status());
  return std::move(*bus);
}

void connectBoth(robot::ipc::Bus& first, robot::ipc::Bus& second) {
  CHECK_OK(first.connect(second.boundEndpoint()));
  CHECK_OK(second.connect(first.boundEndpoint()));
}

humanoid_mpc_msgs::MpcPolicy makePolicyMessage(scalar_t initTime,
                                               scalar_t finalTime,
                                               scalar_t value,
                                               uint64_t resetsServed,
                                               uint64_t fullResetsServed,
                                               uint64_t solveCount,
                                               size_t nodes) {
  CommandData command;
  command.mpcInitObservation_ = observationAt(initTime, value);
  command.mpcTargetTrajectories_ = resetTargetsFor(command.mpcInitObservation_);
  PrimalSolution solution;
  for (size_t node = 0; node < nodes; ++node) {
    const scalar_t fraction = nodes > 1 ? static_cast<scalar_t>(node) / static_cast<scalar_t>(nodes - 1) : 0.0;
    const scalar_t time = initTime + fraction * (finalTime - initTime);
    solution.timeTrajectory_.push_back(time);
    solution.stateTrajectory_.push_back(vector_t::Constant(kStateDim, value + time - initTime));
    solution.inputTrajectory_.push_back(vector_t::Zero(kInputDim));
  }
  solution.controllerPtr_ = std::make_unique<FeedforwardController>(solution.timeTrajectory_, solution.inputTrajectory_);
  humanoid_mpc_msgs::MpcPolicy message;
  CHECK_OK(policyToProto(command, solution, PerformanceIndex(), &message));
  message.set_resets_served(resetsServed);
  message.set_full_resets_served(fullResetsServed);
  message.mutable_solver_status()->set_healthy(true);
  message.mutable_solver_status()->set_solve_count(solveCount);
  return message;
}

// ---------------------------------------------------------------------------------------------------------------------
// Gate
// ---------------------------------------------------------------------------------------------------------------------

void Gate::close() {
  absl::MutexLock lock(mutex_);
  closed_ = true;
  permits_ = 0;
}

void Gate::open() {
  absl::MutexLock lock(mutex_);
  closed_ = false;
}

void Gate::release(size_t count) {
  absl::MutexLock lock(mutex_);
  permits_ += count;
}

void Gate::pass() {
  absl::MutexLock lock(mutex_);
  ++arrivals_;
  mutex_.Await(absl::Condition(this, &Gate::canPass));
  if (closed_) {
    --permits_;
  }
}

size_t Gate::arrivals() const {
  absl::MutexLock lock(mutex_);
  return arrivals_;
}

bool Gate::canPass() const {
  return !closed_ || permits_ > 0;
}

}  // namespace ocs2::humanoid::ipc::test_support
