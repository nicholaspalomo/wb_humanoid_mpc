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

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "ocs2_core/control/FeedforwardController.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_mpc/CommandData.h"
#include "ocs2_oc/multiple_shooting/Helpers.h"
#include "ocs2_oc/oc_data/PerformanceIndex.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_oc/oc_data/TimeDiscretization.h"

#include "humanoid_nmpc/humanoid_mpc_ipc/test/PolicyTestData.h"
#include "robot_ipc/BusOptions.h"
#include "robot_runtime/robot_ipc/test/TestNetwork.h"

namespace ocs2::humanoid::ipc::test_support {

ModelDimensions modelDimensions() {
  return ModelDimensions{
      .stateDim = kStateDim, .inputDim = kInputDim, .numModes = kNumModes};
}

SystemObservation observationAt(scalar_t time, scalar_t value) {
  SystemObservation observation;
  observation.time = time;
  observation.state = vector_t::Constant(kStateDim, value);
  observation.input = vector_t::Zero(kInputDim);
  return observation;
}

TargetTrajectories resetTargetsFor(const SystemObservation& observation) {
  return TargetTrajectories({observation.time}, {observation.state},
                            {observation.input});
}

mpc::Settings mpcSettings(scalar_t solutionTimeWindow) {
  mpc::Settings settings;
  settings.timeHorizon_ = 1.0;
  settings.solutionTimeWindow_ = solutionTimeWindow;
  return settings;
}

std::unique_ptr<mpc_test::ScriptedMpc> makeScriptedMpc(
    const mpc::Settings& settings) {
  std::unique_ptr<mpc_test::ScriptedMpc> mpc =
      std::make_unique<mpc_test::ScriptedMpc>(settings, kInputDim);
  mpc->solver().setPlan(
      [](scalar_t time, scalar_t initTime, const vector_t& initState) {
        return vector_t(initState +
                        vector_t::Constant(initState.size(), time - initTime));
      });
  return mpc;
}

ModelDimensions manifoldModelDimensions() {
  return ModelDimensions{
      .stateDim = test_data::toyStateManifold()->getAmbientDim(),
      .inputDim = kInputDim,
      .numModes = kNumModes};
}

SystemObservation manifoldObservationAt(scalar_t time, scalar_t value) {
  // [E(3), Q, E(5)]: the quaternion (x, y, z, w) at offset 3.
  SystemObservation observation;
  observation.time = time;
  observation.state = vector_t::Constant(
      static_cast<Eigen::Index>(manifoldModelDimensions().stateDim), value);
  rotation_vector_t<scalar_t> rotation;
  rotation << 0.6 * value, -0.48 * value, 0.64 * value;
  observation.state.segment<4>(3) = quaternionExp(rotation);
  observation.input = vector_t::Zero(kInputDim);
  return observation;
}

ManifoldScriptedSolver::ManifoldScriptedSolver(
    std::shared_ptr<const StateManifold> stateManifold, size_t inputDim)
    : stateManifold_(std::move(stateManifold)), inputDim_(inputDim) {}

void ManifoldScriptedSolver::reset() {
  numResets_.fetch_add(1);
  finalTime_ = 0.0;
}

void ManifoldScriptedSolver::getPrimalSolution(
    scalar_t finalTime, PrimalSolution* absl_nonnull primalSolutionPtr) const {
  // Ten intervals, the pre- and post-event nodes of the event halfway sharing
  // its time, as SqpSolver discretizes.
  constexpr size_t kIntervals = 10;
  const scalar_t endTime = std::max(finalTime, initTime_);
  const scalar_t eventTime = initTime_ + 0.5 * (endTime - initTime_);
  std::vector<AnnotatedTime> times;
  for (size_t k = 0; k <= kIntervals; ++k) {
    if (k == kIntervals / 2) {
      times.emplace_back(eventTime, AnnotatedTime::Event::PreEvent);
      times.emplace_back(eventTime, AnnotatedTime::Event::PostEvent);
    } else {
      times.emplace_back(initTime_ + (endTime - initTime_) *
                                         static_cast<scalar_t>(k) /
                                         static_cast<scalar_t>(kIntervals));
    }
  }

  const Eigen::Index tangentDim =
      static_cast<Eigen::Index>(stateManifold_->getTangentDim());
  const vector_t tangentVelocity =
      vector_t::LinSpaced(tangentDim, /*low=*/-0.5, /*high=*/0.5);
  const vector_t jump = vector_t::Constant(tangentDim, 0.05);
  vector_array_t states;
  bool afterEvent = false;
  for (const AnnotatedTime& time : times) {
    afterEvent = afterEvent || time.event == AnnotatedTime::Event::PostEvent;
    vector_t step = (time.time - initTime_) * tangentVelocity;
    if (afterEvent) step += jump;
    vector_t state;
    stateManifold_->retract(initState_, step, /*alpha=*/1.0, state);
    states.push_back(std::move(state));
  }
  // An input and a gain for every node but the last, as the QP gives them.
  vector_array_t inputs;
  matrix_array_t gains;
  for (size_t k = 0; k + 1 < times.size(); ++k) {
    const scalar_t node = static_cast<scalar_t>(k);
    inputs.push_back(vector_t::LinSpaced(static_cast<Eigen::Index>(inputDim_),
                                         initState_(0), initState_(0) + node));
    matrix_t gain(static_cast<Eigen::Index>(inputDim_), tangentDim);
    for (Eigen::Index column = 0; column < gain.cols(); ++column) {
      for (Eigen::Index row = 0; row < gain.rows(); ++row) {
        gain(row, column) = 0.01 * static_cast<scalar_t>(row + 1) -
                            0.02 * static_cast<scalar_t>(column) +
                            0.1 * initState_(0) + 0.003 * node;
      }
    }
    gains.push_back(std::move(gain));
  }
  ModeSchedule modeSchedule(/*eventTimesInput=*/{eventTime},
                            /*modeSequenceInput=*/{0, 1});
  *primalSolutionPtr = multiple_shooting::toPrimalSolution(
      times, std::move(modeSchedule), std::move(states), std::move(inputs),
      std::move(gains), stateManifold_);
}

void ManifoldScriptedSolver::runImpl(scalar_t initTime,
                                     const vector_t& initState,
                                     scalar_t finalTime) {
  initTime_ = initTime;
  initState_ = initState;
  finalTime_ = finalTime;
}

void ManifoldScriptedSolver::runImpl(
    scalar_t initTime,
    const vector_t& initState,
    scalar_t finalTime,
    const ControllerBase* absl_nullable /*externalControllerPtr*/) {
  runImpl(initTime, initState, finalTime);
}

void ManifoldScriptedSolver::runImpl(scalar_t initTime,
                                     const vector_t& initState,
                                     scalar_t finalTime,
                                     const PrimalSolution& /*primalSolution*/) {
  runImpl(initTime, initState, finalTime);
}

std::unique_ptr<ManifoldScriptedMpc> makeManifoldScriptedMpc(
    std::shared_ptr<const StateManifold> stateManifold) {
  return std::make_unique<ManifoldScriptedMpc>(
      mpcSettings(), std::move(stateManifold), kInputDim);
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

std::unique_ptr<robot::ipc::Bus> createNodeBus(const std::string& name,
                                               int domainIndex) {
  robot::ipc::BusOptions options;
  options.nodeName = name;
  options.network = robot::ipc::isolatedTestNetwork({name}, domainIndex);
  absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> bus =
      robot::ipc::Bus::Create(std::move(options));
  CHECK_OK(bus.status());
  return std::move(*bus);
}

std::unique_ptr<robot::ipc::Bus> createSubscriberBus(int domainIndex) {
  robot::ipc::BusOptions options;
  options.network =
      robot::ipc::isolatedTestNetwork({"subscriber"}, domainIndex);
  absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> bus =
      robot::ipc::Bus::Create(std::move(options));
  CHECK_OK(bus.status());
  return std::move(*bus);
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
    const scalar_t fraction = nodes > 1 ? static_cast<scalar_t>(node) /
                                              static_cast<scalar_t>(nodes - 1)
                                        : 0.0;
    const scalar_t time = initTime + fraction * (finalTime - initTime);
    solution.timeTrajectory_.push_back(time);
    solution.stateTrajectory_.push_back(
        vector_t::Constant(kStateDim, value + time - initTime));
    solution.inputTrajectory_.push_back(vector_t::Zero(kInputDim));
  }
  solution.controllerPtr_ = std::make_unique<FeedforwardController>(
      solution.timeTrajectory_, solution.inputTrajectory_);
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
