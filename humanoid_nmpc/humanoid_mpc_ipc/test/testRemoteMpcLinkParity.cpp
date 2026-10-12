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

// Parity of the two MPC links: the same scripted MPC, driven through the same script of observations, reset requests
// and solver failures, once in process (MPC_MRT_Interface with the solve loop of the MRT joint controllers'
// solverWorker()) and once through the network link (RemoteMpcLink and MpcServer over loopback). After every control
// cycle the controller must see the same thing: the same policy in use, bit for bit, evaluating to the same input, and
// the same answers from the predicates it decides on. Transport timing is taken out by waiting, in each cycle, for the
// policy solved from that cycle's observation. The script runs on a flat state with a feedforward policy, and on a state
// manifold with the ManifoldLinearController of an SQP with an event, the MRT of either side given the manifold; there
// the robot's RealtimePolicyEvaluator on the policy it received computes what the in-process MRT_BASE::evaluatePolicy()
// does, on both sides of the event.

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/control/ControllerType.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_mpc/MPC_BASE.h"
#include "ocs2_mpc/MPC_MRT_Interface.h"
#include "ocs2_mpc/MRT_BASE.h"

#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"
#include "humanoid_mpc_ipc/MpcServer.h"
#include "humanoid_mpc_ipc/RealtimePolicyEvaluator.h"
#include "humanoid_mpc_ipc/RemoteMpcLink.h"
#include "humanoid_nmpc/humanoid_mpc_ipc/test/MpcLinkTestSupport.h"
#include "humanoid_nmpc/humanoid_mpc_ipc/test/PolicyTestData.h"
#include "robot_ipc/Bus.h"

namespace ocs2::humanoid::ipc {
namespace {

using test_support::resetTargetsFor;
using test_support::waitFor;
using ResetKind = MpcResetSupervisor::ResetKind;

/** One control cycle of the script. */
struct Step {
  scalar_t time = 0.0;
  scalar_t value = 0.0;
  /** Requested from the controller's supervisor before the cycle's observation is written, in this order. */
  std::vector<ResetKind> resets = {};
  /** The next solve throws, as a solver that fails its QP does; the solve loop retries at once. */
  bool failNextSolve = false;
};

std::vector<Step> script() {
  return {
      {.time = 1.00, .value = 1.0},
      {.time = 1.01, .value = 1.5},
      {.time = 1.02, .value = 2.0, .resets = {ResetKind::kFull}},
      {.time = 1.03, .value = 2.5},
      {.time = 1.04, .value = 3.0, .resets = {ResetKind::kSolver}},
      {.time = 1.05, .value = 3.5, .failNextSolve = true},
      {.time = 1.06, .value = 4.0},
      {.time = 1.07, .value = 4.5, .resets = {ResetKind::kSolver, ResetKind::kFull}},
      {.time = 1.08, .value = 5.0},
      {.time = 1.09, .value = 5.5, .resets = {ResetKind::kFull}, .failNextSolve = true},
      {.time = 1.10, .value = 6.0},
      {.time = 1.11, .value = 6.5, .resets = {ResetKind::kSolver}},
  };
}

/** The model a script runs on, the same in process and over the network. */
struct ParityModel {
  ModelDimensions dimensions;
  /** Null for a flat state; else given to both MRTs, and to the realtime evaluator. */
  std::shared_ptr<const StateManifold> stateManifold;
  /** The observation of a step at `time` with state `value`. */
  std::function<SystemObservation(scalar_t time, scalar_t value)> observationAt;
};

ParityModel flatModel() {
  return {.dimensions = test_support::modelDimensions(), .stateManifold = nullptr, .observationAt = test_support::observationAt};
}

ParityModel manifoldModel() {
  return {.dimensions = test_support::manifoldModelDimensions(),
          .stateManifold = test_data::toyStateManifold(),
          .observationAt = test_support::manifoldObservationAt};
}

/** What the controller sees after a control cycle. */
struct Seen {
  /** isActivePolicyCurrent() && !hasOutstandingReset() right after the cycle's reset requests, before its policy. */
  bool postResetPolicyActiveAfterRequests = false;
  bool initialPolicyReceived = false;
  bool postResetPolicyActive = false;
  bool activePolicyCurrent = false;
  bool outstandingReset = false;
  bool healthy = false;
  SystemObservation initObservation;
  TargetTrajectories targets;
  scalar_array_t times;
  vector_array_t states;
  vector_array_t inputs;
  size_array_t postEventIndices;
  /** evaluatePolicy() a control cycle ahead, and half a horizon ahead and just around it (the manifold policy's event). */
  vector_array_t evaluatedStates;
  vector_array_t evaluatedInputs;
  std::vector<size_t> evaluatedModes;
};

Seen see(MRT_BASE& mrt, const MpcResetSupervisor& supervisor, const ParityModel& model, const SystemObservation& observation) {
  Seen seen;
  seen.initialPolicyReceived = mrt.initialPolicyReceived();
  seen.activePolicyCurrent = mrt.isActivePolicyCurrent();
  seen.outstandingReset = supervisor.hasOutstandingReset();
  seen.postResetPolicyActive = seen.activePolicyCurrent && !seen.outstandingReset;
  seen.healthy = supervisor.isHealthy();
  seen.initObservation = mrt.getCommand().mpcInitObservation_;
  seen.targets = mrt.getCommand().mpcTargetTrajectories_;
  seen.times = mrt.getPolicy().timeTrajectory_;
  seen.states = mrt.getPolicy().stateTrajectory_;
  seen.inputs = mrt.getPolicy().inputTrajectory_;
  seen.postEventIndices = mrt.getPolicy().postEventIndices_;
  // The robot's realtime evaluation of the same policy: the doubles of evaluatePolicy(), on either side.
  RealtimePolicyEvaluator evaluator(model.dimensions, model.stateManifold.get());
  for (const scalar_t lookahead : {0.001, 0.499, 0.5, 0.501}) {
    vector_t state;
    vector_t input;
    size_t mode = 0;
    const scalar_t time = observation.time + lookahead;
    mrt.evaluatePolicy(time, observation.state, state, input, mode);
    seen.evaluatedStates.push_back(state);
    seen.evaluatedInputs.push_back(input);
    seen.evaluatedModes.push_back(mode);
    vector_t realtimeState;
    vector_t realtimeInput;
    size_t realtimeMode = 0;
    EXPECT_EQ(evaluator.evaluate(mrt.getPolicy(), time, observation.state, realtimeState, realtimeInput, realtimeMode),
              RealtimePolicyEvaluator::Outcome::kEvaluated)
        << "t = " << time;
    EXPECT_EQ(realtimeState, state) << "t = " << time;
    EXPECT_EQ(realtimeInput, input) << "t = " << time;
    EXPECT_EQ(realtimeMode, mode) << "t = " << time;
  }
  return seen;
}

void expectSame(const Seen& inProcess, const Seen& remote, size_t step) {
  SCOPED_TRACE(::testing::Message() << "control cycle " << step);
  EXPECT_EQ(remote.postResetPolicyActiveAfterRequests, inProcess.postResetPolicyActiveAfterRequests);
  EXPECT_EQ(remote.initialPolicyReceived, inProcess.initialPolicyReceived);
  EXPECT_EQ(remote.postResetPolicyActive, inProcess.postResetPolicyActive);
  EXPECT_EQ(remote.activePolicyCurrent, inProcess.activePolicyCurrent);
  EXPECT_EQ(remote.outstandingReset, inProcess.outstandingReset);
  EXPECT_EQ(remote.healthy, inProcess.healthy);
  EXPECT_EQ(remote.initObservation.time, inProcess.initObservation.time);
  EXPECT_EQ(remote.initObservation.state, inProcess.initObservation.state);
  EXPECT_EQ(remote.initObservation.mode, inProcess.initObservation.mode);
  EXPECT_EQ(remote.targets.timeTrajectory, inProcess.targets.timeTrajectory);
  EXPECT_EQ(remote.targets.stateTrajectory, inProcess.targets.stateTrajectory);
  EXPECT_EQ(remote.targets.inputTrajectory, inProcess.targets.inputTrajectory);
  EXPECT_EQ(remote.times, inProcess.times);
  EXPECT_EQ(remote.states, inProcess.states);
  EXPECT_EQ(remote.inputs, inProcess.inputs);
  EXPECT_EQ(remote.postEventIndices, inProcess.postEventIndices);
  EXPECT_EQ(remote.evaluatedStates, inProcess.evaluatedStates);
  EXPECT_EQ(remote.evaluatedInputs, inProcess.evaluatedInputs);
  EXPECT_EQ(remote.evaluatedModes, inProcess.evaluatedModes);
}

/**
 * The controller with the solver in process: MPC_MRT_Interface on `mpc`, served as solverWorker() serves it.
 * `failNextSolve` makes the next solve of `mpc` fail; the script asks for it only of an MPC that can fail.
 */
std::vector<Seen> runInProcess(const std::vector<Step>& steps,
                               const ParityModel& model,
                               MPC_BASE& mpc,
                               const std::function<void()>& failNextSolve) {
  MPC_MRT_Interface mrt(mpc);
  mrt.setStateManifold(model.stateManifold);
  MpcResetSupervisor supervisor;
  std::vector<Seen> seen;
  for (size_t index = 0; index < steps.size(); ++index) {
    const Step& step = steps[index];
    for (const ResetKind kind : step.resets) supervisor.requestReset(kind);
    const bool afterRequests = mrt.isActivePolicyCurrent() && !supervisor.hasOutstandingReset();
    const SystemObservation observation = model.observationAt(step.time, step.value);
    mrt.setCurrentObservation(observation);
    if (index == 0) {
      // solverWorker()'s start-up reset, from the first observation.
      mrt.resetMpcNode(resetTargetsFor(mrt.getCurrentObservation()));
    }
    if (step.failNextSolve) failNextSolve();
    // The iterations of the solve loop until one succeeds: a failed solve is retried at once.
    for (int iteration = 0; iteration < 5; ++iteration) {
      if (const std::optional<MpcResetSupervisor::ResetTicket> ticket = supervisor.takeResetRequest()) {
        const SystemObservation current = mrt.getCurrentObservation();
        if (ticket->full) {
          mrt.resetMpcNode(resetTargetsFor(current));
        } else {
          mrt.resetMpcSolver(resetTargetsFor(current));
        }
        supervisor.completeReset(*ticket);
      }
      const absl::Status status = mrt.advanceMpc();
      const std::chrono::duration<scalar_t> retryDelay = supervisor.onSolveResult(status);
      EXPECT_EQ(retryDelay.count(), 0.0) << "the script never fails long enough for a back-off";
      if (status.ok()) break;
    }
    mrt.updatePolicy();
    seen.push_back(see(mrt, supervisor, model, observation));
    seen.back().postResetPolicyActiveAfterRequests = afterRequests;
  }
  return seen;
}

/** The controller with the solver behind the network link: `mpc` served by an MpcServer, as in runInProcess(). */
std::vector<Seen> runRemote(const std::vector<Step>& steps,
                            const ParityModel& model,
                            MPC_BASE& mpc,
                            const std::function<void()>& failNextSolve) {
  std::unique_ptr<robot::ipc::Bus> robotBus = test_support::createNodeBus("robot");
  std::unique_ptr<robot::ipc::Bus> mpcBus = test_support::createNodeBus("mpc");
  test_support::connectBoth(*robotBus, *mpcBus);
  MpcResetSupervisor supervisor;
  RemoteMpcLink::Config linkConfig;
  linkConfig.dimensions = model.dimensions;
  linkConfig.stateManifold = model.stateManifold;
  absl::StatusOr<std::unique_ptr<RemoteMpcLink>> link = RemoteMpcLink::Create(*robotBus, supervisor, linkConfig);
  CHECK_OK(link.status());
  MpcServer::Config serverConfig;
  serverConfig.dimensions = model.dimensions;
  absl::StatusOr<std::unique_ptr<MpcServer>> server = MpcServer::Create(*mpcBus, mpc, resetTargetsFor, serverConfig);
  CHECK_OK(server.status());
  CHECK_OK(robotBus->start());
  CHECK_OK(mpcBus->start());
  CHECK_OK((*server)->start());

  std::vector<Seen> seen;
  for (size_t index = 0; index < steps.size(); ++index) {
    const Step& step = steps[index];
    for (const ResetKind kind : step.resets) supervisor.requestReset(kind);
    const bool afterRequests = (*link)->isActivePolicyCurrent() && !supervisor.hasOutstandingReset();
    const SystemObservation observation = model.observationAt(step.time, step.value);
    if (step.failNextSolve) failNextSolve();
    if (index == 0) {
      // The first observations may go out before the subscriptions have reached the publishers ("slow joiner"): the
      // same observation, until a policy solved from it arrives. Every one of them plans the same policy.
      EXPECT_TRUE(waitFor([&]() {
        (*link)->setCurrentObservation(observation);
        absl::SleepFor(absl::Milliseconds(2));
        return (*link)->statistics().policiesAccepted > 0;
      }));
      EXPECT_TRUE(waitFor([&]() {
        const MpcServer::Statistics statistics = (*server)->statistics();
        return statistics.statusesPublished == statistics.solveAttempts;
      }));
      absl::SleepFor(absl::Milliseconds(20));
    } else {
      const uint64_t accepted = (*link)->statistics().policiesAccepted;
      (*link)->setCurrentObservation(observation);
      EXPECT_TRUE(waitFor([&]() {
        const RemoteMpcLink::Statistics statistics = (*link)->statistics();
        return statistics.policiesAccepted > accepted && statistics.newestPolicyInitTime == step.time;
      })) << "no policy solved from the observation at t = "
          << step.time;
    }
    (*link)->updatePolicy();
    seen.push_back(see(**link, supervisor, model, observation));
    seen.back().postResetPolicyActiveAfterRequests = afterRequests;
  }
  (*server)->stop();
  return seen;
}

TEST(RemoteMpcLinkParityTest, TheControllerSeesTheSameSequenceOfPoliciesInProcessAndOverTheNetwork) {
  const std::vector<Step> steps = script();
  const ParityModel model = flatModel();
  std::unique_ptr<mpc_test::ScriptedMpc> inProcessMpc = test_support::makeScriptedMpc();
  std::unique_ptr<mpc_test::ScriptedMpc> remoteMpc = test_support::makeScriptedMpc();
  const std::vector<Seen> inProcess = runInProcess(steps, model, *inProcessMpc, [&]() { inProcessMpc->solver().failNextSolves(1); });
  const std::vector<Seen> remote = runRemote(steps, model, *remoteMpc, [&]() { remoteMpc->solver().failNextSolves(1); });
  const size_t inProcessResets = inProcessMpc->solver().numResets();
  const size_t remoteResets = remoteMpc->solver().numResets();
  ASSERT_EQ(inProcess.size(), steps.size());
  ASSERT_EQ(remote.size(), steps.size());
  for (size_t step = 0; step < steps.size(); ++step) expectSame(inProcess[step], remote[step], step);
  EXPECT_EQ(remoteResets, inProcessResets) << "the solver was reset a different number of times";

  // Positive controls: the script changes the policy every cycle, and a cycle with a request holds until its policy.
  for (size_t step = 1; step < steps.size(); ++step) {
    EXPECT_NE(inProcess[step].states, inProcess[step - 1].states);
    EXPECT_EQ(inProcess[step].postResetPolicyActiveAfterRequests, steps[step].resets.empty()) << "control cycle " << step;
  }
  for (const Seen& seen : inProcess) EXPECT_TRUE(seen.postResetPolicyActive);
}

TEST(RemoteMpcLinkParityTest, AManifoldPolicyWithAnEventIsTheSameInProcessAndOverTheNetwork) {
  // The script without its solver failures: the stand-in SQP never fails.
  std::vector<Step> steps = script();
  for (Step& step : steps) step.failNextSolve = false;
  const ParityModel model = manifoldModel();
  std::unique_ptr<test_support::ManifoldScriptedMpc> inProcessMpc = test_support::makeManifoldScriptedMpc(model.stateManifold);
  std::unique_ptr<test_support::ManifoldScriptedMpc> remoteMpc = test_support::makeManifoldScriptedMpc(model.stateManifold);
  const std::function<void()> neverFails = []() { ADD_FAILURE() << "the manifold script has no solver failures"; };
  const std::vector<Seen> inProcess = runInProcess(steps, model, *inProcessMpc, neverFails);
  const std::vector<Seen> remote = runRemote(steps, model, *remoteMpc, neverFails);
  ASSERT_EQ(inProcess.size(), steps.size());
  ASSERT_EQ(remote.size(), steps.size());
  for (size_t step = 0; step < steps.size(); ++step) expectSame(inProcess[step], remote[step], step);
  EXPECT_EQ(remoteMpc->solver().numResets(), inProcessMpc->solver().numResets()) << "the solver was reset a different number of times";

  // Positive controls: the policy in use is a manifold-linear one with an event, which changes the input across it,
  // and every cycle has a policy of its own.
  for (size_t step = 0; step < steps.size(); ++step) {
    ASSERT_EQ(inProcess[step].postEventIndices.size(), size_t{1}) << "control cycle " << step;
    EXPECT_NE(inProcess[step].evaluatedInputs[1], inProcess[step].evaluatedInputs[3]) << "control cycle " << step;
    EXPECT_TRUE(inProcess[step].postResetPolicyActive) << "control cycle " << step;
    if (step > 0) {
      EXPECT_NE(inProcess[step].states, inProcess[step - 1].states) << "control cycle " << step;
    }
  }
}

}  // namespace
}  // namespace ocs2::humanoid::ipc
