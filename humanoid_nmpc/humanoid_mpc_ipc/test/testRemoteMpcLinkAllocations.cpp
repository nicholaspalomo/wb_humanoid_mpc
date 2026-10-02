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

// The realtime side of the network MPC link makes no heap allocation once warmed up: what the controller's realtime
// thread calls every cycle - setCurrentObservation(), updatePolicy() with and without a new policy, and the predicates -
// with the bus, the link's IO thread and the MPC server running beside it. The count is the calling thread's own
// (robot::realtime::heapAllocationCountOnThisThread()), since the other threads allocate as they serialize and decode.
//
// The realtime thread evaluates the policy in use with RealtimePolicyEvaluator, which allocates nothing either, for a
// feedforward and a linear controller of the README's size, inside the plan and past both of its ends. OCS2's own
// MRT_BASE::evaluatePolicy() does allocate - ControllerBase::computeInput() and LinearInterpolation::interpolate()
// return by value - which is why the realtime thread does not call it; its count is measured and printed, not
// asserted, so that a fix in OCS2 does not fail this test.
//
// A binary of its own, because the allocation counter replaces malloc for the whole process.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <iostream>
#include <memory>
#include <random>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include <ocs2_core/Types.h>
#include <ocs2_core/control/ControllerType.h>
#include <ocs2_core/reference/TargetTrajectories.h>
#include <ocs2_mpc/CommandData.h>
#include <ocs2_mpc/MRT_BASE.h>
#include <ocs2_oc/oc_data/PerformanceIndex.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>

#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"
#include "humanoid_mpc_ipc/MpcServer.h"
#include "humanoid_mpc_ipc/RealtimePolicyEvaluator.h"
#include "humanoid_mpc_ipc/RemoteMpcLink.h"
#include "humanoid_nmpc/humanoid_mpc_ipc/test/MpcLinkTestSupport.h"
#include "humanoid_nmpc/humanoid_mpc_ipc/test/PolicyTestData.h"
#include "robot_ipc/Bus.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"

namespace ocs2::humanoid::ipc {
namespace {

using robot::realtime::heapAllocationCountOnThisThread;
using test_support::observationAt;
using test_support::waitFor;

constexpr int kCycles = 200;

class RemoteMpcLinkAllocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mpc_ = test_support::makeScriptedMpc();
    robotBus_ = test_support::createNodeBus("robot");
    mpcBus_ = test_support::createNodeBus("mpc");
    test_support::connectBoth(*robotBus_, *mpcBus_);
    RemoteMpcLink::Config linkConfig;
    linkConfig.dimensions = test_support::modelDimensions();
    linkConfig.policyTimeout = 100.0;
    absl::StatusOr<std::unique_ptr<RemoteMpcLink>> link = RemoteMpcLink::Create(*robotBus_, supervisor_, linkConfig);
    CHECK_OK(link.status());
    link_ = std::move(*link);
    MpcServer::Config serverConfig;
    serverConfig.dimensions = test_support::modelDimensions();
    absl::StatusOr<std::unique_ptr<MpcServer>> server = MpcServer::Create(*mpcBus_, *mpc_, test_support::resetTargetsFor, serverConfig);
    CHECK_OK(server.status());
    server_ = std::move(*server);
    CHECK_OK(robotBus_->start());
    CHECK_OK(mpcBus_->start());
    CHECK_OK(server_->start());
    ASSERT_TRUE(waitFor([&]() {
      link_->setCurrentObservation(observationAt(1.0, 1.0));
      absl::SleepFor(absl::Milliseconds(2));
      return link_->statistics().policiesAccepted > 0;
    }));
    ASSERT_TRUE(link_->updatePolicy());
  }

  void TearDown() override { server_->stop(); }

  MpcResetSupervisor supervisor_;
  std::unique_ptr<mpc_test::ScriptedMpc> mpc_;
  std::unique_ptr<robot::ipc::Bus> robotBus_;
  std::unique_ptr<robot::ipc::Bus> mpcBus_;
  std::unique_ptr<RemoteMpcLink> link_;
  std::unique_ptr<MpcServer> server_;
};

// Without this, a counter that saw nothing would make every test below pass.
TEST_F(RemoteMpcLinkAllocationTest, TheCounterSeesThisThreadAllocate) {
  const size_t before = heapAllocationCountOnThisThread();
  const SystemObservation observation = observationAt(1.0, 1.0);
  EXPECT_GE(heapAllocationCountOnThisThread() - before, 2u) << "the state and the input";
  EXPECT_EQ(observation.state.size(), static_cast<Eigen::Index>(test_support::kStateDim));
}

TEST_F(RemoteMpcLinkAllocationTest, TheControllersCallsOnTheLinkDoNotAllocate) {
  SystemObservation observation = observationAt(1.0, 1.0);
  size_t swaps = 0;
  size_t allocations = 0;
  scalar_t time = 1.0;
  for (int cycle = 0; cycle < kCycles; ++cycle) {
    time += 0.001;
    observation.time = time;
    observation.state.setConstant(1.0 + time);
    const size_t before = heapAllocationCountOnThisThread();
    // One realtime cycle of the controller, without evaluatePolicy().
    link_->setCurrentObservation(observation);
    const bool swapped = link_->updatePolicy();
    const bool postResetPolicyActive = link_->isActivePolicyCurrent() && !supervisor_.hasOutstandingReset();
    const bool healthy = supervisor_.isHealthy();
    const bool received = link_->initialPolicyReceived();
    allocations += heapAllocationCountOnThisThread() - before;
    swaps += swapped ? 1 : 0;
    EXPECT_TRUE(postResetPolicyActive && healthy && received);
    absl::SleepFor(absl::Milliseconds(1));
  }
  EXPECT_EQ(allocations, 0u);
  // Positive controls: new policies were swapped in, not just the idle path, and the observations went out. Observations
  // written between two polls of the IO thread go out as the newest one, so fewer than were written.
  EXPECT_GT(swaps, static_cast<size_t>(kCycles / 10));
  EXPECT_GT(link_->statistics().observationsPublished, static_cast<uint64_t>(kCycles / 10));
}

TEST_F(RemoteMpcLinkAllocationTest, AResetRequestAndItsHandshakeDoNotAllocateOnTheRealtimeThread) {
  SystemObservation observation = observationAt(1.0, 1.0);
  size_t allocations = 0;
  scalar_t time = 1.0;
  for (int reset = 0; reset < 10; ++reset) {
    size_t before = heapAllocationCountOnThisThread();
    supervisor_.requestReset(reset % 2 == 0 ? MpcResetSupervisor::ResetKind::kSolver : MpcResetSupervisor::ResetKind::kFull);
    allocations += heapAllocationCountOnThisThread() - before;
    bool served = false;
    for (int cycle = 0; cycle < 5000 && !served; ++cycle) {
      time += 0.001;
      observation.time = time;
      before = heapAllocationCountOnThisThread();
      link_->setCurrentObservation(observation);
      link_->updatePolicy();
      served = link_->isActivePolicyCurrent() && !supervisor_.hasOutstandingReset();
      allocations += heapAllocationCountOnThisThread() - before;
      absl::SleepFor(absl::Milliseconds(1));
    }
    ASSERT_TRUE(served) << "reset " << reset;
  }
  EXPECT_EQ(allocations, 0u);
  EXPECT_EQ(supervisor_.numResetsServed(), 10u);
}

/** An MRT whose buffer the test fills directly, to measure OCS2's evaluatePolicy() on a policy of any controller. */
class BufferedMrt final : public MRT_BASE {
 public:
  void resetMpcNode(const TargetTrajectories& /*initTargetTrajectories*/) override {}
  void setCurrentObservation(const SystemObservation& /*observation*/) override {}
  void fill(const PrimalSolution& solution) {
    moveToBuffer(std::make_unique<CommandData>(), std::make_unique<PrimalSolution>(solution), std::make_unique<PerformanceIndex>());
  }
};

size_t allocationsPerEvaluation(MRT_BASE& mrt, scalar_t startTime, scalar_t endTime, const vector_t& state) {
  vector_t mpcState = state;
  vector_t mpcInput;
  size_t mode = 0;
  mrt.evaluatePolicy(startTime, state, mpcState, mpcInput, mode);  // Sizes the outputs.
  const size_t before = heapAllocationCountOnThisThread();
  for (int cycle = 0; cycle < kCycles; ++cycle) {
    const scalar_t time = startTime + (endTime - startTime) * static_cast<scalar_t>(cycle) / kCycles;
    mrt.evaluatePolicy(time, state, mpcState, mpcInput, mode);
  }
  return (heapAllocationCountOnThisThread() - before) / kCycles;
}

/** Heap allocations per evaluate() of `policy` at times across [startTime, endTime], after one that sizes the outputs. */
size_t allocationsPerRealtimeEvaluation(const PrimalSolution& policy, scalar_t startTime, scalar_t endTime, const vector_t& state) {
  RealtimePolicyEvaluator evaluator(ModelDimensions{.stateDim = static_cast<size_t>(state.size()),
                                                    .inputDim = static_cast<size_t>(policy.inputTrajectory_.front().size()),
                                                    .numModes = 1});
  vector_t mpcState = state;
  vector_t mpcInput = policy.inputTrajectory_.front();
  size_t mode = 0;
  CHECK(evaluator.evaluate(policy, startTime, state, mpcState, mpcInput, mode) == RealtimePolicyEvaluator::Outcome::kEvaluated);
  const size_t before = heapAllocationCountOnThisThread();
  size_t evaluated = 0;
  for (int cycle = 0; cycle < kCycles; ++cycle) {
    const scalar_t time = startTime + (endTime - startTime) * static_cast<scalar_t>(cycle) / kCycles;
    evaluated += evaluator.evaluate(policy, time, state, mpcState, mpcInput, mode) == RealtimePolicyEvaluator::Outcome::kEvaluated ? 1 : 0;
  }
  const size_t allocations = heapAllocationCountOnThisThread() - before;
  CHECK_EQ(evaluated, static_cast<size_t>(kCycles));
  return allocations;
}

TEST_F(RemoteMpcLinkAllocationTest, TheRealtimeEvaluationOfThePolicyInUseDoesNotAllocate) {
  // The scripted MPC's feedforward policy, as the link swapped it in, from before its start to past its end.
  const PrimalSolution& inUse = link_->getPolicy();
  EXPECT_EQ(allocationsPerRealtimeEvaluation(inUse, inUse.timeTrajectory_.front() - 0.1, inUse.timeTrajectory_.back() + 0.1,
                                             observationAt(1.0, 1.0).state),
            0u);

  // The README's bandwidth example, nx = nu = 30 over 60 nodes, with either controller.
  std::mt19937 generator(/*sd=*/17);
  for (const ControllerType type : {ControllerType::FEEDFORWARD, ControllerType::LINEAR}) {
    const test_data::PolicyShape shape{.nodes = 60, .stateDim = 30, .inputDim = 30, .events = 4, .controllerType = type};
    const PrimalSolution policy = test_data::randomPrimalSolution(generator, shape);
    EXPECT_EQ(allocationsPerRealtimeEvaluation(policy, policy.timeTrajectory_.front() - 0.1, policy.timeTrajectory_.back() + 0.1,
                                               policy.stateTrajectory_.front()),
              0u)
        << (type == ControllerType::LINEAR ? "LinearController" : "FeedforwardController");
  }
}

TEST_F(RemoteMpcLinkAllocationTest, MeasuresTheAllocationsOfOcs2sEvaluatePolicy) {
  // The scripted MPC's feedforward policy, through the link.
  const size_t throughLink = allocationsPerEvaluation(*link_, /*startTime=*/1.0, /*endTime=*/1.9, observationAt(1.0, 1.0).state);

  // The README's bandwidth example, nx = nu = 30 over 60 nodes, with either controller.
  std::mt19937 generator(/*sd=*/11);
  std::array<size_t, 2> perController = {0, 0};
  const std::array<ControllerType, 2> types = {ControllerType::FEEDFORWARD, ControllerType::LINEAR};
  for (size_t type = 0; type < types.size(); ++type) {
    const test_data::PolicyShape shape{.nodes = 60, .stateDim = 30, .inputDim = 30, .events = 4, .controllerType = types[type]};
    const PrimalSolution solution = test_data::randomPrimalSolution(generator, shape);
    BufferedMrt mrt;
    mrt.fill(solution);
    ASSERT_TRUE(mrt.updatePolicy());
    perController[type] =
        allocationsPerEvaluation(mrt, solution.timeTrajectory_.front(), solution.timeTrajectory_.back(), solution.stateTrajectory_.front());
  }
  std::cout << "[ MEASURED ] heap allocations per MRT_BASE::evaluatePolicy(): " << throughLink
            << " (scripted feedforward policy, through the link), " << perController[0] << " (FeedforwardController, nx = nu = 30), "
            << perController[1] << " (LinearController, nx = nu = 30)\n";
  RecordProperty("evaluatePolicyAllocationsFeedforward", static_cast<int>(perController[0]));
  RecordProperty("evaluatePolicyAllocationsLinear", static_cast<int>(perController[1]));
}

}  // namespace
}  // namespace ocs2::humanoid::ipc
