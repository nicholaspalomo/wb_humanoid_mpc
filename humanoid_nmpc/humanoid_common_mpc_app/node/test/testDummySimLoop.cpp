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

#include <atomic>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "ocs2_core/dynamics/LinearSystemDynamics.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_oc/rollout/RolloutBase.h"
#include "ocs2_oc/rollout/TimeTriggeredRollout.h"

#include "humanoid_common_mpc_app/node/DummySimLoop.h"
#include "humanoid_mpc_ipc/MpcServer.h"
#include "humanoid_nmpc/humanoid_mpc_ipc/test/MpcLinkTestSupport.h"
#include "robot_ipc/Bus.h"

/*
 * DummySimLoop, the dummy simulator's loop, against an MpcServer over loopback buses. The MPC is OCS2's scripted MPC and
 * the plant a linear system that holds its state, so that the test sees the synchronization alone: in the synchronized
 * mode every MPC update takes the policy solved from the observation sent for it, and the plant's clock advances by one
 * step per step whatever the solves cost.
 */

namespace ocs2::humanoid::node {
namespace {

namespace test_support = ::ocs2::humanoid::ipc::test_support;

constexpr scalar_t kSimulationFrequency = 100.0;  // [Hz]

/** A plant whose every rollout throws, as an OCS2 rollout does when its integration fails. */
class ThrowingRollout final : public RolloutBase {
 public:
  ThrowingRollout() : RolloutBase(rollout::Settings()) {}
  ThrowingRollout* absl_nonnull clone() const override { return new ThrowingRollout(*this); }
  vector_t run(scalar_t /*initTime*/,
               const vector_t& /*initState*/,
               scalar_t /*finalTime*/,
               ControllerBase* absl_nullable /*controller*/,
               ModeSchedule& /*modeSchedule*/,
               scalar_array_t& /*timeTrajectory*/,
               size_array_t& /*postEventIndices*/,
               vector_array_t& /*stateTrajectory*/,
               vector_array_t& /*inputTrajectory*/) override {
    throw std::runtime_error("the integration diverged");
  }
};

/** The scripted MPC on an MPC node's bus, and the dummy simulator's loop on a robot's bus, connected. */
class Harness {
 public:
  /** The plant is `plant` (cloned), or with none a linear system that holds its state. */
  explicit Harness(scalar_t mpcDesiredFrequency, const RolloutBase* absl_nullable plant = nullptr)
      : mpc_(test_support::makeScriptedMpc()),
        dynamics_(matrix_t::Zero(test_support::kStateDim, test_support::kStateDim),
                  matrix_t::Zero(test_support::kStateDim, test_support::kInputDim)),
        rollout_(dynamics_) {
    mpcBus_ = test_support::createNodeBus("mpc");
    std::unique_ptr<robot::ipc::Bus> robotBus = test_support::createNodeBus("robot");
    test_support::connectBoth(*robotBus, *mpcBus_);

    ipc::MpcServer::Config serverConfig;
    serverConfig.dimensions = test_support::modelDimensions();
    serverConfig.mpcDesiredFrequency = mpcDesiredFrequency;
    absl::StatusOr<std::unique_ptr<ipc::MpcServer>> server =
        ipc::MpcServer::Create(*mpcBus_, *mpc_, &test_support::resetTargetsFor, serverConfig);
    EXPECT_TRUE(server.ok()) << server.status();
    server_ = *std::move(server);

    DummySimLoop::Config loopConfig;
    loopConfig.simulationFrequency = kSimulationFrequency;
    loopConfig.mpcDesiredFrequency = mpcDesiredFrequency;
    loopConfig.link.dimensions = test_support::modelDimensions();
    const RolloutBase& rollout = plant != nullptr ? *plant : static_cast<const RolloutBase&>(rollout_);
    absl::StatusOr<std::unique_ptr<DummySimLoop>> loop = DummySimLoop::Create(std::move(robotBus), rollout, loopConfig);
    EXPECT_TRUE(loop.ok()) << loop.status();
    loop_ = *std::move(loop);
  }

  ~Harness() {
    stop_.store(true);
    if (runner_.joinable()) runner_.join();
    loop_.reset();
    server_->stop();
    mpcBus_->stop();
  }
  Harness(const Harness&) = delete;
  Harness& operator=(const Harness&) = delete;

  /** Runs the loop on a thread of its own for `duration` of wall time, then stops it; returns its status. */
  absl::Status runFor(absl::Duration duration) {
    EXPECT_TRUE(mpcBus_->start().ok());
    EXPECT_TRUE(server_->start().ok());
    absl::Status status;
    runner_ = std::thread([this, &status]() {
      status = loop_->run(test_support::observationAt(/*time=*/0.5, /*value=*/1.0), [this]() { return stop_.load(); });
    });
    absl::SleepFor(duration);
    stop_.store(true);
    runner_.join();
    return status;
  }

  DummySimLoop& loop() { return *loop_; }
  ipc::MpcServer& server() { return *server_; }

 private:
  std::unique_ptr<mpc_test::ScriptedMpc> mpc_;
  LinearSystemDynamics dynamics_;
  TimeTriggeredRollout rollout_;
  std::unique_ptr<robot::ipc::Bus> mpcBus_;
  std::unique_ptr<ipc::MpcServer> server_;
  std::unique_ptr<DummySimLoop> loop_;
  std::atomic<bool> stop_{false};
  std::thread runner_;
};

void expectNoResetAfterTheFirst(const ipc::MpcServer::Statistics& server, const ipc::RemoteMpcLink::Statistics& link) {
  EXPECT_EQ(server.fullResets, 1);
  EXPECT_EQ(server.solverResets, 0);
  EXPECT_EQ(server.robotSessions, 1);
  EXPECT_EQ(server.failedAttempts, 0);
  EXPECT_EQ(link.linkLosses, 0);
  EXPECT_EQ(link.stalePoliciesDropped, 0);
  EXPECT_EQ(link.latePoliciesDropped, 0);
  EXPECT_EQ(link.foreignPoliciesDropped, 0);
  EXPECT_EQ(link.invalidPoliciesRejected, 0);
}

TEST(DummySimLoop, SynchronizedWithTheMpcEveryUpdateTakesThePolicySolvedForIt) {
  // Two plant steps per MPC update.
  Harness harness(/*mpcDesiredFrequency=*/50.0);
  ASSERT_TRUE(harness.runFor(absl::Seconds(1.5)).ok());

  const DummySimLoop::Statistics statistics = harness.loop().statistics();
  ASSERT_GT(statistics.steps, 20);
  // An update every second step, the first one at step 0.
  EXPECT_EQ(statistics.synchronizedPolicies, (statistics.steps + 1) / 2);
  // The plant's clock advanced by exactly one step per step.
  const SystemObservation latest = harness.loop().latestObservation();
  EXPECT_NEAR(latest.time, 0.5 + static_cast<scalar_t>(statistics.steps) / kSimulationFrequency, 1.0e-9);
  // The plant holds its state under the scripted MPC's zero input.
  EXPECT_TRUE(latest.state.isApproxToConstant(1.0));
  expectNoResetAfterTheFirst(harness.server().statistics(), statistics.link);
}

TEST(DummySimLoop, InRealTimeEveryStepTakesTheNewestPolicy) {
  Harness harness(/*mpcDesiredFrequency=*/-1.0);
  ASSERT_TRUE(harness.runFor(absl::Seconds(1.0)).ok());

  const DummySimLoop::Statistics statistics = harness.loop().statistics();
  EXPECT_GT(statistics.steps, 20);
  EXPECT_GT(statistics.policyUpdates, 1);
  EXPECT_EQ(statistics.synchronizedPolicies, 0);
  EXPECT_GT(harness.server().statistics().policiesPublished, 1);
  expectNoResetAfterTheFirst(harness.server().statistics(), statistics.link);
}

TEST(DummySimLoop, ARolloutThatThrowsEndsTheLoopWithAnInternalError) {
  const ThrowingRollout plant;
  Harness harness(/*mpcDesiredFrequency=*/-1.0, &plant);
  const absl::Status status = harness.runFor(absl::Seconds(1));
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal) << status;
  EXPECT_TRUE(absl::StrContains(status.message(), "the rollout of the plant failed at t = 0.5")) << status;
  EXPECT_TRUE(absl::StrContains(status.message(), "the integration diverged")) << status;
  // The loop ended at its first step, so the plant never moved.
  EXPECT_EQ(harness.loop().statistics().steps, 0);
  EXPECT_EQ(harness.loop().latestObservation().time, 0.5);
}

TEST(DummySimLoop, StopsWhileWaitingForAnMpcThatIsNotThere) {
  const std::unique_ptr<mpc_test::ScriptedMpc> mpc = test_support::makeScriptedMpc();
  const LinearSystemDynamics dynamics(matrix_t::Zero(test_support::kStateDim, test_support::kStateDim),
                                      matrix_t::Zero(test_support::kStateDim, test_support::kInputDim));
  const TimeTriggeredRollout rollout(dynamics);
  DummySimLoop::Config config;
  config.mpcDesiredFrequency = 50.0;
  config.link.dimensions = test_support::modelDimensions();
  absl::StatusOr<std::unique_ptr<DummySimLoop>> loop = DummySimLoop::Create(test_support::createNodeBus("robot"), rollout, config);
  ASSERT_TRUE(loop.ok()) << loop.status();
  const absl::Time stopAt = absl::Now() + absl::Milliseconds(300);
  EXPECT_TRUE((*loop)->run(test_support::observationAt(/*time=*/0.0, /*value=*/0.0), [stopAt]() { return absl::Now() > stopAt; }).ok());
  EXPECT_EQ((*loop)->statistics().steps, 0);
  EXPECT_GT((*loop)->statistics().link.observationsPublished, 0);
  EXPECT_EQ((*loop)->run(test_support::observationAt(/*time=*/0.0, /*value=*/0.0), []() { return true; }).code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST(DummySimLoop, CreateRefusesAFrequencyThatIsNotPositive) {
  const LinearSystemDynamics dynamics(matrix_t::Zero(test_support::kStateDim, test_support::kStateDim),
                                      matrix_t::Zero(test_support::kStateDim, test_support::kInputDim));
  const TimeTriggeredRollout rollout(dynamics);
  DummySimLoop::Config config;
  config.simulationFrequency = 0.0;
  config.link.dimensions = test_support::modelDimensions();
  EXPECT_EQ(DummySimLoop::Create(test_support::createNodeBus("robot"), rollout, config).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ocs2::humanoid::node
