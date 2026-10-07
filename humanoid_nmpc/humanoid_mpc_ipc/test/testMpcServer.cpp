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

// The MPC side of the network MPC link against a robot the test scripts message by message: which resets the server
// serves from which observation, what it stamps on the policies, the status after every attempt, the solution window,
// the hooks, the pacing, the back-off, and its lifecycle.

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_mpc/CommandData.h"
#include "ocs2_oc/oc_data/PerformanceIndex.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

#include "humanoid_mpc_ipc/MpcMessageConversions.h"
#include "humanoid_mpc_ipc/MpcServer.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/mpc_observation.pb.h"
#include "humanoid_mpc_msgs/mpc_policy.pb.h"
#include "humanoid_mpc_msgs/mpc_status.pb.h"
#include "humanoid_mpc_msgs/viewer_annotations.pb.h"
#include "humanoid_nmpc/humanoid_mpc_ipc/test/MpcLinkTestSupport.h"
#include "robot_ipc/Bus.h"
#include "robot_ipc/Delivery.h"

namespace ocs2::humanoid::ipc {
namespace {

using test_support::modelDimensions;
using test_support::observationAt;
using test_support::resetTargetsFor;
using test_support::waitFor;

/** A robot the test speaks for: it publishes the observations the test makes and records what the server sends. */
class FakeRobot {
 public:
  FakeRobot() : bus_(test_support::createNodeBus("robot")) {
    CHECK_OK(bus_->subscribe<humanoid_mpc_msgs::MpcPolicy>(topics::kMpcPolicy, robot::ipc::Delivery::kAll,
                                                           [this](const humanoid_mpc_msgs::MpcPolicy& message) {
                                                             absl::MutexLock lock(mutex_);
                                                             policies_.push_back(message);
                                                           }));
    CHECK_OK(bus_->subscribe<humanoid_mpc_msgs::MpcStatus>(topics::kMpcStatus, robot::ipc::Delivery::kAll,
                                                           [this](const humanoid_mpc_msgs::MpcStatus& message) {
                                                             absl::MutexLock lock(mutex_);
                                                             statuses_.push_back(message);
                                                           }));
  }

  robot::ipc::Bus& bus() { return *bus_; }

  /** Publishes the observation at `time` with the given counters, as the next control cycle. */
  void publish(scalar_t time, uint64_t requested = 0, uint64_t fullRequested = 0, scalar_t value = 1.0) {
    publishWithSequence(++sequence_, time, requested, fullRequested, value);
  }

  void publishWithSequence(uint64_t sequence, scalar_t time, uint64_t requested, uint64_t fullRequested, scalar_t value = 1.0) {
    humanoid_mpc_msgs::MpcObservation message;
    toProto(observationAt(time, value), message.mutable_observation());
    message.mutable_resets()->set_requested(requested);
    message.mutable_resets()->set_full_requested(fullRequested);
    message.set_sequence(sequence);
    CHECK_OK(bus_->publish(topics::kRobotMpcObservation, message));
  }

  std::vector<humanoid_mpc_msgs::MpcPolicy> policies() const {
    absl::MutexLock lock(mutex_);
    return policies_;
  }
  std::vector<humanoid_mpc_msgs::MpcStatus> statuses() const {
    absl::MutexLock lock(mutex_);
    return statuses_;
  }
  size_t numPolicies() const {
    absl::MutexLock lock(mutex_);
    return policies_.size();
  }
  /** The newest policy solved from the observation at `time`, if one has arrived. */
  bool hasPolicyFrom(scalar_t time) const {
    absl::MutexLock lock(mutex_);
    for (const humanoid_mpc_msgs::MpcPolicy& policy : policies_) {
      if (policy.init_observation().time() == time) return true;
    }
    return false;
  }
  humanoid_mpc_msgs::MpcPolicy lastPolicy() const {
    absl::MutexLock lock(mutex_);
    return policies_.back();
  }

  uint64_t sequence() const { return sequence_; }

 private:
  std::unique_ptr<robot::ipc::Bus> bus_;
  uint64_t sequence_ = 0;
  mutable absl::Mutex mutex_;
  std::vector<humanoid_mpc_msgs::MpcPolicy> policies_ ABSL_GUARDED_BY(mutex_);
  std::vector<humanoid_mpc_msgs::MpcStatus> statuses_ ABSL_GUARDED_BY(mutex_);
};

class MpcServerTest : public ::testing::Test {
 protected:
  static MpcServer::Config defaultConfig() {
    MpcServer::Config config;
    config.dimensions = modelDimensions();
    config.resetSupervisor.initialRetryInterval = 0.005;
    config.resetSupervisor.maxRetryInterval = 0.02;
    return config;
  }

  void build(MpcServer::Config config = defaultConfig(),
             MpcServer::Hooks hooks = MpcServer::Hooks(),
             const mpc::Settings& settings = test_support::mpcSettings(),
             MpcServer::ResetTargetTrajectoriesFunction resetTargets = resetTargetsFor) {
    mpc_ = test_support::makeScriptedMpc(settings);
    mpcBus_ = test_support::createNodeBus("mpc");
    test_support::connectBoth(robot_.bus(), *mpcBus_);
    absl::StatusOr<std::unique_ptr<MpcServer>> server =
        MpcServer::Create(*mpcBus_, *mpc_, std::move(resetTargets), std::move(config), std::move(hooks));
    CHECK_OK(server.status());
    server_ = std::move(*server);
    CHECK_OK(robot_.bus().start());
    CHECK_OK(mpcBus_->start());
    CHECK_OK(server_->start());
  }

  void TearDown() override {
    if (server_ != nullptr) server_->stop();
  }

  /** Publishes observations at `time` until a policy solved from one has arrived ("slow joiner"). */
  void warmUp(scalar_t time = 1.0) {
    ASSERT_TRUE(waitFor([&]() {
      robot_.publish(time);
      absl::SleepFor(absl::Milliseconds(2));
      return robot_.hasPolicyFrom(time);
    }));
    // Whatever was still being solved has been published once the server has answered every attempt.
    ASSERT_TRUE(waitFor([&]() {
      const MpcServer::Statistics statistics = server_->statistics();
      return statistics.statusesPublished == statistics.solveAttempts;
    }));
    absl::SleepFor(absl::Milliseconds(10));
  }

  /** Publishes one observation and waits for the policy solved from it. */
  humanoid_mpc_msgs::MpcPolicy solveFrom(scalar_t time, uint64_t requested = 0, uint64_t fullRequested = 0) {
    robot_.publish(time, requested, fullRequested);
    EXPECT_TRUE(waitFor([&]() { return robot_.hasPolicyFrom(time); })) << "no policy from t = " << time;
    return robot_.lastPolicy();
  }

  FakeRobot robot_;
  std::unique_ptr<mpc_test::ScriptedMpc> mpc_;
  std::unique_ptr<robot::ipc::Bus> mpcBus_;
  std::unique_ptr<MpcServer> server_;
};

// ---------------------------------------------------------------------------------------------------------------------
// Resets and stamps
// ---------------------------------------------------------------------------------------------------------------------

TEST_F(MpcServerTest, ThePolicyCarriesTheCountersItServedAndAStatusFollowsEveryAttempt) {
  build();
  warmUp();
  EXPECT_EQ(server_->statistics().fullResets, 1u) << "the start-up reset, before the first solve";

  const humanoid_mpc_msgs::MpcPolicy full = solveFrom(/*time=*/1.1, /*requested=*/2, /*fullRequested=*/1);
  EXPECT_EQ(full.resets_served(), 2u);
  EXPECT_EQ(full.full_resets_served(), 1u);
  EXPECT_EQ(server_->statistics().fullResets, 2u);

  const humanoid_mpc_msgs::MpcPolicy solver = solveFrom(/*time=*/1.2, /*requested=*/3, /*fullRequested=*/1);
  EXPECT_EQ(solver.resets_served(), 3u);
  EXPECT_EQ(solver.full_resets_served(), 1u);
  EXPECT_EQ(server_->statistics().solverResets, 1u);
  EXPECT_EQ(server_->statistics().fullResets, 2u);

  const humanoid_mpc_msgs::MpcPolicy none = solveFrom(/*time=*/1.3, /*requested=*/3, /*fullRequested=*/1);
  EXPECT_EQ(none.resets_served(), 3u);
  EXPECT_EQ(server_->statistics().solverResets + server_->statistics().fullResets, 3u);
  EXPECT_TRUE(none.solver_status().healthy());
  EXPECT_EQ(none.solver_status().consecutive_failures(), 0u);
  EXPECT_TRUE(none.solver_status().last_error().empty());
  EXPECT_GT(none.solver_status().solve_count(), solver.solver_status().solve_count());
  EXPECT_EQ(none.init_observation().time(), 1.3);
  EXPECT_FALSE(none.has_annotations()) << "no annotations provider, no annotations";

  // The targets of a full reset are the caller's, of the observation it reset from.
  TargetTrajectories targets;
  ASSERT_TRUE(fromProto(full.target_trajectories(), &targets).ok());
  EXPECT_EQ(targets.timeTrajectory, resetTargetsFor(observationAt(1.1, 1.0)).timeTrajectory);

  ASSERT_TRUE(waitFor([&]() { return robot_.statuses().size() >= server_->statistics().solveAttempts; }));
  const std::vector<humanoid_mpc_msgs::MpcStatus> statuses = robot_.statuses();
  const humanoid_mpc_msgs::MpcStatus& last = statuses.back();
  EXPECT_EQ(last.observation_time(), 1.3);
  EXPECT_EQ(last.resets_served(), 3u);
  EXPECT_EQ(last.full_resets_served(), 1u);
  EXPECT_EQ(last.solver_status().solve_count(), none.solver_status().solve_count());
  // One server instance, nonzero, on everything it publishes: the robot tells a restarted node by it.
  EXPECT_NE(server_->serverInstance(), 0u);
  for (const humanoid_mpc_msgs::MpcPolicy& policy : {full, solver, none}) {
    EXPECT_EQ(policy.solver_status().server_instance(), server_->serverInstance());
  }
  for (const humanoid_mpc_msgs::MpcStatus& status : statuses) {
    EXPECT_EQ(status.solver_status().server_instance(), server_->serverInstance());
  }
  uint64_t received = 0;
  for (const humanoid_mpc_msgs::MpcStatus& status : statuses) received += status.observations_received();
  EXPECT_LE(received, server_->statistics().observationsReceived);
}

TEST_F(MpcServerTest, ARestartedRobotProcessGetsAFullReset) {
  build();
  warmUp();
  solveFrom(/*time=*/1.1, /*requested=*/3, /*fullRequested=*/1);
  const MpcServer::Statistics before = server_->statistics();
  EXPECT_EQ(before.robotSessions, 1u);

  // The sequence numbers start over.
  robot_.publishWithSequence(/*sequence=*/1, /*time=*/0.0, /*requested=*/3, /*fullRequested=*/1);
  ASSERT_TRUE(waitFor([&]() { return robot_.hasPolicyFrom(0.0); }));
  EXPECT_EQ(server_->statistics().fullResets, before.fullResets + 1);
  EXPECT_EQ(server_->statistics().robotSessions, 2u);

  // The counters start over.
  robot_.publishWithSequence(/*sequence=*/2, /*time=*/0.01, /*requested=*/0, /*fullRequested=*/0);
  ASSERT_TRUE(waitFor([&]() { return robot_.hasPolicyFrom(0.01); }));
  EXPECT_EQ(server_->statistics().fullResets, before.fullResets + 2);
  EXPECT_EQ(server_->statistics().robotSessions, 3u);
  EXPECT_EQ(robot_.lastPolicy().resets_served(), 0u) << "the counters of the new robot process";
}

TEST_F(MpcServerTest, AnObservationOfOtherDimensionsIsRejectedAndNotSolved) {
  build();
  warmUp();
  const MpcServer::Statistics before = server_->statistics();
  humanoid_mpc_msgs::MpcObservation wrong;
  toProto(observationAt(1.1, 1.0), wrong.mutable_observation());
  wrong.mutable_observation()->add_state(1.0);
  wrong.set_sequence(robot_.sequence() + 1);
  CHECK_OK(robot_.bus().publish(topics::kRobotMpcObservation, wrong));
  ASSERT_TRUE(waitFor([&]() { return server_->statistics().observationsRejected == before.observationsRejected + 1; }));

  // A mode the model does not have.
  humanoid_mpc_msgs::MpcObservation unknownMode;
  toProto(observationAt(1.2, 1.0), unknownMode.mutable_observation());
  unknownMode.mutable_observation()->set_mode(test_support::kNumModes);
  unknownMode.set_sequence(robot_.sequence() + 2);
  CHECK_OK(robot_.bus().publish(topics::kRobotMpcObservation, unknownMode));
  ASSERT_TRUE(waitFor([&]() { return server_->statistics().observationsRejected == before.observationsRejected + 2; }));
  absl::SleepFor(absl::Milliseconds(20));
  EXPECT_EQ(server_->statistics().solveAttempts, before.solveAttempts);
}

TEST_F(MpcServerTest, TheSolutionTimeWindowBoundsThePolicy) {
  build(defaultConfig(), MpcServer::Hooks(), test_support::mpcSettings(/*solutionTimeWindow=*/0.3));
  warmUp();
  const humanoid_mpc_msgs::MpcPolicy policy = solveFrom(/*time=*/1.1);
  ASSERT_GT(policy.time_trajectory_size(), 1);
  EXPECT_NEAR(policy.time_trajectory(policy.time_trajectory_size() - 1), 1.4, 1.0e-12);
}

// ---------------------------------------------------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------------------------------------------------

TEST_F(MpcServerTest, TheHooksSeeEveryPolicyAndAHookThatFailsStopsNothing) {
  std::atomic<uint64_t> observed{0};
  std::atomic<bool> failNext{false};
  MpcServer::Hooks hooks;
  hooks.annotationsProvider = [](const CommandData& command, const PrimalSolution& /*solution*/,
                                 humanoid_mpc_msgs::ViewerAnnotations* absl_nonnull annotations) {
    annotations->set_scaled_velocity_x(command.mpcInitObservation_.time);
    return absl::OkStatus();
  };
  hooks.postSolveObserver = [&](const CommandData& /*command*/, const PrimalSolution& solution, const PerformanceIndex& /*performance*/) {
    EXPECT_FALSE(solution.timeTrajectory_.empty());
    observed.fetch_add(1);
    return failNext.exchange(false) ? absl::InternalError("observer failure") : absl::OkStatus();
  };
  build(defaultConfig(), std::move(hooks));
  warmUp();
  const humanoid_mpc_msgs::MpcPolicy policy = solveFrom(/*time=*/1.1);
  EXPECT_EQ(policy.annotations().scaled_velocity_x(), 1.1);

  failNext.store(true);
  solveFrom(/*time=*/1.2);
  solveFrom(/*time=*/1.3);
  // Joined before the hooks' state goes out of scope; every hook call has returned by then.
  server_->stop();
  EXPECT_EQ(observed.load(), server_->statistics().policiesPublished);
  EXPECT_TRUE(server_->statistics().healthy);
}

TEST_F(MpcServerTest, AnAnnotationsProviderThatFailsSendsThePolicyWithoutAnnotations) {
  std::atomic<bool> failNext{false};
  MpcServer::Hooks hooks;
  hooks.annotationsProvider = [&failNext](const CommandData& command, const PrimalSolution& /*solution*/,
                                          humanoid_mpc_msgs::ViewerAnnotations* absl_nonnull annotations) {
    annotations->set_scaled_velocity_x(command.mpcInitObservation_.time);
    return failNext.exchange(false) ? absl::InternalError("annotations failure") : absl::OkStatus();
  };
  build(defaultConfig(), std::move(hooks));
  warmUp();

  failNext.store(true);
  const humanoid_mpc_msgs::MpcPolicy withoutAnnotations = solveFrom(/*time=*/1.1);
  EXPECT_EQ(withoutAnnotations.annotations().scaled_velocity_x(), 0.0) << "what the provider wrote before it failed is cleared";
  EXPECT_FALSE(withoutAnnotations.time_trajectory().empty());
  const humanoid_mpc_msgs::MpcPolicy annotated = solveFrom(/*time=*/1.2);
  EXPECT_EQ(annotated.annotations().scaled_velocity_x(), 1.2);
  // Joined before the hook's state goes out of scope.
  server_->stop();
  EXPECT_TRUE(server_->statistics().healthy);
  EXPECT_EQ(server_->statistics().failedAttempts, 0u);
}

// ---------------------------------------------------------------------------------------------------------------------
// Pacing and back-off
// ---------------------------------------------------------------------------------------------------------------------

TEST_F(MpcServerTest, TheDesiredFrequencyBoundsTheSolveRate) {
  MpcServer::Config config = defaultConfig();
  config.mpcDesiredFrequency = 20.0;
  build(config);
  warmUp();
  const uint64_t before = server_->statistics().solveAttempts;
  const absl::Time start = absl::Now();
  scalar_t time = 1.0;
  while (absl::Now() - start < absl::Milliseconds(500)) {
    time += 0.001;
    robot_.publish(time);
    absl::SleepFor(absl::Milliseconds(1));
  }
  const uint64_t attempts = server_->statistics().solveAttempts - before;
  EXPECT_GE(attempts, 2u) << "it solves";
  EXPECT_LE(attempts, 13u) << "20 Hz for 0.5 s, plus the attempts at both ends";
}

TEST_F(MpcServerTest, AFailingSolverReportsInEveryStatusBacksOffAndRecovers) {
  build();
  warmUp();
  const size_t policiesBefore = robot_.numPolicies();
  mpc_->solver().failEverySolve(/*fail=*/true);
  scalar_t time = 1.0;
  ASSERT_TRUE(waitFor([&]() {
    time += 0.001;
    robot_.publish(time);
    return !server_->statistics().healthy && server_->statistics().failedAttempts >= 5;
  }));
  absl::SleepFor(absl::Milliseconds(30));
  const std::vector<humanoid_mpc_msgs::MpcStatus> statuses = robot_.statuses();
  const humanoid_mpc_msgs::MpcStatus& last = statuses.back();
  EXPECT_FALSE(last.solver_status().healthy());
  EXPECT_GE(last.solver_status().consecutive_failures(), 3u);
  // OCS2's exception, as the attempt's status.
  EXPECT_THAT(last.solver_status().last_error(), ::testing::StartsWith("MPC solver crashed at t = "));
  EXPECT_THAT(last.solver_status().last_error(), ::testing::HasSubstr("[ScriptedSolver] scripted failure: Failed to solve QP"));
  EXPECT_EQ(robot_.numPolicies(), policiesBefore) << "a failed solve publishes no policy";

  mpc_->solver().failEverySolve(/*fail=*/false);
  ASSERT_TRUE(waitFor([&]() {
    time += 0.001;
    robot_.publish(time);
    return robot_.numPolicies() > policiesBefore;
  }));
  const humanoid_mpc_msgs::MpcPolicy recovered = robot_.lastPolicy();
  EXPECT_TRUE(recovered.solver_status().healthy());
  EXPECT_EQ(recovered.solver_status().consecutive_failures(), 0u);
}

TEST_F(MpcServerTest, ARequestFromTheRobotEndsTheBackOff) {
  MpcServer::Config config = defaultConfig();
  config.resetSupervisor.initialRetryInterval = 5.0;
  config.resetSupervisor.maxRetryInterval = 5.0;
  build(config);
  warmUp();
  mpc_->solver().failEverySolve(/*fail=*/true);
  ASSERT_TRUE(waitFor([&]() {
    robot_.publish(1.0);
    absl::SleepFor(absl::Milliseconds(1));
    return !server_->statistics().healthy;
  }));
  // Now waiting out a five-second back-off.
  absl::SleepFor(absl::Milliseconds(50));
  const uint64_t attempts = server_->statistics().solveAttempts;
  mpc_->solver().failEverySolve(/*fail=*/false);
  const absl::Time requested = absl::Now();
  robot_.publish(/*time=*/1.1, /*requested=*/1, /*fullRequested=*/1);
  ASSERT_TRUE(waitFor([&]() { return robot_.hasPolicyFrom(1.1); }, absl::Seconds(3)));
  EXPECT_LT(absl::Now() - requested, absl::Seconds(1)) << "an operator re-entering WB_MPC waited out the back-off";
  EXPECT_GT(server_->statistics().solveAttempts, attempts);
  EXPECT_EQ(robot_.lastPolicy().resets_served(), 1u);
}

TEST_F(MpcServerTest, AResetThatKeepsFailingWaitsOutTheBackOffInsteadOfSpinning) {
  MpcServer::Config config = defaultConfig();
  config.resetSupervisor.initialRetryInterval = 0.05;
  config.resetSupervisor.maxRetryInterval = 0.1;
  std::atomic<bool> resetFails{false};
  const MpcServer::ResetTargetTrajectoriesFunction resetTargets =
      [&resetFails](const SystemObservation& observation) -> absl::StatusOr<TargetTrajectories> {
    if (resetFails.load()) {
      return absl::InvalidArgumentError("the reset targets cannot be made");
    }
    return resetTargetsFor(observation);
  };
  build(config, MpcServer::Hooks(), test_support::mpcSettings(), resetTargets);
  warmUp();

  // A reset the robot asks for and the server cannot make: the request stays unserved, attempt after attempt.
  resetFails.store(true);
  robot_.publish(/*time=*/1.1, /*requested=*/1, /*fullRequested=*/1);
  ASSERT_TRUE(waitFor([&]() { return !server_->statistics().healthy; }));
  const MpcServer::Statistics before = server_->statistics();
  absl::SleepFor(absl::Milliseconds(300));
  const MpcServer::Statistics after = server_->statistics();
  // Backing off 0.05 s, then 0.1 s: about four attempts in 0.3 s. The unserved request used to end every back-off at
  // once, and the solver thread retried, and published mpc/status, as fast as it could.
  EXPECT_GE(after.solveAttempts - before.solveAttempts, 1u) << "it keeps retrying";
  EXPECT_LE(after.solveAttempts - before.solveAttempts, 10u) << "it did not back off";
  EXPECT_LE(after.statusesPublished - before.statusesPublished, 10u);
  EXPECT_EQ(robot_.lastPolicy().resets_served(), 0u) << "a policy that serves the request without the reset";
  // What the reset function returned, as the failed attempt's status.
  EXPECT_EQ(robot_.statuses().back().solver_status().last_error(), "Resetting the MPC failed: the reset targets cannot be made");

  // Once the reset can be made the next attempt makes it, from the observation that asked for it.
  resetFails.store(false);
  ASSERT_TRUE(waitFor([&]() { return robot_.hasPolicyFrom(1.1); }));
  EXPECT_EQ(robot_.lastPolicy().resets_served(), 1u);
  EXPECT_EQ(robot_.lastPolicy().full_resets_served(), 1u);
  EXPECT_TRUE(server_->statistics().healthy);
}

// ---------------------------------------------------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------------------------------------------------

TEST(MpcServerCreateTest, RefusesWhatItCannotRun) {
  std::unique_ptr<mpc_test::ScriptedMpc> mpc = test_support::makeScriptedMpc();
  std::unique_ptr<robot::ipc::Bus> bus = test_support::createNodeBus("mpc");
  MpcServer::Config config;
  config.dimensions = modelDimensions();

  EXPECT_EQ(MpcServer::Create(*bus, *mpc, /*resetTargetTrajectories=*/nullptr, config).status().code(), absl::StatusCode::kInvalidArgument);
  MpcServer::Config noDimensions = config;
  noDimensions.dimensions.stateDim = 0;
  EXPECT_EQ(MpcServer::Create(*bus, *mpc, resetTargetsFor, noDimensions).status().code(), absl::StatusCode::kInvalidArgument);
  MpcServer::Config noModes = config;
  noModes.dimensions.numModes = 0;
  EXPECT_EQ(MpcServer::Create(*bus, *mpc, resetTargetsFor, noModes).status().code(), absl::StatusCode::kInvalidArgument);
  std::unique_ptr<robot::ipc::Bus> subscriber = test_support::createSubscriberBus("robot", bus->boundPort());
  EXPECT_EQ(MpcServer::Create(*subscriber, *mpc, resetTargetsFor, config).status().code(), absl::StatusCode::kInvalidArgument)
      << "a bus that cannot publish";
  CHECK_OK(bus->start());
  EXPECT_EQ(MpcServer::Create(*bus, *mpc, resetTargetsFor, config).status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(MpcServerLifecycleTest, StartsOnceAndStopsPromptlyWhileWaitingForAnObservation) {
  std::unique_ptr<mpc_test::ScriptedMpc> mpc = test_support::makeScriptedMpc();
  std::unique_ptr<robot::ipc::Bus> bus = test_support::createNodeBus("mpc");
  MpcServer::Config config;
  config.dimensions = modelDimensions();
  absl::StatusOr<std::unique_ptr<MpcServer>> server = MpcServer::Create(*bus, *mpc, resetTargetsFor, config);
  ASSERT_TRUE(server.ok()) << server.status();
  CHECK_OK(bus->start());
  ASSERT_TRUE((*server)->start().ok());
  EXPECT_EQ((*server)->start().code(), absl::StatusCode::kFailedPrecondition);

  absl::SleepFor(absl::Milliseconds(20));
  EXPECT_EQ((*server)->statistics().solveAttempts, 0u) << "nothing is solved before the first observation";
  EXPECT_EQ(mpc->solver().numResets(), 0u) << "nor reset";
  const absl::Time stop = absl::Now();
  (*server)->stop();
  (*server)->stop();
  EXPECT_LT(absl::Now() - stop, absl::Seconds(1));
  EXPECT_EQ((*server)->start().code(), absl::StatusCode::kFailedPrecondition) << "a stopped server stays stopped";
}

}  // namespace
}  // namespace ocs2::humanoid::ipc
