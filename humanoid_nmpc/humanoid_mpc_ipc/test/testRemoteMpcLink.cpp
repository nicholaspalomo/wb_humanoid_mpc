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

// The network MPC link end to end: a RemoteMpcLink on the robot's bus and an MpcServer around OCS2's ScriptedMpc on the
// MPC's bus, over loopback TCP. The test thread is the robot's control thread: it writes the observations and calls
// updatePolicy(), evaluatePolicy() and the predicates, as CentroidalMpcMrtJointController does. A gate in the server's
// annotations provider holds a solved policy in flight; a relay between the robot and the MPC loses observations.

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include <ocs2_core/Types.h>
#include <ocs2_core/reference/TargetTrajectories.h>
#include <ocs2_mpc/CommandData.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>

#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"
#include "humanoid_mpc_ipc/MpcServer.h"
#include "humanoid_mpc_ipc/RemoteMpcLink.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/mpc_observation.pb.h"
#include "humanoid_mpc_msgs/viewer_annotations.pb.h"
#include "humanoid_nmpc/humanoid_mpc_ipc/test/MpcLinkTestSupport.h"
#include "robot_ipc/Bus.h"
#include "robot_ipc/Delivery.h"

namespace ocs2::humanoid::ipc {
namespace {

using test_support::CountingModule;
using test_support::Gate;
using test_support::modelDimensions;
using test_support::observationAt;
using test_support::resetTargetsFor;
using test_support::waitFor;
using ResetKind = MpcResetSupervisor::ResetKind;

/** Forwards the observations from the robot to the MPC, or loses them in transit. */
class ObservationRelay {
 public:
  void setLosing(bool losing) {
    absl::MutexLock lock(mutex_);
    losing_ = losing;
  }
  uint64_t lost() const {
    absl::MutexLock lock(mutex_);
    return lost_;
  }

  /** On the relay's IO thread. */
  void onObservation(const humanoid_mpc_msgs::MpcObservation& message, robot::ipc::Bus& out) {
    absl::MutexLock lock(mutex_);
    last_ = message;
    if (losing_) {
      ++lost_;
      return;
    }
    CHECK_OK(out.publish(topics::kRobotMpcObservation, message));
  }

  /** Sends the last observation that reached the relay once more, as a network that duplicates a message would. */
  void resendLast(robot::ipc::Bus& out) {
    absl::MutexLock lock(mutex_);
    CHECK_OK(out.publish(topics::kRobotMpcObservation, last_));
  }

 private:
  mutable absl::Mutex mutex_;
  bool losing_ ABSL_GUARDED_BY(mutex_) = false;
  uint64_t lost_ ABSL_GUARDED_BY(mutex_) = 0;
  humanoid_mpc_msgs::MpcObservation last_ ABSL_GUARDED_BY(mutex_);
};

class RemoteMpcLinkTest : public ::testing::Test {
 protected:
  /** [s] The robot time of the cycle that ends the warm-up, after the warm-up's own. */
  static constexpr scalar_t kSettleStep = 0.001;

  static RemoteMpcLink::Config defaultLinkConfig() {
    RemoteMpcLink::Config config;
    config.dimensions = modelDimensions();
    config.policyTimeout = 0.2;
    return config;
  }

  static MpcServer::Config defaultServerConfig() {
    MpcServer::Config config;
    config.dimensions = modelDimensions();
    // The back-off of a persistent failure, short enough for a test.
    config.resetSupervisor.initialRetryInterval = 0.005;
    config.resetSupervisor.maxRetryInterval = 0.02;
    return config;
  }

  /** The robot's bus and the MPC's, connected directly or with the observations through the relay. */
  void build(RemoteMpcLink::Config linkConfig = defaultLinkConfig(),
             MpcServer::Config serverConfig = defaultServerConfig(),
             bool observationsThroughRelay = false) {
    mpc_ = test_support::makeScriptedMpc();
    mpc_->getSolverPtr()->addSynchronizedModule(module_);
    robotBus_ = test_support::createNodeBus("robot");
    mpcBus_ = test_support::createNodeBus("mpc");
    if (observationsThroughRelay) {
      relayOut_ = test_support::createNodeBus("relay");
      CHECK_OK(robotBus_->connect(mpcBus_->boundEndpoint()));
      CHECK_OK(mpcBus_->connect(relayOut_->boundEndpoint()));
      relayIn_ = test_support::createSubscriberBus("robot", robotBus_->boundPort());
      CHECK_OK(relayIn_->subscribe<humanoid_mpc_msgs::MpcObservation>(
          topics::kRobotMpcObservation, robot::ipc::Delivery::kAll,
          [this](const humanoid_mpc_msgs::MpcObservation& message) { relay_.onObservation(message, *relayOut_); }));
    } else {
      test_support::connectBoth(*robotBus_, *mpcBus_);
    }

    absl::StatusOr<std::unique_ptr<RemoteMpcLink>> link = RemoteMpcLink::Create(*robotBus_, robotSupervisor_, linkConfig);
    CHECK_OK(link.status());
    link_ = std::move(*link);
    MpcServer::Hooks hooks;
    hooks.annotationsProvider = [this](const CommandData& /*command*/, const PrimalSolution& /*solution*/,
                                       humanoid_mpc_msgs::ViewerAnnotations* annotations) {
      annotations->set_scaled_velocity_x(0.5);
      gate_.pass();
    };
    absl::StatusOr<std::unique_ptr<MpcServer>> server = MpcServer::Create(*mpcBus_, *mpc_, resetTargetsFor, serverConfig, hooks);
    CHECK_OK(server.status());
    server_ = std::move(*server);

    CHECK_OK(robotBus_->start());
    CHECK_OK(mpcBus_->start());
    if (relayOut_ != nullptr) {
      CHECK_OK(relayOut_->start());
      CHECK_OK(relayIn_->start());
    }
    CHECK_OK(server_->start());
  }

  void TearDown() override {
    gate_.open();  // Never leave the solver thread waiting at the gate.
    if (server_ != nullptr) server_->stop();
  }

  /**
   * Writes the observation at robot time `time` until a policy solved from it is in the link's buffer: several of the
   * same time and state, because the first ones may go out before the subscriptions of the other side have reached the
   * publisher (ZeroMQ's "slow joiner"). Then one cycle at time + kSettleStep that waits for its own policy, after which
   * nothing is in flight any more, and swaps that in.
   */
  void warmUp(scalar_t time = 1.0, scalar_t value = 1.0) {
    ASSERT_TRUE(waitFor([&]() {
      link_->setCurrentObservation(observationAt(time, value));
      absl::SleepFor(absl::Milliseconds(2));
      return link_->statistics().policiesAccepted > 0;
    }));
    stepAndAwaitPolicy(time + kSettleStep, value);
  }

  /** One control cycle that waits for its policy: writes the observation, waits until the policy solved from it is
   * accepted, and swaps it in. */
  void stepAndAwaitPolicy(scalar_t time, scalar_t value) {
    robotTime_ = time;
    const uint64_t accepted = link_->statistics().policiesAccepted;
    link_->setCurrentObservation(observationAt(time, value));
    ASSERT_TRUE(waitFor([&]() {
      const RemoteMpcLink::Statistics statistics = link_->statistics();
      return statistics.policiesAccepted > accepted && statistics.newestPolicyInitTime == time;
    })) << "no policy solved from the observation at t = "
        << time;
    ASSERT_TRUE(link_->updatePolicy());
  }

  /**
   * The robot's control loop, about one cycle per millisecond: the robot clock advances by `timeStep`, the observation
   * is written and the policy swapped in. Until `condition`, evaluated after the swap, holds; false after kTimeout.
   */
  bool runRobotUntil(const std::function<bool()>& condition, scalar_t timeStep = 0.001) {
    const absl::Time deadline = absl::Now() + test_support::kTimeout;
    while (absl::Now() < deadline) {
      robotTime_ += timeStep;
      link_->setCurrentObservation(observationAt(robotTime_, /*value=*/1.0));
      link_->updatePolicy();
      if (condition()) return true;
      absl::SleepFor(absl::Milliseconds(1));
    }
    return false;
  }

  /** What the controller checks before it executes the policy in use. */
  bool postResetPolicyActive() const { return link_->isActivePolicyCurrent() && !robotSupervisor_.hasOutstandingReset(); }

  MpcResetSupervisor robotSupervisor_;
  std::shared_ptr<CountingModule> module_ = std::make_shared<CountingModule>();
  Gate gate_;
  ObservationRelay relay_;
  scalar_t robotTime_ = 0.0;
  std::unique_ptr<mpc_test::ScriptedMpc> mpc_;
  std::unique_ptr<robot::ipc::Bus> robotBus_;
  std::unique_ptr<robot::ipc::Bus> mpcBus_;
  std::unique_ptr<robot::ipc::Bus> relayOut_;
  std::unique_ptr<robot::ipc::Bus> relayIn_;
  std::unique_ptr<RemoteMpcLink> link_;
  std::unique_ptr<MpcServer> server_;
};

// ---------------------------------------------------------------------------------------------------------------------
// The first policy
// ---------------------------------------------------------------------------------------------------------------------

TEST_F(RemoteMpcLinkTest, TheFirstPolicyArrivesAndEvaluatesAsTheSolverPlannedIt) {
  build();
  EXPECT_FALSE(link_->initialPolicyReceived());
  EXPECT_FALSE(link_->updatePolicy());
  EXPECT_FALSE(link_->isActivePolicyCurrent()) << "no policy is current before the first one";
  EXPECT_TRUE(robotSupervisor_.isHealthy());

  warmUp(/*time=*/1.0, /*value=*/2.0);
  EXPECT_TRUE(link_->initialPolicyReceived());
  EXPECT_TRUE(link_->isActivePolicyCurrent());
  EXPECT_FALSE(robotSupervisor_.hasOutstandingReset());
  EXPECT_TRUE(robotSupervisor_.isHealthy());

  // Solved from the observation sent, after the start-up reset to the targets of the first observation.
  const scalar_t initTime = 1.0 + kSettleStep;
  const CommandData& command = link_->getCommand();
  EXPECT_EQ(command.mpcInitObservation_.time, initTime);
  EXPECT_EQ(command.mpcInitObservation_.state, vector_t::Constant(test_support::kStateDim, 2.0));
  EXPECT_EQ(command.mpcTargetTrajectories_.timeTrajectory, resetTargetsFor(observationAt(1.0, 2.0)).timeTrajectory);
  EXPECT_EQ(command.mpcTargetTrajectories_.stateTrajectory, resetTargetsFor(observationAt(1.0, 2.0)).stateTrajectory);

  // The plan x(t) = x0 + (t - t0) over the one-second horizon.
  vector_t state;
  vector_t input;
  size_t mode = 99;
  link_->evaluatePolicy(initTime + 0.5, vector_t::Constant(test_support::kStateDim, 2.0), state, input, mode);
  ASSERT_EQ(state.size(), static_cast<Eigen::Index>(test_support::kStateDim));
  EXPECT_NEAR(state(0), 2.5, 1e-12);
  EXPECT_EQ(input, vector_t::Zero(test_support::kInputDim));
  EXPECT_EQ(mode, 0u);
  EXPECT_NEAR(link_->getPolicy().timeTrajectory_.back(), initTime + 1.0, 1e-12);

  const MpcServer::Statistics server = server_->statistics();
  EXPECT_EQ(server.fullResets, 1u) << "one start-up reset, before the first solve";
  EXPECT_EQ(server.solverResets, 0u);
  EXPECT_EQ(server.robotSessions, 1u);
  EXPECT_GE(server.statusesPublished, server.policiesPublished) << "a status after every attempt";
  EXPECT_EQ(module_->resets(), 1u);
}

// ---------------------------------------------------------------------------------------------------------------------
// Resets across the network
// ---------------------------------------------------------------------------------------------------------------------

TEST_F(RemoteMpcLinkTest, APolicySolvedBeforeAResetIsDroppedAndOnlyThePolicyAfterItCompletesTheReset) {
  build();
  warmUp();
  const RemoteMpcLink::Statistics start = link_->statistics();

  // A policy is solved and held in flight.
  gate_.close();
  const size_t arrivals = gate_.arrivals();
  link_->setCurrentObservation(observationAt(1.01, 5.0));
  ASSERT_TRUE(waitFor([&]() { return gate_.arrivals() == arrivals + 1; }));

  // The controller asks for a reset while it is in flight; the link drops the buffered policy at its next poll.
  robotSupervisor_.requestReset(ResetKind::kSolver);
  EXPECT_TRUE(robotSupervisor_.hasOutstandingReset());
  ASSERT_TRUE(waitFor([&]() { return link_->statistics().resetTicketsTaken == start.resetTicketsTaken + 1; }));
  EXPECT_FALSE(link_->updatePolicy());
  EXPECT_FALSE(link_->isActivePolicyCurrent()) << "the policy in use was solved before the reset";

  // The next observation carries the request. The policy in flight arrives first and is dropped; the post-reset one is
  // solved and held.
  link_->setCurrentObservation(observationAt(1.02, 6.0));
  gate_.release(1);
  ASSERT_TRUE(waitFor([&]() { return link_->statistics().stalePoliciesDropped == start.stalePoliciesDropped + 1; }));
  ASSERT_TRUE(waitFor([&]() { return gate_.arrivals() == arrivals + 2; }));
  EXPECT_TRUE(robotSupervisor_.hasOutstandingReset()) << "a policy solved before the reset does not serve it";
  EXPECT_FALSE(link_->updatePolicy()) << "the policy solved before the reset was swapped in after it";
  EXPECT_FALSE(postResetPolicyActive());
  EXPECT_EQ(link_->statistics().policiesAccepted, start.policiesAccepted);

  // The post-reset policy serves it.
  gate_.open();
  ASSERT_TRUE(waitFor([&]() { return link_->statistics().resetTicketsCompleted == start.resetTicketsCompleted + 1; }));
  EXPECT_FALSE(robotSupervisor_.hasOutstandingReset());
  EXPECT_TRUE(link_->updatePolicy());
  EXPECT_TRUE(postResetPolicyActive());
  EXPECT_EQ(link_->getCommand().mpcInitObservation_.time, 1.02);
  EXPECT_EQ(link_->getPolicy().stateTrajectory_.front()(0), 6.0);
  EXPECT_EQ(robotSupervisor_.numResetsServed(), 1u);
  EXPECT_EQ(server_->statistics().robotResetsServed, 1u);
  EXPECT_EQ(server_->statistics().solverResets, 1u);
}

TEST_F(RemoteMpcLinkTest, FullAndSolverResetsAreServedAsTheirKind) {
  build();
  warmUp();
  ASSERT_EQ(server_->statistics().fullResets, 1u);
  ASSERT_EQ(module_->resets(), 1u);

  // A solver reset keeps the reference manager and the modules (the schedule in execution).
  robotSupervisor_.requestReset(ResetKind::kSolver);
  stepAndAwaitPolicy(/*time=*/1.01, /*value=*/1.0);
  EXPECT_TRUE(postResetPolicyActive());
  EXPECT_EQ(server_->statistics().solverResets, 1u);
  EXPECT_EQ(server_->statistics().fullResets, 1u);
  EXPECT_EQ(module_->resets(), 1u);
  EXPECT_EQ(robotSupervisor_.numResetsServed(), 1u);
  EXPECT_EQ(robotSupervisor_.numFullResetsServed(), 0u);

  // A full reset resets them, and starts from the targets of the observation it resets from.
  robotSupervisor_.requestReset(ResetKind::kFull);
  stepAndAwaitPolicy(/*time=*/1.02, /*value=*/3.0);
  EXPECT_TRUE(postResetPolicyActive());
  EXPECT_EQ(server_->statistics().fullResets, 2u);
  EXPECT_EQ(module_->resets(), 2u);
  EXPECT_EQ(robotSupervisor_.numFullResetsServed(), 1u);
  EXPECT_EQ(link_->getCommand().mpcTargetTrajectories_.stateTrajectory, resetTargetsFor(observationAt(1.02, 3.0)).stateTrajectory);

  // resetMpcNode() of MRT_BASE requests a full reset and returns at once.
  link_->resetMpcNode(TargetTrajectories());
  EXPECT_TRUE(robotSupervisor_.hasOutstandingReset());
  stepAndAwaitPolicy(/*time=*/1.03, /*value=*/1.0);
  EXPECT_EQ(server_->statistics().fullResets, 3u);
  EXPECT_EQ(robotSupervisor_.numFullResetsServed(), 2u);

  // A solver and a full reset outstanding together are served as one full reset.
  robotSupervisor_.requestReset(ResetKind::kSolver);
  robotSupervisor_.requestReset(ResetKind::kFull);
  stepAndAwaitPolicy(/*time=*/1.04, /*value=*/1.0);
  EXPECT_FALSE(robotSupervisor_.hasOutstandingReset());
  EXPECT_EQ(server_->statistics().fullResets, 4u);
  EXPECT_EQ(server_->statistics().solverResets, 1u);
  EXPECT_EQ(robotSupervisor_.numResetsServed(), 5u);
  EXPECT_EQ(robotSupervisor_.numFullResetsServed(), 3u);

  // Without a request nothing is reset.
  stepAndAwaitPolicy(/*time=*/1.05, /*value=*/1.0);
  EXPECT_EQ(server_->statistics().fullResets + server_->statistics().solverResets, 5u);
}

TEST_F(RemoteMpcLinkTest, ObservationsLostInTransitDoNotLoseAResetRequest) {
  build(defaultLinkConfig(), defaultServerConfig(), /*observationsThroughRelay=*/true);
  warmUp();
  const MpcServer::Statistics before = server_->statistics();

  // Every observation after the request is lost on its way to the MPC.
  relay_.setLosing(true);
  robotSupervisor_.requestReset(ResetKind::kFull);
  ASSERT_TRUE(runRobotUntil([&]() { return relay_.lost() >= 20; }));
  EXPECT_EQ(server_->statistics().robotResetsServed, before.robotResetsServed) << "positive control: the MPC heard of no request";
  EXPECT_TRUE(robotSupervisor_.hasOutstandingReset());
  EXPECT_FALSE(postResetPolicyActive());

  // The first observation that gets through carries the request all the same.
  relay_.setLosing(false);
  ASSERT_TRUE(runRobotUntil([&]() { return postResetPolicyActive(); }));
  EXPECT_EQ(robotSupervisor_.numFullResetsServed(), 1u);
  EXPECT_EQ(server_->statistics().fullResets, before.fullResets + 1);
  EXPECT_EQ(server_->statistics().robotFullResetsServed, 1u);
}

TEST_F(RemoteMpcLinkTest, ARepeatedObservationChangesNothing) {
  build(defaultLinkConfig(), defaultServerConfig(), /*observationsThroughRelay=*/true);
  warmUp();
  stepAndAwaitPolicy(/*time=*/1.01, /*value=*/1.0);
  ASSERT_TRUE(waitFor([&]() { return server_->statistics().statusesPublished == server_->statistics().solveAttempts; }));
  const MpcServer::Statistics before = server_->statistics();

  relay_.resendLast(*relayOut_);
  ASSERT_TRUE(waitFor([&]() { return server_->statistics().observationsSkipped == before.observationsSkipped + 1; }));
  absl::SleepFor(absl::Milliseconds(20));
  EXPECT_EQ(server_->statistics().solveAttempts, before.solveAttempts) << "an observation solved from already was solved again";
  EXPECT_EQ(server_->statistics().fullResets + server_->statistics().solverResets, before.fullResets + before.solverResets);
}

// ---------------------------------------------------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------------------------------------------------

TEST_F(RemoteMpcLinkTest, SolverFailuresMakeTheRobotUnhealthyUntilASolveSucceeds) {
  // A policy timeout far beyond the test, so that only the solver can make the link unhealthy.
  RemoteMpcLink::Config linkConfig = defaultLinkConfig();
  linkConfig.policyTimeout = 100.0;
  build(linkConfig);
  warmUp();

  mpc_->solver().failEverySolve(true);
  ASSERT_TRUE(runRobotUntil([&]() { return !robotSupervisor_.isHealthy(); }));
  const RemoteMpcLink::Statistics failing = link_->statistics();
  EXPECT_FALSE(failing.solverHealthy);
  EXPECT_TRUE(failing.linkHealthy);
  EXPECT_GE(failing.solverConsecutiveFailures, server_->resetSupervisor().getConfig().maxConsecutiveFailures);
  EXPECT_FALSE(link_->isActivePolicyCurrent()) << "the policy in use stays current through failed solves";
  EXPECT_FALSE(robotSupervisor_.hasOutstandingReset()) << "the MPC side resets itself; the robot requested nothing";
  EXPECT_FALSE(server_->statistics().healthy);

  mpc_->solver().failEverySolve(false);
  ASSERT_TRUE(runRobotUntil([&]() { return robotSupervisor_.isHealthy() && postResetPolicyActive(); }));
  EXPECT_TRUE(link_->statistics().solverHealthy);
  EXPECT_EQ(robotSupervisor_.numConsecutiveFailures(), 0u);
  EXPECT_GE(server_->statistics().failedAttempts, server_->resetSupervisor().getConfig().maxConsecutiveFailures);
  EXPECT_GT(link_->getCommand().mpcInitObservation_.time, failing.newestPolicyInitTime) << "the hold ends on a policy solved after it";
}

TEST_F(RemoteMpcLinkTest, ThePolicyTimeoutTripsAfterItsRobotTimeAndAFreshPolicyClearsIt) {
  RemoteMpcLink::Config linkConfig = defaultLinkConfig();
  linkConfig.policyTimeout = 0.05;
  build(linkConfig);
  warmUp(/*time=*/1.0, /*value=*/1.0);
  const scalar_t newestBefore = link_->statistics().newestPolicyInitTime;
  ASSERT_EQ(newestBefore, robotTime_);

  // The link is lost: every policy from now on is held. The robot clock runs twice as fast as the wall clock.
  gate_.close();
  bool trippedEarly = false;
  ASSERT_TRUE(runRobotUntil(
      [&]() {
        const bool healthy = robotSupervisor_.isHealthy();
        if (!healthy && robotTime_ <= newestBefore + linkConfig.policyTimeout) trippedEarly = true;
        return !healthy;
      },
      /*timeStep=*/0.002));
  EXPECT_FALSE(trippedEarly) << "the link read unhealthy before policyTimeout of robot time had passed";
  // The realtime thread's own check held the robot; the IO thread finds the loss at its next poll.
  ASSERT_TRUE(waitFor([&]() { return link_->statistics().linkLosses == 1; }));
  const RemoteMpcLink::Statistics lost = link_->statistics();
  EXPECT_FALSE(lost.linkHealthy);
  EXPECT_TRUE(lost.solverHealthy);
  EXPECT_EQ(lost.linkLosses, 1u);
  EXPECT_FALSE(link_->isActivePolicyCurrent()) << "the hold must end on a policy that arrives after the loss";
  EXPECT_GT(lost.policyAge, linkConfig.policyTimeout);

  // The link stays lost while the robot runs on, until the policy held in flight is older than the timeout too.
  bool recoveredWhileLost = false;
  ASSERT_TRUE(runRobotUntil(
      [&]() {
        recoveredWhileLost = recoveredWhileLost || robotSupervisor_.isHealthy();
        return robotTime_ > newestBefore + 3.0 * linkConfig.policyTimeout;
      },
      /*timeStep=*/0.002));
  EXPECT_FALSE(recoveredWhileLost);

  // The loss asked the MPC node for a full reset, as a solver declared unhealthy in process does.
  EXPECT_TRUE(robotSupervisor_.hasOutstandingReset());
  const uint64_t fullResetsBefore = server_->statistics().fullResets;

  // The link comes back: the policy that was held was solved before that reset and is too old by now, so it is dropped
  // (or, if the next one arrives in the same drain of the socket, superseded by it); the first policy solved after the
  // reset ends the hold.
  const uint64_t droppedBefore = link_->statistics().latePoliciesDropped + link_->statistics().stalePoliciesDropped +
                                 robotBus_->topicStatistics(topics::kMpcPolicy).superseded;
  gate_.open();
  ASSERT_TRUE(runRobotUntil([&]() { return robotSupervisor_.isHealthy() && postResetPolicyActive(); }, /*timeStep=*/0.002));
  const RemoteMpcLink::Statistics back = link_->statistics();
  EXPECT_TRUE(back.linkHealthy);
  EXPECT_GE(back.latePoliciesDropped + back.stalePoliciesDropped + robotBus_->topicStatistics(topics::kMpcPolicy).superseded,
            droppedBefore + 1);
  EXPECT_GT(server_->statistics().fullResets, fullResetsBefore) << "the hold ended on a policy solved without the full reset";
  EXPECT_EQ(robotSupervisor_.numFullResetsServed(), robotSupervisor_.resetsRequested().fullRequested);
  EXPECT_GT(link_->getCommand().mpcInitObservation_.time, newestBefore + linkConfig.policyTimeout)
      << "the hold ended on a policy solved before the link was lost, the one held in flight";
  EXPECT_LE(back.policyAge, linkConfig.policyTimeout);
  EXPECT_EQ(back.linkLosses, 1u);
}

}  // namespace
}  // namespace ocs2::humanoid::ipc
