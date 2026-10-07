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

// The robot side of the network MPC link against an MPC node the test scripts message by message: which policies the
// link accepts and which it drops, how it serves the reset handshake, how it mirrors the reported health and when the
// link counts as lost. The test thread is the robot's control thread.

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"
#include "humanoid_mpc_ipc/RemoteMpcLink.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/mpc_observation.pb.h"
#include "humanoid_mpc_msgs/mpc_policy.pb.h"
#include "humanoid_mpc_msgs/mpc_status.pb.h"
#include "humanoid_nmpc/humanoid_mpc_ipc/test/MpcLinkTestSupport.h"
#include "robot_ipc/Bus.h"
#include "robot_ipc/BusOptions.h"
#include "robot_ipc/Delivery.h"
#include "robot_ipc/NodeEndpoint.h"

namespace ocs2::humanoid::ipc {
namespace {

using test_support::makePolicyMessage;
using test_support::modelDimensions;
using test_support::observationAt;
using test_support::waitFor;
using ResetKind = MpcResetSupervisor::ResetKind;

/** An MPC node the test speaks for: it records the observations and publishes what the test gives it. */
class FakeMpc {
 public:
  FakeMpc() : bus_(test_support::createNodeBus("mpc")) {
    CHECK_OK(bus_->subscribe<humanoid_mpc_msgs::MpcObservation>(topics::kRobotMpcObservation, robot::ipc::Delivery::kAll,
                                                                [this](const humanoid_mpc_msgs::MpcObservation& message) {
                                                                  absl::MutexLock lock(mutex_);
                                                                  latest_ = message;
                                                                  ++received_;
                                                                }));
  }

  robot::ipc::Bus& bus() { return *bus_; }

  uint64_t received() const {
    absl::MutexLock lock(mutex_);
    return received_;
  }
  humanoid_mpc_msgs::MpcObservation latest() const {
    absl::MutexLock lock(mutex_);
    return latest_;
  }

 private:
  std::unique_ptr<robot::ipc::Bus> bus_;
  mutable absl::Mutex mutex_;
  humanoid_mpc_msgs::MpcObservation latest_ ABSL_GUARDED_BY(mutex_);
  uint64_t received_ ABSL_GUARDED_BY(mutex_) = 0;
};

humanoid_mpc_msgs::MpcStatus makeStatus(
    scalar_t observationTime, bool healthy, uint64_t consecutiveFailures, uint64_t solveCount, uint64_t serverInstance = 0) {
  humanoid_mpc_msgs::MpcStatus status;
  status.set_observation_time(observationTime);
  status.mutable_solver_status()->set_healthy(healthy);
  status.mutable_solver_status()->set_consecutive_failures(consecutiveFailures);
  status.mutable_solver_status()->set_solve_count(solveCount);
  status.mutable_solver_status()->set_server_instance(serverInstance);
  return status;
}

class RemoteMpcLinkProtocolTest : public ::testing::Test {
 protected:
  void SetUp() override { build(defaultConfig()); }
  // A test that stalls the IO thread must not leave it stalled: the bus joins it on destruction.
  void TearDown() override { ioGate_.open(); }

  static RemoteMpcLink::Config defaultConfig() {
    RemoteMpcLink::Config config;
    config.dimensions = modelDimensions();
    config.policyTimeout = 10.0;
    return config;
  }

  void build(RemoteMpcLink::Config config) {
    robotBus_ = test_support::createNodeBus("robot");
    test_support::connectBoth(*robotBus_, mpc_.bus());
    absl::StatusOr<std::unique_ptr<RemoteMpcLink>> link = RemoteMpcLink::Create(*robotBus_, supervisor_, config);
    CHECK_OK(link.status());
    link_ = std::move(*link);
    // Every poll of the robot's IO thread passes the gate, so that closing it stalls the thread the link runs on.
    CHECK_OK(robotBus_->addPeriodicCallback(absl::Milliseconds(1), [this]() { ioGate_.pass(); }));
    CHECK_OK(robotBus_->start());
    CHECK_OK(mpc_.bus().start());
    // Both directions are up once a message of each has gone through (ZeroMQ's "slow joiner").
    ASSERT_TRUE(waitFor([&]() {
      link_->setCurrentObservation(observationAt(kStartTime, /*value=*/1.0));
      CHECK_OK(mpc_.bus().publish(topics::kMpcStatus, makeStatus(/*observationTime=*/-1.0, /*healthy=*/true, /*consecutiveFailures=*/0,
                                                                 /*solveCount=*/0)));
      absl::SleepFor(absl::Milliseconds(2));
      return mpc_.received() > 0 && link_->statistics().statusesReceived > 0;
    }));
  }

  /** Writes the observation at `time` and waits until the MPC node has received it. */
  void sendObservation(scalar_t time) {
    link_->setCurrentObservation(observationAt(time, /*value=*/1.0));
    ASSERT_TRUE(waitFor([&]() { return mpc_.latest().observation().time() == time; })) << "t = " << time;
  }

  /** Publishes `policy` from the MPC node and waits until the link has handled it. */
  void sendPolicy(const humanoid_mpc_msgs::MpcPolicy& policy) {
    const uint64_t received = link_->statistics().policiesReceived;
    CHECK_OK(mpc_.bus().publish(topics::kMpcPolicy, policy));
    ASSERT_TRUE(waitFor([&]() { return link_->statistics().policiesReceived == received + 1; }));
  }

  void sendStatus(const humanoid_mpc_msgs::MpcStatus& status) {
    const uint64_t received = link_->statistics().statusesReceived;
    CHECK_OK(mpc_.bus().publish(topics::kMpcStatus, status));
    ASSERT_TRUE(waitFor([&]() { return link_->statistics().statusesReceived == received + 1; }));
  }

  /** A healthy policy solved from the observation at `initTime`, over one second, serving `resetsServed`. */
  static humanoid_mpc_msgs::MpcPolicy policyFrom(scalar_t initTime,
                                                 uint64_t solveCount,
                                                 uint64_t resetsServed = 0,
                                                 uint64_t fullResetsServed = 0) {
    return makePolicyMessage(initTime, initTime + 1.0, /*value=*/1.0, resetsServed, fullResetsServed, solveCount);
  }

  /** One control cycle of the robot at `time`, which does not wait for the IO thread: the realtime thread's own checks. */
  void controlCycle(scalar_t time) { link_->setCurrentObservation(observationAt(time, /*value=*/1.0)); }

  static constexpr scalar_t kStartTime = 1.0;

  test_support::Gate ioGate_;
  MpcResetSupervisor supervisor_;
  FakeMpc mpc_;
  std::unique_ptr<robot::ipc::Bus> robotBus_;
  std::unique_ptr<RemoteMpcLink> link_;
};

// ---------------------------------------------------------------------------------------------------------------------
// What the link sends
// ---------------------------------------------------------------------------------------------------------------------

TEST_F(RemoteMpcLinkProtocolTest, EveryObservationCarriesTheRequestCountersAndItsSequenceNumber) {
  const humanoid_mpc_msgs::MpcObservation first = mpc_.latest();
  EXPECT_EQ(first.resets().requested(), 0u);
  EXPECT_EQ(first.observation().state_size(), static_cast<int>(test_support::kStateDim));

  supervisor_.requestReset(ResetKind::kSolver);
  supervisor_.requestReset(ResetKind::kFull);
  sendObservation(/*time=*/1.1);
  const humanoid_mpc_msgs::MpcObservation second = mpc_.latest();
  EXPECT_EQ(second.resets().requested(), 2u);
  EXPECT_EQ(second.resets().full_requested(), 1u);
  EXPECT_GT(second.sequence(), first.sequence());

  // Counters only grow, so any later observation still carries every request.
  sendObservation(/*time=*/1.2);
  EXPECT_EQ(mpc_.latest().resets().requested(), 2u);
  EXPECT_EQ(mpc_.latest().resets().full_requested(), 1u);
}

TEST_F(RemoteMpcLinkProtocolTest, AnObservationOfOtherDimensionsIsNotSent) {
  const uint64_t published = link_->statistics().observationsPublished;
  SystemObservation wrong = observationAt(1.1, 1.0);
  wrong.state = vector_t::Zero(test_support::kStateDim + 1);
  link_->setCurrentObservation(wrong);
  ASSERT_TRUE(waitFor([&]() { return link_->statistics().latestObservationTime == 1.1; })) << "positive control: it was taken";
  sendObservation(/*time=*/1.2);
  EXPECT_EQ(link_->statistics().observationsPublished, published + 1) << "only the observation at t = 1.2 went out";
}

// ---------------------------------------------------------------------------------------------------------------------
// Which policies the link takes
// ---------------------------------------------------------------------------------------------------------------------

TEST_F(RemoteMpcLinkProtocolTest, APolicyIsTakenOnlyWhenSolvedFromAnObservationThisRobotSent) {
  sendObservation(/*time=*/1.5);
  sendPolicy(policyFrom(/*initTime=*/2.0, /*solveCount=*/1));
  sendPolicy(policyFrom(/*initTime=*/0.5, /*solveCount=*/2));
  EXPECT_EQ(link_->statistics().foreignPoliciesDropped, 2u) << "from the future, and from before the robot's first observation";
  EXPECT_FALSE(link_->initialPolicyReceived());

  sendPolicy(policyFrom(/*initTime=*/1.5, /*solveCount=*/3));
  EXPECT_EQ(link_->statistics().policiesAccepted, 1u);
  EXPECT_TRUE(link_->initialPolicyReceived());
  EXPECT_TRUE(link_->updatePolicy());
  EXPECT_TRUE(link_->isActivePolicyCurrent());
  EXPECT_EQ(link_->getCommand().mpcInitObservation_.time, 1.5);
}

TEST_F(RemoteMpcLinkProtocolTest, APolicyOfOtherDimensionsIsRejected) {
  humanoid_mpc_msgs::MpcPolicy policy = policyFrom(/*initTime=*/kStartTime, /*solveCount=*/1);
  policy.mutable_state_trajectory(0)->add_data(1.0);
  sendPolicy(policy);
  EXPECT_EQ(link_->statistics().invalidPoliciesRejected, 1u);
  EXPECT_FALSE(link_->initialPolicyReceived());
}

TEST_F(RemoteMpcLinkProtocolTest, APolicyWithAModeTheModelDoesNotHaveIsRejected) {
  // The controller looks the planned contacts up by mode (modeNumber2StanceLeg()), which knows the model's modes only.
  humanoid_mpc_msgs::MpcPolicy policy = policyFrom(/*initTime=*/kStartTime, /*solveCount=*/1);
  ASSERT_GT(policy.mode_schedule().mode_sequence_size(), 0);
  policy.mutable_mode_schedule()->set_mode_sequence(/*index=*/0, test_support::kNumModes);
  sendPolicy(policy);
  EXPECT_EQ(link_->statistics().invalidPoliciesRejected, 1u);
  EXPECT_FALSE(link_->initialPolicyReceived());

  sendPolicy(policyFrom(/*initTime=*/kStartTime, /*solveCount=*/2));
  EXPECT_TRUE(link_->initialPolicyReceived()) << "positive control: the same policy with a mode of the model is taken";
}

/** The protocol fixture with a policy timeout short enough to be reached. */
class RemoteMpcLinkShortTimeoutTest : public RemoteMpcLinkProtocolTest {
 protected:
  void SetUp() override {
    RemoteMpcLink::Config config = defaultConfig();
    config.policyTimeout = 0.1;
    build(config);
  }
};

TEST_F(RemoteMpcLinkShortTimeoutTest, APolicyOlderThanTheTimeoutOnArrivalIsDropped) {
  sendPolicy(policyFrom(/*initTime=*/kStartTime, /*solveCount=*/1));
  ASSERT_EQ(link_->statistics().policiesAccepted, 1u);
  sendObservation(/*time=*/kStartTime + 0.5);
  EXPECT_FALSE(supervisor_.isHealthy()) << "no policy for 0.5 s of robot time: the realtime thread's check, at once";
  ASSERT_TRUE(waitFor([&]() { return link_->statistics().linkLosses == 1; }));

  // Solved from an observation 0.5 s old, though after the reset the loss requested: it would not end the hold, so it
  // is not taken.
  sendPolicy(policyFrom(/*initTime=*/kStartTime, /*solveCount=*/2, /*resetsServed=*/1, /*fullResetsServed=*/1));
  EXPECT_EQ(link_->statistics().latePoliciesDropped, 1u);
  EXPECT_EQ(link_->statistics().policiesAccepted, 1u);
  EXPECT_FALSE(supervisor_.isHealthy());

  sendPolicy(policyFrom(/*initTime=*/kStartTime + 0.5, /*solveCount=*/3, /*resetsServed=*/1, /*fullResetsServed=*/1));
  EXPECT_EQ(link_->statistics().policiesAccepted, 2u);
  controlCycle(kStartTime + 0.5);
  EXPECT_TRUE(supervisor_.isHealthy());
}

TEST_F(RemoteMpcLinkShortTimeoutTest, APolicyWhoseHorizonEndedBeforeItArrivedIsDropped) {
  sendPolicy(policyFrom(/*initTime=*/kStartTime, /*solveCount=*/1));
  sendObservation(/*time=*/kStartTime + 0.05);
  // Within the timeout, but its horizon ended at t = kStartTime + 0.04.
  sendPolicy(makePolicyMessage(kStartTime + 0.01, /*finalTime=*/kStartTime + 0.04, /*value=*/1.0, /*resetsServed=*/0,
                               /*fullResetsServed=*/0, /*solveCount=*/2));
  EXPECT_EQ(link_->statistics().latePoliciesDropped, 1u);
  EXPECT_EQ(link_->statistics().policiesAccepted, 1u);
  EXPECT_EQ(link_->statistics().newestPolicyInitTime, kStartTime);
}

// ---------------------------------------------------------------------------------------------------------------------
// The reset handshake
// ---------------------------------------------------------------------------------------------------------------------

TEST_F(RemoteMpcLinkProtocolTest, AResetIsCompletedByTheFirstPolicyThatServesItAndNoEarlierOne) {
  sendPolicy(policyFrom(/*initTime=*/kStartTime, /*solveCount=*/1));
  ASSERT_TRUE(link_->updatePolicy());
  ASSERT_TRUE(link_->isActivePolicyCurrent());

  supervisor_.requestReset(ResetKind::kSolver);
  sendObservation(/*time=*/1.1);
  EXPECT_EQ(mpc_.latest().resets().requested(), 1u);
  EXPECT_FALSE(link_->isActivePolicyCurrent()) << "the request makes the policy in use stale at once";

  // Solved before the MPC node saw the request: dropped, and the reset stays outstanding.
  sendPolicy(policyFrom(/*initTime=*/1.1, /*solveCount=*/2, /*resetsServed=*/0));
  EXPECT_EQ(link_->statistics().stalePoliciesDropped, 1u);
  EXPECT_TRUE(supervisor_.hasOutstandingReset());
  EXPECT_FALSE(link_->updatePolicy());

  sendPolicy(policyFrom(/*initTime=*/1.1, /*solveCount=*/3, /*resetsServed=*/1));
  EXPECT_FALSE(supervisor_.hasOutstandingReset());
  EXPECT_EQ(supervisor_.numResetsServed(), 1u);
  EXPECT_TRUE(link_->updatePolicy());
  EXPECT_TRUE(link_->isActivePolicyCurrent());

  // A full request is served only by a policy that counts it among the full resets.
  supervisor_.requestReset(ResetKind::kFull);
  sendObservation(/*time=*/1.2);
  sendPolicy(policyFrom(/*initTime=*/1.2, /*solveCount=*/4, /*resetsServed=*/2, /*fullResetsServed=*/0));
  EXPECT_EQ(link_->statistics().stalePoliciesDropped, 2u);
  EXPECT_TRUE(supervisor_.hasOutstandingReset());
  sendPolicy(policyFrom(/*initTime=*/1.2, /*solveCount=*/5, /*resetsServed=*/2, /*fullResetsServed=*/1));
  EXPECT_FALSE(supervisor_.hasOutstandingReset());
  EXPECT_EQ(supervisor_.numFullResetsServed(), 1u);

  // Policies of the epoch keep being taken.
  sendObservation(/*time=*/1.3);
  sendPolicy(policyFrom(/*initTime=*/1.3, /*solveCount=*/6, /*resetsServed=*/2, /*fullResetsServed=*/1));
  EXPECT_EQ(link_->statistics().policiesAccepted, 4u) << "the first one, the two that served a reset, and this one";
  EXPECT_EQ(link_->statistics().resetTicketsTaken, 2u);
  EXPECT_EQ(link_->statistics().resetTicketsCompleted, 2u);
}

// ---------------------------------------------------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------------------------------------------------

TEST_F(RemoteMpcLinkProtocolTest, TheReportedHealthIsMirroredAndAFailureMakesThePolicyInUseStale) {
  sendPolicy(policyFrom(/*initTime=*/kStartTime, /*solveCount=*/5));
  ASSERT_TRUE(link_->updatePolicy());

  // A failed solve, not yet enough to declare the MPC unhealthy: as in process, the policy in use is no longer current.
  sendStatus(makeStatus(kStartTime, /*healthy=*/true, /*consecutiveFailures=*/1, /*solveCount=*/6));
  EXPECT_TRUE(supervisor_.isHealthy());
  EXPECT_EQ(supervisor_.numConsecutiveFailures(), 1u);
  EXPECT_FALSE(link_->isActivePolicyCurrent());
  EXPECT_FALSE(supervisor_.hasOutstandingReset());

  sendStatus(makeStatus(kStartTime, /*healthy=*/false, /*consecutiveFailures=*/3, /*solveCount=*/8));
  EXPECT_FALSE(supervisor_.isHealthy());
  EXPECT_FALSE(link_->statistics().solverHealthy);
  EXPECT_TRUE(link_->statistics().linkHealthy);

  // The status of an attempt older than the newest policy, and a status of another robot, say nothing.
  sendStatus(makeStatus(kStartTime, /*healthy=*/true, /*consecutiveFailures=*/0, /*solveCount=*/4));
  sendStatus(makeStatus(/*observationTime=*/7.0, /*healthy=*/true, /*consecutiveFailures=*/0, /*solveCount=*/99));
  EXPECT_FALSE(supervisor_.isHealthy());

  // The first policy of a solve that succeeded ends it.
  sendPolicy(policyFrom(/*initTime=*/kStartTime, /*solveCount=*/9));
  EXPECT_TRUE(supervisor_.isHealthy());
  EXPECT_EQ(supervisor_.numConsecutiveFailures(), 0u);
  EXPECT_TRUE(link_->updatePolicy());
  EXPECT_TRUE(link_->isActivePolicyCurrent());
}

TEST_F(RemoteMpcLinkProtocolTest, ARestartedMpcNodesStatusesAreMirroredBeforeItsFirstPolicy) {
  // The node that ran, at its 500th solve.
  humanoid_mpc_msgs::MpcPolicy policy = policyFrom(/*initTime=*/kStartTime, /*solveCount=*/500);
  policy.mutable_solver_status()->set_server_instance(/*value=*/11);
  sendPolicy(policy);
  ASSERT_TRUE(link_->updatePolicy());
  EXPECT_TRUE(link_->statistics().solverHealthy);

  // It restarts and keeps failing: its solves count from zero again, and it publishes no policy. Its statuses are what
  // the robot learns the solver's health from, though their solve counts are below the old node's.
  sendStatus(makeStatus(kStartTime, /*healthy=*/false, /*consecutiveFailures=*/3, /*solveCount=*/3, /*serverInstance=*/22));
  EXPECT_FALSE(link_->statistics().solverHealthy);
  EXPECT_FALSE(supervisor_.isHealthy());
  EXPECT_EQ(supervisor_.numConsecutiveFailures(), 3u);
  EXPECT_FALSE(link_->isActivePolicyCurrent()) << "the failure discards the policy in use, as for the node before";

  // Within one node, a status older than its newest policy still says nothing.
  humanoid_mpc_msgs::MpcPolicy recovered = policyFrom(/*initTime=*/kStartTime, /*solveCount=*/6);
  recovered.mutable_solver_status()->set_server_instance(/*value=*/22);
  sendPolicy(recovered);
  EXPECT_TRUE(supervisor_.isHealthy());
  sendStatus(makeStatus(kStartTime, /*healthy=*/false, /*consecutiveFailures=*/2, /*solveCount=*/5, /*serverInstance=*/22));
  EXPECT_TRUE(supervisor_.isHealthy());
}

TEST_F(RemoteMpcLinkProtocolTest, TheLinkIsLostAtTheEndOfTheNewestPolicysHorizon) {
  // A policy timeout far away; the horizon of the policy is short.
  sendPolicy(makePolicyMessage(kStartTime, /*finalTime=*/kStartTime + 0.1, /*value=*/1.0, /*resetsServed=*/0, /*fullResetsServed=*/0,
                               /*solveCount=*/1));
  ASSERT_TRUE(link_->updatePolicy());
  sendObservation(/*time=*/kStartTime + 0.05);
  EXPECT_TRUE(supervisor_.isHealthy());
  sendObservation(/*time=*/kStartTime + 0.1);
  EXPECT_FALSE(supervisor_.isHealthy()) << "the realtime thread's check: the cycle that reaches the end holds the robot";
  ASSERT_TRUE(waitFor([&]() { return link_->statistics().linkLosses == 1; }));
  EXPECT_FALSE(link_->statistics().linkHealthy);
  EXPECT_TRUE(link_->statistics().solverHealthy);
  EXPECT_FALSE(link_->isActivePolicyCurrent());

  // As for a solver declared unhealthy in process, the loss asks the MPC node for a full reset.
  EXPECT_TRUE(supervisor_.hasOutstandingReset());
  sendObservation(/*time=*/kStartTime + 0.11);
  EXPECT_EQ(mpc_.latest().resets().requested(), 1u);
  EXPECT_EQ(mpc_.latest().resets().full_requested(), 1u);

  // A fresh policy that the MPC node solved without the reset - its gait schedule ran on through the loss - does not
  // end the hold; the one solved after the reset does.
  sendPolicy(policyFrom(/*initTime=*/kStartTime + 0.11, /*solveCount=*/2));
  EXPECT_EQ(link_->statistics().stalePoliciesDropped, 1u);
  controlCycle(kStartTime + 0.11);
  EXPECT_FALSE(supervisor_.isHealthy());
  sendPolicy(policyFrom(/*initTime=*/kStartTime + 0.11, /*solveCount=*/3, /*resetsServed=*/1, /*fullResetsServed=*/1));
  controlCycle(kStartTime + 0.11);
  EXPECT_TRUE(supervisor_.isHealthy());
  EXPECT_TRUE(link_->statistics().linkHealthy);
  EXPECT_FALSE(supervisor_.hasOutstandingReset());
  EXPECT_EQ(supervisor_.numFullResetsServed(), 1u);
  EXPECT_TRUE(link_->updatePolicy());
  EXPECT_TRUE(link_->isActivePolicyCurrent());
  EXPECT_EQ(link_->statistics().linkLosses, 1u);
}

TEST_F(RemoteMpcLinkProtocolTest, TheRealtimeThreadHoldsTheRobotAtTheEndOfThePlanWhileTheIoThreadIsStalled) {
  sendPolicy(policyFrom(/*initTime=*/kStartTime, /*solveCount=*/1));  // Its horizon ends at kStartTime + 1.
  ASSERT_TRUE(link_->updatePolicy());

  // The IO thread stops at its next poll: nothing of the link runs on it from here on.
  const size_t arrivals = ioGate_.arrivals();
  ioGate_.close();
  ASSERT_TRUE(waitFor([&]() { return ioGate_.arrivals() > arrivals; }));
  controlCycle(kStartTime + 0.5);
  EXPECT_TRUE(supervisor_.isHealthy());
  controlCycle(kStartTime + 1.0);
  EXPECT_FALSE(supervisor_.isHealthy()) << "the controller would execute the plan past its end";
  absl::SleepFor(absl::Milliseconds(10));
  EXPECT_TRUE(link_->statistics().linkHealthy) << "positive control: the IO thread has not run";
  EXPECT_EQ(link_->statistics().linkLosses, 0u);
  EXPECT_FALSE(supervisor_.isHealthy());

  // Running again, the IO thread finds the same loss: it drops the buffered policy and requests the full reset.
  ioGate_.open();
  ASSERT_TRUE(waitFor([&]() { return link_->statistics().linkLosses == 1; }));
  EXPECT_FALSE(supervisor_.isHealthy());
  EXPECT_TRUE(supervisor_.hasOutstandingReset());
  EXPECT_FALSE(link_->isActivePolicyCurrent());

  sendObservation(/*time=*/kStartTime + 1.01);
  sendPolicy(policyFrom(/*initTime=*/kStartTime + 1.01, /*solveCount=*/2, /*resetsServed=*/1, /*fullResetsServed=*/1));
  controlCycle(kStartTime + 1.01);
  EXPECT_TRUE(supervisor_.isHealthy());
  EXPECT_TRUE(link_->updatePolicy());
  EXPECT_TRUE(link_->isActivePolicyCurrent());
}

TEST_F(RemoteMpcLinkShortTimeoutTest, TheRealtimeThreadHoldsTheRobotAtThePolicyTimeoutWhileTheIoThreadIsStalled) {
  sendPolicy(policyFrom(/*initTime=*/kStartTime, /*solveCount=*/1));
  const size_t arrivals = ioGate_.arrivals();
  ioGate_.close();
  ASSERT_TRUE(waitFor([&]() { return ioGate_.arrivals() > arrivals; }));
  controlCycle(kStartTime + 0.1);
  EXPECT_TRUE(supervisor_.isHealthy()) << "exactly the timeout old is not older than it";
  controlCycle(kStartTime + 0.1001);
  EXPECT_FALSE(supervisor_.isHealthy());
}

TEST_F(RemoteMpcLinkProtocolTest, AClockThatRunsBackwardsRestartsThePolicyTimeoutOnTheNewClock) {
  sendPolicy(policyFrom(/*initTime=*/kStartTime, /*solveCount=*/1));
  ASSERT_TRUE(link_->updatePolicy());
  // The simulation restarts at t = 0: the policy solved at t = 1 is not of this clock, and neither is its horizon.
  sendObservation(/*time=*/0.0);
  ASSERT_TRUE(waitFor([&]() { return std::isnan(link_->statistics().newestPolicyInitTime); }));
  EXPECT_TRUE(supervisor_.isHealthy()) << "the timeout counts from the rewind";
  sendPolicy(policyFrom(/*initTime=*/kStartTime, /*solveCount=*/2));
  EXPECT_EQ(link_->statistics().foreignPoliciesDropped, 1u) << "a policy of the old clock";
  // Past the old policy's init time on the new clock: still no policy of this clock, so no age, not 1.5 - 1.0.
  sendObservation(/*time=*/1.5);
  EXPECT_LT(link_->statistics().policyAge, 0.0);
  EXPECT_TRUE(std::isnan(link_->statistics().newestPolicyInitTime));

  sendObservation(/*time=*/10.5);
  EXPECT_FALSE(supervisor_.isHealthy()) << "no policy for longer than the timeout on the new clock";
  ASSERT_TRUE(waitFor([&]() { return link_->statistics().linkLosses == 1; }));
  sendPolicy(policyFrom(/*initTime=*/10.5, /*solveCount=*/3, /*resetsServed=*/1, /*fullResetsServed=*/1));
  controlCycle(10.5);
  EXPECT_TRUE(supervisor_.isHealthy());
  EXPECT_EQ(link_->statistics().newestPolicyInitTime, 10.5);
  EXPECT_EQ(link_->statistics().policyAge, 0.0);
}

TEST_F(RemoteMpcLinkProtocolTest, ARewindDoesNotEndALoss) {
  sendPolicy(makePolicyMessage(kStartTime, /*finalTime=*/kStartTime + 0.1, /*value=*/1.0, /*resetsServed=*/0, /*fullResetsServed=*/0,
                               /*solveCount=*/1));
  sendObservation(/*time=*/kStartTime + 0.2);
  ASSERT_TRUE(waitFor([&]() { return link_->statistics().linkLosses == 1; }));

  // No policy has arrived: the restarted clock, which restarts the timeout, does not make the link healthy.
  sendObservation(/*time=*/0.0);
  ASSERT_TRUE(waitFor([&]() { return std::isnan(link_->statistics().newestPolicyInitTime); }));
  absl::SleepFor(absl::Milliseconds(10));
  EXPECT_FALSE(link_->statistics().linkHealthy);
  EXPECT_FALSE(supervisor_.isHealthy());
  EXPECT_EQ(link_->statistics().linkLosses, 1u);

  sendPolicy(policyFrom(/*initTime=*/0.0, /*solveCount=*/2, /*resetsServed=*/1, /*fullResetsServed=*/1));
  controlCycle(0.0);
  EXPECT_TRUE(link_->statistics().linkHealthy);
  EXPECT_TRUE(supervisor_.isHealthy());
}

TEST_F(RemoteMpcLinkProtocolTest, AStepBackWithinTheRewindToleranceIsNotARewind) {
  sendPolicy(policyFrom(/*initTime=*/kStartTime, /*solveCount=*/1));
  sendObservation(/*time=*/1.2);
  const scalar_t jitter = 0.5 * supervisor_.getConfig().clockRewindTolerance;
  sendObservation(/*time=*/1.2 - jitter);
  absl::SleepFor(absl::Milliseconds(10));
  // The newest time stays the newest, the newest policy stays of this clock, and its timeout and horizon still count.
  EXPECT_EQ(link_->statistics().latestObservationTime, 1.2);
  EXPECT_EQ(link_->statistics().newestPolicyInitTime, kStartTime);
  sendPolicy(policyFrom(/*initTime=*/1.2, /*solveCount=*/2));
  EXPECT_EQ(link_->statistics().foreignPoliciesDropped, 0u);
  EXPECT_EQ(link_->statistics().policiesAccepted, 2u);

  // A rewind by more than the tolerance is one: a policy of before it is foreign.
  sendObservation(/*time=*/1.2 - 2.0 * supervisor_.getConfig().clockRewindTolerance - 0.01);
  sendPolicy(policyFrom(/*initTime=*/1.2, /*solveCount=*/3));
  EXPECT_EQ(link_->statistics().foreignPoliciesDropped, 1u);
}

// ---------------------------------------------------------------------------------------------------------------------
// Construction and lifetime
// ---------------------------------------------------------------------------------------------------------------------

TEST(RemoteMpcLinkCreateTest, RefusesARunningBusAndAnInvalidConfig) {
  MpcResetSupervisor supervisor;
  std::unique_ptr<robot::ipc::Bus> bus = test_support::createNodeBus("robot");
  RemoteMpcLink::Config config;
  config.dimensions = modelDimensions();

  RemoteMpcLink::Config noDimensions = config;
  noDimensions.dimensions.stateDim = 0;
  EXPECT_EQ(RemoteMpcLink::Create(*bus, supervisor, noDimensions).status().code(), absl::StatusCode::kInvalidArgument);
  RemoteMpcLink::Config noModes = config;
  noModes.dimensions.numModes = 0;
  EXPECT_EQ(RemoteMpcLink::Create(*bus, supervisor, noModes).status().code(), absl::StatusCode::kInvalidArgument);
  RemoteMpcLink::Config noTimeout = config;
  noTimeout.policyTimeout = 0.0;
  EXPECT_EQ(RemoteMpcLink::Create(*bus, supervisor, noTimeout).status().code(), absl::StatusCode::kInvalidArgument);
  RemoteMpcLink::Config noPoll = config;
  noPoll.pollPeriod = absl::ZeroDuration();
  EXPECT_EQ(RemoteMpcLink::Create(*bus, supervisor, noPoll).status().code(), absl::StatusCode::kInvalidArgument);

  CHECK_OK(bus->start());
  EXPECT_EQ(RemoteMpcLink::Create(*bus, supervisor, config).status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST_F(RemoteMpcLinkProtocolTest, ABusThatOutlivesTheLinkCallsNothingOfIt) {
  sendPolicy(policyFrom(/*initTime=*/kStartTime, /*solveCount=*/1));
  const uint64_t delivered = robotBus_->topicStatistics(topics::kMpcStatus).delivered;
  link_.reset();
  // The bus keeps polling and delivering; the guard makes both no-ops.
  for (int message = 0; message < 20; ++message) {
    CHECK_OK(mpc_.bus().publish(topics::kMpcPolicy, policyFrom(/*initTime=*/kStartTime, /*solveCount=*/2)));
    CHECK_OK(mpc_.bus().publish(topics::kMpcStatus, makeStatus(kStartTime, /*healthy=*/false, /*consecutiveFailures=*/5,
                                                               /*solveCount=*/3)));
  }
  ASSERT_TRUE(waitFor([&]() { return robotBus_->topicStatistics(topics::kMpcStatus).delivered >= delivered + 1; }));
  absl::SleepFor(absl::Milliseconds(20));
  EXPECT_TRUE(robotBus_->isRunning());
  EXPECT_TRUE(supervisor_.isHealthy()) << "a destroyed link changed the supervisor";
}

TEST(RemoteMpcLinkOwnBusTest, ALinkWithABusOfItsOwnSendsAndReceives) {
  FakeMpc mpc;
  MpcResetSupervisor supervisor;
  RemoteMpcLink::Config config;
  config.dimensions = modelDimensions();
  // The link's bus binds a port of its own choosing; a random high one, retried if it is taken.
  std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<int> ports(20000, 60000);
  std::unique_ptr<RemoteMpcLink> link;
  int port = 0;
  for (int attempt = 0; attempt < 20 && link == nullptr; ++attempt) {
    port = ports(generator);
    robot::ipc::BusOptions options;
    options.nodeName = "robot";
    options.network.nodes = {robot::ipc::NodeEndpoint{.name = "robot", .host = "127.0.0.1", .port = port, .bindHost = ""},
                             robot::ipc::NodeEndpoint{.name = "mpc", .host = "127.0.0.1", .port = mpc.bus().boundPort(), .bindHost = ""}};
    absl::StatusOr<std::unique_ptr<RemoteMpcLink>> created = RemoteMpcLink::Create(std::move(options), supervisor, config);
    if (created.ok()) {
      link = std::move(*created);
    } else {
      ASSERT_EQ(created.status().code(), absl::StatusCode::kUnavailable) << created.status();
    }
  }
  ASSERT_NE(link, nullptr);
  CHECK_OK(mpc.bus().connect(absl::StrCat("tcp://127.0.0.1:", port)));
  CHECK_OK(mpc.bus().start());

  ASSERT_TRUE(waitFor([&]() {
    link->setCurrentObservation(observationAt(1.0, 1.0));
    CHECK_OK(mpc.bus().publish(topics::kMpcPolicy, makePolicyMessage(1.0, 2.0, 1.0, 0, 0, 1)));
    absl::SleepFor(absl::Milliseconds(2));
    return link->statistics().policiesAccepted > 0;
  }));
  EXPECT_GT(mpc.received(), 0u);
  EXPECT_TRUE(link->updatePolicy());
  link.reset();  // Stops the bus it owns.
}

}  // namespace
}  // namespace ocs2::humanoid::ipc
