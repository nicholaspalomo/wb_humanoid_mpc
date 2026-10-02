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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <optional>
#include <thread>
#include <vector>

#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"

#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"

/*
 * The reset and failure policy of the MRT joint controllers. A solver that failed used to request a reset after every
 * failure and to log each one, so a failure the reset did not cure became a loop of ~100 errors a second for ever, while
 * the robot executed the last policy that had been solved.
 */

namespace ocs2::humanoid {
namespace {

using ::testing::_;
using ::testing::HasSubstr;

const absl::Status kFailure = absl::InternalError("MPC solver crashed: [SqpSolver] Failed to solve QP");

MpcResetSupervisor::Config testConfig() {
  MpcResetSupervisor::Config config;
  config.maxConsecutiveFailures = 3;
  config.initialRetryInterval = 0.1;
  config.maxRetryInterval = 0.4;
  return config;
}

using ResetKind = MpcResetSupervisor::ResetKind;
using ResetTicket = MpcResetSupervisor::ResetTicket;

/** Serves an outstanding reset as the solver thread does; returns whether there was one. */
bool serve(MpcResetSupervisor& supervisor) {
  const std::optional<ResetTicket> ticket = supervisor.takeResetRequest();
  if (!ticket.has_value()) return false;
  supervisor.completeReset(*ticket);
  return true;
}

/** Serves an outstanding reset and returns whether it was a full one; empty when there was none. */
std::optional<bool> serveKind(MpcResetSupervisor& supervisor) {
  const std::optional<ResetTicket> ticket = supervisor.takeResetRequest();
  if (!ticket.has_value()) return std::nullopt;
  supervisor.completeReset(*ticket);
  return ticket->full;
}

TEST(MpcResetSupervisor, AResetIsOutstandingFromTheRequestUntilItIsServed) {
  MpcResetSupervisor supervisor;
  EXPECT_FALSE(supervisor.hasOutstandingReset());
  EXPECT_FALSE(supervisor.takeResetRequest().has_value());

  supervisor.requestReset();
  EXPECT_TRUE(supervisor.hasOutstandingReset());
  const std::optional<ResetTicket> ticket = supervisor.takeResetRequest();
  ASSERT_TRUE(ticket.has_value());
  EXPECT_TRUE(supervisor.hasOutstandingReset()) << "taking a request is not serving it";
  supervisor.completeReset(*ticket);
  EXPECT_FALSE(supervisor.hasOutstandingReset());
  EXPECT_EQ(supervisor.numResetsServed(), 1u);
}

TEST(MpcResetSupervisor, ARequestMadeWhileAResetIsServedIsServedNext) {
  // The control thread asks again while the solver thread is resetting: the reset in progress was started from an
  // observation older than the second request, so the second one must not count as served by it.
  MpcResetSupervisor supervisor;
  supervisor.requestReset();
  const std::optional<ResetTicket> first = supervisor.takeResetRequest();
  ASSERT_TRUE(first.has_value());
  supervisor.requestReset();
  supervisor.completeReset(*first);
  EXPECT_TRUE(supervisor.hasOutstandingReset());
  EXPECT_TRUE(serve(supervisor));
  EXPECT_FALSE(supervisor.hasOutstandingReset());
}

TEST(MpcResetSupervisor, SeveralRequestsBeforeTheSolverLooksAreServedByOneReset) {
  MpcResetSupervisor supervisor;
  supervisor.requestReset();
  supervisor.requestReset();
  supervisor.requestReset();
  EXPECT_TRUE(serve(supervisor));
  EXPECT_FALSE(serve(supervisor));
  EXPECT_EQ(supervisor.numResetsServed(), 3u) << "the counter records the last request served";
}

TEST(MpcResetSupervisor, AFullAndASolverResetOutstandingTogetherAreServedAsOneFullReset) {
  MpcResetSupervisor supervisor;
  supervisor.requestReset(ResetKind::kSolver);
  EXPECT_EQ(serveKind(supervisor), std::optional<bool>(false));
  supervisor.requestReset(ResetKind::kSolver);
  supervisor.requestReset(ResetKind::kFull);
  supervisor.requestReset(ResetKind::kSolver);
  EXPECT_EQ(serveKind(supervisor), std::optional<bool>(true));
  EXPECT_FALSE(serveKind(supervisor).has_value());
  EXPECT_EQ(supervisor.numFullResetsServed(), 1u);
  // A full request is not taken as served by a solver reset that was already under way.
  supervisor.requestReset(ResetKind::kSolver);
  const std::optional<ResetTicket> solverOnly = supervisor.takeResetRequest();
  ASSERT_TRUE(solverOnly.has_value());
  EXPECT_FALSE(solverOnly->full);
  supervisor.requestReset(ResetKind::kFull);
  supervisor.completeReset(*solverOnly);
  EXPECT_EQ(serveKind(supervisor), std::optional<bool>(true));
}

TEST(MpcResetSupervisor, TheFirstFailuresResetTheSolverAndTheEscalationResetsEverything) {
  // A robot in mid-stride keeps the schedule it is executing through a failed solve; failures that persist come from
  // state the solves inherit, which only the full reset clears.
  MpcResetSupervisor supervisor(testConfig());
  supervisor.onSolveResult(kFailure);
  EXPECT_EQ(serveKind(supervisor), std::optional<bool>(false));
  supervisor.onSolveResult(kFailure);
  EXPECT_EQ(serveKind(supervisor), std::optional<bool>(false));
  supervisor.onSolveResult(kFailure);
  EXPECT_EQ(serveKind(supervisor), std::optional<bool>(true));
  supervisor.onSolveResult(kFailure);
  EXPECT_EQ(serveKind(supervisor), std::optional<bool>(true));
}

TEST(MpcResetSupervisor, FailuresRetryAtOnceThenBackOffExponentiallyUpToTheCap) {
  MpcResetSupervisor supervisor(testConfig());
  std::vector<scalar_t> waits;
  for (int failure = 0; failure < 7; ++failure) {
    waits.push_back(supervisor.onSolveResult(kFailure).count());
    EXPECT_TRUE(serve(supervisor)) << "every failure requests a reset, failure " << failure + 1;
  }
  EXPECT_DOUBLE_EQ(waits[0], 0.0);
  EXPECT_DOUBLE_EQ(waits[1], 0.0);
  // From the maxConsecutiveFailures-th on: the initial interval, doubled per failure, capped.
  EXPECT_DOUBLE_EQ(waits[2], 0.1);
  EXPECT_DOUBLE_EQ(waits[3], 0.2);
  EXPECT_DOUBLE_EQ(waits[4], 0.4);
  EXPECT_DOUBLE_EQ(waits[5], 0.4);
  EXPECT_DOUBLE_EQ(waits[6], 0.4);
}

TEST(MpcResetSupervisor, TheMpcIsUnhealthyFromTheLastImmediateRetryUntilASolveSucceeds) {
  MpcResetSupervisor supervisor(testConfig());
  EXPECT_TRUE(supervisor.isHealthy());
  supervisor.onSolveResult(kFailure);
  supervisor.onSolveResult(kFailure);
  EXPECT_TRUE(supervisor.isHealthy()) << "two failures in a row are still retried at once";
  supervisor.onSolveResult(kFailure);
  EXPECT_FALSE(supervisor.isHealthy());
  EXPECT_EQ(supervisor.numConsecutiveFailures(), 3u);
  supervisor.onSolveResult(kFailure);
  EXPECT_FALSE(supervisor.isHealthy());

  EXPECT_DOUBLE_EQ(supervisor.onSolveResult(absl::OkStatus()).count(), 0.0);
  EXPECT_TRUE(supervisor.isHealthy());
  EXPECT_EQ(supervisor.numConsecutiveFailures(), 0u);
  // A failure after the recovery starts a new count.
  EXPECT_DOUBLE_EQ(supervisor.onSolveResult(kFailure).count(), 0.0);
  EXPECT_TRUE(supervisor.isHealthy());
}

TEST(MpcResetSupervisor, APersistentFailureLogsOneErrorNotOnePerAttempt) {
  MpcResetSupervisor supervisor(testConfig());
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  // gMock tries the expectation declared last first: the one error, saying how to recover, and no other.
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _, _)).Times(0);
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _, HasSubstr("switch to JOINT_PD and back to WB_MPC"))).Times(1);
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _, HasSubstr("Failed to solve QP"))).Times(2);
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, _, HasSubstr("recovered"))).Times(1);
  log.StartCapturingLogs();
  for (int failure = 0; failure < 200; ++failure) supervisor.onSolveResult(kFailure);
  supervisor.onSolveResult(absl::OkStatus());
  log.StopCapturingLogs();
}

TEST(MpcResetSupervisor, TheBackOffEndsEarlyWhenAResetIsRequestedFromOutside) {
  MpcResetSupervisor supervisor(testConfig());
  for (int failure = 0; failure < 3; ++failure) supervisor.onSolveResult(kFailure);
  serve(supervisor);

  // Nothing requested: the whole wait.
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  supervisor.waitBeforeRetry(std::chrono::duration<scalar_t>(0.2), []() { return false; });
  EXPECT_GE(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(190));

  // An operator re-entering WB_MPC asks for a reset: the wait ends at once.
  supervisor.onSolveResult(kFailure);
  start = std::chrono::steady_clock::now();
  std::thread requester([&supervisor]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    supervisor.requestReset();
  });
  supervisor.waitBeforeRetry(std::chrono::duration<scalar_t>(5.0), []() { return false; });
  requester.join();
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(1));

  // And so does a stop request (the controller being destroyed).
  supervisor.onSolveResult(kFailure);
  std::atomic<bool> stop{false};
  start = std::chrono::steady_clock::now();
  std::thread stopper([&stop]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    stop.store(true);
  });
  supervisor.waitBeforeRetry(std::chrono::duration<scalar_t>(5.0), [&stop]() { return stop.load(); });
  stopper.join();
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(1));
}

TEST(MpcResetSupervisor, AResetRequestedSinceTheLastFailureIsWhatEndsABackOff) {
  // What a caller that waits out the back-off on its own clock polls (InProcessMpcLink::Execution::kCaller).
  MpcResetSupervisor supervisor(testConfig());
  for (int failure = 0; failure < 3; ++failure) supervisor.onSolveResult(kFailure);
  EXPECT_FALSE(supervisor.resetRequestedSinceLastFailure()) << "the failure's own request does not count";
  serve(supervisor);
  EXPECT_FALSE(supervisor.resetRequestedSinceLastFailure()) << "serving it does not either";
  supervisor.requestReset(MpcResetSupervisor::ResetKind::kSolver);
  EXPECT_TRUE(supervisor.resetRequestedSinceLastFailure());
  // The next failure starts the count again; the wait then ends at once on a request, as the flag says.
  supervisor.onSolveResult(kFailure);
  EXPECT_FALSE(supervisor.resetRequestedSinceLastFailure());
  supervisor.requestReset();
  EXPECT_TRUE(supervisor.resetRequestedSinceLastFailure());
  const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  supervisor.waitBeforeRetry(std::chrono::duration<scalar_t>(5.0), []() { return false; });
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(1));
}

TEST(MpcResetSupervisor, AClockThatRunsBackwardsRequestsOneReset) {
  MpcResetSupervisor supervisor;
  EXPECT_DOUBLE_EQ(supervisor.observeTime(10.0), 0.0);
  EXPECT_DOUBLE_EQ(supervisor.observeTime(10.01), 0.0);
  EXPECT_DOUBLE_EQ(supervisor.observeTime(10.01), 0.0) << "a repeated time is not a rewind";
  EXPECT_FALSE(supervisor.hasOutstandingReset());

  EXPECT_NEAR(supervisor.observeTime(2.0), 8.01, 1e-12);
  EXPECT_EQ(serveKind(supervisor), std::optional<bool>(true)) << "a rewind is a full reset";
  EXPECT_FALSE(serve(supervisor)) << "one rewind, one reset";
  EXPECT_DOUBLE_EQ(supervisor.observeTime(2.01), 0.0) << "the new clock runs forward from there";
  EXPECT_FALSE(supervisor.hasOutstandingReset());
}

// ---------------------------------------------------------------------------------------------------------------------
// The link to a remote solver (RemoteMpcLink)
// ---------------------------------------------------------------------------------------------------------------------

TEST(MpcResetSupervisor, TheRequestCountersOnlyGrowAndCountFullRequestsApart) {
  MpcResetSupervisor supervisor;
  EXPECT_EQ(supervisor.resetsRequested().requested, 0u);
  EXPECT_EQ(supervisor.resetsRequested().fullRequested, 0u);
  supervisor.requestReset(ResetKind::kSolver);
  supervisor.requestReset(ResetKind::kFull);
  supervisor.requestReset(ResetKind::kSolver);
  EXPECT_EQ(supervisor.resetsRequested().requested, 3u);
  EXPECT_EQ(supervisor.resetsRequested().fullRequested, 1u);
  // Serving a reset changes what is served, never what was requested: a lost observation carrying the counters is
  // replaced by any later one.
  EXPECT_TRUE(serve(supervisor));
  EXPECT_EQ(supervisor.resetsRequested().requested, 3u);
  EXPECT_EQ(supervisor.resetsRequested().fullRequested, 1u);
  EXPECT_EQ(supervisor.numResetsServed(), 3u);
  EXPECT_EQ(supervisor.numFullResetsServed(), 1u);
}

TEST(MpcResetSupervisor, TheRemoteHealthIsWhatIsHealthyReports) {
  MpcResetSupervisor supervisor(testConfig());
  supervisor.setRemoteHealth(/*healthy=*/false, /*consecutiveFailures=*/4);
  EXPECT_FALSE(supervisor.isHealthy());
  EXPECT_EQ(supervisor.numConsecutiveFailures(), 4u);
  EXPECT_FALSE(supervisor.hasOutstandingReset()) << "the remote health requests nothing: the MPC side resets itself";
  supervisor.setRemoteHealth(/*healthy=*/true, /*consecutiveFailures=*/0);
  EXPECT_TRUE(supervisor.isHealthy());
  EXPECT_EQ(supervisor.numConsecutiveFailures(), 0u);
}

TEST(MpcResetSupervisor, AnExpiredRemotePolicyIsUnhealthyWhateverTheRemoteHealthSays) {
  MpcResetSupervisor supervisor(testConfig());
  supervisor.setRemoteHealth(/*healthy=*/true, /*consecutiveFailures=*/0);
  supervisor.setRemotePolicyExpired(/*expired=*/true);
  EXPECT_FALSE(supervisor.isHealthy());
  // The link's IO thread, which may not have seen the expiry yet, does not overrule the control thread.
  supervisor.setRemoteHealth(/*healthy=*/true, /*consecutiveFailures=*/0);
  EXPECT_FALSE(supervisor.isHealthy());
  EXPECT_EQ(supervisor.numConsecutiveFailures(), 0u) << "an expired policy is not a failed solve";
  EXPECT_FALSE(supervisor.hasOutstandingReset());
  supervisor.setRemotePolicyExpired(/*expired=*/false);
  EXPECT_TRUE(supervisor.isHealthy());
  // Both must agree for the MPC to be healthy.
  supervisor.setRemoteHealth(/*healthy=*/false, /*consecutiveFailures=*/3);
  EXPECT_FALSE(supervisor.isHealthy());
}

TEST(MpcResetSupervisor, SettingTheRemoteHealthLogsNothing) {
  // The link logs each transition once, with its cause; the supervisor must not add a line per update, which the link
  // makes at every poll of the bus.
  MpcResetSupervisor supervisor(testConfig());
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(_, _, _)).Times(0);
  log.StartCapturingLogs();
  for (int update = 0; update < 100; ++update) supervisor.setRemoteHealth(/*healthy=*/update % 2 == 0, /*consecutiveFailures=*/update);
  log.StopCapturingLogs();
}

}  // namespace
}  // namespace ocs2::humanoid
