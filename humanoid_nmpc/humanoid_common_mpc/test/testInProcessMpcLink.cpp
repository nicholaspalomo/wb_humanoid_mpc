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
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <ocs2_mpc/MPC_Settings.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_mpc_test/ScriptedMpc.h>
#include <ocs2_oc/synchronized_module/SolverSynchronizedModule.h>

#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"

#include "humanoid_common_mpc/mrt/InProcessMpcLink.h"
#include "humanoid_common_mpc/mrt/MpcLink.h"

/*
 * InProcessMpcLink, the solver thread the MRT joint controllers used to run themselves (solverWorker()), around a
 * scripted solver: the start-up reset, the requested resets of either kind served before the next solve from the
 * observation current then, the failure back-off and a reset request cutting it short, the pacing at the desired
 * frequency or back to back, the solve observer, and stop(). Then the same iteration run by the caller
 * (Execution::kCaller), as the lockstep closed loop of humanoid_mpc_validation runs it.
 */

namespace ocs2::humanoid {
namespace {

using ::testing::_;
using ::testing::HasSubstr;

constexpr size_t kStateDim = 3;
constexpr size_t kInputDim = 2;

SystemObservation observationAt(scalar_t time) {
  SystemObservation observation;
  observation.time = time;
  observation.state = vector_t::Constant(kStateDim, time);
  observation.input = vector_t::Zero(kInputDim);
  return observation;
}

/** Counts the resets of the synchronized modules, which only a full reset (MPC_BASE::reset()) performs. */
class ResetCountingModule final : public SolverSynchronizedModule {
 public:
  void preSolverRun(scalar_t /*initTime*/,
                    scalar_t /*finalTime*/,
                    const vector_t& /*initState*/,
                    const ReferenceManagerInterface& /*referenceManager*/) override {}
  void postSolverRun(const PrimalSolution& /*primalSolution*/) override {}
  void reset() override { ++numResets; }
  std::atomic<size_t> numResets{0};
};

/** The observations the link's reset target was asked for, in order. */
class ResetTargetRecorder {
 public:
  MpcLink::ResetTargetFunction function() {
    return [this](const SystemObservation& observation) {
      std::lock_guard<std::mutex> lock(mutex_);
      times_.push_back(observation.time);
      return TargetTrajectories({observation.time}, {observation.state}, {vector_t::Zero(kInputDim)});
    };
  }
  std::vector<scalar_t> times() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return times_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<scalar_t> times_;
};

bool waitFor(const std::function<bool()>& done, std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
  const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + timeout;
  while (!done()) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

class InProcessMpcLinkTest : public ::testing::Test {
 protected:
  InProcessMpcLinkTest() : mpc_(mpc::Settings(), kInputDim), module_(std::make_shared<ResetCountingModule>()) {
    mpc_.getSolverPtr()->addSynchronizedModule(module_);
  }

  std::unique_ptr<InProcessMpcLink> makeLink(scalar_t mpcDesiredFrequency, InProcessMpcLink::SolveObserver observer = nullptr) {
    InProcessMpcLink::Config config;
    config.mpcDesiredFrequency = mpcDesiredFrequency;
    config.solverThreadName = "Test MPC Solver Thread";
    config.solveObserver = std::move(observer);
    return std::make_unique<InProcessMpcLink>(mpc_, resetTargets_.function(), std::move(config));
  }

  /** A link whose iterations the test runs itself (Execution::kCaller). */
  std::unique_ptr<InProcessMpcLink> makeCallerLink(InProcessMpcLink::SolveObserver observer = nullptr) {
    InProcessMpcLink::Config config;
    config.solveObserver = std::move(observer);
    config.execution = InProcessMpcLink::Execution::kCaller;
    return std::make_unique<InProcessMpcLink>(mpc_, resetTargets_.function(), std::move(config));
  }

  /** Waits until the control thread can take a policy solved after the last reset into use. */
  static bool waitForCurrentPolicy(MpcLink& link) {
    return waitFor([&link]() {
      link.updatePolicy();
      return link.initialPolicyReceived() && link.isActivePolicyCurrent() && !link.hasOutstandingReset();
    });
  }

  mpc_test::ScriptedMpc mpc_;
  std::shared_ptr<ResetCountingModule> module_;
  ResetTargetRecorder resetTargets_;
};

TEST_F(InProcessMpcLinkTest, StartResetsFullyFromTheInitialObservationAndServesPolicies) {
  std::unique_ptr<InProcessMpcLink> link = makeLink(/*mpcDesiredFrequency=*/1000.0);
  EXPECT_FALSE(link->initialPolicyReceived());
  link->start(observationAt(/*time=*/1.0));
  ASSERT_TRUE(waitForCurrentPolicy(*link)) << "the solver thread produced no policy";

  ASSERT_FALSE(resetTargets_.times().empty());
  EXPECT_EQ(resetTargets_.times().front(), 1.0) << "the start-up reset was not served from the initial observation";
  EXPECT_EQ(module_->numResets.load(), 1u) << "the start-up reset is a full one";
  EXPECT_EQ(link->getResetSupervisor().numResetsServed(), 0u) << "the start-up reset is no requested one";
  EXPECT_EQ(link->getCommand().mpcInitObservation_.time, 1.0);
  EXPECT_TRUE(link->isHealthy());
}

TEST_F(InProcessMpcLinkTest, ARequestedResetIsServedBeforeTheNextSolveFromTheCurrentObservation) {
  std::unique_ptr<InProcessMpcLink> link = makeLink(/*mpcDesiredFrequency=*/1000.0);
  link->start(observationAt(/*time=*/1.0));
  ASSERT_TRUE(waitForCurrentPolicy(*link));

  // Of the solver alone: the modules keep their state, and the policy in use stops being current until a policy solved
  // after the reset replaces it.
  link->setCurrentObservation(observationAt(/*time=*/2.0));
  link->requestReset(MpcLink::ResetKind::kSolver);
  ASSERT_TRUE(waitForCurrentPolicy(*link)) << "the requested reset was never served";
  EXPECT_EQ(link->getResetSupervisor().numResetsServed(), 1u);
  EXPECT_EQ(link->getResetSupervisor().numFullResetsServed(), 0u);
  EXPECT_EQ(module_->numResets.load(), 1u) << "a reset of the solver alone reset the synchronized modules";
  EXPECT_EQ(resetTargets_.times().back(), 2.0) << "the reset was not served from the observation current then";
  EXPECT_GE(link->getCommand().mpcInitObservation_.time, 2.0);

  // A full one.
  link->setCurrentObservation(observationAt(/*time=*/3.0));
  link->requestReset(MpcLink::ResetKind::kFull);
  ASSERT_TRUE(waitForCurrentPolicy(*link));
  EXPECT_EQ(link->getResetSupervisor().numResetsServed(), 2u);
  EXPECT_EQ(link->getResetSupervisor().numFullResetsServed(), 1u);
  EXPECT_EQ(module_->numResets.load(), 2u) << "a full reset did not reset the synchronized modules";
  EXPECT_EQ(resetTargets_.times().back(), 3.0);
}

TEST_F(InProcessMpcLinkTest, RepeatedFailuresBackOffAndAResetRequestCutsTheWaitShort) {
  std::unique_ptr<InProcessMpcLink> link = makeLink(/*mpcDesiredFrequency=*/1000.0);
  link->start(observationAt(/*time=*/1.0));
  ASSERT_TRUE(waitForCurrentPolicy(*link));

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _, HasSubstr("switch to JOINT_PD and back to WB_MPC"))).Times(1);
  log.StartCapturingLogs();

  mpc_.solver().failEverySolve(true);
  ASSERT_TRUE(waitFor([&link]() { return !link->isHealthy(); })) << "persistent failures never made the MPC unhealthy";
  const size_t failedWhenUnhealthy = mpc_.solver().numFailedSolves();
  EXPECT_EQ(failedWhenUnhealthy, link->getResetSupervisor().getConfig().maxConsecutiveFailures);
  // Backing off from 0.1 s, doubled per failure: attempts after 0.1, 0.3 and 0.7 s, then a wait until 1.5 s; not one
  // per period of the 1 kHz loop.
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  const size_t failedAfterBackOff = mpc_.solver().numFailedSolves();
  EXPECT_GE(failedAfterBackOff - failedWhenUnhealthy, 2u) << "the solver stopped retrying";
  EXPECT_LE(failedAfterBackOff - failedWhenUnhealthy, 8u) << "the attempts did not back off";
  EXPECT_GE(link->getResetSupervisor().numFullResetsServed(), 1u) << "persistent failures escalate to a full reset";

  // The current wait runs until about 1.5 s; a reset requested from outside (the operator re-entering WB_MPC) ends it
  // within the 10 ms the wait polls at.
  link->requestReset();
  EXPECT_TRUE(waitFor([this, failedAfterBackOff]() { return mpc_.solver().numFailedSolves() > failedAfterBackOff; },
                      std::chrono::milliseconds(150)))
      << "a reset request did not cut the back-off short";

  // The first solve that succeeds makes the MPC healthy again.
  mpc_.solver().failEverySolve(false);
  EXPECT_TRUE(waitFor([&link]() { return link->isHealthy(); })) << "the MPC never recovered";
  EXPECT_TRUE(waitForCurrentPolicy(*link));
  log.StopCapturingLogs();
}

TEST_F(InProcessMpcLinkTest, SolvesArePacedAtTheDesiredFrequencyOrRunBackToBack) {
  {
    std::unique_ptr<InProcessMpcLink> link = makeLink(/*mpcDesiredFrequency=*/50.0);
    link->start(observationAt(/*time=*/1.0));
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    link->stop();
    const size_t solves = mpc_.solver().numSolves();
    EXPECT_GE(solves, 20u) << "the solver thread ran far below 50 Hz";
    EXPECT_LE(solves, 56u) << "the solver thread did not pace itself at 50 Hz";
  }
  {
    const size_t before = mpc_.solver().numSolves();
    std::unique_ptr<InProcessMpcLink> link = makeLink(/*mpcDesiredFrequency=*/-1.0);
    link->start(observationAt(/*time=*/1.0));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    link->stop();
    EXPECT_GT(mpc_.solver().numSolves() - before, 100u) << "a frequency <= 0 must run the solves back to back";
  }
}

TEST_F(InProcessMpcLinkTest, TheSolveObserverSeesEveryAttemptAfterTheSupervisorAccountedForIt) {
  struct Seen {
    absl::StatusCode code;
    bool healthy;
  };
  std::mutex mutex;
  std::vector<Seen> seen;
  MpcLink* linkPtr = nullptr;
  std::unique_ptr<InProcessMpcLink> link = makeLink(/*mpcDesiredFrequency=*/1000.0, [&](const absl::Status& status) {
    std::lock_guard<std::mutex> lock(mutex);
    seen.push_back({status.code(), linkPtr->isHealthy()});
  });
  linkPtr = link.get();
  mpc_.solver().failNextSolves(link->getResetSupervisor().getConfig().maxConsecutiveFailures);
  link->start(observationAt(/*time=*/1.0));
  ASSERT_TRUE(waitForCurrentPolicy(*link));
  link->stop();

  std::lock_guard<std::mutex> lock(mutex);
  const size_t failures = link->getResetSupervisor().getConfig().maxConsecutiveFailures;
  ASSERT_GT(seen.size(), failures);
  EXPECT_EQ(seen.size(), mpc_.solver().numSolves()) << "an attempt the observer did not see";
  for (size_t k = 0; k < failures; ++k) EXPECT_EQ(seen[k].code, absl::StatusCode::kInternal) << "attempt " << k;
  EXPECT_FALSE(seen[failures - 1].healthy) << "the observer ran before the supervisor accounted for the failure";
  EXPECT_EQ(seen[failures].code, absl::StatusCode::kOk);
  EXPECT_TRUE(seen[failures].healthy);
}

TEST_F(InProcessMpcLinkTest, StopJoinsTheSolverThreadAndIsIdempotent) {
  std::unique_ptr<InProcessMpcLink> link = makeLink(/*mpcDesiredFrequency=*/1000.0);
  link->start(observationAt(/*time=*/1.0));
  ASSERT_TRUE(waitForCurrentPolicy(*link));
  link->stop();
  const size_t solves = mpc_.solver().numSolves();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(mpc_.solver().numSolves(), solves) << "the solver thread ran on after stop()";
  link->stop();
  link.reset();

  // A link that never started is destroyed without waiting.
  std::unique_ptr<InProcessMpcLink> unstarted = makeLink(/*mpcDesiredFrequency=*/1000.0);
  unstarted.reset();
}

TEST_F(InProcessMpcLinkTest, ASecondStartIsRefused) {
  std::unique_ptr<InProcessMpcLink> link = makeLink(/*mpcDesiredFrequency=*/1000.0);
  link->start(observationAt(/*time=*/1.0));
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _, HasSubstr("started before"))).Times(1);
  log.StartCapturingLogs();
  link->start(observationAt(/*time=*/5.0));
  log.StopCapturingLogs();
  ASSERT_TRUE(waitForCurrentPolicy(*link));
  EXPECT_EQ(resetTargets_.times().front(), 1.0);
}

TEST_F(InProcessMpcLinkTest, TheFactoryMakesALinkOverTheMpcAndAResetTargetIsRequired) {
  const MpcLinkFactory factory = InProcessMpcLink::factory(mpc_, InProcessMpcLink::Config());
  std::unique_ptr<MpcLink> link = factory(resetTargets_.function());
  ASSERT_NE(link, nullptr);
  link->start(observationAt(/*time=*/1.0));
  EXPECT_TRUE(waitForCurrentPolicy(*link));
  link->stop();

  EXPECT_THROW(InProcessMpcLink(mpc_, /*resetTarget=*/nullptr, InProcessMpcLink::Config()), std::invalid_argument);
}

TEST_F(InProcessMpcLinkTest, TheCallerRunsTheIterationsAndNothingSolvesWithoutThem) {
  std::unique_ptr<InProcessMpcLink> link = makeCallerLink();
  link->start(observationAt(/*time=*/1.0));
  // The start-up reset is served by start(), on this thread, before it returns; nothing is solved yet.
  ASSERT_EQ(resetTargets_.times().size(), 1u);
  EXPECT_EQ(resetTargets_.times().front(), 1.0);
  EXPECT_EQ(module_->numResets.load(), 1u) << "the start-up reset is a full one";
  EXPECT_EQ(mpc_.solver().numSolves(), 0u);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(mpc_.solver().numSolves(), 0u) << "a solver thread ran";
  link->updatePolicy();
  EXPECT_FALSE(link->initialPolicyReceived());

  // One iteration, one solve, and its policy is the control thread's to take.
  const InProcessMpcLink::SolverIterationResult first = link->runSolverIteration();
  ASSERT_TRUE(first.status.ok()) << first.status;
  EXPECT_EQ(first.retryDelay.count(), 0.0);
  EXPECT_EQ(mpc_.solver().numSolves(), 1u);
  link->updatePolicy();
  EXPECT_TRUE(link->initialPolicyReceived());
  EXPECT_EQ(link->getResetSupervisor().numResetsServed(), 0u) << "the start-up reset is no requested one";

  // A requested reset waits for the caller's next iteration, which serves it from the observation current then.
  link->setCurrentObservation(observationAt(/*time=*/2.0));
  link->requestReset(MpcLink::ResetKind::kSolver);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_TRUE(link->hasOutstandingReset());
  EXPECT_EQ(link->getResetSupervisor().numResetsServed(), 0u);
  ASSERT_TRUE(link->runSolverIteration().status.ok());
  EXPECT_FALSE(link->hasOutstandingReset());
  EXPECT_EQ(link->getResetSupervisor().numResetsServed(), 1u);
  EXPECT_EQ(module_->numResets.load(), 1u) << "a reset of the solver alone reset the synchronized modules";
  EXPECT_EQ(resetTargets_.times().back(), 2.0);
  EXPECT_EQ(mpc_.solver().numSolves(), 2u);
  link->updatePolicy();
  EXPECT_TRUE(link->isActivePolicyCurrent());
}

TEST_F(InProcessMpcLinkTest, TheCallersIterationsFollowTheFailurePolicyOfTheSolverThread) {
  std::vector<absl::StatusCode> observed;
  std::unique_ptr<InProcessMpcLink> link = makeCallerLink([&observed](const absl::Status& status) { observed.push_back(status.code()); });
  link->start(observationAt(/*time=*/1.0));
  ASSERT_TRUE(link->runSolverIteration().status.ok());

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _, HasSubstr("switch to JOINT_PD and back to WB_MPC"))).Times(1);
  log.StartCapturingLogs();
  // Each failure is followed by a reset served at the next iteration; the last one that makes the MPC unhealthy asks the
  // caller for a pause, which it waits out on its own clock.
  mpc_.solver().failEverySolve(true);
  const size_t maxFailures = link->getResetSupervisor().getConfig().maxConsecutiveFailures;
  InProcessMpcLink::SolverIterationResult iteration;
  for (size_t failure = 1; failure <= maxFailures; ++failure) {
    iteration = link->runSolverIteration();
    EXPECT_EQ(iteration.status.code(), absl::StatusCode::kInternal) << "failure " << failure;
    if (failure < maxFailures) {
      EXPECT_EQ(iteration.retryDelay.count(), 0.0) << "failure " << failure;
      EXPECT_TRUE(link->isHealthy()) << "failure " << failure;
    }
  }
  EXPECT_GT(iteration.retryDelay.count(), 0.0) << "persistent failures back off";
  EXPECT_FALSE(link->isHealthy());
  EXPECT_GE(link->getResetSupervisor().numResetsServed(), maxFailures - 1) << "a failure's reset was not served";

  mpc_.solver().failEverySolve(false);
  iteration = link->runSolverIteration();
  EXPECT_TRUE(iteration.status.ok()) << iteration.status;
  EXPECT_TRUE(link->isHealthy()) << "a solve that succeeds ends it";
  log.StopCapturingLogs();

  ASSERT_EQ(observed.size(), mpc_.solver().numSolves()) << "an attempt the observer did not see";
  EXPECT_EQ(observed.front(), absl::StatusCode::kOk);
  EXPECT_EQ(observed[maxFailures], absl::StatusCode::kInternal);
  EXPECT_EQ(observed.back(), absl::StatusCode::kOk);
}

TEST_F(InProcessMpcLinkTest, RunSolverIterationIsRefusedOutsideTheCallersExecution) {
  // Before start(), after stop(), and on a link with a solver thread of its own, which the caller must not race.
  std::unique_ptr<InProcessMpcLink> callerLink = makeCallerLink();
  EXPECT_EQ(callerLink->runSolverIteration().status.code(), absl::StatusCode::kFailedPrecondition);
  callerLink->start(observationAt(/*time=*/1.0));
  EXPECT_TRUE(callerLink->runSolverIteration().status.ok());
  callerLink->stop();
  EXPECT_EQ(callerLink->runSolverIteration().status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(mpc_.solver().numSolves(), 1u);

  std::unique_ptr<InProcessMpcLink> threadLink = makeLink(/*mpcDesiredFrequency=*/1000.0);
  EXPECT_EQ(threadLink->runSolverIteration().status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(mpc_.solver().numSolves(), 1u);
}

TEST_F(InProcessMpcLinkTest, ASecondStartOfACallersLinkIsRefused) {
  std::unique_ptr<InProcessMpcLink> link = makeCallerLink();
  link->start(observationAt(/*time=*/1.0));
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _, HasSubstr("started before"))).Times(1);
  log.StartCapturingLogs();
  link->start(observationAt(/*time=*/5.0));
  log.StopCapturingLogs();
  EXPECT_EQ(resetTargets_.times().size(), 1u) << "the refused start reset the MPC";
}

TEST_F(InProcessMpcLinkTest, TheReportingFactoryHandsTheCallerTheLinkItMade) {
  InProcessMpcLink* created = nullptr;
  InProcessMpcLink::Config config;
  config.execution = InProcessMpcLink::Execution::kCaller;
  const MpcLinkFactory factory = InProcessMpcLink::factory(mpc_, config, &created);
  EXPECT_EQ(created, nullptr) << "nothing is made before the factory is called";
  std::unique_ptr<MpcLink> link = factory(resetTargets_.function());
  ASSERT_NE(link, nullptr);
  EXPECT_EQ(created, link.get());
  link->start(observationAt(/*time=*/1.0));
  EXPECT_TRUE(created->runSolverIteration().status.ok());

  EXPECT_THROW(InProcessMpcLink::factory(mpc_, config, /*created=*/nullptr), std::invalid_argument);
}

}  // namespace
}  // namespace ocs2::humanoid
