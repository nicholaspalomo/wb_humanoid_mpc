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

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"

#include <ocs2_mpc/MPC_MRT_Interface.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_oc/synchronized_module/SolverSynchronizedModule.h>

#include "ocs2_mpc_test/ScriptedMpc.h"

/*
 * What an MPC reset clears. It used to clear the solver's warm start and nothing else: the reference manager, the
 * synchronized modules and the MRT's policy buffer all kept what they had, so a controller reset after a fall went on
 * executing schedules, plans and a policy made before it.
 */

namespace ocs2 {
namespace {

using mpc_test::ScriptedMpc;

constexpr size_t kStateDim = 3;
constexpr size_t kInputDim = 2;

/** Records every reset() in a shared log, and counts the solves it has seen. */
class RecordingReferenceManager final : public ReferenceManager {
 public:
  explicit RecordingReferenceManager(std::vector<std::string>* log) : log_(log) {}
  void reset() override {
    log_->push_back("reference manager");
    ReferenceManager::reset();
  }

 private:
  std::vector<std::string>* log_;
};

class RecordingModule final : public SolverSynchronizedModule {
 public:
  RecordingModule(std::string name, std::vector<std::string>* log) : name_(std::move(name)), log_(log) {}
  void preSolverRun(scalar_t /*initTime*/,
                    scalar_t /*finalTime*/,
                    const vector_t& /*initState*/,
                    const ReferenceManagerInterface& /*referenceManager*/) override {}
  void postSolverRun(const PrimalSolution& /*primalSolution*/) override {}
  void reset() override { log_->push_back(name_); }

 private:
  std::string name_;
  std::vector<std::string>* log_;
};

SystemObservation observationAt(scalar_t time, scalar_t value) {
  SystemObservation observation;
  observation.time = time;
  observation.state = vector_t::Constant(kStateDim, value);
  observation.input = vector_t::Zero(kInputDim);
  return observation;
}

TargetTrajectories targetAt(scalar_t time) {
  return TargetTrajectories({time}, {vector_t::Zero(kStateDim)}, {vector_t::Zero(kInputDim)});
}

TEST(MpcReset, ResetsTheReferenceManagerAndEveryModuleBeforeTheSolver) {
  std::vector<std::string> log;
  ScriptedMpc mpc(mpc::Settings(), kInputDim);
  mpc.getSolverPtr()->setReferenceManager(std::make_shared<RecordingReferenceManager>(&log));
  mpc.getSolverPtr()->addSynchronizedModule(std::make_shared<RecordingModule>("first module", &log));
  mpc.getSolverPtr()->addSynchronizedModule(std::make_shared<RecordingModule>("second module", &log));
  ASSERT_EQ(mpc.solver().numResets(), 0u);

  mpc.reset();

  EXPECT_EQ(log, (std::vector<std::string>{"reference manager", "first module", "second module"}));
  EXPECT_EQ(mpc.solver().numResets(), 1u);
}

TEST(MpcReset, ResettingTheSolverAloneLeavesTheReferenceManagerAndTheModules) {
  std::vector<std::string> log;
  ScriptedMpc mpc(mpc::Settings(), kInputDim);
  mpc.getSolverPtr()->setReferenceManager(std::make_shared<RecordingReferenceManager>(&log));
  mpc.getSolverPtr()->addSynchronizedModule(std::make_shared<RecordingModule>("module", &log));

  mpc.resetSolver();

  EXPECT_TRUE(log.empty()) << "the schedule and the command state in execution were reset";
  EXPECT_EQ(mpc.solver().numResets(), 1u);
}

TEST(MpcReset, TheReferenceManagerForgetsWhatWasSetBeforeTheReset) {
  const TargetTrajectories initialTarget = targetAt(0.0);
  ReferenceManager manager(initialTarget, ModeSchedule({1.0}, {0, 1}));
  manager.setTargetTrajectories(targetAt(5.0));
  manager.setModeSchedule(ModeSchedule({6.0}, {2, 3}));
  manager.preSolverRun(5.0, 6.0, vector_t::Zero(kStateDim), 0);
  ASSERT_EQ(manager.getTargetTrajectories().timeTrajectory.front(), 5.0) << "positive control: the values were applied";
  // One more of each is waiting in the buffers when the reset comes.
  manager.setTargetTrajectories(targetAt(7.0));
  manager.setModeSchedule(ModeSchedule({8.0}, {4, 5}));

  manager.reset();
  EXPECT_EQ(manager.getTargetTrajectories().timeTrajectory, initialTarget.timeTrajectory);
  EXPECT_EQ(manager.getModeSchedule().eventTimes, (std::vector<scalar_t>{1.0}));
  manager.preSolverRun(9.0, 10.0, vector_t::Zero(kStateDim), 0);
  EXPECT_EQ(manager.getTargetTrajectories().timeTrajectory, initialTarget.timeTrajectory)
      << "a target set before the reset was applied after it";
  EXPECT_EQ(manager.getModeSchedule().modeSequence, (std::vector<size_t>{0, 1})) << "a schedule set before the reset was applied after it";
}

TEST(MpcMrtInterfaceReset, APolicySolvedBeforeTheResetIsNeverSwappedInAfterIt) {
  ScriptedMpc mpc(mpc::Settings(), kInputDim);
  MPC_MRT_Interface mrt(mpc);
  mrt.setCurrentObservation(observationAt(0.0, 1.0));
  mrt.resetMpcNode(targetAt(0.0));
  ASSERT_TRUE(mrt.advanceMpc().ok());
  ASSERT_TRUE(mrt.updatePolicy());
  EXPECT_TRUE(mrt.isActivePolicyCurrent());

  // A second policy reaches the buffer before the reset.
  mrt.setCurrentObservation(observationAt(0.1, 2.0));
  ASSERT_TRUE(mrt.advanceMpc().ok());
  mrt.resetMpcNode(targetAt(0.1));

  EXPECT_FALSE(mrt.updatePolicy()) << "the policy solved before the reset was swapped in after it";
  EXPECT_FALSE(mrt.isActivePolicyCurrent()) << "the policy in use was solved before the reset";
  EXPECT_DOUBLE_EQ(mrt.getPolicy().stateTrajectory_.front()(0), 1.0) << "the policy in use must not have been replaced";

  // The first solve after the reset is the first policy that counts.
  mrt.setCurrentObservation(observationAt(0.2, 3.0));
  ASSERT_TRUE(mrt.advanceMpc().ok());
  ASSERT_TRUE(mrt.updatePolicy());
  EXPECT_TRUE(mrt.isActivePolicyCurrent());
  EXPECT_DOUBLE_EQ(mrt.getPolicy().stateTrajectory_.front()(0), 3.0);
}

TEST(MpcMrtInterfaceReset, ASolverResetAlsoDropsTheBufferedPolicy) {
  ScriptedMpc mpc(mpc::Settings(), kInputDim);
  MPC_MRT_Interface mrt(mpc);
  mrt.setCurrentObservation(observationAt(0.0, 1.0));
  mrt.resetMpcNode(targetAt(0.0));
  ASSERT_TRUE(mrt.advanceMpc().ok());
  ASSERT_TRUE(mrt.updatePolicy());
  mrt.setCurrentObservation(observationAt(0.1, 2.0));
  ASSERT_TRUE(mrt.advanceMpc().ok());
  mrt.resetMpcSolver(targetAt(0.1));
  EXPECT_FALSE(mrt.updatePolicy());
  EXPECT_FALSE(mrt.isActivePolicyCurrent());
  ASSERT_TRUE(mrt.advanceMpc().ok());
  EXPECT_TRUE(mrt.updatePolicy());
  EXPECT_TRUE(mrt.isActivePolicyCurrent());
}

TEST(MpcMrtInterfaceReset, WithoutAResetTheBufferedPolicyIsSwappedIn) {
  // Positive control of the test above: the same sequence without the reset does swap the second policy in.
  ScriptedMpc mpc(mpc::Settings(), kInputDim);
  MPC_MRT_Interface mrt(mpc);
  mrt.setCurrentObservation(observationAt(0.0, 1.0));
  mrt.resetMpcNode(targetAt(0.0));
  ASSERT_TRUE(mrt.advanceMpc().ok());
  ASSERT_TRUE(mrt.updatePolicy());
  mrt.setCurrentObservation(observationAt(0.1, 2.0));
  ASSERT_TRUE(mrt.advanceMpc().ok());
  EXPECT_TRUE(mrt.updatePolicy());
  EXPECT_TRUE(mrt.isActivePolicyCurrent());
  EXPECT_DOUBLE_EQ(mrt.getPolicy().stateTrajectory_.front()(0), 2.0);
}

TEST(MpcMrtInterfaceReset, AnMpcThatCanNoLongerRunIsAFailureThatAResetCures) {
  // The observation time jumps past the end of the previous solution. MPC_BASE::run() then refuses to run, and it used
  // to do so silently, for ever: advanceMpc() reported success and no policy ever reached the buffer again.
  mpc::Settings settings;
  settings.timeHorizon_ = 1.0;
  ScriptedMpc mpc(settings, kInputDim);
  MPC_MRT_Interface mrt(mpc);
  mrt.setCurrentObservation(observationAt(0.0, 1.0));
  mrt.resetMpcNode(targetAt(0.0));
  ASSERT_TRUE(mrt.advanceMpc().ok());
  ASSERT_TRUE(mrt.updatePolicy());

  mrt.setCurrentObservation(observationAt(5.0, 1.0));
  const absl::Status stalled = mrt.advanceMpc();
  EXPECT_EQ(stalled.code(), absl::StatusCode::kFailedPrecondition) << stalled;
  EXPECT_FALSE(mrt.updatePolicy());

  mrt.resetMpcNode(targetAt(5.0));
  EXPECT_TRUE(mrt.advanceMpc().ok());
  EXPECT_TRUE(mrt.updatePolicy());
  EXPECT_TRUE(mrt.isActivePolicyCurrent());
}

TEST(MpcMrtInterfaceReset, ASolverThatThrowsIsReportedAsAnInternalError) {
  ScriptedMpc mpc(mpc::Settings(), kInputDim);
  MPC_MRT_Interface mrt(mpc);
  mrt.setCurrentObservation(observationAt(0.0, 1.0));
  mrt.resetMpcNode(targetAt(0.0));
  mpc.solver().failNextSolves(2);
  EXPECT_EQ(mrt.advanceMpc().code(), absl::StatusCode::kInternal);
  EXPECT_EQ(mrt.advanceMpc().code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(mrt.advanceMpc().ok());
  EXPECT_EQ(mpc.solver().numFailedSolves(), 2u);
}

}  // namespace
}  // namespace ocs2
