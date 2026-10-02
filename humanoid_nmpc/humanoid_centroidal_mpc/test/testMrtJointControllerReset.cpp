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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include <robot_model/RobotDescription.h>
#include <robot_model/RobotJointAction.h>
#include <robot_model/RobotState.h>

#include "absl/log/scoped_mock_log.h"

#include "humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/mrt/InProcessMpcLink.h"
#include "support/AtlasReferenceStack.h"

/*
 * The reset contract of CentroidalMpcMrtJointController, with its link's solver thread running (and in one case with
 * the caller running the link's iterations, as the lockstep closed loop does): the DRC Atlas references wired as in the
 * sim (AtlasReferenceStack) around a scripted solver, which plans what each test tells it to.
 *
 * The user's run that motivated this: after a fall and a reset of the simulator every solve failed, each failure
 * requested a reset that cured nothing, and the controller logged the failure a hundred times a second for ever while
 * WB_MPC waited silently for a policy that never came. Before that the divergence check had requested a reset at every
 * control cycle, in ZERO_TORQUE too, against a robot hanging limp.
 */

namespace ocs2::humanoid {
namespace {

using ::testing::_;
using ::testing::HasSubstr;

constexpr scalar_t kControlPeriod = 0.01;   // [s] of observation time per control cycle
constexpr scalar_t kMpcFrequency = 1000.0;  // [Hz] the scripted solver is fast; the solver thread paces itself at this

/** The controller, its robot and the operator's side of the loop. */
class ControllerHarness {
 public:
  /**
   * With InProcessMpcLink::Execution::kCaller the controller's link starts no solver thread: the test runs its
   * iterations (link().runSolverIteration()), as the lockstep closed loop of humanoid_mpc_validation does.
   */
  explicit ControllerHarness(InProcessMpcLink::Execution execution = InProcessMpcLink::Execution::kSolverThread)
      : description_(stack_.urdfFile()), robotState_(description_), action_(description_) {
    if (execution == InProcessMpcLink::Execution::kCaller) {
      InProcessMpcLink::Config config;
      config.execution = execution;
      controller_ = std::make_unique<CentroidalMpcMrtJointController>(description_, stack_.modelSettings(), stack_.model(),
                                                                      InProcessMpcLink::factory(stack_.mpc(), std::move(config), &link_),
                                                                      stack_.pinocchioInterface());
    } else {
      controller_ = std::make_unique<CentroidalMpcMrtJointController>(description_, stack_.modelSettings(), stack_.model(), stack_.mpc(),
                                                                      stack_.pinocchioInterface(), kMpcFrequency);
    }
    mpcJointIndices_ = description_.getJointIndices(stack_.modelSettings().mpcModelJointNames);
    setState(stack_.initialState());
    // The nominal posture of JOINT_PD is the posture the robot starts in, as SimFsmBridge captures it.
    nominal_.assign(description_.getNumJoints(), 0.0);
    for (size_t joint = 0; joint < description_.getNumJoints(); ++joint) nominal_[joint] = robotState_.getJointPosition(joint);
    controller_->setNominalJointPositions(nominal_);
  }

  ~ControllerHarness() { controller_.reset(); }

  AtlasReferenceStack& stack() { return stack_; }
  CentroidalMpcMrtJointController& controller() { return *controller_; }
  /** The controller's link, for a harness built with InProcessMpcLink::Execution::kCaller. */
  InProcessMpcLink& link() { return *link_; }
  scalar_t time() const { return time_; }
  void setTime(scalar_t time) { time_ = time; }

  /** The robot at the centroidal state `state`, moving with base velocity `baseVelocity` (world frame). */
  void setState(const vector_t& state, const vector3_t& baseVelocity = vector3_t::Zero()) {
    const CentroidalMpcRobotModel<scalar_t>& model = stack_.model();
    robotState_.setConfigurationToZero();
    robotState_.setRootPositionInWorldFrame(model.getBasePosition(state));
    const quaternion_t orientation = getQuaternionFromEulerAnglesZyx(vector3_t(model.getBaseOrientationEulerZYX(state)));
    robotState_.setRootRotationLocalToWorldFrame(orientation);
    robotState_.setRootLinearVelocityInLocalFrame(orientation.toRotationMatrix().transpose() * baseVelocity);
    robotState_.setRootAngularVelocityInLocalFrame(vector3_t::Zero());
    const vector_t joints = model.getJointAngles(state);
    for (size_t i = 0; i < mpcJointIndices_.size(); ++i) robotState_.setJointPosition(mpcJointIndices_[i], joints[i]);
    robotState_.setContactFlag(/*index=*/0, /*contactFlag=*/true);
    robotState_.setContactFlag(/*index=*/1, /*contactFlag=*/true);
  }

  void start() {
    robotState_.setTime(time_);
    controller_->startMpcThread(robotState_);
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (!controller_->ready() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    ASSERT_TRUE(controller_->ready()) << "the solver thread produced no policy";
  }

  /** startMpcThread() at the current time on a caller's link: it serves the start-up reset and starts no thread. */
  void startWithoutThread() {
    robotState_.setTime(time_);
    controller_->startMpcThread(robotState_);
  }

  /** One control cycle at the current time; the time then moves on by kControlPeriod. */
  const robot::model::RobotJointAction& cycle(std::chrono::microseconds pause = std::chrono::microseconds(500)) {
    robotState_.setTime(time_);
    controller_->computeJointControlAction(time_, robotState_, action_);
    time_ += kControlPeriod;
    std::this_thread::sleep_for(pause);  // lets the solver thread run between two cycles
    return action_;
  }

  /** Cycles until `done` holds, at most `maxCycles`; returns whether it did. */
  bool cycleUntil(const std::function<bool()>& done, int maxCycles, std::chrono::microseconds pause = std::chrono::microseconds(500)) {
    for (int k = 0; k < maxCycles; ++k) {
      if (done()) return true;
      cycle(pause);
    }
    return done();
  }

  /** WB_MPC entered from JOINT_PD, run until a policy solved after the entry is in use. */
  void enterMpc() {
    controller_->setControlMode("JOINT_PD");
    cycle();
    controller_->setControlMode("WB_MPC");
    ASSERT_TRUE(cycleUntil([this]() { return controller_->getPlannedContactFlags(time_).has_value(); }, /*maxCycles=*/2000))
        << "no policy solved after the entry reset was ever put in use";
  }

  /** The action JOINT_PD commands at the robot's current state, from a controller of its own. */
  robot::model::RobotJointAction jointPdActionHere() {
    CentroidalMpcMrtJointController reference(description_, stack_.modelSettings(), stack_.model(), stack_.mpc(),
                                              stack_.pinocchioInterface(), kMpcFrequency);
    reference.setNominalJointPositions(nominal_);
    reference.setControlMode("JOINT_PD");
    robot::model::RobotJointAction action(description_);
    robotState_.setTime(time_);
    reference.computeJointControlAction(time_, robotState_, action);
    return action;
  }

  /** The largest difference of any field of the joint actions of the MPC joints. */
  scalar_t maxDifference(const robot::model::RobotJointAction& a, const robot::model::RobotJointAction& b) const {
    scalar_t difference = 0.0;
    for (size_t index : mpcJointIndices_) {
      const robot::model::JointAction& x = a.at(index).value();
      const robot::model::JointAction& y = b.at(index).value();
      difference = std::max({difference, std::abs(x.q_des - y.q_des), std::abs(x.qd_des - y.qd_des), std::abs(x.kp - y.kp),
                             std::abs(x.kd - y.kd), std::abs(x.feed_forward_effort - y.feed_forward_effort)});
    }
    return difference;
  }

  const std::vector<size_t>& mpcJointIndices() const { return mpcJointIndices_; }
  const robot::model::RobotState& robotState() const { return robotState_; }

 private:
  AtlasReferenceStack stack_;
  robot::model::RobotDescription description_;
  robot::model::RobotState robotState_;
  robot::model::RobotJointAction action_;
  std::vector<size_t> mpcJointIndices_;
  std::vector<scalar_t> nominal_;
  scalar_t time_ = 1.0;
  InProcessMpcLink* link_ = nullptr;  // the controller's, with InProcessMpcLink::Execution::kCaller
  std::unique_ptr<CentroidalMpcMrtJointController> controller_;
};

/** A plan whose joints are `offset` [rad] away from the observed ones over the whole horizon. */
mpc_test::ScriptedSolver::PlanFunction jointOffsetPlan(const AtlasReferenceStack& stack, scalar_t offset) {
  const size_t jointStart = stack.model().getJointStartindex();
  const size_t jointDim = stack.model().getJointDim();
  return [jointStart, jointDim, offset](scalar_t /*time*/, scalar_t /*initTime*/, const vector_t& initState) {
    vector_t planned = initState;
    planned.segment(jointStart, jointDim).array() += offset;
    return planned;
  };
}

TEST(MrtJointControllerReset, ZeroTorqueCommandsNothingAndRequestsNoReset) {
  ControllerHarness harness;
  harness.controller().setControlMode("ZERO_TORQUE");
  harness.start();
  harness.stack().mpc().solver().setPlan(jointOffsetPlan(harness.stack(), /*offset=*/1.0));  // a policy the divergence check rejects

  for (int k = 0; k < 150; ++k) {
    const robot::model::RobotJointAction& action = harness.cycle();
    for (size_t index : harness.mpcJointIndices()) {
      ASSERT_EQ(action.at(index)->kp, 0.0);
      ASSERT_EQ(action.at(index)->kd, 0.0);
      ASSERT_EQ(action.at(index)->feed_forward_effort, 0.0);
    }
  }
  EXPECT_EQ(harness.controller().getResetSupervisor().numResetsServed(), 0u)
      << "ZERO_TORQUE executes no policy, so nothing about one may reset the MPC";

  // Positive control: the same policy in WB_MPC is rejected by the divergence check, which does reset the MPC.
  harness.controller().setControlMode("JOINT_PD");
  harness.cycle();
  harness.controller().setControlMode("WB_MPC");
  // The entry reset is the first; the divergence check requests the second.
  EXPECT_TRUE(
      harness.cycleUntil([&harness]() { return harness.controller().getResetSupervisor().numResetsServed() >= 2; }, /*maxCycles=*/1000));
}

TEST(MrtJointControllerReset, TheDivergenceCheckResetsAtMostOncePerPolicyAndInterval) {
  ControllerHarness harness;
  harness.start();
  harness.enterMpc();
  const uint64_t resetsBefore = harness.controller().getResetSupervisor().numResetsServed();
  const uint64_t fullResetsBefore = harness.controller().getResetSupervisor().numFullResetsServed();
  harness.stack().mpc().solver().setPlan(jointOffsetPlan(harness.stack(), /*offset=*/1.0));

  // Three seconds of observation time, every policy diverged.
  const scalar_t duration = 3.0;
  for (int k = 0; k < static_cast<int>(duration / kControlPeriod); ++k) harness.cycle(std::chrono::microseconds(1000));
  const uint64_t resets = harness.controller().getResetSupervisor().numResetsServed() - resetsBefore;
  EXPECT_GE(resets, 1u) << "positive control: the check does fire";
  EXPECT_LE(resets, static_cast<uint64_t>(duration / 0.5) + 2) << "one reset per diverged policy and per 0.5 s, not one per cycle";
  // Of the solver alone: a full reset restarts the gait in stance under a robot in mid-stride.
  EXPECT_EQ(harness.controller().getResetSupervisor().numFullResetsServed(), fullResetsBefore);
}

TEST(MrtJointControllerReset, RepeatedFailuresBackOffHoldTheRobotInJointPdAndLogOneError) {
  ControllerHarness harness;
  harness.start();
  harness.stack().mpc().solver().setPlan(jointOffsetPlan(harness.stack(), /*offset=*/0.1));
  harness.enterMpc();
  const robot::model::RobotJointAction jointPd = harness.jointPdActionHere();
  const robot::model::RobotJointAction mpcAction = harness.cycle();
  ASSERT_GT(harness.maxDifference(mpcAction, jointPd), 0.05) << "positive control: the MPC action is not the JOINT_PD action";

  const uint64_t resetsBefore = harness.controller().getResetSupervisor().numResetsServed();
  const uint64_t fullResetsBefore = harness.controller().getResetSupervisor().numFullResetsServed();
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  // gMock tries the expectation declared last first: one error, saying how to recover, and no other.
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _, _)).Times(0);
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _, HasSubstr("switch to JOINT_PD and back to WB_MPC"))).Times(1);
  log.StartCapturingLogs();

  harness.stack().mpc().solver().failEverySolve(true);
  const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  robot::model::RobotJointAction held = mpcAction;
  while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(1500)) held = harness.cycle(std::chrono::milliseconds(5));

  EXPECT_FALSE(harness.controller().isMpcHealthy());
  EXPECT_GT(harness.controller().getResetSupervisor().numFullResetsServed(), fullResetsBefore)
      << "persistent failures escalate to a full reset";
  const uint64_t resets = harness.controller().getResetSupervisor().numResetsServed() - resetsBefore;
  const uint64_t failedSolves = harness.stack().mpc().solver().numFailedSolves();
  EXPECT_LE(resets, 15u) << "a reset for every attempt, and the attempts back off: " << failedSolves << " failed solves in 1.5 s";
  EXPECT_LT(harness.maxDifference(held, jointPd), 1e-9) << "an unhealthy MPC must hold the robot with the JOINT_PD action";

  // A solve that succeeds again: the MPC is healthy, and its policy is executed again.
  harness.stack().mpc().solver().failEverySolve(false);
  EXPECT_TRUE(harness.cycleUntil(
      [&harness]() {
        return harness.controller().isMpcHealthy() && harness.controller().getPlannedContactFlags(harness.time()).has_value();
      },
      /*maxCycles=*/1000, std::chrono::milliseconds(5)));
  log.StopCapturingLogs();
  const robot::model::RobotJointAction recovered = harness.cycle();
  EXPECT_GT(harness.maxDifference(recovered, jointPd), 0.05) << "the recovered MPC's policy is not executed";
}

TEST(MrtJointControllerReset, AClockThatRunsBackwardsIsOneResetAndTheHandOverCompletes) {
  ControllerHarness harness;
  harness.controller().setMpcEntryBlendTime(0.3);
  harness.start();
  harness.stack().mpc().solver().setPlan(jointOffsetPlan(harness.stack(), /*offset=*/0.1));
  harness.enterMpc();
  ASSERT_TRUE(harness.cycleUntil([&harness]() { return !harness.controller().isEnteringMpc(); }, /*maxCycles=*/200))
      << "the entry never completed";
  const robot::model::RobotJointAction jointPd = harness.jointPdActionHere();

  const uint64_t resetsBefore = harness.controller().getResetSupervisor().numResetsServed();
  harness.setTime(harness.time() - 5.0);
  const robot::model::RobotJointAction& afterRewind = harness.cycle();
  EXPECT_TRUE(harness.controller().isEnteringMpc()) << "the policies planned on the old clock went on reaching the robot";
  EXPECT_LT(harness.maxDifference(afterRewind, jointPd), 1e-9) << "the robot is held with the JOINT_PD action";

  // The new policy is taken into use and ramped in; the hold does not wait for the old clock.
  EXPECT_TRUE(
      harness.cycleUntil([&harness]() { return !harness.controller().isEnteringMpc(); }, /*maxCycles=*/300, std::chrono::milliseconds(1)))
      << "WB_MPC stayed in its hold after the clock ran backwards";
  EXPECT_EQ(harness.controller().getResetSupervisor().numResetsServed() - resetsBefore, 1u) << "one rewind, one reset";
  EXPECT_GT(harness.maxDifference(harness.cycle(), jointPd), 0.05) << "the MPC's policy is executed again";
}

TEST(MrtJointControllerReset, AfterAFallTheFirstPolicyIsPlannedFromTheHeldRobotOnAStanceSchedule) {
  ControllerHarness harness;
  AtlasReferenceStack& stack = harness.stack();
  const CentroidalMpcRobotModel<scalar_t>& model = stack.model();
  harness.start();
  harness.enterMpc();

  // Walking forward, as far as the references are concerned: the scripted solver plans the robot where it is, on the
  // reference manager's schedule, so a swing in the policy is the gait the references walk in.
  const scalar_t speed = 1.2;
  stack.command(speed / 1.2);
  vector_t state = stack.initialState();
  bool walked = false;
  for (int k = 0; k < 400; ++k) {
    state(6) += speed * kControlPeriod;
    harness.setState(state, vector3_t(speed, 0.0, 0.0));
    harness.cycle();
    const std::optional<contact_flag_t> planned = harness.controller().getPlannedContactFlags(harness.time() + 0.3);
    walked = walked || (planned.has_value() && !((*planned)[0] && (*planned)[1]));
  }
  ASSERT_TRUE(walked) << "positive control: the policy steps before the fall";

  // The fall: the robot tips over, the loop catches it on the gantry in JOINT_PD and holds the controller; the remote
  // control re-centers its sticks.
  vector_t fallen = state;
  fallen(11) = 1.2;
  harness.setState(fallen, vector3_t(0.0, -1.0, -0.5));
  harness.controller().requestMpcResetAndHold();
  harness.controller().setControlMode("JOINT_PD");
  stack.command(0.0);
  harness.cycle();

  // Hanging still on the gantry, where it was caught.
  const vector_t held = stack.standingAt(stack.initialState(), /*x=*/3.0, /*y=*/-1.0, /*yaw=*/0.3);
  harness.setState(held);
  for (int k = 0; k < 100; ++k) harness.cycle();

  // Back into WB_MPC: held until a policy solved after that is in use, then that policy.
  harness.controller().setControlMode("WB_MPC");
  const scalar_t entryTime = harness.time();
  ASSERT_TRUE(harness.cycleUntil([&harness]() { return harness.controller().getPlannedContactFlags(harness.time()).has_value(); },
                                 /*maxCycles=*/2000));

  const SystemObservation& solvedFrom = harness.controller().getCommandData().mpcInitObservation_;
  EXPECT_GE(solvedFrom.time, entryTime) << "the policy in use was solved before the entry into WB_MPC";
  EXPECT_LT((model.getBasePose(solvedFrom.state) - model.getBasePose(held)).cwiseAbs().maxCoeff(), 1e-9)
      << "the policy was not planned from the robot held on the gantry";
  EXPECT_LT((model.getJointAngles(solvedFrom.state) - model.getJointAngles(held)).cwiseAbs().maxCoeff(), 1e-9);
  for (scalar_t query = harness.time(); query <= harness.time() + stack.horizon(); query += 0.01) {
    const std::optional<contact_flag_t> planned = harness.controller().getPlannedContactFlags(query);
    ASSERT_TRUE(planned.has_value());
    EXPECT_TRUE((*planned)[0] && (*planned)[1]) << "the first policy after the fall still steps, at t = " << query;
  }
}

TEST(MrtJointControllerReset, WithoutASolverThreadTheCallersIterationsServeResetsAndFailuresOnItsClock) {
  // The lockstep closed loop of humanoid_mpc_validation runs the controller on a link without a solver thread: the
  // iteration the thread would run is called by the caller, and nothing is solved, reset or held off but what it asks
  // for.
  ControllerHarness harness(InProcessMpcLink::Execution::kCaller);
  harness.startWithoutThread();
  EXPECT_FALSE(harness.controller().ready()) << "the start-up reset solves nothing";
  InProcessMpcLink::SolverIterationResult iteration = harness.link().runSolverIteration();
  ASSERT_TRUE(iteration.status.ok()) << iteration.status;
  EXPECT_EQ(iteration.retryDelay.count(), 0.0);
  EXPECT_TRUE(harness.controller().ready()) << "the first iteration produced the first policy";
  EXPECT_EQ(harness.controller().getResetSupervisor().numResetsServed(), 0u) << "the start-up reset is not a requested one";

  // Entering WB_MPC requests a reset and holds; without an iteration nothing serves it, however long the caller cycles.
  harness.controller().setControlMode("JOINT_PD");
  harness.cycle(std::chrono::microseconds(0));
  harness.controller().setControlMode("WB_MPC");
  for (int k = 0; k < 50; ++k) harness.cycle(std::chrono::microseconds(0));
  EXPECT_TRUE(harness.controller().isEnteringMpc());
  EXPECT_FALSE(harness.controller().getPlannedContactFlags(harness.time()).has_value());
  EXPECT_EQ(harness.controller().getResetSupervisor().numResetsServed(), 0u);

  // One iteration serves the reset and solves; the next cycle executes the policy solved after it.
  ASSERT_TRUE(harness.link().runSolverIteration().status.ok());
  EXPECT_EQ(harness.controller().getResetSupervisor().numResetsServed(), 1u);
  harness.cycle(std::chrono::microseconds(0));
  EXPECT_TRUE(harness.controller().getPlannedContactFlags(harness.time()).has_value());

  // Failures: a reset of the solver and no wait for the first, then the back-off the caller waits out on its own clock.
  harness.stack().mpc().solver().failEverySolve(true);
  const size_t maxFailures = harness.controller().getResetSupervisor().getConfig().maxConsecutiveFailures;
  for (size_t failure = 1; failure <= maxFailures; ++failure) {
    iteration = harness.link().runSolverIteration();
    EXPECT_FALSE(iteration.status.ok());
    if (failure < maxFailures) {
      EXPECT_EQ(iteration.retryDelay.count(), 0.0) << "failure " << failure;
      EXPECT_TRUE(harness.controller().isMpcHealthy()) << "failure " << failure;
    }
  }
  EXPECT_GT(iteration.retryDelay.count(), 0.0) << "persistent failures back off";
  EXPECT_FALSE(harness.controller().isMpcHealthy());
  harness.stack().mpc().solver().failEverySolve(false);
  iteration = harness.link().runSolverIteration();
  EXPECT_TRUE(iteration.status.ok()) << iteration.status;
  EXPECT_TRUE(harness.controller().isMpcHealthy()) << "a solve that succeeds ends it";
}

}  // namespace
}  // namespace ocs2::humanoid
