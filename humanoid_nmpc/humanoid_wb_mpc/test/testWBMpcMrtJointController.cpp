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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_mpc/MPC_Settings.h"
#include "ocs2_mpc_test/ScriptedMpc.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/rnea.hpp"

#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/solver/SolverSettingsFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/mrt/ControllerEvent.h"
#include "humanoid_common_mpc/mrt/ControllerEventSink.h"
#include "humanoid_common_mpc/mrt/InProcessMpcLink.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "humanoid_wb_mpc/dynamics/DynamicsHelperFunctions.h"
#include "humanoid_wb_mpc/mrt/WBMpcMrtJointController.h"
#include "robot_model/ContactEstimator.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"

/*
 * The whole-body MRT joint controller's modes and reset contract, on the G1 whole-body model, around a scripted solver.
 * It had neither: the sim never handed it the FSM mode, so JOINT_PD, GRAVITY_COMP and SAFETY left the MPC driving a
 * robot caught on the gantry; a failed solve kept the previous solution and was retried at the solve rate for ever; and
 * its solver thread looped on `while (true)`, so the controller could not be destroyed at all.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kControlPeriod = 0.01;  // [s]

/** The action of joint `index`, which every joint of the description holds: a test failure and a zero action if not. */
const robot::model::JointAction& actionOf(const robot::model::RobotJointAction& action, size_t index) {
  const std::optional<robot::model::JointAction>& entry = action.at(index);
  if (!entry.has_value()) {
    ADD_FAILURE() << "no action for joint " << index;
    static const robot::model::JointAction kNoAction;
    return kNoAction;
  }
  return *entry;
}

std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR")) roots.emplace_back(std::filesystem::path(srcDir) / "_main");
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

/** The G1 whole-body controller around a scripted solver, and the robot it controls. */
class WholeBodyHarness {
 public:
  /**
   * With InProcessMpcLink::Execution::kCaller the controller's link starts no solver thread: the test runs its
   * iterations (link().runSolverIteration()), as the lockstep closed loop of humanoid_mpc_validation does.
   */
  explicit WholeBodyHarness(InProcessMpcLink::Execution execution = InProcessMpcLink::Execution::kSolverThread)
      : task_(loadTaskFile(runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto")).value()),
        urdfFile_(runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf")),
        modelSettings_(ModelSettings::Create(task_, urdfFile_, "wb_mpc_", /*verbose=*/false).value()),
        pinocchioInterface_(loadCustomPinocchioInterface(task_, urdfFile_, modelSettings_).value()),
        model_(modelSettings_),
        mpc_(toMpcSettings(task_.mpc), model_.getInputDim()),
        description_(robot::model::RobotDescription::Create(urdfFile_).value()),
        robotState_(description_),
        action_(description_),
        initialState_(
            stateValuesFromConfig(task_.initial_state, stateInputLayout(modelSettings_, StateInputLayout::Mpc::kWholeBody), "initial_state")
                .value()) {
    mpcJointIndices_ = description_.getJointIndices(modelSettings_.mpcModelJointNames);
    setState(initialState_);
    absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>> created = absl::UnknownError("not created");
    if (execution == InProcessMpcLink::Execution::kCaller) {
      InProcessMpcLink::Config config;
      config.execution = execution;
      created = WBMpcMrtJointController::Create(description_, modelSettings_, InProcessMpcLink::factory(mpc_, std::move(config), &link_),
                                                pinocchioInterface_);
    } else {
      created = WBMpcMrtJointController::Create(description_, modelSettings_, mpc_, pinocchioInterface_, /*mpcDesiredFrequency=*/1000.0);
    }
    EXPECT_TRUE(created.ok()) << created.status();
    if (created.ok()) controller_ = *std::move(created);
  }

  WholeBodyHarness(const WholeBodyHarness&) = delete;
  WholeBodyHarness& operator=(const WholeBodyHarness&) = delete;
  // The controller first: it was declared last, and its link reads the scripted MPC.
  ~WholeBodyHarness() = default;

  WBMpcMrtJointController& controller() { return *controller_; }
  /** The controller's link, for a harness built with InProcessMpcLink::Execution::kCaller. */
  InProcessMpcLink& link() { return *link_; }
  std::unique_ptr<WBMpcMrtJointController>& controllerPtr() { return controller_; }
  mpc_test::ScriptedMpc& mpc() { return mpc_; }
  const WBAccelMpcRobotModel<scalar_t>& model() const { return model_; }
  /// The dynamics helpers take a non-const model; the controller does not share this one.
  WBAccelMpcRobotModel<scalar_t>& mutableModel() { return model_; }
  const vector_t& initialState() const { return initialState_; }
  const std::vector<size_t>& mpcJointIndices() const { return mpcJointIndices_; }
  const robot::model::RobotDescription& description() const { return description_; }
  const ModelSettings& modelSettings() const { return modelSettings_; }
  const PinocchioInterface& pinocchioInterface() const { return pinocchioInterface_; }
  scalar_t time() const { return time_; }

  void setState(const vector_t& state) {
    robotState_.setConfigurationToZero();
    robotState_.setRootPositionInWorldFrame(model_.getBasePosition(state));
    robotState_.setRootRotationLocalToWorldFrame(getQuaternionFromEulerAnglesZyx(vector3_t(model_.getBaseOrientationEulerZYX(state))));
    const vector_t joints = model_.getJointAngles(state);
    for (size_t i = 0; i < mpcJointIndices_.size(); ++i) robotState_.setJointPosition(mpcJointIndices_[i], joints[i]);
    robotState_.setContactFlag(/*index=*/0, /*contactFlag=*/true);
    robotState_.setContactFlag(/*index=*/1, /*contactFlag=*/true);
  }

  /** The contact state the controller measures (its default estimator reads the robot state's flags). */
  void setContactFlags(bool left, bool right) {
    robotState_.setContactFlag(/*index=*/0, /*contactFlag=*/left);
    robotState_.setContactFlag(/*index=*/1, /*contactFlag=*/right);
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

  const robot::model::RobotJointAction& cycle(std::chrono::microseconds pause = std::chrono::microseconds(500)) {
    robotState_.setTime(time_);
    controller_->computeJointControlAction(time_, robotState_, action_);
    time_ += kControlPeriod;
    std::this_thread::sleep_for(pause);
    return action_;
  }

  bool cycleUntil(const std::function<bool()>& done, int maxCycles, std::chrono::microseconds pause = std::chrono::microseconds(500)) {
    for (int k = 0; k < maxCycles; ++k) {
      if (done()) return true;
      cycle(pause);
    }
    return done();
  }

  scalar_t maxDifference(const robot::model::RobotJointAction& a, const robot::model::RobotJointAction& b) const {
    scalar_t difference = 0.0;
    for (size_t index : mpcJointIndices_) {
      const robot::model::JointAction& x = actionOf(a, index);
      const robot::model::JointAction& y = actionOf(b, index);
      difference = std::max({difference, std::abs(x.q_des - y.q_des), std::abs(x.qd_des - y.qd_des), std::abs(x.kp - y.kp),
                             std::abs(x.kd - y.kd), std::abs(x.feed_forward_effort - y.feed_forward_effort)});
    }
    return difference;
  }

 private:
  mpc_config::TaskFile task_;
  std::string urdfFile_;
  ModelSettings modelSettings_;
  PinocchioInterface pinocchioInterface_;
  WBAccelMpcRobotModel<scalar_t> model_;
  mpc_test::ScriptedMpc mpc_;
  robot::model::RobotDescription description_;
  robot::model::RobotState robotState_;
  robot::model::RobotJointAction action_;
  vector_t initialState_;
  std::vector<size_t> mpcJointIndices_;
  scalar_t time_ = 1.0;
  InProcessMpcLink* absl_nullable link_ = nullptr;  // the controller's, with InProcessMpcLink::Execution::kCaller
  std::unique_ptr<WBMpcMrtJointController> controller_;
};

/** A plan whose joints are `offset` [rad] away from the observed ones. */
mpc_test::ScriptedSolver::PlanFunction jointOffsetPlan(const WBAccelMpcRobotModel<scalar_t>& model, scalar_t offset) {
  const size_t jointStart = model.getJointStartindex();
  const size_t jointDim = model.getJointDim();
  return [jointStart, jointDim, offset](scalar_t /*time*/, scalar_t /*initTime*/, const vector_t& initState) {
    vector_t planned = initState;
    planned.segment(jointStart, jointDim).array() += offset;
    return planned;
  };
}

TEST(WBMpcMrtJointController, TheControllerCanBeDestroyedWhileItsSolverRuns) {
  WholeBodyHarness harness;
  harness.start();
  std::atomic<bool> destroyed{false};
  std::thread destroyer([&harness, &destroyed]() {
    harness.controllerPtr().reset();
    destroyed.store(true);
  });
  const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!destroyed.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  if (!destroyed.load()) {
    // The destructor is stuck joining the solver thread; the process could not end, so it is ended here.
    LOG(ERROR) << "WBMpcMrtJointController was not destroyed within 5 s: its solver thread ignores the request to stop.";
    std::_Exit(EXIT_FAILURE);
  }
  destroyer.join();
}

TEST(WBMpcMrtJointController, JointPdHoldsTheNominalPostureWithGravityCompensation) {
  WholeBodyHarness harness;
  const robot::model::RobotDescription& description = harness.description();
  std::vector<scalar_t> nominal(description.getNumJoints(), 0.0);
  for (size_t joint = 0; joint < nominal.size(); ++joint) nominal[joint] = 0.1 * static_cast<scalar_t>(joint % 3) - 0.1;
  harness.controller().setNominalJointPositions(nominal);

  // Positive control: WB_MPC, before any policy, does not command the nominal posture.
  const robot::model::RobotJointAction beforeAnyPolicy = harness.cycle();
  bool differsFromNominal = false;
  for (size_t index : harness.mpcJointIndices()) {
    differsFromNominal = differsFromNominal || actionOf(beforeAnyPolicy, index).q_des != nominal[index];
  }
  ASSERT_TRUE(differsFromNominal);

  harness.controller().setControlMode("JOINT_PD");
  const robot::model::RobotJointAction& action = harness.cycle();

  // The gravity torques of the MPC joints at the robot's configuration, the base held.
  PinocchioInterface pinocchioInterface = harness.pinocchioInterface();
  const vector_t q = harness.model().getGeneralizedCoordinates(harness.initialState());
  pinocchio::nonLinearEffects(pinocchioInterface.getModel(), pinocchioInterface.getData(), q,
                              vector_t::Zero(pinocchioInterface.getModel().nv));
  const vector_t gravity = pinocchioInterface.getData().nle.tail(harness.model().getJointDim());

  for (size_t i = 0; i < harness.mpcJointIndices().size(); ++i) {
    const size_t index = harness.mpcJointIndices()[i];
    EXPECT_DOUBLE_EQ(actionOf(action, index).q_des, nominal[index]) << "joint " << index;
    EXPECT_DOUBLE_EQ(actionOf(action, index).qd_des, 0.0);
    EXPECT_GT(actionOf(action, index).kp, 0.0);
    EXPECT_NEAR(actionOf(action, index).feed_forward_effort, gravity[i], 1.0e-9) << "joint " << index;
  }

  // ZERO_TORQUE commands nothing at all.
  harness.controller().setControlMode("ZERO_TORQUE");
  const robot::model::RobotJointAction& limp = harness.cycle();
  for (size_t index : harness.mpcJointIndices()) {
    EXPECT_EQ(actionOf(limp, index).kp, 0.0);
    EXPECT_EQ(actionOf(limp, index).kd, 0.0);
    EXPECT_EQ(actionOf(limp, index).feed_forward_effort, 0.0);
  }
}

TEST(WBMpcMrtJointController, BeforeAnyPolicyTheFeedforwardCarriesTheWeightOnTheMeasuredFeetWithTheBaseHeld) {
  // Before the first policy the controller holds the current posture, and its feedforward carries the robot's weight on
  // the feet measured in contact with the base held still: computeBaseHeldJointTorques. The floating-base inverse
  // dynamics (computeJointTorques) would instead let the base accelerate under whatever part of the weight the measured
  // feet do not carry about the center of mass - here all of it on one foot.
  WholeBodyHarness harness;
  harness.setContactFlags(/*left=*/true, /*right=*/false);
  const robot::model::RobotJointAction& action = harness.cycle();
  ASSERT_FALSE(harness.controller().getPlannedContactFlags(harness.time()).has_value()) << "a policy is already in use";

  PinocchioInterface pinocchioInterface = harness.pinocchioInterface();
  WBAccelMpcRobotModel<scalar_t>& model = harness.mutableModel();
  const vector_t input = weightCompensatingInput(pinocchioInterface, contact_flag_t{true, false}, model);
  const vector_t baseHeld = computeBaseHeldJointTorques<scalar_t>(harness.initialState(), input, pinocchioInterface, model);
  const vector_t floatingBase = computeJointTorques<scalar_t>(harness.initialState(), input, pinocchioInterface, model);
  // Positive control: the two differ, so the comparison below can tell which one the controller used.
  ASSERT_GT((baseHeld - floatingBase).cwiseAbs().maxCoeff(), 1.0);

  ASSERT_EQ(static_cast<size_t>(baseHeld.size()), harness.mpcJointIndices().size());
  for (size_t i = 0; i < harness.mpcJointIndices().size(); ++i) {
    const size_t index = harness.mpcJointIndices()[i];
    EXPECT_NEAR(actionOf(action, index).feed_forward_effort, baseHeld[i], 1.0e-6) << "joint " << index;
  }
}

TEST(WBMpcMrtJointController, EnteringWbMpcResetsTheMpcAndHoldsJointPdUntilAPolicySolvedAfterIt) {
  WholeBodyHarness harness;
  std::vector<scalar_t> nominal(harness.description().getNumJoints(), 0.0);
  harness.controller().setNominalJointPositions(nominal);
  harness.controller().setControlMode("JOINT_PD");
  harness.start();
  harness.mpc().solver().setPlan(jointOffsetPlan(harness.model(), /*offset=*/0.1));
  for (int k = 0; k < 20; ++k) harness.cycle();
  const robot::model::RobotJointAction jointPd = harness.cycle();
  const uint64_t resetsBefore = harness.controller().getResetSupervisor().numResetsServed();

  harness.controller().setControlMode("WB_MPC");
  const robot::model::RobotJointAction& atEntry = harness.cycle();
  EXPECT_TRUE(harness.controller().isHolding() || harness.controller().getPlannedContactFlags(harness.time()).has_value());
  if (harness.controller().isHolding()) {
    EXPECT_LT(harness.maxDifference(atEntry, jointPd), 1.0e-9) << "the policy solved before the entry reached the robot";
  }
  ASSERT_TRUE(harness.cycleUntil([&harness]() { return harness.controller().getPlannedContactFlags(harness.time()).has_value(); },
                                 /*maxCycles=*/2000));
  EXPECT_GE(harness.controller().getResetSupervisor().numResetsServed(), resetsBefore + 1) << "entering WB_MPC did not reset the MPC";
  const robot::model::RobotJointAction& running = harness.cycle();
  const vector_t joints = harness.model().getJointAngles(harness.initialState());
  for (size_t i = 0; i < harness.mpcJointIndices().size(); ++i) {
    EXPECT_NEAR(actionOf(running, harness.mpcJointIndices()[i]).q_des, joints[i] + 0.1, 1.0e-9)
        << "the policy solved after the entry is executed, joint " << harness.mpcJointIndices()[i];
  }

  // The target the MPC was reset to: two knots, and an input that carries the robot's weight on its two feet.
  harness.controllerPtr().reset();
  const TargetTrajectories& target = harness.mpc().getSolverPtr()->getReferenceManager().getTargetTrajectories();
  ASSERT_EQ(target.timeTrajectory.size(), 2u);
  EXPECT_GT(target.timeTrajectory[1], target.timeTrajectory[0]);
  scalar_t verticalForce = 0.0;
  for (size_t foot = 0; foot < kNumContacts; ++foot) verticalForce += harness.model().getContactForce(target.inputTrajectory[0], foot)(2);
  PinocchioInterface pinocchioInterface = harness.pinocchioInterface();
  EXPECT_NEAR(verticalForce, pinocchio::computeTotalMass(pinocchioInterface.getModel()) * 9.81, 1.0e-6);
}

TEST(WBMpcMrtJointController, WithoutASolverThreadTheCallersIterationsServeResetsAndFailuresOnItsClock) {
  // The lockstep closed loop of humanoid_mpc_validation runs the controller on a link without a solver thread: nothing
  // is solved, reset or held off but what the caller's iterations do.
  WholeBodyHarness harness(InProcessMpcLink::Execution::kCaller);
  std::vector<scalar_t> nominal(harness.description().getNumJoints(), 0.0);
  harness.controller().setNominalJointPositions(nominal);
  harness.controller().setControlMode("JOINT_PD");
  harness.startWithoutThread();
  EXPECT_FALSE(harness.controller().ready()) << "the start-up reset solves nothing";
  InProcessMpcLink::SolverIterationResult iteration = harness.link().runSolverIteration();
  ASSERT_TRUE(iteration.status.ok()) << iteration.status;
  EXPECT_EQ(iteration.retryDelay.count(), 0.0);
  EXPECT_TRUE(harness.controller().ready());
  EXPECT_EQ(harness.controller().getResetSupervisor().numResetsServed(), 0u) << "the start-up reset is not a requested one";

  // The entry into WB_MPC holds until an iteration serves its reset and solves.
  harness.cycle(std::chrono::microseconds(0));
  harness.controller().setControlMode("WB_MPC");
  for (int k = 0; k < 50; ++k) harness.cycle(std::chrono::microseconds(0));
  EXPECT_TRUE(harness.controller().isHolding());
  EXPECT_EQ(harness.controller().getResetSupervisor().numResetsServed(), 0u);
  ASSERT_TRUE(harness.link().runSolverIteration().status.ok());
  EXPECT_EQ(harness.controller().getResetSupervisor().numResetsServed(), 1u);
  harness.cycle(std::chrono::microseconds(0));
  EXPECT_FALSE(harness.controller().isHolding()) << "the policy solved after the entry is in use";

  // Persistent failures back off; the caller waits the delay out on its own clock.
  harness.mpc().solver().failEverySolve(/*fail=*/true);
  const size_t maxFailures = harness.controller().getResetSupervisor().getConfig().maxConsecutiveFailures;
  for (size_t failure = 1; failure <= maxFailures; ++failure) {
    iteration = harness.link().runSolverIteration();
    EXPECT_FALSE(iteration.status.ok());
    if (failure < maxFailures) {
      EXPECT_EQ(iteration.retryDelay.count(), 0.0) << "failure " << failure;
    }
  }
  EXPECT_GT(iteration.retryDelay.count(), 0.0);
  EXPECT_FALSE(harness.controller().isMpcHealthy());
  harness.mpc().solver().failEverySolve(/*fail=*/false);
  EXPECT_TRUE(harness.link().runSolverIteration().status.ok());
  EXPECT_TRUE(harness.controller().isMpcHealthy());
}

TEST(WBMpcMrtJointController, RepeatedFailuresBackOffAndHoldJointPd) {
  WholeBodyHarness harness;
  std::vector<scalar_t> nominal(harness.description().getNumJoints(), 0.0);
  harness.controller().setNominalJointPositions(nominal);
  harness.controller().setControlMode("JOINT_PD");
  harness.start();
  harness.mpc().solver().setPlan(jointOffsetPlan(harness.model(), /*offset=*/0.1));
  const robot::model::RobotJointAction jointPd = harness.cycle();
  harness.controller().setControlMode("WB_MPC");
  ASSERT_TRUE(harness.cycleUntil([&harness]() { return harness.controller().getPlannedContactFlags(harness.time()).has_value(); },
                                 /*maxCycles=*/2000));

  const uint64_t resetsBefore = harness.controller().getResetSupervisor().numResetsServed();
  harness.mpc().solver().failEverySolve(/*fail=*/true);
  robot::model::RobotJointAction held = harness.cycle();
  const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(1500)) held = harness.cycle(std::chrono::milliseconds(5));
  EXPECT_FALSE(harness.controller().isMpcHealthy());
  EXPECT_LE(harness.controller().getResetSupervisor().numResetsServed() - resetsBefore, 15u)
      << harness.mpc().solver().numFailedSolves() << " failed solves in 1.5 s";
  EXPECT_LT(harness.maxDifference(held, jointPd), 1.0e-9) << "an unhealthy MPC must hold the robot with the JOINT_PD action";

  harness.mpc().solver().failEverySolve(/*fail=*/false);
  EXPECT_TRUE(harness.cycleUntil(
      [&harness]() {
        return harness.controller().isMpcHealthy() && harness.controller().getPlannedContactFlags(harness.time()).has_value();
      },
      /*maxCycles=*/1000, std::chrono::milliseconds(5)));
}

// Create() refuses a link factory that makes no link, which the constructor used to throw.
TEST(WBMpcMrtJointController, CreateRefusesALinkFactoryThatMakesNoLink) {
  const WholeBodyHarness harness(InProcessMpcLink::Execution::kCaller);
  const MpcLinkFactory noLink = [](const MpcLink::ResetTargetFunction& /*resetTarget*/) -> std::unique_ptr<MpcLink> { return nullptr; };
  const absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>> created =
      WBMpcMrtJointController::Create(harness.description(), harness.modelSettings(), noLink, harness.pinocchioInterface());
  EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(created.status().message(), "made no link")) << created.status();
}

// Create() refuses a robot description that lacks a joint of the MPC model, naming the joint: the control cycle indexes
// the joints without a check, so the constructor's check would otherwise end the process on a mismatched URDF.
TEST(WBMpcMrtJointController, CreateRefusesAModelJointTheRobotDescriptionDoesNotHave) {
  const WholeBodyHarness harness(InProcessMpcLink::Execution::kCaller);
  const std::filesystem::path urdf = std::filesystem::path(::testing::TempDir()) / "two_joint_robot.urdf";
  std::ofstream(urdf) << R"(<?xml version="1.0"?>
<robot name="two_joints">
  <link name="base"/>
  <link name="arm"/>
  <link name="hand"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="arm"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <joint name="wrist" type="revolute">
    <parent link="arm"/><child link="hand"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>
)";
  const absl::StatusOr<robot::model::RobotDescription> description = robot::model::RobotDescription::Create(urdf.string());
  ASSERT_TRUE(description.ok()) << description.status();
  const MpcLinkFactory noLink = [](const MpcLink::ResetTargetFunction& /*resetTarget*/) -> std::unique_ptr<MpcLink> { return nullptr; };
  const absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>> created =
      WBMpcMrtJointController::Create(*description, harness.modelSettings(), noLink, harness.pinocchioInterface());
  EXPECT_EQ(created.status().code(), absl::StatusCode::kNotFound) << "the joints are refused before the link is made";
  EXPECT_TRUE(absl::StrContains(created.status().message(), harness.modelSettings().mpcModelJointNames.front())) << created.status();
}

/** A contact estimator that breaks its contract: one flag, whatever the number of contact points. */
class SingleFlagContactEstimator final : public ::robot::model::ContactEstimator {
 public:
  void estimateContactFlags(const ::robot::model::RobotState& /*robotState*/, std::vector<bool>& flags) override { flags.assign(1, true); }
  std::string getName() const override { return "SingleFlagContactEstimator"; }
};

/** A sink that keeps the events of one code, as the robot process's realtime event log would hand them on. */
class CodeRecordingSink final : public ControllerEventSink {
 public:
  explicit CodeRecordingSink(ControllerEventCode code) : code_(code) {}
  bool post(const ControllerEvent& event) override {
    if (event.code == code_) events.push_back(event);
    return true;
  }
  std::vector<ControllerEvent> events;

 private:
  const ControllerEventCode code_;
};

// Nothing on the control thread throws or logs: a contact estimate without one flag per contact point, which used to
// throw out of the control cycle, is refused, keeps the measured contact state of the cycle before, and is reported once
// through the event sink.
TEST(WBMpcMrtJointController, AContactEstimateOfTheWrongSizeIsRefusedReportedOnceAndKeepsTheMeasuredContactState) {
  // Declared first, so that it outlives the controller, which may report until it is destroyed.
  CodeRecordingSink sink(ControllerEventCode::kContactEstimateRefused);
  WholeBodyHarness harness(InProcessMpcLink::Execution::kCaller);
  harness.controller().setEventSink(&sink);
  harness.startWithoutThread();
  harness.controller().setControlMode("JOINT_PD");
  harness.setContactFlags(/*left=*/true, /*right=*/false);
  harness.cycle(std::chrono::microseconds(0));
  EXPECT_EQ(harness.controller().getMeasuredContactFlags(), (contact_flag_t{true, false}));
  EXPECT_EQ(harness.controller().getNumRefusedContactEstimates(), 0U) << "positive control: one flag per contact point is taken";

  harness.controller().setContactEstimator(std::make_shared<SingleFlagContactEstimator>());
  harness.cycle(std::chrono::microseconds(0));
  harness.cycle(std::chrono::microseconds(0));
  EXPECT_EQ(harness.controller().getMeasuredContactFlags(), (contact_flag_t{true, false}))
      << "a refused estimate changed the contact state";
  EXPECT_EQ(harness.controller().getNumRefusedContactEstimates(), 2U);
  ASSERT_EQ(sink.events.size(), 1U) << "the refusal is reported once, not on every cycle";
  EXPECT_EQ(sink.events[0].values[0], 1.0);
  EXPECT_EQ(controllerEventText(sink.events[0]), "SingleFlagContactEstimator");
  EXPECT_STREQ(sink.events[0].controller, "WBMpcMrtJointController");
  harness.controller().setEventSink(nullptr);
}

TEST(WBMpcMrtJointController, AnInvalidContactWrenchGateIsRefusedAndTheGateInUseKept) {
  // Declared first, so that it outlives the controller, which may report until it is destroyed.
  CodeRecordingSink sink(ControllerEventCode::kContactWrenchGateRefused);
  WholeBodyHarness harness(InProcessMpcLink::Execution::kCaller);
  harness.controller().setEventSink(&sink);
  harness.controller().setContactWrenchGateConfig({.debounceTime = 0.02, .rampTime = 0.05});
  harness.controller().setContactWrenchGateConfig({.debounceTime = 0.0, .rampTime = -1.0});
  EXPECT_EQ(harness.controller().getContactWrenchGate().getConfig().rampTime, 0.05);
  ASSERT_EQ(sink.events.size(), 1U);
  EXPECT_EQ(sink.events[0].values[1], -1.0);
  harness.controller().setEventSink(nullptr);
}

}  // namespace
}  // namespace ocs2::humanoid
