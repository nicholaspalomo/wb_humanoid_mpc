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
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/control/ManifoldLinearController.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StateManifoldFactory.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/mrt/ControllerEventSink.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/NullMpcLink.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/ScriptedPolicyMpcLink.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "humanoid_wb_mpc/dynamics/DynamicsHelperFunctions.h"
#include "humanoid_wb_mpc/mrt/WBMpcMrtJointController.h"
#include "robot_model/AlwaysInContactEstimator.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"
#include "robot_model/RobotStateContactEstimator.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"

/*
 * What the whole-body MRT joint controller does on the realtime thread of the robot process allocates nothing once it is
 * built: the observation of every cycle, the passive modes (each entry included), the hold of WB_MPC until a policy
 * arrives, the weight compensation of WB_MPC before its first policy, and the execution of a policy in WB_MPC - its
 * evaluation at the lookahead and at the measurement (LookaheadPolicyEvaluator) and its inverse dynamics. Pinocchio's
 * passes over the MPC model's composite base joint made a heap temporary each (the nonlinear effects of the gravity
 * compensation, the mass matrix, nonlinear effects and sole Jacobians of the inverse dynamics); the controller runs them
 * on the model's FreeFlyer twin (RealtimeRigidBodyDynamics), which makes none, and the counts are held at zero. The
 * weight compensation is also held to the base-held inverse dynamics of the weight-compensating input
 * (weightCompensatingInput(), computeBaseHeldJointTorques()) it replaced.
 */

namespace ocs2::humanoid {
namespace {

std::string runfilePath(const std::string& relativePath) {
  const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR");
  const std::filesystem::path candidate =
      srcDir != nullptr ? std::filesystem::path(srcDir) / "_main" / relativePath : std::filesystem::path(relativePath);
  return candidate.string();
}

class CountingSink final : public ControllerEventSink {
 public:
  bool post(const ControllerEvent& /*event*/) override {
    posted.fetch_add(1);
    return true;
  }
  std::atomic<int> posted{0};
};

constexpr const char* absl_nonnull kModes[] = {"ZERO_TORQUE", "JOINT_PD", "GRAVITY_COMP", "SAFETY", "JOINT_PD", "WB_MPC", "ZERO_TORQUE"};
constexpr int kCyclesPerMode = 20;

void runModes(WBMpcMrtJointController& controller,
              const std::vector<scalar_t>& nominal,
              robot::model::RobotState& state,
              robot::model::RobotJointAction& action,
              scalar_t& time,
              std::vector<size_t>* absl_nullable allocationsPerMode = nullptr) {
  for (const char* absl_nonnull mode : kModes) {
    const size_t before = robot::realtime::heapAllocationCountOnThisThread();
    for (int cycle = 0; cycle < kCyclesPerMode; ++cycle) {
      // The whole-body controller takes the posture before the mode, as the robot process hands them (CycleInputOrder).
      controller.setNominalJointPositions(nominal);
      controller.setControlMode(mode);
      time += 0.002;
      state.setTime(time);
      controller.computeJointControlAction(time, state, action);
    }
    if (allocationsPerMode != nullptr) allocationsPerMode->push_back(robot::realtime::heapAllocationCountOnThisThread() - before);
  }
}

TEST(WBMpcMrtJointControllerAllocations, TheObservationThePassiveModesAndTheHoldAllocateNothing) {
  const std::string taskFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto");
  const std::string urdfFile = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
  const ModelSettings modelSettings = ModelSettings::Create(taskFile, urdfFile, "wb_mpc_", /*verbose=*/false).value();
  const PinocchioInterface pinocchioInterface = loadCustomPinocchioInterface(taskFile, urdfFile, modelSettings).value();
  absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus = robot::model::RobotDescription::Create(urdfFile);
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  robot::model::RobotState state(description);
  state.setRootPositionInWorldFrame(vector3_t(0.0, 0.0, 0.75));
  robot::model::RobotJointAction action(description);
  absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>> created = WBMpcMrtJointController::Create(
      description, modelSettings, test_support::NullMpcLink::factory(), pinocchioInterface, jointPdGainsFileBeside(taskFile));
  ASSERT_TRUE(created.ok()) << created.status();
  WBMpcMrtJointController& controller = **created;
  CountingSink sink;
  controller.setEventSink(&sink);
  const std::vector<scalar_t> nominal(description.getNumJoints(), 0.1);
  scalar_t time = 0.0;

  runModes(controller, nominal, state, action, time);  // the first time through, sizing what sizes itself on first use
  const int eventsBefore = sink.posted.load();

  std::vector<size_t> allocationsPerMode;
  allocationsPerMode.reserve(std::size(kModes));
  runModes(controller, nominal, state, action, time, &allocationsPerMode);
  ASSERT_EQ(allocationsPerMode.size(), std::size(kModes));
  for (size_t index = 0; index < allocationsPerMode.size(); ++index) {
    EXPECT_EQ(allocationsPerMode[index], 0u) << kCyclesPerMode << " cycles of " << kModes[index] << " (entry included) allocated";
  }
  EXPECT_GT(sink.posted.load(), eventsBefore) << "SAFETY's entry was reported through the sink";
  EXPECT_EQ(controller.getControlMode(), "ZERO_TORQUE");
}

TEST(WBMpcMrtJointControllerAllocations, AContactEstimatorSwapAllocatesNothingOnTheRealtimeThread) {
  const std::string taskFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto");
  const std::string urdfFile = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
  const ModelSettings modelSettings = ModelSettings::Create(taskFile, urdfFile, "wb_mpc_", /*verbose=*/false).value();
  const PinocchioInterface pinocchioInterface = loadCustomPinocchioInterface(taskFile, urdfFile, modelSettings).value();
  const absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus = robot::model::RobotDescription::Create(urdfFile);
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>> created = WBMpcMrtJointController::Create(
      description, modelSettings, test_support::NullMpcLink::factory(), pinocchioInterface, jointPdGainsFileBeside(taskFile));
  ASSERT_TRUE(created.ok()) << created.status();
  WBMpcMrtJointController& controller = **created;
  CountingSink sink;
  controller.setEventSink(&sink);

  auto estimatorA = std::make_shared<robot::model::RobotStateContactEstimator>();
  auto estimatorB = std::make_shared<robot::model::AlwaysInContactEstimator>();
  controller.setContactEstimator(estimatorA);
  const size_t before = robot::realtime::heapAllocationCountOnThisThread();
  controller.setContactEstimator(estimatorB);
  controller.setContactEstimator(estimatorA);
  EXPECT_EQ(robot::realtime::heapAllocationCountOnThisThread() - before, 0u);
}

/** `count` control cycles in `mode`, 2 ms apart from `time`, which it advances. */
void runCycles(WBMpcMrtJointController& controller,
               const std::vector<scalar_t>& nominal,
               robot::model::RobotState& state,
               robot::model::RobotJointAction& action,
               scalar_t& time,
               const char* absl_nonnull mode,
               int count) {
  for (int cycle = 0; cycle < count; ++cycle) {
    controller.setNominalJointPositions(nominal);
    controller.setControlMode(mode);
    time += 0.002;
    state.setTime(time);
    controller.computeJointControlAction(time, state, action);
  }
}

TEST(WBMpcMrtJointControllerAllocations, ExecutingAPolicyAddsNoAllocationOfItsOwn) {
  const std::string taskFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto");
  const std::string urdfFile = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
  const ModelSettings modelSettings = ModelSettings::Create(taskFile, urdfFile, "wb_mpc_", /*verbose=*/false).value();
  const PinocchioInterface pinocchioInterface = loadCustomPinocchioInterface(taskFile, urdfFile, modelSettings).value();
  const absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus = robot::model::RobotDescription::Create(urdfFile);
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  const WBAccelMpcRobotModel<scalar_t> model(modelSettings);
  const std::shared_ptr<const StateManifold> manifold = stateManifoldOf(model).value();
  test_support::ScriptedPolicyMpcLink* absl_nullable link = nullptr;
  absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>> created =
      WBMpcMrtJointController::Create(description, modelSettings, test_support::ScriptedPolicyMpcLink::factory(&link), pinocchioInterface,
                                      jointPdGainsFileBeside(taskFile));
  ASSERT_TRUE(created.ok()) << created.status();
  ASSERT_NE(link, nullptr);
  WBMpcMrtJointController& controller = **created;
  CountingSink sink;
  controller.setEventSink(&sink);
  robot::model::RobotState state(description);
  state.setRootPositionInWorldFrame(vector3_t(0.0, 0.0, 0.75));
  state.setContactFlag(/*index=*/0, /*contactFlag=*/true);
  state.setContactFlag(/*index=*/1, /*contactFlag=*/true);
  robot::model::RobotJointAction action(description);
  const std::vector<scalar_t> nominal(description.getNumJoints(), 0.1);
  scalar_t time = 0.0;
  runCycles(controller, nominal, state, action, time, "JOINT_PD", kCyclesPerMode);
  runCycles(controller, nominal, state, action, time, "WB_MPC",
            kCyclesPerMode);  // the hold of the entry, until a policy solved after its reset is in use

  // A moving manifold-linear plan with gains: the feedback is evaluated at the measurement and at the lookahead.
  const Eigen::Index nu = static_cast<Eigen::Index>(model.getInputDim());
  const Eigen::Index ndx = static_cast<Eigen::Index>(manifold->getTangentDim());
  vector_t start = link->lastObservation().state;
  vector_t end = start;
  for (Eigen::Index i = 0; i < end.size(); ++i) end(i) += 0.01 * static_cast<scalar_t>(i % 7);
  manifold->project(end);
  vector_t input = vector_t::Zero(nu);
  input(2) = 150.0;  // [N] on the left foot
  input(8) = 150.0;  // [N] on the right foot
  PrimalSolution policy;
  policy.timeTrajectory_ = {time, time + 10.0};
  policy.stateTrajectory_ = {start, end};
  policy.inputTrajectory_ = {input, input};
  policy.modeSchedule_ = ModeSchedule(/*eventTimesInput=*/{}, /*modeSequenceInput=*/{static_cast<size_t>(ModeNumber::kStance)});
  policy.controllerPtr_ = std::make_unique<ManifoldLinearController>(
      policy.timeTrajectory_, policy.stateTrajectory_, policy.inputTrajectory_,
      matrix_array_t{matrix_t::Constant(nu, ndx, 0.5), matrix_t::Constant(nu, ndx, 0.5)}, manifold);
  link->deliver(policy);
  runCycles(controller, nominal, state, action, time, "WB_MPC",
            kCyclesPerMode);  // takes the policy into use and sizes what sizes itself on first use
  ASSERT_FALSE(controller.isHolding()) << "the policy is executed";

  const size_t before = robot::realtime::heapAllocationCountOnThisThread();
  runCycles(controller, nominal, state, action, time, "WB_MPC", kCyclesPerMode);
  EXPECT_EQ(robot::realtime::heapAllocationCountOnThisThread() - before, 0u) << kCyclesPerMode << " cycles executing a policy allocated";
}

/** `count` control cycles, 2 ms apart from `time`, which it advances, in the controller's own mode: no mode is commanded. */
void runCyclesInItsOwnMode(WBMpcMrtJointController& controller,
                           const std::vector<scalar_t>& nominal,
                           robot::model::RobotState& state,
                           robot::model::RobotJointAction& action,
                           scalar_t& time,
                           int count) {
  for (int cycle = 0; cycle < count; ++cycle) {
    controller.setNominalJointPositions(nominal);
    time += 0.002;
    state.setTime(time);
    controller.computeJointControlAction(time, state, action);
  }
}

TEST(WBMpcMrtJointControllerAllocations, TheWeightCompensationBeforeAPolicyIsTheBaseHeldInverseDynamicsAndAllocatesNothing) {
  const std::string taskFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto");
  const std::string urdfFile = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
  const ModelSettings modelSettings = ModelSettings::Create(taskFile, urdfFile, "wb_mpc_", /*verbose=*/false).value();
  PinocchioInterface pinocchioInterface = loadCustomPinocchioInterface(taskFile, urdfFile, modelSettings).value();
  const absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus = robot::model::RobotDescription::Create(urdfFile);
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  const WBAccelMpcRobotModel<scalar_t> model(modelSettings);
  absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>> created = WBMpcMrtJointController::Create(
      description, modelSettings, test_support::NullMpcLink::factory(), pinocchioInterface, jointPdGainsFileBeside(taskFile));
  ASSERT_TRUE(created.ok()) << created.status();
  WBMpcMrtJointController& controller = **created;
  CountingSink sink;
  controller.setEventSink(&sink);
  // A moving, tilted robot, so that the base velocity and the attitude reach the torques.
  robot::model::RobotState state(description);
  state.setRootPositionInWorldFrame(vector3_t(0.1, -0.2, 0.75));
  state.setRootRotationLocalToWorldFrame(quaternion_t(Eigen::AngleAxisd(0.2, vector3_t(1.0, 2.0, -1.0).normalized())));
  state.setRootLinearVelocityInLocalFrame(vector3_t(0.3, -0.1, 0.2));
  state.setRootAngularVelocityInLocalFrame(vector3_t(0.4, 0.5, -0.6));
  for (size_t joint = 0; joint < description.getNumJoints(); ++joint) {
    state.setJointPosition(joint, /*jointPosition=*/0.05 * static_cast<scalar_t>(joint % 5));
    state.setJointVelocity(joint, /*jointVelocity=*/0.1 * static_cast<scalar_t>(joint % 3) - 0.1);
  }
  robot::model::RobotJointAction action(description);
  const std::vector<scalar_t> nominal(description.getNumJoints(), 0.1);
  scalar_t time = 0.0;

  // The controller starts in WB_MPC with no hold armed and no policy: it compensates the weight on the measured feet.
  for (const contact_flag_t& contacts : {contact_flag_t{true, true}, contact_flag_t{true, false}, contact_flag_t{false, false}}) {
    SCOPED_TRACE(::testing::Message() << contacts[0] << contacts[1]);
    state.setContactFlag(/*index=*/0, /*contactFlag=*/contacts[0]);
    state.setContactFlag(/*index=*/1, /*contactFlag=*/contacts[1]);
    runCyclesInItsOwnMode(controller, nominal, state, action, time, /*count=*/1);  // sizes what sizes itself on first use
    const size_t before = robot::realtime::heapAllocationCountOnThisThread();
    runCyclesInItsOwnMode(controller, nominal, state, action, time, kCyclesPerMode);
    EXPECT_EQ(robot::realtime::heapAllocationCountOnThisThread() - before, 0u);
    ASSERT_EQ(controller.getControlMode(), "WB_MPC");

    // The weight-compensating input and the base-held inverse dynamics of it, at the observation.
    const vector_t& observed = controller.getCurrentObservation().state;
    PinocchioInterface reference = pinocchioInterface;
    const vector_t input = weightCompensatingInput(reference, contacts, model);
    EXPECT_LT((controller.getLatestPolicyInput() - input).cwiseAbs().maxCoeff(), 1.0e-9);
    const vector_t torques = computeBaseHeldJointTorques<scalar_t>(observed, input, reference, model);
    const std::vector<size_t> mpcJoints = description.getJointIndices(modelSettings.mpcModelJointNames);
    for (size_t i = 0; i < mpcJoints.size(); ++i) {
      const std::optional<robot::model::JointAction>& slot = action[mpcJoints[i]];
      if (!slot.has_value()) {
        ADD_FAILURE() << "no action for " << modelSettings.mpcModelJointNames[i];
        continue;
      }
      const robot::model::JointAction& jointAction = *slot;
      const scalar_t limit = controller.getPdGains().mpcJointTorqueLimit[i];
      EXPECT_NEAR(jointAction.feed_forward_effort, std::clamp(torques[static_cast<Eigen::Index>(i)], -limit, limit), 1.0e-8)
          << modelSettings.mpcModelJointNames[i];
    }
  }
}

}  // namespace
}  // namespace ocs2::humanoid
