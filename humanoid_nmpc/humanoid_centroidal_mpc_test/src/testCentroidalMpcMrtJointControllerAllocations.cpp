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
#include <iterator>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/AccessHelperFunctions.h"
#include "ocs2_centroidal_model/ModelHelperFunctions.h"
#include "ocs2_core/control/ManifoldLinearController.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h"
#include "humanoid_centroidal_mpc_test/CentroidalTestingModelInterface.h"
#include "humanoid_common_mpc/common/StateManifoldFactory.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/mrt/ControllerEventSink.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/NullMpcLink.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/ScriptedPolicyMpcLink.h"
#include "robot_model/AlwaysInContactEstimator.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"
#include "robot_model/RobotStateContactEstimator.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"
#include "support/TypedConfigFiles.h"

/*
 * What the centroidal MRT joint controller does on the realtime thread of the
 * robot process allocates nothing once it is built: the observation of every
 * cycle, the passive modes (ZERO_TORQUE, JOINT_PD, GRAVITY_COMP, SAFETY, each
 * entry included), the hold of WB_MPC until a policy arrives, the weight
 * compensation of WB_MPC before its first policy, and the execution of a policy
 * - on the wrench-space inputs of the G1 and the basis-vector inputs of the
 * Atlas. A cycle used to make a dozen (the joint vectors of the robot state,
 * the contact flags, the gravity compensation's workspace, a std::function),
 * SAFETY's first cycle resized its posture, Pinocchio's passes over the MPC
 * model's composite base joint made a heap temporary each (the centroidal map
 * of the observation, the nonlinear effects of the gravity compensation, the
 * mass matrix, nonlinear effects and sole Jacobians of the inverse dynamics),
 * and the policy's execution copied the model's vectors and a Pinocchio data
 * per wrench. The controller runs Pinocchio on the model's FreeFlyer twin
 * (RealtimeRigidBodyDynamics), which makes no temporary, and the counts are
 * held at zero. The observation's momentum and the weight compensation are also
 * held to what updateCentroidalDynamics() and computeBaseHeldJointTorques() of
 * the weight-compensating input give.
 */

namespace ocs2::humanoid {
namespace {

/** A sink that counts, as the realtime event log takes events: a copy, no
 * allocation. */
class CountingSink final : public ControllerEventSink {
 public:
  bool post(const ControllerEvent& /*event*/) override {
    posted.fetch_add(1);
    return true;
  }
  std::atomic<int> posted{0};
};

/** The modes the robot process hands the controller, in turn: every entry, and
 * the hold of WB_MPC. */
constexpr const char* absl_nonnull kModes[] = {
    "ZERO_TORQUE", "JOINT_PD", "GRAVITY_COMP", "SAFETY",
    "JOINT_PD",    "WB_MPC",   "ZERO_TORQUE"};
constexpr int kCyclesPerMode = 20;

/** What the robot process calls every cycle - the mode and the posture, then
 * the action - through kModes. */
void runModes(CentroidalMpcMrtJointController& controller,
              const std::vector<scalar_t>& nominal,
              robot::model::RobotState& state,
              robot::model::RobotJointAction& action,
              scalar_t& time,
              std::vector<size_t>* absl_nullable allocationsPerMode = nullptr) {
  for (const char* absl_nonnull mode : kModes) {
    const size_t before = robot::realtime::heapAllocationCountOnThisThread();
    for (int cycle = 0; cycle < kCyclesPerMode; ++cycle) {
      controller.setControlMode(mode);
      controller.setNominalJointPositions(nominal);
      time += 0.01;
      state.setTime(time);
      controller.computeJointControlAction(time, state, action);
    }
    if (allocationsPerMode != nullptr)
      allocationsPerMode->push_back(
          robot::realtime::heapAllocationCountOnThisThread() - before);
  }
}

TEST(CentroidalMpcMrtJointControllerAllocations,
     TheObservationThePassiveModesAndTheHoldAllocateNothing) {
  CentroidalTestingModelInterface model;
  absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus =
      robot::model::RobotDescription::Create(model.urdfFile);
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  robot::model::RobotState state(description);
  state.setRootPositionInWorldFrame(vector3_t(0.0, 0.0, 0.75));
  robot::model::RobotJointAction action(description);
  absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>>
      controllerOrStatus = CentroidalMpcMrtJointController::Create(
          description, model.getModelSettings(), model.getMpcRobotModel(),
          test_support::NullMpcLink::factory(), model.getPinocchioInterface(),
          jointPdGainsFileBeside(model.taskFile));
  ASSERT_TRUE(controllerOrStatus.ok()) << controllerOrStatus.status();
  CentroidalMpcMrtJointController& controller = **controllerOrStatus;
  CountingSink sink;
  controller.setEventSink(&sink);
  const std::vector<scalar_t> nominal(description.getNumJoints(), 0.1);
  scalar_t time = 0.0;

  runModes(
      controller, nominal, state, action,
      time);  // the first time through, sizing what sizes itself on first use
  const int eventsBefore = sink.posted.load();

  std::vector<size_t> allocationsPerMode;
  allocationsPerMode.reserve(std::size(kModes));
  runModes(controller, nominal, state, action, time, &allocationsPerMode);
  ASSERT_EQ(allocationsPerMode.size(), std::size(kModes));
  for (size_t index = 0; index < allocationsPerMode.size(); ++index) {
    EXPECT_EQ(allocationsPerMode[index], 0u)
        << kCyclesPerMode << " cycles of " << kModes[index]
        << " (entry included) allocated";
  }
  EXPECT_GT(sink.posted.load(), eventsBefore)
      << "SAFETY's entry was reported through the sink";
  EXPECT_EQ(controller.getControlMode(), "ZERO_TORQUE");
}

TEST(CentroidalMpcMrtJointControllerAllocations,
     AContactEstimatorSwapAllocatesNothingOnTheRealtimeThread) {
  CentroidalTestingModelInterface model;
  const absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus =
      robot::model::RobotDescription::Create(model.urdfFile);
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  const absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>>
      controllerOrStatus = CentroidalMpcMrtJointController::Create(
          description, model.getModelSettings(), model.getMpcRobotModel(),
          test_support::NullMpcLink::factory(), model.getPinocchioInterface(),
          jointPdGainsFileBeside(model.taskFile));
  ASSERT_TRUE(controllerOrStatus.ok()) << controllerOrStatus.status();
  CentroidalMpcMrtJointController& controller = **controllerOrStatus;
  CountingSink sink;
  controller.setEventSink(&sink);

  auto estimatorA =
      std::make_shared<robot::model::RobotStateContactEstimator>();
  auto estimatorB = std::make_shared<robot::model::AlwaysInContactEstimator>();
  controller.setContactEstimator(estimatorA);
  const size_t before = robot::realtime::heapAllocationCountOnThisThread();
  controller.setContactEstimator(estimatorB);
  controller.setContactEstimator(estimatorA);
  EXPECT_EQ(robot::realtime::heapAllocationCountOnThisThread() - before, 0u);
}

/** A shipped robot, by its files, and the input parameterization its task file
 * selects. */
struct ShippedRobot {
  const char* absl_nonnull name;
  CentroidalRobotFiles (*absl_nonnull files)();
};

void PrintTo(const ShippedRobot& robot, std::ostream* absl_nonnull os) {
  *os << robot.name;
}

// The G1 ships wrench-space inputs, the Atlas basis-vector inputs
// (contact_input_parameterization).
constexpr ShippedRobot kShippedRobots[] = {
    {.name = "unitree_g1", .files = &g1Files},
    {.name = "drc_atlas", .files = &atlasFiles}};

/** `count` control cycles, 10 ms apart from `time`, which it advances; in
 * `mode`, or in the controller's own without one. */
void runCycles(CentroidalMpcMrtJointController& controller,
               const std::vector<scalar_t>& nominal,
               robot::model::RobotState& state,
               robot::model::RobotJointAction& action,
               scalar_t& time,
               const char* absl_nullable mode,
               int count) {
  for (int cycle = 0; cycle < count; ++cycle) {
    if (mode != nullptr) controller.setControlMode(mode);
    controller.setNominalJointPositions(nominal);
    time += 0.01;
    state.setTime(time);
    controller.computeJointControlAction(time, state, action);
  }
}

/**
 * The controller as the robot binary builds it (CentroidalMpcRobotMain.cpp)
 * from the interface of CreateControllerModels(), on a moving, tilted robot
 * standing on both feet.
 */
class CentroidalControllerOnShippedRobot
    : public ::testing::TestWithParam<ShippedRobot> {
 protected:
  void SetUp() override {
    const CentroidalRobotFiles files = GetParam().files();
    taskFile_ = files.taskFile;
    absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
        CentroidalMpcInterface::CreateControllerModels(
            files.taskFile, files.urdfFile, files.referenceFile);
    ASSERT_TRUE(interface.ok()) << interface.status();
    interface_ = *std::move(interface);
    absl::StatusOr<robot::model::RobotDescription> description =
        robot::model::RobotDescription::Create(files.urdfFile);
    ASSERT_TRUE(description.ok()) << description.status();
    description_ = std::make_unique<robot::model::RobotDescription>(
        *std::move(description));
    absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>>
        controller = CentroidalMpcMrtJointController::Create(
            *description_, interface_->modelSettings(),
            interface_->getMpcRobotModel(),
            test_support::ScriptedPolicyMpcLink::factory(&link_),
            interface_->getPinocchioInterface(),
            jointPdGainsFileBeside(taskFile_),
            &interface_->getEffectiveMpcRobotModel());
    ASSERT_TRUE(controller.ok()) << controller.status();
    ASSERT_NE(link_, nullptr);
    controller_ = *std::move(controller);
    controller_->setEventSink(&sink_);

    state_ = std::make_unique<robot::model::RobotState>(*description_);
    state_->setRootPositionInWorldFrame(vector3_t(0.1, -0.2, 0.9));
    state_->setRootRotationLocalToWorldFrame(quaternion_t(
        Eigen::AngleAxisd(0.2, vector3_t(1.0, 2.0, -1.0).normalized())));
    state_->setRootLinearVelocityInLocalFrame(vector3_t(0.3, -0.1, 0.2));
    state_->setRootAngularVelocityInLocalFrame(vector3_t(0.4, 0.5, -0.6));
    for (size_t joint = 0; joint < description_->getNumJoints(); ++joint) {
      state_->setJointPosition(
          joint, /*jointPosition=*/0.05 * static_cast<scalar_t>(joint % 5));
      state_->setJointVelocity(
          joint,
          /*jointVelocity=*/0.1 * static_cast<scalar_t>(joint % 3) - 0.1);
    }
    state_->setContactFlag(/*index=*/0, /*contactFlag=*/true);
    state_->setContactFlag(/*index=*/1, /*contactFlag=*/true);
    action_ = std::make_unique<robot::model::RobotJointAction>(*description_);
    nominal_.assign(description_->getNumJoints(), 0.1);
  }

  std::string taskFile_;
  std::unique_ptr<CentroidalMpcInterface> interface_;
  std::unique_ptr<robot::model::RobotDescription> description_;
  test_support::ScriptedPolicyMpcLink* absl_nullable link_ = nullptr;
  std::unique_ptr<CentroidalMpcMrtJointController> controller_;
  CountingSink sink_;
  std::unique_ptr<robot::model::RobotState> state_;
  std::unique_ptr<robot::model::RobotJointAction> action_;
  std::vector<scalar_t> nominal_;
  scalar_t time_ = 0.0;
};

TEST_P(
    CentroidalControllerOnShippedRobot,
    TheObservedMomentumIsTheCentroidalMapsAndTheWeightCompensationAllocatesNothing) {
  // The controller starts in WB_MPC with no hold armed and no policy: it
  // compensates the weight on the measured feet.
  for (const contact_flag_t& contacts :
       {contact_flag_t{true, true}, contact_flag_t{false, true},
        contact_flag_t{false, false}}) {
    SCOPED_TRACE(::testing::Message() << contacts[0] << contacts[1]);
    state_->setContactFlag(/*index=*/0, /*contactFlag=*/contacts[0]);
    state_->setContactFlag(/*index=*/1, /*contactFlag=*/contacts[1]);
    runCycles(*controller_, nominal_, *state_, *action_, time_,
              /*mode=*/nullptr, /*count=*/1);  // sizes what sizes itself
    const size_t before = robot::realtime::heapAllocationCountOnThisThread();
    runCycles(*controller_, nominal_, *state_, *action_, time_,
              /*mode=*/nullptr, kCyclesPerMode);
    EXPECT_EQ(robot::realtime::heapAllocationCountOnThisThread() - before, 0u);
    ASSERT_EQ(controller_->getControlMode(), "WB_MPC");

    // The observation: h = A(q) v / m of updateCentroidalDynamics() on the MPC
    // model, at the measured q and v, which the model's
    // getGeneralizedVelocities() gives back from the observation.
    const SystemObservation& observation = link_->lastObservation();
    const std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> modelCopy(
        interface_->getMpcRobotModel().clone());
    CentroidalMpcRobotModel<scalar_t>& model = *modelCopy;
    const CentroidalModelInfo& info = model.getCentroidalModelInfo();
    PinocchioInterface reference = interface_->getPinocchioInterface();
    // Through the effective model, whose input layout the observation's is (the
    // basis-vector one on the Atlas).
    const std::unique_ptr<MpcRobotModelBase<scalar_t>> effectiveCopy(
        interface_->getEffectiveMpcRobotModel().clone());
    const vector_t q =
        effectiveCopy->getGeneralizedCoordinates(observation.state);
    const vector_t v = effectiveCopy->getGeneralizedVelocities(
        observation.state, observation.input);
    updateCentroidalDynamics(reference, info, q);
    const vector6_t momentum =
        getCentroidalMomentumMatrix(reference) * v / info.robotMass;
    EXPECT_LT(
        (centroidal_model::getNormalizedMomentum(observation.state, info) -
         momentum)
            .cwiseAbs()
            .maxCoeff(),
        1.0e-10);
    // The base velocity it gives back is the measured one, [dp_W, w_B]: the
    // momentum is A(q) of the measurement.
    const vector3_t measuredLinearVelocity =
        state_->getRootRotationLocalToWorldFrame() *
        state_->getRootLinearVelocityInLocalFrame();
    EXPECT_LT((v.head<3>() - measuredLinearVelocity).cwiseAbs().maxCoeff(),
              1.0e-9)
        << "the observation's base velocity is the measured one";
    EXPECT_LT((v.segment<3>(3) - state_->getRootAngularVelocityInLocalFrame())
                  .cwiseAbs()
                  .maxCoeff(),
              1.0e-9);

    // The feedforward: the base-held inverse dynamics of the weight shared by
    // the feet in contact as vertical world-frame forces. Those are the
    // world-frame wrenches of the weight-compensating input of a wrench-space
    // model exactly; a basis-vector input reproduces them only inside the cone
    // its basis spans (its non-negative least-squares fit), which the tilted
    // Atlas feet here leave, and the controller compensates the weight itself,
    // not that fit.
    const MpcRobotModelBase<scalar_t>& effective =
        interface_->getEffectiveMpcRobotModel();
    const vector_t input = weightCompensatingInput(
        reference, contacts, effective, observation.state);
    const size_t numStanceLegs = numberOfLegsInContacts(contacts);
    feet_array_t<vector6_t> wrenches;
    for (size_t i = 0; i < kNumContacts; ++i) {
      wrenches[i].setZero();
      if (contacts[i])
        wrenches[i](2) = computeRobotWeight(reference) /
                         static_cast<scalar_t>(numStanceLegs);
      if (effective.getContactInputDim(i) == kContactWrenchDim) {
        EXPECT_LT((effective.getContactWrenchInWorldFrame(observation.state,
                                                          input, i) -
                   wrenches[i])
                      .cwiseAbs()
                      .maxCoeff(),
                  1.0e-9);
      }
    }
    const vector_t torques = computeBaseHeldJointTorques<scalar_t>(
        q, v, vector_t::Zero(model.getJointDim()), wrenches,
        getContactFrameIndexArray<scalar_t>(reference, model), reference);
    const std::vector<size_t> mpcJoints = description_->getJointIndices(
        interface_->modelSettings().mpcModelJointNames);
    for (size_t i = 0; i < mpcJoints.size(); ++i) {
      const scalar_t limit = controller_->getPdGains().mpcJointTorqueLimit[i];
      const std::optional<robot::model::JointAction>& jointAction =
          (*action_)[mpcJoints[i]];
      if (!jointAction.has_value()) {
        ADD_FAILURE() << "no action for "
                      << interface_->modelSettings().mpcModelJointNames[i];
        continue;
      }
      EXPECT_NEAR(
          jointAction->feed_forward_effort,
          std::clamp(torques[static_cast<Eigen::Index>(i)], -limit, limit),
          1.0e-8)
          << interface_->modelSettings().mpcModelJointNames[i];
    }
  }
}

TEST_P(CentroidalControllerOnShippedRobot, ExecutingAPolicyAllocatesNothing) {
  runCycles(*controller_, nominal_, *state_, *action_, time_, "JOINT_PD",
            kCyclesPerMode);
  runCycles(*controller_, nominal_, *state_, *action_, time_, "WB_MPC",
            kCyclesPerMode);  // the entry's hold

  // A moving manifold-linear plan with gains and loaded feet: the feedback is
  // evaluated at the measurement and at the lookahead, the wrenches rotated
  // into the world (basis-vector inputs) and gated, the inverse dynamics
  // solved.
  const MpcRobotModelBase<scalar_t>& effective =
      interface_->getEffectiveMpcRobotModel();
  const std::shared_ptr<const StateManifold> manifold =
      stateManifoldOf(interface_->getMpcRobotModel()).value();
  const Eigen::Index nu = static_cast<Eigen::Index>(effective.getInputDim());
  const Eigen::Index ndx = static_cast<Eigen::Index>(manifold->getTangentDim());
  const vector_t start = link_->lastObservation().state;
  vector_t end = start;
  for (Eigen::Index i = 0; i < end.size(); ++i)
    end(i) += 0.01 * static_cast<scalar_t>(i % 7);
  manifold->project(end);
  PinocchioInterface reference = interface_->getPinocchioInterface();
  const vector_t input = weightCompensatingInput(
      reference, contact_flag_t{true, true}, effective, start);
  PrimalSolution policy;
  policy.timeTrajectory_ = {time_, time_ + 10.0};
  policy.stateTrajectory_ = {start, end};
  policy.inputTrajectory_ = {input, input};
  policy.modeSchedule_ = ModeSchedule(
      /*eventTimesInput=*/{},
      /*modeSequenceInput=*/{static_cast<size_t>(ModeNumber::kStance)});
  policy.controllerPtr_ = std::make_unique<ManifoldLinearController>(
      policy.timeTrajectory_, policy.stateTrajectory_, policy.inputTrajectory_,
      matrix_array_t{matrix_t::Constant(nu, ndx, 0.01),
                     matrix_t::Constant(nu, ndx, 0.01)},
      manifold);
  link_->deliver(policy);
  runCycles(*controller_, nominal_, *state_, *action_, time_, "WB_MPC",
            10 * kCyclesPerMode);  // into use, past the entry ramp
  ASSERT_FALSE(controller_->isEnteringMpc()) << "the policy is executed";

  const size_t before = robot::realtime::heapAllocationCountOnThisThread();
  runCycles(*controller_, nominal_, *state_, *action_, time_, "WB_MPC",
            kCyclesPerMode);
  EXPECT_EQ(robot::realtime::heapAllocationCountOnThisThread() - before, 0u)
      << kCyclesPerMode << " cycles executing a policy allocated";
}

INSTANTIATE_TEST_SUITE_P(ShippedRobots,
                         CentroidalControllerOnShippedRobot,
                         ::testing::ValuesIn(kShippedRobots));

}  // namespace
}  // namespace ocs2::humanoid
