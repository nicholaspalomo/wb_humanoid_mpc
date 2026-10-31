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
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/control/FeedforwardController.h"
#include "ocs2_core/control/LinearController.h"
#include "ocs2_core/control/ManifoldLinearController.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

#include "humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h"
#include "humanoid_centroidal_mpc_test/CentroidalTestingModelInterface.h"
#include "humanoid_common_mpc/common/StateManifoldFactory.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/mrt/ControllerEvent.h"
#include "humanoid_common_mpc/mrt/ControllerEventSink.h"
#include "humanoid_common_mpc/mrt/JointActionAccess.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/ScriptedPolicyMpcLink.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"

/*
 * What the centroidal MRT joint controller does in WB_MPC with a policy its
 * realtime evaluator refuses, on the quaternion layout: a LinearController
 * (whose gains would act on the stored quaternion rather than on its tangent)
 * or a ManifoldLinearController with ambient gains is never handed to
 * MRT_BASE::evaluatePolicy(), which allocates on this thread, logs, and aborts
 * on the malformed one. The robot is held with the entry's hold, the cycle
 * allocates exactly what the hold's does, one kPolicyRefused event says why,
 * and a policy the evaluator takes is executed afterwards.
 */

namespace ocs2::humanoid {
namespace {

/** Counts the events of one code, as the realtime event log takes events: no
 * allocation. */
class CodeCountingSink final : public ControllerEventSink {
 public:
  explicit CodeCountingSink(ControllerEventCode code) : code_(code) {}
  bool post(const ControllerEvent& event) override {
    if (event.code == code_) {
      lastValue.store(event.values[0]);
      posted.fetch_add(1);
    }
    return true;
  }
  std::atomic<int> posted{0};
  std::atomic<double> lastValue{0.0};

 private:
  ControllerEventCode code_;
};

constexpr int kCycles = 20;

/** `cycles` control cycles in `mode` at `robotState`, 10 ms apart from `time`,
 * which it advances. */
void runCycles(CentroidalMpcMrtJointController& controller,
               const std::vector<scalar_t>& nominal,
               robot::model::RobotState& robotState,
               robot::model::RobotJointAction& action,
               scalar_t& time,
               const char* absl_nonnull mode,
               int cycles) {
  for (int cycle = 0; cycle < cycles; ++cycle) {
    controller.setControlMode(mode);
    controller.setNominalJointPositions(nominal);
    time += 0.01;
    robotState.setTime(time);
    controller.computeJointControlAction(time, robotState, action);
  }
}

/** The joint actions, flattened: q_des, qd_des, kp, kd and the feedforward of
 * each joint. */
std::vector<scalar_t> actionValuesOf(
    const robot::model::RobotDescription& description,
    const robot::model::RobotJointAction& action) {
  std::vector<scalar_t> values;
  for (size_t joint = 0; joint < description.getNumJoints(); ++joint) {
    const robot::model::JointAction& jointAction =
        jointActionUnchecked(action, joint);
    values.insert(values.end(),
                  {jointAction.q_des, jointAction.qd_des, jointAction.kp,
                   jointAction.kd, jointAction.feed_forward_effort});
  }
  return values;
}

/** A policy of `controller` with two nodes around `time`, `state` and `input`
 * held. */
PrimalSolution policyOf(std::unique_ptr<ControllerBase> controller,
                        const vector_t& state,
                        const vector_t& input,
                        scalar_t time) {
  PrimalSolution policy;
  policy.timeTrajectory_ = {time - 1.0, time + 100.0};
  policy.stateTrajectory_ = {state, state};
  policy.inputTrajectory_ = {input, input};
  policy.modeSchedule_ = ModeSchedule(
      /*eventTimesInput=*/{},
      /*modeSequenceInput=*/{static_cast<size_t>(ModeNumber::kStance)});
  policy.controllerPtr_ = std::move(controller);
  return policy;
}

TEST(CentroidalMpcMrtPolicyExecution,
     ARefusedPolicyHoldsTheRobotWithoutMrtBaseAndIsReportedOnce) {
  CentroidalTestingModelInterface model;
  absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus =
      robot::model::RobotDescription::Create(model.urdfFile);
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  const CentroidalMpcRobotModel<scalar_t>& mpcModel = model.getMpcRobotModel();
  absl::StatusOr<std::shared_ptr<const StateManifold>> manifoldOrStatus =
      stateManifoldOf(mpcModel);
  ASSERT_TRUE(manifoldOrStatus.ok()) << manifoldOrStatus.status();
  const std::shared_ptr<const StateManifold> manifold =
      *std::move(manifoldOrStatus);
  test_support::ScriptedPolicyMpcLink* absl_nullable link = nullptr;
  absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>>
      controllerOrStatus = CentroidalMpcMrtJointController::Create(
          description, model.getModelSettings(), mpcModel,
          test_support::ScriptedPolicyMpcLink::factory(&link),
          model.getPinocchioInterface(),
          jointPdGainsFileBeside(model.taskFile));
  ASSERT_TRUE(controllerOrStatus.ok()) << controllerOrStatus.status();
  ASSERT_NE(link, nullptr);
  CentroidalMpcMrtJointController& controller = **controllerOrStatus;
  CodeCountingSink sink(ControllerEventCode::kPolicyRefused);
  controller.setEventSink(&sink);
  robot::model::RobotState robotState(description);
  robotState.setPosBaseRtWorldEwrtWorld(vector3_t(0.0, 0.0, 0.75));
  for (size_t joint = 0; joint < description.getNumJoints(); ++joint)
    robotState.setJointPosition(joint, /*jointPosition=*/0.1);
  robot::model::RobotJointAction action(description);
  const std::vector<scalar_t> nominal(description.getNumJoints(), 0.1);
  scalar_t time = 0.0;

  runCycles(controller, nominal, robotState, action, time, "JOINT_PD", kCycles);
  // The entry's hold, until a policy solved after the reset is in use.
  runCycles(controller, nominal, robotState, action, time, "WB_MPC", kCycles);
  const std::vector<scalar_t> hold = actionValuesOf(description, action);
  size_t before = robot::realtime::heapAllocationCountOnThisThread();
  runCycles(controller, nominal, robotState, action, time, "WB_MPC", kCycles);
  const size_t holdAllocations =
      robot::realtime::heapAllocationCountOnThisThread() - before;
  EXPECT_EQ(actionValuesOf(description, action), hold);

  const Eigen::Index nx = static_cast<Eigen::Index>(mpcModel.getStateDim());
  const Eigen::Index nu = static_cast<Eigen::Index>(mpcModel.getInputDim());
  vector_t state = vector_t::Zero(nx);
  const Eigen::Index orientation =
      static_cast<Eigen::Index>(mpcModel.getBaseOrientationStartIndex());
  state(orientation - 1) = 0.75;  // the base height, the last of the base
                                  // position before the quaternion
  state.segment<4>(orientation) = vector4_t(0.0, 0.0, 0.0, 1.0);
  // The joint angles follow the base pose [p_W (3), xi (4)].
  state
      .segment(orientation + 4,
               static_cast<Eigen::Index>(mpcModel.getJointDim()))
      .setConstant(0.15);
  const vector_t input = vector_t::Zero(nu);
  const scalar_array_t times = {time - 1.0, time + 100.0};
  int expectedEvents = 0;
  for (const PolicyRefusal refusal : {PolicyRefusal::kUnsupportedController,
                                      PolicyRefusal::kMismatchedPolicy}) {
    SCOPED_TRACE(static_cast<int>(refusal));
    if (refusal == PolicyRefusal::kUnsupportedController) {
      link->deliver(policyOf(
          std::make_unique<LinearController>(
              times, vector_array_t{input, input},
              matrix_array_t{matrix_t::Ones(nu, nx), matrix_t::Ones(nu, nx)}),
          state, input, time));
    } else {
      link->deliver(policyOf(
          std::make_unique<ManifoldLinearController>(
              times, vector_array_t{state, state}, vector_array_t{input, input},
              matrix_array_t{matrix_t::Ones(nu, nx), matrix_t::Ones(nu, nx)},
              manifold),
          state, input, time));
    }
    runCycles(controller, nominal, robotState, action, time, "WB_MPC",
              /*cycles=*/1);  // takes the policy into use
    before = robot::realtime::heapAllocationCountOnThisThread();
    runCycles(controller, nominal, robotState, action, time, "WB_MPC", kCycles);
    EXPECT_EQ(robot::realtime::heapAllocationCountOnThisThread() - before,
              holdAllocations)
        << "a refused policy's cycle allocates what the hold's does: "
           "MRT_BASE::evaluatePolicy() allocates 2 to 3 per call";
    EXPECT_EQ(actionValuesOf(description, action), hold) << "the robot is held";
    ++expectedEvents;
    EXPECT_EQ(sink.posted.load(), expectedEvents)
        << "one report per run of refused cycles";
    EXPECT_EQ(sink.lastValue.load(), static_cast<double>(refusal));

    // A policy the evaluator takes is executed (after the entry ramp): no
    // longer the hold, and no further report.
    link->deliver(policyOf(std::make_unique<FeedforwardController>(
                               times, vector_array_t{input, input}),
                           state, input, time));
    runCycles(controller, nominal, robotState, action, time, "WB_MPC",
              10 * kCycles);
    EXPECT_NE(actionValuesOf(description, action), hold);
    EXPECT_EQ(sink.posted.load(), expectedEvents);
  }
}

// On a moving plan the feedback of the policy acts on the error at the
// measurement time: a robot measured exactly on the plan gets exactly the
// feedforward, at every lookahead - the joint velocity targets and the wrenches
// of the input included - so the action of a manifold-linear policy with large
// gains is that of the same plan without gains. Evaluated at the lookahead on
// the state measured before it, as it used to be, the gains fed the planned
// motion back.
TEST(CentroidalMpcMrtPolicyExecution,
     ARobotMeasuredOnAMovingPlanGetsExactlyTheFeedforwardAtEveryLookahead) {
  CentroidalTestingModelInterface model;
  absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus =
      robot::model::RobotDescription::Create(model.urdfFile);
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  const CentroidalMpcRobotModel<scalar_t>& mpcModel = model.getMpcRobotModel();
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const std::shared_ptr<const StateManifold> manifold,
      stateManifoldOf(mpcModel));
  test_support::ScriptedPolicyMpcLink* absl_nullable link = nullptr;
  absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>>
      controllerOrStatus = CentroidalMpcMrtJointController::Create(
          description, model.getModelSettings(), mpcModel,
          test_support::ScriptedPolicyMpcLink::factory(&link),
          model.getPinocchioInterface(),
          jointPdGainsFileBeside(model.taskFile));
  ASSERT_TRUE(controllerOrStatus.ok()) << controllerOrStatus.status();
  ASSERT_NE(link, nullptr);
  CentroidalMpcMrtJointController& controller = **controllerOrStatus;
  robot::model::RobotState robotState(description);
  robotState.setPosBaseRtWorldEwrtWorld(vector3_t(0.0, 0.0, 0.75));
  for (size_t joint = 0; joint < description.getNumJoints(); ++joint)
    robotState.setJointPosition(joint, /*jointPosition=*/0.1);
  robotState.setContactFlag(/*index=*/0, /*contactFlag=*/true);
  robotState.setContactFlag(/*index=*/1, /*contactFlag=*/true);
  robot::model::RobotJointAction action(description);
  const std::vector<scalar_t> nominal(description.getNumJoints(), 0.1);
  scalar_t time = 0.0;
  runCycles(controller, nominal, robotState, action, time, "JOINT_PD", kCycles);
  runCycles(controller, nominal, robotState, action, time, "WB_MPC", kCycles);

  // Settle the time-dependent parts of the action (the entry, the contact
  // wrench gate's ramp) on a plan that holds the measured state.
  const Eigen::Index nu = static_cast<Eigen::Index>(mpcModel.getInputDim());
  const Eigen::Index ndx = static_cast<Eigen::Index>(manifold->getTangentDim());
  vector_t input = vector_t::Zero(nu);
  for (Eigen::Index i = 0; i < nu; ++i)
    input(i) = 0.2 * std::sin(1.3 * static_cast<scalar_t>(i));
  const vector_t measured = link->lastObservation().state;
  link->deliver(policyOf(std::make_unique<FeedforwardController>(
                             scalar_array_t{time - 1.0, time + 100.0},
                             vector_array_t{input, input}),
                         measured, input, time));
  runCycles(controller, nominal, robotState, action, time, "WB_MPC",
            10 * kCycles);
  ASSERT_FALSE(controller.isEnteringMpc());

  std::mt19937 generator(/*sd=*/37);
  std::normal_distribution<scalar_t> normal(0.0, 1.0);
  matrix_t gain(nu, ndx);
  for (Eigen::Index i = 0; i < gain.size(); ++i)
    gain.data()[i] = 20.0 * normal(generator);
  for (const scalar_t period : {0.002, 0.01, 0.017}) {
    SCOPED_TRACE(period);
    std::vector<std::vector<scalar_t>> actions;
    for (const scalar_t gainScale : {1.0, 0.0}) {
      // The plan starts at the next measurement, on the state the controller
      // measures (the robot does not move), and moves: the momentum, the base
      // and its attitude, and the joints (by less than the divergence bound).
      const scalar_t measurementTime = time + period;
      const vector_t now = link->lastObservation().state;
      vector_t later = now;
      for (Eigen::Index i = 0; i < later.size(); ++i)
        later(i) += 0.3 * std::cos(static_cast<scalar_t>(i));
      manifold->project(later);
      PrimalSolution plan;
      plan.timeTrajectory_ = {measurementTime, measurementTime + 1.0};
      plan.stateTrajectory_ = {now, later};
      plan.inputTrajectory_ = {input, input + vector_t::Constant(nu, 0.4)};
      plan.modeSchedule_ = ModeSchedule(
          /*eventTimesInput=*/{},
          /*modeSequenceInput=*/{static_cast<size_t>(ModeNumber::kStance)});
      const matrix_t scaledGain = gainScale * gain;
      plan.controllerPtr_ = std::make_unique<ManifoldLinearController>(
          plan.timeTrajectory_, plan.stateTrajectory_, plan.inputTrajectory_,
          matrix_array_t{scaledGain, scaledGain}, manifold);
      link->deliver(plan);
      controller.setNominalJointPositions(nominal);
      controller.setControlMode("WB_MPC");
      time = measurementTime;
      robotState.setTime(time);
      controller.computeJointControlAction(time, robotState, action);
      actions.push_back(actionValuesOf(description, action));
    }
    ASSERT_EQ(actions[0].size(), actions[1].size());
    for (size_t i = 0; i < actions[0].size(); ++i) {
      EXPECT_NEAR(actions[0][i], actions[1][i],
                  1.0e-9 * std::max(1.0, std::abs(actions[1][i])))
          << "entry " << i;
    }
  }
}

}  // namespace
}  // namespace ocs2::humanoid
