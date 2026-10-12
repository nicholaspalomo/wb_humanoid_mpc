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
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
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

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StateManifoldFactory.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/mrt/ControllerEvent.h"
#include "humanoid_common_mpc/mrt/ControllerEventSink.h"
#include "humanoid_common_mpc/mrt/JointActionAccess.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/ScriptedPolicyMpcLink.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "humanoid_wb_mpc/mrt/WBMpcMrtJointController.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"

/*
 * What the whole-body MRT joint controller does in WB_MPC with the policy in use, on the quaternion layout:
 *
 *   - a policy its realtime evaluator refuses - a LinearController, whose gains would act on the stored quaternion
 *     rather than on its tangent, or a ManifoldLinearController with ambient gains - is never handed to
 *     MRT_BASE::evaluatePolicy() (which allocates on this thread, logs, and aborts on the malformed one): the robot is
 *     held with the hold's action, the cycle allocates exactly what the hold's does, and one kPolicyRefused event says
 *     why; a policy it evaluates afterwards is executed;
 *   - at gimbal lock of the Euler angles (a pitch of exactly +-90 degrees, a yaw of pi) a manifold-linear policy gives a
 *     finite action, the same to round-off with the measured quaternion negated (the same rotation), and a finite
 *     observation and reset target.
 */

namespace ocs2::humanoid {
namespace {

std::string runfilePath(const std::string& relativePath) {
  const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR");
  const std::filesystem::path candidate =
      srcDir != nullptr ? std::filesystem::path(srcDir) / "_main" / relativePath : std::filesystem::path(relativePath);
  return candidate.string();
}

/** Counts the events of one code, as the realtime event log takes events: no allocation. */
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

/** The controller of the shipped whole-body G1 on a ScriptedPolicyMpcLink, and what the test feeds it. */
class PolicyExecution : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string taskFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto");
    const std::string urdfFile = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
    absl::StatusOr<ModelSettings> modelSettings = ModelSettings::Create(taskFile, urdfFile, "wb_mpc_", /*verbose=*/false);
    ASSERT_TRUE(modelSettings.ok()) << modelSettings.status();
    modelSettings_ = std::make_unique<ModelSettings>(*std::move(modelSettings));
    absl::StatusOr<PinocchioInterface> pinocchioInterface = loadCustomPinocchioInterface(taskFile, urdfFile, *modelSettings_);
    ASSERT_TRUE(pinocchioInterface.ok()) << pinocchioInterface.status();
    absl::StatusOr<robot::model::RobotDescription> description = robot::model::RobotDescription::Create(urdfFile);
    ASSERT_TRUE(description.ok()) << description.status();
    description_ = std::make_unique<robot::model::RobotDescription>(*std::move(description));
    model_ = std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*modelSettings_);
    absl::StatusOr<std::shared_ptr<const StateManifold>> manifold = stateManifoldOf(*model_);
    ASSERT_TRUE(manifold.ok()) << manifold.status();
    manifold_ = *std::move(manifold);
    absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>> controller = WBMpcMrtJointController::Create(
        *description_, *modelSettings_, test_support::ScriptedPolicyMpcLink::factory(&link_), *std::move(pinocchioInterface));
    ASSERT_TRUE(controller.ok()) << controller.status();
    controller_ = *std::move(controller);
    ASSERT_NE(link_, nullptr);
    state_ = std::make_unique<robot::model::RobotState>(*description_);
    state_->setRootPositionInWorldFrame(vector3_t(0.0, 0.0, 0.75));
    for (size_t joint = 0; joint < description_->getNumJoints(); ++joint) state_->setJointPosition(joint, /*jointPosition=*/0.1);
    action_ = std::make_unique<robot::model::RobotJointAction>(*description_);
    nominal_.assign(description_->getNumJoints(), 0.1);
  }

  /** `cycles` control cycles in `mode` at the robot state of the test, 2 ms apart. */
  void run(const char* absl_nonnull mode, int cycles) {
    for (int cycle = 0; cycle < cycles; ++cycle) {
      controller_->setNominalJointPositions(nominal_);
      controller_->setControlMode(mode);
      time_ += 0.002;
      state_->setTime(time_);
      controller_->computeJointControlAction(time_, *state_, *action_);
    }
  }

  /** The joint actions of the robot, flattened: q_des, qd_des, kp, kd and the feedforward of each joint. */
  std::vector<scalar_t> actionValues() const {
    std::vector<scalar_t> values;
    for (size_t joint = 0; joint < description_->getNumJoints(); ++joint) {
      const robot::model::JointAction& action = jointActionUnchecked(*action_, joint);
      values.insert(values.end(), {action.q_des, action.qd_des, action.kp, action.kd, action.feed_forward_effort});
    }
    return values;
  }

  /** How many joints the action sends to `position`. */
  int jointsCommandedTo(scalar_t position) const {
    int count = 0;
    for (size_t joint = 0; joint < description_->getNumJoints(); ++joint) {
      if (jointActionUnchecked(*action_, joint).q_des == position) ++count;
    }
    return count;
  }

  /** A state of the MPC layout: the base at 0.75 m with the orientation `xi` (x, y, z, w), the joints at `joints`. */
  vector_t mpcState(const vector4_t& xi, scalar_t joints) const {
    vector_t state = vector_t::Zero(static_cast<Eigen::Index>(model_->getStateDim()));
    state(2) = 0.75;
    state.segment<4>(static_cast<Eigen::Index>(model_->getBaseOrientationStartIndex())) = xi;
    // The joint angles follow the base pose [p_W (3), xi (4)].
    state.segment(static_cast<Eigen::Index>(model_->getBaseOrientationStartIndex()) + 4, static_cast<Eigen::Index>(model_->getJointDim()))
        .setConstant(joints);
    return state;
  }

  /** A policy of `controller` with two nodes around the test's time, the state and the input held. */
  PrimalSolution policyOf(std::unique_ptr<ControllerBase> controller, const vector_t& state, const vector_t& input) const {
    PrimalSolution policy;
    policy.timeTrajectory_ = {time_ - 1.0, time_ + 100.0};
    policy.stateTrajectory_ = {state, state};
    policy.inputTrajectory_ = {input, input};
    policy.modeSchedule_ = ModeSchedule(/*eventTimesInput=*/{}, /*modeSequenceInput=*/{static_cast<size_t>(ModeNumber::kStance)});
    policy.controllerPtr_ = std::move(controller);
    return policy;
  }

  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<robot::model::RobotDescription> description_;
  std::unique_ptr<WBAccelMpcRobotModel<scalar_t>> model_;
  std::shared_ptr<const StateManifold> manifold_;
  test_support::ScriptedPolicyMpcLink* absl_nullable link_ = nullptr;
  std::unique_ptr<WBMpcMrtJointController> controller_;
  std::unique_ptr<robot::model::RobotState> state_;
  std::unique_ptr<robot::model::RobotJointAction> action_;
  std::vector<scalar_t> nominal_;
  scalar_t time_ = 0.0;
};

constexpr int kCycles = 20;
/** 0.6 s of 2 ms cycles. */
constexpr int kSettlingCycles = 300;

TEST_F(PolicyExecution, ARefusedPolicyHoldsTheRobotWithoutMrtBaseAndIsReportedOnce) {
  CodeCountingSink sink(ControllerEventCode::kPolicyRefused);
  controller_->setEventSink(&sink);
  run("JOINT_PD", kCycles);
  run("WB_MPC", kCycles);  // the hold of the entry, until a policy solved after the reset is in use
  const std::vector<scalar_t> hold = actionValues();
  size_t before = robot::realtime::heapAllocationCountOnThisThread();
  run("WB_MPC", kCycles);
  const size_t holdAllocations = robot::realtime::heapAllocationCountOnThisThread() - before;
  EXPECT_EQ(actionValues(), hold);

  const Eigen::Index nx = static_cast<Eigen::Index>(model_->getStateDim());
  const Eigen::Index nu = static_cast<Eigen::Index>(model_->getInputDim());
  const vector_t state = mpcState(vector4_t(0.0, 0.0, 0.0, 1.0), /*joints=*/0.15);
  const vector_t input = vector_t::Zero(nu);
  const scalar_array_t times = {time_ - 1.0, time_ + 100.0};
  struct RefusedCase {
    const char* absl_nonnull name;
    PrimalSolution policy;
    PolicyRefusal refusal;
  };
  std::vector<RefusedCase> refused;
  refused.push_back(
      RefusedCase{.name = "a LinearController on the quaternion state",
                  .policy = policyOf(std::make_unique<LinearController>(times, vector_array_t{input, input},
                                                                        matrix_array_t{matrix_t::Ones(nu, nx), matrix_t::Ones(nu, nx)}),
                                     state, input),
                  .refusal = PolicyRefusal::kUnsupportedController});
  refused.push_back(RefusedCase{.name = "a ManifoldLinearController with ambient gains",
                                .policy = policyOf(std::make_unique<ManifoldLinearController>(
                                                       times, vector_array_t{state, state}, vector_array_t{input, input},
                                                       matrix_array_t{matrix_t::Ones(nu, nx), matrix_t::Ones(nu, nx)}, manifold_),
                                                   state, input),
                                .refusal = PolicyRefusal::kMismatchedPolicy});
  int expectedEvents = 0;
  for (const RefusedCase& refusedCase : refused) {
    SCOPED_TRACE(refusedCase.name);
    link_->deliver(refusedCase.policy);
    run("WB_MPC", /*cycles=*/1);  // takes the policy into use
    before = robot::realtime::heapAllocationCountOnThisThread();
    run("WB_MPC", kCycles);
    EXPECT_EQ(robot::realtime::heapAllocationCountOnThisThread() - before, holdAllocations)
        << "a refused policy's cycle allocates what the hold's does: MRT_BASE::evaluatePolicy() allocates 2 to 3 per call";
    EXPECT_EQ(actionValues(), hold) << "the robot is held";
    ++expectedEvents;
    EXPECT_EQ(sink.posted.load(), expectedEvents) << "one report per run of refused cycles";
    EXPECT_EQ(sink.lastValue.load(), static_cast<double>(refusedCase.refusal));

    // A policy the evaluator takes is executed: the MPC's joint targets, and no further report.
    link_->deliver(policyOf(std::make_unique<FeedforwardController>(times, vector_array_t{input, input}), state, input));
    run("WB_MPC", kCycles);
    EXPECT_NE(actionValues(), hold);
    EXPECT_EQ(jointsCommandedTo(0.15), static_cast<int>(model_->getJointDim())) << "every MPC joint at the policy's angle";
    EXPECT_EQ(sink.posted.load(), expectedEvents);
  }
}

TEST_F(PolicyExecution, AtGimbalLockAManifoldPolicyGivesTheSameFiniteActionForEitherSignOfTheQuaternion) {
  run("JOINT_PD", kCycles);
  // The entry into WB_MPC requests the reset the first delivery serves.
  run("WB_MPC", kCycles);
  const Eigen::Index nu = static_cast<Eigen::Index>(model_->getInputDim());
  const Eigen::Index ndx = static_cast<Eigen::Index>(manifold_->getTangentDim());
  for (const scalar_t pitch : {0.5 * M_PI, -0.5 * M_PI}) {
    SCOPED_TRACE(pitch);
    // R = R_z(pi) R_y(+-pi / 2): the Euler angles' gimbal lock, at the yaw cut.
    const quaternion_t orientation(Eigen::AngleAxisd(M_PI, vector3_t::UnitZ()) * Eigen::AngleAxisd(pitch, vector3_t::UnitY()));
    std::mt19937 generator(/*sd=*/7);
    std::normal_distribution<scalar_t> normal(0.0, 1.0);
    matrix_t gain(nu, ndx);
    for (Eigen::Index i = 0; i < gain.size(); ++i) gain.data()[i] = 0.1 * normal(generator);
    // The plan's anchor turned a little from the measured attitude, so that the feedback acts on a rotation.
    const quaternion_t anchorOrientation = orientation * quaternion_t(Eigen::AngleAxisd(0.05, vector3_t(0.6, 0.0, 0.8)));
    const vector_t anchor =
        mpcState(vector4_t(anchorOrientation.x(), anchorOrientation.y(), anchorOrientation.z(), anchorOrientation.w()), /*joints=*/0.12);
    const vector_t input = vector_t::Zero(nu);
    const scalar_array_t times = {time_ - 1.0, time_ + 100.0};
    std::vector<std::vector<scalar_t>> actions;
    // Each sign long enough for the time-dependent parts of the action (the contact wrench gate's ramp) to settle; the
    // first sign again at the end shows that they did.
    for (const scalar_t sign : {1.0, -1.0, 1.0}) {
      quaternion_t measured;
      measured.coeffs() = sign * orientation.coeffs();
      state_->setRootRotationLocalToWorldFrame(measured);
      // The policy's own anchors carry the plan's sign; the measured quaternion's sign is the one that changes.
      link_->deliver(
          policyOf(std::make_unique<ManifoldLinearController>(times, vector_array_t{anchor, anchor}, vector_array_t{input, input},
                                                              matrix_array_t{gain, gain}, manifold_),
                   anchor, input));
      run("WB_MPC", kSettlingCycles);
      const std::vector<scalar_t> values = actionValues();
      for (const scalar_t value : values) ASSERT_TRUE(std::isfinite(value));
      actions.push_back(values);
      const SystemObservation& observation = link_->lastObservation();
      ASSERT_TRUE(observation.state.allFinite());
      const TargetTrajectories target = link_->resetTargetOf(observation);
      for (const vector_t& knot : target.stateTrajectory) ASSERT_TRUE(knot.allFinite());
      // The reset target's heading is the measured base's heading, pi, up to the double cover.
      const vector4_t targetXi =
          target.stateTrajectory.front().segment<4>(static_cast<Eigen::Index>(model_->getBaseOrientationStartIndex()));
      const scalar_t heading = 2.0 * std::atan2(targetXi(2), targetXi(3));
      EXPECT_NEAR(std::abs(std::remainder(heading, 2.0 * M_PI)), M_PI, 1.0e-9);
    }
    ASSERT_EQ(actions.size(), 3);
    ASSERT_EQ(actions[1].size(), actions[0].size());
    ASSERT_EQ(actions[2].size(), actions[0].size());
    // The same rotation: the action at -xi is the action at xi to round-off, and so is the first sign's again, which
    // shows that the time-dependent parts had settled (units in the last place were measured in the feedforward: the
    // plan state is interpolated at another time between its two equal anchors).
    for (size_t i = 0; i < actions[0].size(); ++i) {
      const scalar_t tolerance = 1.0e-12 * std::max(1.0, std::abs(actions[0][i]));
      EXPECT_NEAR(actions[2][i], actions[0][i], tolerance) << "the action had not settled: entry " << i;
      EXPECT_NEAR(actions[1][i], actions[0][i], tolerance) << "the action at -xi, entry " << i;
    }
  }
}

}  // namespace
}  // namespace ocs2::humanoid
