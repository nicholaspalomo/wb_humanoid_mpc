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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_mpc/MPC_Settings.h"
#include "ocs2_mpc_test/ScriptedMpc.h"

#include "humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h"
#include "humanoid_centroidal_mpc_test/CentroidalTestingModelInterface.h"
#include "humanoid_common_mpc/common/BasisInputsModelDecorator.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotStateContactEstimator.h"

namespace ocs2::humanoid {
namespace {

/** The action of joint `index`, which every joint of the description holds: a test failure and a zero action if not. */
const ::robot::model::JointAction& actionOf(const ::robot::model::RobotJointAction& action, size_t index) {
  const std::optional<::robot::model::JointAction>& entry = action.at(index);
  if (!entry.has_value()) {
    ADD_FAILURE() << "no action for joint " << index;
    static const ::robot::model::JointAction kNoAction;
    return kNoAction;
  }
  return *entry;
}

/** A URDF of two revolute joints that no MPC model of the test robots names. */
constexpr char kTwoJointUrdf[] = R"(<?xml version="1.0"?>
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

/** The robot description of kTwoJointUrdf, written to the test's temporary directory. */
absl::StatusOr<::robot::model::RobotDescription> twoJointDescription() {
  const std::filesystem::path path = std::filesystem::path(::testing::TempDir()) / "two_joint_robot.urdf";
  std::ofstream(path) << kTwoJointUrdf;
  return ::robot::model::RobotDescription::Create(path.string());
}

}  // namespace

// A contact estimator that answers with a fixed contact state, whatever the RobotState says.
class FixedContactEstimator final : public ::robot::model::ContactEstimator {
 public:
  explicit FixedContactEstimator(std::vector<bool> flags) : flags_(std::move(flags)) {}
  void estimateContactFlags(const ::robot::model::RobotState& /*robotState*/, std::vector<bool>& flags) override {
    ++calls;
    flags.assign(flags_.begin(), flags_.end());
  }
  std::string getName() const override { return "FixedContactEstimator"; }
  void set(std::vector<bool> flags) { flags_ = std::move(flags); }
  size_t calls = 0;

 private:
  std::vector<bool> flags_;
};

class CentroidalMpcMrtJointControllerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tempPdGainsFile_ = std::filesystem::temp_directory_path() / "test_pd_gains.textproto";

    // A PD gains file (humanoid_mpc_config.JointPdGainsFile) that names no joint: every joint has the controller's
    // own gains. (A joint the model does not have would make it another robot's file, which the controller refuses.)
    std::ofstream ofs(tempPdGainsFile_);
    ofs << "# The controller's own gains for every joint.\n";
    ofs.close();
  }

  void TearDown() override {
    if (std::filesystem::exists(tempPdGainsFile_)) {
      std::filesystem::remove(tempPdGainsFile_);
    }
  }

  CentroidalTestingModelInterface testingModelInterface_;
  std::filesystem::path tempPdGainsFile_;
  // The MPC of the controllers below: a scripted solver (ocs2_mpc_test/ScriptedMpc.h), which solves only once a
  // controller's link is started (startMpcThread()). Only testAStartedControllerRunsAgainstTheScriptedMpc starts one; in
  // every other test no policy ever arrives.
  mpc_test::ScriptedMpc scriptedMpc_{mpc::Settings{}, testingModelInterface_.getMpcRobotModel().getInputDim()};
};

TEST_F(CentroidalMpcMrtJointControllerTest, testPdGainsHotReloading) {
  absl::StatusOr<::robot::model::RobotDescription> robotDescOrStatus =
      ::robot::model::RobotDescription::Create(testingModelInterface_.urdfFile);
  ASSERT_TRUE(robotDescOrStatus.ok()) << robotDescOrStatus.status();
  ::robot::model::RobotDescription& robotDesc = *robotDescOrStatus;
  ::robot::model::RobotState robotState(robotDesc);
  ::robot::model::RobotJointAction jointAction(robotDesc);

  // Create controller
  absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>> controllerOrStatus = CentroidalMpcMrtJointController::Create(
      robotDesc, testingModelInterface_.getModelSettings(), testingModelInterface_.getMpcRobotModel(), scriptedMpc_,
      testingModelInterface_.getPinocchioInterface(), /*mpcDesiredFrequency=*/400.0, tempPdGainsFile_.string());
  ASSERT_TRUE(controllerOrStatus.ok()) << controllerOrStatus.status();
  CentroidalMpcMrtJointController& controller = **controllerOrStatus;

  controller.setControlMode("JOINT_PD");

  // Trigger first compute loop
  EXPECT_NO_THROW({ controller.computeJointControlAction(/*time=*/0.01, robotState, jointAction); });

  // Modify file
  std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Ensure timestamp difference
  std::ofstream ofs(tempPdGainsFile_, std::ios::app);
  ofs << "joint_gains { joint: \"right_knee_joint\" kp: 200.0 kd: 20.0 torque_limit: 150.0 }\n";
  ofs.close();

  // Trigger again: the file watcher runs on a non-realtime thread now (pollPdGainsFile()), not inside the control cycle.
  EXPECT_NO_THROW({ controller.pollPdGainsFile(); });
  EXPECT_NO_THROW({ controller.computeJointControlAction(/*time=*/0.02, robotState, jointAction); });
}

// SAFETY must be a damped joint PD about the posture held at mode entry whose torques decay smoothly to zero. Before
// this test the mode had no branch at all in the controller: SimFsmBridge enabled the torques and forwarded the name,
// and the mode then fell through to the active MPC path, so "SAFETY" ran the solver at full authority.
TEST_F(CentroidalMpcMrtJointControllerTest, testSafetyModeDecaysADampedPdToZeroTorque) {
  absl::StatusOr<::robot::model::RobotDescription> robotDescOrStatus =
      ::robot::model::RobotDescription::Create(testingModelInterface_.urdfFile);
  ASSERT_TRUE(robotDescOrStatus.ok()) << robotDescOrStatus.status();
  ::robot::model::RobotDescription& robotDesc = *robotDescOrStatus;
  ::robot::model::RobotState robotState(robotDesc);

  const scalar_t timeConstant = 0.5;
  absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>> controllerOrStatus = CentroidalMpcMrtJointController::Create(
      robotDesc, testingModelInterface_.getModelSettings(), testingModelInterface_.getMpcRobotModel(), scriptedMpc_,
      testingModelInterface_.getPinocchioInterface(), /*mpcDesiredFrequency=*/400.0, tempPdGainsFile_.string());
  ASSERT_TRUE(controllerOrStatus.ok()) << controllerOrStatus.status();
  CentroidalMpcMrtJointController& controller = **controllerOrStatus;
  controller.setSafetyDecayTimeConstant(timeConstant);

  // The gains SAFETY decays from are the ones JOINT_PD commands, so take that mode's action as the reference.
  controller.setControlMode("JOINT_PD");
  ::robot::model::RobotJointAction reference(robotDesc);
  robotState.setTime(0.0);
  controller.computeJointControlAction(/*time=*/0.0, robotState, reference);

  // Move the joints off the nominal posture so that "holds the posture at entry" is distinguishable from "returns to
  // the nominal posture", which is the behavior that would make a safety stop a lunge.
  for (size_t index = 0; index < robotDesc.getNumJoints(); ++index) {
    robotState.setJointPosition(index, robotState.getJointPosition(index) + 0.3);
  }

  controller.setControlMode("SAFETY");
  ::robot::model::RobotJointAction atEntry(robotDesc);
  robotState.setTime(1.0);
  controller.computeJointControlAction(/*time=*/1.0, robotState, atEntry);

  size_t checkedJoints = 0;
  for (size_t index = 0; index < robotDesc.getNumJoints(); ++index) {
    if (!atEntry.at(index).has_value() || !reference.at(index).has_value()) continue;
    const ::robot::model::JointAction& action = actionOf(atEntry, index);
    // Full authority at the instant of entry: the torque must not jump when the mode is switched.
    EXPECT_NEAR(action.kp, actionOf(reference, index).kp, 1.0e-9);
    EXPECT_NEAR(action.kd, actionOf(reference, index).kd, 1.0e-9);
    // The posture held is the measured one, and nothing model-derived is fed forward: SAFETY is what runs when the
    // model or the solver is the problem.
    EXPECT_NEAR(action.q_des, robotState.getJointPosition(index), 1.0e-9);
    EXPECT_DOUBLE_EQ(action.qd_des, 0.0);
    EXPECT_DOUBLE_EQ(action.feed_forward_effort, 0.0);
    ++checkedJoints;
  }
  ASSERT_GT(checkedJoints, 0u) << "the fixture produced no actuated joints, so nothing was verified";

  // The gains decay monotonically and reach exactly zero within four time constants.
  scalar_t previousKp = std::numeric_limits<scalar_t>::max();
  for (int step = 0; step <= 20; ++step) {
    const scalar_t time = 1.0 + 0.2 * step * timeConstant;
    robotState.setTime(time);
    ::robot::model::RobotJointAction action(robotDesc);
    controller.computeJointControlAction(time, robotState, action);
    scalar_t maxKp = 0.0;
    for (size_t index = 0; index < robotDesc.getNumJoints(); ++index) {
      if (!action.at(index).has_value()) continue;
      maxKp = std::max(maxKp, actionOf(action, index).kp);
      EXPECT_DOUBLE_EQ(actionOf(action, index).feed_forward_effort, 0.0);
    }
    EXPECT_LE(maxKp, previousKp + 1.0e-9) << "the SAFETY gains must never climb back up at t = " << time;
    previousKp = maxKp;
  }

  robotState.setTime(1.0 + 4.0 * timeConstant);
  ::robot::model::RobotJointAction afterDecay(robotDesc);
  controller.computeJointControlAction(1.0 + 4.0 * timeConstant, robotState, afterDecay);
  EXPECT_TRUE(controller.isSafetyDecayComplete());
  for (size_t index = 0; index < robotDesc.getNumJoints(); ++index) {
    if (!afterDecay.at(index).has_value()) continue;
    const ::robot::model::JointAction& action = actionOf(afterDecay, index);
    EXPECT_DOUBLE_EQ(action.kp, 0.0);
    EXPECT_DOUBLE_EQ(action.kd, 0.0);
    EXPECT_DOUBLE_EQ(action.feed_forward_effort, 0.0);
    EXPECT_DOUBLE_EQ(action.getTotalFeedbackTorque(robotState.getJointPosition(index), /*qd=*/5.0), 0.0);
  }
}

// Entering WB_MPC from JOINT_PD resets the MPC, and no policy solved before that reset may reach the robot: the controller
// keeps the JOINT_PD action, field by field, until a policy solved after it is in use - whatever the entry blend time,
// which only shapes the ramp that follows. It used to switch at once when the blend time was 0, executing the policy
// solved before the entry (or, before the first policy, a weight-compensating stand-in), for as long as the solver took.
TEST_F(CentroidalMpcMrtJointControllerTest, testEntryHoldsThePreviousModeUntilAPostResetPolicyIsActive) {
  absl::StatusOr<::robot::model::RobotDescription> robotDescOrStatus =
      ::robot::model::RobotDescription::Create(testingModelInterface_.urdfFile);
  ASSERT_TRUE(robotDescOrStatus.ok()) << robotDescOrStatus.status();
  ::robot::model::RobotDescription& robotDesc = *robotDescOrStatus;
  ::robot::model::RobotState robotState(robotDesc);

  using EntryResult = std::tuple<bool, ::robot::model::RobotJointAction, ::robot::model::RobotJointAction>;
  const std::function<EntryResult(scalar_t)> run = [&](scalar_t blendTime) {
    const std::unique_ptr<CentroidalMpcMrtJointController> controllerPtr =
        CentroidalMpcMrtJointController::Create(
            robotDesc, testingModelInterface_.getModelSettings(), testingModelInterface_.getMpcRobotModel(), scriptedMpc_,
            testingModelInterface_.getPinocchioInterface(), /*mpcDesiredFrequency=*/400.0, tempPdGainsFile_.string())
            .value();
    CentroidalMpcMrtJointController& controller = *controllerPtr;
    controller.setMpcEntryBlendTime(blendTime);
    controller.setControlMode("JOINT_PD");
    ::robot::model::RobotJointAction held(robotDesc);
    controller.computeJointControlAction(/*time=*/0.01, robotState, held);
    controller.setControlMode("WB_MPC");  // requests a reset; the link is never started, so no post-reset policy arrives
    ::robot::model::RobotJointAction entered(robotDesc);
    controller.computeJointControlAction(/*time=*/0.02, robotState, entered);
    controller.computeJointControlAction(/*time=*/0.03, robotState, entered);
    return EntryResult(controller.isEnteringMpc(), held, entered);
  };

  for (const scalar_t blendTime : {0.0, 0.3}) {
    const EntryResult result = run(blendTime);
    const bool entering = std::get<0>(result);
    const ::robot::model::RobotJointAction& held = std::get<1>(result);
    const ::robot::model::RobotJointAction& entered = std::get<2>(result);
    EXPECT_TRUE(entering) << "blend time " << blendTime;
    for (size_t index = 0; index < robotDesc.getNumJoints(); ++index) {
      if (!held.at(index).has_value()) continue;
      EXPECT_DOUBLE_EQ(actionOf(entered, index).q_des, actionOf(held, index).q_des) << "joint " << index << ", blend time " << blendTime;
      EXPECT_DOUBLE_EQ(actionOf(entered, index).kp, actionOf(held, index).kp) << "joint " << index << ", blend time " << blendTime;
      EXPECT_DOUBLE_EQ(actionOf(entered, index).kd, actionOf(held, index).kd) << "joint " << index << ", blend time " << blendTime;
      EXPECT_DOUBLE_EQ(actionOf(entered, index).feed_forward_effort, actionOf(held, index).feed_forward_effort)
          << "joint " << index << ", blend time " << blendTime;
    }
  }
}

// With basis-vector inputs the OCP input is [lambda_left, lambda_right, joint velocities] and the input-only accessors of the
// effective model return the LOCAL contact-frame wrench B*lambda. The controller must (a) size its observation input to the
// basis layout and (b) hand WORLD-frame wrenches to the inverse dynamics (which uses LOCAL_WORLD_ALIGNED Jacobians). Pitching
// the ankles makes the local contact frames differ from the world frame, so a local-frame wrench would yield different torques.
TEST_F(CentroidalMpcMrtJointControllerTest, testBasisVectorInputsUseWorldFrameWrenchesForFeedforward) {
  absl::StatusOr<::robot::model::RobotDescription> robotDescOrStatus =
      ::robot::model::RobotDescription::Create(testingModelInterface_.urdfFile);
  ASSERT_TRUE(robotDescOrStatus.ok()) << robotDescOrStatus.status();
  ::robot::model::RobotDescription& robotDesc = *robotDescOrStatus;
  ::robot::model::RobotState robotState(robotDesc);
  ::robot::model::RobotJointAction jointAction(robotDesc);

  // Pitch both feet so that the local contact frames are not aligned with the world frame.
  constexpr scalar_t kAnklePitch = 0.3;
  robotState.setJointPosition(robotDesc.getJointIndex("left_ankle_pitch_joint"), kAnklePitch);
  robotState.setJointPosition(robotDesc.getJointIndex("right_ankle_pitch_joint"), kAnklePitch);

  // Basis decorator built the same way CentroidalMpcInterface builds it in basis mode.
  const ContactWrenchConeConstraint::Config coneConfig(4, 0.7, 0.05, 5.0, 0.0);
  const PolygonBounds footBounds(-0.1, 0.1, -0.05, 0.05);
  absl::StatusOr<ContactWrenchConeBasisMatrix> leftBasis = ContactWrenchConeBasisMatrix::Create(
      coneConfig, ContactRectangle(footBounds, ContactCenterPoint("foot_l_contact", "left_ankle_roll_joint", vector3_t::Zero())));
  ASSERT_TRUE(leftBasis.ok()) << leftBasis.status();
  absl::StatusOr<ContactWrenchConeBasisMatrix> rightBasis = ContactWrenchConeBasisMatrix::Create(
      coneConfig, ContactRectangle(footBounds, ContactCenterPoint("foot_r_contact", "right_ankle_roll_joint", vector3_t::Zero())));
  ASSERT_TRUE(rightBasis.ok()) << rightBasis.status();
  const std::array<ContactWrenchConeBasisMatrix, kNumContacts> basisMatrices = {*std::move(leftBasis), *std::move(rightBasis)};
  BasisInputsModelDecorator<scalar_t> basisModel(
      std::unique_ptr<MpcRobotModelBase<scalar_t>>(testingModelInterface_.getMpcRobotModel().clone()), basisMatrices,
      testingModelInterface_.getPinocchioInterface());
  ASSERT_NE(basisModel.getInputDim(), testingModelInterface_.getMpcRobotModel().getInputDim());

  // A generous torque limit keeps the controller's clamp from masking the torque comparison below.
  const std::filesystem::path gainsFile = std::filesystem::temp_directory_path() / "test_pd_gains_basis.textproto";
  {
    std::ofstream ofs(gainsFile);
    ofs << "default_gains { kp: 100.0 kd: 10.0 torque_limit: 1000000.0 }\n";
  }

  absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>> controllerOrStatus = CentroidalMpcMrtJointController::Create(
      robotDesc, testingModelInterface_.getModelSettings(), testingModelInterface_.getMpcRobotModel(), scriptedMpc_,
      testingModelInterface_.getPinocchioInterface(), /*mpcDesiredFrequency=*/400.0, gainsFile.string(), &basisModel);
  ASSERT_TRUE(controllerOrStatus.ok()) << controllerOrStatus.status();
  CentroidalMpcMrtJointController& controller = **controllerOrStatus;

  // The default mode is WB_MPC and the link is never started, so no policy arrives and the controller takes the
  // weight-compensating feed-forward branch.
  ASSERT_NO_THROW(controller.computeJointControlAction(/*time=*/0.01, robotState, jointAction));
  std::filesystem::remove(gainsFile);

  // (a) The observation input follows the basis layout: a zero contact block with the joint velocities at the tail.
  const SystemObservation& observation = controller.getCurrentObservation();
  ASSERT_EQ(observation.input.size(), static_cast<Eigen::Index>(basisModel.getInputDim()));
  EXPECT_TRUE(observation.input.head(basisModel.getInputDim() - basisModel.getJointDim()).isZero());
  EXPECT_TRUE(basisModel.getJointVelocities(observation.state, observation.input).isZero());

  // (b) Reference torques: the same state-aware weight-compensating input read back in the world frame and projected with the
  // same inverse dynamics the controller uses.
  PinocchioInterface pinocchioInterface = testingModelInterface_.getPinocchioInterface();
  const vector_t weightInput = weightCompensatingInput(pinocchioInterface, {true, true}, basisModel, observation.state);
  const std::array<vector6_t, 2> worldWrenches{basisModel.getContactWrenchInWorldFrame(observation.state, weightInput, /*contactIndex=*/0),
                                               basisModel.getContactWrenchInWorldFrame(observation.state, weightInput, /*contactIndex=*/1)};
  const std::array<vector6_t, 2> localWrenches{basisModel.getContactWrench(weightInput, /*contactIndex=*/0),
                                               basisModel.getContactWrench(weightInput, /*contactIndex=*/1)};

  const vector_t q = basisModel.getGeneralizedCoordinates(observation.state);
  const vector_t qd = basisModel.getGeneralizedVelocities(observation.state, observation.input);
  const vector_t qddZero = vector_t::Zero(basisModel.getJointDim());
  const vector_t expectedTorques = computeBaseHeldJointTorques<scalar_t>(q, qd, qddZero, worldWrenches, pinocchioInterface);
  const vector_t localFrameTorques = computeBaseHeldJointTorques<scalar_t>(q, qd, qddZero, localWrenches, pinocchioInterface);

  // Sanity: with pitched feet the local- and world-frame wrenches (and hence torques) differ, so the check discriminates.
  ASSERT_GT((worldWrenches[0] - localWrenches[0]).norm(), 1.0);
  ASSERT_GT((expectedTorques - localFrameTorques).norm(), 1.0);

  const std::vector<std::string>& mpcJointNames = testingModelInterface_.getModelSettings().mpcModelJointNames;
  const std::vector<::robot::joint_index_t> mpcJointIndices = robotDesc.getJointIndices(mpcJointNames);
  ASSERT_EQ(static_cast<Eigen::Index>(mpcJointIndices.size()), expectedTorques.size());
  for (size_t i = 0; i < mpcJointIndices.size(); ++i) {
    const scalar_t feedforward = actionOf(jointAction, mpcJointIndices[i]).feed_forward_effort;
    EXPECT_TRUE(std::isfinite(feedforward)) << "joint " << mpcJointNames[i];
    EXPECT_NEAR(feedforward, expectedTorques[i], 1.0e-6) << "joint " << mpcJointNames[i];
  }
}

// The measured contact state of the controller comes from its contact estimator: the observation mode handed to the
// MPC follows the estimator, not the contact flags of the RobotState. Without an estimator the RobotState flags are used
// (RobotStateContactEstimator), which is the historical behavior.
TEST_F(CentroidalMpcMrtJointControllerTest, testObservationModeFollowsTheContactEstimator) {
  absl::StatusOr<::robot::model::RobotDescription> robotDescOrStatus =
      ::robot::model::RobotDescription::Create(testingModelInterface_.urdfFile);
  ASSERT_TRUE(robotDescOrStatus.ok()) << robotDescOrStatus.status();
  ::robot::model::RobotDescription& robotDesc = *robotDescOrStatus;
  ::robot::model::RobotState robotState(robotDesc);
  ::robot::model::RobotJointAction jointAction(robotDesc);
  absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>> controllerOrStatus = CentroidalMpcMrtJointController::Create(
      robotDesc, testingModelInterface_.getModelSettings(), testingModelInterface_.getMpcRobotModel(), scriptedMpc_,
      testingModelInterface_.getPinocchioInterface(), /*mpcDesiredFrequency=*/400.0, tempPdGainsFile_.string());
  ASSERT_TRUE(controllerOrStatus.ok()) << controllerOrStatus.status();
  CentroidalMpcMrtJointController& controller = **controllerOrStatus;
  controller.setControlMode("JOINT_PD");

  // Default: the RobotState flags.
  EXPECT_EQ(controller.getContactEstimator().getName(), "RobotStateContactEstimator");
  robotState.setContactFlag(/*index=*/1, /*contactFlag=*/false);
  controller.computeJointControlAction(/*time=*/0.01, robotState, jointAction);
  EXPECT_EQ(controller.getCurrentObservation().mode, stanceLeg2ModeNumber({true, false}));
  EXPECT_EQ(controller.getMeasuredContactFlags(), (contact_flag_t{true, false}));

  // An injected estimator overrides whatever the RobotState carries, and is asked once per control cycle.
  auto estimator = std::make_shared<FixedContactEstimator>(std::vector<bool>{false, true});
  controller.setContactEstimator(estimator);
  EXPECT_EQ(controller.getContactEstimator().getName(), "FixedContactEstimator");
  controller.computeJointControlAction(/*time=*/0.02, robotState, jointAction);
  EXPECT_EQ(estimator->calls, 1u);
  EXPECT_EQ(controller.getCurrentObservation().mode, stanceLeg2ModeNumber({false, true}));
  EXPECT_EQ(controller.getMeasuredContactFlags(), (contact_flag_t{false, true}));
  estimator->set({true, true});
  controller.computeJointControlAction(/*time=*/0.03, robotState, jointAction);
  EXPECT_EQ(estimator->calls, 2u);
  EXPECT_EQ(controller.getCurrentObservation().mode, stanceLeg2ModeNumber({true, true}));

  // An estimator that reports the wrong number of contact points is a configuration error, not a silent mode: the
  // estimate is refused, counted and reported once as a warning through the event sink (kContactEstimateRefused), and
  // the measured contact state stays that of the cycle before.
  estimator->set({true});
  controller.computeJointControlAction(/*time=*/0.04, robotState, jointAction);
  EXPECT_EQ(controller.getNumRefusedContactEstimates(), 1U);
  EXPECT_EQ(controller.getMeasuredContactFlags(), (contact_flag_t{true, true})) << "a refused estimate changed the contact state";

  // A null pointer restores the default.
  controller.setContactEstimator(nullptr);
  EXPECT_EQ(controller.getContactEstimator().getName(), "RobotStateContactEstimator");
  controller.computeJointControlAction(/*time=*/0.05, robotState, jointAction);
  EXPECT_EQ(controller.getCurrentObservation().mode, stanceLeg2ModeNumber({true, false}));
}

// The contact wrenches the inverse dynamics projects are gated by the measured contact state, not by the plan. Before a
// policy arrives the controller compensates the weight through the feet measured in contact: with the right foot reported
// in the air the whole weight goes through the left foot, and the feedforward torques match the inverse dynamics of that
// single-support wrench (a two-foot distribution would give different torques).
TEST_F(CentroidalMpcMrtJointControllerTest, testFeedforwardWrenchesFollowTheMeasuredContactState) {
  absl::StatusOr<::robot::model::RobotDescription> robotDescOrStatus =
      ::robot::model::RobotDescription::Create(testingModelInterface_.urdfFile);
  ASSERT_TRUE(robotDescOrStatus.ok()) << robotDescOrStatus.status();
  ::robot::model::RobotDescription& robotDesc = *robotDescOrStatus;
  ::robot::model::RobotState robotState(robotDesc);

  // A generous torque limit keeps the controller's clamp from masking the torque comparison below.
  const std::filesystem::path gainsFile = std::filesystem::temp_directory_path() / "test_pd_gains_measured_contacts.textproto";
  {
    std::ofstream ofs(gainsFile);
    ofs << "default_gains { kp: 100.0 kd: 10.0 torque_limit: 1000000.0 }\n";
  }
  absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>> controllerOrStatus = CentroidalMpcMrtJointController::Create(
      robotDesc, testingModelInterface_.getModelSettings(), testingModelInterface_.getMpcRobotModel(), scriptedMpc_,
      testingModelInterface_.getPinocchioInterface(), /*mpcDesiredFrequency=*/400.0, gainsFile.string());
  ASSERT_TRUE(controllerOrStatus.ok()) << controllerOrStatus.status();
  CentroidalMpcMrtJointController& controller = **controllerOrStatus;
  std::filesystem::remove(gainsFile);
  auto estimator = std::make_shared<FixedContactEstimator>(std::vector<bool>{true, true});
  controller.setContactEstimator(estimator);

  const std::function<::robot::model::RobotJointAction(const std::vector<bool>&)> feedforward = [&](const std::vector<bool>& measured) {
    estimator->set(measured);
    ::robot::model::RobotJointAction action(robotDesc);
    controller.computeJointControlAction(/*time=*/0.01, robotState, action);  // WB_MPC mode, no policy: weight-compensating branch
    return action;
  };
  const ::robot::model::RobotJointAction bothFeet = feedforward({true, true});
  const ::robot::model::RobotJointAction leftFootOnly = feedforward({true, false});

  // Reference: the inverse dynamics of the controller with the measured single-support wrench.
  std::unique_ptr<MpcRobotModelBase<scalar_t>> modelPtr(testingModelInterface_.getMpcRobotModel().clone());
  MpcRobotModelBase<scalar_t>& model = *modelPtr;
  PinocchioInterface pinocchioInterface = testingModelInterface_.getPinocchioInterface();
  const SystemObservation& observation = controller.getCurrentObservation();
  const vector_t weightInput = weightCompensatingInput(pinocchioInterface, {true, false}, model, observation.state);
  const std::array<vector6_t, 2> wrenches{model.getContactWrenchInWorldFrame(observation.state, weightInput, /*contactIndex=*/0),
                                          model.getContactWrenchInWorldFrame(observation.state, weightInput, /*contactIndex=*/1)};
  ASSERT_TRUE(wrenches[1].isZero());
  const vector_t expected = computeBaseHeldJointTorques<scalar_t>(model.getGeneralizedCoordinates(observation.state),
                                                                  model.getGeneralizedVelocities(observation.state, observation.input),
                                                                  vector_t::Zero(model.getJointDim()), wrenches, pinocchioInterface);
  // Positive control: the floating-base inverse dynamics of the same single-support wrench accelerates the free base and
  // differs, so the comparison below tells the base-held (gantry) feedforward from it.
  const vector_t floatingBase = computeJointTorques<scalar_t>(model.getGeneralizedCoordinates(observation.state),
                                                              model.getGeneralizedVelocities(observation.state, observation.input),
                                                              vector_t::Zero(model.getJointDim()), wrenches, pinocchioInterface);
  ASSERT_GT((floatingBase - expected).cwiseAbs().maxCoeff(), 1.0);

  const std::vector<size_t> mpcJointIndices = robotDesc.getJointIndices(testingModelInterface_.getModelSettings().mpcModelJointNames);
  scalar_t maxDifferenceBetweenContactStates = 0.0;
  for (size_t i = 0; i < mpcJointIndices.size(); ++i) {
    const size_t index = mpcJointIndices[i];
    EXPECT_NEAR(actionOf(leftFootOnly, index).feed_forward_effort, expected[i], 1.0e-6) << "joint " << index;
    maxDifferenceBetweenContactStates =
        std::max(maxDifferenceBetweenContactStates,
                 std::abs(actionOf(leftFootOnly, index).feed_forward_effort - actionOf(bothFeet, index).feed_forward_effort));
  }
  EXPECT_GT(maxDifferenceBetweenContactStates, 1.0e-3);
}

// The controller with its link started: the solver thread resets the scripted MPC from the controller's observation and
// solves through MPC_BASE::getSolverPtr(), and the policy reaches the controller. The stand-in these tests used before,
// a MockMpc whose getSolverPtr() returned null, could not have been started at all; getSolverPtr() is absl_nonnull.
TEST_F(CentroidalMpcMrtJointControllerTest, testAStartedControllerRunsAgainstTheScriptedMpc) {
  absl::StatusOr<::robot::model::RobotDescription> robotDescOrStatus =
      ::robot::model::RobotDescription::Create(testingModelInterface_.urdfFile);
  ASSERT_TRUE(robotDescOrStatus.ok()) << robotDescOrStatus.status();
  ::robot::model::RobotDescription& robotDesc = *robotDescOrStatus;
  ::robot::model::RobotState robotState(robotDesc);
  ::robot::model::RobotJointAction jointAction(robotDesc);
  absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>> controllerOrStatus = CentroidalMpcMrtJointController::Create(
      robotDesc, testingModelInterface_.getModelSettings(), testingModelInterface_.getMpcRobotModel(), scriptedMpc_,
      testingModelInterface_.getPinocchioInterface(), /*mpcDesiredFrequency=*/100.0, tempPdGainsFile_.string());
  ASSERT_TRUE(controllerOrStatus.ok()) << controllerOrStatus.status();
  CentroidalMpcMrtJointController& controller = **controllerOrStatus;
  ASSERT_FALSE(controller.ready()) << "a policy arrived before the link was started";

  controller.startMpcThread(robotState);
  const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!controller.ready() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(controller.ready()) << "the solver thread produced no policy";
  EXPECT_GE(scriptedMpc_.solver().numResets(), 1u) << "the link did not reset the MPC from the start observation";
  EXPECT_GE(scriptedMpc_.solver().numSolves(), 1u);
  EXPECT_EQ(scriptedMpc_.solver().numFailedSolves(), 0u);
  EXPECT_NO_THROW(controller.computeJointControlAction(/*time=*/0.01, robotState, jointAction));
}

// Create() refuses a robot description that lacks a joint of the MPC model, naming the joint: the control cycle indexes
// the joints without a check, so the constructor's check would otherwise end the process on a mismatched URDF.
TEST_F(CentroidalMpcMrtJointControllerTest, testCreateRefusesAModelJointTheRobotDescriptionDoesNotHave) {
  absl::StatusOr<::robot::model::RobotDescription> description = twoJointDescription();
  ASSERT_TRUE(description.ok()) << description.status();
  const absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>> created = CentroidalMpcMrtJointController::Create(
      *description, testingModelInterface_.getModelSettings(), testingModelInterface_.getMpcRobotModel(), scriptedMpc_,
      testingModelInterface_.getPinocchioInterface(), /*mpcDesiredFrequency=*/400.0, tempPdGainsFile_.string());
  EXPECT_EQ(created.status().code(), absl::StatusCode::kNotFound);
  EXPECT_NE(created.status().message().find(testingModelInterface_.getModelSettings().mpcModelJointNames.front()), std::string::npos)
      << created.status();
}

}  // namespace ocs2::humanoid
