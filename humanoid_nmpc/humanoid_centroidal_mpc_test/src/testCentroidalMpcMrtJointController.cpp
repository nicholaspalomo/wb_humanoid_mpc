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
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <thread>

#include <humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h>
#include <ocs2_mpc/MPC_BASE.h>
#include <ocs2_ros2_interfaces/mrt/DummyObserver.h>
#include "humanoid_centroidal_mpc_test/CentroidalTestingModelInterface.h"
#include "humanoid_common_mpc/common/BasisInputsModelDecorator.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

using namespace ocs2;
using namespace ocs2::humanoid;

#include <robot_model/RobotDescription.h>

class MockMpc : public MPC_BASE {
 public:
  MockMpc() : MPC_BASE(mpc::Settings{}) {}

  bool run(scalar_t currentTime, const vector_t& currentState, size_t currentMode = 0) override { return true; }
  void calculateController(scalar_t initTime, const vector_t& initState, size_t initMode, scalar_t finalTime) override {}
  SolverBase* getSolverPtr() override { return nullptr; }
  const SolverBase* getSolverPtr() const override { return nullptr; }
};

class CentroidalMpcMrtJointControllerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tempPdGainsFile_ = std::filesystem::temp_directory_path() / "test_pd_gains.yaml";

    // Create a dummy PD gains file
    std::ofstream ofs(tempPdGainsFile_);
    ofs << "joints:\n"
        << "  LeftKneePitch:\n"
        << "    kp: 100.0\n"
        << "    kd: 10.0\n"
        << "    torque_limit: 150.0\n";
    ofs.close();
  }

  void TearDown() override {
    if (std::filesystem::exists(tempPdGainsFile_)) {
      std::filesystem::remove(tempPdGainsFile_);
    }
  }

  CentroidalTestingModelInterface testingModelInterface;
  std::filesystem::path tempPdGainsFile_;
};

TEST_F(CentroidalMpcMrtJointControllerTest, testPdGainsHotReloading) {
  MockMpc mockMpc;

  ::robot::model::RobotDescription robotDesc(testingModelInterface.urdfFile);
  robot::model::RobotState robotState(robotDesc);
  robot::model::RobotJointAction jointAction(robotDesc);

  // Create controller
  CentroidalMpcMrtJointController controller(robotDesc, testingModelInterface.getModelSettings(), testingModelInterface.getMpcRobotModel(),
                                             mockMpc, testingModelInterface.getPinocchioInterface(), 400.0, nullptr,
                                             tempPdGainsFile_.string());

  controller.setControlMode("JOINT_PD");

  // Trigger first compute loop
  EXPECT_NO_THROW({ controller.computeJointControlAction(0.01, robotState, jointAction); });

  // Modify file
  std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Ensure timestamp difference
  std::ofstream ofs(tempPdGainsFile_, std::ios::app);
  ofs << "  RightKneePitch:\n"
      << "    kp: 200.0\n"
      << "    kd: 20.0\n"
      << "    torque_limit: 150.0\n";
  ofs.close();

  // Trigger again
  EXPECT_NO_THROW({ controller.computeJointControlAction(0.02, robotState, jointAction); });
}

// With basis-vector inputs the OCP input is [lambda_left, lambda_right, joint velocities] and the input-only accessors of the
// effective model return the LOCAL contact-frame wrench B*lambda. The controller must (a) size its observation input to the
// basis layout and (b) hand WORLD-frame wrenches to the inverse dynamics (which uses LOCAL_WORLD_ALIGNED Jacobians). Pitching
// the ankles makes the local contact frames differ from the world frame, so a local-frame wrench would yield different torques.
TEST_F(CentroidalMpcMrtJointControllerTest, testBasisVectorInputsUseWorldFrameWrenchesForFeedforward) {
  MockMpc mockMpc;

  ::robot::model::RobotDescription robotDesc(testingModelInterface.urdfFile);
  robot::model::RobotState robotState(robotDesc);
  robot::model::RobotJointAction jointAction(robotDesc);

  // Pitch both feet so that the local contact frames are not aligned with the world frame.
  constexpr scalar_t kAnklePitch = 0.3;
  robotState.setJointPosition(robotDesc.getJointIndex("left_ankle_pitch_joint"), kAnklePitch);
  robotState.setJointPosition(robotDesc.getJointIndex("right_ankle_pitch_joint"), kAnklePitch);

  // Basis decorator built the same way CentroidalMpcInterface builds it in basis mode.
  const ContactWrenchConeConstraint::Config coneConfig(4, 0.7, 0.05, 5.0, 0.0);
  const PolygonBounds footBounds(-0.1, 0.1, -0.05, 0.05);
  const std::array<ContactWrenchConeBasisMatrix, N_CONTACTS> basisMatrices = {
      ContactWrenchConeBasisMatrix(
          coneConfig, ContactRectangle(footBounds, ContactCenterPoint("foot_l_contact", "left_ankle_roll_joint", vector3_t::Zero()))),
      ContactWrenchConeBasisMatrix(
          coneConfig, ContactRectangle(footBounds, ContactCenterPoint("foot_r_contact", "right_ankle_roll_joint", vector3_t::Zero())))};
  BasisInputsModelDecorator<scalar_t> basisModel(
      std::unique_ptr<MpcRobotModelBase<scalar_t>>(testingModelInterface.getMpcRobotModel().clone()), basisMatrices,
      testingModelInterface.getPinocchioInterface());
  ASSERT_NE(basisModel.getInputDim(), testingModelInterface.getMpcRobotModel().getInputDim());

  // A generous torque limit keeps the controller's clamp from masking the torque comparison below.
  const std::filesystem::path gainsFile = std::filesystem::temp_directory_path() / "test_pd_gains_basis.yaml";
  {
    std::ofstream ofs(gainsFile);
    ofs << "default_gains:\n  kp: 100.0\n  kd: 10.0\n  torque_limit: 1000000.0\n";
  }

  CentroidalMpcMrtJointController controller(robotDesc, testingModelInterface.getModelSettings(), testingModelInterface.getMpcRobotModel(),
                                             mockMpc, testingModelInterface.getPinocchioInterface(), 400.0, nullptr, gainsFile.string(),
                                             &basisModel);

  // The default mode is WB_MPC and the mock MPC never publishes a policy, so the controller takes the weight-compensating
  // feed-forward branch.
  ASSERT_NO_THROW(controller.computeJointControlAction(0.01, robotState, jointAction));
  std::filesystem::remove(gainsFile);

  // (a) The observation input follows the basis layout: a zero contact block with the joint velocities at the tail.
  const SystemObservation& observation = controller.getCurrentObservation();
  ASSERT_EQ(observation.input.size(), static_cast<Eigen::Index>(basisModel.getInputDim()));
  EXPECT_TRUE(observation.input.head(basisModel.getInputDim() - basisModel.getJointDim()).isZero());
  EXPECT_TRUE(basisModel.getJointVelocities(observation.state, observation.input).isZero());

  // (b) Reference torques: the same state-aware weight-compensating input read back in the world frame and projected with the
  // same inverse dynamics the controller uses.
  PinocchioInterface pinocchioInterface = testingModelInterface.getPinocchioInterface();
  const vector_t weightInput = weightCompensatingInput(pinocchioInterface, {true, true}, basisModel, observation.state);
  const std::array<vector6_t, 2> worldWrenches{basisModel.getContactWrenchInWorldFrame(observation.state, weightInput, 0),
                                               basisModel.getContactWrenchInWorldFrame(observation.state, weightInput, 1)};
  const std::array<vector6_t, 2> localWrenches{basisModel.getContactWrench(weightInput, 0), basisModel.getContactWrench(weightInput, 1)};

  const vector_t q = basisModel.getGeneralizedCoordinates(observation.state);
  const vector_t qd = basisModel.getGeneralizedVelocities(observation.state, observation.input);
  const vector_t qddZero = vector_t::Zero(basisModel.getJointDim());
  const vector_t expectedTorques = computeJointTorques<scalar_t>(q, qd, qddZero, worldWrenches, pinocchioInterface);
  const vector_t localFrameTorques = computeJointTorques<scalar_t>(q, qd, qddZero, localWrenches, pinocchioInterface);

  // Sanity: with pitched feet the local- and world-frame wrenches (and hence torques) differ, so the check discriminates.
  ASSERT_GT((worldWrenches[0] - localWrenches[0]).norm(), 1.0);
  ASSERT_GT((expectedTorques - localFrameTorques).norm(), 1.0);

  const auto& mpcJointNames = testingModelInterface.getModelSettings().mpcModelJointNames;
  const auto mpcJointIndices = robotDesc.getJointIndices(mpcJointNames);
  ASSERT_EQ(static_cast<Eigen::Index>(mpcJointIndices.size()), expectedTorques.size());
  for (size_t i = 0; i < mpcJointIndices.size(); ++i) {
    const scalar_t feedforward = jointAction.at(mpcJointIndices[i]).value().feed_forward_effort;
    EXPECT_TRUE(std::isfinite(feedforward)) << "joint " << mpcJointNames[i];
    EXPECT_NEAR(feedforward, expectedTorques[i], 1e-6) << "joint " << mpcJointNames[i];
  }
}
