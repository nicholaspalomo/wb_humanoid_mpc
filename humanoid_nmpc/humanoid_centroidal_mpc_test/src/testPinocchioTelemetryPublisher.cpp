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
#include <rclcpp/rclcpp.hpp>
#include <thread>

#include <geometry_msgs/msg/accel_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>

#include "humanoid_centroidal_mpc_test/CentroidalTestingModelInterface.h"
#include "humanoid_common_mpc_ros2/telemetry/PinocchioTelemetryPublisher.h"

using namespace ocs2;
using namespace ocs2::humanoid;

TEST(TestPinocchioTelemetryPublisher, DofTopicNamingAndPublishing) {
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  auto node = std::make_shared<rclcpp::Node>("test_pinocchio_telemetry_node");

  CentroidalTestingModelInterface testingModelInterface;
  const auto& pinocchioInterface = testingModelInterface.getPinocchioInterface();
  const auto& modelSettings = *testingModelInterface.modelSettingsPtr;
  const auto& mpcRobotModel = testingModelInterface.getMpcRobotModel();

  ::robot::model::RobotDescription robotDesc(testingModelInterface.urdfFile);

  std::vector<std::string> trackedFrames = {"foot_l_contact", "foot_r_contact"};
  PinocchioTelemetryPublisher publisher(node, pinocchioInterface, modelSettings, mpcRobotModel, robotDesc, trackedFrames);

  // 1. Verify DOF names contain base DOFs followed by actuated joint names
  const auto& dofNames = publisher.getDofNames();
  ASSERT_GE(dofNames.size(), 6 + modelSettings.mpc_joint_dim);
  EXPECT_EQ(dofNames[0], "base_x");
  EXPECT_EQ(dofNames[1], "base_y");
  EXPECT_EQ(dofNames[2], "base_z");
  EXPECT_EQ(dofNames[3], "base_yaw");
  EXPECT_EQ(dofNames[4], "base_pitch");
  EXPECT_EQ(dofNames[5], "base_roll");

  // 2. Verify tracked frame names
  const auto& frameNames = publisher.getTrackedFrameNames();
  EXPECT_EQ(frameNames.size(), 2u);
  EXPECT_EQ(frameNames[0], "foot_l_contact");
  EXPECT_EQ(frameNames[1], "foot_r_contact");

  // 3. Subscribe to per-DOF topics formatted as /mpc/desired/generalized_coordinates/[dof_name]
  auto subQos = rclcpp::QoS(10).best_effort();

  bool receivedMpcBaseZ = false;
  double mpcBaseZVal = 0.0;
  auto subMpcBaseZ = node->create_subscription<std_msgs::msg::Float64>("mpc/desired/generalized_coordinates/base_z", subQos,
                                                                       [&](const std_msgs::msg::Float64::SharedPtr msg) {
                                                                         receivedMpcBaseZ = true;
                                                                         mpcBaseZVal = msg->data;
                                                                       });

  bool receivedRobotBaseZ = false;
  double robotBaseZVal = 0.0;
  auto subRobotBaseZ = node->create_subscription<std_msgs::msg::Float64>("robot/generalized_coordinates/base_z", subQos,
                                                                         [&](const std_msgs::msg::Float64::SharedPtr msg) {
                                                                           receivedRobotBaseZ = true;
                                                                           robotBaseZVal = msg->data;
                                                                         });

  bool receivedFrameAccel = false;
  auto subFrameAccel = node->create_subscription<geometry_msgs::msg::AccelStamped>(
      "robot/frames/foot_l_contact/accel", subQos,
      [&](const geometry_msgs::msg::AccelStamped::SharedPtr /*msg*/) { receivedFrameAccel = true; });

  bool receivedGenState = false;
  auto subGenState = node->create_subscription<sensor_msgs::msg::JointState>("robot/generalized_state", subQos,
                                                                             [&](const sensor_msgs::msg::JointState::SharedPtr msg) {
                                                                               receivedGenState = true;
                                                                               EXPECT_EQ(msg->name.size(), dofNames.size());
                                                                             });

  // 4. Publish Pinocchio state
  const auto& model = pinocchioInterface.getModel();
  vector_t q_meas = vector_t::Zero(model.nq);
  q_meas[2] = 0.78;  // base_z = 0.78 m
  vector_t v_meas = vector_t::Zero(model.nv);
  vector_t tau_meas = vector_t::Zero(model.nv);

  vector_t q_des = vector_t::Zero(model.nq);
  q_des[2] = 0.85;  // desired base_z = 0.85 m
  vector_t v_des = vector_t::Zero(model.nv);
  vector_t tau_des = vector_t::Zero(model.nv);

  publisher.publishPinocchioState(node->now(), q_meas, v_meas, tau_meas, q_des, v_des, tau_des);

  // Spin node to process subscribers
  for (int i = 0; i < 20; ++i) {
    rclcpp::spin_some(node);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  EXPECT_TRUE(receivedMpcBaseZ);
  EXPECT_DOUBLE_EQ(mpcBaseZVal, 0.85);

  EXPECT_TRUE(receivedRobotBaseZ);
  EXPECT_DOUBLE_EQ(robotBaseZVal, 0.78);

  EXPECT_TRUE(receivedFrameAccel);
  EXPECT_TRUE(receivedGenState);
}

TEST(TestPinocchioTelemetryPublisher, PublishHighLevelState) {
  if (!rclcpp::ok()) {
    rclcpp::init(0, nullptr);
  }
  auto node = std::make_shared<rclcpp::Node>("test_pinocchio_high_level_node");

  CentroidalTestingModelInterface testingModelInterface;
  const auto& pinocchioInterface = testingModelInterface.getPinocchioInterface();
  const auto& modelSettings = *testingModelInterface.modelSettingsPtr;
  const auto& mpcRobotModel = testingModelInterface.getMpcRobotModel();

  ::robot::model::RobotDescription robotDesc(testingModelInterface.urdfFile);
  ::robot::model::RobotState robotState(robotDesc);
  ::robot::model::RobotJointAction robotJointAction(robotDesc);

  // Set root state
  robotState.setRootPositionInWorldFrame(vector3_t(0.1, 0.2, 0.8));
  robotState.setRootRotationLocalToWorldFrame(quaternion_t::Identity());
  robotState.setRootLinearVelocityInLocalFrame(vector3_t(0.05, 0.0, 0.0));
  robotState.setRootAngularVelocityInLocalFrame(vector3_t::Zero());

  // Set active joint angles
  for (size_t i = 0; i < robotDesc.getNumJoints(); ++i) {
    robotState.setJointPosition(i, 0.02 * static_cast<double>(i));
    robotState.setJointVelocity(i, 0.01);
  }

  SystemObservation observation;
  observation.time = 1.0;
  observation.mode = 3;
  observation.state = vector_t::Zero(mpcRobotModel.getStateDim());
  observation.input = vector_t::Zero(mpcRobotModel.getInputDim());

  vector_t mpcPolicyInput = vector_t::Zero(mpcRobotModel.getInputDim());
  CommandData commandData;

  std::vector<std::string> trackedFrames = {"foot_l_contact", "foot_r_contact"};
  PinocchioTelemetryPublisher publisher(node, pinocchioInterface, modelSettings, mpcRobotModel, robotDesc, trackedFrames);

  auto subQos = rclcpp::QoS(10).best_effort();
  bool receivedJointStates = false;
  auto subJointStates = node->create_subscription<sensor_msgs::msg::JointState>(
      "joint_states", subQos, [&](const sensor_msgs::msg::JointState::SharedPtr msg) {
        receivedJointStates = true;
        EXPECT_EQ(msg->name.size(), modelSettings.fullJointNames.size());
      });

  bool receivedBasePose = false;
  auto subBasePose = node->create_subscription<geometry_msgs::msg::PoseStamped>("robot/base_pose", subQos,
                                                                                [&](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                                                                                  receivedBasePose = true;
                                                                                  EXPECT_DOUBLE_EQ(msg->pose.position.z, 0.8);
                                                                                });

  bool receivedObs = false;
  auto subObs = node->create_subscription<ocs2_ros2_msgs::msg::MpcObservation>(
      "mpc/observation", subQos, [&](const ocs2_ros2_msgs::msg::MpcObservation::SharedPtr msg) {
        receivedObs = true;
        EXPECT_DOUBLE_EQ(msg->time, 1.0);
      });

  publisher.publish(robotState, robotJointAction, observation, mpcPolicyInput, commandData, vector3_t::Zero(), vector3_t::Zero());

  for (int i = 0; i < 20; ++i) {
    rclcpp::spin_some(node);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  EXPECT_TRUE(receivedJointStates);
  EXPECT_TRUE(receivedBasePose);
  EXPECT_TRUE(receivedObs);
}
