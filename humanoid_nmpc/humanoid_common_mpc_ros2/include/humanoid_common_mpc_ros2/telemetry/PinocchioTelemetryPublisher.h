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

#pragma once

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <geometry_msgs/msg/accel_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <ocs2_ros2_msgs/msg/mpc_observation.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>

#include <humanoid_common_mpc/common/ModelSettings.h>
#include <humanoid_common_mpc/common/MpcRobotModelBase.h>
#include <ocs2_mpc/CommandData.h>
#include <ocs2_mpc/SystemObservation.h>
#include <robot_model/RobotDescription.h>
#include <robot_model/RobotJointAction.h>
#include <robot_model/RobotState.h>

namespace ocs2::humanoid {

/**
 * @brief Pinocchio-driven telemetry publisher broadcasting:
 * 1. Entire generalized coordinates, velocities, and forces:
 *    - /robot/generalized_coordinates/[dof_name]
 *    - /robot/generalized_velocities/[dof_name]
 *    - /robot/generalized_forces/[dof_name]
 *    - /mpc/desired/generalized_coordinates/[dof_name]
 *    - /mpc/desired/generalized_velocities/[dof_name]
 *    - /mpc/desired/generalized_forces/[dof_name]
 * 2. Consolidated generalized state messages:
 *    - /robot/generalized_state
 *    - /mpc/desired/generalized_state
 * 3. Targeted list of Pinocchio frames (measured and desired):
 *    - Position & Orientation: /robot/frames/[frame]/pose, /robot/frames/[frame]/euler
 *    - Velocity: /robot/frames/[frame]/twist
 *    - Classical Acceleration: /robot/frames/[frame]/accel
 *    - Contact Force / Wrench: /robot/frames/[frame]/wrench
 * 4. Standard backward-compatible topics:
 *    - /joint_states, /mpc/joint_targets
 *    - /robot/base_pose, /robot/base_euler, /robot/base_twist
 *    - /mpc/target_base_pose, /mpc/target_base_euler, /mpc/target_base_twist
 *    - /mpc/contact_wrench/left|right, /sensors/contact_wrench/left|right
 *    - /mpc/observation
 */
class PinocchioTelemetryPublisher {
 public:
  struct FrameTrackInfo {
    std::string frameName;
    pinocchio::FrameIndex frameId;
    size_t contactIndex = std::numeric_limits<size_t>::max();

    // Measured Frame Publishers
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr measuredPosePub;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr measuredEulerPub;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr measuredTwistPub;
    rclcpp::Publisher<geometry_msgs::msg::AccelStamped>::SharedPtr measuredAccelPub;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr measuredWrenchPub;

    // Desired Frame Publishers
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr desiredPosePub;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr desiredEulerPub;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr desiredTwistPub;
    rclcpp::Publisher<geometry_msgs::msg::AccelStamped>::SharedPtr desiredAccelPub;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr desiredWrenchPub;
  };

  struct DofTrackInfo {
    std::string dofName;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr robotCoordPub;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr robotVelPub;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr robotForcePub;

    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr mpcDesiredCoordPub;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr mpcDesiredVelPub;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr mpcDesiredForcePub;
  };

  PinocchioTelemetryPublisher(rclcpp::Node::SharedPtr nodeHandle,
                              const PinocchioInterface& pinocchioInterface,
                              const ModelSettings& modelSettings,
                              const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                              const ::robot::model::RobotDescription& robotDescription,
                              const std::vector<std::string>& trackedFrames = {});

  void publish(const ::robot::model::RobotState& robotState,
               const ::robot::model::RobotJointAction& robotJointAction,
               const SystemObservation& mpcObservation,
               const vector_t& mpcPolicyInput,
               const CommandData& mpcCommand,
               const vector3_t& leftMeasuredForce,
               const vector3_t& rightMeasuredForce);

  void publishPinocchioState(const rclcpp::Time& stamp,
                             const vector_t& q_meas,
                             const vector_t& v_meas,
                             const vector_t& tau_meas,
                             const vector_t& q_des,
                             const vector_t& v_des,
                             const vector_t& tau_des,
                             const std::unordered_map<std::string, vector3_t>& measuredForces = {},
                             const std::unordered_map<std::string, vector6_t>& desiredWrenches = {});

  static vector3_t quaternionToEulerZYX(const quaternion_t& q);

  const std::vector<std::string>& getDofNames() const { return dofNames_; }
  const std::vector<std::string>& getTrackedFrameNames() const { return trackedFrameNames_; }

 private:
  void initializePublishers(const rclcpp::QoS& qos, const std::vector<std::string>& userTrackedFrames);

  rclcpp::Node::SharedPtr nodeHandle_;
  PinocchioInterface pinocchioInterface_;
  pinocchio::Data dataMeasured_;
  pinocchio::Data dataDesired_;

  const ModelSettings* modelSettingsPtr_ = nullptr;
  const MpcRobotModelBase<scalar_t>* mpcRobotModelPtr_ = nullptr;
  const ::robot::model::RobotDescription* robotDescriptionPtr_ = nullptr;

  std::vector<std::string> dofNames_;
  std::vector<std::string> fullJointNames_;
  std::vector<size_t> mpcJointIndices_;
  std::vector<size_t> descJointIndices_;
  std::vector<size_t> descFullJointIndices_;
  std::vector<std::string> trackedFrameNames_;

  // Per-DOF Publishers
  std::vector<DofTrackInfo> dofTrackInfos_;

  // Per-Frame Publishers
  std::vector<FrameTrackInfo> frameTrackInfos_;

  // Consolidated Generalized State Publishers
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr robotGenStatePub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr mpcDesiredGenStatePub_;

  // Standard Backward-Compatible Publishers
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr jointStatePub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr mpcJointTargetPub_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr robotBasePosePub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr robotBaseEulerPub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr robotBaseTwistPub_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mpcTargetBasePosePub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr mpcTargetBaseEulerPub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr mpcTargetBaseTwistPub_;

  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr mpcContactWrenchLeftPub_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr mpcContactWrenchRightPub_;

  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr simContactWrenchLeftPub_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr simContactWrenchRightPub_;

  rclcpp::Publisher<ocs2_ros2_msgs::msg::MpcObservation>::SharedPtr mpcObservationPub_;

  // Previous velocities for acceleration estimation
  bool hasPreviousVelocities_ = false;
  rclcpp::Time previousStamp_;
  vector_t prevVMeas_;
  vector_t prevVDes_;
};

}  // namespace ocs2::humanoid
