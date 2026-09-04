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

#include "humanoid_common_mpc_ros2/telemetry/PinocchioTelemetryPublisher.h"
#include "humanoid_common_mpc_ros2/telemetry/TelemetryRosHelpers.h"

#include <algorithm>
#include <cmath>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <absl/log/log.h>
#include <ocs2_core/misc/LinearInterpolation.h>
#include <ocs2_robotic_tools/common/RotationDerivativesTransforms.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>

namespace ocs2::humanoid {

PinocchioTelemetryPublisher::PinocchioTelemetryPublisher(rclcpp::Node::SharedPtr nodeHandle,
                                                         const PinocchioInterface& pinocchioInterface,
                                                         const ModelSettings& modelSettings,
                                                         const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                         const ::robot::model::RobotDescription& robotDescription,
                                                         const std::vector<std::string>& trackedFrames)
    : nodeHandle_(std::move(nodeHandle)),
      pinocchioInterface_(pinocchioInterface),
      dataMeasured_(pinocchioInterface_.getModel()),
      dataDesired_(pinocchioInterface_.getModel()),
      modelSettingsPtr_(&modelSettings),
      mpcRobotModelPtr_(&mpcRobotModel),
      robotDescriptionPtr_(&robotDescription) {
  auto qos = rclcpp::QoS(10);
  qos.best_effort();

  initializePublishers(qos, trackedFrames);
}

void PinocchioTelemetryPublisher::initializePublishers(const rclcpp::QoS& qos, const std::vector<std::string>& userTrackedFrames) {
  // 1. Initialize DOF names
  // Base DOFs (JointModelTranslation + JointModelSphericalZYX)
  dofNames_ = getBaseDofNames();

  // Actuated joint names from MPC model settings
  for (const auto& jointName : modelSettingsPtr_->mpcModelJointNames) {
    dofNames_.push_back(jointName);
  }

  fullJointNames_.assign(modelSettingsPtr_->fullJointNames.begin(), modelSettingsPtr_->fullJointNames.end());
  mpcJointIndices_ = modelSettingsPtr_->mpcModelToFullJointsIndices;

  // Pre-calculate RobotDescription joint indices for MPC joints and full joints
  descJointIndices_.clear();
  descJointIndices_.reserve(modelSettingsPtr_->mpcModelJointNames.size());
  for (const auto& jointName : modelSettingsPtr_->mpcModelJointNames) {
    if (robotDescriptionPtr_ && robotDescriptionPtr_->containsJoint(jointName)) {
      descJointIndices_.push_back(robotDescriptionPtr_->getJointIndex(jointName));
    } else {
      descJointIndices_.push_back(std::numeric_limits<size_t>::max());
    }
  }

  descFullJointIndices_.clear();
  descFullJointIndices_.reserve(fullJointNames_.size());
  for (const auto& jointName : fullJointNames_) {
    if (robotDescriptionPtr_ && robotDescriptionPtr_->containsJoint(jointName)) {
      descFullJointIndices_.push_back(robotDescriptionPtr_->getJointIndex(jointName));
    } else {
      descFullJointIndices_.push_back(std::numeric_limits<size_t>::max());
    }
  }

  // 2. Initialize Per-DOF Publishers (/mpc/desired/generalized_* and /robot/generalized_*)
  dofTrackInfos_.reserve(dofNames_.size());
  for (const auto& dofName : dofNames_) {
    DofTrackInfo dofInfo;
    dofInfo.dofName = dofName;

    dofInfo.robotCoordPub = nodeHandle_->create_publisher<std_msgs::msg::Float64>("robot/generalized_coordinates/" + dofName, qos);
    dofInfo.robotVelPub = nodeHandle_->create_publisher<std_msgs::msg::Float64>("robot/generalized_velocities/" + dofName, qos);
    dofInfo.robotForcePub = nodeHandle_->create_publisher<std_msgs::msg::Float64>("robot/generalized_forces/" + dofName, qos);

    dofInfo.mpcDesiredCoordPub =
        nodeHandle_->create_publisher<std_msgs::msg::Float64>("mpc/desired/generalized_coordinates/" + dofName, qos);
    dofInfo.mpcDesiredVelPub = nodeHandle_->create_publisher<std_msgs::msg::Float64>("mpc/desired/generalized_velocities/" + dofName, qos);
    dofInfo.mpcDesiredForcePub = nodeHandle_->create_publisher<std_msgs::msg::Float64>("mpc/desired/generalized_forces/" + dofName, qos);

    dofTrackInfos_.push_back(std::move(dofInfo));
  }

  // 3. Initialize Tracked Frames
  std::vector<std::string> candidateFrames = userTrackedFrames;
  if (candidateFrames.empty()) {
    // Default to contact frames and root link if available
    candidateFrames.insert(candidateFrames.end(), modelSettingsPtr_->contactNames.begin(), modelSettingsPtr_->contactNames.end());
  }

  const auto& model = pinocchioInterface_.getModel();
  for (const auto& frameName : candidateFrames) {
    if (!model.existFrame(frameName)) {
      LOG(WARNING) << "PinocchioTelemetryPublisher: Frame '" << frameName << "' does not exist in Pinocchio model. Skipping.";
      continue;
    }

    FrameTrackInfo fInfo;
    fInfo.frameName = frameName;
    fInfo.frameId = model.getFrameId(frameName);

    // Find if this frame is a known contact
    for (size_t c = 0; c < modelSettingsPtr_->contactNames.size(); ++c) {
      if (modelSettingsPtr_->contactNames[c] == frameName) {
        fInfo.contactIndex = c;
        break;
      }
    }

    // Measured frame publishers
    fInfo.measuredPosePub = nodeHandle_->create_publisher<geometry_msgs::msg::PoseStamped>("robot/frames/" + frameName + "/pose", qos);
    fInfo.measuredEulerPub = nodeHandle_->create_publisher<geometry_msgs::msg::Vector3Stamped>("robot/frames/" + frameName + "/euler", qos);
    fInfo.measuredTwistPub = nodeHandle_->create_publisher<geometry_msgs::msg::TwistStamped>("robot/frames/" + frameName + "/twist", qos);
    fInfo.measuredAccelPub = nodeHandle_->create_publisher<geometry_msgs::msg::AccelStamped>("robot/frames/" + frameName + "/accel", qos);
    fInfo.measuredWrenchPub =
        nodeHandle_->create_publisher<geometry_msgs::msg::WrenchStamped>("robot/frames/" + frameName + "/wrench", qos);

    // Desired frame publishers
    fInfo.desiredPosePub = nodeHandle_->create_publisher<geometry_msgs::msg::PoseStamped>("mpc/desired/frames/" + frameName + "/pose", qos);
    fInfo.desiredEulerPub =
        nodeHandle_->create_publisher<geometry_msgs::msg::Vector3Stamped>("mpc/desired/frames/" + frameName + "/euler", qos);
    fInfo.desiredTwistPub =
        nodeHandle_->create_publisher<geometry_msgs::msg::TwistStamped>("mpc/desired/frames/" + frameName + "/twist", qos);
    fInfo.desiredAccelPub =
        nodeHandle_->create_publisher<geometry_msgs::msg::AccelStamped>("mpc/desired/frames/" + frameName + "/accel", qos);
    fInfo.desiredWrenchPub =
        nodeHandle_->create_publisher<geometry_msgs::msg::WrenchStamped>("mpc/desired/frames/" + frameName + "/wrench", qos);

    frameTrackInfos_.push_back(std::move(fInfo));
    trackedFrameNames_.push_back(frameName);
  }

  // 4. Consolidated Generalized State Publishers
  robotGenStatePub_ = nodeHandle_->create_publisher<sensor_msgs::msg::JointState>("robot/generalized_state", qos);
  mpcDesiredGenStatePub_ = nodeHandle_->create_publisher<sensor_msgs::msg::JointState>("mpc/desired/generalized_state", qos);

  // 5. Standard Backward-Compatible Publishers
  jointStatePub_ = nodeHandle_->create_publisher<sensor_msgs::msg::JointState>("joint_states", qos);
  mpcJointTargetPub_ = nodeHandle_->create_publisher<sensor_msgs::msg::JointState>("mpc/joint_targets", qos);

  robotBasePosePub_ = nodeHandle_->create_publisher<geometry_msgs::msg::PoseStamped>("robot/base_pose", qos);
  robotBaseEulerPub_ = nodeHandle_->create_publisher<geometry_msgs::msg::Vector3Stamped>("robot/base_euler", qos);
  robotBaseTwistPub_ = nodeHandle_->create_publisher<geometry_msgs::msg::TwistStamped>("robot/base_twist", qos);

  mpcTargetBasePosePub_ = nodeHandle_->create_publisher<geometry_msgs::msg::PoseStamped>("mpc/target_base_pose", qos);
  mpcTargetBaseEulerPub_ = nodeHandle_->create_publisher<geometry_msgs::msg::Vector3Stamped>("mpc/target_base_euler", qos);
  mpcTargetBaseTwistPub_ = nodeHandle_->create_publisher<geometry_msgs::msg::TwistStamped>("mpc/target_base_twist", qos);

  mpcContactWrenchLeftPub_ = nodeHandle_->create_publisher<geometry_msgs::msg::WrenchStamped>("mpc/contact_wrench/left", qos);
  mpcContactWrenchRightPub_ = nodeHandle_->create_publisher<geometry_msgs::msg::WrenchStamped>("mpc/contact_wrench/right", qos);

  simContactWrenchLeftPub_ = nodeHandle_->create_publisher<geometry_msgs::msg::WrenchStamped>("sensors/contact_wrench/left", qos);
  simContactWrenchRightPub_ = nodeHandle_->create_publisher<geometry_msgs::msg::WrenchStamped>("sensors/contact_wrench/right", qos);

  mpcObservationPub_ = nodeHandle_->create_publisher<ocs2_ros2_msgs::msg::MpcObservation>("mpc/observation", qos);
}

vector3_t PinocchioTelemetryPublisher::quaternionToEulerZYX(const quaternion_t& q) {
  scalar_t w = q.w();
  scalar_t x = q.x();
  scalar_t y = q.y();
  scalar_t z = q.z();

  scalar_t roll = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
  scalar_t pitch = std::asin(std::clamp(2.0 * (w * y - z * x), -1.0, 1.0));
  scalar_t yaw = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));

  return vector3_t(roll, pitch, yaw);
}

void PinocchioTelemetryPublisher::publishPinocchioState(const rclcpp::Time& stamp,
                                                        const vector_t& q_meas,
                                                        const vector_t& v_meas,
                                                        const vector_t& tau_meas,
                                                        const vector_t& q_des,
                                                        const vector_t& v_des,
                                                        const vector_t& tau_des,
                                                        const std::unordered_map<std::string, vector3_t>& measuredForces,
                                                        const std::unordered_map<std::string, vector6_t>& desiredWrenches) {
  const auto& model = pinocchioInterface_.getModel();

  // Numerical differentiation for generalized acceleration
  vector_t a_meas = vector_t::Zero(model.nv);
  vector_t a_des = vector_t::Zero(model.nv);
  if (hasPreviousVelocities_) {
    double dt = (stamp - previousStamp_).seconds();
    if (dt > 1e-5 && dt < 1.0) {
      if (v_meas.size() == prevVMeas_.size()) {
        a_meas = (v_meas - prevVMeas_) / dt;
      }
      if (v_des.size() == prevVDes_.size()) {
        a_des = (v_des - prevVDes_) / dt;
      }
    }
  }
  hasPreviousVelocities_ = true;
  previousStamp_ = stamp;
  prevVMeas_ = v_meas;
  prevVDes_ = v_des;

  // Forward Kinematics for Measured State
  if (q_meas.size() == model.nq && v_meas.size() == model.nv) {
    pinocchio::forwardKinematics(model, dataMeasured_, q_meas, v_meas, a_meas);
    pinocchio::updateFramePlacements(model, dataMeasured_);
  }

  // Forward Kinematics for Desired State
  if (q_des.size() == model.nq && v_des.size() == model.nv) {
    pinocchio::forwardKinematics(model, dataDesired_, q_des, v_des, a_des);
    pinocchio::updateFramePlacements(model, dataDesired_);
  }

  // 1. Publish Per-DOF Signals (/robot/generalized_* and /mpc/desired/generalized_*)
  for (size_t i = 0; i < dofTrackInfos_.size(); ++i) {
    std_msgs::msg::Float64 msg;

    if (static_cast<int>(i) < q_meas.size()) {
      msg.data = q_meas[i];
      dofTrackInfos_[i].robotCoordPub->publish(msg);
    }
    if (static_cast<int>(i) < v_meas.size()) {
      msg.data = v_meas[i];
      dofTrackInfos_[i].robotVelPub->publish(msg);
    }
    if (static_cast<int>(i) < tau_meas.size()) {
      msg.data = tau_meas[i];
      dofTrackInfos_[i].robotForcePub->publish(msg);
    }

    if (static_cast<int>(i) < q_des.size()) {
      msg.data = q_des[i];
      dofTrackInfos_[i].mpcDesiredCoordPub->publish(msg);
    }
    if (static_cast<int>(i) < v_des.size()) {
      msg.data = v_des[i];
      dofTrackInfos_[i].mpcDesiredVelPub->publish(msg);
    }
    if (static_cast<int>(i) < tau_des.size()) {
      msg.data = tau_des[i];
      dofTrackInfos_[i].mpcDesiredForcePub->publish(msg);
    }
  }

  // 2. Publish Consolidated Generalized State Messages
  sensor_msgs::msg::JointState rGenMsg;
  rGenMsg.header.stamp = stamp;
  rGenMsg.name = dofNames_;
  if (q_meas.size() > 0) {
    rGenMsg.position.assign(q_meas.data(), q_meas.data() + q_meas.size());
  }
  if (v_meas.size() > 0) {
    rGenMsg.velocity.assign(v_meas.data(), v_meas.data() + v_meas.size());
  }
  if (tau_meas.size() > 0) {
    rGenMsg.effort.assign(tau_meas.data(), tau_meas.data() + tau_meas.size());
  }
  robotGenStatePub_->publish(rGenMsg);

  sensor_msgs::msg::JointState mGenMsg;
  mGenMsg.header.stamp = stamp;
  mGenMsg.name = dofNames_;
  if (q_des.size() > 0) {
    mGenMsg.position.assign(q_des.data(), q_des.data() + q_des.size());
  }
  if (v_des.size() > 0) {
    mGenMsg.velocity.assign(v_des.data(), v_des.data() + v_des.size());
  }
  if (tau_des.size() > 0) {
    mGenMsg.effort.assign(tau_des.data(), tau_des.data() + tau_des.size());
  }
  mpcDesiredGenStatePub_->publish(mGenMsg);

  // 3. Publish Per-Frame Signals (Pose, Euler, Twist, Accel, Wrench)
  for (const auto& fInfo : frameTrackInfos_) {
    // Measured Frame
    if (q_meas.size() == model.nq && v_meas.size() == model.nv) {
      const auto& placementMeas = dataMeasured_.oMf[fInfo.frameId];
      const auto vFrameMeas =
          pinocchio::getFrameVelocity(model, dataMeasured_, fInfo.frameId, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);
      const auto aFrameMeas =
          pinocchio::getFrameClassicalAcceleration(model, dataMeasured_, fInfo.frameId, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);

      const quaternion_t qm(placementMeas.rotation());
      fInfo.measuredPosePub->publish(createPoseStamped(stamp, "world", placementMeas.translation(), qm));
      fInfo.measuredEulerPub->publish(createVector3Stamped(stamp, "world", quaternionToEulerZYX(qm)));
      fInfo.measuredTwistPub->publish(createTwistStamped(stamp, "world", vFrameMeas.linear(), vFrameMeas.angular()));
      fInfo.measuredAccelPub->publish(createAccelStamped(stamp, "world", aFrameMeas.linear(), aFrameMeas.angular()));

      auto itForce = measuredForces.find(fInfo.frameName);
      if (itForce != measuredForces.end()) {
        fInfo.measuredWrenchPub->publish(createForceWrenchStamped(stamp, "world", itForce->second));
      } else {
        fInfo.measuredWrenchPub->publish(createForceWrenchStamped(stamp, "world", vector3_t::Zero()));
      }
    }

    // Desired Frame
    if (q_des.size() == model.nq && v_des.size() == model.nv) {
      const auto& placementDes = dataDesired_.oMf[fInfo.frameId];
      const auto vFrameDes =
          pinocchio::getFrameVelocity(model, dataDesired_, fInfo.frameId, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);
      const auto aFrameDes =
          pinocchio::getFrameClassicalAcceleration(model, dataDesired_, fInfo.frameId, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);

      const quaternion_t qd(placementDes.rotation());
      fInfo.desiredPosePub->publish(createPoseStamped(stamp, "world", placementDes.translation(), qd));
      fInfo.desiredEulerPub->publish(createVector3Stamped(stamp, "world", quaternionToEulerZYX(qd)));
      fInfo.desiredTwistPub->publish(createTwistStamped(stamp, "world", vFrameDes.linear(), vFrameDes.angular()));
      fInfo.desiredAccelPub->publish(createAccelStamped(stamp, "world", aFrameDes.linear(), aFrameDes.angular()));

      auto itWrench = desiredWrenches.find(fInfo.frameName);
      if (itWrench != desiredWrenches.end()) {
        fInfo.desiredWrenchPub->publish(createWrenchStamped(stamp, "world", itWrench->second));
      } else {
        fInfo.desiredWrenchPub->publish(createWrenchStamped(stamp, "world", vector6_t::Zero()));
      }
    }
  }
}

void PinocchioTelemetryPublisher::publish(const ::robot::model::RobotState& robotState,
                                          const ::robot::model::RobotJointAction& robotJointAction,
                                          const SystemObservation& mpcObservation,
                                          const vector_t& mpcPolicyInput,
                                          const CommandData& mpcCommand,
                                          const vector3_t& leftMeasuredForce,
                                          const vector3_t& rightMeasuredForce) {
  const auto now = nodeHandle_->now();
  const auto& model = pinocchioInterface_.getModel();
  const size_t numMpcJoints = modelSettingsPtr_->mpc_joint_dim;

  // 1. Compute Measured Generalized State (q_meas, v_meas, tau_meas)
  vector_t q_meas = vector_t::Zero(model.nq);
  vector_t v_meas = vector_t::Zero(model.nv);
  vector_t tau_meas = vector_t::Zero(model.nv);

  const vector3_t rootPos = robotState.getRootPositionInWorldFrame();
  const quaternion_t rootQuat = robotState.getRootRotationLocalToWorldFrame();
  const vector3_t rootEuler = quaternionToEulerZYX(rootQuat);  // roll, pitch, yaw
  const vector3_t rootLinVel = rootQuat * robotState.getRootLinearVelocityInLocalFrame();
  const vector3_t rootAngVel = rootQuat * robotState.getRootAngularVelocityInLocalFrame();

  q_meas.head<BASE_TRANSLATION_DIM>() = rootPos;
  // Pinocchio JointModelSphericalZYX convention: yaw, pitch, roll
  const vector3_t eulerAnglesZyx(rootEuler.z(), rootEuler.y(), rootEuler.x());
  q_meas.segment<BASE_ROTATION_DIM>(BASE_TRANSLATION_DIM) = eulerAnglesZyx;

  v_meas.head<BASE_TRANSLATION_DIM>() = rootLinVel;
  v_meas.segment<BASE_ROTATION_DIM>(BASE_TRANSLATION_DIM) =
      getEulerAnglesZyxDerivativesFromGlobalAngularVelocity<scalar_t>(eulerAnglesZyx, rootAngVel);

  // Actuated Joints
  for (size_t j = 0; j < numMpcJoints; ++j) {
    if (j < descJointIndices_.size()) {
      size_t descIdx = descJointIndices_[j];
      if (descIdx != std::numeric_limits<size_t>::max() && robotDescriptionPtr_ && descIdx < robotDescriptionPtr_->getNumJoints()) {
        try {
          q_meas[JOINT_COORDINATE_OFFSET + j] = robotState.getJointPosition(descIdx);
          v_meas[JOINT_COORDINATE_OFFSET + j] = robotState.getJointVelocity(descIdx);
          if (robotDescriptionPtr_ && descIdx < robotDescriptionPtr_->getNumJoints()) {
            const auto& actionOpt = robotJointAction.at(descIdx);
            if (actionOpt.has_value()) {
              const auto& action = actionOpt.value();
              tau_meas[JOINT_COORDINATE_OFFSET + j] = action.feed_forward_effort +
                                                      action.kp * (action.q_des - q_meas[JOINT_COORDINATE_OFFSET + j]) +
                                                      action.kd * (action.qd_des - v_meas[JOINT_COORDINATE_OFFSET + j]);
            }
          }
        } catch (const std::exception&) {
          // Fall back to default
        }
      }
    }
  }

  // 2. Compute Desired Generalized State (q_des, v_des, tau_des)
  vector_t q_des = q_meas;
  vector_t v_des = vector_t::Zero(model.nv);
  vector_t tau_des = vector_t::Zero(model.nv);
  vector_t targetInput = mpcPolicyInput;

  if (!mpcCommand.mpcTargetTrajectories_.timeTrajectory.empty()) {
    scalar_t time = mpcObservation.time;
    vector_t targetState = LinearInterpolation::interpolate(time, mpcCommand.mpcTargetTrajectories_.timeTrajectory,
                                                            mpcCommand.mpcTargetTrajectories_.stateTrajectory);
    q_des = mpcRobotModelPtr_->getGeneralizedCoordinates(targetState);

    if (!mpcCommand.mpcTargetTrajectories_.inputTrajectory.empty()) {
      targetInput = LinearInterpolation::interpolate(time, mpcCommand.mpcTargetTrajectories_.timeTrajectory,
                                                     mpcCommand.mpcTargetTrajectories_.inputTrajectory);
      v_des = const_cast<MpcRobotModelBase<scalar_t>*>(mpcRobotModelPtr_)->getGeneralizedVelocities(targetState, targetInput);
    }
  }

  for (size_t j = 0; j < numMpcJoints; ++j) {
    if (j < descJointIndices_.size()) {
      size_t descIdx = descJointIndices_[j];
      if (descIdx != std::numeric_limits<size_t>::max() && robotDescriptionPtr_ && descIdx < robotDescriptionPtr_->getNumJoints()) {
        const auto& actionOpt = robotJointAction.at(descIdx);
        if (actionOpt.has_value()) {
          tau_des[JOINT_COORDINATE_OFFSET + j] = actionOpt.value().feed_forward_effort;
        }
      }
    }
  }

  // 3. Contact Wrenches Setup
  std::unordered_map<std::string, vector3_t> measuredForces;
  std::unordered_map<std::string, vector6_t> desiredWrenches;

  if (modelSettingsPtr_->contactNames.size() >= N_CONTACTS) {
    measuredForces[modelSettingsPtr_->contactNames[0]] = leftMeasuredForce;
    measuredForces[modelSettingsPtr_->contactNames[1]] = rightMeasuredForce;
  }

  for (size_t c = 0; c < modelSettingsPtr_->contactNames.size(); ++c) {
    if (targetInput.size() >= static_cast<int>(mpcRobotModelPtr_->getInputDim())) {
      desiredWrenches[modelSettingsPtr_->contactNames[c]] = mpcRobotModelPtr_->getContactWrench(targetInput, c);
    }
  }

  // 4. Publish Pinocchio State & Frames
  publishPinocchioState(now, q_meas, v_meas, tau_meas, q_des, v_des, tau_des, measuredForces, desiredWrenches);

  // 5. Standard Backward-Compatible Topics
  // /joint_states
  sensor_msgs::msg::JointState jointStateMsg;
  jointStateMsg.header.stamp = now;
  jointStateMsg.name = fullJointNames_;
  jointStateMsg.position.resize(fullJointNames_.size(), 0.0);
  jointStateMsg.velocity.resize(fullJointNames_.size(), 0.0);
  jointStateMsg.effort.resize(fullJointNames_.size(), 0.0);

  for (size_t i = 0; i < fullJointNames_.size(); ++i) {
    size_t descIdx = (i < descFullJointIndices_.size()) ? descFullJointIndices_[i] : std::numeric_limits<size_t>::max();
    if (descIdx != std::numeric_limits<size_t>::max() && robotDescriptionPtr_ && descIdx < robotDescriptionPtr_->getNumJoints()) {
      try {
        jointStateMsg.position[i] = robotState.getJointPosition(descIdx);
        jointStateMsg.velocity[i] = robotState.getJointVelocity(descIdx);
      } catch (const std::exception&) {
        jointStateMsg.position[i] = 0.0;
        jointStateMsg.velocity[i] = 0.0;
      }
      if (robotDescriptionPtr_ && descIdx < robotDescriptionPtr_->getNumJoints()) {
        const auto& actionOpt = robotJointAction.at(descIdx);
        if (actionOpt.has_value()) {
          const auto& action = actionOpt.value();
          jointStateMsg.effort[i] = action.feed_forward_effort + action.kp * (action.q_des - jointStateMsg.position[i]) +
                                    action.kd * (action.qd_des - jointStateMsg.velocity[i]);
        }
      }
    }
  }
  jointStatePub_->publish(jointStateMsg);

  // /mpc/joint_targets
  sensor_msgs::msg::JointState targetJointMsg;
  targetJointMsg.header.stamp = now;
  targetJointMsg.name = fullJointNames_;
  targetJointMsg.position.resize(fullJointNames_.size(), 0.0);
  targetJointMsg.velocity.resize(fullJointNames_.size(), 0.0);
  targetJointMsg.effort.resize(fullJointNames_.size(), 0.0);

  for (size_t i = 0; i < fullJointNames_.size(); ++i) {
    size_t descIdx = (i < descFullJointIndices_.size()) ? descFullJointIndices_[i] : std::numeric_limits<size_t>::max();
    if (descIdx != std::numeric_limits<size_t>::max() && robotDescriptionPtr_ && descIdx < robotDescriptionPtr_->getNumJoints()) {
      const auto& actionOpt = robotJointAction.at(descIdx);
      if (actionOpt.has_value()) {
        targetJointMsg.position[i] = actionOpt.value().q_des;
        targetJointMsg.velocity[i] = actionOpt.value().qd_des;
        targetJointMsg.effort[i] = actionOpt.value().feed_forward_effort;
      } else if (robotDescriptionPtr_ && descIdx < robotDescriptionPtr_->getNumJoints()) {
        try {
          targetJointMsg.position[i] = robotState.getJointPosition(descIdx);
        } catch (const std::exception&) {
          targetJointMsg.position[i] = 0.0;
        }
      }
    }
  }
  mpcJointTargetPub_->publish(targetJointMsg);

  // /robot/base_*
  robotBasePosePub_->publish(createPoseStamped(now, "world", rootPos, rootQuat));
  robotBaseEulerPub_->publish(createVector3Stamped(now, "world", rootEuler));
  robotBaseTwistPub_->publish(createTwistStamped(now, "world", rootLinVel, rootAngVel));

  // /mpc/target_base_*
  vector3_t targetPos = rootPos;
  quaternion_t targetQuat = rootQuat;
  vector3_t targetEuler = rootEuler;
  vector3_t targetLinVel = vector3_t::Zero();
  vector3_t targetAngVel = vector3_t::Zero();

  if (!mpcCommand.mpcTargetTrajectories_.timeTrajectory.empty()) {
    scalar_t time = mpcObservation.time;
    vector_t targetState = LinearInterpolation::interpolate(time, mpcCommand.mpcTargetTrajectories_.timeTrajectory,
                                                            mpcCommand.mpcTargetTrajectories_.stateTrajectory);
    targetPos = mpcRobotModelPtr_->getBasePosition(targetState);
    const vector3_t targetEulerZyx = mpcRobotModelPtr_->getBaseOrientationEulerZYX(targetState);
    targetEuler = vector3_t(targetEulerZyx.z(), targetEulerZyx.y(), targetEulerZyx.x());
    targetQuat = getQuaternionFromEulerAnglesZyx(targetEulerZyx);

    if (!mpcCommand.mpcTargetTrajectories_.inputTrajectory.empty()) {
      vector_t tInput = LinearInterpolation::interpolate(time, mpcCommand.mpcTargetTrajectories_.timeTrajectory,
                                                         mpcCommand.mpcTargetTrajectories_.inputTrajectory);
      if (tInput.size() >= static_cast<int>(N_CONTACTS * CONTACT_WRENCH_DIM)) {
        targetLinVel = mpcRobotModelPtr_->getBaseComLinearVelocity(targetState);
      }
    }
  }

  mpcTargetBasePosePub_->publish(createPoseStamped(now, "world", targetPos, targetQuat));
  mpcTargetBaseEulerPub_->publish(createVector3Stamped(now, "world", targetEuler));
  mpcTargetBaseTwistPub_->publish(createTwistStamped(now, "world", targetLinVel, targetAngVel));

  // /mpc/contact_wrench/left|right
  if (mpcPolicyInput.size() >= static_cast<int>(N_CONTACTS * CONTACT_WRENCH_DIM)) {
    mpcContactWrenchLeftPub_->publish(
        createWrenchStamped(now, "world", mpcPolicyInput.segment<CONTACT_WRENCH_DIM>(CONTACT_LEFT_INDEX * CONTACT_WRENCH_DIM)));
    mpcContactWrenchRightPub_->publish(
        createWrenchStamped(now, "world", mpcPolicyInput.segment<CONTACT_WRENCH_DIM>(CONTACT_RIGHT_INDEX * CONTACT_WRENCH_DIM)));
  }

  // /sensors/contact_wrench/left|right
  simContactWrenchLeftPub_->publish(createForceWrenchStamped(now, "world", leftMeasuredForce));
  simContactWrenchRightPub_->publish(createForceWrenchStamped(now, "world", rightMeasuredForce));

  // /mpc/observation
  ocs2_ros2_msgs::msg::MpcObservation obsMsg;
  obsMsg.time = mpcObservation.time;
  obsMsg.mode = mpcObservation.mode;
  if (mpcObservation.state.size() > 0) {
    obsMsg.state.value.assign(mpcObservation.state.data(), mpcObservation.state.data() + mpcObservation.state.size());
  }
  if (mpcPolicyInput.size() > 0) {
    obsMsg.input.value.assign(mpcPolicyInput.data(), mpcPolicyInput.data() + mpcPolicyInput.size());
  }
  mpcObservationPub_->publish(obsMsg);
}

}  // namespace ocs2::humanoid
