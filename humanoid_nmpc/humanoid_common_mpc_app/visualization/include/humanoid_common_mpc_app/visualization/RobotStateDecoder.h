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

#include <array>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_mpc_msgs/robot_state_sample.nproto.h"

namespace ocs2::humanoid::visualization {

/** A robot/state sample in the terms of the MPC's model and of the robot's joint list (ModelSettings::fullJointNames). */
struct DecodedRobotState {
  /** [s] The robot's clock. */
  scalar_t time = 0.0;

  // The root of the floating base, in the world frame.
  vector3_t basePosition = vector3_t::Zero();
  /** Local to world. */
  matrix3_t baseRotation = matrix3_t::Identity();
  quaternion_t baseOrientation = quaternion_t::Identity();
  vector3_t baseLinearVelocity = vector3_t::Zero();
  vector3_t baseAngularVelocity = vector3_t::Zero();
  /** (yaw, pitch, roll). */
  vector3_t baseEulerAnglesZyx = vector3_t::Zero();

  /**
   * The generalized coordinates, velocities and forces of the MPC's Pinocchio model: base position, Euler ZYX angles,
   * then the joints of ModelSettings::mpcModelJointNames. The base velocity is the world linear velocity and the Euler
   * angle rates (JointModelSphericalZYX's velocity), the base force is zero, and a joint's force is the effort
   * applied. A joint the sample does not carry is zero.
   */
  vector_t generalizedCoordinates;
  vector_t generalizedVelocities;
  vector_t generalizedForces;

  /**
   * Per joint of ModelSettings::fullJointNames: the measured position and velocity, the effort applied
   * (feed-forward + kp (q_des - q) + kd (qd_des - qd), 0 without an action), and the action's targets (position target:
   * the measured position without an action; velocity target and feed-forward effort: 0 without one).
   */
  vector_t jointPositions;
  vector_t jointVelocities;
  vector_t jointEfforts;
  vector_t jointPositionTargets;
  vector_t jointVelocityTargets;
  vector_t jointFeedForwardEfforts;

  /** Per contact of ModelSettings::contactNames: the measured wrench [force; torque], world frame; 0 when not sent. */
  feet_array_t<vector6_t> measuredContactWrenches = makeFeetArray<vector6_t>(vector6_t::Zero());
};

/**
 * Reads robot/state samples for the visualization. A sample names its joints, so the decoder maps them onto the MPC's
 * joints and the robot's joint list by name; the mapping is computed again only when the names change.
 *
 * A joint's action: the five action arrays of a sample are each either empty or aligned with joint_names. A joint has
 * an action when the position targets are sent and its target is finite; a sample whose position targets are empty
 * applied no action this cycle. An empty velocity-target, gain or feed-forward array counts as zeros.
 */
class RobotStateDecoder {
 public:
  explicit RobotStateDecoder(const ModelSettings& modelSettings);

  /**
   * Decodes `sample` into `state`.
   *
   * @return InvalidArgument when a joint array is neither empty (the action arrays) nor aligned with joint_names, when
   *         joint positions or velocities are missing, or when the base orientation is not a unit quaternion up to
   *         rounding; `state` is then unspecified.
   */
  absl::Status decode(const msgs::RobotStateSample& sample, DecodedRobotState* absl_nonnull state);

 private:
  void updateJointIndices(const std::vector<std::string>& sampleJointNames);

  std::vector<std::string> fullJointNames_;
  std::vector<std::string> mpcJointNames_;
  /** The joint names the indices below were computed for. */
  std::vector<std::string> sampleJointNames_;
  bool hasJointIndices_ = false;
  /** Index into the sample's arrays per joint of fullJointNames_ / mpcJointNames_; -1 when the sample lacks it. */
  std::vector<int> fullJointToSample_;
  std::vector<int> mpcJointToSample_;
};

}  // namespace ocs2::humanoid::visualization
