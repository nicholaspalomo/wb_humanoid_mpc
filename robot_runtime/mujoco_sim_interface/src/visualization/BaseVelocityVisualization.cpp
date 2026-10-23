/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

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

#include "mujoco_sim_interface/visualization/BaseVelocityVisualization.h"

#include <cmath>
#include <cstring>
#include <functional>

#include "absl/base/nullability.h"
#include "ocs2_core/Dimensions.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/MujocoUtils.h"

namespace robot::mujoco_sim_interface {

using ::ocs2::kPositionDim;
using ::ocs2::kXAxisIndex;
using ::ocs2::kYAxisIndex;
using ::ocs2::kZAxisIndex;

namespace {
// The base is body kBaseBodyId, on a free joint whose qpos holds its position (3), then its quaternion (4).
constexpr int kMinBodies = 2;  // the world and the base
constexpr int kBaseBodyId = 1;

// The yaw rate in a body's row of mjData::cvel, a motion vector (angular velocity, then linear velocity) about the
// subtree center of mass in world-aligned axes: the angular velocity about the world's z axis.
constexpr int kCvelAngularVelocityZIndex = kZAxisIndex;

constexpr double kArrowWidth = 0.02;            // [m]
constexpr double kMinDrawnSpeed = 1.0e-3;       // [m/s] a slower horizontal velocity draws no arrow
constexpr double kMinDrawnYawRate = 1.0e-3;     // [rad/s] a slower yaw rate draws no arrow
constexpr double kYawArrowLengthPerRate = 0.2;  // [m / (rad/s)]
// [m] The offset of the yaw arrows along x and y from the center of mass: + for the measured rate, - for the commanded.
constexpr double kYawArrowOffset = 0.05;
}  // namespace

void BaseVelocityVisualization::addSceneGeoms(const VisualizationFrame& frame) {
  if (frame.sim == nullptr || frame.model == nullptr || frame.state == nullptr || frame.scene == nullptr) return;
  if (frame.state->data == nullptr) return;
  const BaseVelocityCommand command{
      .velocityX = frame.sim->getTargetVelocityX(), .velocityY = frame.sim->getTargetVelocityY(), .yawRate = frame.sim->getTargetYawRate()};
  addArrows(frame.model, frame.state->data, command, *frame.scene);
}

void BaseVelocityVisualization::addArrows(const mjModel* absl_nonnull model,
                                          const mjData* absl_nonnull data,
                                          const BaseVelocityCommand& command,
                                          mjvScene& scene) {
  if (model->nq < kMjFreeJointQposSize || model->nbody < kMinBodies) return;

  // Center of mass of the robot: its root's subtree, NOT body 0's. The world's subtree also holds the thrown ball,
  // which would drag the arrows away from the robot - by meters, for a 5 kg ball parked out of play.
  const RobotCentroidalState robot = robotCentroidalState(model, data);
  if (!robot.valid) return;
  const double* absl_nonnull com = robot.com;

  // The heading of the base, whose free joint's quaternion starts at qpos[3]: the twist of its attitude about the world
  // vertical (design D9), the frame of the commanded velocity. The command is a velocity along the ground in that
  // frame, so it is turned by the heading only and stays horizontal however the base tilts.
  const VerticalTwist twist = verticalTwistOf(data->qpos[kMjFreeJointQuaternionWIndex], data->qpos[kMjFreeJointQuaternionZIndex]);
  const double cosHeading = twist.w * twist.w - twist.z * twist.z;
  const double sinHeading = 2.0 * twist.w * twist.z;

  // The measured velocity is the robot's center-of-mass velocity, already in the world frame (mj_objectVelocity with
  // flg_local = 0): it is drawn as it is, horizontally. Only the command, in the base frame, is rotated.
  const mjtNum v_cur_world[kPositionDim] = {robot.comVelocity[kXAxisIndex], robot.comVelocity[kYAxisIndex], 0.0};
  // The yaw rate of the base, about the world's z axis.
  const double current_yaw = data->cvel[kMjSpatialVectorSize * kBaseBodyId + kCvelAngularVelocityZIndex];

  const mjtNum v_tgt_world[kPositionDim] = {cosHeading * command.velocityX - sinHeading * command.velocityY,
                                            sinHeading * command.velocityX + cosHeading * command.velocityY, 0.0};

  const std::function<void(const mjtNum* absl_nonnull, const float* absl_nonnull, double)> draw_arrow =
      [&](const mjtNum* absl_nonnull dir_world, const float rgba[absl_nonnull kMjRgbaChannels], double scale_factor) {
        if (scene.ngeom >= scene.maxgeom) return;
        const double mag =
            std::sqrt(dir_world[0] * dir_world[0] + dir_world[1] * dir_world[1] + dir_world[kZAxisIndex] * dir_world[kZAxisIndex]);
        if (mag < kMinDrawnSpeed) return;

        mjvGeom* absl_nonnull arrow = &scene.geoms[scene.ngeom++];
        std::memset(arrow, 0, sizeof(mjvGeom));
        const double length = mag * scale_factor;
        const mjtNum from[kPositionDim] = {com[kXAxisIndex], com[kYAxisIndex], com[kZAxisIndex]};
        const mjtNum to[kPositionDim] = {com[0] + length * (dir_world[0] / mag), com[1] + length * (dir_world[1] / mag),
                                         com[kZAxisIndex] + length * (dir_world[kZAxisIndex] / mag)};
        mjv_connector(arrow, mjGEOM_ARROW, kArrowWidth, from, to);
        for (int i = 0; i < kMjRgbaChannels; ++i) arrow->rgba[i] = rgba[i];
        arrow->category = mjCAT_DECOR;
        arrow->emission = 1.0f;
      };

  const float red[kMjRgbaChannels] = {1.0f, 0.0f, 0.0f, 1.0f};
  draw_arrow(v_cur_world, red, /*scale_factor=*/1.0);
  const float green[kMjRgbaChannels] = {0.0f, 1.0f, 0.0f, 1.0f};
  draw_arrow(v_tgt_world, green, /*scale_factor=*/1.0);

  // The yaw rates are rates about the world's vertical, the measured one and the commanded heading rate alike, so they
  // are drawn along the world's z axis, up for a positive rate, offset so the two arrows do not overlap.
  const mjtNum yaw_axis_world[kPositionDim] = {0.0, 0.0, 1.0};

  const std::function<void(double, const float* absl_nonnull, double)> draw_yaw_arrow =
      [&](double yaw_rate, const float rgba[absl_nonnull kMjRgbaChannels], double offset) {
        if (scene.ngeom >= scene.maxgeom || std::abs(yaw_rate) < kMinDrawnYawRate) return;
        mjvGeom* absl_nonnull arrow = &scene.geoms[scene.ngeom++];
        std::memset(arrow, 0, sizeof(mjvGeom));
        const double length = std::abs(yaw_rate) * kYawArrowLengthPerRate;
        const double sign = (yaw_rate > 0) ? 1.0 : -1.0;
        const mjtNum from[kPositionDim] = {com[kXAxisIndex] + offset, com[kYAxisIndex] + offset, com[kZAxisIndex]};
        const mjtNum to[kPositionDim] = {from[0] + sign * length * yaw_axis_world[0], from[1] + sign * length * yaw_axis_world[1],
                                         from[kZAxisIndex] + sign * length * yaw_axis_world[kZAxisIndex]};
        mjv_connector(arrow, mjGEOM_ARROW, kArrowWidth, from, to);
        for (int i = 0; i < kMjRgbaChannels; ++i) arrow->rgba[i] = rgba[i];
        arrow->category = mjCAT_DECOR;
        arrow->emission = 1.0f;
      };

  const float blue[kMjRgbaChannels] = {0.0f, 0.0f, 1.0f, 1.0f};
  draw_yaw_arrow(current_yaw, blue, /*offset=*/kYawArrowOffset);
  const float yellow[kMjRgbaChannels] = {1.0f, 1.0f, 0.0f, 1.0f};
  draw_yaw_arrow(command.yawRate, yellow, /*offset=*/-kYawArrowOffset);
}

}  // namespace robot::mujoco_sim_interface
