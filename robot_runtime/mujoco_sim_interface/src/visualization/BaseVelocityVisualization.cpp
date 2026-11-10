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

#include <array>
#include <cmath>
#include <cstring>

#include "absl/base/nullability.h"
#include "ocs2_core/Dimensions.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/visualization/MujocoVisualization.h"

namespace robot::mujoco_sim_interface {

using ::ocs2::kHorizontalPositionDim;
using ::ocs2::kPositionDim;
using ::ocs2::kXAxisIndex;
using ::ocs2::kYAxisIndex;
using ::ocs2::kZAxisIndex;

namespace {
// The base is body kBaseBodyId, on a free joint whose qpos holds its position
// (3), then its quaternion (4).
constexpr int kMinBodies = 2;  // the world and the base
constexpr int kBaseBodyId = 1;

// The yaw rate in a body's row of mjData::cvel, a motion vector (angular
// velocity, then linear velocity) about the subtree center of mass in
// world-aligned axes: the angular velocity about the world's z axis.
constexpr int kCvelAngularVelocityZIndex = kZAxisIndex;

constexpr double kArrowWidth = 0.02;  // [m]
constexpr double kMinDrawnSpeedLocalRtWorld =
    1.0e-3;  // [m/s] a slower horizontal velocity draws no arrow
constexpr double kMinDrawnAngvelBaseRtWorldEwrtWorldZ =
    1.0e-3;  // [rad/s] a slower yaw rate draws no arrow
constexpr double kYawArrowLengthPerRate = 0.2;  // [m / (rad/s)]
// [m] The offset of the yaw arrows along x and y from the center of mass: + for
// the measured rate, - for the commanded.
constexpr std::array<double, kHorizontalPositionDim> kPosArrowRtComEwrtWorldXy =
    {0.05, 0.05};

/// Adds to `scene` an arrow from the center of mass along
/// `linvelLocalRtWorldEwrtWorld` [m/s], `scaleFactor` [m per m/s] long,
/// unless the scene is full or the speed is below kMinDrawnSpeedLocalRtWorld.
void addLinvelArrow(const double* absl_nonnull posComRtWorldEwrtWorld,
                    const mjtNum* absl_nonnull linvelLocalRtWorldEwrtWorld,
                    const float rgba[absl_nonnull kMjRgbaChannels],
                    double scaleFactor,
                    mjvScene& scene) {
  if (scene.ngeom >= scene.maxgeom) return;
  const double speed_local_rt_world = std::sqrt(
      linvelLocalRtWorldEwrtWorld[0] * linvelLocalRtWorldEwrtWorld[0] +
      linvelLocalRtWorldEwrtWorld[1] * linvelLocalRtWorldEwrtWorld[1] +
      linvelLocalRtWorldEwrtWorld[kZAxisIndex] *
          linvelLocalRtWorldEwrtWorld[kZAxisIndex]);
  if (speed_local_rt_world < kMinDrawnSpeedLocalRtWorld) return;

  mjvGeom* absl_nonnull arrow = &scene.geoms[scene.ngeom++];
  std::memset(arrow, 0, sizeof(mjvGeom));
  const double length = speed_local_rt_world * scaleFactor;
  const mjtNum posArrowRtWorldEwrtWorld[kPositionDim] = {
      posComRtWorldEwrtWorld[kXAxisIndex], posComRtWorldEwrtWorld[kYAxisIndex],
      posComRtWorldEwrtWorld[kZAxisIndex]};
  const mjtNum posArrowheadRtWorldEwrtWorld[kPositionDim] = {
      posComRtWorldEwrtWorld[0] +
          length * (linvelLocalRtWorldEwrtWorld[0] / speed_local_rt_world),
      posComRtWorldEwrtWorld[1] +
          length * (linvelLocalRtWorldEwrtWorld[1] / speed_local_rt_world),
      posComRtWorldEwrtWorld[kZAxisIndex] +
          length * (linvelLocalRtWorldEwrtWorld[kZAxisIndex] /
                    speed_local_rt_world)};
  mjv_connector(arrow, mjGEOM_ARROW, kArrowWidth, posArrowRtWorldEwrtWorld,
                posArrowheadRtWorldEwrtWorld);
  for (int i = 0; i < kMjRgbaChannels; ++i) arrow->rgba[i] = rgba[i];
  arrow->category = mjCAT_DECOR;
  arrow->emission = 1.0f;
}

/// Adds to `scene` a vertical arrow for the yaw rate
/// `angvelBaseRtWorldEwrtWorldZ` [rad/s], up for a positive rate, at
/// `posArrowRtComEwrtWorldXy` from the center of mass, unless the scene
/// is full or the rate is below kMinDrawnAngvelBaseRtWorldEwrtWorldZ.
void addYawArrow(
    const double* absl_nonnull posComRtWorldEwrtWorld,
    double angvelBaseRtWorldEwrtWorldZ,
    const float rgba[absl_nonnull kMjRgbaChannels],
    const std::array<double, kHorizontalPositionDim>& posArrowRtComEwrtWorldXy,
    mjvScene& scene) {
  const mjtNum up_axis_ewrt_world[kPositionDim] = {0.0, 0.0, 1.0};
  if (scene.ngeom >= scene.maxgeom ||
      std::abs(angvelBaseRtWorldEwrtWorldZ) <
          kMinDrawnAngvelBaseRtWorldEwrtWorldZ) {
    return;
  }
  mjvGeom* absl_nonnull arrow = &scene.geoms[scene.ngeom++];
  std::memset(arrow, 0, sizeof(mjvGeom));
  const double length =
      std::abs(angvelBaseRtWorldEwrtWorldZ) * kYawArrowLengthPerRate;
  const double sign = (angvelBaseRtWorldEwrtWorldZ > 0) ? 1.0 : -1.0;
  const mjtNum posArrowRtWorldEwrtWorld[kPositionDim] = {
      posComRtWorldEwrtWorld[kXAxisIndex] +
          posArrowRtComEwrtWorldXy[kXAxisIndex],
      posComRtWorldEwrtWorld[kYAxisIndex] +
          posArrowRtComEwrtWorldXy[kYAxisIndex],
      posComRtWorldEwrtWorld[kZAxisIndex]};
  const mjtNum posArrowheadRtWorldEwrtWorld[kPositionDim] = {
      posArrowRtWorldEwrtWorld[0] + sign * length * up_axis_ewrt_world[0],
      posArrowRtWorldEwrtWorld[1] + sign * length * up_axis_ewrt_world[1],
      posArrowRtWorldEwrtWorld[kZAxisIndex] +
          sign * length * up_axis_ewrt_world[kZAxisIndex]};
  mjv_connector(arrow, mjGEOM_ARROW, kArrowWidth, posArrowRtWorldEwrtWorld,
                posArrowheadRtWorldEwrtWorld);
  for (int i = 0; i < kMjRgbaChannels; ++i) arrow->rgba[i] = rgba[i];
  arrow->category = mjCAT_DECOR;
  arrow->emission = 1.0f;
}

}  // namespace

void BaseVelocityVisualization::addSceneGeoms(const VisualizationFrame& frame) {
  if (frame.sim == nullptr || frame.model == nullptr ||
      frame.state == nullptr || frame.scene == nullptr) {
    return;
  }
  if (frame.state->data == nullptr) return;
  const BaseVelocityCommand command{
      .cmdLinvelBaseRtWorldEwrtHeadingX =
          frame.sim->getCmdLinvelBaseRtWorldEwrtHeadingX(),
      .cmdLinvelBaseRtWorldEwrtHeadingY =
          frame.sim->getCmdLinvelBaseRtWorldEwrtHeadingY(),
      .cmdAngvelBaseRtWorldEwrtWorldZ =
          frame.sim->getCmdAngvelBaseRtWorldEwrtWorldZ()};
  addArrows(frame.model, frame.state->data, command, *frame.scene);
}

void BaseVelocityVisualization::addArrows(const mjModel* absl_nonnull model,
                                          const mjData* absl_nonnull data,
                                          const BaseVelocityCommand& command,
                                          mjvScene& scene) {
  if (model->nq < kMjFreeJointQposSize || model->nbody < kMinBodies) return;

  // Center of mass of the robot: its root's subtree, NOT body 0's. The world's
  // subtree also holds the thrown ball, which would drag the arrows away from
  // the robot - by meters, for a 5 kg ball parked out of play.
  const RobotCentroidalState robot = robotCentroidalState(model, data);
  if (!robot.valid) return;
  const double* absl_nonnull posComRtWorldEwrtWorld =
      robot.posComRtWorldEwrtWorld;

  // The heading of the base, whose free joint's quaternion starts at qpos[3]:
  // the twist of its attitude about the world vertical (design D9), the frame
  // of the commanded velocity. The command is a velocity along the ground in
  // that frame, so it is turned by the heading only and stays horizontal
  // however the base tilts.
  const VerticalTwist quatHeadingRtWorld =
      verticalTwistFromQuaternion(data->qpos[kMjFreeJointQuaternionWIndex],
                                  data->qpos[kMjFreeJointQuaternionZIndex]);
  const double cosYawBaseRtWorld = quatHeadingRtWorld.w * quatHeadingRtWorld.w -
                                   quatHeadingRtWorld.z * quatHeadingRtWorld.z;
  const double sinYawBaseRtWorld =
      2.0 * quatHeadingRtWorld.w * quatHeadingRtWorld.z;

  // The measured velocity is the robot's center-of-mass velocity, already in
  // the world frame (mj_objectVelocity with flg_local = 0): it is drawn as it
  // is, horizontally. Only the command, in the heading frame, is rotated (by
  // the heading).
  const mjtNum planar_meas_linvel_com_rt_world_ewrt_world[kPositionDim] = {
      robot.linvelComRtWorldEwrtWorld[kXAxisIndex],
      robot.linvelComRtWorldEwrtWorld[kYAxisIndex], 0.0};
  // The yaw rate of the base, about the world's z axis.
  const double meas_angvel_base_rt_world_ewrt_world_z =
      data->cvel[kMjSpatialVectorSize * kBaseBodyId +
                 kCvelAngularVelocityZIndex];

  const mjtNum planar_cmd_linvel_base_rt_world_ewrt_world[kPositionDim] = {
      cosYawBaseRtWorld * command.cmdLinvelBaseRtWorldEwrtHeadingX -
          sinYawBaseRtWorld * command.cmdLinvelBaseRtWorldEwrtHeadingY,
      sinYawBaseRtWorld * command.cmdLinvelBaseRtWorldEwrtHeadingX +
          cosYawBaseRtWorld * command.cmdLinvelBaseRtWorldEwrtHeadingY,
      0.0};

  const float red[kMjRgbaChannels] = {1.0f, 0.0f, 0.0f, 1.0f};
  addLinvelArrow(posComRtWorldEwrtWorld,
                 planar_meas_linvel_com_rt_world_ewrt_world, red,
                 /*scaleFactor=*/1.0, scene);
  const float green[kMjRgbaChannels] = {0.0f, 1.0f, 0.0f, 1.0f};
  addLinvelArrow(posComRtWorldEwrtWorld,
                 planar_cmd_linvel_base_rt_world_ewrt_world, green,
                 /*scaleFactor=*/1.0, scene);

  // The yaw rates are rates about the world's vertical, the measured one and
  // the commanded heading rate alike, so they are drawn along the world's z
  // axis, up for a positive rate, offset so the two arrows do not overlap.

  const float blue[kMjRgbaChannels] = {0.0f, 0.0f, 1.0f, 1.0f};
  addYawArrow(posComRtWorldEwrtWorld, meas_angvel_base_rt_world_ewrt_world_z,
              blue,
              /*posArrowRtComEwrtWorldXy=*/kPosArrowRtComEwrtWorldXy, scene);
  const float yellow[kMjRgbaChannels] = {1.0f, 1.0f, 0.0f, 1.0f};
  addYawArrow(posComRtWorldEwrtWorld, command.cmdAngvelBaseRtWorldEwrtWorldZ,
              yellow,
              /*posArrowRtComEwrtWorldXy=*/
              {-kPosArrowRtComEwrtWorldXy[kXAxisIndex],
               -kPosArrowRtComEwrtWorldXy[kYAxisIndex]},
              scene);
}

}  // namespace robot::mujoco_sim_interface
