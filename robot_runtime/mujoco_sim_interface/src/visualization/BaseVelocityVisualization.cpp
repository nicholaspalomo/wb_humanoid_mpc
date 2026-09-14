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

#include "mujoco_sim_interface/MujocoSimInterface.h"

namespace robot::mujoco_sim_interface {

void BaseVelocityVisualization::addSceneGeoms(const VisualizationFrame& frame) {
  if (frame.sim == nullptr || frame.state == nullptr || frame.scene == nullptr) return;
  const mjModel* model = frame.sim->getModel();
  const mjData* data = frame.state->data;
  mjvScene& scene = *frame.scene;
  if (model == nullptr || data == nullptr || model->nq < 7 || model->nbody < 2) return;

  // Centre of mass of the whole robot
  const mjtNum* com = &data->subtree_com[0];

  // Base orientation: the base is body 1 on a free joint whose quaternion starts at qpos[3]
  const mjtNum* quat = &data->qpos[3];
  mjtNum mat[9];
  mju_quat2Mat(mat, quat);

  // Commanded velocities (base frame)
  const double target_vx = frame.sim->getTargetVelocityX();
  const double target_vy = frame.sim->getTargetVelocityY();
  const double target_yaw = frame.sim->getTargetYawRate();

  // Measured velocities: cvel of body 1 is [angular(3), linear(3)] in the local frame
  const mjtNum* cvel_base = &data->cvel[6 * 1];
  const double current_vx = cvel_base[3];
  const double current_vy = cvel_base[4];
  const double current_yaw = cvel_base[2];

  // Rotate the linear velocities to the world frame for rendering
  const mjtNum v_cur_base[3] = {current_vx, current_vy, 0.0};
  mjtNum v_cur_world[3];
  mju_mulMatVec(v_cur_world, mat, v_cur_base, 3, 3);

  const mjtNum v_tgt_base[3] = {target_vx, target_vy, 0.0};
  mjtNum v_tgt_world[3];
  mju_mulMatVec(v_tgt_world, mat, v_tgt_base, 3, 3);

  const auto draw_arrow = [&](const mjtNum* dir_world, const float rgba[4], double scale_factor) {
    if (scene.ngeom >= scene.maxgeom) return;
    const double mag = std::sqrt(dir_world[0] * dir_world[0] + dir_world[1] * dir_world[1]);
    if (mag < 1e-3) return;

    mjvGeom* arrow = &scene.geoms[scene.ngeom++];
    std::memset(arrow, 0, sizeof(mjvGeom));
    const double length = mag * scale_factor;
    const mjtNum from[3] = {com[0], com[1], com[2]};
    const mjtNum to[3] = {com[0] + length * (dir_world[0] / mag), com[1] + length * (dir_world[1] / mag),
                          com[2] + length * (dir_world[2] / mag)};
    mjv_connector(arrow, mjGEOM_ARROW, 0.02, from, to);
    for (int i = 0; i < 4; ++i) arrow->rgba[i] = rgba[i];
    arrow->category = mjCAT_DECOR;
    arrow->emission = 1.0f;
  };

  const float red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
  draw_arrow(v_cur_world, red, 1.0);
  const float green[4] = {0.0f, 1.0f, 0.0f, 1.0f};
  draw_arrow(v_tgt_world, green, 1.0);

  // The yaw rate is drawn along the base's z axis, up for a positive rate, offset so the two arrows do not overlap.
  const mjtNum yaw_axis_base[3] = {0.0, 0.0, 1.0};
  mjtNum yaw_axis_world[3];
  mju_mulMatVec(yaw_axis_world, mat, yaw_axis_base, 3, 3);

  const auto draw_yaw_arrow = [&](double yaw_rate, const float rgba[4], double offset) {
    if (scene.ngeom >= scene.maxgeom || std::abs(yaw_rate) < 1e-3) return;
    mjvGeom* arrow = &scene.geoms[scene.ngeom++];
    std::memset(arrow, 0, sizeof(mjvGeom));
    const double length = std::abs(yaw_rate) * 0.2;
    const double sign = (yaw_rate > 0) ? 1.0 : -1.0;
    const mjtNum from[3] = {com[0] + offset, com[1] + offset, com[2]};
    const mjtNum to[3] = {from[0] + sign * length * yaw_axis_world[0], from[1] + sign * length * yaw_axis_world[1],
                          from[2] + sign * length * yaw_axis_world[2]};
    mjv_connector(arrow, mjGEOM_ARROW, 0.02, from, to);
    for (int i = 0; i < 4; ++i) arrow->rgba[i] = rgba[i];
    arrow->category = mjCAT_DECOR;
    arrow->emission = 1.0f;
  };

  const float blue[4] = {0.0f, 0.0f, 1.0f, 1.0f};
  draw_yaw_arrow(current_yaw, blue, 0.05);
  const float yellow[4] = {1.0f, 1.0f, 0.0f, 1.0f};
  draw_yaw_arrow(target_yaw, yellow, -0.05);
}

}  // namespace robot::mujoco_sim_interface
