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

#include <string>

#include "mujoco/mujoco.h"

#include "mujoco_sim_interface/visualization/MujocoVisualization.h"

namespace robot::mujoco_sim_interface {

/**
 * Returns whether the viewer hides geom `geom` of `model` while
 * HideCollisionGeomsVisualization is on: a geom MuJoCo collides (a nonzero
 * contype or conaffinity) on a body other than the world that also has a geom
 * that does not collide, its visual mesh. The floor and the rest of the world's
 * geoms, the dodgeball and a link without a visual geom stay visible. `geom`
 * must be below model.ngeom.
 */
bool isHiddenCollisionGeom(const mjModel& model, int geom);

/**
 * Hides the collision geoms of the robot, so that the viewer shows its visual
 * meshes only (isHiddenCollisionGeom()). On, it removes those geoms from the
 * scene after mjv_updateScene(); off, the viewer draws them as MuJoCo does. The
 * geom groups the number keys toggle still apply to the geoms it keeps. Not
 * thread-safe: the renderer calls it on its own thread.
 */
class HideCollisionGeomsVisualization final : public MujocoVisualization {
 public:
  std::string name() const override { return "hide_collision_geoms"; }
  std::string description() const override {
    return "hides the collision geoms of every body with a visual mesh; off, "
           "the viewer draws them";
  }
  char hotkey() const override { return 'v'; }
  /** Removes the hidden geoms from frame.scene; the decor geoms of the other
   * visualizations are not the model's geoms and stay. */
  void addSceneGeoms(const VisualizationFrame& frame) override;
};

}  // namespace robot::mujoco_sim_interface
