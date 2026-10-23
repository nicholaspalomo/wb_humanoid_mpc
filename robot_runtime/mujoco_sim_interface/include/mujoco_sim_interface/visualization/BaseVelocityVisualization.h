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

#pragma once

#include <string>

#include "absl/base/nullability.h"
#include "mujoco/mujoco.h"

#include "mujoco_sim_interface/visualization/MujocoVisualization.h"

namespace robot::mujoco_sim_interface {

/**
 * The commanded base velocity the arrows of BaseVelocityVisualization are drawn against: a velocity along the ground in
 * the base's heading frame (its attitude's twist about the world vertical, design D9) and the heading rate.
 */
struct BaseVelocityCommand {
  double velocityX = 0.0;  // [m/s] forward
  double velocityY = 0.0;  // [m/s] lateral
  double yawRate = 0.0;    // [rad/s] about the world vertical
};

/**
 * Arrows at the center of mass: the measured horizontal center-of-mass velocity (red) against the commanded base velocity
 * turned into the world by the base's heading (green), each as long as its speed, and the measured (blue) against the
 * commanded (yellow) yaw rate as arrows along the world's vertical. Not thread-safe; the
 * renderer calls it on its own thread.
 */
class BaseVelocityVisualization final : public MujocoVisualization {
 public:
  std::string name() const override { return "base_velocity"; }
  std::string description() const override { return "measured (red) vs commanded (green) base velocity and yaw rate at the CoM"; }
  void addSceneGeoms(const VisualizationFrame& frame) override;

  /**
   * Appends to `scene`, while it has room, the arrows of the measured velocities of the robot in `data` (on which
   * mj_forward() has run) and of `command`; draws nothing for a model without a robot on a free joint. addSceneGeoms()
   * draws them with the simulator's command. Retains nothing.
   */
  static void addArrows(const mjModel* absl_nonnull model,
                        const mjData* absl_nonnull data,
                        const BaseVelocityCommand& command,
                        mjvScene& scene);
};

}  // namespace robot::mujoco_sim_interface
