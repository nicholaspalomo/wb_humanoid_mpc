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

#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/visualization/MujocoVisualization.h"

namespace robot::mujoco_sim_interface {

/**
 * Yellow-green arrows of the pushes on the bodies: the simulator's last push
 * (MjState::push, the dodgeball's impulse when no ball is compiled into the
 * scene), held for kPushHoldTime of simulation time, since the force itself
 * lasts one physics step, which a frame rarely shows; and any other force of
 * mjData::xfrc_applied. Each starts at the body's center of mass, where MuJoCo
 * applies the force, and is at most 1.5 m long. With a ball in the scene the
 * push is the ball's contact, which contact_forces draws. Not thread-safe; the
 * renderer calls it on its own thread.
 */
class ExternalForceVisualization final : public MujocoVisualization {
 public:
  // [s] of simulation time: how long a push stays drawn after it was applied.
  static constexpr double kPushHoldTime = 0.5;
  // [N s] a smaller push draws no arrow.
  static constexpr double kMinDrawnImpulse = 1.0e-6;

  std::string name() const override { return "external_forces"; }
  std::string description() const override {
    return "arrows of the pushes on the bodies (the dodgeball's impulse "
           "without a ball, held 0.5 s; a ball's push is a contact force)";
  }
  void addSceneGeoms(const VisualizationFrame& frame) override;

  /** Appends to `scene`, while it has room, the arrows of `state` (on which
   * mj_forward() has run). Retains nothing. */
  static void addArrows(const mjModel* absl_nonnull model,
                        const MjState& state,
                        mjvScene& scene);
};

}  // namespace robot::mujoco_sim_interface
