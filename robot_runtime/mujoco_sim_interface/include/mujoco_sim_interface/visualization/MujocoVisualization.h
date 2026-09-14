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

#include <mujoco/mujoco.h>

#include <string>

#include "mujoco_sim_interface/MujocoUtils.h"

namespace robot::mujoco_sim_interface {

class MujocoSimInterface;

/**
 * Everything a visualization may read or draw into during one frame of the viewer. Owned by the renderer; the pointers
 * are valid for the duration of the frame only. All hooks run on the render thread.
 */
struct VisualizationFrame {
  const MujocoSimInterface* sim{nullptr};  // simulator (model, config, data published by the control thread)
  const MjState* state{nullptr};           // copy of the physics state the frame is drawn from (mj_forward has run on it)
  mjvOption* options{nullptr};             // MuJoCo visualization flags of the viewer
  mjvScene* scene{nullptr};                // abstract scene; addSceneGeoms appends decor geoms here
  const mjrContext* context{nullptr};      // GPU context, for text and 2D overlays
  mjrRect viewport{0, 0, 0, 0};            // framebuffer rectangle in pixels, origin bottom-left
  double renderFps{0.0};
  double elapsedRealTime{0.0};  // [s] wall-clock time since the viewer started
};

/**
 * One marker or overlay of the MuJoCo viewer: contact force arrows, the contact timeline, the target contact patches,
 * ... Every visualization is a class of its own, registered under a name in VisualizationRegistry.h; the task file lists
 * the names to enable (`simVisualizations`) and the hotkey, if the class declares one, toggles it in the viewer.
 *
 * The renderer calls the three hooks once per frame in this order:
 *  1. beforeSceneUpdate(): before mjv_updateScene, called whether the visualization is enabled or not, so that a
 *     visualization that acts on the MuJoCo options or the model can also undo its effect once disabled;
 *  2. addSceneGeoms(): after mjv_updateScene and before mjr_render, enabled visualizations only; append decor geoms;
 *  3. renderOverlay(): after mjr_render, enabled visualizations only; draw 2D overlays and text.
 */
class MujocoVisualization {
 public:
  virtual ~MujocoVisualization() = default;

  /** Name under which the visualization is registered and listed in the task file. */
  virtual std::string name() const = 0;
  /** One line for the hotkey cheatsheet. */
  virtual std::string description() const = 0;
  /** Lower-case letter that toggles the visualization in the viewer, or 0 for none. */
  virtual char hotkey() const { return 0; }

  virtual void beforeSceneUpdate(const VisualizationFrame& /*frame*/) {}
  virtual void addSceneGeoms(const VisualizationFrame& /*frame*/) {}
  virtual void renderOverlay(const VisualizationFrame& /*frame*/) {}

  bool enabled() const { return enabled_; }
  void setEnabled(bool enabled) { enabled_ = enabled; }
  void toggle() { enabled_ = !enabled_; }

 private:
  bool enabled_{true};
};

}  // namespace robot::mujoco_sim_interface
