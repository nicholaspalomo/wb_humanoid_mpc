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

#include <memory>

#include "mujoco_sim_interface/visualization/MujocoVisualization.h"

namespace robot::mujoco_sim_interface {

/**
 * A visualization MuJoCo draws itself, switched by a flag of mjvOption (contact points, its contact force arrows, the
 * centres of mass, the inertia ellipsoids, the convex hulls) or, for the transparency, by the alpha of the model's
 * geoms. The flag follows the enabled state every frame, so disabling the visualization clears it again.
 */
class MujocoOptionFlagVisualization final : public MujocoVisualization {
 public:
  /** `flag` is an mjtVisFlag index, or -1 for the model transparency. Starts disabled, like the MuJoCo defaults. */
  MujocoOptionFlagVisualization(std::string name, std::string description, char hotkey, int flag);

  static std::unique_ptr<MujocoOptionFlagVisualization> contactPoints();
  static std::unique_ptr<MujocoOptionFlagVisualization> contactForces();
  static std::unique_ptr<MujocoOptionFlagVisualization> centerOfMass();
  static std::unique_ptr<MujocoOptionFlagVisualization> inertia();
  static std::unique_ptr<MujocoOptionFlagVisualization> convexHull();
  static std::unique_ptr<MujocoOptionFlagVisualization> transparency();

  std::string name() const override { return name_; }
  std::string description() const override { return description_; }
  char hotkey() const override { return hotkey_; }
  int flag() const { return flag_; }
  void beforeSceneUpdate(const VisualizationFrame& frame) override;

  static constexpr int kTransparency = -1;
  static constexpr float kTransparentAlpha = 0.3f;

 private:
  std::string name_;
  std::string description_;
  char hotkey_;
  int flag_;
  bool transparencyApplied_{false};
};

}  // namespace robot::mujoco_sim_interface
