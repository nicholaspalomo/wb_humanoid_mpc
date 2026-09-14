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

#include <array>
#include <vector>

namespace robot::mujoco_sim_interface {

/**
 * Target contact pose of one contact point, in the world frame: where the controller wants the foot's contact frame on
 * the ground (the landing pose of its swing in flight or of its next swing, or its current placement). Fed from the
 * control thread (MujocoSimInterface::setTargetContactPatches) and drawn by the viewer as the foot's contact patch
 * ('g' toggles it), so that the planned position and yaw of every step can be seen against the robot.
 */
struct TargetContactPatch {
  enum class Kind : int {
    STANCE = 0,           // the foot's current placement (no upcoming swing)
    SWING_IN_FLIGHT = 1,  // landing pose of the swing in flight
    NEXT_SWING = 2        // landing pose of the foot's next swing (the foot is still in contact)
  };
  bool valid{false};
  Kind kind{Kind::STANCE};
  double x{0.0};
  double y{0.0};
  double z{0.0};           // [m] contact frame origin
  double yaw{0.0};         // [rad] contact frame yaw about the world z axis
  bool yawPlanned{false};  // the yaw is planned (heading model); otherwise it holds the measured foot yaw
};

/** Corners of the contact patch polygon in the contact frame (x forward, y left), in order around the polygon. */
using ContactPatchCorners = std::vector<std::array<double, 2>>;

/** A generic foot outline (0.2 m x 0.1 m) for contact points without a configured patch. */
ContactPatchCorners defaultContactPatchCorners();

/**
 * World-frame corners of `corners` placed at `patch`: rotated by its yaw about z, translated to its origin and lifted
 * by `heightOffset` above its z.
 */
std::vector<std::array<double, 3>> contactPatchWorldCorners(const TargetContactPatch& patch,
                                                            const ContactPatchCorners& corners,
                                                            double heightOffset = 0.0);

/** How a patch is drawn. */
struct ContactPatchStyle {
  std::array<float, 4> rgba{1.0f, 1.0f, 1.0f, 1.0f};
  bool fill{true};   // thin translucent slab over the bounds of the corners (exact for a rectangle)
  bool arrow{true};  // arrow along the patch's x axis from its origin, shows the yaw
  float emission{0.5f};
};

/**
 * Appends the decor geoms that draw `patch` to `scene`: the outline of the polygon as capsules between consecutive
 * corners, and the filled slab and the heading arrow according to `style`. Returns the number of geoms appended;
 * nothing is appended for an invalid patch, fewer than three corners, or when the scene has no room for all of them.
 */
int addContactPatchGeoms(mjvScene* scene,
                         const TargetContactPatch& patch,
                         const ContactPatchCorners& corners,
                         const ContactPatchStyle& style);

}  // namespace robot::mujoco_sim_interface
