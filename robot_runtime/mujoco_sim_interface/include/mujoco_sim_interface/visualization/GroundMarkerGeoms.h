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

#include <mujoco/mujoco.h>

namespace robot::mujoco_sim_interface {

/** Colour of a ground marker. */
struct MarkerColor {
  float r{1.0f};
  float g{1.0f};
  float b{1.0f};
  float a{1.0f};
};

/**
 * Decor geoms shared by the centroidal markers of the viewer (centre of mass, ZMP, DCM): a flat disc on the ground plane
 * at (x, y), a sphere in space, and a thin vertical line between a point and its projection on the ground. Each call
 * appends one geom to the scene and returns false when the scene is full.
 */
bool addGroundDiscGeom(mjvScene* scene, double x, double y, double radius, const MarkerColor& color);
bool addSphereGeom(mjvScene* scene, const double position[3], double radius, const MarkerColor& color);
bool addVerticalLineGeom(mjvScene* scene, const double position[3], double groundHeight, double width, const MarkerColor& color);

}  // namespace robot::mujoco_sim_interface
