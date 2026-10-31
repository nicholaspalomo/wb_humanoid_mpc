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

#include "mujoco_sim_interface/visualization/CenterOfMassVisualization.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/visualization/GroundMarkerGeoms.h"

namespace robot::mujoco_sim_interface {

namespace {
constexpr double kCenterOfMassMarkerRadius =
    0.03;  // [m] the sphere at the center of mass and the disc below it
constexpr double kShadowLineWidth =
    0.004;  // [m] the line from the center of mass down to the ground
}  // namespace

void CenterOfMassVisualization::addSceneGeoms(const VisualizationFrame& frame) {
  if (frame.model == nullptr || frame.state == nullptr ||
      frame.scene == nullptr)
    return;
  const RobotCentroidalState state =
      robotCentroidalState(frame.model, frame.state->data);
  if (!state.valid) return;
  constexpr MarkerColor kMarkerColor{
      .r = 0.2f, .g = 0.5f, .b = 1.0f, .a = 0.9f};  // blue
  constexpr MarkerColor kShadowColor{
      .r = 0.2f, .g = 0.5f, .b = 1.0f, .a = 0.5f};  // translucent blue
  addSphereGeom(frame.scene, state.posComRtWorldEwrtWorld,
                kCenterOfMassMarkerRadius, kMarkerColor);
  addVerticalLineGeom(frame.scene, state.posComRtWorldEwrtWorld,
                      /*groundHeight=*/0.0, kShadowLineWidth, kShadowColor);
  addGroundDiscGeom(frame.scene, state.posComRtWorldEwrtWorld[0],
                    state.posComRtWorldEwrtWorld[1], kCenterOfMassMarkerRadius,
                    kShadowColor);
}

}  // namespace robot::mujoco_sim_interface
