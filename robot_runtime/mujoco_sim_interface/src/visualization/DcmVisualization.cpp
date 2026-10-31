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

#include "mujoco_sim_interface/visualization/DcmVisualization.h"

#include <cmath>
#include <cstring>

#include "absl/base/nullability.h"
#include "ocs2_core/Dimensions.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/visualization/GroundMarkerGeoms.h"

namespace robot::mujoco_sim_interface {

using ::ocs2::kHorizontalPositionDim;
using ::ocs2::kPositionDim;
using ::ocs2::kZAxisIndex;

namespace {
constexpr double kDcmMarkerRadius = 0.025;  // [m] the disc at the DCM
// The line on the ground from the shadow of the center of mass to the DCM.
constexpr double kDcmLineHeight = 0.004;  // [m] above the ground plane
constexpr double kDcmLineWidth = 0.004;   // [m]
constexpr float kDcmLineAlpha = 0.6f;
constexpr float kDcmLineEmission = 0.6f;
}  // namespace

void DcmVisualization::addSceneGeoms(const VisualizationFrame& frame) {
  if (frame.model == nullptr || frame.state == nullptr ||
      frame.scene == nullptr)
    return;
  const mjModel* absl_nonnull model = frame.model;
  const RobotCentroidalState state =
      robotCentroidalState(model, frame.state->data);
  if (!state.valid) return;
  const double gravity = std::fabs(model->opt.gravity[kZAxisIndex]);
  double posDcmRtWorldEwrtWorldXy[kHorizontalPositionDim];
  divergentComponentOfMotion(state.posComRtWorldEwrtWorld,
                             state.linvelComRtWorldEwrtWorld,
                             state.posComRtWorldEwrtWorld[kZAxisIndex], gravity,
                             posDcmRtWorldEwrtWorldXy);
  constexpr MarkerColor kDcmColor{
      .r = 0.1f, .g = 0.9f, .b = 0.3f, .a = 0.9f};  // green
  // The offset from the CoM's shadow to the DCM is the velocity term v / omega:
  // a line on the ground makes it readable.
  if (frame.scene->ngeom < frame.scene->maxgeom) {
    mjvGeom* absl_nonnull line = &frame.scene->geoms[frame.scene->ngeom++];
    std::memset(line, 0, sizeof(mjvGeom));
    const mjtNum from[kPositionDim] = {state.posComRtWorldEwrtWorld[0],
                                       state.posComRtWorldEwrtWorld[1],
                                       kDcmLineHeight};
    const mjtNum to[kPositionDim] = {posDcmRtWorldEwrtWorldXy[0],
                                     posDcmRtWorldEwrtWorldXy[1],
                                     kDcmLineHeight};
    mjv_connector(line, mjGEOM_CAPSULE, kDcmLineWidth, from, to);
    line->rgba[kRedChannel] = kDcmColor.r;
    line->rgba[kGreenChannel] = kDcmColor.g;
    line->rgba[kBlueChannel] = kDcmColor.b;
    line->rgba[kAlphaChannel] = kDcmLineAlpha;
    line->category = mjCAT_DECOR;
    line->emission = kDcmLineEmission;
  }
  addGroundDiscGeom(frame.scene, posDcmRtWorldEwrtWorldXy[0],
                    posDcmRtWorldEwrtWorldXy[1], kDcmMarkerRadius, kDcmColor);
}

}  // namespace robot::mujoco_sim_interface
