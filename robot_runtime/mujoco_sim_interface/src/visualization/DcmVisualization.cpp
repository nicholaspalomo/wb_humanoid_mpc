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

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/visualization/GroundMarkerGeoms.h"

namespace robot::mujoco_sim_interface {

void DcmVisualization::addSceneGeoms(const VisualizationFrame& frame) {
  if (frame.sim == nullptr || frame.state == nullptr || frame.scene == nullptr) return;
  const mjModel* model = frame.sim->getModel();
  const RobotCentroidalState state = robotCentroidalState(model, frame.state->data);
  if (!state.valid) return;
  const double gravity = std::fabs(model->opt.gravity[2]);
  double dcm[2];
  divergentComponentOfMotion(state.com, state.comVelocity, state.com[2], gravity, dcm);
  const MarkerColor green{0.1f, 0.9f, 0.3f, 0.9f};
  // The offset from the CoM's shadow to the DCM is the velocity term v / omega: a line on the ground makes it readable.
  if (frame.scene->ngeom < frame.scene->maxgeom) {
    mjvGeom* line = &frame.scene->geoms[frame.scene->ngeom++];
    std::memset(line, 0, sizeof(mjvGeom));
    const mjtNum from[3] = {state.com[0], state.com[1], 0.004};
    const mjtNum to[3] = {dcm[0], dcm[1], 0.004};
    mjv_connector(line, mjGEOM_CAPSULE, 0.004, from, to);
    line->rgba[0] = green.r;
    line->rgba[1] = green.g;
    line->rgba[2] = green.b;
    line->rgba[3] = 0.6f;
    line->category = mjCAT_DECOR;
    line->emission = 0.6f;
  }
  addGroundDiscGeom(frame.scene, dcm[0], dcm[1], 0.025, green);
}

}  // namespace robot::mujoco_sim_interface
