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

#include "mujoco_sim_interface/visualization/ExternalForceVisualization.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "absl/base/nullability.h"
#include "ocs2_core/Dimensions.h"

#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "mujoco_sim_interface/MujocoUtils.h"

namespace robot::mujoco_sim_interface {

using ::ocs2::kPositionDim;
using ::ocs2::kXAxisIndex;
using ::ocs2::kYAxisIndex;
using ::ocs2::kZAxisIndex;

namespace {
constexpr double kMinDrawnForce =
    1.0e-6;  // [N] a smaller applied force draws no arrow
// The arrow of a force grows with it, up to kMaxArrowLength.
constexpr double kMinArrowLength = 0.1;         // [m]
constexpr double kArrowLengthPerNewton = 0.05;  // [m/N]
// The arrow of a held push grows with its impulse: 20 N s, a 0.45 kg ball at 44
// m/s, draws 1 m.
constexpr double kArrowLengthPerNewtonSecond = 0.05;  // [m/(N s)]
constexpr double kMaxArrowLength =
    1.5;  // [m] an impulse of one step's force would otherwise draw kilometers
constexpr double kArrowWidth = 0.005;  // [m]
constexpr float kArrowRed = 0.5f;      // the red of the yellow-green arrow

// Appends, while `scene` has room, an arrow of `length` from `origin` along
// `vector` (of `magnitude`).
void addArrow(const double* absl_nonnull origin,
              const double* absl_nonnull vector,
              double magnitude,
              double length,
              mjvScene& scene) {
  if (scene.ngeom >= scene.maxgeom) return;  // out of geom space
  mjvGeom* absl_nonnull arrow = &scene.geoms[scene.ngeom++];
  std::memset(arrow, 0, sizeof(mjvGeom));
  const mjtNum from[kPositionDim] = {origin[kXAxisIndex], origin[kYAxisIndex],
                                     origin[kZAxisIndex]};
  const mjtNum to[kPositionDim] = {
      origin[kXAxisIndex] + length * (vector[kXAxisIndex] / magnitude),
      origin[kYAxisIndex] + length * (vector[kYAxisIndex] / magnitude),
      origin[kZAxisIndex] + length * (vector[kZAxisIndex] / magnitude)};
  mjv_connector(arrow, mjGEOM_ARROW, kArrowWidth, from, to);
  arrow->rgba[kRedChannel] = kArrowRed;
  arrow->rgba[kGreenChannel] = 1.0f;
  arrow->rgba[kBlueChannel] = 0.0f;
  arrow->rgba[kAlphaChannel] = 1.0f;
  arrow->category = mjCAT_DECOR;
  arrow->emission = 1.0f;  // Makes it glow/bright
}

double norm(const double* absl_nonnull vector) {
  return std::sqrt(vector[kXAxisIndex] * vector[kXAxisIndex] +
                   vector[kYAxisIndex] * vector[kYAxisIndex] +
                   vector[kZAxisIndex] * vector[kZAxisIndex]);
}
}  // namespace

void ExternalForceVisualization::addSceneGeoms(
    const VisualizationFrame& frame) {
  if (frame.model == nullptr || frame.state == nullptr ||
      frame.scene == nullptr)
    return;
  addArrows(frame.model, *frame.state, *frame.scene);
}

void ExternalForceVisualization::addArrows(const mjModel* absl_nonnull model,
                                           const MjState& state,
                                           mjvScene& scene) {
  const mjData* absl_nullable data = state.data;
  if (data == nullptr) return;

  // The held push, at the body's center of mass, where MuJoCo applies
  // xfrc_applied (mj_xfrcAccumulate at xipos).
  const AppliedPush& push = state.push;
  const bool pushHeld = push.bodyId >= 0 && push.bodyId < model->nbody &&
                        data->time >= push.simTimeApplied &&
                        data->time - push.simTimeApplied < kPushHoldTime;
  if (pushHeld) {
    const double impulse = norm(push.impulseWorld.data());
    if (impulse >= kMinDrawnImpulse) {
      addArrow(&data->xipos[kPositionDim * push.bodyId],
               push.impulseWorld.data(), impulse,
               std::min(kMinArrowLength + kArrowLengthPerNewtonSecond * impulse,
                        kMaxArrowLength),
               scene);
    }
  }

  for (int body_id = 0; body_id < model->nbody; ++body_id) {
    // The held push is drawn above; its one-step force would be a second arrow
    // on the same body.
    if (pushHeld && body_id == push.bodyId) continue;
    const double* absl_nonnull force =
        &data->xfrc_applied[kMjSpatialVectorSize * body_id];
    const double magnitude = norm(force);
    if (magnitude < kMinDrawnForce) continue;
    addArrow(&data->xipos[kPositionDim * body_id], force, magnitude,
             std::min(kMinArrowLength + kArrowLengthPerNewton * magnitude,
                      kMaxArrowLength),
             scene);
  }
}

}  // namespace robot::mujoco_sim_interface
