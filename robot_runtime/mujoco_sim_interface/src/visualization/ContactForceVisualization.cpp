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

#include "mujoco_sim_interface/visualization/ContactForceVisualization.h"

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
// Where each row of mjContact::frame (the contact normal and its two tangents in the world frame) starts; the force of
// mj_contactForce is along these axes.
constexpr int kNormalRowOffset = kMjContactNormalRow * kPositionDim;
constexpr int kFirstTangentRowOffset = kMjContactFirstTangentRow * kPositionDim;
constexpr int kSecondTangentRowOffset = kMjContactSecondTangentRow * kPositionDim;
// The arrow's length grows with the force.
constexpr double kMinArrowLength = 0.02;         // [m]
constexpr double kArrowLengthPerNewton = 0.002;  // [m/N]: 500 N draw 1 m
constexpr double kArrowWidth = 0.015;            // [m]
constexpr float kArrowAlpha = 0.8f;
constexpr float kArrowEmission = 0.8f;
constexpr double kMinDrawnForce = 1.0;  // [N] a weaker contact draws no arrow

// Whether `bodyId` is the robot's: in the subtree of `robotRootBodyId`, or, for a model without a free joint (a fixed
// base, robotRootBodyId < 0), any body but the world.
bool isRobotBody(const mjModel* absl_nonnull model, int bodyId, int robotRootBodyId) {
  if (robotRootBodyId < 0) return bodyId != 0;
  return isInBodySubtree(model, bodyId, robotRootBodyId);
}
}  // namespace

void ContactForceVisualization::addSceneGeoms(const VisualizationFrame& frame) {
  if (frame.model == nullptr || frame.state == nullptr || frame.scene == nullptr) return;
  const mjModel* absl_nonnull model = frame.model;
  const mjData* absl_nullable data = frame.state->data;
  mjvScene& scene = *frame.scene;
  if (data == nullptr) return;
  const int robotRootBodyId = robotCentroidalState(model, data).rootBodyId;

  for (int i = 0; i < data->ncon; ++i) {
    const mjContact* absl_nonnull contact = &data->contact[i];
    if (contact->exclude != 0 || contact->efc_address < 0) continue;  // only active contacts
    // Contacts with the robot on one side only: not the ball on the floor, nor the robot against itself.
    const bool firstIsRobot = isRobotBody(model, model->geom_bodyid[contact->geom[0]], robotRootBodyId);
    const bool secondIsRobot = isRobotBody(model, model->geom_bodyid[contact->geom[1]], robotRootBodyId);
    if (firstIsRobot == secondIsRobot) continue;

    mjtNum force[kMjSpatialVectorSize];
    mj_contactForce(model, data, i, force);

    // Force in the world frame
    const mjtNum* absl_nonnull normal = &contact->frame[kNormalRowOffset];
    const mjtNum* absl_nonnull firstTangent = &contact->frame[kFirstTangentRowOffset];
    const mjtNum* absl_nonnull secondTangent = &contact->frame[kSecondTangentRowOffset];
    const double normalForce = force[kMjContactNormalRow];
    const double firstTangentForce = force[kMjContactFirstTangentRow];
    const double secondTangentForce = force[kMjContactSecondTangentRow];
    double fx =
        normalForce * normal[kXAxisIndex] + firstTangentForce * firstTangent[kXAxisIndex] + secondTangentForce * secondTangent[kXAxisIndex];
    double fy =
        normalForce * normal[kYAxisIndex] + firstTangentForce * firstTangent[kYAxisIndex] + secondTangentForce * secondTangent[kYAxisIndex];
    double fz =
        normalForce * normal[kZAxisIndex] + firstTangentForce * firstTangent[kZAxisIndex] + secondTangentForce * secondTangent[kZAxisIndex];

    // In the world frame this is the force on geom[1] from geom[0]: the contact normal points from geom[0] to geom[1],
    // and mjData::cfrc_ext adds it to geom[1]'s body and subtracts it from geom[0]'s. It is drawn as it is when geom[1]
    // is the robot's, and negated when geom[0] is: the arrow is always the force on the robot, whichever geom MuJoCo
    // lists first (it orders a pair by geom type, so a ball, a sphere, comes before the robot's capsules and boxes).
    if (firstIsRobot) {
      fx = -fx;
      fy = -fy;
      fz = -fz;
    }

    const double magnitude = std::sqrt(fx * fx + fy * fy + fz * fz);
    if (magnitude < kMinDrawnForce) continue;
    if (scene.ngeom >= scene.maxgeom) return;

    mjvGeom* absl_nonnull arrow = &scene.geoms[scene.ngeom++];
    std::memset(arrow, 0, sizeof(mjvGeom));

    const double scale = kMinArrowLength + kArrowLengthPerNewton * magnitude;

    const mjtNum from[kPositionDim] = {contact->pos[kXAxisIndex], contact->pos[kYAxisIndex], contact->pos[kZAxisIndex]};
    const mjtNum to[kPositionDim] = {contact->pos[0] + scale * (fx / magnitude), contact->pos[1] + scale * (fy / magnitude),
                                     contact->pos[kZAxisIndex] + scale * (fz / magnitude)};
    mjv_connector(arrow, mjGEOM_ARROW, kArrowWidth, from, to);

    arrow->rgba[kRedChannel] = 1.0f;
    arrow->rgba[kGreenChannel] = 0.0f;
    arrow->rgba[kBlueChannel] = 0.0f;
    arrow->rgba[kAlphaChannel] = kArrowAlpha;
    arrow->category = mjCAT_DECOR;
    arrow->emission = kArrowEmission;
  }
}

}  // namespace robot::mujoco_sim_interface
