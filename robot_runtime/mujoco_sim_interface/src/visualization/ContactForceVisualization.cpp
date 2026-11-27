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

#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/visualization/MujocoVisualization.h"

namespace robot::mujoco_sim_interface {

using ::ocs2::kPositionDim;
using ::ocs2::kXAxisIndex;
using ::ocs2::kYAxisIndex;
using ::ocs2::kZAxisIndex;

namespace {
// Where each row of mjContact::frame (the contact normal and its two tangents
// in the world frame) starts; the force of mj_contactForce is along these axes.
constexpr int kNormalRowStartIndex = kMjContactNormalRow * kPositionDim;
constexpr int kFirstTangentRowStartIndex =
    kMjContactFirstTangentRow * kPositionDim;
constexpr int kSecondTangentRowStartIndex =
    kMjContactSecondTangentRow * kPositionDim;
// The arrow's length grows with the force.
constexpr double kMinArrowLength = 0.02;         // [m]
constexpr double kArrowLengthPerNewton = 0.002;  // [m/N]: 500 N draw 1 m
constexpr double kArrowWidth = 0.015;            // [m]
constexpr float kArrowAlpha = 0.8f;
constexpr float kArrowEmission = 0.8f;
constexpr double kMinDrawnNormForceMjcontactEwrtWorld =
    1.0;  // [N] a weaker contact draws no arrow

// Whether `bodyId` is the robot's: in the subtree of `robotRootBodyId`, or, for
// a model without a free joint (a fixed base, robotRootBodyId < 0), any body
// but the world.
bool isRobotBody(const mjModel* absl_nonnull model,
                 int bodyId,
                 int robotRootBodyId) {
  if (robotRootBodyId < 0) return bodyId != 0;
  return isInBodySubtree(model, bodyId, robotRootBodyId);
}
}  // namespace

void ContactForceVisualization::addSceneGeoms(const VisualizationFrame& frame) {
  if (frame.model == nullptr || frame.state == nullptr ||
      frame.scene == nullptr) {
    return;
  }
  const mjModel* absl_nonnull model = frame.model;
  const mjData* absl_nullable data = frame.state->data;
  mjvScene& scene = *frame.scene;
  if (data == nullptr) return;
  const int robotRootBodyId = robotCentroidalState(model, data).rootBodyId;

  for (int i = 0; i < data->ncon; ++i) {
    const mjContact* absl_nonnull contact = &data->contact[i];
    if (contact->exclude != 0 || contact->efc_address < 0) {
      continue;  // only active contacts
    }
    // Contacts with the robot on one side only: not the ball on the floor, nor
    // the robot against itself.
    const bool firstIsRobot = isRobotBody(
        model, model->geom_bodyid[contact->geom[0]], robotRootBodyId);
    const bool secondIsRobot = isRobotBody(
        model, model->geom_bodyid[contact->geom[1]], robotRootBodyId);
    if (firstIsRobot == secondIsRobot) continue;

    mjtNum wrenchMjcontactEwrtMjcontact[kMjSpatialVectorSize];
    mj_contactForce(model, data, i, wrenchMjcontactEwrtMjcontact);

    // Force in the world frame. mjContact::frame stores the rotation of
    // mjcontact relative to the world column-major: each run of three is an
    // axis of mjcontact in world axes, a column of the rotmat.
    const mjtNum* absl_nonnull rotmatMjcontactRtWorld = contact->frame;
    const double forceMjcontactEwrtMjcontactX =
        wrenchMjcontactEwrtMjcontact[kMjContactNormalRow];
    const double forceMjcontactEwrtMjcontactY =
        wrenchMjcontactEwrtMjcontact[kMjContactFirstTangentRow];
    const double forceMjcontactEwrtMjcontactZ =
        wrenchMjcontactEwrtMjcontact[kMjContactSecondTangentRow];
    double forceMjcontactEwrtWorldX =
        forceMjcontactEwrtMjcontactX *
            rotmatMjcontactRtWorld[kNormalRowStartIndex + kXAxisIndex] +
        forceMjcontactEwrtMjcontactY *
            rotmatMjcontactRtWorld[kFirstTangentRowStartIndex + kXAxisIndex] +
        forceMjcontactEwrtMjcontactZ *
            rotmatMjcontactRtWorld[kSecondTangentRowStartIndex + kXAxisIndex];
    double forceMjcontactEwrtWorldY =
        forceMjcontactEwrtMjcontactX *
            rotmatMjcontactRtWorld[kNormalRowStartIndex + kYAxisIndex] +
        forceMjcontactEwrtMjcontactY *
            rotmatMjcontactRtWorld[kFirstTangentRowStartIndex + kYAxisIndex] +
        forceMjcontactEwrtMjcontactZ *
            rotmatMjcontactRtWorld[kSecondTangentRowStartIndex + kYAxisIndex];
    double forceMjcontactEwrtWorldZ =
        forceMjcontactEwrtMjcontactX *
            rotmatMjcontactRtWorld[kNormalRowStartIndex + kZAxisIndex] +
        forceMjcontactEwrtMjcontactY *
            rotmatMjcontactRtWorld[kFirstTangentRowStartIndex + kZAxisIndex] +
        forceMjcontactEwrtMjcontactZ *
            rotmatMjcontactRtWorld[kSecondTangentRowStartIndex + kZAxisIndex];

    // In the world frame this is the force on geom[1] from geom[0]: the contact
    // normal points from geom[0] to geom[1], and mjData::cfrc_ext adds it to
    // geom[1]'s body and subtracts it from geom[0]'s. It is drawn as it is when
    // geom[1] is the robot's, and negated when geom[0] is: the arrow is always
    // the force on the robot, whichever geom MuJoCo lists first (it orders a
    // pair by geom type, so a ball, a sphere, comes before the robot's capsules
    // and boxes).
    if (firstIsRobot) {
      forceMjcontactEwrtWorldX = -forceMjcontactEwrtWorldX;
      forceMjcontactEwrtWorldY = -forceMjcontactEwrtWorldY;
      forceMjcontactEwrtWorldZ = -forceMjcontactEwrtWorldZ;
    }

    const double normForceMjcontactEwrtWorld =
        std::sqrt(forceMjcontactEwrtWorldX * forceMjcontactEwrtWorldX +
                  forceMjcontactEwrtWorldY * forceMjcontactEwrtWorldY +
                  forceMjcontactEwrtWorldZ * forceMjcontactEwrtWorldZ);
    if (normForceMjcontactEwrtWorld < kMinDrawnNormForceMjcontactEwrtWorld) {
      continue;
    }
    if (scene.ngeom >= scene.maxgeom) return;

    mjvGeom* absl_nonnull arrow = &scene.geoms[scene.ngeom++];
    std::memset(arrow, 0, sizeof(mjvGeom));

    const double scale =
        kMinArrowLength + kArrowLengthPerNewton * normForceMjcontactEwrtWorld;

    const mjtNum* absl_nonnull posMjcontactRtWorldEwrtWorld =
        contact->pos;  // NOLINT(frame-naming): MuJoCo's mjContact::pos
    const mjtNum posArrowRtWorldEwrtWorld[kPositionDim] = {
        posMjcontactRtWorldEwrtWorld[kXAxisIndex],
        posMjcontactRtWorldEwrtWorld[kYAxisIndex],
        posMjcontactRtWorldEwrtWorld[kZAxisIndex]};
    const mjtNum posArrowheadRtWorldEwrtWorld[kPositionDim] = {
        posMjcontactRtWorldEwrtWorld[0] +
            scale * (forceMjcontactEwrtWorldX / normForceMjcontactEwrtWorld),
        posMjcontactRtWorldEwrtWorld[1] +
            scale * (forceMjcontactEwrtWorldY / normForceMjcontactEwrtWorld),
        posMjcontactRtWorldEwrtWorld[kZAxisIndex] +
            scale * (forceMjcontactEwrtWorldZ / normForceMjcontactEwrtWorld)};
    mjv_connector(arrow, mjGEOM_ARROW, kArrowWidth, posArrowRtWorldEwrtWorld,
                  posArrowheadRtWorldEwrtWorld);

    arrow->rgba[kRedChannel] = 1.0f;
    arrow->rgba[kGreenChannel] = 0.0f;
    arrow->rgba[kBlueChannel] = 0.0f;
    arrow->rgba[kAlphaChannel] = kArrowAlpha;
    arrow->category = mjCAT_DECOR;
    arrow->emission = kArrowEmission;
  }
}

}  // namespace robot::mujoco_sim_interface
