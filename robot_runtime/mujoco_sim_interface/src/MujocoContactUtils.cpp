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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/Dimensions.h"
#include "urdfdom/urdf_parser/urdf_parser.h"

#include "mujoco_sim_interface/MujocoUtils.h"

namespace robot::mujoco_sim_interface {

using ::ocs2::kHorizontalPositionDim;
using ::ocs2::kPositionDim;
using ::ocs2::kXAxisIndex;
using ::ocs2::kYAxisIndex;
using ::ocs2::kZAxisIndex;

namespace {

/// [m] The lowest pendulum height divergentComponentOfMotion() uses, so that a
/// center of mass at or below the ground still gives a finite natural
/// frequency.
constexpr double kMinPendulumHeight = 0.05;

/// True when `body` belongs to the robot whose root body is `robotRoot`: it
/// shares that root. The world body never does.
bool isRobotBody(const mjModel* absl_nonnull model, int body, int robotRoot) {
  return body != 0 && model->body_rootid[body] == robotRoot;
}

}  // namespace

absl::StatusOr<MjDataPtr> makeMjData(const mjModel* absl_nonnull model) {
  MjDataPtr data(mj_makeData(model));
  if (data == nullptr) {
    return absl::InternalError(
        "MuJoCo could not create the simulation data of the model (mj_makeData "
        "returned null).");
  }
  return data;
}

MjState::MjState(const mjModel* absl_nonnull model)
    : model(model),
      data(ABSL_DIE_IF_NULL(mj_makeData(model))),
      modelFields(model) {}

// LINT.IfChange(mj_state_fields)
MjState::MjState(const MjState& other)
    : model(other.model),
      timestamp(other.timestamp),
      data(other.model != nullptr ? ABSL_DIE_IF_NULL(mj_makeData(other.model))
                                  : nullptr),
      metrics(other.metrics),
      modelFields(other.modelFields),
      push(other.push) {
  if (data != nullptr && other.data != nullptr)
    mj_copyData(data, model, other.data);
}

MjState& MjState::operator=(const MjState& other) {
  if (this != &other) {
    if (data != nullptr) mj_deleteData(data);
    model = other.model;
    timestamp = other.timestamp;
    data = other.model != nullptr ? ABSL_DIE_IF_NULL(mj_makeData(other.model))
                                  : nullptr;
    metrics = other.metrics;
    modelFields = other.modelFields;
    push = other.push;
    if (data != nullptr && other.data != nullptr)
      mj_copyData(data, model, other.data);
  }
  return *this;
}

MjState::MjState(MjState&& other) noexcept
    : model(other.model),
      timestamp(other.timestamp),
      data(other.data),
      metrics(other.metrics),
      modelFields(std::move(other.modelFields)),
      push(other.push) {
  other.data = nullptr;
  other.model = nullptr;
  other.modelFields = MjRuntimeModelFields();
}

MjState& MjState::operator=(MjState&& other) noexcept {
  if (this != &other) {
    if (data != nullptr) mj_deleteData(data);
    model = other.model;
    timestamp = other.timestamp;
    data = other.data;
    metrics = other.metrics;
    modelFields = std::move(other.modelFields);
    push = other.push;
    other.data = nullptr;
    other.model = nullptr;
    other.modelFields = MjRuntimeModelFields();
  }
  return *this;
}
// LINT.ThenChange(//robot_runtime/mujoco_sim_interface/include/mujoco_sim_interface/MujocoUtils.h:mj_state_fields)

MjState::~MjState() {
  if (data != nullptr) mj_deleteData(data);
}

void ContactTimeline::append(const ContactTimelineSample& sample) {
  if (!samples_.empty() && sample.time < samples_.back().time) {
    samples_.clear();  // the simulation was reset
  }
  samples_.push_back(sample);
  const double oldest = sample.time - window_;
  while (!samples_.empty() && samples_.front().time < oldest) {
    samples_.pop_front();
  }
}

bool isInBodySubtree(const mjModel* absl_nonnull model,
                     int bodyId,
                     int ancestorId) {
  if (bodyId < 0 || ancestorId < 0) return false;
  int body = bodyId;
  while (true) {
    if (body == ancestorId) return true;
    if (body <= 0)
      return false;  // reached the world body without meeting the ancestor
    body = model->body_parentid[body];
  }
}

std::vector<int> resolveContactBodies(
    const mjModel* absl_nonnull model,
    const std::string& urdfPath,
    const std::vector<std::string>& contactFrameNames,
    const std::vector<std::string>& contactParentJointNames,
    std::vector<std::string>* absl_nullable errors) {
  std::vector<int> bodyIds(contactFrameNames.size(), -1);

  urdf::ModelInterfaceSharedPtr urdfModel;
  const std::function<void()> loadUrdf = [&]() {
    if (urdfModel) return;
    std::ifstream file(urdfPath);
    if (file) {
      std::stringstream content;
      content << file.rdbuf();
      urdfModel = urdf::parseURDF(content.str());
    }
  };

  for (size_t i = 0; i < contactFrameNames.size(); ++i) {
    // 1. The body driven by the joint that carries the contact frame.
    if (i < contactParentJointNames.size() &&
        !contactParentJointNames[i].empty()) {
      const int jointId =
          mj_name2id(model, mjOBJ_JOINT, contactParentJointNames[i].c_str());
      if (jointId >= 0) {
        bodyIds[i] = model->jnt_bodyid[jointId];
        continue;
      }
      if (errors != nullptr) {
        errors->push_back(absl::StrCat(contactFrameNames[i],
                                       ": the parent joint '",
                                       contactParentJointNames[i],
                                       "' is not a joint of the MuJoCo model; "
                                       "falling back to the frame name"));
      }
    }
    // 2. A body named like the frame, 3. the URDF walked up through fixed
    // joints.
    std::string linkName = contactFrameNames[i];
    int bodyId = mj_name2id(model, mjOBJ_BODY, linkName.c_str());
    if (bodyId < 0) loadUrdf();
    bool reported = false;
    // Walk up through fixed joints: the MJCF conversion merges fixed-joint
    // children into their parent body.
    while (bodyId < 0 && urdfModel) {
      const urdf::LinkConstSharedPtr link = urdfModel->getLink(linkName);
      if (!link || !link->parent_joint) break;
      if (link->parent_joint->type != urdf::Joint::FIXED) {
        if (errors != nullptr) {
          errors->push_back(absl::StrCat(
              contactFrameNames[i], ": link '", linkName,
              "' hangs on the movable joint '", link->parent_joint->name,
              "' but is not a body of the MuJoCo model"));
        }
        reported = true;
        break;
      }
      linkName = link->parent_joint->parent_link_name;
      bodyId = mj_name2id(model, mjOBJ_BODY, linkName.c_str());
    }
    if (bodyId < 0 && !reported && errors != nullptr) {
      errors->push_back(absl::StrCat(
          contactFrameNames[i],
          ": no MuJoCo body found for the frame or its fixed-joint ancestors",
          urdfModel ? std::string()
                    : absl::StrCat(" (the URDF could not be parsed: ", urdfPath,
                                   ")")));
    }
    bodyIds[i] = bodyId;
  }
  return bodyIds;
}

uint32_t groundTruthContactMask(const mjModel* absl_nonnull model,
                                const mjData* absl_nonnull data,
                                const std::vector<int>& contactBodyIds,
                                double minForceMjcontactEwrtMjcontactX,
                                int ignoreBodyId) {
  uint32_t mask = 0;
  const size_t numContacts =
      std::min<size_t>(contactBodyIds.size(), kContactMaskCapacity);
  for (int c = 0; c < data->ncon; ++c) {
    const mjContact& contact = data->contact[c];
    if (contact.exclude != 0 || contact.efc_address < 0)
      continue;  // detected but not active in the constraint solver
    if (contact.geom[0] < 0 || contact.geom[1] < 0)
      continue;  // flex contacts carry no geoms
    mjtNum wrenchMjcontactEwrtMjcontact[kMjSpatialVectorSize];
    mj_contactForce(model, data, c,
                    wrenchMjcontactEwrtMjcontact);  // [0] is the normal force
    if (wrenchMjcontactEwrtMjcontact[0] < minForceMjcontactEwrtMjcontactX)
      continue;
    const int body1 = model->geom_bodyid[contact.geom[0]];
    const int body2 = model->geom_bodyid[contact.geom[1]];
    // Neither the robot nor the ground: a thrown ball touching a foot says
    // nothing about whether that foot is down.
    if (ignoreBodyId >= 0 && (body1 == ignoreBodyId || body2 == ignoreBodyId))
      continue;
    for (size_t i = 0; i < numContacts; ++i) {
      const int contactBody = contactBodyIds[i];
      if (contactBody < 0) continue;
      const int robotRoot = model->body_rootid[contactBody];
      // A body belongs to the robot when it shares the contact body's root; the
      // world body never does. (For a robot welded to the world every static
      // body counts as the robot, a limitation accepted for walking robots.)
      const bool oneIsContact = isInBodySubtree(model, body1, contactBody);
      const bool twoIsContact = isInBodySubtree(model, body2, contactBody);
      if ((oneIsContact && !isRobotBody(model, body2, robotRoot)) ||
          (twoIsContact && !isRobotBody(model, body1, robotRoot))) {
        mask |= (1u << i);
      }
    }
  }
  return mask;
}

RobotCentroidalState robotCentroidalState(const mjModel* absl_nullable model,
                                          const mjData* absl_nullable data) {
  RobotCentroidalState state;
  if (model == nullptr || data == nullptr) return state;
  for (int body = 1; body < model->nbody && state.rootBodyId < 0; ++body) {
    if (model->body_jntnum[body] > 0 &&
        model->jnt_type[model->body_jntadr[body]] == mjJNT_FREE)
      state.rootBodyId = body;
  }
  if (state.rootBodyId < 0) return state;
  state.mass = model->body_subtreemass[state.rootBodyId];
  if (state.mass <= 0.0) return state;
  for (int axis = 0; axis < kPositionDim; ++axis)
    state.posComRtWorldEwrtWorld[axis] =
        data->subtree_com[kPositionDim * state.rootBodyId + axis];
  // mj_forward does not fill subtree_linvel; the mass-weighted mean of the body
  // center velocities is the same thing.
  double linmomComRtWorldEwrtWorld[kPositionDim] = {0.0, 0.0, 0.0};
  for (int body = state.rootBodyId; body < model->nbody; ++body) {
    if (!isInBodySubtree(model, body, state.rootBodyId)) continue;
    const double mass = model->body_mass[body];
    if (mass <= 0.0) continue;
    // MuJoCo's motion vector, angular first: the velocity of the body's
    // inertial frame (its center of mass), in world-aligned axes.
    mjtNum mjTwistLinkRtWorldEwrtWorld[kMjSpatialVectorSize];
    mj_objectVelocity(model, data, mjOBJ_BODY, body,
                      mjTwistLinkRtWorldEwrtWorld, /*flg_local=*/0);
    for (int axis = 0; axis < kPositionDim; ++axis)
      linmomComRtWorldEwrtWorld[axis] +=
          mass * mjTwistLinkRtWorldEwrtWorld[kMjMotionLinearOffset + axis];
  }
  for (int axis = 0; axis < kPositionDim; ++axis)
    state.linvelComRtWorldEwrtWorld[axis] =
        linmomComRtWorldEwrtWorld[axis] / state.mass;
  state.valid = true;
  return state;
}

GroundReaction groundReaction(const mjModel* absl_nullable model,
                              const mjData* absl_nullable data,
                              int rootBodyId,
                              double minForceWorldEwrtWorldZ,
                              int ignoreBodyId) {
  GroundReaction reaction;
  if (model == nullptr || data == nullptr || rootBodyId < 0) return reaction;
  for (int c = 0; c < data->ncon; ++c) {
    const mjContact& contact = data->contact[c];
    if (contact.exclude != 0 || contact.efc_address < 0) continue;
    if (contact.geom[0] < 0 || contact.geom[1] < 0) continue;
    const int body1 = model->geom_bodyid[contact.geom[0]];
    const int body2 = model->geom_bodyid[contact.geom[1]];
    if (ignoreBodyId >= 0 && (body1 == ignoreBodyId || body2 == ignoreBodyId))
      continue;  // a thrown ball, not the ground
    const bool oneIsRobot = isInBodySubtree(model, body1, rootBodyId);
    const bool twoIsRobot = isInBodySubtree(model, body2, rootBodyId);
    if (oneIsRobot == twoIsRobot)
      continue;  // self-contact, or a contact that does not involve the robot
    mjtNum wrenchMjcontactEwrtMjcontact[kMjSpatialVectorSize];
    mj_contactForce(model, data, c, wrenchMjcontactEwrtMjcontact);
    // The contact frame is stored row-major (rows = the frame axes, the normal
    // first, pointing from geom[0] to geom[1]); the normal force is
    // non-negative along it, so the world-frame force below is the one acting
    // on geom[1].
    double forceMjcontactEwrtWorld[kPositionDim];
    for (int axis = 0; axis < kPositionDim; ++axis) {
      forceMjcontactEwrtWorld[axis] =
          wrenchMjcontactEwrtMjcontact[kMjContactNormalRow] *
              contact.frame[kMjContactNormalRow * kPositionDim + axis] +
          wrenchMjcontactEwrtMjcontact[kMjContactFirstTangentRow] *
              contact.frame[kMjContactFirstTangentRow * kPositionDim + axis] +
          wrenchMjcontactEwrtMjcontact[kMjContactSecondTangentRow] *
              contact.frame[kMjContactSecondTangentRow * kPositionDim + axis];
    }
    if (oneIsRobot) {
      for (double& f : forceMjcontactEwrtWorld)
        f = -f;  // the robot is geom[0]: it receives the reaction
    }
    for (int axis = 0; axis < kPositionDim; ++axis)
      reaction.forceWorldEwrtWorld[axis] += forceMjcontactEwrtWorld[axis];
    // The moment about the origin, r x f.
    const mjtNum* absl_nonnull posMjcontactRtWorldEwrtWorld =
        contact.pos;  // NOLINT(frame-naming): MuJoCo's mjContact::pos
    reaction.torqueWorldEwrtWorld[kXAxisIndex] +=
        posMjcontactRtWorldEwrtWorld[kYAxisIndex] *
            forceMjcontactEwrtWorld[kZAxisIndex] -
        posMjcontactRtWorldEwrtWorld[kZAxisIndex] *
            forceMjcontactEwrtWorld[kYAxisIndex];
    reaction.torqueWorldEwrtWorld[kYAxisIndex] +=
        posMjcontactRtWorldEwrtWorld[kZAxisIndex] *
            forceMjcontactEwrtWorld[kXAxisIndex] -
        posMjcontactRtWorldEwrtWorld[kXAxisIndex] *
            forceMjcontactEwrtWorld[kZAxisIndex];
    reaction.torqueWorldEwrtWorld[kZAxisIndex] +=
        posMjcontactRtWorldEwrtWorld[kXAxisIndex] *
            forceMjcontactEwrtWorld[kYAxisIndex] -
        posMjcontactRtWorldEwrtWorld[kYAxisIndex] *
            forceMjcontactEwrtWorld[kXAxisIndex];
  }
  if (reaction.forceWorldEwrtWorld[kZAxisIndex] <= minForceWorldEwrtWorldZ)
    return reaction;
  reaction.posZmpRtWorldEwrtWorldXy[kXAxisIndex] =
      -reaction.torqueWorldEwrtWorld[kYAxisIndex] /
      reaction.forceWorldEwrtWorld[kZAxisIndex];
  reaction.posZmpRtWorldEwrtWorldXy[kYAxisIndex] =
      reaction.torqueWorldEwrtWorld[kXAxisIndex] /
      reaction.forceWorldEwrtWorld[kZAxisIndex];
  reaction.valid = true;
  return reaction;
}

void divergentComponentOfMotion(
    const double posComRtWorldEwrtWorld[absl_nonnull kPositionDim],
    const double linvelComRtWorldEwrtWorld[absl_nonnull kPositionDim],
    double height,
    double gravity,
    double posDcmRtWorldEwrtWorldXy[absl_nonnull kHorizontalPositionDim]) {
  const double omega =
      std::sqrt(std::max(gravity, 0.0) / std::max(height, kMinPendulumHeight));
  for (int axis = 0; axis < kHorizontalPositionDim; ++axis)
    posDcmRtWorldEwrtWorldXy[axis] =
        posComRtWorldEwrtWorld[axis] +
        (omega > 0.0 ? linvelComRtWorldEwrtWorld[axis] / omega : 0.0);
}

std::array<mjtNum, kPositionDim> computeMeasForceFootEwrtWorld(
    const mjModel* absl_nonnull model,
    const mjData* absl_nonnull data,
    int sensorId) {
  const mjtNum* absl_nonnull reading =
      data->sensordata + model->sensor_adr[sensorId];
  // The site's world orientation, row-major: it maps the site frame into the
  // world frame.
  const mjtNum* absl_nonnull siteToWorld =
      data->site_xmat + kMjMatrixSize * model->sensor_objid[sensorId];
  std::array<mjtNum, kPositionDim> measForceFootEwrtWorld{};
  mju_mulMatVec3(measForceFootEwrtWorld.data(), siteToWorld, reading);
  // MuJoCo's reading is the force on the parent's side; the foot carries its
  // opposite.
  for (mjtNum& component : measForceFootEwrtWorld) component = -component;
  return measForceFootEwrtWorld;
}

bool isFootForceSensor(  // NOLINT(frame-naming): a sensor's type, not a force
    const mjModel* absl_nonnull model,
    int sensorId) {
  return sensorId >= 0 && sensorId < model->nsensor &&
         model->sensor_type[sensorId] == mjSENS_FORCE &&
         model->sensor_objtype[sensorId] == mjOBJ_SITE;
}

VerticalTwist verticalTwistFromQuaternion(double w, double z) {
  // The hemisphere w > 0, or w = 0 <= z, as BaseOrientation's
  // headingFromTwist() takes it. |w| keeps a negative zero from turning
  // atan2(+-0, -0) into +-pi.
  const bool flip = w < 0.0 || (w == 0.0 && z < 0.0);
  const double sign = flip ? -1.0 : 1.0;
  // atan2(0, 0) = 0: an exactly upside-down attitude has the identity twist,
  // and the half angle is continuous above it.
  const double halfHeading = std::atan2(sign * z, std::abs(sign * w));
  return {.w = std::cos(halfHeading), .z = std::sin(halfHeading)};
}

double yawOf(double w, double z) {
  const VerticalTwist quatHeadingRtWorld = verticalTwistFromQuaternion(w, z);
  return 2.0 * std::atan2(quatHeadingRtWorld.z, quatHeadingRtWorld.w);
}

}  // namespace robot::mujoco_sim_interface
