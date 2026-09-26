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

#include "mujoco_sim_interface/MujocoUtils.h"

#include <urdfdom/urdf_parser/urdf_parser.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace robot::mujoco_sim_interface {

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

bool isInBodySubtree(const mjModel* model, int bodyId, int ancestorId) {
  if (bodyId < 0 || ancestorId < 0) return false;
  int body = bodyId;
  while (true) {
    if (body == ancestorId) return true;
    if (body <= 0) return false;  // reached the world body without meeting the ancestor
    body = model->body_parentid[body];
  }
}

std::vector<int> resolveContactBodies(const mjModel* model,
                                      const std::string& urdfPath,
                                      const std::vector<std::string>& contactFrameNames,
                                      const std::vector<std::string>& contactParentJointNames,
                                      std::vector<std::string>* errors) {
  std::vector<int> bodyIds(contactFrameNames.size(), -1);

  urdf::ModelInterfaceSharedPtr urdfModel;
  const auto loadUrdf = [&]() {
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
    if (i < contactParentJointNames.size() && !contactParentJointNames[i].empty()) {
      const int jointId = mj_name2id(model, mjOBJ_JOINT, contactParentJointNames[i].c_str());
      if (jointId >= 0) {
        bodyIds[i] = model->jnt_bodyid[jointId];
        continue;
      }
      if (errors) {
        errors->push_back(contactFrameNames[i] + ": the parent joint '" + contactParentJointNames[i] +
                          "' is not a joint of the MuJoCo model; falling back to the frame name");
      }
    }
    // 2. A body named like the frame, 3. the URDF walked up through fixed joints.
    std::string linkName = contactFrameNames[i];
    int bodyId = mj_name2id(model, mjOBJ_BODY, linkName.c_str());
    if (bodyId < 0) loadUrdf();
    bool reported = false;
    // Walk up through fixed joints: the MJCF conversion merges fixed-joint children into their parent body.
    while (bodyId < 0 && urdfModel) {
      const urdf::LinkConstSharedPtr link = urdfModel->getLink(linkName);
      if (!link || !link->parent_joint) break;
      if (link->parent_joint->type != urdf::Joint::FIXED) {
        if (errors) {
          errors->push_back(contactFrameNames[i] + ": link '" + linkName + "' hangs on the movable joint '" + link->parent_joint->name +
                            "' but is not a body of the MuJoCo model");
        }
        reported = true;
        break;
      }
      linkName = link->parent_joint->parent_link_name;
      bodyId = mj_name2id(model, mjOBJ_BODY, linkName.c_str());
    }
    if (bodyId < 0 && !reported && errors) {
      errors->push_back(contactFrameNames[i] + ": no MuJoCo body found for the frame or its fixed-joint ancestors" +
                        (urdfModel ? std::string() : " (the URDF could not be parsed: " + urdfPath + ")"));
    }
    bodyIds[i] = bodyId;
  }
  return bodyIds;
}

uint32_t groundTruthContactMask(
    const mjModel* model, const mjData* data, const std::vector<int>& contactBodyIds, double forceThreshold, int ignoreBodyId) {
  uint32_t mask = 0;
  const size_t numContacts = std::min<size_t>(contactBodyIds.size(), 32);
  for (int c = 0; c < data->ncon; ++c) {
    const mjContact& contact = data->contact[c];
    if (contact.exclude != 0 || contact.efc_address < 0) continue;  // detected but not active in the constraint solver
    if (contact.geom[0] < 0 || contact.geom[1] < 0) continue;       // flex contacts carry no geoms
    mjtNum force[6];
    mj_contactForce(model, data, c, force);  // contact frame: force[0] is the normal component
    if (force[0] < forceThreshold) continue;
    const int body1 = model->geom_bodyid[contact.geom[0]];
    const int body2 = model->geom_bodyid[contact.geom[1]];
    // Neither the robot nor the ground: a thrown ball touching a foot says nothing about whether that foot is down.
    if (ignoreBodyId >= 0 && (body1 == ignoreBodyId || body2 == ignoreBodyId)) continue;
    for (size_t i = 0; i < numContacts; ++i) {
      const int contactBody = contactBodyIds[i];
      if (contactBody < 0) continue;
      const int robotRoot = model->body_rootid[contactBody];
      // A body belongs to the robot when it shares the contact body's root; the world body never does. (For a robot
      // welded to the world every static body counts as the robot, a limitation accepted for walking robots.)
      const auto isRobot = [&](int body) { return body != 0 && model->body_rootid[body] == robotRoot; };
      const bool oneIsContact = isInBodySubtree(model, body1, contactBody);
      const bool twoIsContact = isInBodySubtree(model, body2, contactBody);
      if ((oneIsContact && !isRobot(body2)) || (twoIsContact && !isRobot(body1))) {
        mask |= (1u << i);
      }
    }
  }
  return mask;
}

RobotCentroidalState robotCentroidalState(const mjModel* model, const mjData* data) {
  RobotCentroidalState state;
  if (model == nullptr || data == nullptr) return state;
  for (int body = 1; body < model->nbody && state.rootBodyId < 0; ++body) {
    if (model->body_jntnum[body] > 0 && model->jnt_type[model->body_jntadr[body]] == mjJNT_FREE) state.rootBodyId = body;
  }
  if (state.rootBodyId < 0) return state;
  state.mass = model->body_subtreemass[state.rootBodyId];
  if (state.mass <= 0.0) return state;
  for (int axis = 0; axis < 3; ++axis) state.com[axis] = data->subtree_com[3 * state.rootBodyId + axis];
  // mj_forward does not fill subtree_linvel; the mass-weighted mean of the body centre velocities is the same thing.
  double momentum[3] = {0.0, 0.0, 0.0};
  for (int body = state.rootBodyId; body < model->nbody; ++body) {
    if (!isInBodySubtree(model, body, state.rootBodyId)) continue;
    const double mass = model->body_mass[body];
    if (mass <= 0.0) continue;
    mjtNum velocity[6];  // [angular, linear] of the body's inertial frame, world orientation
    mj_objectVelocity(model, data, mjOBJ_BODY, body, velocity, 0);
    for (int axis = 0; axis < 3; ++axis) momentum[axis] += mass * velocity[3 + axis];
  }
  for (int axis = 0; axis < 3; ++axis) state.comVelocity[axis] = momentum[axis] / state.mass;
  state.valid = true;
  return state;
}

GroundReaction groundReaction(const mjModel* model, const mjData* data, int rootBodyId, double minNormalForce) {
  GroundReaction reaction;
  if (model == nullptr || data == nullptr || rootBodyId < 0) return reaction;
  for (int c = 0; c < data->ncon; ++c) {
    const mjContact& contact = data->contact[c];
    if (contact.exclude != 0 || contact.efc_address < 0) continue;
    if (contact.geom[0] < 0 || contact.geom[1] < 0) continue;
    const int body1 = model->geom_bodyid[contact.geom[0]];
    const int body2 = model->geom_bodyid[contact.geom[1]];
    const bool oneIsRobot = isInBodySubtree(model, body1, rootBodyId);
    const bool twoIsRobot = isInBodySubtree(model, body2, rootBodyId);
    if (oneIsRobot == twoIsRobot) continue;  // self-contact, or a contact that does not involve the robot
    mjtNum forceInContactFrame[6];
    mj_contactForce(model, data, c, forceInContactFrame);
    // The contact frame is stored row-major (rows = the frame axes, the normal first, pointing from geom[0] to geom[1]);
    // the normal force is non-negative along it, so the world-frame force below is the one acting on geom[1].
    double force[3];
    for (int axis = 0; axis < 3; ++axis) {
      force[axis] = forceInContactFrame[0] * contact.frame[axis] + forceInContactFrame[1] * contact.frame[3 + axis] +
                    forceInContactFrame[2] * contact.frame[6 + axis];
    }
    if (oneIsRobot) {
      for (double& f : force) f = -f;  // the robot is geom[0]: it receives the reaction
    }
    for (int axis = 0; axis < 3; ++axis) reaction.force[axis] += force[axis];
    reaction.moment[0] += contact.pos[1] * force[2] - contact.pos[2] * force[1];
    reaction.moment[1] += contact.pos[2] * force[0] - contact.pos[0] * force[2];
    reaction.moment[2] += contact.pos[0] * force[1] - contact.pos[1] * force[0];
  }
  if (reaction.force[2] <= minNormalForce) return reaction;
  reaction.zmp[0] = -reaction.moment[1] / reaction.force[2];
  reaction.zmp[1] = reaction.moment[0] / reaction.force[2];
  reaction.valid = true;
  return reaction;
}

void divergentComponentOfMotion(const double com[3], const double comVelocity[3], double height, double gravity, double dcm[2]) {
  const double omega = std::sqrt(std::max(gravity, 0.0) / std::max(height, 0.05));
  for (int axis = 0; axis < 2; ++axis) dcm[axis] = com[axis] + (omega > 0.0 ? comVelocity[axis] / omega : 0.0);
}

}  // namespace robot::mujoco_sim_interface
