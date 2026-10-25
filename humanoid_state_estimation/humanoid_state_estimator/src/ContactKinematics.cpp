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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "humanoid_state_estimator/ContactKinematics.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/parsers/urdf.hpp"

namespace ocs2::humanoid::estimation {

struct ContactKinematics::Model {
  /** How a joint of the model reads the robot's joint vectors. */
  struct Joint {
    /** The robot joint index of this model joint. */
    size_t robotJoint = 0;
    /** The joint's first entry in the model's q and v. */
    int qIndex = 0;
    int vIndex = 0;
    /** A continuous joint: q = (cos theta, sin theta). */
    bool continuous = false;
  };

  // pinocchio::Model has no move constructor, so the model is copied once, from
  // a reference.
  Model(const pinocchio::Model& pinocchioModel,
        std::vector<pinocchio::FrameIndex> frames,
        std::vector<vector3_t> frameOffsets,
        std::vector<Joint> robotJoints)
      : model(pinocchioModel),
        data(model),
        contactFrames(std::move(frames)),
        offsets(std::move(frameOffsets)),
        joints(std::move(robotJoints)),
        q(pinocchio::neutral(model)),
        v(vector_t::Zero(model.nv)) {}

  pinocchio::Model model;
  pinocchio::Data data;
  std::vector<pinocchio::FrameIndex> contactFrames;
  std::vector<vector3_t> offsets;
  /** The model joints the robot names; the others stay neutral. */
  std::vector<Joint> joints;
  vector_t q;
  vector_t v;
};

namespace {

/** The coordinates of a continuous joint in Pinocchio's configuration vector:
 * (cos theta, sin theta). */
constexpr int kContinuousJointCoordinates = 2;

/** The names of the frames of `model`, for an error message. */
std::string frameNames(const pinocchio::Model& model) {
  std::vector<std::string> names;
  names.reserve(model.frames.size());
  for (const pinocchio::Frame& frame : model.frames)
    names.push_back(frame.name);
  return absl::StrJoin(names, ", ");
}

absl::StatusOr<pinocchio::Model> fixedBaseModel(const std::string& urdfFile) {
  pinocchio::Model model;
  try {  // NOLINT(exceptions): Pinocchio's URDF parser throws; converted at once
    pinocchio::urdf::buildModel(urdfFile, model);
    // NOLINTNEXTLINE(exceptions): Pinocchio's URDF parser throws; converted at once
  } catch (const std::exception& error) {
    return absl::InvalidArgumentError(
        absl::StrCat("ContactKinematics: the URDF '", urdfFile,
                     "' does not load: ", error.what()));
  }
  return model;
}

}  // namespace

absl::StatusOr<std::unique_ptr<ContactKinematics>> ContactKinematics::Create(
    const std::string& urdfFile,
    const std::vector<ContactPointSpec>& contacts,
    const std::vector<std::string>& jointNames) {
  if (contacts.empty()) {
    return absl::InvalidArgumentError("ContactKinematics: no contact point");
  }
  absl::StatusOr<pinocchio::Model> model = fixedBaseModel(urdfFile);
  if (!model.ok()) return model.status();

  std::vector<pinocchio::FrameIndex> frames;
  std::vector<vector3_t> offsets;
  std::vector<std::string> names;
  for (const ContactPointSpec& contact : contacts) {
    const std::vector<std::string>::const_iterator found = std::find_if(
        contact.frameNames.begin(), contact.frameNames.end(),
        [&model](const std::string& name) { return model->existFrame(name); });
    if (found == contact.frameNames.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "ContactKinematics: the URDF '", urdfFile,
          "' has none of the frames ", absl::StrJoin(contact.frameNames, ", "),
          " of a contact point; its frames: ", frameNames(*model)));
    }
    frames.push_back(model->getFrameId(*found));
    names.push_back(*found);
    offsets.push_back(contact.offsetInFrame);
  }

  absl::flat_hash_map<std::string, size_t> robotJointByName;
  for (size_t joint = 0; joint < jointNames.size(); ++joint)
    robotJointByName.emplace(jointNames[joint], joint);
  std::vector<Model::Joint> modelJoints;
  // Joint 0 is the universe: the base link of the fixed-base model.
  for (pinocchio::JointIndex joint = 1;
       joint < static_cast<pinocchio::JointIndex>(model->njoints); ++joint) {
    const int nq = model->nqs[joint];
    const int nv = model->nvs[joint];
    const bool continuous = nv == 1 && nq == kContinuousJointCoordinates;
    if (nv != 1 || (nq != 1 && !continuous)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "ContactKinematics: the joint '", model->names[joint], "' of '",
          urdfFile, "' has ", nq, " coordinates and ", nv,
          " velocities; only revolute, prismatic and continuous joints are "
          "supported"));
    }
    const absl::flat_hash_map<std::string, size_t>::const_iterator robotJoint =
        robotJointByName.find(model->names[joint]);
    if (robotJoint == robotJointByName.end())
      continue;  // not the robot's: stays neutral
    modelJoints.push_back(Model::Joint{.robotJoint = robotJoint->second,
                                       .qIndex = model->idx_qs[joint],
                                       .vIndex = model->idx_vs[joint],
                                       .continuous = continuous});
  }
  std::unique_ptr<Model> kinematics = std::make_unique<Model>(
      *model, std::move(frames), std::move(offsets), std::move(modelJoints));
  return absl::WrapUnique(new ContactKinematics(
      std::move(kinematics), std::move(names), jointNames.size()));
}

ContactKinematics::ContactKinematics(std::unique_ptr<Model> model,
                                     std::vector<std::string> contactFrameNames,
                                     size_t numRobotJoints)
    : model_(std::move(model)),
      contactFrameNames_(std::move(contactFrameNames)),
      numRobotJoints_(numRobotJoints),
      gravity_(model_->model.gravity.linear()),
      contactPositions_(contactFrameNames_.size(), vector3_t::Zero()),
      contactVelocities_(contactFrameNames_.size(), vector3_t::Zero()) {}

ContactKinematics::~ContactKinematics() = default;

void ContactKinematics::update(const vector_t& jointPositions,
                               const vector_t& jointVelocities) {
  Model& kinematics = *model_;
  for (const Model::Joint& joint : kinematics.joints) {
    const scalar_t position =
        jointPositions[static_cast<Eigen::Index>(joint.robotJoint)];
    if (joint.continuous) {
      kinematics.q[joint.qIndex] = std::cos(position);
      kinematics.q[joint.qIndex + 1] = std::sin(position);
    } else {
      kinematics.q[joint.qIndex] = position;
    }
    kinematics.v[joint.vIndex] =
        jointVelocities[static_cast<Eigen::Index>(joint.robotJoint)];
  }
  pinocchio::forwardKinematics(kinematics.model, kinematics.data, kinematics.q,
                               kinematics.v);
  for (size_t contact = 0; contact < kinematics.contactFrames.size();
       ++contact) {
    const pinocchio::FrameIndex frame = kinematics.contactFrames[contact];
    const pinocchio::SE3& placement = pinocchio::updateFramePlacement(
        kinematics.model, kinematics.data, frame);
    const vector3_t offsetInBase =
        placement.rotation() * kinematics.offsets[contact];
    contactPositions_[contact] = placement.translation() + offsetInBase;
    // The frame origin's velocity in the base axes, then the offset point's on
    // the same rigid body.
    const pinocchio::Motion velocity =
        pinocchio::getFrameVelocity(kinematics.model, kinematics.data, frame,
                                    pinocchio::LOCAL_WORLD_ALIGNED);
    contactVelocities_[contact] =
        velocity.linear() + velocity.angular().cross(offsetInBase);
  }
}

}  // namespace ocs2::humanoid::estimation
