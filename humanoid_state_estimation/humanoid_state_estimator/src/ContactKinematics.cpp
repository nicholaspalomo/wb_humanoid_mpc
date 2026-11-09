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
#include "absl/types/span.h"
#include "ocs2_core/Types.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody.hpp"
#include "pinocchio/multibody/fwd.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/spatial/fwd.hpp"

#include "humanoid_common_mpc/common/Types.h"

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
        std::vector<vector3_t> posContactsRtLinkEwrtLink,
        std::vector<Joint> robotJoints)
      : model(pinocchioModel),
        data(model),
        contactFrames(std::move(frames)),
        posContactsRtLinkEwrtLink(std::move(posContactsRtLinkEwrtLink)),
        joints(std::move(robotJoints)),
        q(pinocchio::neutral(model)),
        v(vector_t::Zero(model.nv)) {}

  pinocchio::Model model;
  pinocchio::Data data;
  std::vector<pinocchio::FrameIndex> contactFrames;
  std::vector<vector3_t> posContactsRtLinkEwrtLink;
  /** The model joints the robot names; the others stay neutral. */
  std::vector<Joint> joints;
  vector_t q;
  vector_t v;
};

namespace {

/** The coordinates of a continuous joint in Pinocchio's configuration vector:
 * (cos theta, sin theta). */
constexpr int kContinuousJointCoordinates = 2;

/** The universe's frame, which the fixed-base model's root link hangs from. */
constexpr pinocchio::FrameIndex kUniverseFrame = 0;

/** The names of the links (BODY frames) of `model`, for an error message. */
std::string linkNames(const pinocchio::Model& model) {
  std::vector<std::string> names;
  names.reserve(model.frames.size());
  for (const pinocchio::Frame& frame : model.frames) {
    if (frame.type == pinocchio::BODY) names.push_back(frame.name);
  }
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

/**
 * The link whose frame `frame` (a joint's, of type JOINT or FIXED_JOINT, or the
 * universe's) places: the BODY frame whose parent frame it is. NotFound when no
 * link hangs from it.
 */
absl::StatusOr<std::string> childLinkOfFrame(const pinocchio::Model& model,
                                             pinocchio::FrameIndex frame) {
  for (const pinocchio::Frame& candidate : model.frames) {
    if (candidate.type == pinocchio::BODY && candidate.parentFrame == frame) {
      return candidate.name;
    }
  }
  return absl::NotFoundError(
      absl::StrCat("ContactKinematics: no link hangs "
                   "from the frame '",
                   model.frames[frame].name, "'"));
}

}  // namespace

absl::StatusOr<std::vector<std::string>> childLinkNamesOfJoints(
    const std::string& urdfFile, absl::Span<const std::string> jointNames) {
  absl::StatusOr<pinocchio::Model> model = fixedBaseModel(urdfFile);
  if (!model.ok()) return model.status();
  const pinocchio::FrameType jointFrameTypes =
      static_cast<pinocchio::FrameType>(pinocchio::JOINT |
                                        pinocchio::FIXED_JOINT);
  std::vector<std::string> links;
  links.reserve(jointNames.size());
  for (const std::string& joint : jointNames) {
    pinocchio::FrameIndex frame = kUniverseFrame;
    if (joint != kMpcRootJointName) {
      if (!model->existFrame(joint, jointFrameTypes)) {
        return absl::InvalidArgumentError(
            absl::StrCat("ContactKinematics: the URDF '", urdfFile,
                         "' has no joint '", joint, "'"));
      }
      frame = model->getFrameId(joint, jointFrameTypes);
    }
    absl::StatusOr<std::string> link = childLinkOfFrame(*model, frame);
    if (!link.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat(link.status().message(), " in the URDF '", urdfFile,
                       "' (the child link of '", joint, "')"));
    }
    links.push_back(*std::move(link));
  }
  return links;
}

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
  std::vector<vector3_t> posContactsRtLinkEwrtLink;
  std::vector<std::string> names;
  for (const ContactPointSpec& contact : contacts) {
    if (!model->existBodyName(contact.linkName)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "ContactKinematics: the URDF '", urdfFile, "' has no link '",
          contact.linkName,
          "' for a contact point; its links: ", linkNames(*model)));
    }
    frames.push_back(model->getBodyId(contact.linkName));
    names.push_back(contact.linkName);
    posContactsRtLinkEwrtLink.push_back(contact.posContactRtLinkEwrtLink);
  }

  absl::flat_hash_map<std::string, size_t> robotJointByName;
  for (size_t joint = 0; joint < jointNames.size(); ++joint) {
    robotJointByName.emplace(jointNames[joint], joint);
  }
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
    if (robotJoint == robotJointByName.end()) {
      continue;  // not the robot's: stays neutral
    }
    modelJoints.push_back(Model::Joint{.robotJoint = robotJoint->second,
                                       .qIndex = model->idx_qs[joint],
                                       .vIndex = model->idx_vs[joint],
                                       .continuous = continuous});
  }
  std::unique_ptr<Model> kinematics = std::make_unique<Model>(
      *model, std::move(frames), std::move(posContactsRtLinkEwrtLink),
      std::move(modelJoints));
  return absl::WrapUnique(new ContactKinematics(
      std::move(kinematics), std::move(names), jointNames.size()));
}

ContactKinematics::ContactKinematics(std::unique_ptr<Model> model,
                                     std::vector<std::string> contactLinkNames,
                                     size_t numRobotJoints)
    : model_(std::move(model)),
      contactLinkNames_(std::move(contactLinkNames)),
      numRobotJoints_(numRobotJoints),
      gravityEwrtWorld_(model_->model.gravity.linear()),
      posContactsRtBaseEwrtBase_(contactLinkNames_.size(), vector3_t::Zero()),
      linvelContactsRtBaseEwrtBase_(contactLinkNames_.size(),
                                    vector3_t::Zero()) {}

ContactKinematics::~ContactKinematics() = default;

void ContactKinematics::update(const vector_t& jointPositions,
                               const vector_t& jointVelocities) {
  Model& kinematics = *model_;
  for (const Model::Joint& joint : kinematics.joints) {
    const scalar_t jointPosition =
        jointPositions[static_cast<Eigen::Index>(joint.robotJoint)];
    if (joint.continuous) {
      kinematics.q[joint.qIndex] = std::cos(jointPosition);
      kinematics.q[joint.qIndex + 1] = std::sin(jointPosition);
    } else {
      kinematics.q[joint.qIndex] = jointPosition;
    }
    kinematics.v[joint.vIndex] =
        jointVelocities[static_cast<Eigen::Index>(joint.robotJoint)];
  }
  pinocchio::forwardKinematics(kinematics.model, kinematics.data, kinematics.q,
                               kinematics.v);
  for (size_t contact = 0; contact < kinematics.contactFrames.size();
       ++contact) {
    const pinocchio::FrameIndex frame = kinematics.contactFrames[contact];
    const pinocchio::SE3& poseLinkRtBase = pinocchio::updateFramePlacement(
        kinematics.model, kinematics.data, frame);
    const vector3_t posContactRtLinkEwrtBase =
        poseLinkRtBase.rotation() *
        kinematics.posContactsRtLinkEwrtLink[contact];
    posContactsRtBaseEwrtBase_[contact] =
        poseLinkRtBase.translation() + posContactRtLinkEwrtBase;
    // The frame origin's velocity in the base axes, then the offset point's on
    // the same rigid body.
    const pinocchio::Motion twistLinkRtBaseEwrtBase =
        pinocchio::getFrameVelocity(kinematics.model, kinematics.data, frame,
                                    pinocchio::LOCAL_WORLD_ALIGNED);
    linvelContactsRtBaseEwrtBase_[contact] =
        twistLinkRtBaseEwrtBase.linear() +
        twistLinkRtBaseEwrtBase.angular().cross(posContactRtLinkEwrtBase);
  }
}

}  // namespace ocs2::humanoid::estimation
