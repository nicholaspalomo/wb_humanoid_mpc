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

#include "humanoid_multi_contact/collision/CollisionModel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Dimensions.h"
#include "pinocchio/multibody/fwd.hpp"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_multi_contact/collision/collision_model.nproto.h"
#include "humanoid_multi_contact/collision/collision_model.nproto.pb.h"  // IWYU pragma: keep
#include "humanoid_multi_contact/collision/collision_model.pb.h"
#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/stability/RobotStanceModel.h"
#include "humanoid_multi_contact/terrain/TerrainModel.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid::multi_contact {
namespace {

using Link = config::CollisionModel::Link;

// Below this length [m] a capsule is a sphere at its start.
constexpr double kDegenerateLength = 1.0e-9;

absl::Status linkError(const config::CollisionModel& config,
                       absl::string_view link,
                       absl::string_view what) {
  return absl::InvalidArgumentError(absl::StrCat(
      "collision model of '", config.robot, "', link '", link, "': ", what));
}

absl::StatusOr<Eigen::Vector3d> point3(const Eigen::VectorXd& values,
                                       const config::CollisionModel& config,
                                       absl::string_view link,
                                       absl::string_view field) {
  if (values.size() != kPositionDim || !values.allFinite()) {
    return linkError(config, link,
                     absl::StrCat(field, " needs ", kPositionDim,
                                  " finite values, not ", values.size()));
  }
  return Eigen::Vector3d(values);
}

absl::Status checkRadius(double radius,
                         const config::CollisionModel& config,
                         absl::string_view link) {
  if (!(radius > 0.0) || !std::isfinite(radius)) {
    return linkError(config, link, "a radius must be positive");
  }
  return absl::OkStatus();
}

/** The number of intervals of at most `spacing` that cover `length`. */
int intervalsOf(double length, double spacing) {
  // A length within this fraction of a whole number of spacings needs no more.
  constexpr double kRoundingSlack = 1.0e-9;
  return std::max(
      1, static_cast<int>(std::ceil(length / spacing - kRoundingSlack)));
}

/** Computes the centers of spheres of the capsule's radius from its start to
 * its end, at most one radius apart, in the joint frame. */
absl::StatusOr<std::vector<Eigen::Vector3d>> computePosSpheresRtLinkEwrtLink(
    const config::CollisionModel::Capsule& capsule,
    const config::CollisionModel& config,
    absl::string_view link) {
  RETURN_IF_ERROR(checkRadius(capsule.radius, config, link));
  ASSIGN_OR_RETURN(const Eigen::Vector3d posCapsulestartRtLinkEwrtLink,
                   point3(capsule.pos_capsulestart_rt_link_ewrt_link, config,
                          link, "pos_capsulestart_rt_link_ewrt_link"));
  ASSIGN_OR_RETURN(const Eigen::Vector3d posCapsuleendRtLinkEwrtLink,
                   point3(capsule.pos_capsuleend_rt_link_ewrt_link, config,
                          link, "pos_capsuleend_rt_link_ewrt_link"));
  const Eigen::Vector3d posCapsuleendRtCapsulestartEwrtLink =
      posCapsuleendRtLinkEwrtLink - posCapsulestartRtLinkEwrtLink;
  const double length = posCapsuleendRtCapsulestartEwrtLink.norm();
  if (length < kDegenerateLength) {
    return std::vector<Eigen::Vector3d>{posCapsulestartRtLinkEwrtLink};
  }
  const int intervals = intervalsOf(length, capsule.radius);
  std::vector<Eigen::Vector3d> posSpheresRtLinkEwrtLink;
  for (int k = 0; k <= intervals; ++k) {
    posSpheresRtLinkEwrtLink.push_back(posCapsulestartRtLinkEwrtLink +
                                       posCapsuleendRtCapsulestartEwrtLink *
                                           static_cast<double>(k) /
                                           static_cast<double>(intervals));
  }
  return posSpheresRtLinkEwrtLink;
}

/** The coordinates along one box axis of half extent `halfExtent`: 0 when the
 * spheres span it, else a grid from -(h - r) to h - r at most r apart. */
std::vector<double> boxAxisCoordinates(double halfExtent, double radius) {
  const double span = halfExtent - radius;
  // Below this [m], the spheres already span the axis.
  constexpr double kNoSpan = 1.0e-9;
  if (span < kNoSpan) return {0.0};
  const int intervals = intervalsOf(span + span, radius);
  std::vector<double> coordinates;
  for (int k = 0; k <= intervals; ++k) {
    coordinates.push_back(-span + (span + span) * static_cast<double>(k) /
                                      static_cast<double>(intervals));
  }
  return coordinates;
}

struct BoxSpheres {
  std::vector<Eigen::Vector3d> posSpheresRtLinkEwrtLink;
  double radius = 0.0;
};

/** Spheres of the box's smallest half extent, inside the box. */
absl::StatusOr<BoxSpheres> boxSpheres(const config::CollisionModel::Box& box,
                                      const config::CollisionModel& config,
                                      absl::string_view link) {
  ASSIGN_OR_RETURN(const Eigen::Vector3d posBoxRtLinkEwrtLink,
                   point3(box.pos_box_rt_link_ewrt_link, config, link,
                          "pos_box_rt_link_ewrt_link"));
  ASSIGN_OR_RETURN(const Eigen::Vector3d halfExtents,
                   point3(box.half_extents, config, link, "half_extents"));
  if (!(halfExtents.minCoeff() > 0.0)) {
    return linkError(config, link, "half extents must be positive");
  }
  BoxSpheres result;
  result.radius = halfExtents.minCoeff();
  const std::vector<double> posSpheresRtBoxEwrtLinkX =
      boxAxisCoordinates(halfExtents.x(), result.radius);
  const std::vector<double> posSpheresRtBoxEwrtLinkY =
      boxAxisCoordinates(halfExtents.y(), result.radius);
  const std::vector<double> posSpheresRtBoxEwrtLinkZ =
      boxAxisCoordinates(halfExtents.z(), result.radius);
  for (const double posSphereRtBoxEwrtLinkX : posSpheresRtBoxEwrtLinkX) {
    for (const double posSphereRtBoxEwrtLinkY : posSpheresRtBoxEwrtLinkY) {
      for (const double posSphereRtBoxEwrtLinkZ : posSpheresRtBoxEwrtLinkZ) {
        result.posSpheresRtLinkEwrtLink.push_back(
            posBoxRtLinkEwrtLink + Eigen::Vector3d(posSphereRtBoxEwrtLinkX,
                                                   posSphereRtBoxEwrtLinkY,
                                                   posSphereRtBoxEwrtLinkZ));
      }
    }
  }
  return result;
}

/** Appends the spheres of `link` (its index `linkIndex`) to `spheres`. */
absl::Status addLinkSpheres(const config::CollisionModel& config,
                            const Link& link,
                            int linkIndex,
                            pinocchio::JointIndex joint,
                            std::vector<CollisionSphere>& spheres) {
  switch (link.shape.index()) {
    case Link::kCapsuleIndex: {
      const config::CollisionModel::Capsule& capsule =
          std::get<Link::kCapsuleIndex>(link.shape);
      ASSIGN_OR_RETURN(
          const std::vector<Eigen::Vector3d> posSpheresRtLinkEwrtLink,
          computePosSpheresRtLinkEwrtLink(capsule, config, link.name));
      for (const Eigen::Vector3d& posSphereRtLinkEwrtLink :
           posSpheresRtLinkEwrtLink) {
        spheres.push_back(
            CollisionSphere{.link = linkIndex,
                            .joint = joint,
                            .posSphereRtLinkEwrtLink = posSphereRtLinkEwrtLink,
                            .radius = capsule.radius});
      }
      return absl::OkStatus();
    }
    case Link::kBoxIndex: {
      ASSIGN_OR_RETURN(
          const BoxSpheres box,
          boxSpheres(std::get<Link::kBoxIndex>(link.shape), config, link.name));
      for (const Eigen::Vector3d& posSphereRtLinkEwrtLink :
           box.posSpheresRtLinkEwrtLink) {
        spheres.push_back(
            CollisionSphere{.link = linkIndex,
                            .joint = joint,
                            .posSphereRtLinkEwrtLink = posSphereRtLinkEwrtLink,
                            .radius = box.radius});
      }
      return absl::OkStatus();
    }
    case Link::kSpheresIndex: {
      const config::CollisionModel::Spheres& given =
          std::get<Link::kSpheresIndex>(link.shape);
      if (given.spheres.empty()) {
        return linkError(config, link.name, "spheres lists no sphere");
      }
      for (const config::CollisionModel::Sphere& sphere : given.spheres) {
        RETURN_IF_ERROR(checkRadius(sphere.radius, config, link.name));
        ASSIGN_OR_RETURN(const Eigen::Vector3d posSphereRtLinkEwrtLink,
                         point3(sphere.pos_sphere_rt_link_ewrt_link, config,
                                link.name, "pos_sphere_rt_link_ewrt_link"));
        spheres.push_back(
            CollisionSphere{.link = linkIndex,
                            .joint = joint,
                            .posSphereRtLinkEwrtLink = posSphereRtLinkEwrtLink,
                            .radius = sphere.radius});
      }
      return absl::OkStatus();
    }
    default:
      return linkError(config, link.name,
                       "sets none of capsule, box and spheres");
  }
}

}  // namespace

double distanceToPad(const PadRegion& pad,
                     const Eigen::Vector3d& posQueryRtWorldEwrtWorld) {
  if (!pad.rectangle.has_value()) {
    return (posQueryRtWorldEwrtWorld - pad.posPadRtWorldEwrtWorld).norm();
  }
  const Eigen::Vector3d posQueryRtPadEwrtPad =
      pad.rotmatPadRtWorld.transpose() *
      (posQueryRtWorldEwrtWorld - pad.posPadRtWorldEwrtWorld);
  const ContactRectangle& rectangle = *pad.rectangle;
  const Eigen::Vector3d clampedPosQueryRtPadEwrtPad(
      std::clamp(posQueryRtPadEwrtPad.x(), rectangle.minPosCornersRtPadEwrtPadX,
                 rectangle.maxPosCornersRtPadEwrtPadX),
      std::clamp(posQueryRtPadEwrtPad.y(), rectangle.minPosCornersRtPadEwrtPadY,
                 rectangle.maxPosCornersRtPadEwrtPadY),
      0.0);
  return (posQueryRtPadEwrtPad - clampedPosQueryRtPadEwrtPad).norm();
}

bool exemptsJoint(const PadRegion& pad, pinocchio::JointIndex joint) {
  return pad.joints.empty() || std::find(pad.joints.begin(), pad.joints.end(),
                                         joint) != pad.joints.end();
}

absl::StatusOr<LinkAbove> linkAboveOf(absl::string_view name) {
  for (size_t i = 0; i < kLinkAboveNames.size(); ++i) {
    if (kLinkAboveNames[i] == name) return static_cast<LinkAbove>(i);
  }
  return absl::InvalidArgumentError(
      absl::StrCat("unknown point_pad_link_above '", name,
                   "' (known: ", absl::StrJoin(kLinkAboveNames, ", "), ")"));
}

namespace {

/** The joint of the link above the pad of candidate `index`; std::nullopt
 * when the pad's joint is the root's. */
std::optional<pinocchio::JointIndex> jointAbove(const RobotStanceModel& model,
                                                int index) {
  const pinocchio::JointIndex parent =
      model.model().parents[model.padJoint(index)];
  if (parent == 0) return std::nullopt;
  return parent;
}

}  // namespace

std::vector<pinocchio::JointIndex> exemptJointsOf(const RobotStanceModel& model,
                                                  int index,
                                                  LinkAbove linkAbove) {
  if (model.contactSet().candidate(index).kind == ContactKind::kSurface) {
    return {};
  }
  std::vector<pinocchio::JointIndex> joints = ownJointsOf(model, index);
  if (const std::optional<pinocchio::JointIndex> above =
          jointAbove(model, index);
      linkAbove == LinkAbove::kExempt && above.has_value()) {
    joints.push_back(*above);
  }
  return joints;
}

std::vector<pinocchio::JointIndex> restingJointsOf(
    const RobotStanceModel& model, int index, LinkAbove linkAbove) {
  if (model.contactSet().candidate(index).kind == ContactKind::kSurface ||
      linkAbove != LinkAbove::kRests) {
    return {};
  }
  const std::optional<pinocchio::JointIndex> above = jointAbove(model, index);
  if (!above.has_value()) return {};
  return {*above};
}

std::vector<pinocchio::JointIndex> ownJointsOf(const RobotStanceModel& model,
                                               int index) {
  std::vector<pinocchio::JointIndex> joints;
  const pinocchio::Model& pinocchioModel = model.model();
  const pinocchio::JointIndex root = model.padJoint(index);
  for (pinocchio::JointIndex joint = 0;
       joint < static_cast<pinocchio::JointIndex>(pinocchioModel.njoints);
       ++joint) {
    // The universe (0) is no joint's descendant.
    for (pinocchio::JointIndex up = joint; up > 0;
         up = pinocchioModel.parents[up]) {
      if (up == root) {
        joints.push_back(joint);
        break;
      }
    }
  }
  return joints;
}

absl::StatusOr<CollisionModel> CollisionModel::Create(
    const config::CollisionModel& config, const RobotStanceModel& model) {
  if (config.links.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "the collision model of '", config.robot, "' has no link"));
  }
  CollisionModel result;
  absl::flat_hash_map<std::string, int> linkIndex;
  for (const Link& link : config.links) {
    if (link.name.empty()) {
      return linkError(config, link.name, "a link needs a name");
    }
    if (!linkIndex
             .try_emplace(link.name, static_cast<int>(result.linkNames_.size()))
             .second) {
      return linkError(config, link.name, "the name is given twice");
    }
    ASSIGN_OR_RETURN(const pinocchio::JointIndex joint,
                     model.jointIndex(link.parent_joint));
    RETURN_IF_ERROR(addLinkSpheres(config, link,
                                   static_cast<int>(result.linkNames_.size()),
                                   joint, result.spheres_));
    result.linkNames_.push_back(link.name);
  }
  for (const config::CollisionModel::SelfCollisionPair& pair :
       config.self_collision_pairs) {
    const absl::flat_hash_map<std::string, int>::const_iterator first =
        linkIndex.find(pair.first);
    const absl::flat_hash_map<std::string, int>::const_iterator second =
        linkIndex.find(pair.second);
    if (first == linkIndex.end() || second == linkIndex.end() ||
        first->second == second->second) {
      return linkError(config, absl::StrCat(pair.first, "/", pair.second),
                       "a self-collision pair needs two different links");
    }
    for (int a = 0; a < static_cast<int>(result.spheres_.size()); ++a) {
      if (result.spheres_[a].link != first->second) continue;
      for (int b = 0; b < static_cast<int>(result.spheres_.size()); ++b) {
        if (result.spheres_[b].link != second->second) continue;
        result.selfCollisionPairs_.push_back(
            SpherePair{.first = a, .second = b});
      }
    }
  }
  return result;
}

Eigen::Vector3d CollisionModel::getPosSphereRtWorldEwrtWorld(
    const RobotStanceModel& model, int index) const {
  const CollisionSphere& sphere = spheres_[index];
  return model.getPoseLinkRtWorld(sphere.joint)
      .act(sphere.posSphereRtLinkEwrtLink);
}

std::vector<PadRegion> CollisionModel::activePadRegions(
    const RobotStanceModel& model, ContactMask mask, LinkAbove linkAbove) {
  std::vector<PadRegion> pads;
  for (int index = 0; index < model.contactSet().size(); ++index) {
    if (!isContactActive(mask, index)) continue;
    pads.push_back(PadRegion{
        .posPadRtWorldEwrtWorld = model.getPosPadRtWorldEwrtWorld(index),
        .rotmatPadRtWorld = model.getRotmatPadRtWorld(index),
        .rectangle = model.contactSet().candidate(index).rectangle,
        .joints = exemptJointsOf(model, index, linkAbove),
        .restingJoints = restingJointsOf(model, index, linkAbove)});
  }
  return pads;
}

bool CollisionModel::isExempt(const std::vector<PadRegion>& pads,
                              pinocchio::JointIndex joint,
                              const Eigen::Vector3d& posSphereRtWorldEwrtWorld,
                              double radius,
                              double neighborhood) {
  for (const PadRegion& pad : pads) {
    if (exemptsJoint(pad, joint) &&
        distanceToPad(pad, posSphereRtWorldEwrtWorld) < radius + neighborhood) {
      return true;
    }
  }
  return false;
}

double CollisionModel::restingAllowance(
    const std::vector<PadRegion>& pads,
    pinocchio::JointIndex joint,
    const Eigen::Vector3d& posSphereRtWorldEwrtWorld,
    double radius,
    double neighborhood) {
  for (const PadRegion& pad : pads) {
    if (std::find(pad.restingJoints.begin(), pad.restingJoints.end(), joint) !=
            pad.restingJoints.end() &&
        distanceToPad(pad, posSphereRtWorldEwrtWorld) < radius + neighborhood) {
      return kRestingLinkDepth;
    }
  }
  return 0.0;
}

SphereClearance CollisionModel::terrainClearance(
    const RobotStanceModel& model,
    const TerrainModel& terrain,
    ContactMask mask,
    double neighborhood,
    std::optional<double> restingTolerance) const {
  std::vector<PadRegion> pads = activePadRegions(model, mask, linkAbove_);
  if (restingTolerance.has_value()) {
    for (int index = 0; index < model.contactSet().size(); ++index) {
      if (isContactActive(mask, index)) continue;
      const Eigen::Vector3d posPadRtWorldEwrtWorld =
          model.getPosPadRtWorldEwrtWorld(index);
      if (std::abs(terrain.signedDistance(posPadRtWorldEwrtWorld)) >
          *restingTolerance) {
        continue;
      }
      // A resting pad exempts its own link and those below, and treats the
      // end of a point pad's link above as an active one does: a knee
      // settling onto a box.
      const bool surface =
          model.contactSet().candidate(index).kind == ContactKind::kSurface;
      pads.push_back(PadRegion{
          .posPadRtWorldEwrtWorld = posPadRtWorldEwrtWorld,
          .rotmatPadRtWorld = model.getRotmatPadRtWorld(index),
          .rectangle = model.contactSet().candidate(index).rectangle,
          .joints = surface ? ownJointsOf(model, index)
                            : exemptJointsOf(model, index, linkAbove_),
          .restingJoints = restingJointsOf(model, index, linkAbove_)});
    }
  }
  SphereClearance result;
  for (int index = 0; index < static_cast<int>(spheres_.size()); ++index) {
    const Eigen::Vector3d posSphereRtWorldEwrtWorld =
        getPosSphereRtWorldEwrtWorld(model, index);
    const double radius = spheres_[index].radius;
    if (isExempt(pads, spheres_[index].joint, posSphereRtWorldEwrtWorld, radius,
                 neighborhood)) {
      continue;
    }
    const double clearance =
        terrain.sphereDistance(posSphereRtWorldEwrtWorld, radius) +
        restingAllowance(pads, spheres_[index].joint, posSphereRtWorldEwrtWorld,
                         radius, neighborhood);
    if (!result.clearance.has_value() || clearance < *result.clearance) {
      result.clearance = clearance;
      result.sphere = index;
      result.link = linkNames_[spheres_[index].link];
    }
  }
  return result;
}

SphereClearance CollisionModel::selfClearance(
    const RobotStanceModel& model) const {
  SphereClearance result;
  for (const SpherePair& pair : selfCollisionPairs_) {
    const double clearance = (getPosSphereRtWorldEwrtWorld(model, pair.first) -
                              getPosSphereRtWorldEwrtWorld(model, pair.second))
                                 .norm() -
                             spheres_[pair.first].radius -
                             spheres_[pair.second].radius;
    if (!result.clearance.has_value() || clearance < *result.clearance) {
      result.clearance = clearance;
      result.sphere = pair.first;
      result.link = absl::StrCat(linkNames_[spheres_[pair.first].link], "/",
                                 linkNames_[spheres_[pair.second].link]);
    }
  }
  return result;
}

absl::StatusOr<config::CollisionModel> loadCollisionModelFile(
    absl::string_view path) {
  return nproto::LoadTextprotoFile<config::CollisionModel,
                                   humanoid_multi_contact::CollisionModel>(
      path);
}

}  // namespace ocs2::humanoid::multi_contact
