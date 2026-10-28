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

#pragma once

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <optional>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_multi_contact/collision/collision_model.nproto.h"
#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/stability/RobotStanceModel.h"
#include "humanoid_multi_contact/terrain/TerrainModel.h"

namespace ocs2::humanoid::multi_contact {

/**
 * [m] The neighborhood of an active pad whose spheres are exempt from the
 * terrain clearance (design section 5.4): without it, every contact would
 * collide with the surface it rests on.
 */
inline constexpr double kDefaultPadNeighborhood = 0.05;

/** One collision sphere, fixed to the link of a joint. Passive data. */
struct CollisionSphere {
  // The index of its link in CollisionModel::linkNames().
  int link = 0;
  pinocchio::JointIndex joint = 0;
  // [m] The center in the joint's frame.
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  // [m] Positive.
  double radius = 0.0;
};

/** Two spheres of a self-collision pair of links. Passive data. */
struct SpherePair {
  int first = 0;
  int second = 0;
};

/** The region of an active pad: its center, frame and, for a surface
 * candidate, its rectangle. Passive data. */
struct PadRegion {
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  // world_R_pad (ContactCandidate::padRotation).
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  std::optional<ContactRectangle> rectangle;
  // The joints whose links' spheres the pad exempts: of a point pad, its
  // link's joint, every joint below it and the joint of the link above
  // (exemptJointsOf()), so that a knee on a box exempts the shin, the foot and
  // the end of the thigh at the knee but not the other leg or the arms;
  // empty: the spheres of every link (a surface pad, whose region the
  // neighboring links rest on too: the thighs beside the seat).
  std::vector<pinocchio::JointIndex> joints;
};

/** [m] The distance of `point` to the pad region: to its center for a point
 * pad, to its rectangle (in the pad's plane) for a surface pad. */
double distanceToPad(const PadRegion& pad, const Eigen::Vector3d& point);

/** Whether `pad` exempts the spheres of the link of `joint` (PadRegion::
 * joints). */
bool exemptsJoint(const PadRegion& pad, pinocchio::JointIndex joint);

/** The joint of candidate `index`'s pad in `model` and every joint below
 * it. */
std::vector<pinocchio::JointIndex> ownJointsOf(const RobotStanceModel& model,
                                               int index);

/** The joints whose spheres the pad of candidate `index` of `model` exempts
 * (PadRegion::joints): none listed for a surface pad, ownJointsOf() and the
 * joint of the link above for a point pad. */
std::vector<pinocchio::JointIndex> exemptJointsOf(const RobotStanceModel& model,
                                                  int index);

/** The smallest clearance among some spheres, and where. Passive data. */
struct SphereClearance {
  // [m] Negative when a sphere penetrates; std::nullopt when no sphere counts.
  std::optional<double> clearance;
  // The sphere with that clearance, and its link's name.
  int sphere = 0;
  std::string link;
};

/**
 * The environment collision model of a robot (design section 5.4): spheres on
 * its links, placed along each capsule's axis at a spacing of at most one
 * radius and filling each box with spheres of its smallest half extent, and the
 * sphere pairs of the links that must not touch each other.
 *
 * A sphere counts against the terrain unless it lies within the neighborhood
 * of an active pad that exempts its link: its center closer than its radius
 * plus the neighborhood to the pad region (distanceToPad()), and, for a point
 * pad, on the pad's link or one below it (PadRegion::joints).
 *
 * A value type, immutable after Create(); the methods are const and safe to
 * call concurrently, but they read a RobotStanceModel, which is not.
 */
class CollisionModel {
 public:
  /**
   * Validates `config` against `model` and builds the spheres.
   * InvalidArgument names the link at fault: no link, a duplicate or empty
   * name, no shape, a vector of the wrong size, a radius or half extent that is
   * not positive, or a self-collision pair naming an unknown link or one link
   * twice; NotFound for a parent joint that is not a joint of the model.
   */
  static absl::StatusOr<CollisionModel> Create(
      const config::CollisionModel& config, const RobotStanceModel& model);

  CollisionModel(const CollisionModel&) = default;
  CollisionModel& operator=(const CollisionModel&) = default;
  CollisionModel(CollisionModel&&) noexcept = default;
  CollisionModel& operator=(CollisionModel&&) noexcept = default;
  ~CollisionModel() = default;

  const std::vector<std::string>& linkNames() const { return linkNames_; }
  const std::vector<CollisionSphere>& spheres() const { return spheres_; }
  const std::vector<SpherePair>& selfCollisionPairs() const {
    return selfCollisionPairs_;
  }

  /** [m] The world center of sphere `index` at `model`'s configuration;
   * requires 0 <= index < spheres().size(). */
  Eigen::Vector3d worldCenter(const RobotStanceModel& model, int index) const;

  /** The regions of the active pads of `mask` at `model`'s configuration. */
  static std::vector<PadRegion> activePadRegions(const RobotStanceModel& model,
                                                 ContactMask mask);

  /** Whether a sphere on the link of `joint` at `worldCenter` of `radius`
   * lies within `neighborhood` of one of `pads` that exempts that link. */
  static bool isExempt(const std::vector<PadRegion>& pads,
                       pinocchio::JointIndex joint,
                       const Eigen::Vector3d& worldCenter,
                       double radius,
                       double neighborhood);

  /** The smallest terrain clearance (TerrainModel::sphereDistance()) of the
   * spheres that are not exempt for the active pads of `mask`, at `model`'s
   * configuration. With `restingTolerance`, a pad outside `mask` whose center
   * is within that distance of the terrain [m] still rests on it, and exempts
   * the spheres of its own link and those below (ownJointsOf()), a point
   * pad's also those of the link above (exemptJointsOf()): a knee whose
   * contact the plan has just broken has not moved off the box yet, and its
   * shin and the end of its thigh rest there with it. */
  SphereClearance terrainClearance(
      const RobotStanceModel& model,
      const TerrainModel& terrain,
      ContactMask mask,
      double neighborhood,
      std::optional<double> restingTolerance = std::nullopt) const;

  /** The smallest distance between the surfaces of the spheres of the
   * self-collision pairs at `model`'s configuration; negative when two
   * overlap. */
  SphereClearance selfClearance(const RobotStanceModel& model) const;

 private:
  CollisionModel() = default;

  std::vector<std::string> linkNames_;
  std::vector<CollisionSphere> spheres_;
  std::vector<SpherePair> selfCollisionPairs_;
};

/** Reads the collision-model textproto at `path` (its header names
 * humanoid_multi_contact.CollisionModel). */
absl::StatusOr<config::CollisionModel> loadCollisionModelFile(
    absl::string_view path);

}  // namespace ocs2::humanoid::multi_contact
