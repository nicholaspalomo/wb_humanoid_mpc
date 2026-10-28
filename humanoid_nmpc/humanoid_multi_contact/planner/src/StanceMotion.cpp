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

#include "humanoid_multi_contact/planner/StanceMotion.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "pinocchio/algorithm/joint-configuration.hpp"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_multi_contact/planner/SwingPath.h"
#include "humanoid_multi_contact/stability/ConvexPolygon.h"
#include "humanoid_multi_contact/stability/StabilityCertifier.h"
#include "humanoid_multi_contact/stability/TorqueFeasibility.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// [m] The resolution of a stance key: stances whose pads are on the same
// primitives and within one cell of this grid are one, so the search does not
// expand the same stance with a hand moved by a grid step.
constexpr double kKeyResolution = 0.05;
// The fraction of a segment over which a pad that broke contact lifts off.
constexpr double kLiftOffFraction = 0.3;
// [rad or m] Two knots closer than this in every coordinate are one.
constexpr double kSameKnot = 1.0e-9;
// A polygon of fewer vertices has no inside.
constexpr size_t kMinPolygonVertices = 3;
// A segment has at least one knot between its ends.
constexpr int kMinSegmentIntervals = 2;
// The --v levels that log every knot of every motion, and a segment's end
// solved again.
constexpr int kKnotLogLevel = 3;
constexpr int kEndLogLevel = 2;
// Below this fraction of a segment, its end is reached.
constexpr double kSegmentEnd = 1.0e-9;
// The middle of an interval of a segment.
constexpr double kMiddle = 0.5;

/** world_R_pad of a placement: z along the normal, x along its x axis. */
Eigen::Matrix3d placementRotation(const ContactPlacement& placement) {
  Eigen::Matrix3d rotation;
  rotation.col(kXAxisIndex) = placement.xAxis;
  rotation.col(kYAxisIndex) = placement.normal.cross(placement.xAxis);
  rotation.col(kZAxisIndex) = placement.normal;
  return rotation;
}

}  // namespace

ContactMask maskOf(const StancePlacements& stance) {
  ContactMask mask = 0;
  for (const ContactPlacement& placement : stance) {
    mask |= contactBit(placement.candidate);
  }
  return mask;
}

std::string stanceKey(const StancePlacements& stance) {
  std::string key;
  for (const ContactPlacement& placement : stance) {
    const Eigen::Vector3d cell = placement.position / kKeyResolution;
    absl::StrAppend(
        &key, placement.candidate, "@", placement.primitive, ":",
        placement.face.has_value() ? static_cast<int>(*placement.face) : -1,
        ":", std::lround(cell.x()), ",", std::lround(cell.y()), ",",
        std::lround(cell.z()), ";");
  }
  return key;
}

StancePlacements withPlacement(const StancePlacements& stance,
                               const ContactPlacement& placement) {
  StancePlacements result = stance;
  result.insert(std::upper_bound(result.begin(), result.end(), placement,
                                 [](const ContactPlacement& first,
                                    const ContactPlacement& second) {
                                   return first.candidate < second.candidate;
                                 }),
                placement);
  return result;
}

StancePlacements withoutCandidate(const StancePlacements& stance,
                                  int candidate) {
  StancePlacements result;
  for (const ContactPlacement& placement : stance) {
    if (placement.candidate != candidate) result.push_back(placement);
  }
  return result;
}

std::optional<ContactPlacement> placementOf(const StancePlacements& stance,
                                            int candidate) {
  for (const ContactPlacement& placement : stance) {
    if (placement.candidate == candidate) return placement;
  }
  return std::nullopt;
}

namespace {

/** The corners of a surface placement's rectangle in the world. */
std::vector<Eigen::Vector3d> cornersOf(const ContactPlacement& placement,
                                       const ContactRectangle& rectangle) {
  const Eigen::Vector3d yAxis = placement.normal.cross(placement.xAxis);
  std::vector<Eigen::Vector3d> corners;
  for (const double x : {rectangle.xMin, rectangle.xMax}) {
    for (const double y : {rectangle.yMin, rectangle.yMax}) {
      corners.push_back(placement.position + x * placement.xAxis + y * yAxis);
    }
  }
  return corners;
}

/** Whether the projections of `first` and `second` on `axis` are apart by at
 * least `margin`. */
bool separatedAlong(const std::vector<Eigen::Vector3d>& first,
                    const std::vector<Eigen::Vector3d>& second,
                    const Eigen::Vector3d& axis,
                    double margin) {
  double firstLow = std::numeric_limits<double>::infinity();
  double firstHigh = -firstLow;
  double secondLow = firstLow;
  double secondHigh = -firstLow;
  for (const Eigen::Vector3d& corner : first) {
    firstLow = std::min(firstLow, axis.dot(corner));
    firstHigh = std::max(firstHigh, axis.dot(corner));
  }
  for (const Eigen::Vector3d& corner : second) {
    secondLow = std::min(secondLow, axis.dot(corner));
    secondHigh = std::max(secondHigh, axis.dot(corner));
  }
  return secondLow - firstHigh >= margin || firstLow - secondHigh >= margin;
}

}  // namespace

bool placementsOverlap(const ContactPlacement& first,
                       const ContactPlacement& second,
                       const ContactSet& contactSet,
                       double margin) {
  // Normals within about 25 degrees, planes within 5 cm: one surface.
  constexpr double kSameSurfaceCosine = 0.9;
  constexpr double kSameSurfaceOffset = 0.05;
  if (first.normal.dot(second.normal) < kSameSurfaceCosine ||
      std::abs(first.normal.dot(second.position - first.position)) >
          kSameSurfaceOffset) {
    return false;
  }
  const ContactCandidate& firstCandidate =
      contactSet.candidate(first.candidate);
  const ContactCandidate& secondCandidate =
      contactSet.candidate(second.candidate);
  if (!firstCandidate.rectangle.has_value() &&
      !secondCandidate.rectangle.has_value()) {
    return (first.position - second.position).norm() <
           firstCandidate.geomRadius + secondCandidate.geomRadius + margin;
  }
  if (!firstCandidate.rectangle.has_value() ||
      !secondCandidate.rectangle.has_value()) {
    const ContactPlacement& point =
        firstCandidate.rectangle.has_value() ? second : first;
    const ContactPlacement& surface =
        firstCandidate.rectangle.has_value() ? first : second;
    return distanceToPad(padRegionOf(surface, contactSet), point.position) <
           contactSet.candidate(point.candidate).geomRadius + margin;
  }
  const std::vector<Eigen::Vector3d> firstCorners =
      cornersOf(first, *firstCandidate.rectangle);
  const std::vector<Eigen::Vector3d> secondCorners =
      cornersOf(second, *secondCandidate.rectangle);
  for (const Eigen::Vector3d& axis :
       {first.xAxis, Eigen::Vector3d(first.normal.cross(first.xAxis)),
        second.xAxis, Eigen::Vector3d(second.normal.cross(second.xAxis))}) {
    if (separatedAlong(firstCorners, secondCorners, axis, margin)) return false;
  }
  return true;
}

PadRegion padRegionOf(const ContactPlacement& placement,
                      const ContactSet& contactSet) {
  return PadRegion{
      .center = placement.position,
      .rotation = placementRotation(placement),
      .rectangle = contactSet.candidate(placement.candidate).rectangle};
}

PadRegion exemptRegionOf(const ContactPlacement& placement,
                         const RobotStanceModel& model) {
  PadRegion region = padRegionOf(placement, model.contactSet());
  region.joints = exemptJointsOf(model, placement.candidate);
  return region;
}

Eigen::VectorXd knotInterpolation(const Eigen::VectorXd& from,
                                  const Eigen::VectorXd& to,
                                  double fraction) {
  Eigen::VectorXd q = from + fraction * (to - from);
  // The quaternion slerp of the MPC's interpolateWithSlerpedOrientation().
  q.segment<kQuaternionCoefficients>(kPositionDim) = quaternionSlerp<double>(
      from.segment<kQuaternionCoefficients>(kPositionDim),
      to.segment<kQuaternionCoefficients>(kPositionDim), fraction);
  return q;
}

StanceMotion::StanceMotion(std::unique_ptr<StanceInverseKinematics> ik,
                           const StanceMotionOptions& options)
    : ik_(std::move(ik)), options_(options) {}

absl::StatusOr<std::unique_ptr<StanceMotion>> StanceMotion::Create(
    std::unique_ptr<StanceInverseKinematics> ik,
    const StanceMotionOptions& options) {
  if (ik == nullptr) {
    return absl::InvalidArgumentError("the stance motion needs an IK");
  }
  if (!(options.comMargin >= 0.0) || !(options.swingClearance > 0.0) ||
      !(options.maxKnotComStep > 0.0) || !(options.maxKnotBaseStep > 0.0) ||
      !(options.maxKnotPadStep > 0.0) || !(options.maxKnotJointStep > 0.0) ||
      options.maxSegmentKnots < kMinSegmentIntervals ||
      !(options.sphereTolerance >= 0.0) || !(options.padNeighborhood >= 0.0) ||
      !(options.pointTouchdownPostureLead > 0.0 &&
        options.pointTouchdownPostureLead <= 1.0)) {
    return absl::InvalidArgumentError(
        "the stance motion's margins and steps must be positive");
  }
  if (options.sweep.has_value() &&
      !(options.sweep->joint > 0.0 && options.sweep->position > 0.0 &&
        options.sweep->tolerance >= 0.0 && options.sweep->minIntervals >= 1)) {
    return absl::InvalidArgumentError(
        "the stance motion's sweep resolution must be positive");
  }
  return absl::WrapUnique(new StanceMotion(std::move(ik), options));
}

absl::StatusOr<std::vector<Eigen::Vector2d>> StanceMotion::polygonOf(
    const StancePlacements& stance) {
  const std::string key = stanceKey(stance);
  if (const absl::flat_hash_map<
          std::string, std::vector<Eigen::Vector2d>>::const_iterator cached =
          polygons_.find(key);
      cached != polygons_.end()) {
    return cached->second;
  }
  const ContactSet& contactSet = model().contactSet();
  Stance contacts;
  for (const ContactPlacement& placement : stance) {
    const ContactCandidate& candidate =
        contactSet.candidate(placement.candidate);
    contacts.push_back(StanceContact{
        .name = candidate.name,
        .candidateIndex = placement.candidate,
        .kind = candidate.kind,
        .position = placement.position,
        .rotation = coneRotationOnSurface(placementRotation(placement),
                                          placement.normal),
        .friction = ik_->terrain().primitive(placement.primitive).friction,
        .rectangle = candidate.rectangle});
  }
  std::vector<Eigen::Vector2d> polygon;
  if (!contacts.empty()) {
    ASSIGN_OR_RETURN(
        const StaticEquilibrium equilibrium,
        StaticEquilibrium::Create(std::move(contacts), options_.equilibrium));
    // An LP HPIPM cannot solve (a degenerate stance) leaves the stance
    // without a polygon: no configuration is planned on it.
    if (absl::StatusOr<std::vector<Eigen::Vector2d>> support =
            equilibrium.supportPolygon();
        support.ok()) {
      polygon = *std::move(support);
    } else if (support.status().code() != absl::StatusCode::kInternal) {
      return support.status();
    }
  }
  polygons_[key] = polygon;
  return polygon;
}

StanceIkProblem StanceMotion::holdingProblem(
    const StancePlacements& held, std::vector<Eigen::Vector2d> polygon) const {
  StanceIkProblem problem;
  for (const ContactPlacement& placement : held) {
    problem.contacts.push_back(ContactTarget{.candidate = placement.candidate,
                                             .position = placement.position,
                                             .normal = placement.normal,
                                             .xAxis = placement.xAxis});
    problem.exemptPads.push_back(exemptRegionOf(placement, ik_->model()));
  }
  problem.comPolygon = std::move(polygon);
  problem.comMargin = options_.comMargin;
  return problem;
}

absl::StatusOr<std::optional<std::string>> StanceMotion::checkStatics(
    const Eigen::VectorXd& q, const StancePlacements& stance) {
  RobotStanceModel& robot = model();
  RETURN_IF_ERROR(robot.setGeneralizedPosition(q));
  ASSIGN_OR_RETURN(const StanceOnTerrain onTerrain,
                   stanceOnTerrain(robot, ik_->terrain(), maskOf(stance)));
  if (onTerrain.stance.empty()) {
    return std::optional<std::string>("the stance has no contact");
  }
  if (onTerrain.maxPadDistance > ik_->options().reachTolerance) {
    return std::optional<std::string>(absl::StrFormat(
        "a pad is %.4f m from the terrain", onTerrain.maxPadDistance));
  }
  ASSIGN_OR_RETURN(
      const StaticEquilibrium equilibrium,
      StaticEquilibrium::Create(onTerrain.stance, options_.equilibrium));
  const Eigen::Vector2d com =
      robot.centerOfMass().head<kHorizontalPositionDim>();
  const absl::StatusOr<double> residual = equilibrium.equilibriumResidual(com);
  if (!residual.ok()) {
    return std::optional<std::string>(absl::StrCat(
        "the equilibrium LP fails: ", residual.status().message()));
  }
  if (!(*residual <= options_.equilibrium.feasibilityTolerance)) {
    return std::optional<std::string>(
        absl::StrFormat("no static equilibrium (residual %.3g)", *residual));
  }
  if (options_.exactStaticMargin.has_value()) {
    const absl::StatusOr<std::vector<Eigen::Vector2d>> support =
        equilibrium.supportPolygon();
    if (!support.ok()) {
      return std::optional<std::string>(absl::StrCat(
          "the support polygon's LPs fail: ", support.status().message()));
    }
    const std::vector<Eigen::Vector2d>& polygon = *support;
    const std::optional<double> margin = staticMargin(polygon, com);
    if (!margin.has_value() || *margin < *options_.exactStaticMargin) {
      return std::optional<std::string>(
          absl::StrFormat("static margin %.4f m (needs %.4f m)",
                          margin.value_or(-1.0), *options_.exactStaticMargin));
    }
  }
  if (!enforcesEffortLimits(options_.jointLimits)) {
    return std::optional<std::string>();
  }
  const absl::StatusOr<TorqueFeasibilityResult> torques =
      minimumTorqueUtilization(robot, equilibrium);
  if (!torques.ok()) {
    return std::optional<std::string>(
        absl::StrCat("the torque LP fails: ", torques.status().message()));
  }
  if (!isTorqueFeasible(*torques, options_.torqueMargin)) {
    return std::optional<std::string>(absl::StrFormat(
        "torque utilization %.3f at %s (at most %.3f)", torques->utilization,
        torques->maxTorqueJoint, options_.torqueMargin));
  }
  return std::optional<std::string>();
}

absl::StatusOr<std::optional<std::string>> StanceMotion::checkSphereClearance(
    const Eigen::VectorXd& q, const StancePlacements& stance) {
  return sphereClearance(q, stance, options_.sphereTolerance);
}

absl::StatusOr<std::optional<std::string>> StanceMotion::sphereClearance(
    const Eigen::VectorXd& q,
    const StancePlacements& stance,
    double tolerance) {
  RobotStanceModel& robot = model();
  RETURN_IF_ERROR(robot.setGeneralizedPosition(q));
  const SphereClearance clearance = ik_->collision().terrainClearance(
      robot, ik_->terrain(), maskOf(stance), options_.padNeighborhood,
      /*restingTolerance=*/ik_->options().reachTolerance);
  if (clearance.clearance.has_value() && *clearance.clearance < -tolerance) {
    return std::optional<std::string>(
        absl::StrFormat("a collision sphere of %s is %.4f m inside the terrain",
                        clearance.link, -*clearance.clearance));
  }
  return std::optional<std::string>();
}

absl::StatusOr<std::optional<std::string>> StanceMotion::sweptClearance(
    const Eigen::VectorXd& from,
    const Eigen::VectorXd& to,
    const StancePlacements& stance) {
  if (!options_.sweep.has_value()) return std::optional<std::string>();
  const Eigen::VectorXd tangent =
      pinocchio::difference(model().model(), from, to);
  const double samples = std::max(
      tangent.tail(tangent.size() - kPositionDim).lpNorm<Eigen::Infinity>() /
          options_.sweep->joint,
      tangent.head<kPositionDim>().norm() / options_.sweep->position);
  const int intervals = std::max(options_.sweep->minIntervals,
                                 static_cast<int>(std::ceil(samples)));
  ASSIGN_OR_RETURN(const std::optional<std::string> chord,
                   padChordCollision(from, to, stance, intervals));
  if (chord.has_value()) return chord;
  std::vector<Eigen::Vector2d> polygon;
  if (!stance.empty()) {
    ASSIGN_OR_RETURN(polygon, polygonOf(stance));
  }
  const double comMargin = options_.comMargin - options_.sweep->comMarginSlack;
  for (int i = 1; i < intervals; ++i) {
    const double fraction = static_cast<double>(i) / intervals;
    const Eigen::VectorXd q = knotInterpolation(from, to, fraction);
    ASSIGN_OR_RETURN(const std::optional<std::string> penetration,
                     sphereClearance(q, stance, options_.sweep->tolerance));
    if (penetration.has_value()) {
      return std::optional<std::string>(absl::StrFormat(
          "%.2f of the way between two knots, %s", fraction, *penetration));
    }
    // sphereClearance() set the model's configuration.
    if (polygon.size() < kMinPolygonVertices) continue;
    const double margin = signedDistanceToConvexPolygon(
        polygon, model().centerOfMass().head<kHorizontalPositionDim>());
    if (margin < comMargin) {
      return std::optional<std::string>(absl::StrFormat(
          "%.2f of the way between two knots, the CoM is %.4f m inside its "
          "polygon (needs %.4f m)",
          fraction, margin, comMargin));
    }
  }
  return std::optional<std::string>();
}

absl::StatusOr<std::optional<std::string>> StanceMotion::padChordCollision(
    const Eigen::VectorXd& from,
    const Eigen::VectorXd& to,
    const StancePlacements& stance,
    int intervals) {
  RobotStanceModel& robot = model();
  const ContactSet& contactSet = robot.contactSet();
  // The points of each free pad's geometry, in its frame (z = -padNormal),
  // with their radii: a point pad's sphere behind its center, as the MPC's
  // terrain_non_penetration barrier places it; a sole's four corners.
  struct PadPoint {
    int candidate = 0;
    Eigen::Vector3d offset = Eigen::Vector3d::Zero();
    double radius = 0.0;
  };
  std::vector<PadPoint> points;
  for (const int candidate : options_.clearedPads) {
    if (placementOf(stance, candidate).has_value()) continue;
    const ContactCandidate& pad = contactSet.candidate(candidate);
    if (pad.rectangle.has_value()) {
      for (const double x : {pad.rectangle->xMin, pad.rectangle->xMax}) {
        for (const double y : {pad.rectangle->yMin, pad.rectangle->yMax}) {
          points.push_back(PadPoint{.candidate = candidate,
                                    .offset = Eigen::Vector3d(x, y, 0.0)});
        }
      }
    } else if (pad.geomRadius > 0.0) {
      points.push_back(
          PadPoint{.candidate = candidate,
                   .offset = pad.geomRadius * Eigen::Vector3d::UnitZ(),
                   .radius = pad.geomRadius});
    }
  }
  if (points.empty()) return std::optional<std::string>();
  std::vector<Eigen::Vector3d> fromPositions;
  std::vector<Eigen::Quaterniond> fromRotations;
  RETURN_IF_ERROR(robot.setGeneralizedPosition(from));
  for (const PadPoint& point : points) {
    fromPositions.push_back(robot.padPosition(point.candidate));
    fromRotations.emplace_back(robot.padRotation(point.candidate));
  }
  RETURN_IF_ERROR(robot.setGeneralizedPosition(to));
  for (size_t p = 0; p < points.size(); ++p) {
    const Eigen::Vector3d toPosition = robot.padPosition(points[p].candidate);
    const Eigen::Quaterniond toRotation(robot.padRotation(points[p].candidate));
    for (int i = 1; i < intervals; ++i) {
      const double fraction = static_cast<double>(i) / intervals;
      const Eigen::Vector3d point =
          fromPositions[p] + fraction * (toPosition - fromPositions[p]) +
          fromRotations[p].slerp(fraction, toRotation) * points[p].offset;
      const double depth =
          points[p].radius - ik_->terrain().signedDistance(point);
      if (depth > options_.sweep->tolerance) {
        return std::optional<std::string>(absl::StrFormat(
            "%.2f of the way between two knots, the straight path of the pad "
            "of %s is %.4f m inside the terrain",
            fraction, contactSet.candidate(points[p].candidate).name, depth));
      }
    }
  }
  return std::optional<std::string>();
}

std::optional<std::string> StanceMotion::padInTerrain(
    const Eigen::VectorXd& q, const StancePlacements& held) {
  RobotStanceModel& robot = model();
  if (!robot.setGeneralizedPosition(q).ok()) {
    return std::optional<std::string>("the configuration is not valid");
  }
  for (const int candidate : options_.clearedPads) {
    if (placementOf(held, candidate).has_value()) continue;
    const double distance =
        ik_->terrain().signedDistance(robot.padPosition(candidate));
    if (distance < -options_.padPenetrationTolerance) {
      return std::optional<std::string>(absl::StrFormat(
          "the pad of %s is %.4f m inside the terrain",
          robot.contactSet().candidate(candidate).name, -distance));
    }
  }
  return std::nullopt;
}

absl::StatusOr<MotionCheck> StanceMotion::transition(
    const StancePlacements& held,
    const StancePlacements& supporting,
    const Eigen::VectorXd& start,
    const std::vector<Eigen::VectorXd>& references) {
  ASSIGN_OR_RETURN(std::vector<Eigen::Vector2d> polygon, polygonOf(supporting));
  MotionCheck check;
  if (polygon.size() < kMinPolygonVertices) {
    check.failure = "the supporting stance has no static equilibrium polygon";
    return check;
  }
  StanceIkProblem problem = holdingProblem(held, std::move(polygon));
  StanceIkProblem reachOnly = problem;
  reachOnly.comPolygon.clear();
  for (const Eigen::VectorXd& reference : references) {
    problem.postureReference = reference;
    ASSIGN_OR_RETURN(StanceIkResult result, ik_->solve(problem, start));
    if (!result.feasible) {
      // Staged: the pads first, without the CoM's polygon, then the polygon
      // from there; a far target and the balance fight less one at a time.
      reachOnly.postureReference = reference;
      ASSIGN_OR_RETURN(const StanceIkResult reached,
                       ik_->solve(reachOnly, start));
      ASSIGN_OR_RETURN(result, ik_->solve(problem, reached.q));
    }
    if (!result.feasible) {
      check.failure = absl::StrCat("IK: ", result.failure);
      continue;
    }
    if (const std::optional<std::string> pad = padInTerrain(result.q, held);
        pad.has_value()) {
      check.failure = *pad;
      continue;
    }
    ASSIGN_OR_RETURN(const std::optional<std::string> statics,
                     checkStatics(result.q, supporting));
    if (statics.has_value()) {
      check.failure = absl::StrCat("statics: ", *statics);
      continue;
    }
    check.feasible = true;
    check.failure.clear();
    check.q = result.q;
    return check;
  }
  return check;
}

absl::StatusOr<std::optional<std::string>> StanceMotion::knotJump(
    const Eigen::VectorXd& from,
    const Eigen::VectorXd& to,
    const std::vector<int>& movingPads) {
  // A knot may lag its interpolation: twice the steps the intervals allow.
  constexpr double kAllowedLag = 2.0;
  RobotStanceModel& robot = model();
  RETURN_IF_ERROR(robot.setGeneralizedPosition(from));
  const Eigen::Vector3d comFrom = robot.centerOfMass();
  std::vector<Eigen::Vector3d> padsFrom;
  for (int pad = 0; pad < robot.contactSet().size(); ++pad) {
    padsFrom.push_back(robot.padPosition(pad));
  }
  RETURN_IF_ERROR(robot.setGeneralizedPosition(to));
  const double com = (robot.centerOfMass() - comFrom).norm();
  if (com > kAllowedLag * options_.maxKnotComStep) {
    return std::optional<std::string>(
        absl::StrFormat("the CoM moves %.3f m between two knots", com));
  }
  for (const int pad : movingPads) {
    const double moved = (robot.padPosition(pad) - padsFrom[pad]).norm();
    if (moved > kAllowedLag * options_.maxKnotPadStep) {
      return std::optional<std::string>(
          absl::StrFormat("the pad of %s moves %.3f m between two knots",
                          robot.contactSet().candidate(pad).name, moved));
    }
  }
  const Eigen::VectorXd tangent =
      pinocchio::difference(robot.model(), from, to);
  Eigen::Index largest = 0;
  const double joint =
      tangent.tail(tangent.size() - kPositionDim).cwiseAbs().maxCoeff(&largest);
  if (joint > kAllowedLag * options_.maxKnotJointStep) {
    std::string name = "the base";
    for (const ActuatedJoint& actuated : robot.actuatedJoints()) {
      if (actuated.velocityIndex == largest + kPositionDim) {
        name = actuated.name;
      }
    }
    return std::optional<std::string>(
        absl::StrFormat("%s turns %.3f rad between two knots", name, joint));
  }
  return std::optional<std::string>();
}

absl::StatusOr<int> StanceMotion::segmentIntervals(const Eigen::VectorXd& from,
                                                   const Eigen::VectorXd& to,
                                                   double swingLength) {
  RobotStanceModel& robot = model();
  RETURN_IF_ERROR(robot.setGeneralizedPosition(from));
  const Eigen::Vector3d comFrom = robot.centerOfMass();
  std::vector<Eigen::Vector3d> padsFrom;
  for (int pad = 0; pad < robot.contactSet().size(); ++pad) {
    padsFrom.push_back(robot.padPosition(pad));
  }
  RETURN_IF_ERROR(robot.setGeneralizedPosition(to));
  double intervals =
      (robot.centerOfMass() - comFrom).norm() / options_.maxKnotComStep;
  for (int pad = 0; pad < robot.contactSet().size(); ++pad) {
    intervals =
        std::max(intervals, (robot.padPosition(pad) - padsFrom[pad]).norm() /
                                options_.maxKnotPadStep);
  }
  intervals = std::max(intervals, swingLength / options_.maxKnotPadStep);
  intervals = std::max(
      intervals, (to.head<kPositionDim>() - from.head<kPositionDim>()).norm() /
                     options_.maxKnotBaseStep);
  const Eigen::VectorXd tangent =
      pinocchio::difference(robot.model(), from, to);
  intervals = std::max(
      intervals,
      tangent.tail(tangent.size() - kPositionDim).lpNorm<Eigen::Infinity>() /
          options_.maxKnotJointStep);
  return std::clamp(static_cast<int>(std::ceil(intervals)),
                    kMinSegmentIntervals, options_.maxSegmentKnots);
}

namespace {

/** A pad that swings in a segment: a touchdown along a SwingPath, or a pad
 * that only lifts off along its surface's normal. */
struct Swing {
  int candidate = 0;
  // The touchdown's path; std::nullopt for a lift-off alone.
  std::optional<SwingPath> path;
  // [m] Where the pad starts, and the outward normal there.
  Eigen::Vector3d start = Eigen::Vector3d::Zero();
  Eigen::Vector3d startNormal = Eigen::Vector3d::UnitZ();
  // The regions exempt from the clearance near its start and its target.
  PadRegion startRegion;
  std::optional<PadRegion> targetRegion;
  // Whether the pad starts on the terrain and lifts off.
  bool lifts = false;
  // Whether the pad is a surface pad (a sole, the seat, the torso).
  bool surface = false;
  // The normal a touchdown's sole is held level with
  // (SegmentContacts::levelTouchdownSole); std::nullopt: it turns freely.
  std::optional<Eigen::Vector3d> levelNormal;
  // The fraction of the segment its motion spans.
  double span = 1.0;
  // [m] How far it lifts off.
  double clearance = 0.0;
  // [m] Of a point pad that only lifts off: where the segment's end has it,
  // which it goes to from where its lift-off ended over the rest of the
  // segment (a knee rising off a box as the leg folds forward).
  std::optional<Eigen::Vector3d> end;

  /** The pad's target at `fraction` of its motion; past 1, `s` of the
   * segment moves a lift-off with an end toward it. */
  Eigen::Vector3d targetAt(double fraction, double s) const {
    if (path.has_value()) return path->pointAt(fraction);
    const Eigen::Vector3d lifted =
        start + std::clamp(fraction, 0.0, 1.0) * clearance * startNormal;
    if (!end.has_value() || fraction <= 1.0) return lifted;
    const double share = std::clamp((s - span) / (1.0 - span), 0.0, 1.0);
    return lifted + share * (*end - lifted);
  }
  /** Until which fraction its start region is exempt, and from which its
   * target region is. */
  double liftEnd() const { return path.has_value() ? path->liftEnd() : 1.0; }
  double approachStart() const {
    return path.has_value() ? path->approachStart() : 1.0;
  }
  double length() const {
    return path.has_value() ? path->length() : clearance;
  }
};

/** [m] How far a candidate's pad reaches from its center: a rectangle's
 * farthest corner, a point pad's geometry radius. */
double padExtent(const ContactCandidate& candidate) {
  if (!candidate.rectangle.has_value()) return candidate.geomRadius;
  const ContactRectangle& rectangle = *candidate.rectangle;
  return std::max(
      std::hypot(rectangle.xMin, rectangle.yMin),
      std::max(std::hypot(rectangle.xMax, rectangle.yMin),
               std::max(std::hypot(rectangle.xMin, rectangle.yMax),
                        std::hypot(rectangle.xMax, rectangle.yMax))));
}

/** The swing of `candidate` from where `robot` has its pad. */
Swing swingFrom(const RobotStanceModel& robot,
                const TerrainModel& terrain,
                int candidate,
                const StanceMotionOptions& options) {
  Swing swing;
  swing.candidate = candidate;
  swing.start = robot.padPosition(candidate);
  swing.startNormal = terrain.normal(swing.start);
  swing.lifts =
      std::abs(terrain.signedDistance(swing.start)) < options.touchingDistance;
  swing.startRegion =
      PadRegion{.center = swing.start,
                .rotation = robot.padRotation(candidate),
                .rectangle = robot.contactSet().candidate(candidate).rectangle,
                .joints = exemptJointsOf(robot, candidate)};
  swing.clearance = options.swingClearance;
  swing.surface =
      robot.contactSet().candidate(candidate).kind == ContactKind::kSurface;
  return swing;
}

}  // namespace

namespace {

/** What every knot of one segment shares. */
struct SegmentPath {
  Eigen::VectorXd from;
  Eigen::VectorXd to;
  Eigen::Vector3d comFrom = Eigen::Vector3d::Zero();
  Eigen::Vector3d comTo = Eigen::Vector3d::Zero();
  std::vector<PadRegion> heldRegions;
  std::vector<Swing> swings;
  std::vector<int> movingPads;
  // The fraction of the segment by which the posture reference reaches `to`'s
  // (StanceMotionOptions::pointTouchdownPostureLead), in (0, 1].
  double postureLead = 1.0;
};

/** The progress of `path`'s posture reference at fraction `s` of it. */
double postureProgress(const SegmentPath& path, double s) {
  return std::min(1.0, s / path.postureLead);
}

/** Sets `problem` to the knot at fraction `s` of `path`, after the knot
 * `previous` at fraction `previousS`: the posture reference the previous knot
 * moved toward the end by the step's share of the rest of the way (so that
 * consecutive knots stay on one branch of the IK and the last reaches the
 * end), the CoM interpolated, the swinging pads' targets and the exempt
 * regions. */
void configureKnot(const pinocchio::Model& model,
                   const SegmentPath& path,
                   double previousS,
                   const Eigen::VectorXd& previous,
                   double s,
                   StanceIkProblem& problem) {
  const double previousProgress = postureProgress(path, previousS);
  const double share =
      std::clamp((postureProgress(path, s) - previousProgress) /
                     std::max(1.0 - previousProgress, kSegmentEnd),
                 0.0, 1.0);
  problem.postureReference =
      pinocchio::interpolate(model, previous, path.to, share);
  problem.comTarget = path.comFrom + s * (path.comTo - path.comFrom);
  problem.padTargets.clear();
  problem.exemptPads = path.heldRegions;
  for (const Swing& swing : path.swings) {
    const double fraction = s / swing.span;
    PadTarget target{.candidate = swing.candidate,
                     .position = swing.targetAt(fraction, s)};
    // A sole that only lifts off (its contact is not the one the segment
    // makes) stays level while it lifts: free, the posture's pull would tilt
    // it and press its toe or heel into the surface it left. One that goes on
    // to the end's position turns freely after its lift-off.
    if (swing.lifts && !swing.path.has_value() && swing.surface &&
        (!swing.end.has_value() || fraction <= 1.0)) {
      target.normal = swing.startNormal;
    }
    if (swing.levelNormal.has_value()) target.normal = swing.levelNormal;
    problem.padTargets.push_back(target);
    if (swing.lifts && fraction <= swing.liftEnd()) {
      problem.exemptPads.push_back(swing.startRegion);
    }
    if (swing.targetRegion.has_value() && fraction >= swing.approachStart()) {
      problem.exemptPads.push_back(*swing.targetRegion);
    }
  }
}

/** A knot of a segment, or why there is none. */
struct KnotSolution {
  bool feasible = false;
  std::string failure;
  Eigen::VectorXd q;
};

/** The knot at fraction `s` of `path`, solved from `initial`: feasible, its
 * swinging pads out of the terrain and statically feasible on `stance`. */
absl::StatusOr<KnotSolution> solveKnot(StanceMotion& motion,
                                       const StancePlacements& stance,
                                       const SegmentPath& path,
                                       double previousS,
                                       const Eigen::VectorXd& previous,
                                       double s,
                                       const Eigen::VectorXd& initial,
                                       StanceIkProblem& problem) {
  RobotStanceModel& robot = motion.model();
  configureKnot(robot.model(), path, previousS, previous, s, problem);
  ASSIGN_OR_RETURN(const StanceIkResult result,
                   motion.ik().solve(problem, initial));
  const std::string at = absl::StrFormat("s = %.3f: ", s);
  KnotSolution solution;
  if (!result.feasible) {
    solution.failure = absl::StrCat(at, "IK: ", result.failure);
    return solution;
  }
  for (const Swing& swing : path.swings) {
    VLOG(kKnotLogLevel) << absl::StrFormat(
        "knot s=%.3f: the pad of %s is %.3f m from its swing target", s,
        robot.contactSet().candidate(swing.candidate).name,
        (robot.padPosition(swing.candidate) - swing.targetAt(s / swing.span, s))
            .norm());
    const double distance = motion.ik().terrain().signedDistance(
        robot.padPosition(swing.candidate));
    if (distance < -motion.options().padPenetrationTolerance) {
      solution.failure = absl::StrFormat(
          "%sthe swinging pad of %s is %.4f m inside the terrain", at,
          robot.contactSet().candidate(swing.candidate).name, -distance);
      return solution;
    }
  }
  // The IK exempts the swinging pads' start and target regions from its
  // clearance; the certifier exempts only the stance's pads.
  ASSIGN_OR_RETURN(const std::optional<std::string> penetration,
                   motion.checkSphereClearance(result.q, stance));
  if (penetration.has_value()) {
    solution.failure = absl::StrCat(at, *penetration);
    return solution;
  }
  ASSIGN_OR_RETURN(const std::optional<std::string> statics,
                   motion.checkStatics(result.q, stance));
  if (statics.has_value()) {
    solution.failure = absl::StrCat(at, "statics: ", *statics);
    return solution;
  }
  solution.feasible = true;
  solution.q = result.q;
  return solution;
}

/** What is wrong with the step between two knots: a jump (knotJump()) or a
 * sweep through the terrain (StanceMotion::sweptClearance()). Passive data. */
struct StepDefect {
  std::optional<std::string> jump;
  std::optional<std::string> sweep;

  bool any() const { return jump.has_value() || sweep.has_value(); }
  std::string describe() const {
    return jump.has_value() ? *jump : sweep.value_or(std::string());
  }
};

/** The defects of the step from knot `from` to knot `to` of `path` in
 * `stance`. */
absl::StatusOr<StepDefect> stepDefect(StanceMotion& motion,
                                      const StancePlacements& stance,
                                      const SegmentPath& path,
                                      const Eigen::VectorXd& from,
                                      const Eigen::VectorXd& to) {
  StepDefect defect;
  ASSIGN_OR_RETURN(defect.jump, motion.knotJump(from, to, path.movingPads));
  ASSIGN_OR_RETURN(defect.sweep, motion.sweptClearance(from, to, stance));
  return defect;
}

/** The knots that continue a motion between two knots, or why none does.
 * Passive data. */
struct Bridge {
  bool feasible = false;
  std::string failure;
  std::vector<Eigen::VectorXd> knots;
};

/** The knots after `fromKnot` (at fraction `fromS`) up to `toKnot` (at `toS`)
 * of a continuous motion between them: none in between when the step has no
 * defect (stepDefect()), else the knot solved at the middle from the
 * configurations' midpoint and, recursively, the knots of both halves, down to
 * `depth` halvings, where a remaining jump is accepted and a remaining sweep
 * through the terrain is not. Infeasible when a middle knot is: the motion has
 * no continuous way through. */
absl::StatusOr<Bridge> bridge(StanceMotion& motion,
                              const StancePlacements& stance,
                              const SegmentPath& path,
                              double fromS,
                              const Eigen::VectorXd& fromKnot,
                              double toS,
                              const Eigen::VectorXd& toKnot,
                              int depth,
                              StanceIkProblem& problem) {
  ASSIGN_OR_RETURN(const StepDefect defect,
                   stepDefect(motion, stance, path, fromKnot, toKnot));
  Bridge result;
  // At the last halving, a step that still jumps between knots that are all
  // feasible is a fast motion, not a change of branch: the timing law slows
  // the motion down to it. A sweep through the terrain is a collision the MPC
  // would track.
  if (!defect.any() || (depth == 0 && !defect.sweep.has_value())) {
    result.feasible = true;
    result.knots.push_back(toKnot);
    return result;
  }
  if (depth == 0 && defect.sweep.has_value()) {
    result.failure = absl::StrFormat("s = %.4f: %s", toS, *defect.sweep);
    return result;
  }
  const double middleS = 0.5 * (fromS + toS);
  ASSIGN_OR_RETURN(const KnotSolution middle,
                   solveKnot(motion, stance, path, fromS, fromKnot, middleS,
                             pinocchio::interpolate(motion.model().model(),
                                                    fromKnot, toKnot, kMiddle),
                             problem));
  if (!middle.feasible) {
    result.failure = middle.failure;
    return result;
  }
  ASSIGN_OR_RETURN(Bridge first, bridge(motion, stance, path, fromS, fromKnot,
                                        middleS, middle.q, depth - 1, problem));
  if (!first.feasible) return first;
  ASSIGN_OR_RETURN(const Bridge second,
                   bridge(motion, stance, path, middleS, middle.q, toS, toKnot,
                          depth - 1, problem));
  if (!second.feasible) return second;
  first.knots.insert(first.knots.end(), second.knots.begin(),
                     second.knots.end());
  return first;
}

/** With the sweeps checked, the end of a segment that sets a pad down, solved
 * from `knot` toward it, when the touch-down's pad already reached its target
 * there (within the reach tolerance) and the step to the end has no defect:
 * the rest of the motion then happens with the contact made, in the next
 * segment, rather than with the pad hovering on its patch for seconds while
 * the MPC holds it in the air. std::nullopt otherwise. */
absl::StatusOr<std::optional<Eigen::VectorXd>> endOnReach(
    StanceMotion& motion,
    const StancePlacements& stance,
    const SegmentPath& path,
    const SegmentContacts& contacts,
    const Eigen::VectorXd& knot) {
  if (!motion.options().sweep.has_value() || !contacts.touchdown.has_value() ||
      !contacts.endHeld.has_value() || !contacts.endSupporting.has_value()) {
    return std::optional<Eigen::VectorXd>();
  }
  RobotStanceModel& robot = motion.model();
  RETURN_IF_ERROR(robot.setGeneralizedPosition(knot));
  const double reach = (robot.padPosition(contacts.touchdown->candidate) -
                        contacts.touchdown->position)
                           .norm();
  if (reach > motion.ik().options().reachTolerance) {
    return std::optional<Eigen::VectorXd>();
  }
  ASSIGN_OR_RETURN(const MotionCheck end,
                   motion.transition(*contacts.endHeld, *contacts.endSupporting,
                                     knot, {knot}));
  if (!end.feasible) {
    VLOG(kEndLogLevel) << "segment end on reach (" << reach
                       << " m): " << end.failure;
    return std::optional<Eigen::VectorXd>();
  }
  ASSIGN_OR_RETURN(const StepDefect defect,
                   stepDefect(motion, stance, path, knot, end.q));
  if (defect.any()) {
    VLOG(kEndLogLevel) << "segment end on reach (" << reach
                       << " m): " << defect.describe();
    return std::optional<Eigen::VectorXd>();
  }
  VLOG(kEndLogLevel) << "segment end: the touch-down's pad reached its target "
                        "before the end";
  return std::optional<Eigen::VectorXd>(end.q);
}

}  // namespace

absl::StatusOr<SegmentCheck> StanceMotion::segment(
    const StancePlacements& stance,
    const Eigen::VectorXd& from,
    const Eigen::VectorXd& to,
    const SegmentContacts& contacts) {
  RobotStanceModel& robot = model();
  const TerrainModel& terrain = ik_->terrain();
  RETURN_IF_ERROR(robot.setGeneralizedPosition(from));
  SegmentPath path{.from = from, .to = to, .comFrom = robot.centerOfMass()};
  if (contacts.touchdown.has_value()) {
    Swing swing =
        swingFrom(robot, terrain, contacts.touchdown->candidate, options_);
    swing.path = SwingPath::Create(
        SwingEnds{.start = swing.start,
                  .startNormal = swing.startNormal,
                  .lifts = swing.lifts,
                  .target = contacts.touchdown->position,
                  .targetNormal = contacts.touchdown->normal,
                  .extent = padExtent(robot.contactSet().candidate(
                      contacts.touchdown->candidate))},
        options_.swingClearance, terrain);
    swing.targetRegion = exemptRegionOf(*contacts.touchdown, robot);
    if (contacts.levelTouchdownSole && swing.surface) {
      swing.levelNormal = contacts.touchdown->normal;
    }
    if (!swing.surface) path.postureLead = options_.pointTouchdownPostureLead;
    path.swings.push_back(std::move(swing));
  }
  if (contacts.liftOff.has_value() &&
      (!contacts.touchdown.has_value() ||
       contacts.touchdown->candidate != *contacts.liftOff)) {
    Swing swing = swingFrom(robot, terrain, *contacts.liftOff, options_);
    swing.span = kLiftOffFraction;
    if (swing.lifts) path.swings.push_back(std::move(swing));
  }
  double swingLength = 0.0;
  for (const Swing& swing : path.swings) {
    swingLength = std::max(swingLength, swing.length() / swing.span);
    path.movingPads.push_back(swing.candidate);
  }
  ASSIGN_OR_RETURN(const int intervals,
                   segmentIntervals(from, to, swingLength));
  RETURN_IF_ERROR(robot.setGeneralizedPosition(to));
  path.comTo = robot.centerOfMass();
  // A point pad that only lifts off goes on to where the end has it; a sole
  // stays where its lift-off ended, unless the pad the segment sets down is
  // above it on its own leg (the foot of a knee that goes onto a box, which
  // cannot stay by the floor).
  std::vector<pinocchio::JointIndex> touchdownLeg;
  if (contacts.touchdown.has_value()) {
    touchdownLeg = ownJointsOf(robot, contacts.touchdown->candidate);
  }
  for (Swing& swing : path.swings) {
    if (swing.path.has_value()) continue;
    const bool belowTouchdown =
        std::find(touchdownLeg.begin(), touchdownLeg.end(),
                  robot.padJoint(swing.candidate)) != touchdownLeg.end();
    if (!swing.surface || belowTouchdown) {
      swing.end = robot.padPosition(swing.candidate);
    }
  }
  if (contacts.comGoal.has_value()) {
    path.comTo.head<kHorizontalPositionDim>() = *contacts.comGoal;
    path.comTo.z() = path.comFrom.z();
  }
  ASSIGN_OR_RETURN(std::vector<Eigen::Vector2d> polygon, polygonOf(stance));
  StanceIkProblem problem = holdingProblem(stance, std::move(polygon));
  problem.postureWeight = options_.segmentPostureWeight;
  path.heldRegions = problem.exemptPads;

  // The knots advance s by at most 1 / intervals; a knot that jumps (it moves
  // more than twice the knot limits, as when a near-straight leg lifts its
  // foot) is retried at half the step, down to 1 / maxSegmentKnots, and then
  // bridged by knots solved between (bridge()).
  constexpr double kEndOfSegment = 1.0 - 1.0e-9;
  constexpr int kBridgeDepth = 4;
  const double largestStep = 1.0 / intervals;
  const double smallestStep = 1.0 / options_.maxSegmentKnots;
  SegmentCheck check;
  Eigen::VectorXd q = from;
  double s = 0.0;
  double ds = largestStep;
  while (s < kEndOfSegment) {
    const double next = std::min(1.0, s + ds);
    Eigen::VectorXd knot = to;
    if (next >= kEndOfSegment && contacts.endHeld.has_value() &&
        contacts.endSupporting.has_value()) {
      ASSIGN_OR_RETURN(
          MotionCheck end,
          transition(*contacts.endHeld, *contacts.endSupporting, q, {to, q}));
      // With the sweeps checked, an end toward `to` that the last knot cannot
      // reach without a defect (the free shin folding the other way through a
      // box's edge) is solved again toward the last knot alone, and taken when
      // the step to it has none.
      if (end.feasible && options_.sweep.has_value()) {
        ASSIGN_OR_RETURN(const StepDefect defect,
                         stepDefect(*this, stance, path, q, end.q));
        if (defect.any()) {
          ASSIGN_OR_RETURN(
              const MotionCheck near,
              transition(*contacts.endHeld, *contacts.endSupporting, q, {q}));
          StepDefect nearDefect;
          if (near.feasible) {
            ASSIGN_OR_RETURN(nearDefect,
                             stepDefect(*this, stance, path, q, near.q));
          }
          VLOG(kEndLogLevel) << absl::StrFormat(
              "segment end: %s; toward the last knot: %s", defect.describe(),
              !near.feasible     ? near.failure
              : nearDefect.any() ? nearDefect.describe()
                                 : std::string("continues"));
          if (near.feasible && !nearDefect.any()) end = near;
        }
      }
      if (end.feasible) {
        knot = end.q;
      } else if (contacts.endMustBeSolved) {
        check.failure = absl::StrCat("the end: ", end.failure);
        return check;
      }
    }
    if (next < kEndOfSegment) {
      ASSIGN_OR_RETURN(const KnotSolution solution,
                       solveKnot(*this, stance, path, s, q, next, q, problem));
      if (!solution.feasible) {
        check.failure = solution.failure;
        return check;
      }
      knot = solution.q;
    }
    ASSIGN_OR_RETURN(const StepDefect defect,
                     stepDefect(*this, stance, path, q, knot));
    VLOG(kKnotLogLevel) << absl::StrFormat(
        "segment s=%.3f ds=%.4f%s", next, ds,
        defect.any() ? " DEFECT " + defect.describe() : std::string());
    if (defect.any() && ds > smallestStep) {
      ds = std::max(smallestStep, ds * 0.5);
      continue;
    }
    if (defect.any()) {
      ASSIGN_OR_RETURN(
          const Bridge bridged,
          bridge(*this, stance, path, s, q, next, knot, kBridgeDepth, problem));
      if (!bridged.feasible) {
        check.failure = absl::StrFormat(
            "s = %.3f: the step to the next knot fails (%s), and no motion "
            "between the knots continues it: %s",
            next, defect.describe(), bridged.failure);
        return check;
      }
      // A bridge to the end can bring the touch-down's pad to its target
      // well before the body reaches the end's posture: the segment ends at
      // the first bridged knot it can end on (endOnReach()).
      bool endedEarly = false;
      for (size_t b = 0; b + 1 < bridged.knots.size() && !endedEarly; ++b) {
        check.knots.push_back(bridged.knots[b]);
        ASSIGN_OR_RETURN(
            std::optional<Eigen::VectorXd> early,
            endOnReach(*this, stance, path, contacts, bridged.knots[b]));
        if (early.has_value()) {
          check.knots.push_back(*std::move(early));
          endedEarly = true;
        }
      }
      if (endedEarly) break;
      check.knots.push_back(bridged.knots.back());
    } else if (pinocchio::difference(robot.model(), q, knot)
                   .lpNorm<Eigen::Infinity>() < kSameKnot) {
      // A knot where the last one is (the end solved from a last knot that
      // already holds it) adds no motion, and the timing law would put the two
      // a millisecond apart: the end replaces the last knot.
      if (next >= kEndOfSegment && !check.knots.empty()) {
        check.knots.back() = knot;
      } else if (next >= kEndOfSegment) {
        check.knots.push_back(knot);
      }
    } else {
      check.knots.push_back(knot);
    }
    q = knot;
    s = next;
    ds = std::min(largestStep, ds + ds);
    if (s < kEndOfSegment) {
      ASSIGN_OR_RETURN(std::optional<Eigen::VectorXd> early,
                       endOnReach(*this, stance, path, contacts, q));
      if (early.has_value()) {
        // An end where the knot is replaces it, as at the segment's end.
        if (pinocchio::difference(robot.model(), q, *early)
                .lpNorm<Eigen::Infinity>() < kSameKnot) {
          check.knots.back() = *std::move(early);
        } else {
          check.knots.push_back(*std::move(early));
        }
        break;
      }
    }
  }
  check.feasible = true;
  return check;
}

}  // namespace ocs2::humanoid::multi_contact
