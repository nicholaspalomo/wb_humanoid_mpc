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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"

#include "humanoid_multi_contact/collision/CollisionModel.h"
#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/planner/StanceInverseKinematics.h"
#include "humanoid_multi_contact/stability/JointLimits.h"
#include "humanoid_multi_contact/stability/StabilityCertifier.h"
#include "humanoid_multi_contact/stability/StaticEquilibrium.h"
#include "humanoid_multi_contact/terrain/PatchFrame.h"

namespace ocs2::humanoid::multi_contact {

/** One contact of a planned stance: a candidate on a point of a face of the
 * terrain. Passive data. */
struct ContactPlacement {
  int candidate = 0;
  // The primitive, and its box face or std::nullopt on a capsule.
  int primitive = 0;
  std::optional<BoxFace> face;
  // [m] The pad's center; the surface's outward normal; a surface pad's x
  // axis on the surface.
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
  Eigen::Vector3d xAxis = Eigen::Vector3d::UnitX();
};

/** A stance of the stance graph: its placements, ordered by candidate, one per
 * candidate. */
using StancePlacements = std::vector<ContactPlacement>;

/** The contact mode of `stance`. */
ContactMask maskOf(const StancePlacements& stance);

/** A key that names `stance` up to 5 cm, for duplicate detection. */
std::string stanceKey(const StancePlacements& stance);

/** `stance` with `placement` added (its candidate must not be in it), or the
 * placement of `candidate` removed. */
StancePlacements withPlacement(const StancePlacements& stance,
                               const ContactPlacement& placement);
StancePlacements withoutCandidate(const StancePlacements& stance,
                                  int candidate);

/** The placement of `candidate` in `stance`, or std::nullopt. */
std::optional<ContactPlacement> placementOf(const StancePlacements& stance,
                                            int candidate);

/** The defaults of StanceMotionOptions; its fields say what each is. */
inline constexpr double kDefaultComMargin = 0.03;
inline constexpr double kDefaultSwingClearance = 0.06;
inline constexpr double kDefaultMaxKnotComStep = 0.08;
inline constexpr double kDefaultMaxKnotBaseStep = 0.08;
inline constexpr double kDefaultMaxKnotPadStep = 0.08;
inline constexpr double kDefaultMaxKnotJointStep = 0.25;
inline constexpr int kDefaultMaxSegmentKnots = 48;
inline constexpr double kDefaultTouchingDistance = 0.01;
inline constexpr double kDefaultPadPenetrationTolerance = 0.005;
inline constexpr double kDefaultSweepJointResolution = 0.02;
inline constexpr double kDefaultSweepPositionResolution = 0.01;
inline constexpr int kDefaultSweepMinIntervals = 10;
inline constexpr double kDefaultSweepComMarginSlack = 0.01;
inline constexpr double kDefaultSweepTolerance = 0.0025;

/** How finely StanceMotion samples the configurations between two knots
 * (StanceMotion::sweptClearance()). Passive data. */
struct SweepResolution {
  // [rad] The largest turn of a joint or of the base between two samples.
  double joint = kDefaultSweepJointResolution;
  // [m] The largest move of the base between two samples.
  double position = kDefaultSweepPositionResolution;
  // The fewest intervals a step is sampled in: a knee lifting off a box dips
  // the end of its thigh in for a tenth of a step that turns no joint far.
  int minIntervals = kDefaultSweepMinIntervals;
  // [m] How much less than the knots' CoM margin (StanceMotionOptions::
  // comMargin) the CoM between two knots may keep inside the stance's static
  // equilibrium polygon: a fold of the arms between two knots swings the CoM
  // centimeters off the line between theirs.
  double comMarginSlack = kDefaultSweepComMarginSlack;
  // [m] How far a collision sphere between two knots may penetrate the
  // terrain: half the certifier's contact distance tolerance, which the
  // tracked robot then still keeps, for the millimeters by which a thigh
  // settling onto a box beside its knee's pad dips between two knots that
  // both clear it.
  double tolerance = kDefaultSweepTolerance;
};

/**
 * The configuration at `fraction` in [0, 1] between the knots `from` and `to`
 * as the MPC's reference interpolates a plan's states
 * (MpcRobotModelBase::interpolateStateTrajectory()): the base position and the
 * joint positions linearly, the base orientation, the quaternion (x, y, z, w)
 * after the position, slerped along the shorter way. Not pinocchio's
 * interpolate(), which moves the free flyer along a screw.
 */
Eigen::VectorXd knotInterpolation(const Eigen::VectorXd& from,
                                  const Eigen::VectorXd& to,
                                  double fraction);

/** The settings of StanceMotion. */
struct StanceMotionOptions {
  // [m] The CoM's margin inside the static equilibrium polygon of every
  // planned configuration.
  double comMargin = kDefaultComMargin;
  // Whether a configuration must hold its stance within `torqueMargin` of the
  // effort limits (condition 1, JointLimits::kPositionAndEffort) or only
  // within the joint position limits (kPositionOnly: no torque LP), and that
  // margin.
  JointLimits jointLimits = JointLimits::kPositionAndEffort;
  double torqueMargin = 1.0;
  // Whether to check the exact static margin of a transition configuration's
  // stance (its polygon at the reached pads) against `exactStaticMargin`.
  std::optional<double> exactStaticMargin;
  StaticEquilibriumOptions equilibrium;
  // [m] How high a swinging pad lifts off and approaches its target.
  double swingClearance = kDefaultSwingClearance;
  // [m, rad] The largest motion between two knots of a segment: of the CoM,
  // the base, a pad and a joint.
  double maxKnotComStep = kDefaultMaxKnotComStep;
  double maxKnotBaseStep = kDefaultMaxKnotBaseStep;
  double maxKnotPadStep = kDefaultMaxKnotPadStep;
  double maxKnotJointStep = kDefaultMaxKnotJointStep;
  int maxSegmentKnots = kDefaultMaxSegmentKnots;
  // [1/rad^2] The posture weight of a segment's knots, which follow the
  // interpolation between its ends closely (StanceIkProblem::postureWeight).
  double segmentPostureWeight = 1.0;
  // The fraction of a segment that sets a point pad down (a knee onto a box)
  // by which its knots' posture reference reaches the end's, in (0, 1]: below
  // 1 the leg folds into the end's shape first and the pad then travels with
  // it, rather than reaching the box's edge half folded, where the shin
  // cannot pass. 1: the reference reaches the end at the end, as in every
  // other segment.
  double pointTouchdownPostureLead = 1.0;
  // [m] Below this distance to the terrain, a pad that leaves its stance
  // lifts off along the surface normal.
  double touchingDistance = kDefaultTouchingDistance;
  // [m] How far below the terrain a swinging pad may dip.
  double padPenetrationTolerance = kDefaultPadPenetrationTolerance;
  // The candidates whose pads a transition configuration keeps out of the
  // terrain (within padPenetrationTolerance) when it does not hold them: the
  // scenario's allowed contacts, whose pads the segments swing.
  std::vector<int> clearedPads;
  // [m] How far a collision sphere of a segment's knot may penetrate the
  // terrain outside `padNeighborhood` of the stance's pads: the certifier's
  // collision_free test (its contact distance tolerance and pad neighborhood),
  // which, unlike the IK's clearance, exempts no swinging pad.
  double sphereTolerance = kDefaultContactDistanceTolerance;
  double padNeighborhood = kDefaultPadNeighborhood;
  // How finely the configurations between two consecutive knots of a segment
  // are checked like its knots (sweptClearance()): the MPC tracks the
  // interpolation between them, which can sweep a shin through an edge that
  // both knots clear. A step whose sweep collides is halved like a jump, then
  // bridged by knots solved between, and the segment fails when a sweep still
  // collides at the deepest halving. std::nullopt: only the knots are checked.
  std::optional<SweepResolution> sweep;
};

/** A configuration found by StanceMotion, or why there is none. Passive
 * data. */
struct MotionCheck {
  bool feasible = false;
  std::string failure;
  Eigen::VectorXd q;
};

/** The knots of a motion inside one stance, or why there is none. Passive
 * data. */
struct SegmentCheck {
  bool feasible = false;
  std::string failure;
  // The knots after the start, the last one the end configuration.
  std::vector<Eigen::VectorXd> knots;
};

/** Where a segment's swinging pads go. Passive data. */
struct SegmentContacts {
  // The candidate whose contact the segment's stance broke at its start: it
  // lifts off, unless it is `touchdown`'s.
  std::optional<int> liftOff;
  // The contact made at the segment's end: its pad swings to it.
  std::optional<ContactPlacement> touchdown;
  // The stances of the transition the segment ends in (transition()): when
  // set, the end is solved again from the last knot toward `to`, so that it
  // continues the motion rather than jumping to another solution of the same
  // stance; `to` is the end when that fails.
  std::optional<StancePlacements> endHeld;
  std::optional<StancePlacements> endSupporting;
  // Whether the end must be solved from the last knot: `to` is then only a
  // posture guide, not a transition, and the segment fails when the end
  // cannot be solved.
  bool endMustBeSolved = false;
  // [m] Where the CoM's horizontal position goes over the segment instead of
  // to `to`'s: the weight shifts there gradually, as before a contact is
  // broken.
  std::optional<Eigen::Vector2d> comGoal;
  // Whether a sole that swings to its touchdown is held level with its target
  // surface the whole way (the free leg of a climb folds far, and a free sole
  // flips its ankle on the way); otherwise it turns freely until it lands.
  bool levelTouchdownSole = false;
};

/**
 * The motions of the stance graph planner (design section 5.1): the IK of a
 * stance with its CoM in the static equilibrium polygon of a supporting
 * stance, the statics check of a configuration (static equilibrium, torque
 * feasibility and, optionally, the exact static margin), and the motion inside
 * a stance from one configuration to another, by projecting the interpolation
 * between them onto the stance's constraints knot by knot while the pads that
 * leave or join the stance swing (SwingPath). The polygons are cached by
 * stance.
 *
 * Not thread-safe: it moves its inverse kinematics' model.
 */
class StanceMotion {
 public:
  /** InvalidArgument for a negative margin or step. */
  static absl::StatusOr<std::unique_ptr<StanceMotion>> Create(
      std::unique_ptr<StanceInverseKinematics> ik,
      const StanceMotionOptions& options);

  StanceMotion(const StanceMotion&) = delete;
  StanceMotion& operator=(const StanceMotion&) = delete;
  ~StanceMotion() = default;

  /** The static equilibrium polygon of `stance` at its placements (empty when
   * no CoM position is in equilibrium); Internal when an LP fails. */
  absl::StatusOr<std::vector<Eigen::Vector2d>> polygonOf(
      const StancePlacements& stance);

  /**
   * A configuration that holds `held` with its CoM in the polygon of
   * `supporting` (a subset of `held`) with the CoM margin, collision-free,
   * with no free pad in the terrain (padInTerrain()) and statically feasible
   * on `supporting` (checkStatics()), solved from `start`
   * toward each of `references` in turn until one is feasible; when the IK
   * fails, it is solved again in two stages, the pads without the CoM's
   * polygon first.
   */
  absl::StatusOr<MotionCheck> transition(
      const StancePlacements& held,
      const StancePlacements& supporting,
      const Eigen::VectorXd& start,
      const std::vector<Eigen::VectorXd>& references);

  /** Why `q` cannot hold `stance` statically, or std::nullopt when it can
   * (design section 2, condition 1, with the options' checks). */
  absl::StatusOr<std::optional<std::string>> checkStatics(
      const Eigen::VectorXd& q, const StancePlacements& stance);

  /** Why a collision sphere of `q` penetrates the terrain by more than the
   * sphere tolerance outside the pad neighborhood of `stance`'s pads (the
   * certifier's collision_free test), or std::nullopt when none does. Every
   * knot of a segment passes it. */
  absl::StatusOr<std::optional<std::string>> checkSphereClearance(
      const Eigen::VectorXd& q, const StancePlacements& stance);

  /** Why a configuration between the knots `from` and `to`, as the MPC's
   * reference interpolates them (knotInterpolation()), fails
   * checkSphereClearance() on `stance` or keeps its CoM less than the CoM
   * margin less the sweep's slack inside the stance's polygon, or why the
   * straight path of a free pad between them goes into the terrain
   * (padChordCollision()), or
   * std::nullopt when neither does or the options set no
   * StanceMotionOptions::sweep, with the sweep's tolerance in place of the
   * sphere tolerance. Samples the interior at the sweep's resolution: one
   * sphere clearance per sample. */
  absl::StatusOr<std::optional<std::string>> sweptClearance(
      const Eigen::VectorXd& from,
      const Eigen::VectorXd& to,
      const StancePlacements& stance);

  /** Why the pad of a candidate of StanceMotionOptions::clearedPads that
   * `held` does not hold is inside the terrain at `q`, or std::nullopt when
   * none is. Sets the model's configuration. */
  std::optional<std::string> padInTerrain(const Eigen::VectorXd& q,
                                          const StancePlacements& held);

  /** The motion inside `stance` from `from` to `to`, both of which hold it,
   * with the swings of `contacts`; its last knot is the end configuration,
   * `to` or the one solved from the last knot (SegmentContacts::endHeld). */
  absl::StatusOr<SegmentCheck> segment(const StancePlacements& stance,
                                       const Eigen::VectorXd& from,
                                       const Eigen::VectorXd& to,
                                       const SegmentContacts& contacts);

  /** The IK problem of holding `held` with the CoM in `polygon`. */
  StanceIkProblem holdingProblem(const StancePlacements& held,
                                 std::vector<Eigen::Vector2d> polygon) const;

  /** Why the step from knot `from` to knot `to` is more than twice the
   * intervals' limits allow for the CoM, the joints and the pads of
   * `movingPads`, or std::nullopt when it is not: a knot that could not follow
   * the interpolation. */
  absl::StatusOr<std::optional<std::string>> knotJump(
      const Eigen::VectorXd& from,
      const Eigen::VectorXd& to,
      const std::vector<int>& movingPads);

  StanceInverseKinematics& ik() { return *ik_; }
  RobotStanceModel& model() { return ik_->model(); }
  const StanceMotionOptions& options() const { return options_; }

 private:
  StanceMotion(std::unique_ptr<StanceInverseKinematics> ik,
               const StanceMotionOptions& options);

  /** Why the straight path between knots `from` and `to` of the sphere of a
   * pad of StanceMotionOptions::clearedPads that `stance` does not hold - the
   * path the MPC's reference has a pad follow between two knots of a long
   * swing (PlannedMotionSettings::arcSwingDurationLimit) - goes deeper into
   * the terrain than the sweep's tolerance at one of `intervals` - 1 interior
   * samples, or std::nullopt. A point pad's geometry is its sphere
   * (ContactCandidate::geomRadius) behind its center, as the MPC's
   * terrain_non_penetration barrier keeps it out; a sole's, its rectangle's
   * corners, turned with the pad (its orientation slerped). Requires
   * options().sweep. */
  absl::StatusOr<std::optional<std::string>> padChordCollision(
      const Eigen::VectorXd& from,
      const Eigen::VectorXd& to,
      const StancePlacements& stance,
      int intervals);

  /** checkSphereClearance() with the tolerance `tolerance` [m]. */
  absl::StatusOr<std::optional<std::string>> sphereClearance(
      const Eigen::VectorXd& q,
      const StancePlacements& stance,
      double tolerance);

  /** The number of intervals of the segment from `from` to `to` with a swing
   * of `swingLength`. */
  absl::StatusOr<int> segmentIntervals(const Eigen::VectorXd& from,
                                       const Eigen::VectorXd& to,
                                       double swingLength);

  std::unique_ptr<StanceInverseKinematics> ik_;
  StanceMotionOptions options_;
  absl::flat_hash_map<std::string, std::vector<Eigen::Vector2d>> polygons_;
};

/**
 * Whether the pads of two placements on one surface (normals within about 25
 * degrees, planes within 5 cm) come closer than `margin` [m]: two point pads
 * by their geom radii, a point and a surface pad by the point's distance to the
 * rectangle, two surface pads by the separating axes of their rectangles. Pads
 * on different surfaces never overlap.
 */
bool placementsOverlap(const ContactPlacement& first,
                       const ContactPlacement& second,
                       const ContactSet& contactSet,
                       double margin);

/** The pad region of `placement` (CollisionModel), for `contactSet`'s
 * candidate. */
PadRegion padRegionOf(const ContactPlacement& placement,
                      const ContactSet& contactSet);

/** The pad region of `placement` with the joints whose spheres it exempts
 * (exemptJointsOf()), for the IK's clearance. */
PadRegion exemptRegionOf(const ContactPlacement& placement,
                         const RobotStanceModel& model);

}  // namespace ocs2::humanoid::multi_contact
