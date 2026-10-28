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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/status/statusor.h"

#include "humanoid_multi_contact/collision/CollisionModel.h"
#include "humanoid_multi_contact/stability/RobotStanceModel.h"
#include "humanoid_multi_contact/terrain/TerrainModel.h"

namespace ocs2::humanoid::multi_contact {

/** [rad] How far a point pad's normal may tilt from the surface normal by
 * default (design section 3.1, max_pad_tilt). */
inline constexpr double kDefaultMaxPadTilt = 0.5235987755982988;  // 30 deg

/** [m] How far a held pad may be from its target (design section 5.1). */
inline constexpr double kDefaultReachTolerance = 0.005;

/** A contact the inverse kinematics holds on the terrain. Passive data. */
struct ContactTarget {
  // The candidate of the contact set.
  int candidate = 0;
  // [m] The pad's center in the world.
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  // The surface's outward unit normal: the pad's z axis (ContactCandidate).
  Eigen::Vector3d normal = Eigen::Vector3d::UnitZ();
  // A surface pad's x axis, a unit vector orthogonal to `normal`: its yaw on
  // the surface. Unused for a point pad, whose normal only stays in the cone
  // of the maximal tilt.
  Eigen::Vector3d xAxis = Eigen::Vector3d::UnitX();
};

/** A pad the inverse kinematics pulls toward a point, softly: a swinging
 * contact. Passive data. */
struct PadTarget {
  int candidate = 0;
  // [m]
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  // A unit vector the pad's z axis (its outward surface normal, the opposite
  // of ContactCandidate::padNormal) is pulled toward, softly, at the
  // orientation weight: a sole that lifts off stays level. std::nullopt: the
  // pad turns freely.
  std::optional<Eigen::Vector3d> normal;
};

/** One inverse kinematics problem (design section 5.1). Passive data. */
struct StanceIkProblem {
  // The contacts held, each within the reach tolerance of its target.
  std::vector<ContactTarget> contacts;
  // The polygon the CoM's horizontal position must stay in, `comMargin` from
  // its edges (counterclockwise, StaticEquilibrium::supportPolygon()); empty:
  // no CoM constraint.
  std::vector<Eigen::Vector2d> comPolygon;
  // [m]
  double comMargin = 0.0;
  // A point the CoM is pulled toward, softly.
  std::optional<Eigen::Vector3d> comTarget;
  // Swinging pads pulled toward points, softly.
  std::vector<PadTarget> padTargets;
  // The pose the redundancy is pulled toward (a generalized position), and
  // the weight of the pull [1/rad^2] (StanceIkOptions::postureWeight when
  // absent; the base position's is a tenth of it).
  Eigen::VectorXd postureReference;
  std::optional<double> postureWeight;
  // The pads near which the collision spheres are exempt: the held contacts
  // and the pads about to touch or just lifted.
  std::vector<PadRegion> exemptPads;
};

/** The defaults of StanceIkOptions; its fields say what each is. */
inline constexpr int kDefaultMaxIterations = 100;
inline constexpr double kDefaultSurfaceTiltTolerance = 0.02;
inline constexpr double kDefaultTerrainClearance = 0.01;
inline constexpr double kDefaultSelfClearance = 0.01;
inline constexpr double kDefaultActivationDistance = 0.08;
inline constexpr double kDefaultSelfActivationDistance = 0.05;
inline constexpr double kDefaultContactWeight = 1.0e3;
inline constexpr double kDefaultOrientationWeight = 1.0e2;
inline constexpr double kDefaultPadTargetWeight = 1.0e3;
inline constexpr double kDefaultComTargetWeight = 1.0e2;
inline constexpr double kDefaultPostureWeight = 1.0e-1;
inline constexpr double kDefaultDamping = 1.0e-3;
inline constexpr double kDefaultContactPenalty = 1.0e5;
inline constexpr double kDefaultSlackPenalty = 1.0e3;
inline constexpr double kDefaultJointLimitMargin = 0.1;
inline constexpr double kDefaultMaxJointStep = 0.2;
inline constexpr double kDefaultMaxBaseStep = 0.08;
inline constexpr double kDefaultConvergedStep = 1.0e-4;

/** The weights and limits of StanceInverseKinematics. */
struct StanceIkOptions {
  int maxIterations = kDefaultMaxIterations;
  double reachTolerance = kDefaultReachTolerance;
  // [m] How deep a collision sphere that is not exempt may be in the terrain
  // for the result to be feasible: the certifier's contact tolerance by
  // default.
  double penetrationTolerance = kDefaultReachTolerance;
  // [rad] Of a surface pad's normal from the surface normal.
  double surfaceTiltTolerance = kDefaultSurfaceTiltTolerance;
  double maxPadTilt = kDefaultMaxPadTilt;
  // [m] The clearance the spheres keep from the terrain and from each other.
  double terrainClearance = kDefaultTerrainClearance;
  double selfClearance = kDefaultSelfClearance;
  // [m] The pad neighborhood exempt from the terrain clearance.
  double padNeighborhood = kDefaultPadNeighborhood;
  // [m] Spheres farther than this from the terrain, and pairs of links whose
  // closest spheres are farther than this apart, do not enter an iteration's
  // QP: a step moves no point by more than about a decimeter.
  double activationDistance = kDefaultActivationDistance;
  double selfActivationDistance = kDefaultSelfActivationDistance;
  // The least-squares weights: held contacts [1/m^2], their orientation
  // [1/rad^2], soft pad and CoM targets [1/m^2], the posture [1/rad^2, 1/m^2].
  double contactWeight = kDefaultContactWeight;
  double orientationWeight = kDefaultOrientationWeight;
  double padTargetWeight = kDefaultPadTargetWeight;
  double comTargetWeight = kDefaultComTargetWeight;
  double postureWeight = kDefaultPostureWeight;
  double damping = kDefaultDamping;
  // The exact penalties, per meter or radian: of a held pad off its target or
  // out of its normal cone, and of a violated CoM polygon or clearance. The
  // pads' is the larger, so that the IK never trades a contact for balance.
  double contactPenalty = kDefaultContactPenalty;
  double slackPenalty = kDefaultSlackPenalty;
  // [rad] How far inside its limits the IK keeps a joint, where the range
  // allows: a straight knee at its limit is a singularity from which lifting
  // the foot snaps the leg into a bent one.
  double jointLimitMargin = kDefaultJointLimitMargin;
  // [rad, m] The largest step of one iteration per joint and base coordinate.
  double maxJointStep = kDefaultMaxJointStep;
  double maxBaseStep = kDefaultMaxBaseStep;
  // [rad or m] An iteration whose largest step is below this has converged.
  double convergedStep = kDefaultConvergedStep;
};

/** What StanceInverseKinematics::solve() found. Passive data. */
struct StanceIkResult {
  Eigen::VectorXd q;
  int iterations = 0;
  // Why the iterations stopped: "converged" (a step below convergedStep),
  // "stalled" (no step decreased the merit), "qp" (HPIPM failed on a
  // linearization) or "iterations".
  std::string termination;
  // Whether every condition below holds.
  bool feasible = false;
  // What fails, for messages; empty when feasible.
  std::string failure;
  // [m] The largest distance of a held pad to its target.
  double maxReachError = 0.0;
  // [rad] The largest tilt of a held pad's normal from its surface's.
  double maxPointTilt = 0.0;
  double maxSurfaceTilt = 0.0;
  // [m] The CoM's signed distance to the polygon's boundary; std::nullopt
  // without a polygon.
  std::optional<double> comMargin;
  // [m] The smallest terrain clearance of the spheres that are not exempt,
  // and the smallest self clearance.
  std::optional<double> terrainClearance;
  std::optional<double> selfClearance;
  std::string terrainClearanceLink;
  std::string selfClearanceLinks;
};

/**
 * The whole-body inverse kinematics of a stance (design section 5.1): a
 * configuration whose held pads reach their targets with their normals on the
 * surfaces, whose CoM lies in a polygon with a margin, within the joint limits
 * and collision-free, closest to a posture reference.
 *
 * A sequential QP over Pinocchio: each iteration linearizes the pads, the CoM
 * and the collision spheres at the current configuration and solves one dense
 * QP on HPIPM (solveQuadraticProgram()) for the step in the tangent space,
 * within a trust region that shrinks when a step increases the merit (the
 * least squares plus the exact penalty of the violations) and grows when one
 * decreases it. The held pads are equalities with an exact L1 penalty (and a
 * least-squares term), the CoM polygon and the clearances inequalities with a
 * smaller exact penalty, so every QP is feasible and a contact is never traded
 * for balance; the soft targets and the posture are least squares; the joint
 * limits and the trust region are bounds. Each trial step is corrected back
 * onto the held pads before the merit judges it (a second-order correction).
 * The result is judged against the
 * exact conditions at the final configuration (StanceIkResult::feasible).
 *
 * Not thread-safe: it sets the configuration of its RobotStanceModel.
 */
class StanceInverseKinematics {
 public:
  /** InvalidArgument for options that are not positive where they must be;
   * `collision` must be built for `model`. */
  static absl::StatusOr<std::unique_ptr<StanceInverseKinematics>> Create(
      std::unique_ptr<RobotStanceModel> model,
      TerrainModel terrain,
      CollisionModel collision,
      const StanceIkOptions& options);

  StanceInverseKinematics(const StanceInverseKinematics&) = delete;
  StanceInverseKinematics& operator=(const StanceInverseKinematics&) = delete;
  ~StanceInverseKinematics() = default;

  /**
   * Solves `problem` from the generalized position `initial`, clamped into
   * the joint limits. InvalidArgument for vectors of the wrong size or a
   * candidate out of range; Internal when HPIPM fails. An infeasible problem
   * is not an error: the result says what fails.
   */
  absl::StatusOr<StanceIkResult> solve(const StanceIkProblem& problem,
                                       const Eigen::VectorXd& initial);

  /** Judges the configuration `q` against `problem` without moving it. */
  absl::StatusOr<StanceIkResult> evaluate(const StanceIkProblem& problem,
                                          const Eigen::VectorXd& q);

  RobotStanceModel& model() { return *model_; }
  const RobotStanceModel& model() const { return *model_; }
  const TerrainModel& terrain() const { return terrain_; }
  const CollisionModel& collision() const { return collision_; }
  const StanceIkOptions& options() const { return options_; }

  /** `q` with its joints clamped into their limits, `jointLimitMargin` inside
   * them where the range is wider than twice the margin. */
  Eigen::VectorXd clampToLimits(const Eigen::VectorXd& q) const;

 private:
  StanceInverseKinematics(std::unique_ptr<RobotStanceModel> model,
                          TerrainModel terrain,
                          CollisionModel collision,
                          const StanceIkOptions& options);

  /** One SQP step at the model's configuration, its bounds scaled by
   * `radius` in (0, 1]. */
  absl::StatusOr<Eigen::VectorXd> step(const StanceIkProblem& problem,
                                       double radius);

  /** `trial` with its held pads corrected onto their targets by minimum-norm
   * Gauss-Newton steps (the second-order correction of an SQP step, against
   * the Maratos effect of the contacts' exact penalty); sets the model's
   * configuration to the result. */
  absl::StatusOr<Eigen::VectorXd> correctContacts(
      const StanceIkProblem& problem, const Eigen::VectorXd& trial);

  std::unique_ptr<RobotStanceModel> model_;
  TerrainModel terrain_;
  CollisionModel collision_;
  StanceIkOptions options_;
};

}  // namespace ocs2::humanoid::multi_contact
