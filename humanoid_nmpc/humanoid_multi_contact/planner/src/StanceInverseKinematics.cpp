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

#include "humanoid_multi_contact/planner/StanceInverseKinematics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Eigen/Cholesky"
#include "Eigen/Geometry"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "ocs2_core/Dimensions.h"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/spatial/skew.hpp"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_multi_contact/stability/ConvexPolygon.h"
#include "humanoid_multi_contact/stability/LinearProgram.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// The first velocity index of the base's rotation (Pinocchio's free flyer:
// the linear velocity, then the angular one, in the base frame).
constexpr int kBaseRotationIndex = kPositionDim;
// The quadratic weight of a slack, which keeps the QP's Hessian positive
// definite; the exact penalty is the linear one.
constexpr double kSlackHessian = 1.0;
// The inner cone a point pad's normal is constrained to, as a fraction of the
// maximal tilt, so that the judged tilt keeps a little room.
constexpr double kTiltConeFraction = 0.9;
// [m] How far below the required CoM margin a solution may end: the QP's
// tolerance.
constexpr double kComMarginSlack = 1.0e-3;
// The trust region: shrunk by this factor when a step increases the merit,
// grown by this one when a step decreases it; below this scale of the step
// bounds, the iterations have stalled.
constexpr double kRadiusShrink = 0.25;
constexpr double kRadiusGrowth = 2.0;
constexpr double kSmallestRadius = 1.0e-3;
// HPIPM's iterations and tolerance on an IK step.
constexpr int kQpMaxIterations = 100;
constexpr double kQpTolerance = 1.0e-7;
// A step that decreases the merit by less than this fraction of it ends the
// iterations.
constexpr double kConvergedMeritDecrease = 1.0e-9;
// A polygon of fewer vertices has no inside.
constexpr size_t kMinPolygonVertices = 3;

constexpr double kInfinity = std::numeric_limits<double>::infinity();

/** world_R_target of a surface pad: z along the normal, x along `xAxis`. */
Eigen::Matrix3d surfaceRotation(const ContactTarget& target) {
  Eigen::Matrix3d rotation;
  rotation.col(kXAxisIndex) = target.xAxis;
  rotation.col(kYAxisIndex) = target.normal.cross(target.xAxis);
  rotation.col(kZAxisIndex) = target.normal;
  return rotation;
}

/** `error` scaled down to the norm `largest` when it is longer. */
Eigen::Vector3d capped(const Eigen::Vector3d& error, double largest) {
  const double norm = error.norm();
  return norm > largest ? Eigen::Vector3d(error * (largest / norm)) : error;
}

/** The rotation vector from `current` to `target`, in world axes. */
Eigen::Vector3d rotationError(const Eigen::Matrix3d& target,
                              const Eigen::Matrix3d& current) {
  const Eigen::AngleAxisd error(target * current.transpose());
  return error.angle() * error.axis();
}

/** [rad] The angle between two unit vectors. */
double angleBetween(const Eigen::Vector3d& first,
                    const Eigen::Vector3d& second) {
  return std::acos(std::clamp(first.dot(second), -1.0, 1.0));
}

/** One linear inequality row' dq >= bound, whose violation costs `penalty`
 * per unit. */
struct InequalityRow {
  Eigen::RowVectorXd row;
  double bound = 0.0;
  double penalty = 0.0;
};

/** The weighted least-squares terms and inequalities of one iteration. */
struct QpTerms {
  // Whether to compute the Jacobians; without them, only the merit's values.
  bool linearize = true;
  Eigen::MatrixXd hessian;
  Eigen::VectorXd gradient;
  std::vector<InequalityRow> inequalities;
  // Of the merit: the least squares' value and the inequalities' penalized
  // violation.
  double leastSquares = 0.0;
  double violation = 0.0;
};

/** The posture weight of each velocity coordinate: the problem's or the
 * options', a tenth of it for the base position. */
Eigen::VectorXd postureWeightsOf(const StanceIkProblem& problem,
                                 const StanceIkOptions& options,
                                 int velocities) {
  // The base position's weight relative to the joints' and the base turn's.
  constexpr double kBasePositionWeightFraction = 0.1;
  const double weight = problem.postureWeight.value_or(options.postureWeight);
  Eigen::VectorXd weights = Eigen::VectorXd::Constant(velocities, weight);
  weights.head<kPositionDim>().setConstant(kBasePositionWeightFraction *
                                           weight);
  return weights;
}

/** Adds w |J dq - error|^2 to the terms; `jacobian` is called only when
 * the terms linearize. */
void addLeastSquares(absl::FunctionRef<Eigen::MatrixXd()> jacobian,
                     const Eigen::VectorXd& error,
                     double weight,
                     QpTerms& terms) {
  terms.leastSquares += weight * error.squaredNorm();
  if (!terms.linearize) return;
  const Eigen::MatrixXd matrix = jacobian();
  terms.hessian.noalias() += (weight + weight) * matrix.transpose() * matrix;
  terms.gradient.noalias() -= (weight + weight) * matrix.transpose() * error;
}

/** Adds row dq >= bound, whose violation at dq = 0 is max(0, bound); `row` is
 * called only when the terms linearize. */
void addInequality(absl::FunctionRef<Eigen::RowVectorXd()> row,
                   double bound,
                   double penalty,
                   QpTerms& terms) {
  terms.violation += penalty * std::max(0.0, bound);
  if (!terms.linearize) return;
  terms.inequalities.push_back(
      InequalityRow{.row = row(), .bound = bound, .penalty = penalty});
}

/** Adds J dq = error as two inequalities, |J dq - error| with the exact
 * penalty `penalty`, and w |J dq - error|^2. */
void addEquality(absl::FunctionRef<Eigen::MatrixXd()> jacobian,
                 const Eigen::VectorXd& error,
                 double weight,
                 double penalty,
                 QpTerms& terms) {
  terms.leastSquares += weight * error.squaredNorm();
  terms.violation += penalty * error.lpNorm<1>();
  if (!terms.linearize) return;
  const Eigen::MatrixXd matrix = jacobian();
  terms.hessian.noalias() += (weight + weight) * matrix.transpose() * matrix;
  terms.gradient.noalias() -= (weight + weight) * matrix.transpose() * error;
  for (Eigen::Index k = 0; k < error.size(); ++k) {
    terms.inequalities.push_back(InequalityRow{
        .row = matrix.row(k), .bound = error(k), .penalty = penalty});
    terms.inequalities.push_back(InequalityRow{
        .row = -matrix.row(k), .bound = -error(k), .penalty = penalty});
  }
}

}  // namespace

StanceInverseKinematics::StanceInverseKinematics(
    std::unique_ptr<RobotStanceModel> model,
    TerrainModel terrain,
    CollisionModel collision,
    const StanceIkOptions& options)
    : model_(std::move(model)),
      terrain_(std::move(terrain)),
      collision_(std::move(collision)),
      options_(options) {}

absl::StatusOr<std::unique_ptr<StanceInverseKinematics>>
StanceInverseKinematics::Create(std::unique_ptr<RobotStanceModel> model,
                                TerrainModel terrain,
                                CollisionModel collision,
                                const StanceIkOptions& options) {
  if (model == nullptr) {
    return absl::InvalidArgumentError("the inverse kinematics needs a model");
  }
  if (options.maxIterations < 1 || !(options.reachTolerance > 0.0) ||
      !(options.penetrationTolerance >= 0.0) || !(options.maxPadTilt > 0.0) ||
      !(options.contactWeight > 0.0) || !(options.damping > 0.0) ||
      !(options.slackPenalty > 0.0) || !(options.maxJointStep > 0.0) ||
      !(options.maxBaseStep > 0.0) || !(options.terrainClearance >= 0.0) ||
      !(options.selfClearance >= 0.0)) {
    return absl::InvalidArgumentError(
        "the inverse kinematics' iterations, tolerances, weights and steps "
        "must be positive, its clearances not negative");
  }
  return absl::WrapUnique(new StanceInverseKinematics(
      std::move(model), std::move(terrain), std::move(collision), options));
}

namespace {

/** The limits of `joint` the IK keeps to from `value`: `margin` inside its
 * limits (or the middle of a narrower range), widened to `value` when it is
 * already between them and the margin, so that a pose near a limit, such as a
 * seated hip, is not pushed out. */
std::pair<double, double> innerLimits(const ActuatedJoint& joint,
                                      double margin,
                                      double value) {
  const double inner =
      std::min(margin, 0.5 * (joint.upperLimit - joint.lowerLimit));
  const double lower =
      std::clamp(value, joint.lowerLimit, joint.lowerLimit + inner);
  const double upper =
      std::clamp(value, joint.upperLimit - inner, joint.upperLimit);
  return {lower, upper};
}

}  // namespace

Eigen::VectorXd StanceInverseKinematics::clampToLimits(
    const Eigen::VectorXd& q) const {
  Eigen::VectorXd clamped = q;
  for (const ActuatedJoint& joint : model_->actuatedJoints()) {
    const std::pair<double, double> limits = innerLimits(
        joint, options_.jointLimitMargin, clamped(joint.positionIndex));
    clamped(joint.positionIndex) =
        std::clamp(clamped(joint.positionIndex), limits.first, limits.second);
  }
  return clamped;
}

namespace {

/** The least squares of the held contacts and the soft targets. */
void addTargets(const RobotStanceModel& model,
                const StanceIkProblem& problem,
                const StanceIkOptions& options,
                QpTerms& terms) {
  for (const ContactTarget& contact : problem.contacts) {
    const Eigen::Vector3d position = model.padPosition(contact.candidate);
    addEquality(
        [&]() -> Eigen::MatrixXd {
          return model.pointJacobian(contact.candidate, position);
        },
        contact.position - position, options.contactWeight,
        options.contactPenalty, terms);
    const Eigen::Matrix3d rotation = model.padRotation(contact.candidate);
    const pinocchio::JointIndex joint = model.padJoint(contact.candidate);
    const ContactCandidate& candidate =
        model.contactSet().candidate(contact.candidate);
    if (candidate.kind == ContactKind::kSurface) {
      addEquality(
          [&]() -> Eigen::MatrixXd {
            return model.jointAngularJacobian(joint);
          },
          rotationError(surfaceRotation(contact), rotation),
          options.orientationWeight, options.contactPenalty, terms);
      continue;
    }
    // A point pad's normal z stays in the cone about the surface normal n:
    // d(n.z) = omega . (z x n).
    const Eigen::Vector3d z = rotation.col(kZAxisIndex);
    addInequality(
        [&]() -> Eigen::RowVectorXd {
          return z.cross(contact.normal).transpose() *
                 model.jointAngularJacobian(joint);
        },
        std::cos(kTiltConeFraction * options.maxPadTilt) -
            contact.normal.dot(z),
        options.contactPenalty, terms);
  }
  for (const PadTarget& target : problem.padTargets) {
    const Eigen::Vector3d position = model.padPosition(target.candidate);
    addLeastSquares(
        [&]() -> Eigen::MatrixXd {
          return model.pointJacobian(target.candidate, position);
        },
        target.position - position, options.padTargetWeight, terms);
    if (!target.normal.has_value()) continue;
    // dz = omega x z: the pad's z axis toward the normal.
    const Eigen::Vector3d z =
        model.padRotation(target.candidate).col(kZAxisIndex);
    const pinocchio::JointIndex joint = model.padJoint(target.candidate);
    addLeastSquares(
        [&]() -> Eigen::MatrixXd {
          return -pinocchio::skew(z) * model.jointAngularJacobian(joint);
        },
        *target.normal - z, options.orientationWeight, terms);
  }
  if (problem.comTarget.has_value()) {
    addLeastSquares(
        [&]() -> Eigen::MatrixXd { return model.centerOfMassJacobian(); },
        *problem.comTarget - model.centerOfMass(), options.comTargetWeight,
        terms);
  }
}

/** The CoM's polygon: n_e . c <= d_e - margin per edge. */
void addComPolygon(const RobotStanceModel& model,
                   const StanceIkProblem& problem,
                   double penalty,
                   QpTerms& terms) {
  const std::vector<Eigen::Vector2d>& polygon = problem.comPolygon;
  if (polygon.size() < kMinPolygonVertices) return;
  const Eigen::Vector2d com =
      model.centerOfMass().head<kHorizontalPositionDim>();
  for (size_t i = 0; i < polygon.size(); ++i) {
    const Eigen::Vector2d edge = polygon[(i + 1) % polygon.size()] - polygon[i];
    const Eigen::Vector2d outward =
        Eigen::Vector2d(edge.y(), -edge.x()).normalized();
    const double offset = outward.dot(polygon[i]);
    addInequality(
        [&]() -> Eigen::RowVectorXd {
          return -outward.transpose() *
                 model.centerOfMassJacobian().topRows<kHorizontalPositionDim>();
        },
        outward.dot(com) - offset + problem.comMargin, penalty, terms);
  }
}

/** The clearances of the spheres within the activation distance. */
void addCollisions(const RobotStanceModel& model,
                   const TerrainModel& terrain,
                   const CollisionModel& collision,
                   const StanceIkProblem& problem,
                   const StanceIkOptions& options,
                   QpTerms& terms) {
  const std::vector<CollisionSphere>& spheres = collision.spheres();
  std::vector<Eigen::Vector3d> centers(spheres.size());
  for (int i = 0; i < static_cast<int>(spheres.size()); ++i) {
    centers[i] = collision.worldCenter(model, i);
  }
  // Inside a pad's neighborhood the clearance required of a sphere falls off
  // with this slope [m/m] instead of vanishing at once: the exemption is
  // continuous, so the merit is, and the IK keeps at least the judged
  // requirement (none inside the neighborhood, the clearance outside it).
  constexpr double kExemptionRamp = 10.0;
  for (int i = 0; i < static_cast<int>(spheres.size()); ++i) {
    double nearestPad = std::numeric_limits<double>::infinity();
    for (const PadRegion& pad : problem.exemptPads) {
      if (!exemptsJoint(pad, spheres[i].joint)) continue;
      nearestPad = std::min(nearestPad, distanceToPad(pad, centers[i]));
    }
    const double relaxed =
        kExemptionRamp *
        std::max(0.0, spheres[i].radius + options.padNeighborhood - nearestPad);
    const double distance =
        terrain.sphereDistance(centers[i], spheres[i].radius) +
        CollisionModel::restingAllowance(problem.exemptPads, spheres[i].joint,
                                         centers[i], spheres[i].radius,
                                         options.padNeighborhood);
    const double bound = options.terrainClearance - relaxed - distance;
    if (bound < -options.activationDistance) continue;
    addInequality(
        [&]() -> Eigen::RowVectorXd {
          return terrain.normal(centers[i]).transpose() *
                 model.jointPointJacobian(spheres[i].joint, centers[i]);
        },
        bound, options.slackPenalty, terms);
  }
  // Per pair of links, the closest pair of spheres only.
  absl::flat_hash_map<std::pair<int, int>, SpherePair> closest;
  absl::flat_hash_map<std::pair<int, int>, double> closestClearance;
  for (const SpherePair& pair : collision.selfCollisionPairs()) {
    const double clearance =
        (centers[pair.first] - centers[pair.second]).norm() -
        spheres[pair.first].radius - spheres[pair.second].radius;
    const std::pair<int, int> links(spheres[pair.first].link,
                                    spheres[pair.second].link);
    const absl::flat_hash_map<std::pair<int, int>, double>::iterator found =
        closestClearance.find(links);
    if (found == closestClearance.end() || clearance < found->second) {
      closestClearance[links] = clearance;
      closest[links] = pair;
    }
  }
  // Sorted, so that the QP's rows do not depend on the hash order.
  std::vector<SpherePair> pairs;
  for (const std::pair<const std::pair<int, int>, SpherePair>& entry :
       closest) {
    pairs.push_back(entry.second);
  }
  std::sort(pairs.begin(), pairs.end(),
            [](const SpherePair& first, const SpherePair& second) {
              return std::tie(first.first, first.second) <
                     std::tie(second.first, second.second);
            });
  for (const SpherePair& pair : pairs) {
    const Eigen::Vector3d offset = centers[pair.first] - centers[pair.second];
    const double distance = offset.norm();
    const double clearance =
        distance - spheres[pair.first].radius - spheres[pair.second].radius;
    // Below this [m], the centers coincide and the direction is undefined.
    constexpr double kCoincident = 1.0e-9;
    if (clearance > options.selfActivationDistance || distance < kCoincident) {
      continue;
    }
    const Eigen::Vector3d direction = offset / distance;
    addInequality(
        [&]() -> Eigen::RowVectorXd {
          return direction.transpose() *
                 (model.jointPointJacobian(spheres[pair.first].joint,
                                           centers[pair.first]) -
                  model.jointPointJacobian(spheres[pair.second].joint,
                                           centers[pair.second]));
        },
        options.selfClearance - clearance, options.slackPenalty, terms);
  }
}

}  // namespace

absl::StatusOr<Eigen::VectorXd> StanceInverseKinematics::step(
    const StanceIkProblem& problem, double radius) {
  const int velocities = model_->model().nv;
  QpTerms terms;
  terms.hessian =
      options_.damping * Eigen::MatrixXd::Identity(velocities, velocities);
  terms.gradient = Eigen::VectorXd::Zero(velocities);
  addTargets(*model_, problem, options_, terms);
  // The posture reference, per coordinate of the tangent space.
  const Eigen::VectorXd toReference = pinocchio::difference(
      model_->model(), model_->generalizedPosition(), problem.postureReference);
  const Eigen::VectorXd postureWeights =
      postureWeightsOf(problem, options_, velocities);
  terms.hessian.diagonal() += (postureWeights + postureWeights);
  terms.gradient -= (postureWeights + postureWeights).cwiseProduct(toReference);
  addComPolygon(*model_, problem, options_.slackPenalty, terms);
  addCollisions(*model_, terrain_, collision_, problem, options_, terms);

  const int slacks = static_cast<int>(terms.inequalities.size());
  const int variables = velocities + slacks;
  LinearProgram program = LinearProgram::Unconstrained(variables);
  program.cost.head(velocities) = terms.gradient;
  for (int r = 0; r < slacks; ++r) {
    program.cost(velocities + r) = terms.inequalities[r].penalty;
  }
  Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(variables, variables);
  hessian.topLeftCorner(velocities, velocities) = terms.hessian;
  hessian.bottomRightCorner(slacks, slacks)
      .diagonal()
      .setConstant(kSlackHessian);
  // The trust region, the joint limits and the slacks' sign.
  const double baseStep = radius * options_.maxBaseStep;
  const double jointStep = radius * options_.maxJointStep;
  program.lowerBound.head(kPositionDim).setConstant(-baseStep);
  program.upperBound.head(kPositionDim).setConstant(baseStep);
  program.lowerBound.segment<kRotationTangentDim>(kBaseRotationIndex)
      .setConstant(-jointStep);
  program.upperBound.segment<kRotationTangentDim>(kBaseRotationIndex)
      .setConstant(jointStep);
  const Eigen::VectorXd& q = model_->generalizedPosition();
  for (const ActuatedJoint& joint : model_->actuatedJoints()) {
    const std::pair<double, double> limits =
        innerLimits(joint, options_.jointLimitMargin, q(joint.positionIndex));
    const double lower = std::min(
        0.0, std::max(-jointStep, limits.first - q(joint.positionIndex)));
    const double upper = std::max(
        0.0, std::min(jointStep, limits.second - q(joint.positionIndex)));
    program.lowerBound(joint.velocityIndex) = lower;
    program.upperBound(joint.velocityIndex) = upper;
  }
  program.lowerBound.tail(slacks).setZero();
  program.inequalityMatrix = Eigen::MatrixXd::Zero(slacks, variables);
  program.inequalityLower.resize(slacks);
  program.inequalityUpper = Eigen::VectorXd::Constant(slacks, kInfinity);
  for (int r = 0; r < slacks; ++r) {
    program.inequalityMatrix.row(r).head(velocities) =
        terms.inequalities[r].row;
    program.inequalityMatrix(r, velocities + r) = 1.0;
    program.inequalityLower(r) = terms.inequalities[r].bound;
  }
  // HPIPM's tolerances for a step: a step is refined by the next iteration,
  // and the penalties make the objective large.
  QpSolverSettings settings;
  settings.maxIterations = kQpMaxIterations;
  settings.stationarityTolerance = kQpTolerance;
  settings.equalityTolerance = kQpTolerance;
  settings.inequalityTolerance = kQpTolerance;
  settings.complementarityTolerance = kQpTolerance;
  ASSIGN_OR_RETURN(const LinearProgramSolution solution,
                   solveQuadraticProgram(hessian, program, settings));
  return Eigen::VectorXd(solution.x.head(velocities));
}

namespace {

/** The merit of a configuration: the least squares plus the exact penalty of
 * the violated inequalities (the QP terms at dq = 0). */
double meritOf(const RobotStanceModel& model,
               const TerrainModel& terrain,
               const CollisionModel& collision,
               const StanceIkProblem& problem,
               const StanceIkOptions& options) {
  const int velocities = model.model().nv;
  QpTerms terms;
  terms.linearize = false;
  addTargets(model, problem, options, terms);
  addComPolygon(model, problem, options.slackPenalty, terms);
  addCollisions(model, terrain, collision, problem, options, terms);
  const Eigen::VectorXd toReference = pinocchio::difference(
      model.model(), model.generalizedPosition(), problem.postureReference);
  const double posture =
      toReference
          .cwiseProduct(postureWeightsOf(problem, options, velocities)
                            .cwiseProduct(toReference))
          .sum();
  return terms.leastSquares + posture + terms.violation;
}

}  // namespace

absl::StatusOr<Eigen::VectorXd> StanceInverseKinematics::correctContacts(
    const StanceIkProblem& problem, const Eigen::VectorXd& trial) {
  // Gauss-Newton corrections of the held pads, as many as needed for the
  // second-order drift of a step; their damping keeps them finite near a
  // singular configuration.
  constexpr int kCorrections = 2;
  constexpr double kCorrectionDamping = 1.0e-8;
  constexpr double kCorrected = 1.0e-7;
  // [m, rad] The most a correction moves a pad: the second-order drift of a
  // step, not the way to a far target, which the QP steps cover within the
  // trust region.
  constexpr double kLargestCorrection = 0.01;
  const ContactSet& contactSet = model_->contactSet();
  int rows = 0;
  for (const ContactTarget& contact : problem.contacts) {
    rows +=
        contactSet.candidate(contact.candidate).kind == ContactKind::kSurface
            ? kPoseTangentDim
            : kPositionDim;
  }
  Eigen::VectorXd q = trial;
  RETURN_IF_ERROR(model_->setGeneralizedPosition(q));
  for (int correction = 0; correction < kCorrections && rows > 0;
       ++correction) {
    Eigen::MatrixXd jacobian(rows, model_->model().nv);
    Eigen::VectorXd error(rows);
    int row = 0;
    for (const ContactTarget& contact : problem.contacts) {
      const Eigen::Vector3d position = model_->padPosition(contact.candidate);
      jacobian.middleRows<kPositionDim>(row) =
          model_->pointJacobian(contact.candidate, position);
      error.segment<kPositionDim>(row) =
          capped(contact.position - position, kLargestCorrection);
      row += kPositionDim;
      if (contactSet.candidate(contact.candidate).kind !=
          ContactKind::kSurface) {
        continue;
      }
      jacobian.middleRows<kRotationTangentDim>(row) =
          model_->jointAngularJacobian(model_->padJoint(contact.candidate));
      error.segment<kRotationTangentDim>(row) =
          capped(rotationError(surfaceRotation(contact),
                               model_->padRotation(contact.candidate)),
                 kLargestCorrection);
      row += kRotationTangentDim;
    }
    if (error.lpNorm<Eigen::Infinity>() < kCorrected) break;
    const Eigen::MatrixXd gram =
        jacobian * jacobian.transpose() +
        kCorrectionDamping * Eigen::MatrixXd::Identity(rows, rows);
    const Eigen::VectorXd dq = jacobian.transpose() * gram.ldlt().solve(error);
    q = clampToLimits(pinocchio::integrate(model_->model(), q, dq));
    RETURN_IF_ERROR(model_->setGeneralizedPosition(q));
  }
  return q;
}

absl::StatusOr<StanceIkResult> StanceInverseKinematics::solve(
    const StanceIkProblem& problem, const Eigen::VectorXd& initial) {
  if (initial.size() != model_->model().nq ||
      problem.postureReference.size() != model_->model().nq) {
    return absl::InvalidArgumentError(
        absl::StrCat("the inverse kinematics needs generalized positions of ",
                     model_->model().nq, " values"));
  }
  for (const ContactTarget& contact : problem.contacts) {
    if (contact.candidate < 0 ||
        contact.candidate >= model_->contactSet().size()) {
      return absl::InvalidArgumentError("a held contact names no candidate");
    }
  }
  Eigen::VectorXd q = clampToLimits(initial);
  RETURN_IF_ERROR(model_->setGeneralizedPosition(q));
  double merit = meritOf(*model_, terrain_, collision_, problem, options_);
  // A trust-region SQP: the QP's step bounds are scaled by `radius`, which
  // shrinks when a step increases the merit (and the QP is solved again) and
  // grows when one decreases it.
  int iteration = 0;
  std::string termination = "iterations";
  double radius = 1.0;
  for (; iteration < options_.maxIterations; ++iteration) {
    const absl::StatusOr<Eigen::VectorXd> stepped = step(problem, radius);
    if (!stepped.ok()) {
      // HPIPM did not converge on this linearization: judge where we are.
      if (stepped.status().code() != absl::StatusCode::kInternal) {
        return stepped.status();
      }
      RETURN_IF_ERROR(model_->setGeneralizedPosition(q));
      termination = "qp";
      break;
    }
    ASSIGN_OR_RETURN(
        const Eigen::VectorXd next,
        correctContacts(problem, clampToLimits(pinocchio::integrate(
                                     model_->model(), q, *stepped))));
    const double nextMerit =
        meritOf(*model_, terrain_, collision_, problem, options_);
    if (!(nextMerit <= merit)) {
      RETURN_IF_ERROR(model_->setGeneralizedPosition(q));
      radius *= kRadiusShrink;
      if (radius < kSmallestRadius) {
        termination = "stalled";
        break;
      }
      continue;
    }
    const double largestStep = stepped->lpNorm<Eigen::Infinity>();
    q = next;
    const double decrease = merit - nextMerit;
    merit = nextMerit;
    radius = std::min(1.0, radius * kRadiusGrowth);
    if (largestStep < options_.convergedStep ||
        decrease < kConvergedMeritDecrease * (1.0 + merit)) {
      termination = "converged";
      break;
    }
  }
  ASSIGN_OR_RETURN(StanceIkResult result, evaluate(problem, q));
  result.iterations = iteration;
  result.termination = termination;
  if (!result.feasible) {
    absl::StrAppend(&result.failure, " (", termination, " after ", iteration,
                    " iterations)");
  }
  return result;
}

absl::StatusOr<StanceIkResult> StanceInverseKinematics::evaluate(
    const StanceIkProblem& problem, const Eigen::VectorXd& q) {
  RETURN_IF_ERROR(model_->setGeneralizedPosition(q));
  StanceIkResult result;
  result.q = q;
  std::vector<std::string> failures;
  for (const ContactTarget& contact : problem.contacts) {
    const ContactCandidate& candidate =
        model_->contactSet().candidate(contact.candidate);
    const double reach =
        (model_->padPosition(contact.candidate) - contact.position).norm();
    result.maxReachError = std::max(result.maxReachError, reach);
    const double tilt =
        angleBetween(model_->padRotation(contact.candidate).col(kZAxisIndex),
                     contact.normal);
    if (candidate.kind == ContactKind::kSurface) {
      result.maxSurfaceTilt = std::max(result.maxSurfaceTilt, tilt);
    } else {
      result.maxPointTilt = std::max(result.maxPointTilt, tilt);
    }
  }
  if (result.maxReachError > options_.reachTolerance) {
    failures.push_back(absl::StrFormat("a pad is %.4f m from its target",
                                       result.maxReachError));
  }
  if (result.maxPointTilt > options_.maxPadTilt ||
      result.maxSurfaceTilt > options_.surfaceTiltTolerance) {
    failures.push_back(absl::StrFormat(
        "a pad tilts %.3f rad (point) / %.3f rad (surface) from its surface",
        result.maxPointTilt, result.maxSurfaceTilt));
  }
  if (!problem.comPolygon.empty()) {
    result.comMargin = signedDistanceToConvexPolygon(
        problem.comPolygon,
        model_->centerOfMass().head<kHorizontalPositionDim>());
    if (*result.comMargin < problem.comMargin - kComMarginSlack) {
      failures.push_back(
          absl::StrFormat("the CoM is %.4f m inside its polygon (needs %.4f m)",
                          *result.comMargin, problem.comMargin));
    }
  }
  const std::vector<CollisionSphere>& spheres = collision_.spheres();
  for (int i = 0; i < static_cast<int>(spheres.size()); ++i) {
    const Eigen::Vector3d center = collision_.worldCenter(*model_, i);
    if (CollisionModel::isExempt(problem.exemptPads, spheres[i].joint, center,
                                 spheres[i].radius, options_.padNeighborhood)) {
      continue;
    }
    const double clearance =
        terrain_.sphereDistance(center, spheres[i].radius) +
        CollisionModel::restingAllowance(problem.exemptPads, spheres[i].joint,
                                         center, spheres[i].radius,
                                         options_.padNeighborhood);
    if (!result.terrainClearance.has_value() ||
        clearance < *result.terrainClearance) {
      result.terrainClearance = clearance;
      result.terrainClearanceLink = collision_.linkNames()[spheres[i].link];
    }
  }
  // Judged as the certifier judges it: no deeper than the contact tolerance.
  if (result.terrainClearance.has_value() &&
      *result.terrainClearance < -options_.penetrationTolerance) {
    failures.push_back(absl::StrFormat("%s penetrates the terrain by %.4f m",
                                       result.terrainClearanceLink,
                                       -*result.terrainClearance));
  }
  const SphereClearance self = collision_.selfClearance(*model_);
  result.selfClearance = self.clearance;
  result.selfClearanceLinks = self.link;
  if (self.clearance.has_value() && *self.clearance < 0.0) {
    failures.push_back(
        absl::StrFormat("%s collide by %.4f m", self.link, -*self.clearance));
  }
  result.feasible = failures.empty();
  for (const std::string& failure : failures) {
    absl::StrAppend(&result.failure, result.failure.empty() ? "" : "; ",
                    failure);
  }
  return result;
}

}  // namespace ocs2::humanoid::multi_contact
