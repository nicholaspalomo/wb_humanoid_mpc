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

#include "humanoid_multi_contact/stability/StabilityCertifier.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "ocs2_core/Dimensions.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_multi_contact/stability/TorqueFeasibility.h"

namespace ocs2::humanoid::multi_contact {
namespace {

using ReportKnot = config::CertifierReport::Knot;
using PlanKnot = config::MultiContactPlan::Knot;

/** The CoM's horizontal position. */
Eigen::Vector2d horizontal(const Eigen::Vector3d& point) {
  return point.head<kHorizontalPositionDim>();
}

absl::StatusOr<Eigen::Vector3d> vector3Of(const Eigen::VectorXd& values,
                                          absl::string_view field,
                                          int knot,
                                          bool emptyIsZero) {
  if (values.size() == 0 && emptyIsZero) return Eigen::Vector3d::Zero();
  if (values.size() != kPositionDim || !values.allFinite()) {
    return absl::InvalidArgumentError(
        absl::StrCat("plan knot ", knot, ": ", field, " must hold ",
                     kPositionDim, " finite values, not ", values.size()));
  }
  return Eigen::Vector3d(values);
}

absl::StatusOr<RobotConfiguration> configurationOf(
    const config::MultiContactPlan& plan, const PlanKnot& knot, int index) {
  RobotConfiguration configuration;
  ASSIGN_OR_RETURN(configuration.basePosition,
                   vector3Of(knot.base_position, "base_position", index,
                             /*emptyIsZero=*/false));
  if (knot.base_orientation_xyzw.size() != kQuaternionCoefficients) {
    return absl::InvalidArgumentError(absl::StrCat(
        "plan knot ", index, ": base_orientation_xyzw must hold a quaternion"));
  }
  configuration.baseOrientation =
      Eigen::Quaterniond(Eigen::Vector4d(knot.base_orientation_xyzw));
  if (knot.joint_positions.size() !=
      static_cast<Eigen::Index>(plan.joint_names.size())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "plan knot ", index, ": ", knot.joint_positions.size(),
        " joint positions for ", plan.joint_names.size(), " joint names"));
  }
  configuration.jointNames = plan.joint_names;
  configuration.jointPositions = knot.joint_positions;
  return configuration;
}

/** d/dt of the momenta by differences between the neighboring knots. */
absl::Status setMomentumRates(const config::MultiContactPlan& plan,
                              std::vector<CertifierKnot>& knots) {
  const int count = static_cast<int>(knots.size());
  std::vector<CentroidalMomentumRate> momenta(count);
  for (int k = 0; k < count; ++k) {
    ASSIGN_OR_RETURN(momenta[k].linear,
                     vector3Of(plan.knots[k].linear_momentum, "linear_momentum",
                               k, /*emptyIsZero=*/true));
    ASSIGN_OR_RETURN(momenta[k].angular,
                     vector3Of(plan.knots[k].angular_momentum,
                               "angular_momentum", k, /*emptyIsZero=*/true));
  }
  if (count == 1) return absl::OkStatus();
  for (int k = 0; k < count; ++k) {
    const int before = k == 0 ? 0 : k - 1;
    const int after = k + 1 == count ? k : k + 1;
    const double span = knots[after].time - knots[before].time;
    knots[k].momentumRate.linear =
        (momenta[after].linear - momenta[before].linear) / span;
    knots[k].momentumRate.angular =
        (momenta[after].angular - momenta[before].angular) / span;
  }
  return absl::OkStatus();
}

}  // namespace

Eigen::Matrix3d coneRotationOnSurface(const Eigen::Matrix3d& padRotation,
                                      const Eigen::Vector3d& normal) {
  // Below this, the pad's x axis is nearly the normal: use any tangent.
  constexpr double kDegenerateProjection = 1.0e-6;
  Eigen::Vector3d xAxis = padRotation.col(kXAxisIndex);
  xAxis -= xAxis.dot(normal) * normal;
  if (xAxis.norm() < kDegenerateProjection) xAxis = normal.unitOrthogonal();
  xAxis.normalize();
  Eigen::Matrix3d rotation;
  rotation.col(kXAxisIndex) = xAxis;
  rotation.col(kYAxisIndex) = normal.cross(xAxis);
  rotation.col(kZAxisIndex) = normal;
  return rotation;
}

absl::StatusOr<StanceOnTerrain> stanceOnTerrain(const RobotStanceModel& model,
                                                const TerrainModel& terrain,
                                                ContactMask mask) {
  ASSIGN_OR_RETURN(const std::vector<int> active,
                   model.contactSet().activeIndices(mask));
  StanceOnTerrain result;
  for (const int index : active) {
    const ContactCandidate& candidate = model.contactSet().candidate(index);
    const Eigen::Vector3d position = model.padPosition(index);
    const PatchFrame patch = terrain.patchFrameAt(position);
    const double distance =
        terrain.primitiveQuery(patch.primitiveIndex, position).distance;
    result.maxPadDistance = std::max(result.maxPadDistance, std::abs(distance));
    result.stance.push_back(StanceContact{
        .name = candidate.name,
        .candidateIndex = index,
        .kind = candidate.kind,
        .position = position,
        .rotation =
            coneRotationOnSurface(model.padRotation(index), patch.normal()),
        .friction = patch.friction,
        .rectangle = candidate.rectangle});
  }
  return result;
}

absl::StatusOr<std::vector<CertifierKnot>> certifierKnotsOf(
    const config::MultiContactPlan& plan) {
  if (plan.knots.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("plan '", plan.scenario, "' has no knot"));
  }
  std::vector<CertifierKnot> knots;
  knots.reserve(plan.knots.size());
  for (int k = 0; k < static_cast<int>(plan.knots.size()); ++k) {
    const PlanKnot& knot = plan.knots[k];
    if (!std::isfinite(knot.time) ||
        (k > 0 && !(knot.time > plan.knots[k - 1].time))) {
      return absl::InvalidArgumentError(absl::StrCat(
          "plan knot ", k, ": the times must be finite and increase"));
    }
    CertifierKnot certifierKnot;
    certifierKnot.time = knot.time;
    certifierKnot.activeContacts = knot.active_contacts;
    ASSIGN_OR_RETURN(certifierKnot.configuration,
                     configurationOf(plan, knot, k));
    knots.push_back(std::move(certifierKnot));
  }
  RETURN_IF_ERROR(setMomentumRates(plan, knots));
  return knots;
}

StabilityCertifier::StabilityCertifier(
    std::unique_ptr<RobotStanceModel> model,
    TerrainModel terrain,
    const CertifierOptions& options,
    std::optional<CollisionModel> collisionModel)
    : model_(std::move(model)),
      terrain_(std::move(terrain)),
      options_(options),
      collisionModel_(std::move(collisionModel)) {}

absl::StatusOr<std::unique_ptr<StabilityCertifier>> StabilityCertifier::Create(
    std::unique_ptr<RobotStanceModel> model,
    TerrainModel terrain,
    const CertifierOptions& options,
    std::optional<CollisionModel> collisionModel) {
  if (model == nullptr) {
    return absl::InvalidArgumentError("the certifier needs a robot model");
  }
  if (!(options.torqueMargin >= 0.0) ||
      !(options.contactDistanceTolerance >= 0.0) ||
      !(options.padNeighborhood >= 0.0) ||
      (options.staticMargin.has_value() && std::isnan(*options.staticMargin))) {
    return absl::InvalidArgumentError(
        "the torque margin, the contact tolerance and the pad neighborhood "
        "must not be negative, and the static margin not NaN");
  }
  return absl::WrapUnique(new StabilityCertifier(std::move(model),
                                                 std::move(terrain), options,
                                                 std::move(collisionModel)));
}

absl::StatusOr<StanceOnTerrain> StabilityCertifier::stanceOf(
    ContactMask mask) const {
  return stanceOnTerrain(*model_, terrain_, mask);
}

absl::StatusOr<bool> StabilityCertifier::staticCondition(
    const RobotConfiguration& configuration, ContactMask mask) {
  if (!options_.staticMargin.has_value()) return true;
  RETURN_IF_ERROR(model_->setConfiguration(configuration));
  ASSIGN_OR_RETURN(const StanceOnTerrain stance, stanceOf(mask));
  if (stance.stance.empty()) return false;
  ASSIGN_OR_RETURN(
      const StaticEquilibrium equilibrium,
      StaticEquilibrium::Create(stance.stance, options_.equilibrium));
  // An LP HPIPM cannot solve: not statically stable.
  const absl::StatusOr<std::vector<Eigen::Vector2d>> polygon =
      equilibrium.supportPolygon();
  if (!polygon.ok()) return false;
  const std::optional<double> margin =
      staticMargin(*polygon, horizontal(model_->centerOfMass()));
  if (!margin.has_value() || !(*margin >= *options_.staticMargin)) return false;
  const absl::StatusOr<double> residual =
      equilibrium.equilibriumResidual(horizontal(model_->centerOfMass()));
  if (!residual.ok() ||
      !(*residual <= options_.equilibrium.feasibilityTolerance)) {
    return false;
  }
  if (!enforcesEffortLimits(options_.jointLimits)) return true;
  ASSIGN_OR_RETURN(const TorqueFeasibilityResult torques,
                   minimumTorqueUtilization(*model_, equilibrium));
  return isTorqueFeasible(torques, options_.torqueMargin);
}

absl::StatusOr<bool> StabilityCertifier::reportTorques(
    const StaticEquilibrium& equilibrium, ReportKnot& report) const {
  const absl::StatusOr<TorqueFeasibilityResult> torques =
      minimumTorqueUtilization(*model_, equilibrium);
  if (!enforcesEffortLimits(options_.jointLimits)) {
    // Information only: an LP HPIPM cannot solve leaves the utilization
    // absent.
    if (torques.ok()) {
      report.torque_utilization = torques->utilization;
      report.max_torque_joint = torques->maxTorqueJoint;
    }
    return true;
  }
  if (!torques.ok()) return torques.status();
  report.torque_utilization = torques->utilization;
  report.max_torque_joint = torques->maxTorqueJoint;
  return isTorqueFeasible(*torques, options_.torqueMargin);
}

void StabilityCertifier::reportCollision(ContactMask mask,
                                         ReportKnot& report) const {
  if (!collisionModel_.has_value()) return;
  const SphereClearance clearance = collisionModel_->terrainClearance(
      *model_, terrain_, mask, options_.padNeighborhood,
      /*restingTolerance=*/options_.contactDistanceTolerance);
  report.collision_free = true;
  if (!clearance.clearance.has_value()) return;
  report.min_sphere_clearance = clearance.clearance;
  report.min_sphere_clearance_link = clearance.link;
  if (*clearance.clearance < -options_.contactDistanceTolerance) {
    report.collision_free = false;
    report.failures.push_back(absl::StrFormat(
        "a collision sphere of %s penetrates the terrain by %.4f m "
        "(tolerance %.4f m)",
        clearance.link, -*clearance.clearance,
        options_.contactDistanceTolerance));
  }
}

absl::StatusOr<ReportKnot> StabilityCertifier::evaluateKnot(
    const CertifierKnot& knot) {
  RETURN_IF_ERROR(model_->setConfiguration(knot.configuration));
  ReportKnot report;
  report.time = knot.time;
  report.active_contact_mask = knot.activeContacts;
  ASSIGN_OR_RETURN(report.active_contacts,
                   model_->contactSet().namesOf(knot.activeContacts));
  const Eigen::Vector3d com = model_->centerOfMass();
  report.com = com;
  ASSIGN_OR_RETURN(const StanceOnTerrain stance, stanceOf(knot.activeContacts));
  report.max_active_contact_distance = stance.maxPadDistance;
  const bool onTerrain =
      stance.maxPadDistance <= options_.contactDistanceTolerance;
  if (!onTerrain) {
    report.failures.push_back(absl::StrFormat(
        "an active pad is %.4f m from the terrain (tolerance %.4f m)",
        stance.maxPadDistance, options_.contactDistanceTolerance));
  }

  // The clearance of the inactive candidates' geometry.
  for (int index = 0; index < model_->contactSet().size(); ++index) {
    if (isContactActive(knot.activeContacts, index)) continue;
    const ContactCandidate& candidate = model_->contactSet().candidate(index);
    const Eigen::Matrix3d rotation = model_->padRotation(index);
    // The pad is at geomRadius along the outward normal, -z, from the center.
    const Eigen::Vector3d center =
        model_->padPosition(index) +
        candidate.geomRadius * rotation.col(kZAxisIndex);
    const double clearance =
        terrain_.sphereDistance(center, candidate.geomRadius);
    if (!report.min_inactive_clearance.has_value() ||
        clearance < *report.min_inactive_clearance) {
      report.min_inactive_clearance = clearance;
      report.min_inactive_clearance_contact = candidate.name;
    }
  }

  reportCollision(knot.activeContacts, report);

  bool staticPass = false;
  bool coneLpFailed = false;
  if (stance.stance.empty()) {
    // In flight: only a free fall is feasible.
    report.momentum_rate_residual =
        requiredContactWrench(model_->mass(), com, knot.momentumRate, com)
            .lpNorm<1>();
    report.momentum_rate_feasible = report.momentum_rate_residual <=
                                    options_.equilibrium.feasibilityTolerance;
  } else {
    ASSIGN_OR_RETURN(
        const StaticEquilibrium equilibrium,
        StaticEquilibrium::Create(stance.stance, options_.equilibrium));
    // An LP HPIPM cannot solve fails the knot, naming it, rather than the
    // plan's certification.
    const absl::StatusOr<std::vector<Eigen::Vector2d>> polygon =
        equilibrium.supportPolygon();
    if (polygon.ok()) {
      for (const Eigen::Vector2d& vertex : *polygon) {
        report.support_polygon.push_back(
            config::CertifierReport::Point2{.x = vertex.x(), .y = vertex.y()});
      }
      report.static_margin = staticMargin(*polygon, horizontal(com));
    } else {
      report.failures.push_back(absl::StrCat("the support polygon's LPs fail: ",
                                             polygon.status().message()));
    }
    const absl::StatusOr<double> residual =
        equilibrium.equilibriumResidual(horizontal(com));
    if (!residual.ok()) {
      report.failures.push_back(absl::StrCat(
          "the static equilibrium's LP fails: ", residual.status().message()));
    } else if (*residual <= options_.equilibrium.feasibilityTolerance) {
      ASSIGN_OR_RETURN(const bool torquesPass,
                       reportTorques(equilibrium, report));
      staticPass = options_.staticMargin.has_value() &&
                   report.static_margin.has_value() &&
                   *report.static_margin >= *options_.staticMargin &&
                   torquesPass;
    }
    const absl::StatusOr<ContactWrenchConeResult> cone = checkContactWrenchCone(
        equilibrium, model_->mass(), com, knot.momentumRate);
    if (cone.ok()) {
      report.momentum_rate_feasible = cone->feasible;
      report.momentum_rate_residual = cone->residual;
    } else {
      report.momentum_rate_feasible = false;
      coneLpFailed = true;
      report.failures.push_back(absl::StrCat(
          "the contact wrench cone's LP fails: ", cone.status().message()));
    }
  }
  if (!report.momentum_rate_feasible && !coneLpFailed) {
    report.failures.push_back(absl::StrFormat(
        "the momentum rate is outside the contact wrench cone (residual %.3g)",
        report.momentum_rate_residual));
  }
  report.dynamic_condition = report.momentum_rate_feasible && onTerrain;
  if (options_.staticMargin.has_value()) {
    report.static_condition = staticPass;
    if (!staticPass) {
      const std::string margin =
          absl::StrFormat("condition 1: static margin %s (needs %.3f m)",
                          report.static_margin.has_value()
                              ? absl::StrFormat("%.4f m", *report.static_margin)
                              : std::string("none"),
                          *options_.staticMargin);
      report.failures.push_back(
          enforcesEffortLimits(options_.jointLimits)
              ? absl::StrFormat(
                    "%s, torque utilization %s (at most %.3f)", margin,
                    report.torque_utilization.has_value()
                        ? absl::StrFormat("%.3f at %s",
                                          *report.torque_utilization,
                                          report.max_torque_joint)
                        : std::string("none"),
                    options_.torqueMargin)
              : margin);
    }
  }
  return report;
}

absl::StatusOr<config::CertifierReport> StabilityCertifier::certify(
    const config::MultiContactPlan& plan) {
  ASSIGN_OR_RETURN(const std::vector<CertifierKnot> knots,
                   certifierKnotsOf(plan));
  return certifyKnots(knots);
}

absl::StatusOr<config::CertifierReport> StabilityCertifier::certifyKnots(
    const std::vector<CertifierKnot>& knots) {
  if (knots.empty()) {
    return absl::InvalidArgumentError("there is no knot to certify");
  }
  config::CertifierReport report;
  report.static_margin = options_.staticMargin;
  report.joint_limits = std::string(jointLimitsName(options_.jointLimits));
  if (enforcesEffortLimits(options_.jointLimits)) {
    report.torque_margin = options_.torqueMargin;
  }
  report.passed = true;
  for (int k = 0; k < static_cast<int>(knots.size()); ++k) {
    ASSIGN_OR_RETURN(ReportKnot knotReport, evaluateKnot(knots[k]));
    const ContactMask previous =
        k == 0 ? knots[k].activeContacts : knots[k - 1].activeContacts;
    if (options_.staticMargin.has_value() &&
        previous != knots[k].activeContacts) {
      const ContactMask common = previous & knots[k].activeContacts;
      const bool oneChange =
          std::popcount(previous ^ knots[k].activeContacts) == 1;
      ASSIGN_OR_RETURN(const bool before,
                       staticCondition(knots[k - 1].configuration, common));
      ASSIGN_OR_RETURN(const bool after,
                       staticCondition(knots[k].configuration, common));
      knotReport.transition_condition = oneChange && before && after;
      if (!*knotReport.transition_condition) {
        knotReport.failures.push_back(
            absl::StrCat("condition 3: the stance changes by ",
                         std::popcount(previous ^ knots[k].activeContacts),
                         " contacts; the common stance is ",
                         before ? "" : "not ", "statically stable before and ",
                         after ? "" : "not ", "after the change"));
      }
    }
    knotReport.passed = knotReport.dynamic_condition &&
                        knotReport.collision_free.value_or(true) &&
                        knotReport.static_condition.value_or(true) &&
                        knotReport.transition_condition.value_or(true);
    if (!knotReport.passed) {
      report.passed = false;
      ++report.failed_knots;
    }
    report.knots.push_back(std::move(knotReport));
  }
  return report;
}

}  // namespace ocs2::humanoid::multi_contact
