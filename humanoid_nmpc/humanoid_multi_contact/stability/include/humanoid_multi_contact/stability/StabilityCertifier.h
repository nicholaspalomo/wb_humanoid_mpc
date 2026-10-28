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
#include <vector>

#include "absl/status/statusor.h"

#include "humanoid_multi_contact/collision/CollisionModel.h"
#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"
#include "humanoid_multi_contact/stability/ContactWrenchConeCheck.h"
#include "humanoid_multi_contact/stability/JointLimits.h"
#include "humanoid_multi_contact/stability/RobotStanceModel.h"
#include "humanoid_multi_contact/stability/StaticEquilibrium.h"
#include "humanoid_multi_contact/stability/certifier_report.nproto.h"
#include "humanoid_multi_contact/terrain/TerrainModel.h"

namespace ocs2::humanoid::multi_contact {

/** [m] How far an active contact's pad may be from the terrain by default. */
inline constexpr double kDefaultContactDistanceTolerance = 0.005;

/** The settings of StabilityCertifier, from the scenario. */
struct CertifierOptions {
  // [m] delta_s of condition 1; std::nullopt: conditions 1 and 3 are not judged
  // (design section 11, decision 5).
  std::optional<double> staticMargin;
  // Whether condition 1 judges the torques against the effort limits; under
  // JointLimits::kPositionOnly they are reported only.
  JointLimits jointLimits = JointLimits::kPositionAndEffort;
  // The fraction of the effort limits the torques may use, under
  // JointLimits::kPositionAndEffort.
  double torqueMargin = 1.0;
  StaticEquilibriumOptions equilibrium;
  double contactDistanceTolerance = kDefaultContactDistanceTolerance;
  // [m] The neighborhood of an active pad whose collision spheres are exempt
  // from the terrain clearance (CollisionModel).
  double padNeighborhood = kDefaultPadNeighborhood;
};

/** The stance of a contact mode on the terrain, and the largest distance of its
 * pads to the terrain. Passive data. */
struct StanceOnTerrain {
  Stance stance;
  double maxPadDistance = 0.0;  // [m]
};

/**
 * The stance of the active candidates of `mask` at `model`'s configuration:
 * each pad at its position, on the patch under it
 * (TerrainModel::patchFrameAt()), its cone's z axis the patch normal and its x
 * axis the pad's own, projected on the patch (so that a surface pad's rectangle
 * keeps its yaw), with the patch's friction. InvalidArgument for a mask beyond
 * the contact set.
 */
absl::StatusOr<StanceOnTerrain> stanceOnTerrain(const RobotStanceModel& model,
                                                const TerrainModel& terrain,
                                                ContactMask mask);

/** world_R_cone of a pad of rotation `padRotation` resting on a surface of
 * outward normal `normal` (stanceOnTerrain()). */
Eigen::Matrix3d coneRotationOnSurface(const Eigen::Matrix3d& padRotation,
                                      const Eigen::Vector3d& normal);

/** One knot to certify: a configuration, its contact mode and its momentum
 * rate. Passive data. */
struct CertifierKnot {
  double time = 0.0;
  RobotConfiguration configuration;
  ContactMask activeContacts = 0;
  CentroidalMomentumRate momentumRate;
};

/**
 * The stability certifier of the multi-contact plans (design section 6). At
 * every knot it finds the active stance on the terrain (the patch under each
 * active pad, TerrainModel::patchFrameAt()) and reports the static margin
 * rho_s with its polygon, the smallest torque utilization and its joint,
 * whether the planned momentum rate lies in the contact wrench cone, the
 * distance of the active pads to the terrain and the clearance of the inactive
 * ones, with a collision model the clearance of the robot's collision spheres
 * outside the neighborhood of the active pads (section 5.4), and conditions 1
 * to 3 of section 2:
 *
 *  1. rho_s >= staticMargin and, under JointLimits::kPositionAndEffort, the
 *     torques within torqueMargin of the effort limits;
 *  2. the momentum rate in CWC(S), with every active pad on the terrain;
 *  3. where the stance changes, by one contact, the common stance satisfies
 *     condition 1 at both knots.
 *
 * Conditions 1 and 3 are judged only when the options set a static margin.
 * Under JointLimits::kPositionOnly the torque utilization is information: a
 * torque LP that HPIPM cannot solve leaves it absent instead of failing the
 * certification, and condition 3 solves no torque LP.
 * With a collision model, a knot also fails when a sphere penetrates the
 * terrain by more than the contact distance tolerance.
 * The momentum rate of a plan's knot is the central difference of its momenta
 * (one-sided at the ends).
 *
 * Not thread-safe: it sets the configuration of its RobotStanceModel.
 */
class StabilityCertifier {
 public:
  /** InvalidArgument for a negative torque margin, contact tolerance or pad
   * neighborhood. `collisionModel`, built for `model`, may be absent: the
   * sphere clearances are then neither reported nor judged. */
  static absl::StatusOr<std::unique_ptr<StabilityCertifier>> Create(
      std::unique_ptr<RobotStanceModel> model,
      TerrainModel terrain,
      const CertifierOptions& options,
      std::optional<CollisionModel> collisionModel);

  StabilityCertifier(const StabilityCertifier&) = delete;
  StabilityCertifier& operator=(const StabilityCertifier&) = delete;
  ~StabilityCertifier() = default;

  /**
   * Certifies every knot of `plan`. InvalidArgument for a plan without knots,
   * with times that do not increase, vectors of the wrong size or a mask beyond
   * the contact set; Internal when an LP fails (except the torque LP under
   * JointLimits::kPositionOnly).
   */
  absl::StatusOr<config::CertifierReport> certify(
      const config::MultiContactPlan& plan);

  /**
   * Certifies a sequence of knots; `knots` must not be empty and its times
   * increase.
   */
  absl::StatusOr<config::CertifierReport> certifyKnots(
      const std::vector<CertifierKnot>& knots);

  const RobotStanceModel& model() const { return *model_; }
  const TerrainModel& terrain() const { return terrain_; }
  const CertifierOptions& options() const { return options_; }
  const std::optional<CollisionModel>& collisionModel() const {
    return collisionModel_;
  }

 private:
  StabilityCertifier(std::unique_ptr<RobotStanceModel> model,
                     TerrainModel terrain,
                     const CertifierOptions& options,
                     std::optional<CollisionModel> collisionModel);

  /** The stance of `mask` at the model's current configuration
   * (stanceOnTerrain()). */
  absl::StatusOr<StanceOnTerrain> stanceOf(ContactMask mask) const;

  /** Whether condition 1 holds for `mask` at `configuration`. */
  absl::StatusOr<bool> staticCondition(const RobotConfiguration& configuration,
                                       ContactMask mask);

  /** Reports the torque utilization of `equilibrium`'s stance at the model's
   * configuration in `report`, and returns whether the torques pass condition
   * 1: always under JointLimits::kPositionOnly, where an LP failure leaves the
   * utilization absent; within the torque margin otherwise, where an LP
   * failure is the error. */
  absl::StatusOr<bool> reportTorques(
      const StaticEquilibrium& equilibrium,
      config::CertifierReport::Knot& report) const;

  /** Reports the sphere clearance of a knot at the model's configuration. */
  void reportCollision(ContactMask mask,
                       config::CertifierReport::Knot& report) const;

  /** Every measure of one knot, without the transition condition. */
  absl::StatusOr<config::CertifierReport::Knot> evaluateKnot(
      const CertifierKnot& knot);

  std::unique_ptr<RobotStanceModel> model_;
  TerrainModel terrain_;
  CertifierOptions options_;
  std::optional<CollisionModel> collisionModel_;
};

/**
 * The knots of `plan` for certification: the configurations by joint name and
 * the momentum rates by differences of the momenta. InvalidArgument as
 * StabilityCertifier::certify().
 */
absl::StatusOr<std::vector<CertifierKnot>> certifierKnotsOf(
    const config::MultiContactPlan& plan);

}  // namespace ocs2::humanoid::multi_contact
