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

#include <array>
#include <optional>
#include <vector>

#include "Eigen/Core"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "humanoid_multi_contact/stability/Stance.h"

namespace ocs2::humanoid::multi_contact {

/** The horizontal directions of the static margin by default (design section
 * 2). */
inline constexpr int kDefaultMarginDirections = 16;

/**
 * The L1 norm of the normalized wrench mismatch (force over the weight, moment
 * over the weight in meters) below which a feasibility LP counts as feasible.
 */
inline constexpr double kDefaultFeasibilityTolerance = 1.0e-6;

/** [m] StaticEquilibriumOptions::searchRadius by default. */
inline constexpr double kDefaultSearchRadius = 5.0;

/** How supportPolygon() approximates the static equilibrium polygon, selected
 * by name (Scenario::support_polygon, kSupportPolygonNames). */
enum class SupportPolygon {
  // The hull of the support points along the margin directions alone.
  kSampled,
  // That hull grown by a support point along the outward normal of each of
  // its edges until no edge moves by more than kRefinementTolerance, or
  // kMaxRefinementPoints more points (Bretl and Lall's projection): a thin
  // polygon whose corner lies between two margin directions (two pads along a
  // box's edge and a foot) keeps that corner, which the sampled hull cuts off
  // or keeps by how an LP breaks a tie.
  kRefined,
};

/** The names of SupportPolygon's values, in their order. */
// LINT.IfChange(support_polygon_names)
inline constexpr std::array<absl::string_view, 2> kSupportPolygonNames = {
    "sampled", "refined"};
// LINT.ThenChange(//humanoid_nmpc/humanoid_multi_contact/scenarios/scenario.proto:support_polygon_names)

/** The SupportPolygon named `name`; InvalidArgument listing
 * kSupportPolygonNames for another name. */
absl::StatusOr<SupportPolygon> supportPolygonOf(absl::string_view name);

/** [m] How far beyond an edge of the refined hull a support point must lie to
 * grow it (SupportPolygon::kRefined). */
inline constexpr double kRefinementTolerance = 1.0e-4;

/** The most support points SupportPolygon::kRefined adds to the sampled ones.
 */
inline constexpr int kMaxRefinementPoints = 32;

/** The settings of StaticEquilibrium. */
struct StaticEquilibriumOptions {
  int frictionConeEdges = kDefaultFrictionConeEdges;
  // At least three.
  int marginDirections = kDefaultMarginDirections;
  SupportPolygon supportPolygon = SupportPolygon::kSampled;
  // [m] How far from the stance's reference point the support LPs look for
  // equilibrium CoM positions: a bound for the stances whose equilibrium set is
  // unbounded, such as two hands pressing on opposing walls.
  double searchRadius = kDefaultSearchRadius;
  double feasibilityTolerance = kDefaultFeasibilityTolerance;
};

/**
 * The static equilibrium of a stance with gravity along -z (design section 2,
 * Bretl and Lall): contact forces in their linearized cones that balance the
 * weight, for a center of mass at a horizontal position c,
 *
 *   sum_g lambda_g w_g = (e_z, (c - r) x e_z),  lambda >= 0,
 *
 * with the generators w_g of the stance (WrenchGenerators, moments about r) and
 * the forces in units of the robot's weight, so the CoM height does not matter.
 * The set of such c is a convex polygon.
 *
 * supportPolygon() computes an inner approximation of that polygon without
 * building it: one LP per horizontal direction d_k finds the equilibrium CoM
 * position farthest along d_k, and the convex hull of those support points lies
 * inside the polygon, touching its boundary at each of them. Its signed
 * distance to the CoM (staticMargin()) is therefore a lower bound on the true
 * margin rho_s that tightens with more directions, and its sign flips where the
 * CoM crosses the hull's boundary. SupportPolygon::kRefined then grows the
 * hull along the outward normals of its edges until it stops growing. The LPs
 * run on HPIPM's dense QP (solveLinearProgram()).
 *
 * A value type, immutable after Create(); the methods are const and safe to
 * call concurrently.
 */
class StaticEquilibrium {
 public:
  /** Builds the generators of `stance`; InvalidArgument as
   * buildWrenchGenerators(), or for fewer than three margin directions or a
   * search radius or tolerance that is not positive. */
  static absl::StatusOr<StaticEquilibrium> Create(
      Stance stance, const StaticEquilibriumOptions& options);

  StaticEquilibrium(const StaticEquilibrium&) = default;
  StaticEquilibrium& operator=(const StaticEquilibrium&) = default;
  StaticEquilibrium(StaticEquilibrium&&) noexcept = default;
  StaticEquilibrium& operator=(StaticEquilibrium&&) noexcept = default;
  ~StaticEquilibrium() = default;

  const Stance& stance() const { return stance_; }
  const WrenchGenerators& generators() const { return generators_; }
  const StaticEquilibriumOptions& options() const { return options_; }

  /** The required normalized contact wrench, about the generators' reference
   * point, that holds the robot still with its CoM above `comXy`. */
  Wrench staticWrench(const Eigen::Vector2d& comXy) const;

  /** The L1 norm of the smallest mismatch between the stance's wrenches and
   * `wrench` (normalized, about the reference point): 0 exactly when the
   * stance can apply it. One LP. */
  absl::StatusOr<double> wrenchResidual(const Wrench& wrench) const;

  /** wrenchResidual(staticWrench(comXy)). */
  absl::StatusOr<double> equilibriumResidual(
      const Eigen::Vector2d& comXy) const;

  /**
   * The inner approximation of the static equilibrium polygon, counterclockwise
   * (convexHull()); empty when no CoM position within the search radius is in
   * equilibrium, as for a stance on a vertical wall alone. One LP to test that,
   * then one per direction, and under SupportPolygon::kRefined one per edge
   * of each grown hull, at most kMaxRefinementPoints more. Internal when HPIPM
   * fails.
   */
  absl::StatusOr<std::vector<Eigen::Vector2d>> supportPolygon() const;

  /**
   * Scalings lambda >= 0 of the generators that apply `wrench` exactly, with
   * the dependent rows of the generator matrix removed, minimizing
   * cost' lambda; the program's other rows are added by the caller: `rows` and
   * `lower` <= rows * [lambda; extra] <= `upper` over `extraVariables` more
   * variables with cost `extraCost`. Shared by the torque LP; InvalidArgument
   * for inconsistent sizes.
   */
  absl::StatusOr<Eigen::VectorXd> solveWithExtraVariables(
      const Wrench& wrench,
      const Eigen::VectorXd& extraCost,
      const Eigen::MatrixXd& rows,
      const Eigen::VectorXd& lower,
      const Eigen::VectorXd& upper) const;

 private:
  StaticEquilibrium(Stance stance,
                    WrenchGenerators generators,
                    const StaticEquilibriumOptions& options);

  /** The CoM position farthest along `direction` among those in equilibrium,
   * relative to the reference point. */
  absl::StatusOr<Eigen::Vector2d> supportPoint(
      const Eigen::Vector2d& direction) const;

  /** The hull of `points` (support points, world frame) grown along the
   * outward normals of its edges (SupportPolygon::kRefined). */
  absl::StatusOr<std::vector<Eigen::Vector2d>> refinedHull(
      std::vector<Eigen::Vector2d> points) const;

  Stance stance_;
  WrenchGenerators generators_;
  StaticEquilibriumOptions options_;
};

/**
 * [m] The signed distance of `comXy` to the boundary of `polygon`
 * (supportPolygon()), positive inside; std::nullopt for an empty polygon, when
 * no CoM position is in equilibrium. A polygon of one or two vertices has no
 * inside, so its margin is at most 0.
 */
std::optional<double> staticMargin(absl::Span<const Eigen::Vector2d> polygon,
                                   const Eigen::Vector2d& comXy);

}  // namespace ocs2::humanoid::multi_contact
