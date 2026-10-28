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

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_core/constraint/StateConstraintCppAd.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

namespace ocs2::humanoid::multi_contact {

/** The most edges of a support polygon: one per support direction of the
 * static margin's LPs (StaticEquilibriumOptions::marginDirections,
 * kDefaultMarginDirections). */
inline constexpr int kMaxPolygonEdges = 16;

/**
 * A support polygon in H-representation, a_k . c_xy <= b_k, over an interval
 * of time: the static equilibrium polygon of the stance of that interval.
 * Passive data.
 */
struct TimedPolygon {
  double startTime = 0.0;  // [s]
  double endTime = 0.0;    // [s], >= startTime
  // One row per edge, the outward unit normals; at most kMaxPolygonEdges.
  Eigen::MatrixX2d normals;
  // [m] One per edge.
  Eigen::VectorXd offsets;
};

/**
 * The H-representation of the convex polygon `vertices`, counterclockwise
 * (StaticEquilibrium::supportPolygon()): one outward normal and offset per
 * edge. InvalidArgument for fewer than three vertices, more than
 * kMaxPolygonEdges, or a polygon that is not counterclockwise and convex.
 */
absl::StatusOr<TimedPolygon> polygonOf(
    const std::vector<Eigen::Vector2d>& vertices,
    double startTime,
    double endTime);

/**
 * The static margin of design section 4.3 as linear constraints on the
 * center of mass in planner A's solve 3:
 *
 *   b_k - margin - a_k . c_xy(q) >= 0
 *
 * for every edge of the polygon of the stance at the time (TimedPolygon), the
 * CoM c(q) taped from the configuration of the centroidal state, the edges
 * parameters of the tape (unused rows a = 0, b = 1). Inactive where no
 * polygon covers the time. Wrapped in a relaxed barrier.
 *
 * Cloned per solver worker; a single instance is not thread-safe.
 */
class SupportPolygonConstraint final : public StateConstraintCppAd {
 public:
  /** The polygons `polygons` at `margin` [m] on the MPC's model `pinocchio`
   * (copied), taped as `libraryName` in `folder`. */
  SupportPolygonConstraint(const PinocchioInterface& pinocchio,
                           const CentroidalModelInfo& info,
                           std::vector<TimedPolygon> polygons,
                           scalar_t margin,
                           const std::string& libraryName,
                           const std::string& folder,
                           bool recompile,
                           bool verbose);
  ~SupportPolygonConstraint() override = default;
  SupportPolygonConstraint* absl_nonnull clone() const override {
    return new SupportPolygonConstraint(*this);
  }
  SupportPolygonConstraint& operator=(const SupportPolygonConstraint&) = delete;
  SupportPolygonConstraint(SupportPolygonConstraint&&) = delete;
  SupportPolygonConstraint& operator=(SupportPolygonConstraint&&) = delete;

  bool isActive(scalar_t time) const override;
  size_t getNumConstraints(scalar_t /*time*/) const override {
    return kMaxPolygonEdges;
  }
  vector_t getParameters(scalar_t time,
                         const PreComputation& preComputation) const override;

  /** The number of parameters: the normals' two coordinates and the offset
   * of each edge. */
  static constexpr int kNumParameters = 3 * kMaxPolygonEdges;

 private:
  SupportPolygonConstraint(const SupportPolygonConstraint& other) = default;

  /** The polygon covering `time`; std::nullopt where none does. */
  std::optional<size_t> polygonAt(scalar_t time) const;

  ad_vector_t constraintFunction(ad_scalar_t time,
                                 const ad_vector_t& state,
                                 const ad_vector_t& parameters) const override;

  PinocchioInterfaceCppAd pinocchioCppAd_;
  CentroidalModelInfoCppAd infoCppAd_;
  std::vector<TimedPolygon> polygons_;
  scalar_t margin_ = 0.0;
};

}  // namespace ocs2::humanoid::multi_contact
