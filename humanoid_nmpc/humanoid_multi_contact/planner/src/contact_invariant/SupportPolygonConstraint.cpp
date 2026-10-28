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

#include "humanoid_multi_contact/planner/contact_invariant/SupportPolygonConstraint.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ocs2_centroidal_model/CentroidalModelPinocchioMapping.h"
#include "ocs2_core/Dimensions.h"
#include "pinocchio/algorithm/center-of-mass.hpp"

namespace ocs2::humanoid::multi_contact {
namespace {

// The polygon has at least a triangle's vertices.
constexpr size_t kMinimumVertices = 3;
// [m] An edge shorter than this has no normal, and a vertex this far outside
// an edge's line makes the polygon not convex.
constexpr double kGeometricTolerance = 1.0e-9;
// Where the offsets follow the normals among the parameters.
constexpr int kOffsetParameters = 2 * kMaxPolygonEdges;

}  // namespace

absl::StatusOr<TimedPolygon> polygonOf(
    const std::vector<Eigen::Vector2d>& vertices,
    double startTime,
    double endTime) {
  const size_t n = vertices.size();
  if (n < kMinimumVertices || n > static_cast<size_t>(kMaxPolygonEdges)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "support polygon: ", n, " vertices, not 3 to ", kMaxPolygonEdges));
  }
  TimedPolygon polygon;
  polygon.startTime = startTime;
  polygon.endTime = endTime;
  polygon.normals.resize(static_cast<Eigen::Index>(n), kHorizontalPositionDim);
  polygon.offsets.resize(static_cast<Eigen::Index>(n));
  for (size_t k = 0; k < n; ++k) {
    const Eigen::Vector2d edge = vertices[(k + 1) % n] - vertices[k];
    if (edge.norm() < kGeometricTolerance) {
      return absl::InvalidArgumentError(
          absl::StrCat("support polygon: edge ", k, " has no length"));
    }
    // Counterclockwise: the outward normal is the edge turned clockwise.
    const Eigen::Vector2d normal =
        Eigen::Vector2d(edge.y(), -edge.x()) / edge.norm();
    polygon.normals.row(static_cast<Eigen::Index>(k)) = normal.transpose();
    polygon.offsets(static_cast<Eigen::Index>(k)) = normal.dot(vertices[k]);
  }
  for (size_t k = 0; k < n; ++k) {
    for (const Eigen::Vector2d& vertex : vertices) {
      if (polygon.normals.row(static_cast<Eigen::Index>(k)).dot(vertex) >
          polygon.offsets(static_cast<Eigen::Index>(k)) + kGeometricTolerance) {
        return absl::InvalidArgumentError(absl::StrCat(
            "support polygon: not counterclockwise and convex at edge ", k));
      }
    }
  }
  return polygon;
}

SupportPolygonConstraint::SupportPolygonConstraint(
    const PinocchioInterface& pinocchio,
    const CentroidalModelInfo& info,
    std::vector<TimedPolygon> polygons,
    scalar_t margin,
    const std::string& libraryName,
    const std::string& folder,
    bool recompile,
    bool verbose)
    : StateConstraintCppAd(ConstraintOrder::Linear),
      pinocchioCppAd_(pinocchio.toCppAd()),
      infoCppAd_(info.toCppAd()),
      polygons_(std::move(polygons)),
      margin_(margin) {
  initialize(info.stateDim, kNumParameters, libraryName, folder, recompile,
             verbose);
}

std::optional<size_t> SupportPolygonConstraint::polygonAt(scalar_t time) const {
  for (size_t k = 0; k < polygons_.size(); ++k) {
    if (time >= polygons_[k].startTime && time <= polygons_[k].endTime) {
      return k;
    }
  }
  return std::nullopt;
}

bool SupportPolygonConstraint::isActive(scalar_t time) const {
  return polygonAt(time).has_value();
}

vector_t SupportPolygonConstraint::getParameters(
    scalar_t time, const PreComputation& /*preComputation*/) const {
  vector_t parameters = vector_t::Zero(kNumParameters);
  // Unused rows: a = 0, b = 1, h = 1 - margin > 0 whatever the CoM.
  parameters.tail(kMaxPolygonEdges).setOnes();
  if (const std::optional<size_t> index = polygonAt(time); index.has_value()) {
    const TimedPolygon& polygon = polygons_[*index];
    for (Eigen::Index k = 0; k < polygon.offsets.size(); ++k) {
      parameters(kHorizontalPositionDim * k) = polygon.normals.row(k).x();
      parameters(kHorizontalPositionDim * k + 1) = polygon.normals.row(k).y();
      parameters(kOffsetParameters + k) = polygon.offsets(k);
    }
  }
  return parameters;
}

ad_vector_t SupportPolygonConstraint::constraintFunction(
    ad_scalar_t /*time*/,
    const ad_vector_t& state,
    const ad_vector_t& parameters) const {
  PinocchioInterfaceCppAd interface = pinocchioCppAd_;
  const CentroidalModelPinocchioMappingCppAd mapping(infoCppAd_);
  const ad_vector_t q = mapping.getPinocchioJointPosition(state);
  const Eigen::Matrix<ad_scalar_t, kPositionDim, 1> com =
      pinocchio::centerOfMass(interface.getModel(), interface.getData(), q,
                              /*computeSubtreeComs=*/false);
  ad_vector_t value(kMaxPolygonEdges);
  for (int k = 0; k < kMaxPolygonEdges; ++k) {
    value(k) = parameters(kOffsetParameters + k) - ad_scalar_t(margin_) -
               parameters(kHorizontalPositionDim * k) * com.x() -
               parameters(kHorizontalPositionDim * k + 1) * com.y();
  }
  return value;
}

}  // namespace ocs2::humanoid::multi_contact
