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

#include "humanoid_multi_contact/planner/ContactSampler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <variant>
#include <vector>

#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "ocs2_core/Dimensions.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// Below this norm, a projected axis is degenerate.
constexpr double kDegenerateNorm = 1.0e-6;
// Two capsule normals closer than this cosine (about 30 degrees) give one
// sample.
constexpr double kSameNormalCosine = 0.866;

/** The bounds of a face's rectangle in its patch coordinates. */
struct FaceBounds {
  Eigen::Vector2d lower = Eigen::Vector2d::Zero();
  Eigen::Vector2d upper = Eigen::Vector2d::Zero();
};

FaceBounds boundsOf(const PatchFrame& patch) {
  FaceBounds bounds{.lower = patch.polygon.front(),
                    .upper = patch.polygon.front()};
  for (const Eigen::Vector2d& vertex : patch.polygon) {
    bounds.lower = bounds.lower.cwiseMin(vertex);
    bounds.upper = bounds.upper.cwiseMax(vertex);
  }
  return bounds;
}

bool inside(const Eigen::Vector2d& point,
            const FaceBounds& bounds,
            double margin) {
  // [m] The rounding a grid point exactly at the margin may carry.
  constexpr double kRounding = 1.0e-9;
  return point.x() >= bounds.lower.x() + margin - kRounding &&
         point.x() <= bounds.upper.x() - margin + kRounding &&
         point.y() >= bounds.lower.y() + margin - kRounding &&
         point.y() <= bounds.upper.y() - margin + kRounding;
}

/** Whether the point `point` of a face of normal `normal`, lifted off the face
 * by a little, is that far from the terrain: no other primitive covers the
 * face there. */
bool clearAbove(const TerrainModel& terrain,
                const Eigen::Vector3d& point,
                const Eigen::Vector3d& normal,
                const ContactSamplerOptions& options) {
  // [m] How far the point is lifted.
  constexpr double kLift = 0.02;
  return terrain.signedDistance(Eigen::Vector3d(point + kLift * normal)) >=
         kLift - options.penetrationTolerance;
}

/** Whether the pad at `position` on a surface of normal `normal` is free of
 * the other primitives: its center (and a surface pad's corners) not inside
 * the terrain, and a round pad's geometry not penetrating it. */
bool freeOfTerrain(const TerrainModel& terrain,
                   const ContactCandidate& candidate,
                   const Eigen::Vector3d& position,
                   const Eigen::Vector3d& normal,
                   const Eigen::Vector3d& xAxis,
                   const ContactSamplerOptions& options) {
  if (!clearAbove(terrain, position, normal, options)) return false;
  if (candidate.rectangle.has_value()) {
    const ContactRectangle& rectangle = *candidate.rectangle;
    const Eigen::Vector3d yAxis = normal.cross(xAxis);
    for (const double x : {rectangle.xMin, rectangle.xMax}) {
      for (const double y : {rectangle.yMin, rectangle.yMax}) {
        if (!clearAbove(terrain,
                        Eigen::Vector3d(position + x * xAxis + y * yAxis),
                        normal, options)) {
          return false;
        }
      }
    }
    return true;
  }
  return terrain.sphereDistance(
             Eigen::Vector3d(position + candidate.geomRadius * normal),
             candidate.geomRadius) >= -options.penetrationTolerance;
}

/** Whether the candidate's pad at `point` (face coordinates), its x axis along
 * `axis`, lies inside the face `margin` from its edges. */
bool padFits(const ContactCandidate& candidate,
             const Eigen::Vector2d& point,
             const Eigen::Vector2d& axis,
             const FaceBounds& bounds,
             double margin) {
  if (!candidate.rectangle.has_value()) {
    return inside(point, bounds, candidate.geomRadius + margin);
  }
  const ContactRectangle& rectangle = *candidate.rectangle;
  const Eigen::Vector2d perpendicular(-axis.y(), axis.x());
  for (const double x : {rectangle.xMin, rectangle.xMax}) {
    for (const double y : {rectangle.yMin, rectangle.yMax}) {
      if (!inside(point + x * axis + y * perpendicular, bounds, margin)) {
        return false;
      }
    }
  }
  return true;
}

/** Appends the samples of one box face. */
void sampleFace(const TerrainModel& terrain,
                const PatchFrame& patch,
                const ContactCandidate& candidate,
                int candidateIndex,
                const Eigen::Vector3d& center,
                double reach,
                const Eigen::Vector3d& xAxisHint,
                const ContactSamplerOptions& options,
                std::vector<ContactSample>& samples) {
  const Eigen::Vector3d local =
      patch.rotation.transpose() * (center - patch.origin);
  if (std::abs(local.z()) > reach) return;
  const double radius =
      std::sqrt(reach * reach - local.z() * local.z());  // in the face plane
  const FaceBounds bounds = boundsOf(patch);
  const Eigen::Vector3d xAxis = tangentAxis(xAxisHint, patch.normal());
  const Eigen::Vector2d axis =
      (patch.rotation.transpose() * xAxis).head<kHorizontalPositionDim>();
  const double spacing = options.gridSpacing;
  const int iLow = static_cast<int>(
      std::ceil(std::max(bounds.lower.x(), local.x() - radius) / spacing));
  const int iHigh = static_cast<int>(
      std::floor(std::min(bounds.upper.x(), local.x() + radius) / spacing));
  const int jLow = static_cast<int>(
      std::ceil(std::max(bounds.lower.y(), local.y() - radius) / spacing));
  const int jHigh = static_cast<int>(
      std::floor(std::min(bounds.upper.y(), local.y() + radius) / spacing));
  for (int i = iLow; i <= iHigh; ++i) {
    for (int j = jLow; j <= jHigh; ++j) {
      const Eigen::Vector2d point(i * spacing, j * spacing);
      if (!padFits(candidate, point, axis, bounds, options.edgeMargin)) {
        continue;
      }
      const Eigen::Vector3d position =
          patch.origin + patch.rotation.col(kXAxisIndex) * point.x() +
          patch.rotation.col(kYAxisIndex) * point.y();
      if ((position - center).norm() > reach ||
          !freeOfTerrain(terrain, candidate, position, patch.normal(), xAxis,
                         options)) {
        continue;
      }
      samples.push_back(ContactSample{.candidate = candidateIndex,
                                      .primitive = patch.primitiveIndex,
                                      .face = patch.face,
                                      .position = position,
                                      .normal = patch.normal(),
                                      .xAxis = xAxis});
    }
  }
}

/** Appends the samples of one capsule: on its top and its side facing
 * `center`. */
void sampleCapsule(const TerrainModel& terrain,
                   const CapsuleGeometry& capsule,
                   int primitive,
                   const ContactCandidate& candidate,
                   int candidateIndex,
                   const Eigen::Vector3d& center,
                   double reach,
                   const ContactSamplerOptions& options,
                   std::vector<ContactSample>& samples) {
  const Eigen::Vector3d axis = capsule.end - capsule.start;
  const double length = axis.norm();
  if (length < kDegenerateNorm) return;
  const Eigen::Vector3d direction = axis / length;
  const int count = static_cast<int>(std::floor(length / options.gridSpacing));
  for (int k = 0; k <= count; ++k) {
    const Eigen::Vector3d onAxis =
        capsule.start + direction * (k * options.gridSpacing);
    std::vector<Eigen::Vector3d> normals;
    for (const Eigen::Vector3d& toward :
         {Eigen::Vector3d(Eigen::Vector3d::UnitZ()),
          Eigen::Vector3d(center - onAxis)}) {
      Eigen::Vector3d normal = toward - toward.dot(direction) * direction;
      if (normal.norm() < kDegenerateNorm) continue;
      normal.normalize();
      if (normal.z() < options.minPointNormalZ) continue;
      if (!normals.empty() && normals.front().dot(normal) > kSameNormalCosine) {
        continue;
      }
      normals.push_back(normal);
    }
    for (const Eigen::Vector3d& normal : normals) {
      const Eigen::Vector3d position = onAxis + capsule.radius * normal;
      const Eigen::Vector3d xAxis = tangentAxis(direction, normal);
      if ((position - center).norm() > reach ||
          !freeOfTerrain(terrain, candidate, position, normal, xAxis,
                         options)) {
        continue;
      }
      samples.push_back(ContactSample{.candidate = candidateIndex,
                                      .primitive = primitive,
                                      .face = std::nullopt,
                                      .position = position,
                                      .normal = normal,
                                      .xAxis = xAxis});
    }
  }
}

}  // namespace

Eigen::Vector3d tangentAxis(const Eigen::Vector3d& hint,
                            const Eigen::Vector3d& normal) {
  Eigen::Vector3d axis = hint - hint.dot(normal) * normal;
  if (axis.norm() < kDegenerateNorm) return normal.unitOrthogonal();
  return axis.normalized();
}

std::vector<ContactSample> sampleContacts(
    const TerrainModel& terrain,
    const ContactCandidate& candidate,
    int candidateIndex,
    const Eigen::Vector3d& center,
    double reach,
    const Eigen::Vector3d& xAxisHint,
    const ContactSamplerOptions& options) {
  std::vector<ContactSample> samples;
  const bool surface = candidate.kind == ContactKind::kSurface;
  const double minNormalZ =
      surface ? options.minSurfaceNormalZ : options.minPointNormalZ;
  for (int primitive = 0; primitive < terrain.numPrimitives(); ++primitive) {
    const TerrainPrimitive& shape = terrain.primitive(primitive);
    if (const CapsuleGeometry* absl_nullable capsule =
            std::get_if<CapsuleGeometry>(&shape.geometry)) {
      if (!surface) {
        sampleCapsule(terrain, *capsule, primitive, candidate, candidateIndex,
                      center, reach, options, samples);
      }
      continue;
    }
    for (int face = 0; face < kBoxFaceCount; ++face) {
      const absl::StatusOr<PatchFrame> patch =
          terrain.boxFaceFrame(primitive, static_cast<BoxFace>(face));
      if (!patch.ok() || patch->normal().z() < minNormalZ) continue;
      sampleFace(terrain, *patch, candidate, candidateIndex, center, reach,
                 xAxisHint, options, samples);
    }
  }
  return samples;
}

}  // namespace ocs2::humanoid::multi_contact
