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
#include <cmath>
#include <optional>
#include <variant>
#include <vector>

#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "ocs2_core/Dimensions.h"

#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/terrain/PatchFrame.h"
#include "humanoid_multi_contact/terrain/TerrainModel.h"
#include "humanoid_multi_contact/terrain/TerrainPrimitives.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// Below this norm, a projected axis is degenerate.
constexpr double kDegenerateNorm = 1.0e-6;
// Two capsule normals closer than this cosine (about 30 degrees) give one
// sample.
constexpr double kSameNormalCosine = 0.866;

/** The bounds of a face's rectangle in its patch coordinates. */
struct FaceBounds {
  Eigen::Vector2d minPosCornersRtSurfaceEwrtSurfaceXy = Eigen::Vector2d::Zero();
  Eigen::Vector2d maxPosCornersRtSurfaceEwrtSurfaceXy = Eigen::Vector2d::Zero();
};

FaceBounds boundsOf(const PatchFrame& patch) {
  FaceBounds bounds{.minPosCornersRtSurfaceEwrtSurfaceXy =
                        patch.posCornersRtSurfaceEwrtSurfaceXy.front(),
                    .maxPosCornersRtSurfaceEwrtSurfaceXy =
                        patch.posCornersRtSurfaceEwrtSurfaceXy.front()};
  for (const Eigen::Vector2d& vertex : patch.posCornersRtSurfaceEwrtSurfaceXy) {
    bounds.minPosCornersRtSurfaceEwrtSurfaceXy =
        bounds.minPosCornersRtSurfaceEwrtSurfaceXy.cwiseMin(vertex);
    bounds.maxPosCornersRtSurfaceEwrtSurfaceXy =
        bounds.maxPosCornersRtSurfaceEwrtSurfaceXy.cwiseMax(vertex);
  }
  return bounds;
}

bool inside(const Eigen::Vector2d& point,
            const FaceBounds& bounds,
            double margin) {
  // [m] The rounding a grid point exactly at the margin may carry.
  constexpr double kRounding = 1.0e-9;
  return point.x() >= bounds.minPosCornersRtSurfaceEwrtSurfaceXy.x() + margin -
                          kRounding &&
         point.x() <= bounds.maxPosCornersRtSurfaceEwrtSurfaceXy.x() - margin +
                          kRounding &&
         point.y() >= bounds.minPosCornersRtSurfaceEwrtSurfaceXy.y() + margin -
                          kRounding &&
         point.y() <= bounds.maxPosCornersRtSurfaceEwrtSurfaceXy.y() - margin +
                          kRounding;
}

/** Whether the point `posQueryRtWorldEwrtWorld` of a face of normal
 * `normalSurfaceEwrtWorld`, lifted off the face by a little, is that far from
 * the terrain: no other primitive covers the face there. */
bool clearAbove(const TerrainModel& terrain,
                const Eigen::Vector3d& posQueryRtWorldEwrtWorld,
                const Eigen::Vector3d& normalSurfaceEwrtWorld,
                const ContactSamplerOptions& options) {
  // [m] How far the point is lifted.
  constexpr double kLift = 0.02;
  return terrain.signedDistance(Eigen::Vector3d(
             posQueryRtWorldEwrtWorld + kLift * normalSurfaceEwrtWorld)) >=
         kLift - options.penetrationTolerance;
}

/** Whether the pad at `posPadRtWorldEwrtWorld`, its x axis
 * `xAxisPadEwrtWorld`, on a surface of outward normal `normalSurfaceEwrtWorld`
 * is free of the other primitives: its center (and a surface pad's corners) not
 * inside the terrain, and a round pad's geometry not penetrating it. */
bool freeOfTerrain(const TerrainModel& terrain,
                   const ContactCandidate& candidate,
                   const Eigen::Vector3d& posPadRtWorldEwrtWorld,
                   const Eigen::Vector3d& normalSurfaceEwrtWorld,
                   const Eigen::Vector3d& xAxisPadEwrtWorld,
                   const ContactSamplerOptions& options) {
  if (!clearAbove(terrain, posPadRtWorldEwrtWorld, normalSurfaceEwrtWorld,
                  options)) {
    return false;
  }
  if (candidate.rectangle.has_value()) {
    const ContactRectangle& rectangle = *candidate.rectangle;
    const Eigen::Vector3d yAxisPadEwrtWorld =
        normalSurfaceEwrtWorld.cross(xAxisPadEwrtWorld);
    for (const double x : {rectangle.minPosCornersRtPadEwrtPadX,
                           rectangle.maxPosCornersRtPadEwrtPadX}) {
      for (const double y : {rectangle.minPosCornersRtPadEwrtPadY,
                             rectangle.maxPosCornersRtPadEwrtPadY}) {
        if (!clearAbove(
                terrain,
                Eigen::Vector3d(posPadRtWorldEwrtWorld + x * xAxisPadEwrtWorld +
                                y * yAxisPadEwrtWorld),
                normalSurfaceEwrtWorld, options)) {
          return false;
        }
      }
    }
    return true;
  }
  return terrain.sphereDistance(
             Eigen::Vector3d(posPadRtWorldEwrtWorld +
                             candidate.geomRadius * normalSurfaceEwrtWorld),
             candidate.geomRadius) >= -options.penetrationTolerance;
}

/** Whether the candidate's pad at `posPadRtSurfaceEwrtSurfaceXy` (face
 * coordinates), its x axis along `xAxisPadEwrtSurfaceXy`, lies inside the face
 * `margin` from its edges. */
bool padFits(const ContactCandidate& candidate,
             const Eigen::Vector2d& posPadRtSurfaceEwrtSurfaceXy,
             const Eigen::Vector2d& xAxisPadEwrtSurfaceXy,
             const FaceBounds& bounds,
             double margin) {
  if (!candidate.rectangle.has_value()) {
    return inside(posPadRtSurfaceEwrtSurfaceXy, bounds,
                  candidate.geomRadius + margin);
  }
  const ContactRectangle& rectangle = *candidate.rectangle;
  const Eigen::Vector2d yAxisPadEwrtSurfaceXy(-xAxisPadEwrtSurfaceXy.y(),
                                              xAxisPadEwrtSurfaceXy.x());
  for (const double x : {rectangle.minPosCornersRtPadEwrtPadX,
                         rectangle.maxPosCornersRtPadEwrtPadX}) {
    for (const double y : {rectangle.minPosCornersRtPadEwrtPadY,
                           rectangle.maxPosCornersRtPadEwrtPadY}) {
      if (!inside(posPadRtSurfaceEwrtSurfaceXy + x * xAxisPadEwrtSurfaceXy +
                      y * yAxisPadEwrtSurfaceXy,
                  bounds, margin)) {
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
                const Eigen::Vector3d& posReachcenterRtWorldEwrtWorld,
                double maxDistPadRtReachcenter,
                const Eigen::Vector3d& hintXAxisPadEwrtWorld,
                const ContactSamplerOptions& options,
                std::vector<ContactSample>& samples) {
  const Eigen::Vector3d posReachcenterRtSurfaceEwrtSurface =
      patch.rotmatSurfaceRtWorld.transpose() *
      (posReachcenterRtWorldEwrtWorld - patch.posSurfaceRtWorldEwrtWorld);
  if (std::abs(posReachcenterRtSurfaceEwrtSurface.z()) >
      maxDistPadRtReachcenter) {
    return;
  }
  const double radius = std::sqrt(
      maxDistPadRtReachcenter * maxDistPadRtReachcenter -
      posReachcenterRtSurfaceEwrtSurface.z() *
          posReachcenterRtSurfaceEwrtSurface.z());  // in the face plane
  const FaceBounds bounds = boundsOf(patch);
  const Eigen::Vector3d xAxisPadEwrtWorld = computeXAxisPadEwrtWorld(
      hintXAxisPadEwrtWorld, patch.getNormalSurfaceEwrtWorld());
  const Eigen::Vector2d xAxisPadEwrtSurfaceXy =
      (patch.rotmatSurfaceRtWorld.transpose() * xAxisPadEwrtWorld)
          .head<kHorizontalPositionDim>();
  const double spacing = options.gridSpacing;
  const int iLow = static_cast<int>(
      std::ceil(std::max(bounds.minPosCornersRtSurfaceEwrtSurfaceXy.x(),
                         posReachcenterRtSurfaceEwrtSurface.x() - radius) /
                spacing));
  const int iHigh = static_cast<int>(
      std::floor(std::min(bounds.maxPosCornersRtSurfaceEwrtSurfaceXy.x(),
                          posReachcenterRtSurfaceEwrtSurface.x() + radius) /
                 spacing));
  const int jLow = static_cast<int>(
      std::ceil(std::max(bounds.minPosCornersRtSurfaceEwrtSurfaceXy.y(),
                         posReachcenterRtSurfaceEwrtSurface.y() - radius) /
                spacing));
  const int jHigh = static_cast<int>(
      std::floor(std::min(bounds.maxPosCornersRtSurfaceEwrtSurfaceXy.y(),
                          posReachcenterRtSurfaceEwrtSurface.y() + radius) /
                 spacing));
  for (int i = iLow; i <= iHigh; ++i) {
    for (int j = jLow; j <= jHigh; ++j) {
      const Eigen::Vector2d posPadRtSurfaceEwrtSurfaceXy(i * spacing,
                                                         j * spacing);
      if (!padFits(candidate, posPadRtSurfaceEwrtSurfaceXy,
                   xAxisPadEwrtSurfaceXy, bounds, options.edgeMargin)) {
        continue;
      }
      const Eigen::Vector3d posPadRtWorldEwrtWorld =
          patch.posSurfaceRtWorldEwrtWorld +
          patch.rotmatSurfaceRtWorld.col(kXAxisIndex) *
              posPadRtSurfaceEwrtSurfaceXy.x() +
          patch.rotmatSurfaceRtWorld.col(kYAxisIndex) *
              posPadRtSurfaceEwrtSurfaceXy.y();
      if ((posPadRtWorldEwrtWorld - posReachcenterRtWorldEwrtWorld).norm() >
              maxDistPadRtReachcenter ||
          !freeOfTerrain(terrain, candidate, posPadRtWorldEwrtWorld,
                         patch.getNormalSurfaceEwrtWorld(), xAxisPadEwrtWorld,
                         options)) {
        continue;
      }
      samples.push_back(ContactSample{
          .candidate = candidateIndex,
          .primitive = patch.primitiveIndex,
          .face = patch.face,
          .posPadRtWorldEwrtWorld = posPadRtWorldEwrtWorld,
          .normalSurfaceEwrtWorld = patch.getNormalSurfaceEwrtWorld(),
          .xAxisPadEwrtWorld = xAxisPadEwrtWorld});
    }
  }
}

/** Appends the samples of one capsule: on its top and its side facing
 * `posReachcenterRtWorldEwrtWorld`. */
void sampleCapsule(const TerrainModel& terrain,
                   const CapsuleGeometry& capsule,
                   int primitive,
                   const ContactCandidate& candidate,
                   int candidateIndex,
                   const Eigen::Vector3d& posReachcenterRtWorldEwrtWorld,
                   double maxDistPadRtReachcenter,
                   const ContactSamplerOptions& options,
                   std::vector<ContactSample>& samples) {
  const Eigen::Vector3d posCapsuleendRtCapsulestartEwrtWorld =
      capsule.posCapsuleendRtWorldEwrtWorld -
      capsule.posCapsulestartRtWorldEwrtWorld;
  const double distCapsuleendRtCapsulestart =
      posCapsuleendRtCapsulestartEwrtWorld.norm();
  if (distCapsuleendRtCapsulestart < kDegenerateNorm) return;
  const Eigen::Vector3d zAxisCapsulestartEwrtWorld =
      posCapsuleendRtCapsulestartEwrtWorld / distCapsuleendRtCapsulestart;
  const int count = static_cast<int>(
      std::floor(distCapsuleendRtCapsulestart / options.gridSpacing));
  for (int k = 0; k <= count; ++k) {
    const Eigen::Vector3d posSegmentpointRtWorldEwrtWorld =
        capsule.posCapsulestartRtWorldEwrtWorld +
        zAxisCapsulestartEwrtWorld * (k * options.gridSpacing);
    std::vector<Eigen::Vector3d> normalSurfacesEwrtWorld;
    for (const Eigen::Vector3d& towardDirEwrtWorld :
         {Eigen::Vector3d(Eigen::Vector3d::UnitZ()),
          Eigen::Vector3d(posReachcenterRtWorldEwrtWorld -
                          posSegmentpointRtWorldEwrtWorld)}) {
      Eigen::Vector3d normalSurfaceEwrtWorld =
          towardDirEwrtWorld -
          towardDirEwrtWorld.dot(zAxisCapsulestartEwrtWorld) *
              zAxisCapsulestartEwrtWorld;
      if (normalSurfaceEwrtWorld.norm() < kDegenerateNorm) continue;
      normalSurfaceEwrtWorld.normalize();
      if (normalSurfaceEwrtWorld.z() <
          options.pointMinNormalSurfaceEwrtWorldZ) {
        continue;
      }
      if (!normalSurfacesEwrtWorld.empty() &&
          normalSurfacesEwrtWorld.front().dot(normalSurfaceEwrtWorld) >
              kSameNormalCosine) {
        continue;
      }
      normalSurfacesEwrtWorld.push_back(normalSurfaceEwrtWorld);
    }
    for (const Eigen::Vector3d& normalSurfaceEwrtWorld :
         normalSurfacesEwrtWorld) {
      const Eigen::Vector3d posPadRtWorldEwrtWorld =
          posSegmentpointRtWorldEwrtWorld +
          capsule.radius * normalSurfaceEwrtWorld;
      const Eigen::Vector3d xAxisPadEwrtWorld = computeXAxisPadEwrtWorld(
          zAxisCapsulestartEwrtWorld, normalSurfaceEwrtWorld);
      if ((posPadRtWorldEwrtWorld - posReachcenterRtWorldEwrtWorld).norm() >
              maxDistPadRtReachcenter ||
          !freeOfTerrain(terrain, candidate, posPadRtWorldEwrtWorld,
                         normalSurfaceEwrtWorld, xAxisPadEwrtWorld, options)) {
        continue;
      }
      samples.push_back(
          ContactSample{.candidate = candidateIndex,
                        .primitive = primitive,
                        .face = std::nullopt,
                        .posPadRtWorldEwrtWorld = posPadRtWorldEwrtWorld,
                        .normalSurfaceEwrtWorld = normalSurfaceEwrtWorld,
                        .xAxisPadEwrtWorld = xAxisPadEwrtWorld});
    }
  }
}

}  // namespace

Eigen::Vector3d computeXAxisPadEwrtWorld(
    const Eigen::Vector3d& hintXAxisPadEwrtWorld,
    const Eigen::Vector3d& normalSurfaceEwrtWorld) {
  Eigen::Vector3d xAxisPadEwrtWorld =
      hintXAxisPadEwrtWorld -
      hintXAxisPadEwrtWorld.dot(normalSurfaceEwrtWorld) *
          normalSurfaceEwrtWorld;
  if (xAxisPadEwrtWorld.norm() < kDegenerateNorm) {
    return normalSurfaceEwrtWorld.unitOrthogonal();
  }
  return xAxisPadEwrtWorld.normalized();
}

std::vector<ContactSample> sampleContacts(
    const TerrainModel& terrain,
    const ContactCandidate& candidate,
    int candidateIndex,
    const Eigen::Vector3d& posReachcenterRtWorldEwrtWorld,
    double maxDistPadRtReachcenter,
    const Eigen::Vector3d& hintXAxisPadEwrtWorld,
    const ContactSamplerOptions& options) {
  std::vector<ContactSample> samples;
  const bool surface = candidate.kind == ContactKind::kSurface;
  const double minNormalSurfaceEwrtWorldZ =
      surface ? options.rectangleMinNormalSurfaceEwrtWorldZ
              : options.pointMinNormalSurfaceEwrtWorldZ;
  for (int primitive = 0; primitive < terrain.numPrimitives(); ++primitive) {
    const TerrainPrimitive& shape = terrain.primitive(primitive);
    if (const CapsuleGeometry* absl_nullable capsule =
            std::get_if<CapsuleGeometry>(&shape.geometry)) {
      if (!surface) {
        sampleCapsule(terrain, *capsule, primitive, candidate, candidateIndex,
                      posReachcenterRtWorldEwrtWorld, maxDistPadRtReachcenter,
                      options, samples);
      }
      continue;
    }
    for (int face = 0; face < kBoxFaceCount; ++face) {
      const absl::StatusOr<PatchFrame> patch =
          terrain.boxFaceFrame(primitive, static_cast<BoxFace>(face));
      if (!patch.ok() ||
          patch->getNormalSurfaceEwrtWorld().z() < minNormalSurfaceEwrtWorldZ) {
        continue;
      }
      sampleFace(terrain, *patch, candidate, candidateIndex,
                 posReachcenterRtWorldEwrtWorld, maxDistPadRtReachcenter,
                 hintXAxisPadEwrtWorld, options, samples);
    }
  }
  return samples;
}

}  // namespace ocs2::humanoid::multi_contact
