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

#include <cmath>
#include <string>
#include <variant>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_multi_contact/terrain/PatchFrame.h"
#include "humanoid_multi_contact/terrain/TerrainPrimitives.h"
#include "humanoid_multi_contact/terrain/terrain_scene.nproto.h"

namespace ocs2::humanoid::multi_contact {

/**
 * [m] The scale of the smooth minimum across primitives: -s log(sum_i
 * exp(-d_i / s)). It lies below the true minimum by at most s log(n), s log(2)
 * = 3.5 mm where two primitives are equally near, such as at the inside corner
 * of a stair, and it equals the minimum where one primitive is nearer than the
 * others by several s.
 */
inline constexpr double kSmoothMinimumScale = 0.005;

/** One validated primitive of a TerrainScene. Passive data. */
struct TerrainPrimitive {
  std::string name;
  double friction = 0.0;
  std::variant<BoxGeometry, CapsuleGeometry> geometry;
};

/** The primitive nearest to a point, and its exact query there. */
struct NearestPrimitive {
  int index = 0;
  PrimitiveQuery<double> query;
};

/**
 * The terrain of a scenario (design section 3.2): the signed distance of a
 * point to the scene, its nearest surface point and outward normal, the
 * distance of a sphere, and the patch frames of the faces that contacts rest
 * on.
 *
 * The distance across primitives is their smooth minimum (kSmoothMinimumScale),
 * so its gradient is defined where two primitives meet; a scene of one
 * primitive, such as a flat floor, gives that primitive's exact distance. The
 * templated queries run in double and on a CppAD tape (TerrainPrimitives.h):
 * record them with the AD scalar of the tape, and the scene enters the tape as
 * constants, so a tape recorded from one TerrainModel is valid for that scene
 * only.
 *
 * A value type, immutable after Create(): every method is const and safe to
 * call concurrently.
 */
class TerrainModel {
 public:
  /** Validates `scene` and builds its model. InvalidArgument names the
   * primitive and the field at fault: no primitive, a duplicate or empty name,
   * a friction coefficient that is not positive, a vector of the wrong size, a
   * quaternion that is not unit, a half extent or radius that is not positive,
   * or no shape. */
  static absl::StatusOr<TerrainModel> Create(const config::TerrainScene& scene);

  TerrainModel(const TerrainModel&) = default;
  TerrainModel& operator=(const TerrainModel&) = default;
  TerrainModel(TerrainModel&&) noexcept = default;
  TerrainModel& operator=(TerrainModel&&) noexcept = default;
  ~TerrainModel() = default;

  const std::string& name() const { return name_; }
  int numPrimitives() const { return static_cast<int>(primitives_.size()); }
  /** The primitive at `index`; requires 0 <= index < numPrimitives(). */
  const TerrainPrimitive& primitive(int index) const {
    return primitives_[index];
  }
  /** The index of the primitive `name`; NotFound names the scene. */
  absl::StatusOr<int> primitiveIndex(absl::string_view name) const;

  /** The exact query of one primitive; requires 0 <= index < numPrimitives().
   */
  template <typename Scalar>
  PrimitiveQuery<Scalar> primitiveQuery(int index,
                                        const Vector3<Scalar>& point) const {
    const TerrainPrimitive& primitive = primitives_[index];
    if (const BoxGeometry* absl_nullable box =
            std::get_if<BoxGeometry>(&primitive.geometry)) {
      return boxQuery(*box, point);
    }
    return capsuleQuery(std::get<CapsuleGeometry>(primitive.geometry), point);
  }

  /** [m] The smooth minimum of the primitives' signed distances at `point`,
   * negative inside the scene. */
  template <typename Scalar>
  Scalar signedDistance(const Vector3<Scalar>& point) const {
    return blend(point).distance;
  }

  /** The unit outward normal at `point`: the gradient of signedDistance(), the
   * softmin-weighted sum of the primitives' normals, normalized. */
  template <typename Scalar>
  Vector3<Scalar> normal(const Vector3<Scalar>& point) const {
    return blend(point).normal;
  }

  /** point - signedDistance(point) * normal(point): exact where one primitive
   * is nearest by several kSmoothMinimumScale, a blend where several are. */
  template <typename Scalar>
  Vector3<Scalar> nearestPoint(const Vector3<Scalar>& point) const {
    const PrimitiveQuery<Scalar> query = blend(point);
    return point - query.distance * query.normal;
  }

  /** [m] The signed distance of a sphere to the scene: negative when it
   * penetrates. Exact for one primitive. */
  template <typename Scalar>
  Scalar sphereDistance(const Vector3<Scalar>& center,
                        const Scalar& radius) const {
    return signedDistance(center) - radius;
  }

  /** The primitive with the smallest exact signed distance at `point` (the
   * first one on a tie). */
  NearestPrimitive nearestPrimitive(const Eigen::Vector3d& point) const;

  /** The frame and polygon of a face of a box primitive; InvalidArgument when
   * the index is out of range or the primitive is a capsule. */
  absl::StatusOr<PatchFrame> boxFaceFrame(int index, BoxFace face) const;

  /**
   * The patch a contact at `point` rests on: the nearest primitive, and on a
   * box the face whose plane is nearest to the point (the largest excess over
   * the box's half extents, which on an edge or a corner is the face the point
   * is farthest outside of), or on a capsule its tangent plane at the nearest
   * point.
   */
  PatchFrame patchFrameAt(const Eigen::Vector3d& point) const;

 private:
  TerrainModel() = default;

  /** The smooth minimum of the distances, and the normalized softmin-weighted
   * normal, at `point`; nearestPoint is not set. */
  template <typename Scalar>
  PrimitiveQuery<Scalar> blend(const Vector3<Scalar>& point) const {
    using std::exp;
    using std::log;
    std::vector<PrimitiveQuery<Scalar>> queries;
    queries.reserve(primitives_.size());
    for (int index = 0; index < numPrimitives(); ++index) {
      queries.push_back(primitiveQuery(index, point));
    }
    Scalar minimum = queries.front().distance;
    for (const PrimitiveQuery<Scalar>& query : queries) {
      minimum = internal::minOf(minimum, query.distance);
    }
    const Scalar scale = Scalar(kSmoothMinimumScale);
    Scalar weightSum = Scalar(0.0);
    Vector3<Scalar> weightedNormal = Vector3<Scalar>::Zero();
    for (const PrimitiveQuery<Scalar>& query : queries) {
      // In (0, 1]: the nearest primitive's weight is 1.
      const Scalar weight = exp(Scalar(-(query.distance - minimum) / scale));
      weightSum += weight;
      weightedNormal += weight * query.normal;
    }
    PrimitiveQuery<Scalar> result;
    result.distance = minimum - scale * log(weightSum);
    const Scalar normalNorm = internal::guardedSqrt(
        Scalar(weightedNormal.squaredNorm()), kPrimitiveNormGuard);
    result.normal = weightedNormal / normalNorm;
    return result;
  }

  std::string name_;
  std::vector<TerrainPrimitive> primitives_;  // at least one
  absl::flat_hash_map<std::string, int> indexByName_;
};

}  // namespace ocs2::humanoid::multi_contact
