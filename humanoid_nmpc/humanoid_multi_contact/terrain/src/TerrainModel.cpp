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

#include "humanoid_multi_contact/terrain/TerrainModel.h"

#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Dimensions.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// How far from 1 the norm of a scene quaternion may be; it is normalized.
constexpr double kQuaternionNormTolerance = 1.0e-6;
// Below this squared length a capsule's ends coincide: it is a sphere.
constexpr double kDegenerateAxisSquaredLength = 1.0e-18;

using Primitive = config::TerrainScene::Primitive;

absl::Status primitiveError(absl::string_view scene,
                            absl::string_view name,
                            absl::string_view what) {
  return absl::InvalidArgumentError(absl::StrCat(
      "terrain scene '", scene, "': primitive '", name, "': ", what));
}

absl::StatusOr<Eigen::Vector3d> point3(const Eigen::VectorXd& values,
                                       absl::string_view scene,
                                       absl::string_view name,
                                       absl::string_view field) {
  if (values.size() != kPositionDim || !values.allFinite()) {
    return primitiveError(scene, name,
                          absl::StrCat(field, " must hold ", kPositionDim,
                                       " finite values, not ", values.size()));
  }
  return Eigen::Vector3d(values);
}

absl::StatusOr<BoxGeometry> boxGeometry(const config::TerrainScene::Box& box,
                                        absl::string_view scene,
                                        absl::string_view name) {
  BoxGeometry geometry;
  ASSIGN_OR_RETURN(geometry.center, point3(box.center, scene, name, "center"));
  ASSIGN_OR_RETURN(geometry.halfExtents,
                   point3(box.half_extents, scene, name, "half_extents"));
  if (!(geometry.halfExtents.minCoeff() > 0.0)) {
    return primitiveError(scene, name, "half_extents must be positive");
  }
  if (box.orientation_xyzw.size() != 0) {
    if (box.orientation_xyzw.size() != kQuaternionCoefficients ||
        !box.orientation_xyzw.allFinite() ||
        !(std::abs(box.orientation_xyzw.norm() - 1.0) <=
          kQuaternionNormTolerance)) {
      return primitiveError(
          scene, name,
          "orientation_xyzw must be empty or a unit quaternion (x, y, z, w)");
    }
    const Eigen::Vector4d coefficients(box.orientation_xyzw);
    geometry.rotation =
        Eigen::Quaterniond(coefficients).normalized().toRotationMatrix();
  }
  return geometry;
}

absl::StatusOr<CapsuleGeometry> capsuleGeometry(
    const config::TerrainScene::Capsule& capsule,
    absl::string_view scene,
    absl::string_view name) {
  CapsuleGeometry geometry;
  ASSIGN_OR_RETURN(geometry.start, point3(capsule.start, scene, name, "start"));
  ASSIGN_OR_RETURN(geometry.end, point3(capsule.end, scene, name, "end"));
  if (!(capsule.radius > 0.0) || !std::isfinite(capsule.radius)) {
    return primitiveError(scene, name, "radius must be positive");
  }
  geometry.radius = capsule.radius;
  const Eigen::Vector3d axis = geometry.end - geometry.start;
  const double squaredLength = axis.squaredNorm();
  if (squaredLength > kDegenerateAxisSquaredLength) {
    geometry.inverseSquaredLength = 1.0 / squaredLength;
    geometry.fallbackNormal = axis.unitOrthogonal();
  }
  return geometry;
}

absl::StatusOr<TerrainPrimitive> terrainPrimitive(const Primitive& primitive,
                                                  absl::string_view scene) {
  TerrainPrimitive result;
  result.name = primitive.name;
  if (primitive.name.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("terrain scene '", scene, "': a primitive has no name"));
  }
  if (!(primitive.friction > 0.0) || !std::isfinite(primitive.friction)) {
    return primitiveError(scene, primitive.name,
                          "friction must be positive and finite");
  }
  result.friction = primitive.friction;
  switch (primitive.shape.index()) {
    case Primitive::kBoxIndex: {
      ASSIGN_OR_RETURN(
          result.geometry,
          boxGeometry(std::get<Primitive::kBoxIndex>(primitive.shape), scene,
                      primitive.name));
      return result;
    }
    case Primitive::kCapsuleIndex: {
      ASSIGN_OR_RETURN(
          result.geometry,
          capsuleGeometry(std::get<Primitive::kCapsuleIndex>(primitive.shape),
                          scene, primitive.name));
      return result;
    }
    default:
      return primitiveError(scene, primitive.name,
                            "sets neither box nor capsule");
  }
}

/** The axis of the box frame a face's normal points along, and its sign. */
struct FaceAxis {
  int axis = 0;
  double sign = 1.0;
};

FaceAxis faceAxis(BoxFace face) {
  switch (face) {
    case BoxFace::kPositiveX:
      return FaceAxis{.axis = kXAxisIndex, .sign = 1.0};
    case BoxFace::kNegativeX:
      return FaceAxis{.axis = kXAxisIndex, .sign = -1.0};
    case BoxFace::kPositiveY:
      return FaceAxis{.axis = kYAxisIndex, .sign = 1.0};
    case BoxFace::kNegativeY:
      return FaceAxis{.axis = kYAxisIndex, .sign = -1.0};
    case BoxFace::kPositiveZ:
      return FaceAxis{.axis = kZAxisIndex, .sign = 1.0};
    case BoxFace::kNegativeZ:
      return FaceAxis{.axis = kZAxisIndex, .sign = -1.0};
  }
  return FaceAxis{};
}

BoxFace faceOf(int axis, double sign) {
  constexpr std::array<BoxFace, kPositionDim> kPositive = {
      BoxFace::kPositiveX, BoxFace::kPositiveY, BoxFace::kPositiveZ};
  constexpr std::array<BoxFace, kPositionDim> kNegative = {
      BoxFace::kNegativeX, BoxFace::kNegativeY, BoxFace::kNegativeZ};
  return sign > 0.0 ? kPositive[axis] : kNegative[axis];
}

PatchFrame faceFrame(const BoxGeometry& box, BoxFace face) {
  const FaceAxis normalAxis = faceAxis(face);
  const int firstTangentAxis = (normalAxis.axis + 1) % kPositionDim;
  const int secondTangentAxis =
      (normalAxis.axis + kPositionDim - 1) % kPositionDim;
  const Eigen::Vector3d normal =
      normalAxis.sign * box.rotation.col(normalAxis.axis);
  const Eigen::Vector3d firstTangent = box.rotation.col(firstTangentAxis);
  // n x t1 = sign * e_(k+2): the face's second axis, up to its sign.
  const Eigen::Vector3d secondTangent = normal.cross(firstTangent);
  PatchFrame frame;
  frame.face = face;
  frame.origin = box.center + normalAxis.sign *
                                  box.halfExtents(normalAxis.axis) *
                                  box.rotation.col(normalAxis.axis);
  frame.rotation.col(kXAxisIndex) = firstTangent;
  frame.rotation.col(kYAxisIndex) = secondTangent;
  frame.rotation.col(kZAxisIndex) = normal;
  const double halfX = box.halfExtents(firstTangentAxis);
  const double halfY = box.halfExtents(secondTangentAxis);
  frame.polygon = {
      Eigen::Vector2d(halfX, halfY), Eigen::Vector2d(-halfX, halfY),
      Eigen::Vector2d(-halfX, -halfY), Eigen::Vector2d(halfX, -halfY)};
  return frame;
}

PatchFrame tangentFrame(const CapsuleGeometry& capsule,
                        const PrimitiveQuery<double>& query) {
  const Eigen::Vector3d normal = query.normal;
  Eigen::Vector3d firstTangent = capsule.end - capsule.start;
  firstTangent -= firstTangent.dot(normal) * normal;
  if (firstTangent.squaredNorm() <= kDegenerateAxisSquaredLength) {
    firstTangent = normal.unitOrthogonal();
  }
  firstTangent.normalize();
  PatchFrame frame;
  frame.origin = query.nearestPoint;
  frame.rotation.col(kXAxisIndex) = firstTangent;
  frame.rotation.col(kYAxisIndex) = normal.cross(firstTangent);
  frame.rotation.col(kZAxisIndex) = normal;
  return frame;
}

}  // namespace

absl::StatusOr<TerrainModel> TerrainModel::Create(
    const config::TerrainScene& scene) {
  if (scene.primitives.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("terrain scene '", scene.name, "' lists no primitive"));
  }
  TerrainModel model;
  model.name_ = scene.name;
  for (const Primitive& primitive : scene.primitives) {
    ASSIGN_OR_RETURN(TerrainPrimitive validated,
                     terrainPrimitive(primitive, scene.name));
    if (!model.indexByName_.try_emplace(validated.name, model.numPrimitives())
             .second) {
      return primitiveError(scene.name, validated.name, "is listed twice");
    }
    model.primitives_.push_back(std::move(validated));
  }
  return model;
}

absl::StatusOr<int> TerrainModel::primitiveIndex(absl::string_view name) const {
  const absl::flat_hash_map<std::string, int>::const_iterator found =
      indexByName_.find(name);
  if (found == indexByName_.end()) {
    return absl::NotFoundError(absl::StrCat("terrain scene '", name_,
                                            "' has no primitive '", name, "'"));
  }
  return found->second;
}

NearestPrimitive TerrainModel::nearestPrimitive(
    const Eigen::Vector3d& point) const {
  NearestPrimitive nearest{.index = 0,
                           .query = primitiveQuery(/*index=*/0, point)};
  for (int index = 1; index < numPrimitives(); ++index) {
    const PrimitiveQuery<double> query = primitiveQuery(index, point);
    if (query.distance < nearest.query.distance) {
      nearest = NearestPrimitive{.index = index, .query = query};
    }
  }
  return nearest;
}

absl::StatusOr<PatchFrame> TerrainModel::boxFaceFrame(int index,
                                                      BoxFace face) const {
  if (index < 0 || index >= numPrimitives()) {
    return absl::InvalidArgumentError(
        absl::StrCat("terrain scene '", name_, "' has no primitive ", index));
  }
  const BoxGeometry* absl_nullable box =
      std::get_if<BoxGeometry>(&primitives_[index].geometry);
  if (box == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("terrain scene '", name_, "': primitive '",
                     primitives_[index].name, "' is not a box"));
  }
  PatchFrame frame = faceFrame(*box, face);
  frame.primitiveIndex = index;
  frame.friction = primitives_[index].friction;
  return frame;
}

PatchFrame TerrainModel::patchFrameAt(const Eigen::Vector3d& point) const {
  const NearestPrimitive nearest = nearestPrimitive(point);
  const TerrainPrimitive& primitive = primitives_[nearest.index];
  PatchFrame frame;
  if (const BoxGeometry* absl_nullable box =
          std::get_if<BoxGeometry>(&primitive.geometry)) {
    const Eigen::Vector3d local =
        box->rotation.transpose() * (point - box->center);
    const Eigen::Vector3d excess = local.cwiseAbs() - box->halfExtents;
    Eigen::Index axis = 0;
    excess.maxCoeff(&axis);
    const double sign = local(axis) > 0.0 ? 1.0 : -1.0;
    frame = faceFrame(*box, faceOf(static_cast<int>(axis), sign));
  } else {
    frame = tangentFrame(std::get<CapsuleGeometry>(primitive.geometry),
                         nearest.query);
  }
  frame.primitiveIndex = nearest.index;
  frame.friction = primitive.friction;
  return frame;
}

}  // namespace ocs2::humanoid::multi_contact
