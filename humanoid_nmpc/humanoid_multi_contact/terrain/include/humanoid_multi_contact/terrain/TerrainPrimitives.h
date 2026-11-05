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

#include "Eigen/Core"
#include "ocs2_core/Dimensions.h"

/*
 * The exact signed distance, nearest surface point and outward normal of the
 * two convex primitives of a TerrainScene, a box and a capsule
 * (humanoid_nmpc/docs/multi_contact_planning/README.md, section 3.2).
 *
 * Every query is a template on the scalar, so the same code runs in double and
 * on a CppAD tape. On a tape no branch may be a plain `if`, which would record
 * only the branch taken at the taping point: every branch goes through
 * internal::ifGreaterThan(), a CppAD conditional expression for AD scalars, and
 * both branches stay finite at every input. A norm is never the square root of
 * a quantity that can be 0 in a branch selected away: the squared quantity is
 * guarded before the square root, because the reverse sweep of sqrt at 0 is 0 *
 * inf. The geometry itself is double data, which a tape records as constants.
 *
 * For an AD scalar, include "cppad/cppad.hpp" (or "cppad/cg.hpp") and Eigen's
 * NumTraits of the scalar before instantiating these templates.
 */

namespace ocs2::humanoid::multi_contact {

template <typename Scalar>
using Vector3 = Eigen::Matrix<Scalar, kPositionDim, 1>;

template <typename Scalar>
using Matrix3 = Eigen::Matrix<Scalar, kPositionDim, kPositionDim>;

/**
 * Below this distance [m] from a box's surface (outside) or from a capsule's
 * axis, a norm is guarded: the box query reads the point as on or inside the
 * box, and the capsule query takes the capsule's fallback normal. Far below any
 * contact tolerance, and its square (1e-18) is a normal double.
 */
inline constexpr double kPrimitiveNormGuard = 1.0e-9;

/** The box of a TerrainScene, validated: a rotation matrix and positive half
 * extents. */
struct BoxGeometry {
  Eigen::Vector3d posBoxRtWorldEwrtWorld = Eigen::Vector3d::Zero();
  // world_R_box: the columns are the box's axes in the world frame.
  Eigen::Matrix3d rotmatBoxRtWorld = Eigen::Matrix3d::Identity();
  Eigen::Vector3d halfExtents = Eigen::Vector3d::Zero();
};

/** The capsule of a TerrainScene, validated, with what its queries precompute.
 */
struct CapsuleGeometry {
  // [m] The ends of the axis.
  Eigen::Vector3d posCapsulestartRtWorldEwrtWorld = Eigen::Vector3d::Zero();
  Eigen::Vector3d posCapsuleendRtWorldEwrtWorld = Eigen::Vector3d::Zero();
  double radius = 0.0;
  // 1 / |end - start|^2, or 0 when the ends coincide (a sphere at the start).
  double inverseSquaredLength = 0.0;
  // A unit vector orthogonal to the axis: the normal on the axis itself, where
  // the outward direction is undefined.
  Eigen::Vector3d fallbackNormalSurfaceEwrtWorld = Eigen::Vector3d::UnitZ();
};

/** What a query of one primitive returns. Passive data. */
template <typename Scalar>
struct PrimitiveQuery {
  // [m] Negative inside the primitive.
  Scalar distance = Scalar(0.0);
  // The point of the surface nearest to the query point.
  Vector3<Scalar> nearestPosSurfaceRtWorldEwrtWorld = Vector3<Scalar>::Zero();
  // The unit outward normal of the surface at
  // nearestPosSurfaceRtWorldEwrtWorld: the gradient of the signed distance at
  // the query point.
  Vector3<Scalar> normalSurfaceEwrtWorld = Vector3<Scalar>::UnitZ();
};

namespace internal {

/** left > right ? ifTrue : ifFalse, as a plain conditional for double. */
inline double ifGreaterThan(double left,
                            double right,
                            double ifTrue,
                            double ifFalse) {
  return left > right ? ifTrue : ifFalse;
}

/** left > right ? ifTrue : ifFalse, as a CppAD conditional expression for AD
 * scalars (found by argument-dependent lookup). */
template <typename Scalar>
Scalar ifGreaterThan(const Scalar& left,
                     const Scalar& right,
                     const Scalar& ifTrue,
                     const Scalar& ifFalse) {
  return CondExpGt(left, right, ifTrue, ifFalse);
}

template <typename Scalar>
Scalar maxOf(const Scalar& left, const Scalar& right) {
  return ifGreaterThan(left, right, left, right);
}

template <typename Scalar>
Scalar minOf(const Scalar& left, const Scalar& right) {
  return ifGreaterThan(left, right, right, left);
}

template <typename Scalar>
Scalar absOf(const Scalar& value) {
  return ifGreaterThan(value, Scalar(0.0), value, Scalar(-value));
}

template <typename Scalar>
Scalar clampOf(const Scalar& value, const Scalar& lower, const Scalar& upper) {
  return minOf(maxOf(value, lower), upper);
}

/** sqrt(max(squared, guard^2)): finite, with finite derivatives, everywhere. */
template <typename Scalar>
Scalar guardedSqrt(const Scalar& squared, double guard) {
  using std::sqrt;
  const Scalar guardSquared = Scalar(guard * guard);
  return sqrt(ifGreaterThan(squared, guardSquared, squared, guardSquared));
}

/** Each component of `ifTrue` where left > right, else of `ifFalse`. */
template <typename Scalar>
Vector3<Scalar> selectVector(const Scalar& left,
                             const Scalar& right,
                             const Vector3<Scalar>& ifTrue,
                             const Vector3<Scalar>& ifFalse) {
  Vector3<Scalar> result;
  for (int axis = 0; axis < kPositionDim; ++axis) {
    result(axis) = ifGreaterThan(left, right, ifTrue(axis), ifFalse(axis));
  }
  return result;
}

}  // namespace internal

/**
 * Queries a box at `posQueryRtWorldEwrtWorld` (world frame), exactly: outside,
 * the distance to the nearest face, edge or corner; inside, minus the distance
 * to the nearest face. On a tie between faces inside the box, the first axis
 * wins (x, then y, then z), and a coordinate exactly at the center plane counts
 * as negative.
 */
template <typename Scalar>
PrimitiveQuery<Scalar> boxQuery(
    const BoxGeometry& box, const Vector3<Scalar>& posQueryRtWorldEwrtWorld) {
  using internal::ifGreaterThan;
  const Matrix3<Scalar> rotmatBoxRtWorld = box.rotmatBoxRtWorld.cast<Scalar>();
  const Vector3<Scalar> posQueryRtBoxEwrtBox =
      rotmatBoxRtWorld.transpose() *
      (posQueryRtWorldEwrtWorld - box.posBoxRtWorldEwrtWorld.cast<Scalar>());
  Vector3<Scalar> excess;  // |x_i| - h_i, positive outside slab i
  Vector3<Scalar> outsideNearestPosSurfaceRtBoxEwrtBox;  // the nearest point of
                                                         // the box when outside
  Vector3<Scalar> signOf;  // +-1, the side of each center plane
  for (int axis = 0; axis < kPositionDim; ++axis) {
    const Scalar halfExtent = Scalar(box.halfExtents(axis));
    excess(axis) = internal::absOf(posQueryRtBoxEwrtBox(axis)) - halfExtent;
    outsideNearestPosSurfaceRtBoxEwrtBox(axis) = internal::clampOf(
        posQueryRtBoxEwrtBox(axis), Scalar(-halfExtent), halfExtent);
    signOf(axis) = ifGreaterThan(posQueryRtBoxEwrtBox(axis), Scalar(0.0),
                                 Scalar(1.0), Scalar(-1.0));
  }
  const Vector3<Scalar> outsidePosQueryRtSurfaceEwrtBox =
      posQueryRtBoxEwrtBox -
      outsideNearestPosSurfaceRtBoxEwrtBox;  // zero inside
  const Scalar outsideSquared = outsidePosQueryRtSurfaceEwrtBox.squaredNorm();
  const Scalar outsideNorm =
      internal::guardedSqrt(outsideSquared, kPrimitiveNormGuard);
  const Scalar guardSquared = Scalar(kPrimitiveNormGuard * kPrimitiveNormGuard);

  // Inside: the face of the largest excess, as indicators that sum to one.
  const Scalar xBeatsY = ifGreaterThan(excess(kYAxisIndex), excess(kXAxisIndex),
                                       Scalar(0.0), Scalar(1.0));
  const Scalar xBeatsZ = ifGreaterThan(excess(kZAxisIndex), excess(kXAxisIndex),
                                       Scalar(0.0), Scalar(1.0));
  const Scalar yBeatsZ = ifGreaterThan(excess(kZAxisIndex), excess(kYAxisIndex),
                                       Scalar(0.0), Scalar(1.0));
  Vector3<Scalar> chosen;
  chosen(kXAxisIndex) = xBeatsY * xBeatsZ;
  chosen(kYAxisIndex) = (Scalar(1.0) - chosen(kXAxisIndex)) * yBeatsZ;
  chosen(kZAxisIndex) = Scalar(1.0) - chosen(kXAxisIndex) - chosen(kYAxisIndex);
  Vector3<Scalar> insideNormalSurfaceEwrtBox;
  Vector3<Scalar> insideNearestPosSurfaceRtBoxEwrtBox;
  for (int axis = 0; axis < kPositionDim; ++axis) {
    insideNormalSurfaceEwrtBox(axis) = chosen(axis) * signOf(axis);
    insideNearestPosSurfaceRtBoxEwrtBox(axis) =
        posQueryRtBoxEwrtBox(axis) +
        chosen(axis) * (signOf(axis) * Scalar(box.halfExtents(axis)) -
                        posQueryRtBoxEwrtBox(axis));
  }
  const Scalar largestExcess =
      internal::maxOf(internal::maxOf(excess(kXAxisIndex), excess(kYAxisIndex)),
                      excess(kZAxisIndex));

  PrimitiveQuery<Scalar> query;
  query.distance = ifGreaterThan(outsideSquared, guardSquared, outsideNorm,
                                 internal::minOf(largestExcess, Scalar(0.0)));
  const Vector3<Scalar> outsideNormalSurfaceEwrtBox =
      outsidePosQueryRtSurfaceEwrtBox / outsideNorm;
  query.normalSurfaceEwrtWorld =
      rotmatBoxRtWorld * internal::selectVector(outsideSquared, guardSquared,
                                                outsideNormalSurfaceEwrtBox,
                                                insideNormalSurfaceEwrtBox);
  query.nearestPosSurfaceRtWorldEwrtWorld =
      box.posBoxRtWorldEwrtWorld.cast<Scalar>() +
      rotmatBoxRtWorld *
          internal::selectVector(outsideSquared, guardSquared,
                                 outsideNearestPosSurfaceRtBoxEwrtBox,
                                 insideNearestPosSurfaceRtBoxEwrtBox);
  return query;
}

/**
 * Queries a capsule at `posQueryRtWorldEwrtWorld` (world frame), exactly: the
 * distance to its axis segment minus its radius. On the axis itself, the normal
 * is the capsule's fallbackNormalSurfaceEwrtWorld.
 */
template <typename Scalar>
PrimitiveQuery<Scalar> capsuleQuery(
    const CapsuleGeometry& capsule,
    const Vector3<Scalar>& posQueryRtWorldEwrtWorld) {
  const Vector3<Scalar> posCapsulestartRtWorldEwrtWorld =
      capsule.posCapsulestartRtWorldEwrtWorld.cast<Scalar>();
  const Vector3<Scalar> posCapsuleendRtCapsulestartEwrtWorld =
      (capsule.posCapsuleendRtWorldEwrtWorld -
       capsule.posCapsulestartRtWorldEwrtWorld)
          .cast<Scalar>();
  const Scalar fraction = internal::clampOf(
      Scalar((posQueryRtWorldEwrtWorld - posCapsulestartRtWorldEwrtWorld)
                 .dot(posCapsuleendRtCapsulestartEwrtWorld) *
             Scalar(capsule.inverseSquaredLength)),
      Scalar(0.0), Scalar(1.0));
  const Vector3<Scalar> posSegmentpointRtWorldEwrtWorld =
      posCapsulestartRtWorldEwrtWorld +
      fraction * posCapsuleendRtCapsulestartEwrtWorld;
  const Vector3<Scalar> posQueryRtSegmentpointEwrtWorld =
      posQueryRtWorldEwrtWorld - posSegmentpointRtWorldEwrtWorld;
  const Scalar squared = posQueryRtSegmentpointEwrtWorld.squaredNorm();
  const Scalar norm = internal::guardedSqrt(squared, kPrimitiveNormGuard);
  const Scalar guardSquared = Scalar(kPrimitiveNormGuard * kPrimitiveNormGuard);

  PrimitiveQuery<Scalar> query;
  query.distance = norm - Scalar(capsule.radius);
  query.normalSurfaceEwrtWorld = internal::selectVector(
      squared, guardSquared,
      Vector3<Scalar>(posQueryRtSegmentpointEwrtWorld / norm),
      Vector3<Scalar>(capsule.fallbackNormalSurfaceEwrtWorld.cast<Scalar>()));
  query.nearestPosSurfaceRtWorldEwrtWorld =
      posSegmentpointRtWorldEwrtWorld +
      Scalar(capsule.radius) * query.normalSurfaceEwrtWorld;
  return query;
}

}  // namespace ocs2::humanoid::multi_contact
