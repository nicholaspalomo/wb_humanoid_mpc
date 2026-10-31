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

#include <cmath>
#include <limits>
#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/manifold/ProductStateManifold.h"
#include "ocs2_core/manifold/StateManifoldSegment.h"

#include "humanoid_mpc_validation/closed_loop/ClosedLoopDriver.h"

namespace ocs2::humanoid::validation {
namespace {

// A centroidal-like state: the momentum (6) and the base position (3), the
// orientation at 9, then 5 joints.
constexpr size_t kOrientationStart = 9;
constexpr Eigen::Index kJoints = 5;

/** A step whose orientation part is `rotation` and whose other entries are
 * small and nonzero. */
vector_t stepWithRotation(const vector3_t& rotation, Eigen::Index stateDim) {
  vector_t delta = vector_t::LinSpaced(stateDim, -0.03, 0.04);
  delta.segment<3>(static_cast<Eigen::Index>(kOrientationStart)) = rotation;
  return delta;
}

// On a manifold, the step is a tangent vector: the rotation is the angle of its
// rotation block, which the manifold measures, wherever its own layout puts the
// block.
TEST(InitialStateGapTest, OnAManifoldTheRotationIsTheManifoldsRotationAngle) {
  const absl::StatusOr<std::shared_ptr<const ProductStateManifold>> manifold =
      ProductStateManifold::create(
          {StateManifoldSegment::euclidean(kOrientationStart),
           StateManifoldSegment::unitQuaternion(),
           StateManifoldSegment::euclidean(static_cast<size_t>(kJoints))});
  ASSERT_TRUE(manifold.ok()) << manifold.status();
  const vector3_t rotation(0.3, -0.1, 0.05);
  const vector_t delta = stepWithRotation(
      rotation, static_cast<Eigen::Index>((*manifold)->getTangentDim()));
  const InitialStateGap gap = initialStateGapOf(delta, **manifold);
  EXPECT_EQ(gap.rotation, (*manifold)->getMaximumTangentAngle(delta));
  EXPECT_NEAR(gap.rotation, rotation.norm(), 1.0e-15);
  EXPECT_EQ(gap.norm, delta.norm());

  // Without rotation blocks there is no rotation.
  const absl::StatusOr<std::shared_ptr<const ProductStateManifold>> flat =
      ProductStateManifold::create(
          {StateManifoldSegment::euclidean(static_cast<size_t>(delta.size()))});
  ASSERT_TRUE(flat.ok()) << flat.status();
  EXPECT_EQ(initialStateGapOf(delta, **flat).rotation, 0.0);
}

// The stored quaternions of the solver's nodes, unnormalized: the largest
// distance of one from the unit sphere, read at the orientation start and
// nowhere else.
TEST(QuaternionNormDeviationTest,
     IsTheLargestDistanceOfAStoredQuaternionFromTheUnitSphere) {
  const Eigen::Index start = static_cast<Eigen::Index>(kOrientationStart);
  // Every other entry far from a unit norm, and the quaternion (1/2, 1/2, 1/2,
  // 1/2), whose norm is 1 exactly.
  vector_t onTheSphere = vector_t::LinSpaced(start + 4 + kJoints, -2.0, 3.0);
  onTheSphere.segment<4>(start).setConstant(0.5);
  vector_t outside = onTheSphere;
  outside.segment<4>(start) *= 1.0 + 3.0e-10;
  vector_t inside = onTheSphere;
  inside.segment<4>(start) *= 1.0 - 5.0e-10;

  const std::optional<double> unit =
      maxQuaternionNormDeviationOf({onTheSphere}, kOrientationStart);
  ASSERT_TRUE(unit.has_value());
  EXPECT_EQ(unit.value_or(-1.0), 0.0);
  const std::optional<double> deviation = maxQuaternionNormDeviationOf(
      {onTheSphere, outside, inside, onTheSphere}, kOrientationStart);
  ASSERT_TRUE(deviation.has_value());
  EXPECT_NEAR(deviation.value_or(-1.0), 5.0e-10, 1.0e-15);
  const std::optional<double> outward =
      maxQuaternionNormDeviationOf({outside}, kOrientationStart);
  ASSERT_TRUE(outward.has_value());
  EXPECT_NEAR(outward.value_or(-1.0), 3.0e-10, 1.0e-15);
}

// A NaN is not dropped: a node with a NaN quaternion has no distance from the
// sphere, and the maximum over the nodes is NaN wherever the node is, so that
// the band max | |xi| - 1 | < 1e-9 cannot pass it.
TEST(QuaternionNormDeviationTest, ANanQuaternionMakesTheDeviationNan) {
  const Eigen::Index start = static_cast<Eigen::Index>(kOrientationStart);
  vector_t onTheSphere = vector_t::Zero(start + 4 + kJoints);
  onTheSphere(start + 3) = 1.0;
  vector_t nanQuaternion = onTheSphere;
  nanQuaternion(start) = std::numeric_limits<double>::quiet_NaN();
  for (const vector_array_t& states :
       {vector_array_t{onTheSphere, nanQuaternion},
        vector_array_t{nanQuaternion, onTheSphere},
        vector_array_t{nanQuaternion}}) {
    const std::optional<double> deviation =
        maxQuaternionNormDeviationOf(states, kOrientationStart);
    ASSERT_TRUE(deviation.has_value());
    EXPECT_TRUE(std::isnan(deviation.value_or(0.0)))
        << deviation.value_or(0.0) << " over " << states.size() << " nodes";
  }
}

// The same for the rotation of a gap with a NaN rotation block.
TEST(InitialStateGapTest, ANanRotationMakesTheRotationNan) {
  const absl::StatusOr<std::shared_ptr<const ProductStateManifold>> manifold =
      ProductStateManifold::create(
          {StateManifoldSegment::euclidean(kOrientationStart),
           StateManifoldSegment::unitQuaternion(),
           StateManifoldSegment::euclidean(static_cast<size_t>(kJoints))});
  ASSERT_TRUE(manifold.ok()) << manifold.status();
  const vector_t delta = stepWithRotation(
      vector3_t(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0),
      static_cast<Eigen::Index>((*manifold)->getTangentDim()));
  EXPECT_TRUE(std::isnan(initialStateGapOf(delta, **manifold).rotation));
}

TEST(QuaternionNormDeviationTest, IsEmptyWithoutAnyNode) {
  EXPECT_FALSE(maxQuaternionNormDeviationOf(/*states=*/{}, kOrientationStart)
                   .has_value());
}

}  // namespace
}  // namespace ocs2::humanoid::validation
