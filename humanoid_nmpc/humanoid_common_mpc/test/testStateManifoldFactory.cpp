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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"
#include "ocs2_core/manifold/ProductStateManifold.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/manifold/StateManifoldSegment.h"

#include "humanoid_common_mpc/common/StateManifoldFactory.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"

/*
 * stateManifoldOf() on the tuning layouts of both MPCs: the product of the
 * blocks with a unit quaternion for the base orientation, which every robot
 * model stores as one, and the refusal of a layout that does not describe the
 * state - an orientation on other rows than the quaternion's 4 among them.
 */

namespace ocs2::humanoid {
namespace {

/** Five joints and two contacts: a centroidal tuning layout of 17 rows, a
 * whole-body one of 22. */
StateInputLayout layoutOf(StateInputLayout::Mpc mpc) {
  return StateInputLayout{.mpc = mpc,
                          .jointNames = {"a", "b", "c", "d", "e"},
                          .fixedJointNames = {},
                          .contactNames = {"left", "right"}};
}

/**
 * A state of `layout` (placed with a quaternion) with distinct entries from
 * `first` on and the unit quaternion `xi` in its orientation block.
 */
vector_t stateWithAttitude(const TuningLayout& layout,
                           scalar_t first,
                           const vector4_t& xi) {
  vector_t state = vector_t::LinSpaced(
      static_cast<Eigen::Index>(getTuningLayoutStateDim(layout)), first,
      first + 2.2);
  for (const TuningSegment& segment : layout.segments) {
    if (segment.kind == TuningSegmentKind::kEulerZyxOrientation)
      state.segment<4>(static_cast<Eigen::Index>(segment.stateStart)) = xi;
  }
  return state;
}

TEST(StateManifoldFactory, AQuaternionLayoutIsTheProductOfItsBlocks) {
  // Centroidal [h(6), p(3), xi(4), q_j(5)] and whole body [p(3), xi(4), q_j(5),
  // pd(3), w_B(3), qd_j(5)].
  const std::vector<
      std::pair<StateInputLayout::Mpc, std::vector<StateManifoldSegment>>>
      expectations = {
          {StateInputLayout::Mpc::kCentroidal,
           {StateManifoldSegment::euclidean(9),
            StateManifoldSegment::unitQuaternion(),
            StateManifoldSegment::euclidean(5)}},
          {StateInputLayout::Mpc::kWholeBody,
           {StateManifoldSegment::euclidean(3),
            StateManifoldSegment::unitQuaternion(),
            StateManifoldSegment::euclidean(16)}},
      };
  for (const std::pair<StateInputLayout::Mpc,
                       std::vector<StateManifoldSegment>>& expected :
       expectations) {
    const TuningLayout layout =
        placeTuningLayout(tuningLayoutOf(layoutOf(expected.first)));
    const size_t stateDim = getTuningLayoutStateDim(layout);
    ASSERT_EQ(stateDim, getTuningLayoutDim(layout) + 1);
    const absl::StatusOr<std::shared_ptr<const StateManifold>> manifold =
        stateManifoldOf(layout, stateDim);
    ASSERT_TRUE(manifold.ok()) << manifold.status();
    ASSERT_NE(*manifold, nullptr);
    EXPECT_EQ((*manifold)->getAmbientDim(), stateDim);
    EXPECT_EQ((*manifold)->getTangentDim(), getTuningLayoutDim(layout));

    // It is the manifold of the expected blocks: the same difference and
    // retraction, bit for bit.
    const absl::StatusOr<std::shared_ptr<const ProductStateManifold>>
        reference = ProductStateManifold::create(expected.second);
    ASSERT_TRUE(reference.ok()) << reference.status();
    const vector_t x0 = stateWithAttitude(
        layout, /*first=*/0.1, vector4_t(0.1, -0.2, 0.3, 0.9).normalized());
    const vector_t x1 = stateWithAttitude(
        layout, /*first=*/0.35, vector4_t(-0.3, 0.1, 0.2, 0.8).normalized());
    const vector_t delta = (*manifold)->difference(x0, x1);
    EXPECT_EQ(delta, (*reference)->difference(x0, x1));
    vector_t retracted;
    vector_t referenceRetracted;
    (*manifold)->retract(x0, delta, /*alpha=*/0.5, retracted);
    (*reference)->retract(x0, delta, /*alpha=*/0.5, referenceRetracted);
    EXPECT_EQ(retracted, referenceRetracted);
  }
}

TEST(StateManifoldFactory, ALayoutThatDoesNotDescribeTheStateIsRefused) {
  const TuningLayout placed = placeTuningLayout(
      tuningLayoutOf(layoutOf(StateInputLayout::Mpc::kCentroidal)));

  // A state of another size.
  const absl::StatusOr<std::shared_ptr<const StateManifold>> wrongSize =
      stateManifoldOf(placed, getTuningLayoutStateDim(placed) - 1);
  ASSERT_FALSE(wrongSize.ok());
  EXPECT_EQ(wrongSize.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(wrongSize.status().message(),
                                "centroidal tuning layout"))
      << wrongSize.status();

  // A layout that placeTuningLayout() has not placed.
  const absl::StatusOr<std::shared_ptr<const StateManifold>> unplaced =
      stateManifoldOf(
          tuningLayoutOf(layoutOf(StateInputLayout::Mpc::kCentroidal)),
          getTuningLayoutStateDim(placed));
  ASSERT_FALSE(unplaced.ok());
  EXPECT_EQ(unplaced.status().code(), absl::StatusCode::kInvalidArgument);

  // An orientation placed by hand on another number of state rows than the
  // quaternion's 4, the blocks after it moved to follow on: the 3 of its Euler
  // angles and 5.
  const std::vector<size_t> malformedOrientationSlotCounts = {3, 5};
  for (const size_t orientationSlotCount : malformedOrientationSlotCounts) {
    TuningLayout malformed = placed;
    size_t stateStart = 0;
    for (TuningSegment& segment : malformed.segments) {
      segment.stateStart = stateStart;
      if (segment.kind == TuningSegmentKind::kEulerZyxOrientation)
        segment.stateDim = orientationSlotCount;
      stateStart += segment.stateDim;
    }
    const absl::StatusOr<std::shared_ptr<const StateManifold>>
        malformedManifold =
            stateManifoldOf(malformed, getTuningLayoutStateDim(malformed));
    ASSERT_FALSE(malformedManifold.ok())
        << orientationSlotCount << " orientation rows got a manifold";
    EXPECT_EQ(malformedManifold.status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(absl::StrContains(malformedManifold.status().message(),
                                  "'eulerzyx_base_rt_world'"))
        << malformedManifold.status();
    EXPECT_TRUE(absl::StrContains(malformedManifold.status().message(),
                                  "as a quaternion, on 4 rows"))
        << malformedManifold.status();
  }
}

}  // namespace
}  // namespace ocs2::humanoid
