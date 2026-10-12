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

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

#include "humanoid_common_mpc/command/TargetTrajectoriesCalculatorBase.h"
#include "humanoid_common_mpc/common/Types.h"

/*
 * The subdivision of a pose command's two knots (TargetTrajectoriesCalculatorBase::subdivideYawKnots(), design section
 * 2.9): the fewest equal parts in time for which consecutive knots are at most a quarter turn apart in yaw, every
 * coordinate interpolated linearly, the end knots exactly the commanded poses, a turn past kMaxSegmentYawChange clamped
 * and a turn that is not finite left in one part. Builds no model.
 */

namespace ocs2::humanoid {
namespace {

using Calculator = TargetTrajectoriesCalculatorBase;

/** The level base pose [x, y, z, yaw, 0, 0]. */
vector6_t pose(scalar_t x, scalar_t y, scalar_t z, scalar_t yaw) {
  vector6_t basePose;
  basePose << x, y, z, yaw, 0.0, 0.0;
  return basePose;
}

TEST(SubdivideYawKnots, AQuarterTurnOrLessKeepsTheTwoKnots) {
  for (const scalar_t yawChange : {0.0, 0.3, -0.5 * M_PI, 0.5 * M_PI}) {
    SCOPED_TRACE(yawChange);
    const vector6_t start = pose(0.1, -0.2, 0.9, 3.0);
    const vector6_t target = pose(/*x=*/0.5, /*y=*/0.4, /*z=*/0.95, 3.0 + yawChange);
    const std::vector<Calculator::PoseCommandKnot> knots =
        Calculator::subdivideYawKnots(/*startTime=*/1.0, start, /*targetTime=*/3.0, target);
    ASSERT_EQ(knots.size(), 2u);
    EXPECT_EQ(knots.front().time, 1.0);
    EXPECT_EQ(knots.back().time, 3.0);
    EXPECT_EQ(knots.front().basePose, start);
    EXPECT_EQ(knots.back().basePose, target);
  }
}

TEST(SubdivideYawKnots, ALargerTurnIsCutIntoQuarterTurnsOnTheLineBetweenThePoses) {
  for (const scalar_t yawChange : {0.5 * M_PI + 1.0e-9, M_PI, -1.5 * M_PI, 3.5, -4.0 * M_PI, 20.0}) {
    SCOPED_TRACE(yawChange);
    const vector6_t start = pose(1.0, 2.0, 0.8, -2.5);
    const vector6_t target = pose(/*x=*/-1.0, /*y=*/0.5, /*z=*/0.85, -2.5 + yawChange);
    const scalar_t startTime = 0.25;
    const scalar_t targetTime = 4.25;
    const std::vector<Calculator::PoseCommandKnot> knots = Calculator::subdivideYawKnots(startTime, start, targetTime, target);
    const size_t parts = static_cast<size_t>(std::ceil(std::abs(yawChange) / Calculator::kMaxKnotYawSpacing));
    ASSERT_EQ(knots.size(), parts + 1);
    EXPECT_EQ(knots.front().basePose, start);
    EXPECT_EQ(knots.back().basePose, target);
    EXPECT_EQ(knots.back().time, targetTime);
    for (size_t k = 0; k < knots.size(); ++k) {
      const scalar_t fraction = static_cast<scalar_t>(k) / static_cast<scalar_t>(parts);
      EXPECT_NEAR(knots[k].time, startTime + fraction * (targetTime - startTime), 1.0e-12);
      EXPECT_LT((knots[k].basePose - (start + fraction * (target - start))).norm(), 1.0e-12);
      if (k > 0) {
        EXPECT_GT(knots[k].time, knots[k - 1].time);
        EXPECT_LE(std::abs(knots[k].basePose(3) - knots[k - 1].basePose(3)), Calculator::kMaxKnotYawSpacing + 1.0e-12);
      }
    }
  }
}

TEST(SubdivideYawKnots, ATurnBeyondTheLargestSegmentTurnIsClampedToIt) {
  const vector6_t start = pose(0.0, 0.0, 0.9, 0.0);
  for (const scalar_t sign : {1.0, -1.0}) {
    SCOPED_TRACE(sign);
    const std::vector<Calculator::PoseCommandKnot> knots =
        Calculator::subdivideYawKnots(/*startTime=*/0.0, start, /*targetTime=*/10.0, pose(/*x=*/1.0, /*y=*/0.0, /*z=*/0.9, sign * 1.0e3));
    ASSERT_EQ(knots.size(), Calculator::kMaxKnotSubdivisions + 1);
    EXPECT_NEAR(knots.back().basePose(3), sign * Calculator::kMaxSegmentYawChange, 1.0e-12);
    EXPECT_EQ(knots.back().basePose(0), 1.0);
    for (size_t k = 1; k < knots.size(); ++k) {
      EXPECT_LE(std::abs(knots[k].basePose(3) - knots[k - 1].basePose(3)), Calculator::kMaxKnotYawSpacing + 1.0e-12);
    }
  }
}

TEST(SubdivideYawKnots, ATurnThatIsNotFiniteStaysInOnePart) {
  const vector6_t start = pose(0.0, 0.0, 0.9, 0.0);
  const vector6_t target = pose(/*x=*/1.0, /*y=*/0.0, /*z=*/0.9, std::numeric_limits<scalar_t>::quiet_NaN());
  const std::vector<Calculator::PoseCommandKnot> knots =
      Calculator::subdivideYawKnots(/*startTime=*/0.0, start, /*targetTime=*/1.0, target);
  ASSERT_EQ(knots.size(), 2u);
  EXPECT_TRUE(std::isnan(knots.back().basePose(3)));
}

}  // namespace
}  // namespace ocs2::humanoid
