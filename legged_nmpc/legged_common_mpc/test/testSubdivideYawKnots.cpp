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
#include <numbers>
#include <vector>

#include "gtest/gtest.h"
#include "ocs2_core/Types.h"

#include "legged_common_mpc/command/TargetTrajectoriesCalculatorBase.h"
#include "legged_common_mpc/common/Types.h"

/*
 * The subdivision of a pose command's two knots
 * (TargetTrajectoriesCalculatorBase::subdivideYawKnots(), design section 2.9):
 * the fewest equal parts in time for which consecutive knots are at most a
 * quarter turn apart in yaw, every coordinate interpolated linearly, the end
 * knots exactly the commanded poses, a turn past kSegmentMaxErrYawBaseRtWorld
 * clamped and a turn that is not finite left in one part. Builds no model.
 */

namespace ocs2::legged {
namespace {

using Calculator = TargetTrajectoriesCalculatorBase;

/** The level base pose [x, y, z, yaw, 0, 0]. */
vector6_t makePoseeulerzyxBaseRtWorld(scalar_t posBaseRtWorldEwrtWorldX,
                                      scalar_t posBaseRtWorldEwrtWorldY,
                                      scalar_t posBaseRtWorldEwrtWorldZ,
                                      scalar_t yawBaseRtWorld) {
  vector6_t poseeulerzyxBaseRtWorld;
  poseeulerzyxBaseRtWorld << posBaseRtWorldEwrtWorldX, posBaseRtWorldEwrtWorldY,
      posBaseRtWorldEwrtWorldZ, yawBaseRtWorld, 0.0, 0.0;
  return poseeulerzyxBaseRtWorld;
}

TEST(SubdivideYawKnots, AQuarterTurnOrLessKeepsTheTwoKnots) {
  for (const scalar_t errYawBaseRtWorld :
       {0.0, 0.3, -0.5 * std::numbers::pi, 0.5 * std::numbers::pi}) {
    SCOPED_TRACE(errYawBaseRtWorld);
    const vector6_t initPoseeulerzyxBaseRtWorld =
        makePoseeulerzyxBaseRtWorld(0.1, -0.2, 0.9, 3.0);
    const vector6_t finalPoseeulerzyxBaseRtWorld = makePoseeulerzyxBaseRtWorld(
        /*posBaseRtWorldEwrtWorldX=*/0.5, /*posBaseRtWorldEwrtWorldY=*/0.4,
        /*posBaseRtWorldEwrtWorldZ=*/0.95, 3.0 + errYawBaseRtWorld);
    const std::vector<Calculator::PoseCommandKnot> knots =
        Calculator::subdivideYawKnots(
            /*startTime=*/1.0, initPoseeulerzyxBaseRtWorld,
            /*targetTime=*/3.0, finalPoseeulerzyxBaseRtWorld);
    ASSERT_EQ(knots.size(), 2u);
    EXPECT_EQ(knots.front().time, 1.0);
    EXPECT_EQ(knots.back().time, 3.0);
    EXPECT_EQ(knots.front().poseeulerzyxBaseRtWorld,
              initPoseeulerzyxBaseRtWorld);
    EXPECT_EQ(knots.back().poseeulerzyxBaseRtWorld,
              finalPoseeulerzyxBaseRtWorld);
  }
}

TEST(SubdivideYawKnots,
     ALargerTurnIsCutIntoQuarterTurnsOnTheLineBetweenThePoses) {
  for (const scalar_t errYawBaseRtWorld :
       {0.5 * std::numbers::pi + 1.0e-9, std::numbers::pi,
        -1.5 * std::numbers::pi, 3.5, -4.0 * std::numbers::pi, 20.0}) {
    SCOPED_TRACE(errYawBaseRtWorld);
    const vector6_t initPoseeulerzyxBaseRtWorld =
        makePoseeulerzyxBaseRtWorld(1.0, 2.0, 0.8, -2.5);
    const vector6_t finalPoseeulerzyxBaseRtWorld = makePoseeulerzyxBaseRtWorld(
        /*posBaseRtWorldEwrtWorldX=*/-1.0, /*posBaseRtWorldEwrtWorldY=*/0.5,
        /*posBaseRtWorldEwrtWorldZ=*/0.85, -2.5 + errYawBaseRtWorld);
    const scalar_t startTime = 0.25;
    const scalar_t targetTime = 4.25;
    const std::vector<Calculator::PoseCommandKnot> knots =
        Calculator::subdivideYawKnots(startTime, initPoseeulerzyxBaseRtWorld,
                                      targetTime, finalPoseeulerzyxBaseRtWorld);
    const size_t parts = static_cast<size_t>(std::ceil(
        std::abs(errYawBaseRtWorld) / Calculator::kKnotMaxErrYawBaseRtWorld));
    ASSERT_EQ(knots.size(), parts + 1);
    EXPECT_EQ(knots.front().poseeulerzyxBaseRtWorld,
              initPoseeulerzyxBaseRtWorld);
    EXPECT_EQ(knots.back().poseeulerzyxBaseRtWorld,
              finalPoseeulerzyxBaseRtWorld);
    EXPECT_EQ(knots.back().time, targetTime);
    for (size_t k = 0; k < knots.size(); ++k) {
      const scalar_t fraction =
          static_cast<scalar_t>(k) / static_cast<scalar_t>(parts);
      EXPECT_NEAR(knots[k].time,
                  startTime + fraction * (targetTime - startTime), 1.0e-12);
      EXPECT_LT((knots[k].poseeulerzyxBaseRtWorld -
                 (initPoseeulerzyxBaseRtWorld +
                  fraction * (finalPoseeulerzyxBaseRtWorld -
                              initPoseeulerzyxBaseRtWorld)))
                    .norm(),
                1.0e-12);
      if (k > 0) {
        EXPECT_GT(knots[k].time, knots[k - 1].time);
        EXPECT_LE(std::abs(knots[k].poseeulerzyxBaseRtWorld(3) -
                           knots[k - 1].poseeulerzyxBaseRtWorld(3)),
                  Calculator::kKnotMaxErrYawBaseRtWorld + 1.0e-12);
      }
    }
  }
}

TEST(SubdivideYawKnots, ATurnBeyondTheLargestSegmentTurnIsClampedToIt) {
  const vector6_t initPoseeulerzyxBaseRtWorld =
      makePoseeulerzyxBaseRtWorld(0.0, 0.0, 0.9, 0.0);
  for (const scalar_t sign : {1.0, -1.0}) {
    SCOPED_TRACE(sign);
    const std::vector<Calculator::PoseCommandKnot> knots =
        Calculator::subdivideYawKnots(
            /*startTime=*/0.0, initPoseeulerzyxBaseRtWorld, /*targetTime=*/10.0,
            makePoseeulerzyxBaseRtWorld(/*posBaseRtWorldEwrtWorldX=*/1.0,
                                        /*posBaseRtWorldEwrtWorldY=*/0.0,
                                        /*posBaseRtWorldEwrtWorldZ=*/0.9,
                                        sign * 1.0e3));
    ASSERT_EQ(knots.size(), Calculator::kMaxKnotSubdivisions + 1);
    EXPECT_NEAR(knots.back().poseeulerzyxBaseRtWorld(3),
                sign * Calculator::kSegmentMaxErrYawBaseRtWorld, 1.0e-12);
    EXPECT_EQ(knots.back().poseeulerzyxBaseRtWorld(0), 1.0);
    for (size_t k = 1; k < knots.size(); ++k) {
      EXPECT_LE(std::abs(knots[k].poseeulerzyxBaseRtWorld(3) -
                         knots[k - 1].poseeulerzyxBaseRtWorld(3)),
                Calculator::kKnotMaxErrYawBaseRtWorld + 1.0e-12);
    }
  }
}

TEST(SubdivideYawKnots, ATurnThatIsNotFiniteStaysInOnePart) {
  const vector6_t initPoseeulerzyxBaseRtWorld =
      makePoseeulerzyxBaseRtWorld(0.0, 0.0, 0.9, 0.0);
  const vector6_t finalPoseeulerzyxBaseRtWorld = makePoseeulerzyxBaseRtWorld(
      /*posBaseRtWorldEwrtWorldX=*/1.0, /*posBaseRtWorldEwrtWorldY=*/0.0,
      /*posBaseRtWorldEwrtWorldZ=*/0.9,
      std::numeric_limits<scalar_t>::quiet_NaN());
  const std::vector<Calculator::PoseCommandKnot> knots =
      Calculator::subdivideYawKnots(
          /*startTime=*/0.0, initPoseeulerzyxBaseRtWorld,
          /*targetTime=*/1.0, finalPoseeulerzyxBaseRtWorld);
  ASSERT_EQ(knots.size(), 2u);
  EXPECT_TRUE(std::isnan(knots.back().poseeulerzyxBaseRtWorld(3)));
}

}  // namespace
}  // namespace ocs2::legged
