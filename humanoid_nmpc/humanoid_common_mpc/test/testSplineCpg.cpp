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
#include <random>

#include "gtest/gtest.h"

#include "humanoid_common_mpc/swing_foot_planner/CubicSpline.h"
#include "humanoid_common_mpc/swing_foot_planner/SplineCpg.h"

/**
 * The time derivatives of CubicSpline and SplineCpg against central finite
 * differences of re-built splines. SplineCpg's apex time is the mean of the
 * lift-off and touch-down times, so moving either end moves the apex by half as
 * much.
 */
namespace ocs2::humanoid {
namespace {

constexpr scalar_t kStep = 1.0e-6;  // [s] finite-difference step
constexpr scalar_t kTol = 1.0e-6;

/** The swing ends and apex of one random trial. */
struct SwingNodes {
  CubicSpline::Node liftOff;
  scalar_t apex;
  CubicSpline::Node touchDown;
};

SwingNodes randomNodes(std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> start(-1.0, 1.0);
  std::uniform_real_distribution<scalar_t> duration(0.1, 1.5);
  std::normal_distribution<scalar_t> value(0.0, 0.1);
  const scalar_t liftOffTime = start(generator);
  const scalar_t touchDownTime = liftOffTime + duration(generator);
  return SwingNodes{.liftOff = {.time = liftOffTime,
                                .position = value(generator),
                                .velocity = value(generator)},
                    .apex = 0.1 + std::abs(value(generator)),
                    .touchDown = {.time = touchDownTime,
                                  .position = value(generator),
                                  .velocity = value(generator)}};
}

SplineCpg shifted(const SwingNodes& nodes,
                  scalar_t liftOffShift,
                  scalar_t touchDownShift) {
  CubicSpline::Node liftOff = nodes.liftOff;
  CubicSpline::Node touchDown = nodes.touchDown;
  liftOff.time += liftOffShift;
  touchDown.time += touchDownShift;
  return SplineCpg(liftOff, nodes.apex, touchDown);
}

}  // namespace

TEST(SplineCpg, TimeDerivativesMatchFiniteDifferences) {
  std::mt19937 generator(5);
  std::uniform_real_distribution<scalar_t> fraction(0.01, 0.99);
  for (int trial = 0; trial < 2000; ++trial) {
    const SwingNodes nodes = randomNodes(generator);
    const SplineCpg spline(nodes.liftOff, nodes.apex, nodes.touchDown);
    const scalar_t duration = nodes.touchDown.time - nodes.liftOff.time;
    const scalar_t time = nodes.liftOff.time + fraction(generator) * duration;
    // Keep the evaluation away from the apex, where the active half changes
    // under a shift of the ends.
    if (std::abs(time - 0.5 * (nodes.liftOff.time + nodes.touchDown.time)) <
        1.0e-3)
      continue;

    const scalar_t startDifference =
        (shifted(nodes, kStep, /*touchDownShift=*/0.0).position(time) -
         shifted(nodes, -kStep, /*touchDownShift=*/0.0).position(time)) /
        (2.0 * kStep);
    const scalar_t finalDifference =
        (shifted(nodes, /*liftOffShift=*/0.0, kStep).position(time) -
         shifted(nodes, /*liftOffShift=*/0.0, -kStep).position(time)) /
        (2.0 * kStep);
    EXPECT_NEAR(spline.startTimeDerivative(time), startDifference, kTol)
        << "trial " << trial;
    EXPECT_NEAR(spline.finalTimeDerivative(time), finalDifference, kTol)
        << "trial " << trial;
  }
}

TEST(SplineCpg, CubicSplineTimeDerivativesMatchFiniteDifferences) {
  std::mt19937 generator(9);
  std::uniform_real_distribution<scalar_t> fraction(0.0, 1.0);
  for (int trial = 0; trial < 2000; ++trial) {
    const SwingNodes nodes = randomNodes(generator);
    const CubicSpline spline(nodes.liftOff, nodes.touchDown);
    const scalar_t time =
        nodes.liftOff.time +
        fraction(generator) * (nodes.touchDown.time - nodes.liftOff.time);
    CubicSpline::Node earlier = nodes.liftOff;
    CubicSpline::Node later = nodes.liftOff;
    earlier.time -= kStep;
    later.time += kStep;
    const scalar_t startDifference =
        (CubicSpline(later, nodes.touchDown).position(time) -
         CubicSpline(earlier, nodes.touchDown).position(time)) /
        (2.0 * kStep);
    CubicSpline::Node earlierEnd = nodes.touchDown;
    CubicSpline::Node laterEnd = nodes.touchDown;
    earlierEnd.time -= kStep;
    laterEnd.time += kStep;
    const scalar_t finalDifference =
        (CubicSpline(nodes.liftOff, laterEnd).position(time) -
         CubicSpline(nodes.liftOff, earlierEnd).position(time)) /
        (2.0 * kStep);
    EXPECT_NEAR(spline.startTimeDerivative(time), startDifference, kTol)
        << "trial " << trial;
    EXPECT_NEAR(spline.finalTimeDerivative(time), finalDifference, kTol)
        << "trial " << trial;
  }
}

}  // namespace ocs2::humanoid
