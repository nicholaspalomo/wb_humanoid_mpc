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

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "Eigen/Geometry"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/command/BasePlanarMotion.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/reference_manager/StrideMotionHistory.h"

/*
 * The stride mean of the base's planar motion (design README section 2.9): over
 * a window of one gait cycle it is the robot's mean heading rate and
 * heading-frame velocity, the gait's periodic sway removed, exactly for a
 * steady twist and to the second order in the sample spacing for a swaying one;
 * a window of another length leaks the sway by the boxcar's |sinc(pi W / P)|. A
 * stall of up to kMaxBridgedGap is bridged without a seed, exactly for a steady
 * twist and with the heading rate exact while the window starts before it; a
 * mean is measured (isMeasuredOver()) only over a window of dense samples after
 * the first one. Against closed forms and a fine quadrature of the motion;
 * builds no model.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kCycle = 1.4;  // [s] the shipped walk's

/** R_z(angle) v. */
vector2_t rotated(scalar_t angle, const vector2_t& vector) {
  return vector2_t(std::cos(angle) * vector(0) - std::sin(angle) * vector(1),
                   std::sin(angle) * vector(0) + std::cos(angle) * vector(1));
}

/** A planar motion given by its heading Psi(t) and its heading-frame velocity
 * v_h(t): the base's p(t) = int R_z(Psi) v_h. */
struct PlanarTrajectory {
  std::function<scalar_t(scalar_t)> yawBaseRtWorld;
  std::function<vector2_t(scalar_t)> linvelBaseRtWorldEwrtHeadingXy;

  /** [m] p(t1) - p(t0), by composite Simpson with 200 steps: off by far below
   * 1e-15 m on the motions below. */
  vector2_t displacement(scalar_t t0, scalar_t t1) const {
    constexpr int kSteps = 200;
    const scalar_t step = (t1 - t0) / kSteps;
    vector2_t sum = vector2_t::Zero();
    for (int i = 0; i <= kSteps; ++i) {
      const scalar_t time = t0 + i * step;
      const scalar_t weight =
          (i == 0 || i == kSteps) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0);
      sum += weight * rotated(yawBaseRtWorld(time),
                              linvelBaseRtWorldEwrtHeadingXy(time));
    }
    return sum * step / 3.0;
  }
};

/** The robot walking at a steady body twist (forward 0.4, sideways 0.05 m/s,
 * turning at 0.3 rad/s) with a sway of period kCycle. */
PlanarTrajectory swayingWalk(
    scalar_t headingSway, const vector2_t& swayLinvelBaseRtWorldEwrtHeadingXy) {
  return PlanarTrajectory{
      .yawBaseRtWorld =
          [headingSway](scalar_t t) {
            return 0.3 * t + headingSway * std::sin(2.0 * M_PI * t / kCycle);
          },
      .linvelBaseRtWorldEwrtHeadingXy =
          [swayLinvelBaseRtWorldEwrtHeadingXy](scalar_t t) {
            return vector2_t(vector2_t(0.4, 0.05) +
                             swayLinvelBaseRtWorldEwrtHeadingXy *
                                 std::sin(2.0 * M_PI * t / kCycle + 0.7));
          }};
}

/** No motion before the first sample. */
BasePlanarMotion atRest() {
  return BasePlanarMotion{};
}

/** `history` fed the base of `trajectory` at `times` (in increasing order), the
 * motion before the first being `seed`. */
void feed(StrideMotionHistory& history,
          const PlanarTrajectory& trajectory,
          const std::vector<scalar_t>& times,
          const BasePlanarMotion& seed) {
  vector2_t posBaseRtWorldEwrtWorldXy = vector2_t(1.0, -2.0);
  for (size_t i = 0; i < times.size(); ++i) {
    if (i > 0)
      posBaseRtWorldEwrtWorldXy +=
          trajectory.displacement(times[i - 1], times[i]);
    history.addSample(times[i], posBaseRtWorldEwrtWorldXy,
                      trajectory.yawBaseRtWorld(times[i]),
                      [&seed]() { return seed; });
  }
}

/** Sample times from 0 to `end`, `spacing` apart. */
std::vector<scalar_t> uniformTimes(scalar_t spacing, scalar_t end) {
  std::vector<scalar_t> times;
  for (int i = 0; i * spacing <= end + 1.0e-12; ++i)
    times.push_back(i * spacing);
  return times;
}

/** Sample times from 0 to past `end`, 8 to 12 ms apart, as the solves come. */
std::vector<scalar_t> irregularTimes(scalar_t end, unsigned int seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<scalar_t> spacing(0.008, 0.012);
  std::vector<scalar_t> times{0.0};
  while (times.back() < end) times.push_back(times.back() + spacing(generator));
  return times;
}

/** The largest deviation of the stride mean from the steady twist over every
 * sample a full window after the first. */
struct MeanDeviation {
  scalar_t rateYawBaseRtWorld = 0.0;
  scalar_t normErrLinvelBaseRtWorldEwrtHeadingXy = 0.0;
};

MeanDeviation deviationOverTheRun(const PlanarTrajectory& trajectory,
                                  const std::vector<scalar_t>& times,
                                  scalar_t window) {
  StrideMotionHistory history(/*retention=*/2.0);
  MeanDeviation deviation;
  vector2_t posBaseRtWorldEwrtWorldXy = vector2_t::Zero();
  for (size_t i = 0; i < times.size(); ++i) {
    if (i > 0)
      posBaseRtWorldEwrtWorldXy +=
          trajectory.displacement(times[i - 1], times[i]);
    history.addSample(times[i], posBaseRtWorldEwrtWorldXy,
                      trajectory.yawBaseRtWorld(times[i]), atRest);
    if (times[i] < window) continue;
    const BasePlanarMotion mean = history.mean(window);
    deviation.rateYawBaseRtWorld = std::max(
        deviation.rateYawBaseRtWorld, std::abs(mean.rateYawBaseRtWorld - 0.3));
    deviation.normErrLinvelBaseRtWorldEwrtHeadingXy =
        std::max(deviation.normErrLinvelBaseRtWorldEwrtHeadingXy,
                 (mean.linvelBaseRtWorldEwrtHeadingXy - vector2_t(0.4, 0.05))
                     .cwiseAbs()
                     .maxCoeff());
  }
  return deviation;
}

TEST(StrideMotionHistory, TheMeanOfASteadyTwistIsTheTwistAtAnySampling) {
  // dD = R_z(Psi_mid)^T dp / sinc(dPsi / 2) is exact on the arc of a constant
  // twist, and Psi and D are linear in time between samples: round-off is all
  // that is left, also on irregular samples.
  const PlanarTrajectory steady =
      swayingWalk(/*headingSway=*/0.0,
                  /*swayLinvelBaseRtWorldEwrtHeadingXy=*/vector2_t::Zero());
  for (const scalar_t window : {0.5, kCycle, 1.7}) {
    SCOPED_TRACE(window);
    const MeanDeviation deviation = deviationOverTheRun(
        steady, irregularTimes(/*end=*/4.0, /*seed=*/3), window);
    EXPECT_LT(deviation.rateYawBaseRtWorld, 1.0e-12);
    EXPECT_LT(deviation.normErrLinvelBaseRtWorldEwrtHeadingXy, 1.0e-12);
  }
}

TEST(StrideMotionHistory, AMeanOverOneCycleRemovesTheGaitsSway) {
  // The pelvis yaws by 0.08 rad (0.36 rad/s, more than slow_walk's 0.15 rad/s
  // threshold) and sways sideways at 0.2 m/s with the gait cycle; the boxcar
  // over one cycle has a zero at every harmonic of it.
  const PlanarTrajectory walk =
      swayingWalk(/*headingSway=*/0.08,
                  /*swayLinvelBaseRtWorldEwrtHeadingXy=*/vector2_t(0.05, 0.2));
  // Samples a whole number to the cycle: the start of the window is a sample,
  // so the heading rate is exact, and the velocity is off by the
  // sample-to-sample quadrature only (second order in the spacing).
  const MeanDeviation onTheCycle = deviationOverTheRun(
      walk, uniformTimes(/*spacing=*/kCycle / 140.0, /*end=*/3.0 * kCycle),
      kCycle);
  EXPECT_LT(onTheCycle.rateYawBaseRtWorld, 1.0e-12);
  EXPECT_LT(onTheCycle.normErrLinvelBaseRtWorldEwrtHeadingXy, 1.0e-5);
  // As the solves come, 8 to 12 ms apart: the linear interpolation at the
  // window's start adds a second-order error.
  const MeanDeviation irregular = deviationOverTheRun(
      walk, irregularTimes(/*end=*/3.0 * kCycle, /*seed=*/5), kCycle);
  EXPECT_LT(irregular.rateYawBaseRtWorld, 1.0e-4);
  EXPECT_LT(irregular.normErrLinvelBaseRtWorldEwrtHeadingXy, 1.0e-4);
}

TEST(StrideMotionHistory,
     AWindowOfAnotherLengthLeaksTheSwayByTheBoxcarsResponse) {
  // A sway a sin(2 pi t / P) of the heading adds a (2 pi / P) cos(...) to its
  // rate; its mean over W is that times sinc(pi W / P) at a phase, so the worst
  // deviation over a run is a (2 pi / P) |sinc(pi W / P)|.
  constexpr scalar_t kHeadingSway = 0.08;
  const PlanarTrajectory walk = swayingWalk(
      kHeadingSway, /*swayLinvelBaseRtWorldEwrtHeadingXy=*/vector2_t::Zero());
  for (const scalar_t window : {1.2, 1.7}) {
    SCOPED_TRACE(window);
    const scalar_t x = M_PI * window / kCycle;
    const scalar_t predicted =
        kHeadingSway * 2.0 * M_PI / kCycle * std::abs(std::sin(x) / x);
    const MeanDeviation deviation = deviationOverTheRun(
        walk, uniformTimes(/*spacing=*/0.001, /*end=*/4.0), window);
    EXPECT_NEAR(deviation.rateYawBaseRtWorld, predicted, 1.0e-3 * predicted);
  }
}

TEST(StrideMotionHistory, BeforeTheFirstSampleTheSeedsMotionIsAveragedIn) {
  StrideMotionHistory history(/*retention=*/2.0);
  EXPECT_TRUE(history.empty());
  EXPECT_EQ(history.mean(/*window=*/1.0).rateYawBaseRtWorld, 0.0)
      << "an empty history is at rest";
  // Not const: the seed function returns it by value.
  BasePlanarMotion seed{.linvelBaseRtWorldEwrtHeadingXy = vector2_t(0.3, -0.1),
                        .rateYawBaseRtWorld = 0.5};
  int seedCalls = 0;
  const std::function<BasePlanarMotion()> countedSeed = [&seed, &seedCalls]() {
    ++seedCalls;
    return seed;
  };
  // The first sample: the window holds the seed alone.
  history.addSample(/*time=*/10.0, vector2_t(1.0, 1.0), /*yawBaseRtWorld=*/0.2,
                    countedSeed);
  EXPECT_EQ(seedCalls, 1);
  EXPECT_NEAR(history.mean(/*window=*/1.0).rateYawBaseRtWorld, 0.5, 1.0e-15);
  EXPECT_LT((history.mean(/*window=*/1.0).linvelBaseRtWorldEwrtHeadingXy -
             seed.linvelBaseRtWorldEwrtHeadingXy)
                .norm(),
            1.0e-15);
  // 0.04 s later the base has moved forward at 0.1 m/s in its heading frame
  // while turning at 0.1 rad/s, along the chord 0.04 sinc(0.002) R_z(0.202)
  // (0.1, 0) of that arc: the 1 s window is 0.96 s of the seed and 0.04 s of
  // that.
  const scalar_t chord = 0.004 * std::sin(0.002) / 0.002;
  history.addSample(
      /*time=*/10.04,
      vector2_t(1.0, 1.0) + rotated(/*angle=*/0.202, vector2_t(chord, 0.0)),
      /*heading=*/0.204, countedSeed);
  EXPECT_EQ(seedCalls, 1)
      << "a sample that continues the history needs no seed";
  const BasePlanarMotion mean = history.mean(/*window=*/1.0);
  EXPECT_NEAR(mean.rateYawBaseRtWorld, 0.96 * 0.5 + 0.04 * 0.1, 1.0e-12);
  EXPECT_NEAR(mean.linvelBaseRtWorldEwrtHeadingXy(0), 0.96 * 0.3 + 0.04 * 0.1,
              1.0e-12);
  EXPECT_NEAR(mean.linvelBaseRtWorldEwrtHeadingXy(1), 0.96 * -0.1, 1.0e-12);
}

TEST(StrideMotionHistory, ItRestartsWhenTheClockRunsBackwardsOrSkips) {
  StrideMotionHistory history(/*retention=*/2.0);
  // Not const: the seed function returns it by value.
  BasePlanarMotion turning{.linvelBaseRtWorldEwrtHeadingXy = vector2_t::Zero(),
                           .rateYawBaseRtWorld = 0.4};
  int seedCalls = 0;
  const std::function<BasePlanarMotion()> seed = [&turning, &seedCalls]() {
    ++seedCalls;
    return turning;
  };
  history.addSample(/*time=*/1.0, vector2_t::Zero(), /*yawBaseRtWorld=*/0.0,
                    seed);
  history.addSample(/*time=*/1.01, vector2_t::Zero(), /*yawBaseRtWorld=*/0.0,
                    seed);
  history.addSample(/*time=*/1.01, vector2_t::Zero(), /*yawBaseRtWorld=*/3.0,
                    seed);
  EXPECT_EQ(seedCalls, 1);
  EXPECT_LT(history.mean(/*window=*/0.01).rateYawBaseRtWorld, 1.0e-12)
      << "a sample at the last one's time is dropped";
  // Backwards: a new history, from its seed.
  history.addSample(/*time=*/0.5, vector2_t::Zero(), /*yawBaseRtWorld=*/1.0,
                    seed);
  EXPECT_EQ(seedCalls, 2);
  EXPECT_NEAR(history.mean(/*window=*/1.0).rateYawBaseRtWorld, 0.4, 1.0e-15);
  // A gap up to kMaxBridgedGap is bridged; one beyond it starts a new history
  // too.
  history.addSample(/*time=*/0.5 + StrideMotionHistory::kMaxBridgedGap,
                    vector2_t::Zero(), /*yawBaseRtWorld=*/1.0, seed);
  EXPECT_EQ(seedCalls, 2);
  history.addSample(
      /*time=*/0.5 + 2.0 * StrideMotionHistory::kMaxBridgedGap + 1.0e-3,
      vector2_t::Zero(), /*yawBaseRtWorld=*/1.0, seed);
  EXPECT_EQ(seedCalls, 3);
  history.reset();
  EXPECT_TRUE(history.empty());
  EXPECT_FALSE(history.isMeasuredOver(/*window=*/1.0));
}

TEST(StrideMotionHistory, AMeanIsMeasuredOnceItsWindowLiesAfterTheFirstSample) {
  // Before, the window reaches back to the seed: a mean, but not a measured
  // one.
  StrideMotionHistory history(/*retention=*/2.0);
  const PlanarTrajectory steady =
      swayingWalk(/*headingSway=*/0.0,
                  /*swayLinvelBaseRtWorldEwrtHeadingXy=*/vector2_t::Zero());
  vector2_t posBaseRtWorldEwrtWorldXy = vector2_t::Zero();
  for (int i = 0; i <= 200; ++i) {
    const scalar_t time = 0.01 * i;
    if (i > 0)
      posBaseRtWorldEwrtWorldXy += steady.displacement(time - 0.01, time);
    history.addSample(time, posBaseRtWorldEwrtWorldXy,
                      steady.yawBaseRtWorld(time), atRest);
    EXPECT_EQ(history.isMeasuredOver(kCycle), time >= kCycle - 1.0e-12) << time;
  }
  // The window is cut to the retention, so a longer one is measured once the
  // retention is.
  EXPECT_TRUE(history.isMeasuredOver(/*window=*/5.0));
}

/** `trajectory` sampled every 10 ms from 0 to `end`, without the samples
 * strictly inside (gapStart, gapStart + gap). */
std::vector<scalar_t> timesWithAGap(scalar_t end,
                                    scalar_t gapStart,
                                    scalar_t gap) {
  std::vector<scalar_t> times;
  for (int i = 0; 0.01 * i <= end + 1.0e-12; ++i) {
    const scalar_t time = 0.01 * i;
    if (time > gapStart + 1.0e-9 && time < gapStart + gap - 1.0e-9) continue;
    times.push_back(time);
  }
  return times;
}

TEST(StrideMotionHistory,
     AStallUpToTheBridgedGapKeepsAConstantTwistsMeanExact) {
  // A network stall or a starved solver: up to kMaxBridgedGap the history goes
  // on across the gap, no seed is read, and the chord step is exact on the arc
  // of a constant body twist however long the gap.
  const PlanarTrajectory steady =
      swayingWalk(/*headingSway=*/0.0,
                  /*swayLinvelBaseRtWorldEwrtHeadingXy=*/vector2_t::Zero());
  // The longest gap a sample short of kMaxBridgedGap, so that the sample times'
  // round-off cannot take it past.
  for (const scalar_t gap :
       {0.15, 0.5, 1.0, StrideMotionHistory::kMaxBridgedGap - 0.01}) {
    SCOPED_TRACE(gap);
    constexpr scalar_t kGapStart = 2.5;
    StrideMotionHistory history(/*retention=*/2.0);
    int seedCalls = 0;
    const std::vector<scalar_t> times =
        timesWithAGap(/*end=*/7.0, kGapStart, gap);
    vector2_t posBaseRtWorldEwrtWorldXy = vector2_t(1.0, -2.0);
    for (size_t i = 0; i < times.size(); ++i) {
      if (i > 0)
        posBaseRtWorldEwrtWorldXy +=
            steady.displacement(times[i - 1], times[i]);
      history.addSample(times[i], posBaseRtWorldEwrtWorldXy,
                        steady.yawBaseRtWorld(times[i]), [&seedCalls]() {
                          ++seedCalls;
                          return BasePlanarMotion{};
                        });
      if (times[i] < kCycle) continue;
      const BasePlanarMotion mean = history.mean(kCycle);
      EXPECT_NEAR(mean.rateYawBaseRtWorld, 0.3, 1.0e-12) << times[i];
      EXPECT_LT((mean.linvelBaseRtWorldEwrtHeadingXy - vector2_t(0.4, 0.05))
                    .cwiseAbs()
                    .maxCoeff(),
                1.0e-12)
          << times[i];
      // A window that holds the gap is a mean, not a measured one (the sample a
      // window after the gap left out, at which the comparison is one of
      // round-off).
      const scalar_t afterTheGap = times[i] - kCycle - (kGapStart + gap);
      if (times[i] <= kGapStart) {
        EXPECT_TRUE(history.isMeasuredOver(kCycle)) << times[i];
      } else if (std::abs(afterTheGap) > 1.0e-6) {
        EXPECT_EQ(history.isMeasuredOver(kCycle), afterTheGap > 0.0)
            << times[i];
      }
    }
    EXPECT_EQ(seedCalls, 1) << "only the first sample is seeded";
  }
}

TEST(StrideMotionHistory,
     AcrossABridgedGapTheHeadingRateIsExactUntilTheWindowStartsInsideIt) {
  // The robot sways (0.08 rad of heading, 0.36 rad/s of heading rate) through a
  // stall. While the window starts before the gap its ends are samples and the
  // heading rate is exact; once it starts inside, Psi there is the line between
  // the gap's two samples, so the mean is off by (Psi_line(s) - Psi(s)) / W at
  // the window's start s.
  const PlanarTrajectory walk =
      swayingWalk(/*headingSway=*/0.08,
                  /*swayLinvelBaseRtWorldEwrtHeadingXy=*/vector2_t::Zero());
  for (const scalar_t gap : {0.15, 0.5, 1.0}) {
    SCOPED_TRACE(gap);
    constexpr scalar_t kGapStart = 2.5;
    StrideMotionHistory history(/*retention=*/2.0);
    const std::vector<scalar_t> times =
        timesWithAGap(/*end=*/7.0, kGapStart, gap);
    vector2_t posBaseRtWorldEwrtWorldXy = vector2_t::Zero();
    scalar_t largestStartInsideError = 0.0;
    for (size_t i = 0; i < times.size(); ++i) {
      if (i > 0)
        posBaseRtWorldEwrtWorldXy += walk.displacement(times[i - 1], times[i]);
      history.addSample(times[i], posBaseRtWorldEwrtWorldXy,
                        walk.yawBaseRtWorld(times[i]), atRest);
      const scalar_t start = times[i] - kCycle;
      if (start < 0.0) continue;
      const scalar_t exact =
          (walk.yawBaseRtWorld(times[i]) - walk.yawBaseRtWorld(start)) / kCycle;
      const scalar_t mean = history.mean(kCycle).rateYawBaseRtWorld;
      if (start <= kGapStart || start >= kGapStart + gap) {
        // On the 10 ms grid every window start is a sample.
        EXPECT_NEAR(mean, exact, 1.0e-12) << times[i];
      } else {
        const scalar_t fraction = (start - kGapStart) / gap;
        const scalar_t line =
            (1.0 - fraction) * walk.yawBaseRtWorld(kGapStart) +
            fraction * walk.yawBaseRtWorld(kGapStart + gap);
        EXPECT_NEAR(mean - exact, (walk.yawBaseRtWorld(start) - line) / kCycle,
                    1.0e-12)
            << times[i];
        largestStartInsideError =
            std::max(largestStartInsideError, std::abs(mean - exact));
      }
    }
    // At most the heading's peak-to-peak sway over W: 2 * 0.08 / 1.4.
    EXPECT_GT(largestStartInsideError, 1.0e-3)
        << "the windows must start inside the gap";
    EXPECT_LT(largestStartInsideError, 2.0 * 0.08 / kCycle);
  }
}

TEST(StrideMotionHistory,
     AcrossABridgedGapTheHeadingIsUnwrappedBelowTheStatedRate) {
  // Across a gap of kMaxBridgedGap the heading takes the branch of |dPsi| <=
  // pi: a steady turn at 1.9 rad/s changes it by 2.85 rad there and is read
  // exactly; at 2.2 rad/s (3.3 rad) the gap is read as a turn the other way.
  for (const scalar_t yawRate : {1.9, -1.9, 2.2}) {
    SCOPED_TRACE(yawRate);
    StrideMotionHistory history(/*retention=*/2.0);
    for (int i = 0; i <= 100; ++i) {
      const scalar_t time = 0.01 * i;
      history.addSample(time, vector2_t::Zero(),
                        std::remainder(yawRate * time, 2.0 * M_PI), atRest);
    }
    const scalar_t after = 1.0 + StrideMotionHistory::kMaxBridgedGap;
    history.addSample(after, vector2_t::Zero(),
                      std::remainder(yawRate * after, 2.0 * M_PI), atRest);
    const scalar_t window = StrideMotionHistory::kMaxBridgedGap + 0.5;
    if (std::abs(yawRate) * StrideMotionHistory::kMaxBridgedGap < M_PI) {
      EXPECT_NEAR(history.mean(window).rateYawBaseRtWorld, yawRate, 1.0e-12);
    } else {
      EXPECT_NEAR(history.mean(window).rateYawBaseRtWorld,
                  yawRate - 2.0 * M_PI / window, 1.0e-12);
    }
  }
}

TEST(StrideMotionHistory, TheHeadingIsUnwrappedThroughPlusMinusPi) {
  // Turning left at 1 rad/s through pi: the wrapped twist heading jumps from pi
  // to -pi, the turn does not.
  StrideMotionHistory history(/*retention=*/2.0);
  for (int i = 0; i <= 150; ++i) {
    const scalar_t time = 0.01 * i;
    history.addSample(time, vector2_t::Zero(),
                      std::remainder(2.5 + time, 2.0 * M_PI), atRest);
  }
  EXPECT_NEAR(history.mean(/*window=*/1.0).rateYawBaseRtWorld, 1.0, 1.0e-12);
}

TEST(StrideMotionHistory, ALongRunKeepsItsRetentionAndCutsLongerWindows) {
  const PlanarTrajectory steady =
      swayingWalk(/*headingSway=*/0.0,
                  /*swayLinvelBaseRtWorldEwrtHeadingXy=*/vector2_t::Zero());
  StrideMotionHistory history(/*retention=*/1.0);
  feed(history, steady, uniformTimes(/*spacing=*/0.01, /*end=*/30.0), atRest());
  EXPECT_NEAR(history.mean(/*window=*/1.0).rateYawBaseRtWorld, 0.3, 1.0e-12);
  // A window longer than the retention is cut to it, and still averages the
  // measured motion, not the seed.
  EXPECT_NEAR(history.mean(/*window=*/5.0).rateYawBaseRtWorld, 0.3, 1.0e-12);
  EXPECT_LT((history.mean(/*window=*/5.0).linvelBaseRtWorldEwrtHeadingXy -
             vector2_t(0.4, 0.05))
                .norm(),
            1.0e-12);
}

TEST(InstantaneousPlanarMotion,
     IsTheRateOfTheTwistHeadingAndTheVelocityInItsFrame) {
  // At tilted attitudes the heading rate differs from the world yaw rate at
  // second order in the tilt; it is the derivative of yawOf() along the
  // body rotation, here by central differences of xi (x) Exp(w_B h).
  std::mt19937 generator(7);
  std::uniform_real_distribution<scalar_t> uniform(-1.0, 1.0);
  scalar_t largestYawRateGap = 0.0;
  for (int sample = 0; sample < 50; ++sample) {
    const Eigen::Quaterniond attitude =
        Eigen::AngleAxisd(3.0 * uniform(generator), vector3_t::UnitZ()) *
        Eigen::AngleAxisd(0.5 * uniform(generator), vector3_t::UnitY()) *
        Eigen::AngleAxisd(0.5 * uniform(generator), vector3_t::UnitX());
    const vector3_t angvelBaseRtWorldEwrtBase(
        uniform(generator), uniform(generator), uniform(generator));
    vector6_t twistBaseRtWorldEwrtWorld;
    twistBaseRtWorldEwrtWorld << uniform(generator), uniform(generator),
        uniform(generator), attitude * angvelBaseRtWorldEwrtBase;
    const BasePlanarMotion motion =
        instantaneousPlanarMotion(attitude.coeffs(), twistBaseRtWorldEwrtWorld);

    constexpr scalar_t kStep = 1.0e-6;
    const Eigen::Quaterniond increment(
        Eigen::AngleAxisd(angvelBaseRtWorldEwrtBase.norm() * kStep,
                          angvelBaseRtWorldEwrtBase.normalized()));
    const scalar_t after = yawOf((attitude * increment).coeffs());
    const scalar_t before = yawOf((attitude * increment.conjugate()).coeffs());
    EXPECT_NEAR(motion.rateYawBaseRtWorld,
                std::remainder(after - before, 2.0 * M_PI) / (2.0 * kStep),
                1.0e-7);
    const vector2_t expectedLinvelBaseRtWorldEwrtHeadingXy =
        rotated(-yawOf(attitude.coeffs()),
                vector2_t(twistBaseRtWorldEwrtWorld.head<2>()));
    EXPECT_LT((motion.linvelBaseRtWorldEwrtHeadingXy -
               expectedLinvelBaseRtWorldEwrtHeadingXy)
                  .norm(),
              1.0e-14);
    largestYawRateGap = std::max(
        largestYawRateGap,
        std::abs(motion.rateYawBaseRtWorld - twistBaseRtWorldEwrtWorld(5)));
  }
  EXPECT_GT(largestYawRateGap, 1.0e-3)
      << "the attitudes must tell the heading rate from the world yaw rate";
}

}  // namespace
}  // namespace ocs2::humanoid
