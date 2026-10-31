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
#include <cstddef>
#include <limits>

#include "gtest/gtest.h"

#include "humanoid_common_mpc/command/TargetTrajectoriesCalculatorBase.h"
#include "humanoid_common_mpc/common/Types.h"

/*
 * The arc of a velocity command's knots (design README section 2.9), their
 * subdivision and the assumption of the first segment: the displacement of a
 * constant planar twist against the closed form (v / omega) [[sin, cos - 1], [1
 * - cos, sin]] and an RK4 integration of the unicycle, bit for bit T v when it
 * does not turn; the number of parts that keeps knots at most pi / 2 apart and
 * their linear interpolation within the tolerances of the arc, measured on a
 * dense grid; and the held mean twist against an RK4 integration of a linearly
 * ramping twist, whose difference is the documented leading term. Builds no
 * model.
 */

namespace ocs2::humanoid {
namespace {

/** R_z(angle) v. */
vector2_t rotated(scalar_t angle, const vector2_t& vector) {
  return vector2_t(std::cos(angle) * vector(0) - std::sin(angle) * vector(1),
                   std::sin(angle) * vector(0) + std::cos(angle) * vector(1));
}

/** p(T) - p(0) of dp/dt = R_z(omega t) v0, by RK4 with 10000 steps. */
vector2_t integratedArc(const vector2_t& initLinvelBaseRtWorldEwrtWorldXy,
                        scalar_t rateYawBaseRtWorld,
                        scalar_t duration) {
  constexpr int kSteps = 10000;
  const scalar_t step = duration / kSteps;
  vector2_t posBaseRtWorldEwrtWorldXy = vector2_t::Zero();
  for (int i = 0; i < kSteps; ++i) {
    const scalar_t t = i * step;
    const vector2_t k1 =
        rotated(rateYawBaseRtWorld * t, initLinvelBaseRtWorldEwrtWorldXy);
    const vector2_t k23 = rotated(rateYawBaseRtWorld * (t + 0.5 * step),
                                  initLinvelBaseRtWorldEwrtWorldXy);
    const vector2_t k4 = rotated(rateYawBaseRtWorld * (t + step),
                                 initLinvelBaseRtWorldEwrtWorldXy);
    posBaseRtWorldEwrtWorldXy += step / 6.0 * (k1 + 4.0 * k23 + k4);
  }
  return posBaseRtWorldEwrtWorldXy;
}

TEST(VelocityCommandArc, TheDisplacementIsTheArcOfTheTwist) {
  for (const scalar_t rateYawBaseRtWorld : {0.3, -1.0, 4.0, 1.0e-7}) {
    for (const scalar_t duration : {0.84, 0.36}) {
      SCOPED_TRACE(rateYawBaseRtWorld);
      SCOPED_TRACE(duration);
      const vector2_t initLinvelBaseRtWorldEwrtWorldXy(0.3, -0.12);
      const vector2_t displacement =
          TargetTrajectoriesCalculatorBase::planarArcDisplacement(
              initLinvelBaseRtWorldEwrtWorldXy, rateYawBaseRtWorld, duration);
      EXPECT_LT((displacement - integratedArc(initLinvelBaseRtWorldEwrtWorldXy,
                                              rateYawBaseRtWorld, duration))
                    .norm(),
                1.0e-12);
      const scalar_t theta = rateYawBaseRtWorld * duration;
      if (std::abs(theta) > 1.0e-3) {
        // (1 / omega) [[sin(theta), cos(theta) - 1], [1 - cos(theta),
        // sin(theta)]] v0.
        const vector2_t closedForm(
            std::sin(theta) * initLinvelBaseRtWorldEwrtWorldXy(0) +
                (std::cos(theta) - 1.0) * initLinvelBaseRtWorldEwrtWorldXy(1),
            (1.0 - std::cos(theta)) * initLinvelBaseRtWorldEwrtWorldXy(0) +
                std::sin(theta) * initLinvelBaseRtWorldEwrtWorldXy(1));
        EXPECT_LT((displacement - closedForm / rateYawBaseRtWorld).norm(),
                  1.0e-14);
      }
    }
  }
}

TEST(VelocityCommandArc, WithoutATurnItIsTheStraightLineBitForBit) {
  const vector2_t initLinvelBaseRtWorldEwrtWorldXy(0.37, -0.11);
  const vector2_t displacement =
      TargetTrajectoriesCalculatorBase::planarArcDisplacement(
          initLinvelBaseRtWorldEwrtWorldXy, /*rateYawBaseRtWorld=*/0.0,
          /*duration=*/0.7);
  EXPECT_EQ(displacement(0), initLinvelBaseRtWorldEwrtWorldXy(0) * 0.7);
  EXPECT_EQ(displacement(1), initLinvelBaseRtWorldEwrtWorldXy(1) * 0.7);
}

TEST(VelocityCommandArc, TheDesignReadmesArcIsReproduced) {
  // Design README section 2.9: at 0.3 m/s forward and 0.3 rad/s over 1.2 s the
  // end of the horizon lies (v / omega)(sin, 1 - cos) from the start, 0.065 m
  // off the straight chord (0.36, 0) the knots used to take.
  const vector2_t displacement =
      TargetTrajectoriesCalculatorBase::planarArcDisplacement(
          vector2_t(0.3, 0.0), /*rateYawBaseRtWorld=*/0.3, /*duration=*/1.2);
  EXPECT_NEAR(displacement(0), std::sin(0.36), 1.0e-15);
  EXPECT_NEAR(displacement(1), 1.0 - std::cos(0.36), 1.0e-15);
  EXPECT_NEAR((displacement - vector2_t(0.36, 0.0)).norm(), 0.0646, 1.0e-4);
}

using Extent = TargetTrajectoriesCalculatorBase::KnotSegmentExtent;

/** knotSubdivisionCount() of a segment that only turns, by `yawChange`. */
size_t turnParts(scalar_t yawChange) {
  return TargetTrajectoriesCalculatorBase::knotSubdivisionCount(
      Extent{.yawChange = yawChange, .arcLength = 0.0, .targetSpeed = 0.0});
}

TEST(VelocityCommandArc, ASegmentIsCutIntoPartsAtMostAQuarterTurnApart) {
  EXPECT_EQ(turnParts(0.0), 1);
  EXPECT_EQ(turnParts(0.84), 1);
  EXPECT_EQ(turnParts(0.5 * M_PI), 1);
  EXPECT_EQ(turnParts(0.5 * M_PI + 1.0e-9), 2);
  EXPECT_EQ(turnParts(-3.15), 3);
  EXPECT_EQ(turnParts(-2.5 * M_PI), 5);
  for (const scalar_t change : {0.3, 2.0, 3.15, 7.0, -11.0}) {
    SCOPED_TRACE(change);
    const size_t parts = turnParts(change);
    EXPECT_LE(std::abs(change) / static_cast<scalar_t>(parts),
              TargetTrajectoriesCalculatorBase::kMaxKnotYawSpacing);
    if (parts > 1) {
      EXPECT_GT(std::abs(change) / static_cast<scalar_t>(parts - 1), 0.5 * M_PI)
          << "the fewest parts";
    }
  }
  // A segment that only translates is not cut.
  EXPECT_EQ(TargetTrajectoriesCalculatorBase::knotSubdivisionCount(
                Extent{.yawChange = 0.0, .arcLength = 5.0, .targetSpeed = 3.0}),
            1);
  EXPECT_EQ(turnParts(1.0e9),
            TargetTrajectoriesCalculatorBase::kMaxKnotSubdivisions);
  // The cap's turn is the largest a segment makes (velocityCommandKnots()
  // clamps the yaw rate there): exactly pi / 2 apart.
  EXPECT_EQ(turnParts(TargetTrajectoriesCalculatorBase::kMaxSegmentYawChange),
            TargetTrajectoriesCalculatorBase::kMaxKnotSubdivisions);
  EXPECT_NEAR(TargetTrajectoriesCalculatorBase::kMaxSegmentYawChange /
                  static_cast<scalar_t>(
                      TargetTrajectoriesCalculatorBase::kMaxKnotSubdivisions),
              0.5 * M_PI, 1.0e-15);
  const scalar_t nan = std::numeric_limits<scalar_t>::quiet_NaN();
  const scalar_t infinity = std::numeric_limits<scalar_t>::infinity();
  EXPECT_EQ(turnParts(nan), 1);
  EXPECT_EQ(turnParts(infinity), 1);
  EXPECT_EQ(TargetTrajectoriesCalculatorBase::knotSubdivisionCount(
                Extent{.yawChange = 1.0, .arcLength = nan, .targetSpeed = 1.0}),
            1);
  EXPECT_EQ(TargetTrajectoriesCalculatorBase::knotSubdivisionCount(Extent{
                .yawChange = 1.0, .arcLength = 1.0, .targetSpeed = infinity}),
            1);
}

/** Largest distances, over a dense grid, of the linear interpolation of n knots
 * on an arc from the arc. Passive data. */
struct ChordDeviation {
  scalar_t maxErrPosBaseRtWorldEwrtWorld = 0.0;
  scalar_t maxErrLinvelBaseRtWorldEwrtWorld = 0.0;
};

/**
 * The deviation from the arc of a segment cut into `parts`: the base starts
 * with the world velocity `startVelocity` and turns at `yawRate` for
 * `duration`, the knots carry the velocity target R_z(psi_k) `targetVelocity`,
 * and between them the positions and targets are interpolated linearly, as OCS2
 * interpolates a TargetTrajectories.
 */
ChordDeviation chordDeviation(
    const vector2_t& initLinvelBaseRtWorldEwrtWorldXy,
    const vector2_t& cmdLinvelBaseRtWorldEwrtHeadingXy,
    scalar_t rateYawBaseRtWorld,
    scalar_t duration,
    size_t parts) {
  ChordDeviation deviation;
  const scalar_t interval = duration / static_cast<scalar_t>(parts);
  for (size_t part = 0; part < parts; ++part) {
    const scalar_t start = interval * static_cast<scalar_t>(part);
    const vector2_t from =
        TargetTrajectoriesCalculatorBase::planarArcDisplacement(
            initLinvelBaseRtWorldEwrtWorldXy, rateYawBaseRtWorld, start);
    const vector2_t to =
        TargetTrajectoriesCalculatorBase::planarArcDisplacement(
            initLinvelBaseRtWorldEwrtWorldXy, rateYawBaseRtWorld,
            start + interval);
    const vector2_t fromTarget =
        rotated(rateYawBaseRtWorld * start, cmdLinvelBaseRtWorldEwrtHeadingXy);
    const vector2_t toTarget = rotated(rateYawBaseRtWorld * (start + interval),
                                       cmdLinvelBaseRtWorldEwrtHeadingXy);
    constexpr int kSamples = 400;
    for (int sample = 0; sample <= kSamples; ++sample) {
      const scalar_t fraction = static_cast<scalar_t>(sample) / kSamples;
      const scalar_t time = start + fraction * interval;
      const vector2_t onTheArc =
          TargetTrajectoriesCalculatorBase::planarArcDisplacement(
              initLinvelBaseRtWorldEwrtWorldXy, rateYawBaseRtWorld, time);
      deviation.maxErrPosBaseRtWorldEwrtWorld =
          std::max(deviation.maxErrPosBaseRtWorldEwrtWorld,
                   ((1.0 - fraction) * from + fraction * to - onTheArc).norm());
      deviation.maxErrLinvelBaseRtWorldEwrtWorld =
          std::max(deviation.maxErrLinvelBaseRtWorldEwrtWorld,
                   ((1.0 - fraction) * fromTarget + fraction * toTarget -
                    rotated(rateYawBaseRtWorld * time,
                            cmdLinvelBaseRtWorldEwrtHeadingXy))
                       .norm());
    }
  }
  return deviation;
}

TEST(VelocityCommandArc,
     BetweenTheKnotsTheInterpolationStaysWithinTheToleranceOfTheArc) {
  // The solver interpolates the knots linearly, so between knots dpsi apart the
  // reference is the chord, rho (1 - cos(dpsi / 2)) off the arc of radius rho =
  // |v| / |omega|, and the velocity target |v_c| (1 - cos(dpsi / 2)) off its
  // turned value. Steady state at the shipped command limits (the first segment
  // of the Unitree G1 at 2.4 m/s and 1 rad/s over 0.84 s, of the EngineAI SA01
  // at 0.8 m/s over 0.7 s, the second segment of the G1) and the design
  // README's 0.3 m/s at 0.3 rad/s (section 2.9).
  struct Case {
    vector2_t initLinvelBaseRtWorldEwrtWorldXy = vector2_t::Zero();
    scalar_t rateYawBaseRtWorld = 0.0;
    scalar_t duration = 0.0;
  };
  for (const Case& segment :
       {Case{.initLinvelBaseRtWorldEwrtWorldXy = vector2_t(2.4, 1.2),
             .rateYawBaseRtWorld = 1.0,
             .duration = 0.84},
        Case{.initLinvelBaseRtWorldEwrtWorldXy = vector2_t(0.8, 0.2),
             .rateYawBaseRtWorld = -1.0,
             .duration = 0.7},
        Case{.initLinvelBaseRtWorldEwrtWorldXy = vector2_t(2.4, 0.0),
             .rateYawBaseRtWorld = 1.0,
             .duration = 0.36},
        Case{.initLinvelBaseRtWorldEwrtWorldXy = vector2_t(0.3, 0.0),
             .rateYawBaseRtWorld = 0.3,
             .duration = 0.84}}) {
    SCOPED_TRACE(segment.initLinvelBaseRtWorldEwrtWorldXy.transpose());
    const Extent extent{
        .yawChange = segment.rateYawBaseRtWorld * segment.duration,
        .arcLength =
            segment.initLinvelBaseRtWorldEwrtWorldXy.norm() * segment.duration,
        .targetSpeed = segment.initLinvelBaseRtWorldEwrtWorldXy.norm()};
    const size_t parts =
        TargetTrajectoriesCalculatorBase::knotSubdivisionCount(extent);
    const ChordDeviation cut =
        chordDeviation(segment.initLinvelBaseRtWorldEwrtWorldXy,
                       segment.initLinvelBaseRtWorldEwrtWorldXy,
                       segment.rateYawBaseRtWorld, segment.duration, parts);
    EXPECT_LE(
        cut.maxErrPosBaseRtWorldEwrtWorld,
        TargetTrajectoriesCalculatorBase::kKnotMaxErrPosBaseRtWorldEwrtWorld +
            1.0e-12);
    EXPECT_LE(cut.maxErrLinvelBaseRtWorldEwrtWorld,
              TargetTrajectoriesCalculatorBase::
                      kKnotMaxErrLinvelBaseRtWorldEwrtWorld +
                  1.0e-12);
    // The deviation is the documented one, at the middle of an interval.
    const scalar_t halfSpacing =
        0.5 * std::abs(extent.yawChange) / static_cast<scalar_t>(parts);
    EXPECT_NEAR(cut.maxErrPosBaseRtWorldEwrtWorld,
                segment.initLinvelBaseRtWorldEwrtWorldXy.norm() /
                    std::abs(segment.rateYawBaseRtWorld) *
                    (1.0 - std::cos(halfSpacing)),
                1.0e-9);
    EXPECT_NEAR(cut.maxErrLinvelBaseRtWorldEwrtWorld,
                segment.initLinvelBaseRtWorldEwrtWorldXy.norm() *
                    (1.0 - std::cos(halfSpacing)),
                1.0e-9);
    if (parts > 1) {
      // The fewest parts for the bounds: with one less, one of them is
      // exceeded.
      const scalar_t fewer = static_cast<scalar_t>(parts - 1);
      EXPECT_TRUE(extent.arcLength * std::abs(extent.yawChange) /
                          (8.0 * fewer * fewer) >
                      TargetTrajectoriesCalculatorBase::
                          kKnotMaxErrPosBaseRtWorldEwrtWorld ||
                  extent.targetSpeed * extent.yawChange * extent.yawChange /
                          (8.0 * fewer * fewer) >
                      TargetTrajectoriesCalculatorBase::
                          kKnotMaxErrLinvelBaseRtWorldEwrtWorld ||
                  std::abs(extent.yawChange) / fewer >
                      TargetTrajectoriesCalculatorBase::kMaxKnotYawSpacing);
    }
  }
  // Uncut, the G1's first segment at its limits is 0.21 m off the arc between
  // its two knots.
  const ChordDeviation uncut = chordDeviation(
      vector2_t(2.4, 0.0), vector2_t(2.4, 0.0), /*rateYawBaseRtWorld=*/1.0,
      /*duration=*/0.84, /*parts=*/1);
  EXPECT_NEAR(uncut.maxErrPosBaseRtWorldEwrtWorld, 2.4 * (1.0 - std::cos(0.42)),
              1.0e-9);
  EXPECT_GT(uncut.maxErrPosBaseRtWorldEwrtWorld, 0.2);
}

/** A body planar twist ramping linearly over `duration` from the measured to
 * the commanded one. Passive data. */
struct TwistRamp {
  scalar_t initYawBaseRtWorld = 0.0;
  vector2_t measLinvelBaseRtWorldEwrtHeadingXy = vector2_t::Zero();
  scalar_t measRateYawBaseRtWorld = 0.0;
  vector2_t cmdLinvelBaseRtWorldEwrtHeadingXy = vector2_t::Zero();
  scalar_t cmdRateYawBaseRtWorld = 0.0;
  scalar_t duration = 0.0;
};

/** The ramp's world velocity at `time` [s]: R_z(psi(s)) v(s), psi(s) = psi0 +
 * omega_m s + (omega_c - omega_m) s^2 / (2 tau). */
vector2_t computeRampLinvelBaseRtWorldEwrtWorldXy(const TwistRamp& ramp,
                                                  scalar_t time) {
  const scalar_t fraction = time / ramp.duration;
  const scalar_t yawBaseRtWorld =
      ramp.initYawBaseRtWorld + ramp.measRateYawBaseRtWorld * time +
      0.5 * (ramp.cmdRateYawBaseRtWorld - ramp.measRateYawBaseRtWorld) *
          fraction * time;
  return rotated(
      yawBaseRtWorld,
      vector2_t(ramp.measLinvelBaseRtWorldEwrtHeadingXy +
                fraction * (ramp.cmdLinvelBaseRtWorldEwrtHeadingXy -
                            ramp.measLinvelBaseRtWorldEwrtHeadingXy)));
}

/** p(tau) - p(0) of the ramp, by RK4 with 20000 steps. */
vector2_t rampedDisplacement(const TwistRamp& ramp) {
  constexpr int kSteps = 20000;
  const scalar_t step = ramp.duration / kSteps;
  vector2_t posBaseRtWorldEwrtWorldXy = vector2_t::Zero();
  for (int i = 0; i < kSteps; ++i) {
    const scalar_t time = i * step;
    posBaseRtWorldEwrtWorldXy +=
        step / 6.0 *
        (computeRampLinvelBaseRtWorldEwrtWorldXy(ramp, time) +
         4.0 *
             computeRampLinvelBaseRtWorldEwrtWorldXy(ramp, time + 0.5 * step) +
         computeRampLinvelBaseRtWorldEwrtWorldXy(ramp, time + step));
  }
  return posBaseRtWorldEwrtWorldXy;
}

TEST(VelocityCommandArc,
     TheHeldMeanTwistDiffersFromALinearRampByTheDocumentedLeadingTerm) {
  // The first segment holds the mean twist instead of ramping from the measured
  // to the commanded one. Its end heading is the ramp's; its position differs
  // by (tau^2 / 12) R_z(psi0) J (omega_m v_c - omega_c v_m) + O(tau^3 omega^2
  // |v|), J = R_z(pi / 2): halving the rates halves the term and quarters the
  // remainder. A steady turn whose speed changes (omega_m = omega_c) is among
  // the cases: the difference is not zero there.
  const TwistRamp steadyTurn{
      .initYawBaseRtWorld = 0.4,
      .measLinvelBaseRtWorldEwrtHeadingXy = vector2_t(0.15, 0.0),
      .measRateYawBaseRtWorld = 0.5,
      .cmdLinvelBaseRtWorldEwrtHeadingXy = vector2_t(0.3, 0.0),
      .cmdRateYawBaseRtWorld = 0.5,
      .duration = 0.84};
  const TwistRamp slowingTurn{
      .initYawBaseRtWorld = 0.4,
      .measLinvelBaseRtWorldEwrtHeadingXy = vector2_t(0.2, -0.1),
      .measRateYawBaseRtWorld = 0.6,
      .cmdLinvelBaseRtWorldEwrtHeadingXy = vector2_t(0.3, 0.1),
      .cmdRateYawBaseRtWorld = 0.18,
      .duration = 0.84};
  for (const TwistRamp& motion : {steadyTurn, slowingTurn}) {
    SCOPED_TRACE(motion.measRateYawBaseRtWorld);
    scalar_t previousRemainder = 0.0;
    for (const scalar_t scale : {1.0, 0.5, 0.25}) {
      SCOPED_TRACE(scale);
      TwistRamp ramp = motion;
      ramp.measRateYawBaseRtWorld *= scale;
      ramp.cmdRateYawBaseRtWorld *= scale;
      const vector2_t held =
          TargetTrajectoriesCalculatorBase::planarArcDisplacement(
              rotated(ramp.initYawBaseRtWorld,
                      0.5 * (ramp.measLinvelBaseRtWorldEwrtHeadingXy +
                             ramp.cmdLinvelBaseRtWorldEwrtHeadingXy)),
              0.5 * (ramp.measRateYawBaseRtWorld + ramp.cmdRateYawBaseRtWorld),
              ramp.duration);
      const vector2_t leadingTerm =
          ramp.duration * ramp.duration / 12.0 *
          rotated(ramp.initYawBaseRtWorld + 0.5 * M_PI,
                  vector2_t(ramp.measRateYawBaseRtWorld *
                                ramp.cmdLinvelBaseRtWorldEwrtHeadingXy -
                            ramp.cmdRateYawBaseRtWorld *
                                ramp.measLinvelBaseRtWorldEwrtHeadingXy));
      const scalar_t remainder =
          (rampedDisplacement(ramp) - held - leadingTerm).norm();
      EXPECT_GT(leadingTerm.norm(), 1.0e-4);
      EXPECT_LT(remainder, 0.3 * leadingTerm.norm());
      if (previousRemainder > 0.0) {
        EXPECT_NEAR(remainder / previousRemainder, 0.25, 0.01);
      }
      previousRemainder = remainder;
    }
  }
  // The class comment's steady turn, 0.5 rad/s from 0.15 to 0.3 m/s over 0.84
  // s: 4.4 mm.
  const vector2_t held =
      TargetTrajectoriesCalculatorBase::planarArcDisplacement(
          rotated(steadyTurn.initYawBaseRtWorld, vector2_t(0.225, 0.0)),
          steadyTurn.measRateYawBaseRtWorld, steadyTurn.duration);
  EXPECT_NEAR((rampedDisplacement(steadyTurn) - held).norm(), 4.39e-3, 1.0e-5);
}

}  // namespace
}  // namespace ocs2::humanoid
