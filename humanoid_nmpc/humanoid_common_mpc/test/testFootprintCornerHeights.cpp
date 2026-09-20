/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <gtest/gtest.h>

#include <cmath>
#include <initializer_list>
#include <limits>

#include "humanoid_common_mpc/contact/FootprintCornerHeights.h"

namespace ocs2::humanoid {
namespace {

/** [m] the shipped smoothing length, so the numbers below are the ones the robot actually runs with. */
constexpr scalar_t kSmoothing = 1.0e-3;

vector_t heights(std::initializer_list<scalar_t> values) {
  vector_t result(static_cast<long>(values.size()));
  long index = 0;
  for (scalar_t value : values) {
    result(index++) = value;
  }
  return result;
}

TEST(SmoothMinimumHeight, isExactOnAFlatFoot) {
  // THE property the 1/N normalisation buys, and the reason the plain log-sum-exp softmin could not be used. A flat
  // foot is where a walking robot spends most of its stance; an un-normalised softmin reports it as log(4)*s = 1.39 mm
  // LOWER than it is, and since the complementarity penalty is two-sided the solver answers a negative reported gap by
  // lifting the foot. The robot would hover under full load and never close the contact.
  for (scalar_t height : {-0.02, 0.0, 0.05, 1.3}) {
    const SmoothMinimumHeight result = smoothMinimumHeight(heights({height, height, height, height}), kSmoothing);
    EXPECT_NEAR(result.value, height, 1e-15) << "at " << height;
    // And exact in the gradient too: every corner carries the same share, so the state Jacobian is their plain mean.
    for (long corner = 0; corner < result.weights.size(); ++corner) {
      EXPECT_NEAR(result.weights(corner), 0.25, 1e-15) << "corner " << corner;
    }
  }
}

TEST(SmoothMinimumHeight, neverReportsLessClearanceThanTheLowestCorner) {
  // One-sided in the safe direction: over-reporting the gap only asks the solver to take load off a foot that is up on
  // an edge, whereas under-reporting it moves the penalty's zero below the ground and buys a permanent hover.
  const vector_t samples[] = {heights({0.0, 0.0, 0.0, 0.0}),    heights({0.0, 0.001, 0.002, 0.003}), heights({0.0, 0.02, 0.02, 0.04}),
                              heights({0.05, 0.05, 0.05, 0.2}), heights({-0.004, 0.0, 0.01, 0.03}),  heights({0.1, 0.1, 0.1, 0.1000001})};
  for (const vector_t& sample : samples) {
    const SmoothMinimumHeight result = smoothMinimumHeight(sample, kSmoothing);
    EXPECT_GE(result.value, sample.minCoeff() - 1e-15);
    EXPECT_LE(result.value, sample.minCoeff() + std::log(static_cast<scalar_t>(sample.size())) * kSmoothing + 1e-15);
  }
}

TEST(SmoothMinimumHeight, theBiasIsLogOfTheFractionOfCornersThatTouch) {
  // The bias is not one number, it is s * log(N / k) where k is how many corners sit at the minimum. Spelling out the
  // whole family matters because the two cases that actually occur are NOT the worst case, and an earlier version of
  // this file quoted the two-corner number as if it were the bound - which made the assertion below unsatisfiable.
  //
  //   k = 4, a flat foot:                       exact.
  //   k = 2, an EDGE down - the ordinary heel strike or toe-off of a rectangular sole: s * log 2 = 0.693 mm.
  //   k = 1, a single corner, which needs pitch AND roll at once: s * log 4 = 1.386 mm. This is the bound.
  //
  // Against the shipped penetration hinge (penetrationWeight 5e4 against a complementarity curvature of
  // complementarityWeight / heightReference^2 = 7812 at full body weight) those become 0.094 mm and 0.187 mm of
  // equilibrium penetration respectively - the price of a differentiable minimum.
  const SmoothMinimumHeight flat = smoothMinimumHeight(heights({0.0, 0.0, 0.0, 0.0}), kSmoothing);
  EXPECT_NEAR(flat.value, 0.0, 1e-15);

  const SmoothMinimumHeight edge = smoothMinimumHeight(heights({0.0, 0.0, 0.05, 0.05}), kSmoothing);
  EXPECT_NEAR(edge.value, std::log(2.0) * kSmoothing, 1e-9);
  // Each of the two touching corners carries half the gradient; the raised pair carry none.
  EXPECT_NEAR(edge.weights(0), 0.5, 1e-9);
  EXPECT_NEAR(edge.weights(1), 0.5, 1e-9);
  EXPECT_NEAR(edge.weights(2), 0.0, 1e-9);

  const SmoothMinimumHeight corner = smoothMinimumHeight(heights({0.0, 0.05, 0.05, 0.05}), kSmoothing);
  EXPECT_NEAR(corner.value, std::log(4.0) * kSmoothing, 1e-9);
  EXPECT_LT(corner.value, 1.5e-3) << "the worst-case error must stay well under two millimetres";
  // The touching corner carries essentially the whole gradient: 0.05 m is fifty smoothing lengths away.
  EXPECT_NEAR(corner.weights(0), 1.0, 1e-9);

  // And the ordering is monotone: fewer corners down means more bias, never less.
  EXPECT_LT(flat.value, edge.value);
  EXPECT_LT(edge.value, corner.value);
}

TEST(SmoothMinimumHeight, weightsAreAConvexCombinationThatFavoursTheLowestCorner) {
  const vector_t sample = heights({0.004, 0.0, 0.002, 0.010});
  const SmoothMinimumHeight result = smoothMinimumHeight(sample, kSmoothing);
  ASSERT_EQ(result.weights.size(), sample.size());
  EXPECT_NEAR(result.weights.sum(), 1.0, 1e-12) << "the gap moves one-for-one with a rigid translation of the foot";
  EXPECT_TRUE((result.weights.array() >= 0.0).all());
  // Monotone in the height: a lower corner always carries at least as much of the gradient as a higher one.
  EXPECT_GT(result.weights(1), result.weights(2));
  EXPECT_GT(result.weights(2), result.weights(0));
  EXPECT_GT(result.weights(0), result.weights(3));
}

TEST(SmoothMinimumHeight, weightsAreItsGradient) {
  // The weights are handed straight to the solver as d(gap)/d(corner height), so a value and a gradient that do not
  // belong to the same function would be a silently wrong linearisation rather than a visibly wrong number.
  const vector_t sample = heights({0.004, 0.0, 0.002, 0.010});
  const SmoothMinimumHeight result = smoothMinimumHeight(sample, kSmoothing);
  constexpr scalar_t kStep = 1.0e-7;
  for (long corner = 0; corner < sample.size(); ++corner) {
    vector_t perturbed = sample;
    perturbed(corner) += kStep;
    const scalar_t forward = smoothMinimumHeight(perturbed, kSmoothing).value;
    perturbed(corner) -= 2.0 * kStep;
    const scalar_t backward = smoothMinimumHeight(perturbed, kSmoothing).value;
    EXPECT_NEAR(result.weights(corner), (forward - backward) / (2.0 * kStep), 1e-6) << "corner " << corner;
  }
}

TEST(SmoothMinimumHeight, approachesTheTrueMinimumAsTheSmoothingShrinks) {
  const vector_t sample = heights({0.0, 0.05, 0.05, 0.05});
  scalar_t previousError = std::numeric_limits<scalar_t>::max();
  for (scalar_t smoothing : {1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5}) {
    const scalar_t error = smoothMinimumHeight(sample, smoothing).value - sample.minCoeff();
    EXPECT_GE(error, 0.0);
    EXPECT_LT(error, previousError);
    previousError = error;
  }
  EXPECT_LT(previousError, 1.0e-4);
}

TEST(SmoothMinimumHeight, survivesCornersFarEnoughApartToUnderflow) {
  // exp(-(h_i - m)/s) underflows to zero for a corner a metre up at a millimetre of smoothing. The shift by the true
  // minimum keeps the touching corner's term at exactly one, so the sum can never be empty and the logarithm never
  // sees zero - which would have come back as an infinite gap and an infinite complementarity residual.
  const SmoothMinimumHeight result = smoothMinimumHeight(heights({0.0, 1.0, 2.0, 5.0}), kSmoothing);
  EXPECT_TRUE(std::isfinite(result.value));
  EXPECT_NEAR(result.value, std::log(4.0) * kSmoothing, 1e-12);
  EXPECT_NEAR(result.weights(0), 1.0, 1e-12);
  EXPECT_NEAR(result.weights.sum(), 1.0, 1e-12);
}

TEST(SmoothMinimumHeight, isTranslationEquivariant) {
  // The gap of a foot lifted by a centimetre is a centimetre larger, exactly. Without the shift-by-the-minimum this
  // would be the first thing to break at large heights.
  const vector_t sample = heights({0.004, 0.0, 0.002, 0.010});
  const scalar_t base = smoothMinimumHeight(sample, kSmoothing).value;
  const vector_t lifted = sample.array() + 0.01;
  EXPECT_NEAR(smoothMinimumHeight(lifted, kSmoothing).value, base + 0.01, 1e-12);
}

TEST(SmoothMinimumHeight, handlesASinglePoint) {
  const SmoothMinimumHeight result = smoothMinimumHeight(heights({0.037}), kSmoothing);
  EXPECT_NEAR(result.value, 0.037, 1e-15) << "with one point there is nothing to smooth";
  ASSERT_EQ(result.weights.size(), 1);
  EXPECT_NEAR(result.weights(0), 1.0, 1e-15);
}

TEST(SmoothMinimumHeight, rejectsANonPositiveSmoothing) {
  EXPECT_DEATH(smoothMinimumHeight(heights({0.0, 0.0, 0.0, 0.0}), 0.0), "gapSmoothing");
  EXPECT_DEATH(smoothMinimumHeight(heights({0.0, 0.0, 0.0, 0.0}), -1.0e-3), "gapSmoothing");
}

}  // namespace
}  // namespace ocs2::humanoid
