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

// contactTrustScale(): the factor of KalmanBaseEstimator's contact-trust ramp
// on a stance contact's noise near its planned switch.

#include <cmath>
#include <limits>

#include "gtest/gtest.h"
#include "ocs2_core/Types.h"

#include "legged_state_estimator/ContactTrustRamp.h"

namespace ocs2::legged::estimation {
namespace {

constexpr scalar_t kRampDuration = 0.05;
constexpr scalar_t kMaxScale = 4.0;

TEST(ContactTrustRampTest, IsOneOutsideTheWindowAndWhenOff) {
  EXPECT_EQ(contactTrustScale(kRampDuration, kRampDuration, kMaxScale), 1.0);
  EXPECT_EQ(
      contactTrustScale(/*timeToNearestSwitch=*/1.0, kRampDuration, kMaxScale),
      1.0);
  // Off: no window, whatever the distance.
  EXPECT_EQ(contactTrustScale(/*timeToNearestSwitch=*/0.0, /*rampDuration=*/0.0,
                              kMaxScale),
            1.0);
  EXPECT_EQ(contactTrustScale(/*timeToNearestSwitch=*/0.01,
                              /*rampDuration=*/0.0, kMaxScale),
            1.0);
}

TEST(ContactTrustRampTest, NoKnownSwitchIsOne) {
  EXPECT_EQ(contactTrustScale(std::numeric_limits<scalar_t>::infinity(),
                              kRampDuration, kMaxScale),
            1.0);
  EXPECT_EQ(contactTrustScale(std::nan(""), kRampDuration, kMaxScale), 1.0);
}

TEST(ContactTrustRampTest, IsTheMaximumAtTheSwitch) {
  EXPECT_EQ(
      contactTrustScale(/*timeToNearestSwitch=*/0.0, kRampDuration, kMaxScale),
      kMaxScale);
  // A switch already passed counts as the switch itself.
  EXPECT_EQ(contactTrustScale(/*timeToNearestSwitch=*/-0.01, kRampDuration,
                              kMaxScale),
            kMaxScale);
  // Half-way the smoothstep is one half.
  EXPECT_DOUBLE_EQ(
      contactTrustScale(0.5 * kRampDuration, kRampDuration, kMaxScale),
      1.0 + 0.5 * (kMaxScale - 1.0));
  // A maximum of 1 is no ramp.
  EXPECT_EQ(contactTrustScale(/*timeToNearestSwitch=*/0.0, kRampDuration,
                              /*maxScale=*/1.0),
            1.0);
}

TEST(ContactTrustRampTest, DecreasesMonotonicallyAndIsSmoothAtTheEdges) {
  constexpr int kSamples = 1000;
  scalar_t previous =
      contactTrustScale(/*timeToNearestSwitch=*/0.0, kRampDuration, kMaxScale);
  for (int sample = 1; sample <= kSamples; ++sample) {
    const scalar_t distance = kRampDuration * sample / kSamples;
    const scalar_t scale =
        contactTrustScale(distance, kRampDuration, kMaxScale);
    EXPECT_LE(scale, previous) << "at d = " << distance;
    EXPECT_GE(scale, 1.0);
    previous = scale;
  }
  // C1: the one-sided slopes at the window's edge and at the switch vanish;
  // the difference quotient over kStep is 9 kStep / duration^2 there, against
  // about 90 / s in the middle.
  constexpr scalar_t kStep = 1.0e-6;
  const scalar_t slopeInsideTheEdge =
      (contactTrustScale(kRampDuration - kStep, kRampDuration, kMaxScale) -
       1.0) /
      kStep;
  EXPECT_LT(std::abs(slopeInsideTheEdge), 1.0e-2);
  const scalar_t slopeAtTheSwitch =
      (kMaxScale - contactTrustScale(kStep, kRampDuration, kMaxScale)) / kStep;
  EXPECT_LT(std::abs(slopeAtTheSwitch), 1.0e-2);
  // While the slope in the middle is of the order (maxScale - 1) / duration.
  const scalar_t slopeInTheMiddle =
      (contactTrustScale(0.5 * kRampDuration - kStep, kRampDuration,
                         kMaxScale) -
       contactTrustScale(0.5 * kRampDuration + kStep, kRampDuration,
                         kMaxScale)) /
      (2.0 * kStep);
  EXPECT_NEAR(slopeInTheMiddle, 1.5 * (kMaxScale - 1.0) / kRampDuration, 1.0);
}

}  // namespace
}  // namespace ocs2::legged::estimation
