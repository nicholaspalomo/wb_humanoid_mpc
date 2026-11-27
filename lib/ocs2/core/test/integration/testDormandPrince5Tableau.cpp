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

// The Dormand-Prince 5(4) tableau both adaptive steppers share (DormandPrince5Tableau.h): the published coefficients,
// and the consistency conditions of a Runge-Kutta tableau they satisfy to round-off.

#include "gtest/gtest.h"
#include "ocs2_core/integration/DormandPrince5Tableau.h"

namespace ocs2::dormand_prince5 {
namespace {

// A sum of a few coefficients of magnitude below 12 carries a few ulps of round-off.
constexpr scalar_t kRoundOff = 1.0e-14;

TEST(DormandPrince5Tableau, HasThePublishedCoefficients) {
  EXPECT_EQ(kA2, 0.2);
  EXPECT_EQ(kA4, 0.8);
  EXPECT_EQ(kB42, -56.0 / 15.0);
  EXPECT_EQ(kB65, -5103.0 / 18656.0);
  EXPECT_EQ(kC1, 35.0 / 384.0);
  EXPECT_EQ(kDc7, -0.025);
  EXPECT_EQ(kStepSizeSafetyFactor, 0.9);
  EXPECT_EQ(kMinStepDecreaseFactor, 0.2);
  EXPECT_EQ(kStepIncreaseErrorThreshold, 0.5);
  EXPECT_EQ(kStepIncreaseLimit, 5.0);
}

TEST(DormandPrince5Tableau, EachStageIsEvaluatedAtTheSumOfItsWeights) {
  EXPECT_NEAR(kB21, kA2, kRoundOff);
  EXPECT_NEAR(kB31 + kB32, kA3, kRoundOff);
  EXPECT_NEAR(kB41 + kB42 + kB43, kA4, kRoundOff);
  EXPECT_NEAR(kB51 + kB52 + kB53 + kB54, kA5, kRoundOff);
  // Stage 6 is evaluated at the end of the step.
  EXPECT_NEAR(kB61 + kB62 + kB63 + kB64 + kB65, 1.0, kRoundOff);
}

TEST(DormandPrince5Tableau, BothSolutionsAreConsistent) {
  // The fifth-order weights sum to one, and so do the fourth-order ones: their differences sum to zero.
  EXPECT_NEAR(kC1 + kC3 + kC4 + kC5 + kC6, 1.0, kRoundOff);
  EXPECT_NEAR(kDc1 + kDc3 + kDc4 + kDc5 + kDc6 + kDc7, 0.0, kRoundOff);
}

}  // namespace
}  // namespace ocs2::dormand_prince5
