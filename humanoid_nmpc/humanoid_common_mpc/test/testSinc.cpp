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

// sinc(): sin(x) / x away from zero, its two-term series near it, and
// continuous across the threshold between them.

#include <cmath>
#include <limits>

#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/Sinc.h"

namespace ocs2::humanoid {
namespace {

TEST(Sinc, IsOneAtZero) {
  EXPECT_EQ(sinc(0.0), 1.0);
}

TEST(Sinc, IsTheQuotientAwayFromZero) {
  for (const scalar_t x : {-3.0, -0.5, 1.0e-3, 0.25, 1.0, 2.0, 10.0}) {
    EXPECT_DOUBLE_EQ(sinc(x), std::sin(x) / x) << "x = " << x;
  }
}

TEST(Sinc, IsTheTwoTermSeriesNearZero) {
  for (const scalar_t x : {-5.0e-5, -1.0e-8, 1.0e-12, 3.0e-5, 9.9e-5}) {
    EXPECT_DOUBLE_EQ(sinc(x), 1.0 - x * x / 6.0) << "x = " << x;
    // The series is the quotient to within the next term, x^4 / 5!, below
    // 1e-18.
    EXPECT_NEAR(sinc(x), std::sin(x) / x, 1.0e-15) << "x = " << x;
  }
}

TEST(Sinc, IsEvenAndContinuousAtTheThreshold) {
  EXPECT_DOUBLE_EQ(sinc(0.3), sinc(-0.3));
  // An |x| just below the threshold, where sinc() is still the series.
  const scalar_t below =
      kSincSeriesThreshold * (1.0 - std::numeric_limits<scalar_t>::epsilon());
  EXPECT_NEAR(sinc(below), sinc(kSincSeriesThreshold), 1.0e-15);
}

}  // namespace
}  // namespace ocs2::humanoid
