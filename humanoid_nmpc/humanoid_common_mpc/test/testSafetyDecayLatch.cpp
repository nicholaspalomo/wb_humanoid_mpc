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

#include <random>

#include "gtest/gtest.h"

#include "humanoid_common_mpc/mrt/SafetyDecay.h"

/*
 * The SAFETY decay as the MRT joint controllers apply it (safety_decay::latchedFactor()): alpha never rises within one
 * stay in the mode, whatever the observation clock and the time constant do - a clock that steps back by less than
 * the rewind tolerance (which the controllers do not compensate), a time constant made longer mid-decay, or both - and
 * once it has reached zero torque it stays there. factor() alone rises in both cases: the positive control.
 */

namespace ocs2::humanoid {
namespace {

TEST(SafetyDecayLatch, NeverRisesUnderClockStepsBackAndLongerTimeConstants) {
  std::mt19937 generator(/*sd=*/23);
  std::uniform_real_distribution<scalar_t> step(-0.004, 0.01);  // [s] per cycle; back by up to 4 ms
  std::uniform_real_distribution<scalar_t> timeConstant(0.1, 5.0);
  bool unlatchedRose = false;
  for (int run = 0; run < 50; ++run) {
    scalar_t elapsed = 0.0;
    scalar_t tau = timeConstant(generator);
    scalar_t alpha = 1.0;
    scalar_t previousUnlatched = 1.0;
    for (int cycle = 0; cycle < 2000; ++cycle) {
      elapsed += step(generator);
      if (cycle % 97 == 0) tau = timeConstant(generator);
      const scalar_t latched = safety_decay::latchedFactor(alpha, elapsed, tau);
      EXPECT_LE(latched, alpha);
      EXPECT_GE(latched, 0.0);
      EXPECT_LE(latched, safety_decay::factor(elapsed, tau));
      const scalar_t unlatched = safety_decay::factor(elapsed, tau);
      if (unlatched > previousUnlatched) unlatchedRose = true;
      previousUnlatched = unlatched;
      alpha = latched;
    }
  }
  EXPECT_TRUE(unlatchedRose) << "the positive control: the bare factor rises in these runs";
}

TEST(SafetyDecayLatch, ZeroTorqueStaysZeroWhenTheTimeConstantGrows) {
  const scalar_t tau = 0.5;
  scalar_t alpha = safety_decay::latchedFactor(/*previous=*/1.0, /*elapsedSinceEntry=*/10.0 * tau, tau);
  ASSERT_EQ(alpha, 0.0);
  // Ten times the time constant at the same elapsed time: the bare factor is back above the cutoff.
  ASSERT_GT(safety_decay::factor(10.0 * tau, 10.0 * tau), safety_decay::kCutoff);
  alpha = safety_decay::latchedFactor(alpha, 10.0 * tau, 10.0 * tau);
  EXPECT_EQ(alpha, 0.0);
}

TEST(SafetyDecayLatch, FollowsTheBareFactorOnAMonotoneClock) {
  const scalar_t tau = 0.4;
  scalar_t alpha = 1.0;
  for (int cycle = 0; cycle <= 1000; ++cycle) {
    const scalar_t elapsed = 0.002 * cycle;
    alpha = safety_decay::latchedFactor(alpha, elapsed, tau);
    EXPECT_EQ(alpha, safety_decay::factor(elapsed, tau));
  }
}

}  // namespace
}  // namespace ocs2::humanoid
