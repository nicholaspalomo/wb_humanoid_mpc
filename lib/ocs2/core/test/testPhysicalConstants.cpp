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

// The constants of ocs2_core/PhysicalConstants.h are the literals they replaced: the code that names a 9.81 or a 1.0e3
// with them must compute what it computed with the literal, so each keeps the value and the type (double) of that
// literal.

#include <type_traits>

#include "gtest/gtest.h"
#include "ocs2_core/PhysicalConstants.h"

namespace ocs2 {
namespace {

// The type of a floating-point literal such as 9.81, which the constants replace.
static_assert(std::is_same_v<decltype(9.81), double>);
static_assert(std::is_same_v<decltype(kGravityMagnitude), const double>);
static_assert(std::is_same_v<decltype(kMillisecondsPerSecond), const double>);
static_assert(std::is_same_v<decltype(kMicrosecondsPerSecond), const double>);
static_assert(std::is_same_v<decltype(kNanosecondsPerSecond), const double>);

TEST(PhysicalConstants, EveryConstantHasTheValueOfTheLiteralItNames) {
  EXPECT_EQ(kGravityMagnitude, 9.81);
  EXPECT_EQ(kMillisecondsPerSecond, 1.0e3);
  EXPECT_EQ(kMicrosecondsPerSecond, 1.0e6);
  EXPECT_EQ(kNanosecondsPerSecond, 1.0e9);
}

TEST(PhysicalConstants, TheTimeFactorsAreExactIntegers) {
  // The integer literals some of them replaced (1'000'000 microseconds) convert to these doubles exactly.
  EXPECT_EQ(kMillisecondsPerSecond, static_cast<double>(1'000));
  EXPECT_EQ(kMicrosecondsPerSecond, static_cast<double>(1'000'000));
  EXPECT_EQ(kNanosecondsPerSecond, static_cast<double>(1'000'000'000));
}

}  // namespace
}  // namespace ocs2
