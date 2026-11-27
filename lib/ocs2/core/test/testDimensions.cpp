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

// The constants of ocs2_core/Dimensions.h are the literals they replaced: the code that names a 3, a 4 or a 6 with
// them must compute what it computed with the literal, so each keeps the value and the type (int) of that literal.

#include <type_traits>

#include "gtest/gtest.h"
#include "ocs2_core/Dimensions.h"

namespace ocs2 {
namespace {

// The type of an integer literal such as 3, which the constants replace.
static_assert(std::is_same_v<decltype(3), int>);
static_assert(std::is_same_v<decltype(kPositionDim), const int>);
static_assert(std::is_same_v<decltype(kHorizontalPositionDim), const int>);
static_assert(std::is_same_v<decltype(kXAxisIndex), const int>);
static_assert(std::is_same_v<decltype(kYAxisIndex), const int>);
static_assert(std::is_same_v<decltype(kZAxisIndex), const int>);
static_assert(std::is_same_v<decltype(kRotationTangentDim), const int>);
static_assert(std::is_same_v<decltype(kQuaternionCoefficients), const int>);
static_assert(std::is_same_v<decltype(kQuaternionXIndex), const int>);
static_assert(std::is_same_v<decltype(kQuaternionYIndex), const int>);
static_assert(std::is_same_v<decltype(kQuaternionZIndex), const int>);
static_assert(std::is_same_v<decltype(kQuaternionWIndex), const int>);
static_assert(std::is_same_v<decltype(kForceDim), const int>);
static_assert(std::is_same_v<decltype(kTorqueDim), const int>);
static_assert(std::is_same_v<decltype(kWrenchDim), const int>);
static_assert(std::is_same_v<decltype(kTwistDim), const int>);
static_assert(std::is_same_v<decltype(kPoseCoefficients), const int>);
static_assert(std::is_same_v<decltype(kPoseTangentDim), const int>);

TEST(Dimensions, EveryConstantHasTheValueOfTheLiteralItNames) {
  EXPECT_EQ(kPositionDim, 3);
  EXPECT_EQ(kHorizontalPositionDim, 2);
  EXPECT_EQ(kXAxisIndex, 0);
  EXPECT_EQ(kYAxisIndex, 1);
  EXPECT_EQ(kZAxisIndex, 2);
  EXPECT_EQ(kRotationTangentDim, 3);
  EXPECT_EQ(kQuaternionCoefficients, 4);
  EXPECT_EQ(kForceDim, 3);
  EXPECT_EQ(kTorqueDim, 3);
  EXPECT_EQ(kWrenchDim, 6);
  EXPECT_EQ(kTwistDim, 6);
  EXPECT_EQ(kPoseCoefficients, 7);
  EXPECT_EQ(kPoseTangentDim, 6);
}

TEST(Dimensions, TheQuaternionIndicesAreTheStorageOrderXyzw) {
  EXPECT_EQ(kQuaternionXIndex, 0);
  EXPECT_EQ(kQuaternionYIndex, 1);
  EXPECT_EQ(kQuaternionZIndex, 2);
  EXPECT_EQ(kQuaternionWIndex, 3);
  EXPECT_EQ(kQuaternionWIndex + 1, kQuaternionCoefficients);
}

}  // namespace
}  // namespace ocs2
