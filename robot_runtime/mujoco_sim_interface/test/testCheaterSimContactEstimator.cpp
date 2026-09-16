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

#include <gtest/gtest.h>

#include "mujoco_sim_interface/CheaterSimContactEstimator.h"

using robot::mujoco_sim_interface::CheaterSimContactEstimator;

TEST(CheaterSimContactEstimator, flagsFollowTheGroundTruthMask) {
  EXPECT_EQ(CheaterSimContactEstimator::flagsFromMasks(0b01u, 0u, 2, 2), (std::vector<bool>{true, false}));
  EXPECT_EQ(CheaterSimContactEstimator::flagsFromMasks(0b10u, 0u, 2, 2), (std::vector<bool>{false, true}));
  EXPECT_EQ(CheaterSimContactEstimator::flagsFromMasks(0b11u, 0u, 2, 2), (std::vector<bool>{true, true}));
  EXPECT_EQ(CheaterSimContactEstimator::flagsFromMasks(0b00u, 0u, 2, 2), (std::vector<bool>{false, false}));
}

TEST(CheaterSimContactEstimator, unresolvedContactPointsAreReportedAsTouching) {
  // Contact point 1 has no MuJoCo body: it keeps its planned wrenches, like the RobotState flags of the simulator.
  EXPECT_EQ(CheaterSimContactEstimator::flagsFromMasks(0b00u, 0b10u, 2, 2), (std::vector<bool>{false, true}));
}

TEST(CheaterSimContactEstimator, contactPointsBeyondTheDetectedOnesAreReportedAsTouching) {
  EXPECT_EQ(CheaterSimContactEstimator::flagsFromMasks(0b0u, 0u, 1, 2), (std::vector<bool>{false, true}));
}

TEST(CheaterSimContactEstimator, withoutContactDetectionEveryContactPointIsTouching) {
  EXPECT_EQ(CheaterSimContactEstimator::flagsFromMasks(0u, 0u, 0, 2), (std::vector<bool>{true, true}));
  EXPECT_TRUE(CheaterSimContactEstimator::flagsFromMasks(0u, 0u, 0, 0).empty());
}
