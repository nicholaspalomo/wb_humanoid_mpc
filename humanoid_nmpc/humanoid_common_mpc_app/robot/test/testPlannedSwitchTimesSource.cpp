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

// PlannedSwitchTimesSource: the controller's planned switch times written into
// the state estimator's contact inputs, cleared without a policy, and no heap
// allocation.

#include <cstddef>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc_app/robot/PlannedSwitchTimesSource.h"
#include "humanoid_common_mpc_app/robot/RobotController.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/RobotTestSupport.h"
#include "humanoid_state_estimator/ContactEstimationInputs.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kUnknown = std::numeric_limits<scalar_t>::infinity();
constexpr size_t kNumFeet = 2;

TEST(PlannedSwitchTimesSourceTest, WritesTheControllersTimes) {
  robot_test::ScriptedRobotController controller;
  controller.setPlannedContactSwitchTimes(/*timeSinceSwitch=*/{0.1, kUnknown},
                                          /*timeToSwitch=*/{0.2, 0.05});
  estimation::ContactEstimationInputs inputs(kNumFeet);
  PlannedSwitchTimesSource source(kNumFeet);
  source.fill(controller, inputs);
  EXPECT_EQ(inputs.plannedTimeSinceSwitch(0), 0.1);
  EXPECT_EQ(inputs.plannedTimeToSwitch(0), 0.2);
  EXPECT_EQ(inputs.plannedTimeSinceSwitch(1), kUnknown);
  EXPECT_EQ(inputs.plannedTimeToSwitch(1), 0.05);
}

TEST(PlannedSwitchTimesSourceTest, WithoutAPolicyNoSwitchIsKnown) {
  robot_test::ScriptedRobotController controller;  // answers false
  estimation::ContactEstimationInputs inputs(kNumFeet);
  inputs.setPlannedContactSwitchTimes(/*contact=*/0, /*timeSinceSwitch=*/0.1,
                                      /*timeToSwitch=*/0.2);
  PlannedSwitchTimesSource source(kNumFeet);
  source.fill(controller, inputs);
  for (size_t contact = 0; contact < kNumFeet; ++contact) {
    EXPECT_EQ(inputs.plannedTimeSinceSwitch(contact), kUnknown);
    EXPECT_EQ(inputs.plannedTimeToSwitch(contact), kUnknown);
  }
}

TEST(PlannedSwitchTimesSourceTest,
     AControllerOfFewerContactsLeavesTheRestUnknown) {
  robot_test::ScriptedRobotController controller;
  controller.setPlannedContactSwitchTimes(/*timeSinceSwitch=*/{0.1},
                                          /*timeToSwitch=*/{0.2});
  estimation::ContactEstimationInputs inputs(kNumFeet);
  PlannedSwitchTimesSource source(kNumFeet);
  source.fill(controller, inputs);
  EXPECT_EQ(inputs.plannedTimeToSwitch(0), 0.2);
  EXPECT_EQ(inputs.plannedTimeToSwitch(1), kUnknown);
}

TEST(PlannedSwitchTimesSourceTest, FillAllocatesNothing) {
  robot_test::ScriptedRobotController withTimes;
  withTimes.setPlannedContactSwitchTimes(/*timeSinceSwitch=*/{0.1, 0.3},
                                         /*timeToSwitch=*/{0.2, 0.4});
  const robot_test::ScriptedRobotController withoutTimes;
  const RobotController& answering = withTimes;
  const RobotController& notAnswering = withoutTimes;
  estimation::ContactEstimationInputs inputs(kNumFeet);
  PlannedSwitchTimesSource source(kNumFeet);
  const size_t before = robot::realtime::heapAllocationCountOnThisThread();
  for (int cycle = 0; cycle < 100; ++cycle) {
    source.fill(cycle % 2 == 0 ? answering : notAnswering, inputs);
  }
  EXPECT_EQ(robot::realtime::heapAllocationCountOnThisThread() - before, 0u);
}

}  // namespace
}  // namespace ocs2::humanoid
