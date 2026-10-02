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

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

#include "humanoid_common_mpc/mrt/ControllerEvent.h"
#include "humanoid_common_mpc/mrt/ControllerEventSink.h"
#include "humanoid_common_mpc/mrt/LoggingControllerEventSink.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"

/*
 * The MRT joint controllers report from the realtime thread through a ControllerEventSink instead of logging there: an
 * event is a few plain values, made without allocating, and the communication thread formats it into the line the
 * controller used to log.
 */

namespace ocs2::humanoid {
namespace {

constexpr ControllerEventCode kEveryCode[] = {
    ControllerEventCode::kPolicyDiverged,
    ControllerEventCode::kClockRewind,
    ControllerEventCode::kSafetyEntered,
    ControllerEventCode::kSafetyDecayComplete,
    ControllerEventCode::kContactWrenchGateChanged,
    ControllerEventCode::kContactEstimatorChanged,
    ControllerEventCode::kNoPolicyWeightCompensation,
};

/** A sink that keeps what it is handed, as the realtime event log does. */
class RecordingSink final : public ControllerEventSink {
 public:
  bool post(const ControllerEvent& event) override {
    events.push_back(event);
    return true;
  }
  std::vector<ControllerEvent> events;
};

TEST(ControllerEvent, IsPlainDataMadeWithoutAllocating) {
  static_assert(std::is_trivially_copyable_v<ControllerEvent>, "an event is copied into a queue slot");
  const std::size_t before = robot::realtime::heapAllocationCountOnThisThread();
  const ControllerEvent event = makeControllerEvent(ControllerEventCode::kContactEstimatorChanged, "Controller", /*value0=*/1.0,
                                                    /*value1=*/2.0, "CheaterSimContactEstimator");
  EXPECT_EQ(robot::realtime::heapAllocationCountOnThisThread() - before, 0u);
  EXPECT_EQ(controllerEventText(event), "CheaterSimContactEstimator");
  EXPECT_EQ(event.values[1], 2.0);
}

TEST(ControllerEvent, ALongTextIsCutToFit) {
  std::string name;
  name.resize(100, 'x');
  const ControllerEvent event = makeControllerEvent(ControllerEventCode::kContactEstimatorChanged, "Controller", /*value0=*/0.0,
                                                    /*value1=*/0.0, name);
  EXPECT_EQ(controllerEventText(event).size(), event.text.size() - 1);
}

TEST(ControllerEvent, EveryCodeFormatsToALineNamingItsController) {
  for (const ControllerEventCode code : kEveryCode) {
    const std::string line = formatControllerEvent(makeControllerEvent(code, "WBMpcMrtJointController", /*value0=*/0.25));
    EXPECT_EQ(line.rfind("[WBMpcMrtJointController] ", 0), 0u) << line;
    EXPECT_EQ(line.find("controller event"), std::string::npos) << "a code without its own line: " << line;
  }
  EXPECT_NE(formatControllerEvent(makeControllerEvent(ControllerEventCode::kPolicyDiverged, "C", /*value0=*/0.75)).find("0.75"),
            std::string::npos);
  EXPECT_TRUE(isWarningControllerEvent(makeControllerEvent(ControllerEventCode::kSafetyEntered, "C")));
  EXPECT_FALSE(isWarningControllerEvent(makeControllerEvent(ControllerEventCode::kContactWrenchGateChanged, "C")));
}

TEST(ControllerEvent, ASinkGetsTheEventAsPosted) {
  RecordingSink sink;
  ControllerEventSink& base = sink;
  EXPECT_TRUE(base.post(makeControllerEvent(ControllerEventCode::kClockRewind, "C", /*value0=*/0.5, /*value1=*/12.0)));
  ASSERT_EQ(sink.events.size(), 1u);
  EXPECT_EQ(sink.events[0].code, ControllerEventCode::kClockRewind);
  EXPECT_EQ(sink.events[0].values[0], 0.5);
  EXPECT_TRUE(LoggingControllerEventSink::instance().post(sink.events[0]));
}

}  // namespace
}  // namespace ocs2::humanoid
