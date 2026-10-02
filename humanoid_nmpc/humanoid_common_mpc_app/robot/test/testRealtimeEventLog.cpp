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

#include <string>
#include <vector>

#include "humanoid_common_mpc_app/robot/FallRecoveryTypes.h"
#include "humanoid_common_mpc_app/robot/RealtimeEventLog.h"

/*
 * The reports of the realtime thread: posted in order, logged by the communication thread with the wording of the lines
 * the ROS sims logged from their loops, and dropped (and counted) rather than waited for when the queue is full.
 */

namespace ocs2::humanoid {
namespace {

TEST(RealtimeEventLog, DrainsTheReportsInTheOrderTheyWerePosted) {
  RealtimeEventLog log(/*capacity=*/8);
  ASSERT_TRUE(log.post(RealtimeEventCode::kTorquesEnabled, /*detail=*/0, "JOINT_PD"));
  ASSERT_TRUE(log.post(RealtimeEventCode::kGantryUnlockCommanded));
  ASSERT_TRUE(log.post(RealtimeEventCode::kTorquesDisabled, /*detail=*/0, "ZERO_TORQUE"));
  std::vector<std::string> lines;
  EXPECT_EQ(log.drain([&](const RealtimeEvent& event) { lines.push_back(formatRealtimeEvent(event)); }), 3u);
  ASSERT_EQ(lines.size(), 3u);
  EXPECT_EQ(lines[0], "FSM command received: JOINT_PD - enabling torques.");
  EXPECT_EQ(lines[1], "FSM command received: Unlocking gantry.");
  EXPECT_EQ(lines[2], "FSM command received: ZERO_TORQUE - zero-torque mode.");
  EXPECT_EQ(log.drain([](const RealtimeEvent&) {}), 0u) << "drained once";
}

TEST(RealtimeEventLog, ACatchNamesItsCauseAndTheLift) {
  RealtimeEventLog log;
  ASSERT_TRUE(log.post(RealtimeEventCode::kCaughtAndSettling, static_cast<std::int32_t>(DiscontinuityCause::kSimulatorReset),
                       /*text=*/{}, /*value0=*/0.0, /*value1=*/1.0, /*value2=*/0.15, /*count=*/3));
  ASSERT_TRUE(log.post(RealtimeEventCode::kDiscontinuity, static_cast<std::int32_t>(DiscontinuityCause::kTiltCaught), /*text=*/{},
                       /*value0=*/1.2, /*value1=*/1.0));
  std::vector<RealtimeEvent> events;
  log.drain([&](const RealtimeEvent& event) { events.push_back(event); });
  ASSERT_EQ(events.size(), 2u);
  const std::string caught = formatRealtimeEvent(events[0]);
  EXPECT_NE(caught.find("reset epoch 3"), std::string::npos) << caught;
  EXPECT_NE(caught.find("0.15 m (simGantryCatchLift)"), std::string::npos) << caught;
  EXPECT_TRUE(isWarningEvent(events[0]));
  const std::string tilted = formatRealtimeEvent(events[1]);
  EXPECT_NE(tilted.find("the base tilted 1.2 rad, past simMaxBaseTiltAngle 1 rad"), std::string::npos) << tilted;
}

TEST(RealtimeEventLog, TheTextIsCutToFitAndTerminated) {
  RealtimeEventLog log;
  const std::string longName(100, 'x');
  ASSERT_TRUE(log.post(RealtimeEventCode::kContactEstimatorSwapped, /*detail=*/0, longName));
  log.drain([&](const RealtimeEvent& event) { EXPECT_EQ(realtimeEventText(event), std::string(event.text.size() - 1, 'x')); });
}

TEST(RealtimeEventLog, AFullQueueDropsAndCountsInsteadOfWaiting) {
  RealtimeEventLog log(/*capacity=*/2);
  EXPECT_TRUE(log.post(RealtimeEventCode::kSettled));
  EXPECT_TRUE(log.post(RealtimeEventCode::kSettled));
  EXPECT_FALSE(log.post(RealtimeEventCode::kSettled));
  EXPECT_EQ(log.dropped(), 1u);
  EXPECT_EQ(log.drainToLog(), 2u);
  EXPECT_TRUE(log.post(RealtimeEventCode::kSettled)) << "room again once drained";
}

TEST(RealtimeEventLog, IsTheControllersSinkAndLogsItsReportsAsTheControllerDid) {
  RealtimeEventLog log;
  ControllerEventSink& sink = log;
  ASSERT_TRUE(sink.post(makeControllerEvent(ControllerEventCode::kPolicyDiverged, "CentroidalMpcMrtJointController", /*value0=*/0.62)));
  ASSERT_TRUE(sink.post(makeControllerEvent(ControllerEventCode::kContactEstimatorChanged, "WBMpcMrtJointController", /*value0=*/0.0,
                                            /*value1=*/0.0, "CheaterSimContactEstimator")));
  std::vector<RealtimeEvent> events;
  EXPECT_EQ(log.drain([&](const RealtimeEvent& event) { events.push_back(event); }), 2u);
  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[0].code, RealtimeEventCode::kControllerEvent);
  EXPECT_TRUE(isWarningEvent(events[0])) << "a diverged policy is a warning";
  const std::string diverged = formatRealtimeEvent(events[0]);
  EXPECT_EQ(diverged.rfind("[CentroidalMpcMrtJointController] MPC policy diverged", 0), 0u) << diverged;
  EXPECT_NE(diverged.find("0.62"), std::string::npos) << diverged;
  EXPECT_FALSE(isWarningEvent(events[1]));
  EXPECT_NE(formatRealtimeEvent(events[1]).find("CheaterSimContactEstimator"), std::string::npos);
}

TEST(SettlePhaseDescription, EveryPhaseSaysWhatTheRobotWaitsFor) {
  EXPECT_EQ(settlePhaseDescription(SettlePhase::kLifting), "lifting the robot clear of the ground");
  EXPECT_EQ(settlePhaseDescription(SettlePhase::kSettlingOnFeet), "waiting for the robot to come to rest on its feet");
  EXPECT_EQ(discontinuityReason(DiscontinuityCause::kGantryLocked, /*resetEpoch=*/0, /*tilt=*/0.0, /*maxTilt=*/0.0),
            "the gantry was locked");
}

}  // namespace
}  // namespace ocs2::humanoid
