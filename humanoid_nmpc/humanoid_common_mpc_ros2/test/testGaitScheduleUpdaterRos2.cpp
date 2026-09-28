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

#include <unistd.h>

#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include <rclcpp/rclcpp.hpp>

#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_ros2_msgs/msg/mode_schedule.hpp>

#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc_ros2/gait/GaitScheduleUpdaterRos2.h"
#include "humanoid_common_mpc_ros2/gait/ModeSequenceTemplateRos.h"

/*
 * GaitScheduleUpdaterRos2 fed through its topic. The subclass used to keep a "gait received" flag of its own and a
 * preSolverRun() of its own that read only that flag, while GaitScheduleUpdater::reset() cleared the base class's: a
 * gait received before an MPC reset was still inserted after it. It now sets the base class's (atomic) flag, which the
 * reset clears.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kHorizon = 1.0;  // [s]

ModeSequenceTemplate stanceTemplate() {
  return ModeSequenceTemplate({0.0, 0.5}, {ModeNumber::STANCE});
}

ModeSequenceTemplate walkTemplate() {
  return ModeSequenceTemplate({0.0, 0.35, 0.7}, {ModeNumber::LF, ModeNumber::RF});
}

bool hasSwing(const ModeSchedule& schedule, scalar_t from, scalar_t to) {
  for (scalar_t time = from; time <= to; time += 0.01) {
    if (schedule.modeAtTime(time) != ModeNumber::STANCE) return true;
  }
  return false;
}

class GaitScheduleUpdaterRos2Test : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { rclcpp::init(/*argc=*/0, /*argv=*/nullptr); }
  static void TearDownTestSuite() {
    if (rclcpp::ok()) rclcpp::shutdown();
  }

  void SetUp() override {
    // A robot name of the test's own, so that no other process shares the topic; intra-process delivery keeps the
    // middleware's discovery out of the test.
    const std::string robotName = absl::StrCat("gait_schedule_updater_test_", getpid());
    rclcpp::NodeOptions nodeOptions;
    nodeOptions.use_intra_process_comms(true);
    node_ = std::make_shared<rclcpp::Node>(robotName, nodeOptions);
    schedule_ = std::make_shared<GaitSchedule>(ModeSchedule({0.5}, {ModeNumber::STANCE, ModeNumber::STANCE}), stanceTemplate(),
                                               /*phaseTransitionStanceTime=*/0.0);
    updater_ = std::make_unique<GaitScheduleUpdaterRos2>(node_, schedule_, robotName);
    publisher_ = node_->create_publisher<ocs2_ros2_msgs::msg::ModeSchedule>(absl::StrCat(robotName, "_mpc_mode_schedule"),
                                                                            rclcpp::QoS(1).best_effort());
  }

  void TearDown() override {
    publisher_.reset();
    updater_.reset();
    node_.reset();
  }

  /** Publishes `gait` and spins until the updater holds it. */
  bool publishAndReceive(const ModeSequenceTemplate& gait) {
    publisher_->publish(createModeSequenceTemplateMsg(gait));
    for (int attempt = 0; attempt < 200; ++attempt) {
      rclcpp::spin_some(node_);
      const ModeSequenceTemplate received = updater_->getReceivedGait();
      if (received.switchingTimes == gait.switchingTimes && received.modeSequence == gait.modeSequence) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
  }

  /** Runs the updater's pre-solve hook at `initTime` and returns whether the schedule the solve reads has a swing. */
  bool solveSeesASwing(scalar_t initTime) {
    updater_->preSolverRun(initTime, initTime + kHorizon, vector_t(), ReferenceManager());
    return hasSwing(schedule_->getModeSchedule(initTime, initTime + 2.0 * kHorizon), initTime, initTime + 2.0 * kHorizon);
  }

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<GaitSchedule> schedule_;
  std::unique_ptr<GaitScheduleUpdaterRos2> updater_;
  rclcpp::Publisher<ocs2_ros2_msgs::msg::ModeSchedule>::SharedPtr publisher_;
};

TEST_F(GaitScheduleUpdaterRos2Test, aGaitFromTheTopicIsInsertedBeforeTheNextSolve) {
  ASSERT_FALSE(solveSeesASwing(/*initTime=*/1.0)) << "the schedule starts in stance";
  ASSERT_TRUE(publishAndReceive(walkTemplate())) << "the published gait never reached the updater";
  EXPECT_TRUE(solveSeesASwing(/*initTime=*/1.0)) << "the received gait was not inserted";
}

TEST_F(GaitScheduleUpdaterRos2Test, aGaitReceivedBeforeAResetIsDroppedWithIt) {
  ASSERT_TRUE(publishAndReceive(walkTemplate())) << "the published gait never reached the updater";
  updater_->reset();
  EXPECT_FALSE(solveSeesASwing(/*initTime=*/1.0)) << "a gait received before the reset was inserted after it";

  // Positive control: a gait received after the reset is inserted, through the same path.
  ASSERT_TRUE(publishAndReceive(ModeSequenceTemplate({0.0, 0.4, 0.8}, {ModeNumber::LF, ModeNumber::RF})));
  EXPECT_TRUE(solveSeesASwing(/*initTime=*/1.0));
}

}  // namespace
}  // namespace ocs2::humanoid
