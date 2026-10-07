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

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/mrt/ContactEstimateIntake.h"
#include "robot_model/ContactEstimator.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"

/*
 * The measured contact state of the MRT joint controllers: an estimate with one flag per contact point is taken, any
 * other is refused and reported once per run of refusals, and a wrong estimator is refused before it is installed.
 */

namespace ocs2::humanoid {
namespace {

/** A sink that keeps what it is handed. */
class RecordingSink final : public ControllerEventSink {
 public:
  RecordingSink() { events.reserve(16); }
  bool post(const ControllerEvent& event) override {
    events.push_back(event);
    return true;
  }
  std::vector<ControllerEvent> events;
};

/** An estimator that reports the same flags every cycle. */
class FixedContactEstimator final : public robot::model::ContactEstimator {
 public:
  explicit FixedContactEstimator(std::vector<bool> flags) : flags_(std::move(flags)) {}
  void estimateContactFlags(const robot::model::RobotState& /*robotState*/, std::vector<bool>& flags) override { flags = flags_; }
  std::string getName() const override { return "fixed"; }

 private:
  std::vector<bool> flags_;
};

TEST(ContactEstimateIntake, TakesAnEstimateWithOneFlagPerContactPoint) {
  ContactEstimateIntake intake("Controller");
  RecordingSink sink;
  contact_flag_t measured{};
  EXPECT_TRUE(intake.take({true, false}, measured, sink));
  EXPECT_EQ(measured, (contact_flag_t{true, false}));
  EXPECT_EQ(intake.numRefused(), 0U);
  EXPECT_TRUE(sink.events.empty());
}

TEST(ContactEstimateIntake, RefusesAnyOtherEstimateKeepsTheLastFlagsAndReportsOncePerRun) {
  ContactEstimateIntake intake("Controller");
  intake.resetEstimator("per_corner");
  RecordingSink sink;
  contact_flag_t measured{};
  ASSERT_TRUE(intake.take({true, true}, measured, sink));
  for (int cycle = 0; cycle < 5; ++cycle) {
    EXPECT_FALSE(intake.take({true, true, false, false}, measured, sink));
  }
  EXPECT_EQ(measured, (contact_flag_t{true, true})) << "a refused estimate keeps the flags of the cycle before";
  EXPECT_EQ(intake.numRefused(), 5U);
  ASSERT_EQ(sink.events.size(), 1U) << "one report per run of refusals, not one per cycle";
  const ControllerEvent& report = sink.events[0];
  EXPECT_EQ(report.code, ControllerEventCode::kContactEstimateRefused);
  EXPECT_EQ(report.values[0], 4.0);
  EXPECT_EQ(report.values[1], static_cast<double>(kNumContacts));
  EXPECT_EQ(controllerEventText(report), "per_corner");
  EXPECT_TRUE(isWarningControllerEvent(report));

  // A taken estimate ends the run: the next refusal is reported again, and so is the first one of a new estimator.
  ASSERT_TRUE(intake.take({false, true}, measured, sink));
  EXPECT_FALSE(intake.take({false}, measured, sink));
  EXPECT_EQ(sink.events.size(), 2U);
  intake.resetEstimator("another");
  EXPECT_FALSE(intake.take({false}, measured, sink));
  ASSERT_EQ(sink.events.size(), 3U);
  EXPECT_EQ(controllerEventText(sink.events[2]), "another");
}

TEST(ContactEstimateIntake, AllocatesNothing) {
  ContactEstimateIntake intake("Controller");
  RecordingSink sink;
  contact_flag_t measured{};
  const std::vector<bool> good = {true, false};
  const std::vector<bool> bad = {true};
  const size_t before = robot::realtime::heapAllocationCountOnThisThread();
  intake.resetEstimator("a name that is longer than the text of an event can hold");
  intake.take(good, measured, sink);
  intake.take(bad, measured, sink);
  EXPECT_EQ(robot::realtime::heapAllocationCountOnThisThread() - before, 0U);
}

class CheckContactEstimatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    directory_ = std::filesystem::path(::testing::TempDir()) / "contact_estimate_intake_test";
    std::filesystem::create_directories(directory_);
    const std::filesystem::path urdf = directory_ / "biped.urdf";
    std::ofstream(urdf) << R"(<?xml version="1.0"?>
        <robot name="biped">
            <link name="base_link"/>
            <link name="foot_l"/>
            <link name="foot_r"/>
            <joint name="hip_l" type="revolute">
                <parent link="base_link"/>
                <child link="foot_l"/>
                <axis xyz="0 1 0"/>
                <limit lower="-1.57" upper="1.57" effort="100" velocity="2.0"/>
            </joint>
            <joint name="hip_r" type="revolute">
                <parent link="base_link"/>
                <child link="foot_r"/>
                <axis xyz="0 1 0"/>
                <limit lower="-1.57" upper="1.57" effort="100" velocity="2.0"/>
            </joint>
        </robot>)";
    absl::StatusOr<robot::model::RobotDescription> description = robot::model::RobotDescription::Create(urdf.string());
    ASSERT_TRUE(description.ok()) << description.status();
    description_ = std::make_unique<robot::model::RobotDescription>(*std::move(description));
  }

  void TearDown() override {
    std::error_code ignored;
    std::filesystem::remove_all(directory_, ignored);
  }

  std::filesystem::path directory_;
  std::unique_ptr<robot::model::RobotDescription> description_;
};

TEST_F(CheckContactEstimatorTest, AcceptsOneFlagPerContactPointAndRefusesAnyOtherCount) {
  const robot::model::RobotState state(*description_);
  FixedContactEstimator good({true, true});
  EXPECT_TRUE(checkContactEstimator(good, state, "fixed").ok());
  FixedContactEstimator one({true});
  const absl::Status refused = checkContactEstimator(one, state, "fixed");
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(refused.message().find("'fixed' reports 1 contact flags"), std::string::npos) << refused;
}

}  // namespace
}  // namespace ocs2::humanoid
