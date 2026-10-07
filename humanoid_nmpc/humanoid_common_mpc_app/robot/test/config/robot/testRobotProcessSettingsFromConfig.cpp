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

// The robot process's settings from a typed task file: a file that says nothing gives the defaults of
// RobotProcessSettings except the lists it leaves out, which are empty, every field is read, a value the robot process cannot
// run with is refused by its field, the retired fields name their replacement, and the schema's link timeout is the
// link's.

#include <limits>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/contact/ContactWrenchGate.h"
#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "humanoid_common_mpc_app/robot/config/robot/RobotProcessSettingsFromConfig.h"
#include "humanoid_mpc_config/contact_wrench_gate_config.nproto.h"
#include "humanoid_mpc_config/mpc_link_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.pb.h"
#include "humanoid_mpc_ipc/RemoteMpcLink.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::HasSubstr;
using ::testing::Optional;

// The times of a gate the tests set and read back [s].
constexpr double kDebounceTime = 0.01;
constexpr double kRampTime = 0.04;

TEST(RobotProcessSettingsFromConfigTest, AFileThatSaysNothingGivesTheLoadersDefaultsButNoLists) {
  const absl::StatusOr<RobotProcessSettings> settings = robotProcessSettingsFromConfig(mpc_config::TaskFile{});
  ASSERT_TRUE(settings.ok()) << settings.status();
  const RobotProcessSettings defaults;
  EXPECT_EQ(settings->contactEstimator, defaults.contactEstimator);
  EXPECT_EQ(settings->simulator.contactForceThreshold, defaults.simulator.contactForceThreshold);
  EXPECT_EQ(settings->simulator.contactTimelineWindow, defaults.simulator.contactTimelineWindow);
  EXPECT_EQ(settings->simulator.gantryHold, defaults.simulator.gantryHold);
  EXPECT_EQ(settings->simulator.projectile, defaults.simulator.projectile);
  EXPECT_EQ(settings->fallRecovery.maxBaseTiltAngle, defaults.fallRecovery.maxBaseTiltAngle) << "the catch stays off";
  EXPECT_EQ(settings->fallRecovery.catchLift, defaults.fallRecovery.catchLift);
  EXPECT_EQ(settings->mpcEntryBlendTime, defaults.mpcEntryBlendTime);
  EXPECT_EQ(settings->safetyDecayTimeConstant, defaults.safetyDecayTimeConstant);
  EXPECT_FALSE(settings->contactWrenchGate.has_value()) << "the instantaneous gate";
  EXPECT_EQ(settings->wbMpcFeedforward, defaults.wbMpcFeedforward);
  EXPECT_EQ(settings->telemetryFrequency, defaults.telemetryFrequency);
  EXPECT_EQ(settings->mpcLinkPolicyTimeout, defaults.mpcLinkPolicyTimeout);
  // A repeated field the file leaves out is empty: every shipped file lists both.
  EXPECT_TRUE(settings->telemetrySinks.empty());
  EXPECT_THAT(settings->simulator.visualizations, Optional(std::vector<std::string>{}));
}

TEST(RobotProcessSettingsFromConfigTest, EveryFieldIsRead) {
  mpc_config::TaskFile task;
  task.contact_estimator = "robot_state";
  task.sim_contact_force_threshold = 7.5;
  task.sim_contact_timeline_window = 3.0;
  task.sim_visualizations = {"metrics", "zmp"};
  task.gantry_hold = "kinematic_teleport";
  task.sim_projectile = "dodgeball";
  task.sim_max_base_tilt_angle = 1.0;
  task.sim_gantry_catch_lift = 0.15;
  task.mpc_entry_blend_time = 0.4;
  task.safety_decay_time_constant = 0.5;
  mpc_config::ContactWrenchGateConfig gate;
  gate.debounce_time = kDebounceTime;
  gate.ramp_time = kRampTime;
  task.contact_wrench_gate = gate;
  task.wb_mpc_feedforward = "gravity_compensation";
  task.telemetry_sinks = {"bus"};
  task.telemetry_frequency = 50.0;
  task.mpc_link.policy_timeout = 0.75;

  const absl::StatusOr<RobotProcessSettings> settings = robotProcessSettingsFromConfig(task);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_EQ(settings->contactEstimator, "robot_state");
  EXPECT_EQ(settings->simulator.contactForceThreshold, 7.5);
  EXPECT_EQ(settings->simulator.contactTimelineWindow, 3.0);
  EXPECT_THAT(settings->simulator.visualizations, Optional(ElementsAre("metrics", "zmp")));
  EXPECT_EQ(settings->simulator.gantryHold, "kinematic_teleport");
  EXPECT_EQ(settings->simulator.projectile, "dodgeball");
  EXPECT_EQ(settings->fallRecovery.maxBaseTiltAngle, 1.0);
  EXPECT_EQ(settings->fallRecovery.catchLift, 0.15);
  EXPECT_THAT(settings->mpcEntryBlendTime, Optional(0.4));
  EXPECT_THAT(settings->safetyDecayTimeConstant, Optional(0.5));
  EXPECT_THAT(settings->contactWrenchGate, Optional(AllOf(Field(&ContactWrenchGate::Config::debounceTime, kDebounceTime),
                                                          Field(&ContactWrenchGate::Config::rampTime, kRampTime))));
  EXPECT_EQ(settings->wbMpcFeedforward, WbMpcFeedforward::kGravityCompensation);
  EXPECT_THAT(settings->telemetrySinks, ElementsAre("bus"));
  EXPECT_THAT(settings->telemetryFrequency, Optional(50.0));
  EXPECT_EQ(settings->mpcLinkPolicyTimeout, 0.75);
}

TEST(RobotProcessSettingsFromConfigTest, AValueTheRobotProcessCannotRunWithIsRefusedByItsField) {
  struct Case {
    mpc_config::TaskFile task;
    std::string field;
  };
  std::vector<Case> cases;
  {
    Case unknownFeedforward{.task = {}, .field = "wb_mpc_feedforward is 'torque_feedforward'"};
    unknownFeedforward.task.wb_mpc_feedforward = "torque_feedforward";
    cases.push_back(unknownFeedforward);
  }
  for (const double bad : {-0.01, std::numeric_limits<double>::quiet_NaN()}) {
    Case negativeGate{.task = {}, .field = "contact_wrench_gate"};
    mpc_config::ContactWrenchGateConfig gate;
    gate.debounce_time = bad;
    negativeGate.task.contact_wrench_gate = gate;
    cases.push_back(negativeGate);
  }
  for (const double bad : {0.0, -100.0, std::numeric_limits<double>::quiet_NaN()}) {
    Case frequency{.task = {}, .field = "telemetry_frequency must be positive"};
    frequency.task.telemetry_frequency = bad;
    cases.push_back(frequency);
    Case timeout{.task = {}, .field = "mpc_link.policy_timeout must be a positive number of seconds"};
    timeout.task.mpc_link.policy_timeout = bad;
    cases.push_back(timeout);
  }
  for (const Case& refused : cases) {
    const absl::StatusOr<RobotProcessSettings> settings = robotProcessSettingsFromConfig(refused.task);
    EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument) << refused.field;
    EXPECT_THAT(settings.status().message(), HasSubstr(refused.field));
  }
}

TEST(RobotProcessSettingsFromConfigTest, TheRetiredKeysNameTheirReplacement) {
  struct Retired {
    const char* absl_nonnull text;
    const char* absl_nonnull replacement;
  };
  const Retired retired[] = {
      {.text = "enableTelemetry: true\n", .replacement = "telemetry_sinks"},
      {.text = "enable_telemetry: false\n", .replacement = "telemetry_sinks"},
      {.text = "useGravityCompFeedforward: true\n", .replacement = "wb_mpc_feedforward"},
  };
  for (const Retired& key : retired) {
    const absl::StatusOr<humanoid_mpc_config::TaskFile> parsed =
        nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(key.text, "task.textproto");
    EXPECT_EQ(parsed.status().code(), absl::StatusCode::kInvalidArgument) << key.text;
    EXPECT_THAT(parsed.status().message(), HasSubstr("is retired: "));
    EXPECT_THAT(parsed.status().message(), HasSubstr(key.replacement));
  }
}

TEST(RobotProcessSettingsFromConfigTest, TheSchemasLinkTimeoutIsTheLinksAndTheSettings) {
  EXPECT_EQ(mpc_config::MpcLinkConfig{}.policy_timeout, RobotProcessSettings{}.mpcLinkPolicyTimeout);
  EXPECT_EQ(mpc_config::MpcLinkConfig{}.policy_timeout, ipc::RemoteMpcLink::Config{}.policyTimeout);
}

}  // namespace
}  // namespace ocs2::humanoid
