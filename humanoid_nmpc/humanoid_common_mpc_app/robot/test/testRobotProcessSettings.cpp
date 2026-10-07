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

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "humanoid_common_mpc_app/robot/TelemetrySinkRegistry.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "robot_model/ContactEstimatorRegistry.h"

/*
 * The robot process's settings of the task file through loadRobotProcessSettings(), which reads the typed task file
 * strictly and names the file in its errors: a file that says nothing gets the defaults of the ROS sims, except the
 * lists; every field is read; a value of the wrong type is refused
 * where it is written and a value the robot process cannot run with by its field; the retired booleans are refused with
 * the name that replaced them; and every shipped task file reads with names the registries know.
 */

namespace ocs2::humanoid {
namespace {

using ::testing::AllOf;
using ::testing::DoubleEq;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::HasSubstr;
using ::testing::Optional;

// LINT.IfChange(shipped_task_files)
constexpr absl::string_view kShippedTaskFiles[] = {
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto",
    "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto",
    "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto",
    "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto",
    "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.textproto",
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/BUILD.bazel:shipped_task_files)

/** `content` written to `name` in the test's temporary directory; its path. */
std::string writeTemporaryFile(absl::string_view name, absl::string_view content) {
  const std::string file = (std::filesystem::path(std::getenv("TEST_TMPDIR")) / std::string(name)).string();
  std::ofstream(file, std::ios::trunc) << content;
  return file;
}

/** The settings of a task file whose whole text is `text`. */
absl::StatusOr<RobotProcessSettings> settingsOfText(absl::string_view text) {
  return loadRobotProcessSettings(writeTemporaryFile("robot_settings_task.textproto", text));
}

TEST(RobotProcessSettings, ATaskFileIsReadStrictlyAndItsErrorsNameIt) {
  const std::string valid = writeTemporaryFile("robot_settings_valid.textproto",
                                               "contact_estimator: \"robot_state\"\ntelemetry_sinks: \"bus\"\ntelemetry_frequency: 50\n");
  const absl::StatusOr<RobotProcessSettings> settings = loadRobotProcessSettings(valid);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_EQ(settings->contactEstimator, "robot_state");
  EXPECT_EQ(settings->telemetrySinks, std::vector<std::string>{"bus"});
  EXPECT_THAT(settings->telemetryFrequency, Optional(DoubleEq(50.0)));

  // A field the schema does not have, where it is written; a value the conversion refuses, by its field.
  const std::string unknown = writeTemporaryFile("robot_settings_unknown.textproto", "telemetry_sinks: \"bus\"\ntelemetry_rate: 50\n");
  const absl::StatusOr<RobotProcessSettings> unknownField = loadRobotProcessSettings(unknown);
  EXPECT_EQ(unknownField.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(unknownField.status().message(), absl::StrCat(unknown, ":2:"))) << unknownField.status();
  const std::string refused = writeTemporaryFile("robot_settings_refused.textproto", "telemetry_frequency: 0\n");
  const absl::StatusOr<RobotProcessSettings> refusedValue = loadRobotProcessSettings(refused);
  EXPECT_EQ(refusedValue.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StartsWith(refusedValue.status().message(), refused)) << refusedValue.status();
  EXPECT_TRUE(absl::StrContains(refusedValue.status().message(), "telemetry_frequency")) << refusedValue.status();
  EXPECT_EQ(loadRobotProcessSettings("/nonexistent/task.textproto").status().code(), absl::StatusCode::kNotFound);
}

TEST(RobotProcessSettings, AYamlTaskFileIsRefusedNamingTheFile) {
  // The task files are textprotos read strictly; a file in the YAML of the old task files does not parse.
  const std::string yaml = writeTemporaryFile("robot_settings_task.yaml", "contactEstimator: robot_state\ntelemetrySinks: []\n");
  const absl::StatusOr<RobotProcessSettings> settings = loadRobotProcessSettings(yaml);
  EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument) << settings.status();
  EXPECT_TRUE(absl::StrContains(settings.status().message(), yaml)) << settings.status();
}

TEST(RobotProcessSettings, AFileThatSaysNothingGetsTheDefaultsOfTheSimsButNoLists) {
  const absl::StatusOr<RobotProcessSettings> settings = settingsOfText("mpc { time_horizon: 1.0 }\n");
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_EQ(settings->contactEstimator, "cheater_sim");
  EXPECT_DOUBLE_EQ(settings->simulator.contactForceThreshold, 5.0);
  EXPECT_DOUBLE_EQ(settings->simulator.contactTimelineWindow, 5.0);
  EXPECT_EQ(settings->simulator.gantryHold, "weld_constraint");
  EXPECT_TRUE(settings->simulator.projectile.empty());
  EXPECT_DOUBLE_EQ(settings->fallRecovery.maxBaseTiltAngle, 0.0) << "the catch is off unless the file turns it on";
  EXPECT_DOUBLE_EQ(settings->fallRecovery.catchLift, 0.0);
  EXPECT_FALSE(settings->mpcEntryBlendTime.has_value());
  EXPECT_FALSE(settings->safetyDecayTimeConstant.has_value());
  EXPECT_FALSE(settings->contactWrenchGate.has_value());
  EXPECT_EQ(settings->wbMpcFeedforward, WbMpcFeedforward::kInverseDynamics);
  EXPECT_FALSE(settings->telemetryFrequency.has_value());
  // A list the file leaves out is empty: no telemetry, no viewer markers.
  EXPECT_TRUE(settings->telemetrySinks.empty());
  EXPECT_THAT(settings->simulator.visualizations, Optional(std::vector<std::string>{}));
  // One lost TCP segment on the bus is retransmitted after Linux's minimum retransmission timeout, 200 ms: the default
  // link-loss timeout outlasts it with margin, so that a single lost packet does not hold the robot.
  constexpr scalar_t kLinuxMinimumTcpRetransmissionTimeout = 0.2;
  EXPECT_GE(settings->mpcLinkPolicyTimeout, 2.0 * kLinuxMinimumTcpRetransmissionTimeout);
}

TEST(RobotProcessSettings, EveryFieldIsRead) {
  const absl::StatusOr<RobotProcessSettings> settings = settingsOfText(
      "contact_estimator: \"robot_state\"\n"
      "sim_contact_force_threshold: 8\n"
      "sim_contact_timeline_window: 3\n"
      "sim_visualizations: [\"zmp\", \"dcm\"]\n"
      "gantry_hold: \"kinematic_teleport\"\n"
      "sim_projectile: \"dodgeball\"\n"
      "sim_max_base_tilt_angle: 1\n"
      "sim_gantry_catch_lift: 0.15\n"
      "mpc_entry_blend_time: 2\n"
      "safety_decay_time_constant: 0.4\n"
      "contact_wrench_gate { debounce_time: 0.01 ramp_time: 0.04 }\n"
      "wb_mpc_feedforward: \"gravity_compensation\"\n"
      "telemetry_sinks: []\n"
      "telemetry_frequency: 50\n"
      "mpc_link { policy_timeout: 0.35 }\n");
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_EQ(settings->contactEstimator, "robot_state");
  EXPECT_DOUBLE_EQ(settings->simulator.contactForceThreshold, 8.0);
  EXPECT_DOUBLE_EQ(settings->simulator.contactTimelineWindow, 3.0);
  EXPECT_THAT(settings->simulator.visualizations, Optional(ElementsAre("zmp", "dcm")));
  EXPECT_EQ(settings->simulator.gantryHold, "kinematic_teleport");
  EXPECT_EQ(settings->simulator.projectile, "dodgeball");
  EXPECT_DOUBLE_EQ(settings->fallRecovery.maxBaseTiltAngle, 1.0);
  EXPECT_DOUBLE_EQ(settings->fallRecovery.catchLift, 0.15);
  EXPECT_THAT(settings->mpcEntryBlendTime, Optional(DoubleEq(2.0)));
  EXPECT_THAT(settings->safetyDecayTimeConstant, Optional(DoubleEq(0.4)));
  EXPECT_THAT(settings->contactWrenchGate, Optional(AllOf(Field(&ContactWrenchGate::Config::debounceTime, DoubleEq(0.01)),
                                                          Field(&ContactWrenchGate::Config::rampTime, DoubleEq(0.04)))));
  EXPECT_EQ(settings->wbMpcFeedforward, WbMpcFeedforward::kGravityCompensation);
  EXPECT_TRUE(settings->telemetrySinks.empty()) << "telemetry turned off by an empty list of sinks";
  EXPECT_THAT(settings->telemetryFrequency, Optional(DoubleEq(50.0)));
  EXPECT_DOUBLE_EQ(settings->mpcLinkPolicyTimeout, 0.35);
}

TEST(RobotProcessSettings, AValueOfTheWrongTypeIsRefusedWhereItIsWrittenAndOneThatCannotRunByItsField) {
  for (const std::string& text : {std::string("sim_gantry_catch_lift: \"high\"\n"), std::string("sim_visualizations: zmp\n"),
                                  std::string("contact_wrench_gate { ramp_time: \"soon\" }\n"),
                                  std::string("mpc_link { policy_timeout: x }\n"), std::string("[1, 2]\n")}) {
    const std::string file = writeTemporaryFile("robot_settings_wrong_type.textproto", text);
    const absl::StatusOr<RobotProcessSettings> settings = loadRobotProcessSettings(file);
    EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument) << text;
    EXPECT_THAT(settings.status().message(), HasSubstr(absl::StrCat(file, ":1:"))) << text;
  }
  struct Refused {
    std::string text;
    std::string field;
  };
  const Refused refused[] = {
      {.text = "wb_mpc_feedforward: \"magic\"\n", .field = "wb_mpc_feedforward"},
      {.text = "telemetry_frequency: 0\n", .field = "telemetry_frequency"},
      {.text = "mpc_link { policy_timeout: 0 }\n", .field = "mpc_link.policy_timeout"},
      {.text = "contact_wrench_gate { ramp_time: -0.1 }\n", .field = "contact_wrench_gate.ramp_time is -0.1 [s]"},
      // NaN passed the old sign check and then stopped the process in ContactWrenchGate::setConfig().
      {.text = "contact_wrench_gate { debounce_time: nan }\n", .field = "contact_wrench_gate.debounce_time is nan [s]"},
  };
  for (const Refused& value : refused) {
    const absl::StatusOr<RobotProcessSettings> settings = settingsOfText(value.text);
    EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument) << value.text;
    EXPECT_THAT(settings.status().message(), HasSubstr(value.field)) << value.text;
  }
}

TEST(RobotProcessSettings, TheRetiredBooleansAreRefusedWithTheirReplacement) {
  struct Retired {
    std::string text;
    std::string replacement;
  };
  const Retired retired[] = {
      {.text = "enable_telemetry: true\n", .replacement = "telemetry_sinks"},
      {.text = "enableTelemetry: false\n", .replacement = "telemetry_sinks"},
      {.text = "use_gravity_comp_feedforward: false\n", .replacement = "wb_mpc_feedforward"},
      {.text = "useGravityCompFeedforward: true\n", .replacement = "wb_mpc_feedforward"},
  };
  for (const Retired& field : retired) {
    const absl::StatusOr<RobotProcessSettings> settings = settingsOfText(field.text);
    EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument) << field.text;
    EXPECT_THAT(settings.status().message(), HasSubstr("is retired")) << field.text;
    EXPECT_THAT(settings.status().message(), HasSubstr(field.replacement)) << field.text;
  }
}

TEST(RobotProcessSettings, EveryShippedTaskFileReadsWithRegisteredNamesAndTheShippedDefaults) {
  robot::model::ContactEstimatorRegistry estimators;
  const TelemetrySinkRegistry sinks;
  for (const absl::string_view taskFile : kShippedTaskFiles) {
    const absl::StatusOr<RobotProcessSettings> settings = loadRobotProcessSettings(std::string(taskFile));
    ASSERT_TRUE(settings.ok()) << taskFile << ": " << settings.status();
    const std::string estimator = robot::model::ContactEstimatorRegistry::canonicalName(settings->contactEstimator);
    EXPECT_TRUE(estimators.has(estimator) || estimator == "cheater_sim") << taskFile << ": contact_estimator " << estimator;
    for (const std::string& sink : settings->telemetrySinks) EXPECT_TRUE(sinks.has(sink)) << taskFile << ": telemetry sink " << sink;
    EXPECT_FALSE(settings->telemetrySinks.empty()) << taskFile << ": every robot publishes its telemetry";
    EXPECT_TRUE(robot::mujoco_sim_interface::gantryHoldFromName(settings->simulator.gantryHold).ok()) << taskFile;
    // The debugging feedforward stays off in what ships.
    EXPECT_EQ(settings->wbMpcFeedforward, WbMpcFeedforward::kInverseDynamics) << taskFile;
  }
}

TEST(TelemetrySinkRegistry, TheNamesAreTheRegisteredSinksOnceEach) {
  const TelemetrySinkRegistry sinks;
  const std::vector<std::string> names = sinks.names();
  ASSERT_FALSE(names.empty());
  for (const std::string& name : names) {
    EXPECT_TRUE(sinks.has(name)) << name;
    EXPECT_EQ(std::count(names.begin(), names.end(), name), 1) << name;
    EXPECT_THAT(sinks.availableNames(), HasSubstr(name));
  }
  EXPECT_FALSE(sinks.has("no_such_sink"));
}

TEST(WbMpcFeedforward, EveryNameIsReadAsItsFeedforwardAndNamesItBack) {
  const std::vector<std::string> names = wbMpcFeedforwardNames();
  ASSERT_FALSE(names.empty());
  for (const std::string& name : names) {
    EXPECT_EQ(std::count(names.begin(), names.end(), name), 1) << name;
    const absl::StatusOr<RobotProcessSettings> settings = settingsOfText(absl::StrCat("wb_mpc_feedforward: \"", name, "\""));
    ASSERT_TRUE(settings.ok()) << name << ": " << settings.status();
    EXPECT_EQ(wbMpcFeedforwardName(settings->wbMpcFeedforward), name);
  }
}

}  // namespace
}  // namespace ocs2::humanoid
