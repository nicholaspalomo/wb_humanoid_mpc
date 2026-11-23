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

// The gait file's gait_scheduler and rolling_lateral_debt block
// (rollingStepSchedulerSettingsFromConfig()) on the shipped robots: the wheeled
// gait files step the wheels of the LimX Tron1 and the wheeled ANYmal C in
// their groups, a gait file of the gait switching table reads no block, and
// every malformed selection or block is refused with its field.

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/config/ConfigFiles.h"
#include "legged_common_mpc/gait/RollingStepScheduler.h"
#include "legged_mpc_config/gait_file.nproto.h"
#include "legged_mpc_config/rolling_step_scheduler_config.nproto.h"

namespace ocs2::legged {
namespace {

using ::testing::HasSubstr;

constexpr char kTron1Task[] =
    "robot_models/limx_tron1/tron1_centroidal_mpc/config/mpc/task.textproto";
constexpr char kTron1Urdf[] =
    "robot_models/limx_tron1/tron1_description/urdf/tron1.urdf";
constexpr char kAnymalCWheelsTask[] =
    "robot_models/anybotics_anymal_c_wheels/anymal_c_wheels_centroidal_mpc/"
    "config/mpc/task.textproto";
constexpr char kAnymalCWheelsUrdf[] =
    "robot_models/anybotics_anymal_c_wheels/anymal_c_wheels_description/urdf/"
    "anymal_c_wheels.urdf";
constexpr char kAnymalBTask[] =
    "robot_models/anybotics_anymal_b/anymal_b_centroidal_mpc/config/mpc/"
    "task.textproto";
constexpr char kAnymalBUrdf[] =
    "robot_models/anybotics_anymal_b/anymal_b_description/urdf/anymal_b.urdf";
constexpr char kWheeledGait[] =
    "legged_nmpc/legged_common_mpc/config/command/wheeled_gait.textproto";
constexpr char kWheeledQuadrupedGait[] =
    "legged_nmpc/legged_common_mpc/config/command/"
    "wheeled_quadruped_gait.textproto";
constexpr char kLeftWheel[] = "wheel_L_contact";
constexpr char kRightWheel[] = "wheel_R_contact";

std::string runfilesPath(absl::string_view relativePath) {
  const char* absl_nullable directory = std::getenv("TEST_SRCDIR");
  return directory == nullptr
             ? std::string(relativePath)
             : absl::StrCat(directory, "/_main/", relativePath);
}

absl::StatusOr<ModelSettings> settingsOf(absl::string_view task,
                                         absl::string_view urdf) {
  return ModelSettings::Create(runfilesPath(task), runfilesPath(urdf),
                               "testRollingStepSchedulerConfig_",
                               /*verbose=*/false);
}

mpc_config::RollingStepSchedulerConfig::StepGroup group(
    std::vector<std::string> contacts) {
  mpc_config::RollingStepSchedulerConfig::StepGroup entry;
  entry.contacts = std::move(contacts);
  return entry;
}

/** A gait file that steps Tron1's wheels one at a time with the default
 * durations. */
mpc_config::GaitFile tron1Steps() {
  mpc_config::GaitFile gaitFile;
  gaitFile.gait_scheduler = kRollingLateralDebtSchedulerName;
  mpc_config::RollingStepSchedulerConfig block;
  block.step_groups = {group({kLeftWheel}), group({kRightWheel})};
  gaitFile.rolling_lateral_debt = block;
  return gaitFile;
}

/** The status of the settings of `gaitFile` on Tron1. */
absl::Status tron1Status(const mpc_config::GaitFile& gaitFile) {
  const absl::StatusOr<ModelSettings> tron1 =
      settingsOf(kTron1Task, kTron1Urdf);
  if (!tron1.ok()) return tron1.status();
  return rollingStepSchedulerSettingsFromConfig(gaitFile, *tron1).status();
}

/** The contact names of each group of `settings`. */
std::vector<std::vector<std::string>> groupNames(
    const RollingStepSchedulerSettings& settings,
    const ModelSettings& modelSettings) {
  std::vector<std::vector<std::string>> names;
  for (const std::vector<size_t>& contacts : settings.stepGroups) {
    std::vector<std::string>& groupContacts = names.emplace_back();
    for (const size_t contact : contacts) {
      groupContacts.push_back(modelSettings.contactNames[contact]);
    }
  }
  return names;
}

TEST(RollingStepSchedulerConfig, TheShippedWheeledGaitFilesStepTheirWheels) {
  struct Robot {
    const char* absl_nonnull task;
    const char* absl_nonnull urdf;
    const char* absl_nonnull gaitFile;
    std::vector<std::vector<std::string>> groups;
  };
  const std::vector<Robot> robots = {
      {.task = kTron1Task,
       .urdf = kTron1Urdf,
       .gaitFile = kWheeledGait,
       .groups = {{kLeftWheel}, {kRightWheel}}},
      {.task = kAnymalCWheelsTask,
       .urdf = kAnymalCWheelsUrdf,
       .gaitFile = kWheeledQuadrupedGait,
       .groups = {{"LF_wheel_contact", "RH_wheel_contact"},
                  {"RF_wheel_contact", "LH_wheel_contact"}}}};
  for (const Robot& robot : robots) {
    SCOPED_TRACE(robot.gaitFile);
    const absl::StatusOr<ModelSettings> modelSettings =
        settingsOf(robot.task, robot.urdf);
    ASSERT_TRUE(modelSettings.ok()) << modelSettings.status();
    const absl::StatusOr<mpc_config::GaitFile> gaitFile =
        loadGaitFile(runfilesPath(robot.gaitFile));
    ASSERT_TRUE(gaitFile.ok()) << gaitFile.status();
    const absl::StatusOr<std::optional<RollingStepSchedulerSettings>> settings =
        rollingStepSchedulerSettingsFromConfig(*gaitFile, *modelSettings);
    ASSERT_TRUE(settings.ok()) << settings.status();
    ASSERT_TRUE(settings->has_value());
    EXPECT_EQ(groupNames(**settings, *modelSettings), robot.groups);
    EXPECT_GE((*settings)->commitWindow, (*settings)->minLiftOffLead);
  }
}

TEST(RollingStepSchedulerConfig, TheGaitSwitchingTableReadsNoBlock) {
  const absl::StatusOr<ModelSettings> tron1 =
      settingsOf(kTron1Task, kTron1Urdf);
  ASSERT_TRUE(tron1.ok()) << tron1.status();
  const mpc_config::GaitFile table;
  EXPECT_EQ(table.gait_scheduler, kGaitSwitchingTableSchedulerName);
  const absl::StatusOr<std::optional<RollingStepSchedulerSettings>> settings =
      rollingStepSchedulerSettingsFromConfig(table, *tron1);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_FALSE(settings->has_value());
}

TEST(RollingStepSchedulerConfig, EveryNameOfTheRegistryResolves) {
  for (const std::string& name : gaitSchedulerNames()) {
    EXPECT_TRUE(gaitSchedulerFromName(name).ok()) << name;
  }
  EXPECT_THAT(gaitSchedulerFromName("lateral_debt").status().message(),
              HasSubstr("gait_switching_table, rolling_lateral_debt"));
}

TEST(RollingStepSchedulerConfig, AMalformedSelectionIsRefused) {
  mpc_config::GaitFile unknown = tron1Steps();
  unknown.gait_scheduler = "wheel_steps";
  EXPECT_THAT(tron1Status(unknown).message(),
              HasSubstr("unknown gait_scheduler 'wheel_steps'"));

  mpc_config::GaitFile unread = tron1Steps();
  unread.gait_scheduler = kGaitSwitchingTableSchedulerName;
  EXPECT_THAT(tron1Status(unread).message(),
              HasSubstr("only gait_scheduler: \"rolling_lateral_debt\" reads"));

  mpc_config::GaitFile noBlock = tron1Steps();
  noBlock.rolling_lateral_debt.reset();
  EXPECT_THAT(tron1Status(noBlock).message(),
              HasSubstr("needs the gait file's rolling_lateral_debt block"));
}

TEST(RollingStepSchedulerConfig, AMalformedGroupIsRefused) {
  mpc_config::GaitFile empty = tron1Steps();
  empty.rolling_lateral_debt->step_groups.push_back(group({}));
  EXPECT_THAT(tron1Status(empty).message(),
              HasSubstr("step_groups[2].contacts is empty"));

  mpc_config::GaitFile unknown = tron1Steps();
  unknown.rolling_lateral_debt->step_groups[1] = group({"wheel_X_contact"});
  EXPECT_THAT(tron1Status(unknown).message(),
              HasSubstr("'wheel_X_contact', which is not a contact"));

  mpc_config::GaitFile twice = tron1Steps();
  twice.rolling_lateral_debt->step_groups[1] = group({kRightWheel, kLeftWheel});
  EXPECT_THAT(tron1Status(twice).message(),
              HasSubstr("which step_groups[0] already has"));

  mpc_config::GaitFile missing = tron1Steps();
  missing.rolling_lateral_debt->step_groups.pop_back();
  EXPECT_THAT(tron1Status(missing).message(),
              HasSubstr("no group with the contact 'wheel_R_contact'"));
}

TEST(RollingStepSchedulerConfig, ADurationOrDebtOutOfRangeIsRefused) {
  mpc_config::GaitFile debt = tron1Steps();
  debt.rolling_lateral_debt->max_lateral_debt = 0.0;
  EXPECT_THAT(tron1Status(debt).message(),
              HasSubstr("max_lateral_debt is 0; it must be finite"));

  mpc_config::GaitFile swing = tron1Steps();
  swing.rolling_lateral_debt->swing_duration = std::nan("");
  EXPECT_THAT(tron1Status(swing).message(), HasSubstr("swing_duration is nan"));

  mpc_config::GaitFile stance = tron1Steps();
  stance.rolling_lateral_debt->min_stance_duration = -0.1;
  EXPECT_THAT(tron1Status(stance).message(),
              HasSubstr("min_stance_duration is -0.1"));

  mpc_config::GaitFile lead = tron1Steps();
  lead.rolling_lateral_debt->min_lift_off_lead = 0.0;
  EXPECT_THAT(tron1Status(lead).message(), HasSubstr("min_lift_off_lead is 0"));

  mpc_config::GaitFile window = tron1Steps();
  window.rolling_lateral_debt->commit_window = 0.05;
  EXPECT_THAT(tron1Status(window).message(),
              HasSubstr("commit_window is 0.05; it must be finite and at least "
                        "min_lift_off_lead (0.1)"));
}

TEST(RollingStepSchedulerConfig, AContactSetWithAFixedPointIsRefused) {
  const absl::StatusOr<ModelSettings> anymalB =
      settingsOf(kAnymalBTask, kAnymalBUrdf);
  ASSERT_TRUE(anymalB.ok()) << anymalB.status();
  mpc_config::GaitFile gaitFile = tron1Steps();
  gaitFile.rolling_lateral_debt->step_groups = {
      group({anymalB->contactNames[0]})};
  EXPECT_THAT(rollingStepSchedulerSettingsFromConfig(gaitFile, *anymalB)
                  .status()
                  .message(),
              HasSubstr(absl::StrCat("the contact '", anymalB->contactNames[0],
                                     "' is not one")));
}

}  // namespace
}  // namespace ocs2::legged
