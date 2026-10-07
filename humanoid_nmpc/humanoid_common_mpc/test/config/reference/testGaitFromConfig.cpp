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

// The gait conversions: a gait pattern and a mode schedule are read by mode name and checked as ModeSequenceTemplate
// and ModeSchedule check them, the gait file's gait_list and gaits name each other exactly, and the reference file's start
// of the gait schedule is read from its two blocks.

#include <cstddef>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/reference/GaitFromConfig.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_mpc_config/gait_file.nproto.h"
#include "humanoid_mpc_config/mode_schedule_config.nproto.h"
#include "humanoid_mpc_config/mode_sequence_template_config.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.h"

namespace ocs2::humanoid {
namespace {

Eigen::VectorXd times(const std::vector<double>& values) {
  return Eigen::Map<const Eigen::VectorXd>(values.data(), static_cast<Eigen::Index>(values.size()));
}

mpc_config::ModeSequenceTemplateConfig pattern(absl::string_view name,
                                               std::vector<std::string> modes,
                                               const std::vector<double>& switching) {
  return mpc_config::ModeSequenceTemplateConfig{
      .name = std::string(name), .mode_sequence = std::move(modes), .switching_times = times(switching)};
}

mpc_config::GaitFile gaitFile(std::vector<std::string> list, std::vector<mpc_config::ModeSequenceTemplateConfig> gaits) {
  return mpc_config::GaitFile{.gait_list = std::move(list), .gaits = std::move(gaits)};
}

mpc_config::ModeSequenceTemplateConfig stance() {
  return pattern("stance", {"STANCE"}, {0.0, 0.5});
}

mpc_config::ModeSequenceTemplateConfig trot() {
  return pattern("trot", {"LF", "RF"}, {0.0, 0.5, 1.0});
}

TEST(ModeSequenceTemplateFromConfigTest, TheModesAndTimesAreThePatterns) {
  const absl::StatusOr<ModeSequenceTemplate> converted = modeSequenceTemplateFromConfig(trot(), "gaits[trot]");
  ASSERT_TRUE(converted.ok()) << converted.status();
  EXPECT_EQ(converted->modeSequence, (std::vector<size_t>{ModeNumber::kLf, ModeNumber::kRf}));
  EXPECT_EQ(converted->switchingTimes, (std::vector<scalar_t>{0.0, 0.5, 1.0}));
}

TEST(ModeSequenceTemplateFromConfigTest, ItAcceptsExactlyTheTemplatesThatDescribeAGait) {
  // At least one mode, one more switching time than modes, and the times strictly increasing.
  struct Case {
    std::vector<std::string> modes;
    std::vector<double> times;
    bool accepted;
  };
  const Case cases[] = {
      {.modes = {"STANCE"}, .times = {0.0, 0.5}, .accepted = true},
      {.modes = {"LF", "RF"}, .times = {0.0, 0.5, 1.0}, .accepted = true},
      {.modes = {"LF", "RF"}, .times = {0.0, 1.0}, .accepted = false},
      {.modes = {"LF"}, .times = {0.0, 0.5, 1.0}, .accepted = false},
      {.modes = {"LF", "RF"}, .times = {0.0, 0.5, 0.5}, .accepted = false},
      {.modes = {"LF", "RF"}, .times = {0.0, 0.7, 0.5}, .accepted = false},
      {.modes = {"STANCE"}, .times = {}, .accepted = false},
      {.modes = {}, .times = {0.0}, .accepted = false},
      {.modes = {}, .times = {}, .accepted = false},
      {.modes = {"STANCE", "FLY"}, .times = {0.3, 0.4, 0.6}, .accepted = true},
  };
  for (const Case& testCase : cases) {
    const absl::StatusOr<ModeSequenceTemplate> converted =
        modeSequenceTemplateFromConfig(pattern("x", testCase.modes, testCase.times), "gaits[x]");
    EXPECT_EQ(converted.ok(), testCase.accepted) << converted.status();
    if (!converted.ok()) {
      EXPECT_EQ(converted.status().code(), absl::StatusCode::kInvalidArgument);
      EXPECT_TRUE(absl::StartsWith(converted.status().message(), "gaits[x].")) << converted.status();
    }
  }
}

TEST(ModeSequenceTemplateFromConfigTest, AModeNameThatIsNoModeIsRefusedWithItsIndex) {
  const absl::StatusOr<ModeSequenceTemplate> converted =
      modeSequenceTemplateFromConfig(pattern("hop", {"LF", "LEFT"}, {0.0, 0.5, 1.0}), "gaits[hop]");
  EXPECT_EQ(converted.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(converted.status().message(), "gaits[hop].mode_sequence[1]: 'LEFT' is not a mode")) << converted.status();
}

TEST(ModeSequenceTemplateFromConfigTest, ATimeThatIsNotAFiniteNumberIsRefused) {
  for (const double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    const absl::StatusOr<ModeSequenceTemplate> converted =
        modeSequenceTemplateFromConfig(pattern("x", {"STANCE"}, {0.0, bad}), "default_mode_sequence_template");
    EXPECT_TRUE(absl::StrContains(converted.status().message(), "default_mode_sequence_template.switching_times[1]")) << converted.status();
  }
}

TEST(ModeScheduleFromConfigTest, TheScheduleHasOneEventTimeFewerThanModes) {
  mpc_config::ModeScheduleConfig config{.mode_sequence = {"STANCE", "LF"}, .event_times = times({0.5})};
  const absl::StatusOr<ModeSchedule> schedule = modeScheduleFromConfig(config, "initial_mode_schedule");
  ASSERT_TRUE(schedule.ok()) << schedule.status();
  EXPECT_EQ(schedule->modeSequence, (std::vector<size_t>{ModeNumber::kStance, ModeNumber::kLf}));
  EXPECT_EQ(schedule->eventTimes, (std::vector<scalar_t>{0.5}));

  config.event_times = times({0.5, 1.0});
  EXPECT_TRUE(absl::StrContains(modeScheduleFromConfig(config, "initial_mode_schedule").status().message(),
                                "initial_mode_schedule.event_times has 2 times for the 2 modes"));
  config.event_times = times({std::numeric_limits<double>::infinity()});
  EXPECT_TRUE(absl::StrContains(modeScheduleFromConfig(config, "initial_mode_schedule").status().message(),
                                "initial_mode_schedule.event_times[0]"));
  const mpc_config::ModeScheduleConfig empty;
  EXPECT_TRUE(absl::StrContains(modeScheduleFromConfig(empty, "initial_mode_schedule").status().message(),
                                "initial_mode_schedule.mode_sequence is empty"));
}

TEST(GaitMapFromConfigTest, EveryListedGaitByName) {
  const absl::StatusOr<std::map<std::string, ModeSequenceTemplate>> gaits =
      gaitMapFromConfig(gaitFile({"trot", "stance"}, {stance(), trot()}));
  ASSERT_TRUE(gaits.ok()) << gaits.status();
  ASSERT_EQ(gaits->size(), size_t{2});
  EXPECT_EQ(gaits->at("trot").switchingTimes, (std::vector<scalar_t>{0.0, 0.5, 1.0}));
  EXPECT_EQ(gaits->at("stance").modeSequence, (std::vector<size_t>{ModeNumber::kStance}));
}

TEST(GaitMapFromConfigTest, AnEmptyFileHasNoGaits) {
  const absl::StatusOr<std::map<std::string, ModeSequenceTemplate>> gaits = gaitMapFromConfig(mpc_config::GaitFile{});
  ASSERT_TRUE(gaits.ok()) << gaits.status();
  EXPECT_TRUE(gaits->empty()) << "the motion manager refuses a file without the gaits it selects";
}

TEST(GaitMapFromConfigTest, TheListAndTheGaitsNameEachOtherExactly) {
  const std::pair<mpc_config::GaitFile, absl::string_view> refused[] = {
      {gaitFile({"stance", "stance"}, {stance()}), "gait_list names the gait 'stance' more than once"},
      {gaitFile({"stance", "trot"}, {stance()}), "gait_list names the gait 'trot', which gaits does not define"},
      {gaitFile({"stance"}, {stance(), trot()}), "gaits[trot] is not named in gait_list"},
      {gaitFile({"stance"}, {stance(), stance()}), "gaits defines the gait 'stance' more than once"},
      {gaitFile({"stance"}, {stance(), pattern(/*name=*/"", {"STANCE"}, {0.0, 1.0})}), "gaits[1] has no name"},
      {gaitFile({"trot"}, {pattern("trot", {"LF", "RF"}, {0.0, 0.5})}), "gaits[trot].switching_times has 2 times"},
  };
  for (const std::pair<mpc_config::GaitFile, absl::string_view>& file : refused) {
    const absl::StatusOr<std::map<std::string, ModeSequenceTemplate>> gaits = gaitMapFromConfig(file.first);
    EXPECT_EQ(gaits.status().code(), absl::StatusCode::kInvalidArgument) << file.second;
    EXPECT_TRUE(absl::StrContains(gaits.status().message(), file.second)) << gaits.status();
  }
}

TEST(GaitScheduleStartFromConfigTest, TheReferenceFilesTwoBlocksNameThemselvesInTheirErrors) {
  mpc_config::ReferenceFile file;
  EXPECT_TRUE(absl::StartsWith(initialModeScheduleFromConfig(file).status().message(), "initial_mode_schedule."));
  EXPECT_TRUE(absl::StartsWith(defaultModeSequenceTemplateFromConfig(file).status().message(), "default_mode_sequence_template."));

  // One mode is a schedule, but not one the gait schedule can start from.
  file.initial_mode_schedule = mpc_config::ModeScheduleConfig{.mode_sequence = {"STANCE"}, .event_times = times(/*values=*/{})};
  EXPECT_TRUE(modeScheduleFromConfig(file.initial_mode_schedule, "initial_mode_schedule").ok());
  EXPECT_TRUE(absl::StrContains(initialModeScheduleFromConfig(file).status().message(), "needs at least two modes"));

  file.initial_mode_schedule = mpc_config::ModeScheduleConfig{.mode_sequence = {"STANCE", "STANCE"}, .event_times = times({0.5})};
  file.default_mode_sequence_template = pattern(/*name=*/"", {"STANCE"}, {0.0, 0.5});
  EXPECT_TRUE(initialModeScheduleFromConfig(file).ok());
  const absl::StatusOr<ModeSequenceTemplate> defaultTemplate = defaultModeSequenceTemplateFromConfig(file);
  ASSERT_TRUE(defaultTemplate.ok()) << defaultTemplate.status();
  EXPECT_EQ(defaultTemplate->switchingTimes, (std::vector<scalar_t>{0.0, 0.5}));

  // The default template is no gait of the gait file: a name would mean nothing.
  file.default_mode_sequence_template.name = "walk";
  EXPECT_TRUE(absl::StrContains(defaultModeSequenceTemplateFromConfig(file).status().message(), "default_mode_sequence_template.name"));
}

}  // namespace
}  // namespace ocs2::humanoid
