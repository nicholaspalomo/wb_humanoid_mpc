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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/reference/GaitFromConfig.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_mpc_config/gait_file.nproto.h"
#include "humanoid_mpc_config/mode_schedule_config.nproto.h"
#include "humanoid_mpc_config/mode_sequence_template_config.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.h"

namespace ocs2::humanoid {
namespace {

constexpr char kGaitFile[] = "humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto";

/** The shipped gait file in this test's runfiles, or empty when it is not there. */
std::string gaitFilePath() {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
    roots.emplace_back(std::filesystem::path(srcDir) / "wb_humanoid_mpc");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / kGaitFile;
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

/** Writes `content` to a file of the test's scratch directory and returns its path. */
std::string writeGaitFile(const std::string& name, const std::string& content) {
  const std::string file = absl::StrCat(testing::TempDir(), "/", name);
  std::ofstream(file) << content;
  return file;
}

/** The gait `name` of `modes` switching at `times`. */
mpc_config::ModeSequenceTemplateConfig gait(const std::string& name,
                                            const std::vector<std::string>& modes,
                                            const std::vector<double>& times) {
  mpc_config::ModeSequenceTemplateConfig config;
  config.name = name;
  config.mode_sequence = modes;
  config.switching_times = Eigen::Map<const Eigen::VectorXd>(times.data(), static_cast<Eigen::Index>(times.size()));
  return config;
}

/** A gait file of `gaits`, each listed in gait_list. */
mpc_config::GaitFile gaitFileOf(const std::vector<mpc_config::ModeSequenceTemplateConfig>& gaits) {
  mpc_config::GaitFile file;
  for (const mpc_config::ModeSequenceTemplateConfig& config : gaits) file.gait_list.push_back(config.name);
  file.gaits = gaits;
  return file;
}

TEST(ModeSequenceTemplateTest, EveryShippedGaitIsValid) {
  // gaitMapFromConfig() converts every gait through modeSequenceTemplateFromConfig(), which refuses one validate
  // rejects. The `skip` gait once had a switching time of 0.08 between 0.75 and 1.0 - a phase of -0.67 s - and loaded
  // without complaint.
  const std::string gaitFile = gaitFilePath();
  ASSERT_FALSE(gaitFile.empty()) << kGaitFile << " is not in the runfiles";
  const absl::StatusOr<mpc_config::GaitFile> file = loadGaitFile(gaitFile);
  ASSERT_TRUE(file.ok()) << file.status();
  const absl::StatusOr<std::map<std::string, ModeSequenceTemplate>> gaits = gaitMapFromConfig(*file);
  ASSERT_TRUE(gaits.ok()) << gaits.status();
  ASSERT_GT(gaits->size(), 10U) << "the gait list was not read";
  for (const std::pair<const std::string, ModeSequenceTemplate>& entry : *gaits) {
    const std::vector<scalar_t>& times = entry.second.switchingTimes;
    EXPECT_EQ(times.size(), entry.second.modeSequence.size() + 1) << entry.first;
    for (size_t i = 1; i < times.size(); ++i) {
      EXPECT_LT(times[i - 1], times[i]) << entry.first << " at " << i;
    }
  }
}

TEST(ModeSequenceTemplateTest, TheDocumentedLayoutsLoad) {
  // The layouts of a gait (gait.textproto's gaits) and of a mode schedule (reference.textproto's
  // initial_mode_schedule), read back by their conversions: the schema describes files the MPC accepts.
  const std::string gaitPath = writeGaitFile("documented_gait_layout.textproto",
                                             "gait_list: \"walk\"\n"
                                             "gaits {\n"
                                             "  name: \"walk\"\n"
                                             "  mode_sequence: [\"LF\", \"RF\"]\n"
                                             "  switching_times: [0.0, 0.4, 0.8]\n"
                                             "}\n");
  const absl::StatusOr<mpc_config::GaitFile> gaitFile = loadGaitFile(gaitPath);
  ASSERT_TRUE(gaitFile.ok()) << gaitFile.status();
  const absl::StatusOr<std::map<std::string, ModeSequenceTemplate>> gaits = gaitMapFromConfig(*gaitFile);
  ASSERT_TRUE(gaits.ok()) << gaits.status();
  ASSERT_TRUE(gaits->contains("walk"));
  EXPECT_EQ(gaits->at("walk").modeSequence, (std::vector<size_t>{ModeNumber::kLf, ModeNumber::kRf}));
  EXPECT_EQ(gaits->at("walk").switchingTimes, (std::vector<scalar_t>{0.0, 0.4, 0.8}));

  const std::string referencePath = writeGaitFile("documented_schedule_layout.textproto",
                                                  "initial_mode_schedule {\n"
                                                  "  mode_sequence: [\"STANCE\", \"LF\", \"STANCE\"]\n"
                                                  "  event_times: [0.5, 1.0]\n"
                                                  "}\n");
  const absl::StatusOr<mpc_config::ReferenceFile> referenceFile = loadReferenceFile(referencePath);
  ASSERT_TRUE(referenceFile.ok()) << referenceFile.status();
  const absl::StatusOr<ModeSchedule> modeSchedule = initialModeScheduleFromConfig(*referenceFile);
  ASSERT_TRUE(modeSchedule.ok()) << modeSchedule.status();
  EXPECT_EQ(modeSchedule->modeSequence, (std::vector<size_t>{ModeNumber::kStance, ModeNumber::kLf, ModeNumber::kStance}));
  EXPECT_EQ(modeSchedule->eventTimes, (std::vector<scalar_t>{0.5, 1.0}));
}

TEST(ModeSequenceTemplateTest, AFileThatCannotBeReadIsNotFound) {
  const std::string missing = testing::TempDir() + "/no_such_gait_file.textproto";
  EXPECT_EQ(loadGaitFile(missing).status().code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(loadReferenceFile(missing).status().code(), absl::StatusCode::kNotFound);
}

TEST(ModeSequenceTemplateTest, AGaitTheListNamesButTheFileDoesNotHaveIsRefusedNamingIt) {
  mpc_config::GaitFile file = gaitFileOf({gait("other", {"STANCE"}, {0.0, 0.5})});
  file.gait_list = {"walk", "other"};
  const absl::StatusOr<std::map<std::string, ModeSequenceTemplate>> gaits = gaitMapFromConfig(file);
  EXPECT_EQ(gaits.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(gaits.status().message(), ::testing::HasSubstr("walk"));
}

TEST(ModeSequenceTemplateTest, AnUnknownModeNameIsRefusedInsteadOfReadAsFly) {
  // An unknown name used to become mode 0, FLY, without a word: a typo in a gait file lifted both feet.
  const mpc_config::ModeSequenceTemplateConfig typo = gait("walk", {"LF", "RFF"}, {0.0, 0.4, 0.8});
  const absl::StatusOr<ModeSequenceTemplate> converted = modeSequenceTemplateFromConfig(typo, "gaits[walk]");
  EXPECT_EQ(converted.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(converted.status().message(), ::testing::HasSubstr("'RFF' is not a mode"));
  EXPECT_THAT(converted.status().message(), ::testing::HasSubstr("gaits[walk]"));
  mpc_config::ModeScheduleConfig schedule;
  schedule.mode_sequence = {"STANCE", "STANCE_"};
  schedule.event_times = Eigen::VectorXd::Constant(1, 0.5);
  EXPECT_EQ(modeScheduleFromConfig(schedule, "initial_mode_schedule").status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(gaitMapFromConfig(gaitFileOf({typo})).status().code(), absl::StatusCode::kInvalidArgument) << "the map fails with its gait";
}

TEST(ModeSequenceTemplateTest, AnInvalidTemplateIsRefusedNamingWhereItIs) {
  const absl::StatusOr<ModeSequenceTemplate> converted =
      modeSequenceTemplateFromConfig(gait("walk", {"LF", "RF"}, {0.0, 0.9, 0.8}), "gaits[walk]");
  EXPECT_EQ(converted.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(converted.status().message(), ::testing::HasSubstr("gaits[walk].switching_times"));
}

TEST(MotionPhaseDefinitionTest, ModeNamesRoundTripAndStanceLegsMatchTheModeNumbers) {
  for (const ModeNumber mode : {ModeNumber::kFly, ModeNumber::kRf, ModeNumber::kLf, ModeNumber::kStance}) {
    const absl::StatusOr<size_t> parsed = parseModeNumber(modeNumber2String(mode));
    ASSERT_TRUE(parsed.ok()) << parsed.status();
    EXPECT_EQ(*parsed, static_cast<size_t>(mode));
    EXPECT_EQ(stanceLeg2ModeNumber(modeNumber2StanceLeg(mode)), static_cast<size_t>(mode));
  }
  EXPECT_EQ(modeNumber2StanceLeg(ModeNumber::kLf), (contact_flag_t{true, false}));
  EXPECT_EQ(modeNumber2StanceLeg(ModeNumber::kRf), (contact_flag_t{false, true}));
  EXPECT_EQ(modeNumber2StanceLeg(/*modeNumber=*/7), (contact_flag_t{false, false})) << "a number that is not a mode";
  EXPECT_EQ(modeNumber2String(/*modeNumber=*/7), "");
  EXPECT_EQ(parseModeNumber("stance").status().code(), absl::StatusCode::kInvalidArgument) << "the names are upper case";
}

TEST(ModeSequenceTemplateTest, SwitchingTimesOutOfOrderAreRejectedNamingTheField) {
  const absl::StatusOr<ModeSequenceTemplate> skip = modeSequenceTemplateFromConfig(
      gait("skip", {"LF", "FLY", "LF", "RF", "FLY", "RF"}, {0.0, 0.25, 0.3, 0.5, 0.75, 0.08, 1.0}), "gaits[skip]");
  EXPECT_EQ(skip.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(skip.status().message(), ::testing::HasSubstr("gaits[skip].switching_times"));
  EXPECT_THAT(skip.status().message(), ::testing::HasSubstr("not strictly increasing at 5"));

  // Positive control: the same gait with 0.8 in its place is valid.
  const absl::StatusOr<ModeSequenceTemplate> fixed = modeSequenceTemplateFromConfig(
      gait("skip", {"LF", "FLY", "LF", "RF", "FLY", "RF"}, {0.0, 0.25, 0.3, 0.5, 0.75, 0.8, 1.0}), "gaits[skip]");
  EXPECT_TRUE(fixed.ok()) << fixed.status();
}

TEST(ModeSequenceTemplateTest, ARepeatedSwitchingTimeIsRejected) {
  // A zero-length mode is as meaningless as a negative one: the strict inequality is the property.
  EXPECT_FALSE(modeSequenceTemplateFromConfig(gait("repeated", {"LF", "STANCE", "RF"}, {0.0, 0.5, 0.5, 1.0}), "gaits[repeated]").ok());
}

TEST(ModeSequenceTemplateTest, AMismatchedLengthIsRejected) {
  const absl::StatusOr<ModeSequenceTemplate> tooFew =
      modeSequenceTemplateFromConfig(gait("short", {"LF", "RF"}, {0.0, 1.0}), "gaits[short]");
  EXPECT_EQ(tooFew.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(tooFew.status().message(), ::testing::HasSubstr("gaits[short].switching_times has 2 times"));
}

}  // namespace
}  // namespace ocs2::humanoid
