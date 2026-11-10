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
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/reference/GaitFromConfig.h"
#include "humanoid_common_mpc/gait/BipedModes.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_mpc_config/gait_file.nproto.h"
#include "humanoid_mpc_config/mode_schedule_config.nproto.h"
#include "humanoid_mpc_config/mode_sequence_template_config.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.h"

namespace ocs2::humanoid {
namespace {

constexpr char kGaitFile[] =
    "humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto";

/** The shipped gait file in this test's runfiles, or empty when it is not
 * there. */
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

/** Writes `content` to a file of the test's scratch directory and returns its
 * path. */
std::string writeGaitFile(absl::string_view name, absl::string_view content) {
  const std::string file = absl::StrCat(testing::TempDir(), "/", name);
  std::ofstream(file) << content;
  return file;
}

/** The gait `name` of `modes` switching at `times`. */
mpc_config::ModeSequenceTemplateConfig gait(std::string name,
                                            std::vector<std::string> modes,
                                            absl::Span<const double> times) {
  mpc_config::ModeSequenceTemplateConfig config;
  config.name = std::move(name);
  config.mode_sequence = std::move(modes);
  config.switching_times = Eigen::Map<const Eigen::VectorXd>(
      times.data(), static_cast<Eigen::Index>(times.size()));
  return config;
}

/** A gait file of `gaits`, each listed in gait_list. */
mpc_config::GaitFile gaitFileOf(
    std::vector<mpc_config::ModeSequenceTemplateConfig> gaits) {
  mpc_config::GaitFile file;
  for (const mpc_config::ModeSequenceTemplateConfig& config : gaits) {
    file.gait_list.push_back(config.name);
  }
  file.gaits = std::move(gaits);
  return file;
}

TEST(ModeSequenceTemplateTest, EveryShippedGaitIsValid) {
  // gaitMapFromConfig() converts every gait through
  // modeSequenceTemplateFromConfig(), which refuses one validate rejects. The
  // `skip` gait once had a switching time of 0.08 between 0.75 and 1.0 - a
  // phase of -0.67 s - and loaded without complaint.
  const std::string gaitFile = gaitFilePath();
  ASSERT_FALSE(gaitFile.empty()) << kGaitFile << " is not in the runfiles";
  const absl::StatusOr<mpc_config::GaitFile> file = loadGaitFile(gaitFile);
  ASSERT_TRUE(file.ok()) << file.status();
  const absl::StatusOr<absl::btree_map<std::string, ModeSequenceTemplate>>
      gaits = gaitMapFromConfig(*file);
  ASSERT_TRUE(gaits.ok()) << gaits.status();
  ASSERT_GT(gaits->size(), 10U) << "the gait list was not read";
  for (const std::pair<const std::string, ModeSequenceTemplate>& entry :
       *gaits) {
    const std::vector<scalar_t>& times = entry.second.switchingTimes;
    EXPECT_EQ(times.size(), entry.second.modeSequence.size() + 1)
        << entry.first;
    for (size_t i = 1; i < times.size(); ++i) {
      EXPECT_LT(times[i - 1], times[i]) << entry.first << " at " << i;
    }
  }
}

/** The biped's layout as gaitModeLayoutOf() gives it for a shipped robot: two
 * 6-DoF feet, the default legs "left" and "right", and the foot names. */
GaitModeLayout bipedLayoutWithLegs() {
  return GaitModeLayout{
      .numContacts = 2,
      .stanceMode = kBipedModeStance,
      .legs = {{.name = "left", .modeBits = kBipedModeLf},
               {.name = "right", .modeBits = kBipedModeRf}},
      .contactNames = {"left_foot_contact", "right_foot_contact"}};
}

/** A quadruped's layout: contacts and legs LF, RF, LH, RH, one contact each. */
GaitModeLayout quadrupedLayout() {
  return GaitModeLayout{
      .numContacts = 4,
      .stanceMode = 0b1111,
      .legs = {{.name = "LF", .modeBits = 0b1000},
               {.name = "RF", .modeBits = 0b0100},
               {.name = "LH", .modeBits = 0b0010},
               {.name = "RH", .modeBits = 0b0001}},
      .contactNames = {"lf_foot", "rf_foot", "lh_foot", "rh_foot"}};
}

/** The DRC Atlas sole-corner set: four point contacts, toe and heel per sole,
 * one leg per contact as testModelSettingsLegs names them. */
GaitModeLayout soleCornerLayout() {
  return GaitModeLayout{
      .numContacts = 4,
      .stanceMode = 0b1111,
      .legs = {{.name = "LF", .modeBits = 0b1000},
               {.name = "RF", .modeBits = 0b0100},
               {.name = "LH", .modeBits = 0b0010},
               {.name = "RH", .modeBits = 0b0001}},
      .contactNames = {"sole_l_toe_contact", "sole_r_toe_contact",
                       "sole_l_heel_contact", "sole_r_heel_contact"}};
}

// The shipped biped gait file read by the mode names of the contact set: every
// gait has the modes and times the FLY/RF/LF/STANCE-only reader gave it, on
// the default layout and on a biped layout with legs and contact names.
TEST(ModeSequenceTemplateTest, EveryShippedBipedGaitKeepsItsModesAndTimes) {
  constexpr size_t kFly = kBipedModeFly;
  constexpr size_t kRf = kBipedModeRf;
  constexpr size_t kLf = kBipedModeLf;
  constexpr size_t kStance = kBipedModeStance;
  const absl::btree_map<std::string, ModeSequenceTemplate> expected = {
      {"stance", ModeSequenceTemplate({0.0, 0.5}, {kStance})},
      {"left_leg", ModeSequenceTemplate({0.0, 0.5}, {kLf})},
      {"right_leg", ModeSequenceTemplate({0.0, 0.5}, {kRf})},
      {"trot", ModeSequenceTemplate({0.0, 0.5, 1.0}, {kLf, kRf})},
      {"fast_trot", ModeSequenceTemplate({0.0, 0.4, 0.8}, {kLf, kRf})},
      {"slow_trot", ModeSequenceTemplate({0.0, 0.55, 1.1}, {kLf, kRf})},
      {"slower_trot", ModeSequenceTemplate({0.0, 0.6, 1.2}, {kLf, kRf})},
      {"walk", ModeSequenceTemplate({0.0, 0.6, 0.7, 1.3, 1.4},
                                    {kLf, kStance, kRf, kStance})},
      {"fast_walk", ModeSequenceTemplate({0.0, 0.3, 0.5, 0.8, 1.0},
                                         {kLf, kStance, kRf, kStance})},
      {"slow_walk", ModeSequenceTemplate({0.0, 0.65, 0.85, 1.5, 1.7},
                                         {kLf, kStance, kRf, kStance})},
      {"very_slow_walk", ModeSequenceTemplate({0.0, 0.7, 1.1, 1.8, 2.2},
                                              {kRf, kStance, kLf, kStance})},
      {"run",
       ModeSequenceTemplate({0.0, 0.3, 0.4, 0.7, 0.8}, {kLf, kFly, kRf, kFly})},
      {"fast_run",
       ModeSequenceTemplate({0.0, 0.3, 0.4, 0.7, 0.8}, {kLf, kFly, kRf, kFly})},
      {"skip", ModeSequenceTemplate({0.0, 0.25, 0.3, 0.5, 0.75, 0.8, 1.0},
                                    {kLf, kFly, kLf, kRf, kFly, kRf})},
      {"jump", ModeSequenceTemplate({0.0, 0.3, 0.5}, {kStance, kFly})},
  };
  const std::string gaitFile = gaitFilePath();
  ASSERT_FALSE(gaitFile.empty()) << kGaitFile << " is not in the runfiles";
  const absl::StatusOr<mpc_config::GaitFile> file = loadGaitFile(gaitFile);
  ASSERT_TRUE(file.ok()) << file.status();
  for (const GaitModeLayout& layout :
       {GaitModeLayout{}, bipedLayoutWithLegs()}) {
    const absl::StatusOr<absl::btree_map<std::string, ModeSequenceTemplate>>
        gaits = gaitMapFromConfig(*file, layout);
    ASSERT_TRUE(gaits.ok()) << gaits.status();
    ASSERT_EQ(gaits->size(), expected.size());
    for (const std::pair<const std::string, ModeSequenceTemplate>& entry :
         expected) {
      ASSERT_TRUE(gaits->contains(entry.first)) << entry.first;
      const ModeSequenceTemplate& parsed = gaits->find(entry.first)->second;
      EXPECT_EQ(parsed.modeSequence, entry.second.modeSequence) << entry.first;
      EXPECT_EQ(parsed.switchingTimes, entry.second.switchingTimes)
          << entry.first;
    }
  }
}

TEST(ModeSequenceTemplateTest, TheDocumentedLayoutsLoad) {
  // The layouts of a gait (gait.textproto's gaits) and of a mode schedule
  // (reference.textproto's initial_mode_schedule), read back by their
  // conversions: the schema describes files the MPC accepts.
  const std::string gaitPath =
      writeGaitFile("documented_gait_layout.textproto",
                    "# proto-message: humanoid_mpc_config.GaitFile\n"
                    "gait_list: \"walk\"\n"
                    "gaits {\n"
                    "  name: \"walk\"\n"
                    "  mode_sequence: [\"LF\", \"RF\"]\n"
                    "  switching_times: [0.0, 0.4, 0.8]\n"
                    "}\n");
  const absl::StatusOr<mpc_config::GaitFile> gaitFile = loadGaitFile(gaitPath);
  ASSERT_TRUE(gaitFile.ok()) << gaitFile.status();
  const absl::StatusOr<absl::btree_map<std::string, ModeSequenceTemplate>>
      gaits = gaitMapFromConfig(*gaitFile);
  ASSERT_TRUE(gaits.ok()) << gaits.status();
  ASSERT_TRUE(gaits->contains("walk"));
  EXPECT_EQ(gaits->at("walk").modeSequence,
            (std::vector<size_t>{kBipedModeLf, kBipedModeRf}));
  EXPECT_EQ(gaits->at("walk").switchingTimes,
            (std::vector<scalar_t>{0.0, 0.4, 0.8}));

  const std::string referencePath =
      writeGaitFile("documented_schedule_layout.textproto",
                    "# proto-message: humanoid_mpc_config.ReferenceFile\n"
                    "initial_mode_schedule {\n"
                    "  mode_sequence: [\"STANCE\", \"LF\", \"STANCE\"]\n"
                    "  event_times: [0.5, 1.0]\n"
                    "}\n");
  const absl::StatusOr<mpc_config::ReferenceFile> referenceFile =
      loadReferenceFile(referencePath);
  ASSERT_TRUE(referenceFile.ok()) << referenceFile.status();
  const absl::StatusOr<ModeSchedule> modeSchedule =
      initialModeScheduleFromConfig(*referenceFile);
  ASSERT_TRUE(modeSchedule.ok()) << modeSchedule.status();
  EXPECT_EQ(
      modeSchedule->modeSequence,
      (std::vector<size_t>{kBipedModeStance, kBipedModeLf, kBipedModeStance}));
  EXPECT_EQ(modeSchedule->eventTimes, (std::vector<scalar_t>{0.5, 1.0}));
}

TEST(ModeSequenceTemplateTest, AFileThatCannotBeReadIsNotFound) {
  const std::string missing =
      testing::TempDir() + "/no_such_gait_file.textproto";
  EXPECT_EQ(loadGaitFile(missing).status().code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(loadReferenceFile(missing).status().code(),
            absl::StatusCode::kNotFound);
}

TEST(ModeSequenceTemplateTest,
     AGaitTheListNamesButTheFileDoesNotHaveIsRefusedNamingIt) {
  mpc_config::GaitFile file =
      gaitFileOf({gait("other", {"STANCE"}, {0.0, 0.5})});
  file.gait_list = {"walk", "other"};
  const absl::StatusOr<absl::btree_map<std::string, ModeSequenceTemplate>>
      gaits = gaitMapFromConfig(file);
  EXPECT_EQ(gaits.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(gaits.status().message(), ::testing::HasSubstr("walk"));
}

TEST(ModeSequenceTemplateTest, AnUnknownModeNameIsRefusedInsteadOfReadAsFly) {
  // An unknown name used to become mode 0, FLY, without a word: a typo in a
  // gait file lifted both feet.
  const mpc_config::ModeSequenceTemplateConfig typo =
      gait("walk", {"LF", "RFF"}, {0.0, 0.4, 0.8});
  const absl::StatusOr<ModeSequenceTemplate> converted =
      modeSequenceTemplateFromConfig(typo, "gaits[walk]");
  EXPECT_EQ(converted.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(converted.status().message(),
              ::testing::HasSubstr("'RFF' is not a mode"));
  EXPECT_THAT(converted.status().message(),
              ::testing::HasSubstr("gaits[walk]"));
  mpc_config::ModeScheduleConfig schedule;
  schedule.mode_sequence = {"STANCE", "STANCE_"};
  schedule.event_times = Eigen::VectorXd::Constant(1, 0.5);
  EXPECT_EQ(
      modeScheduleFromConfig(schedule, "initial_mode_schedule").status().code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(gaitMapFromConfig(gaitFileOf({typo})).status().code(),
            absl::StatusCode::kInvalidArgument)
      << "the map fails with its gait";
}

TEST(ModeSequenceTemplateTest, AnInvalidTemplateIsRefusedNamingWhereItIs) {
  const absl::StatusOr<ModeSequenceTemplate> converted =
      modeSequenceTemplateFromConfig(
          gait("walk", {"LF", "RF"}, {0.0, 0.9, 0.8}), "gaits[walk]");
  EXPECT_EQ(converted.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(converted.status().message(),
              ::testing::HasSubstr("gaits[walk].switching_times"));
}

TEST(MotionPhaseDefinitionTest,
     ModeNamesRoundTripAndStanceLegsMatchTheModeNumbers) {
  for (const size_t mode :
       {kBipedModeFly, kBipedModeRf, kBipedModeLf, kBipedModeStance}) {
    const absl::StatusOr<size_t> parsed =
        parseModeNumber(modeNumber2String(mode), GaitModeLayout{});
    ASSERT_TRUE(parsed.ok()) << parsed.status();
    EXPECT_EQ(*parsed, mode);
    EXPECT_EQ(bipedModeOfFootContacts(bipedFootContactsOfMode(mode)), mode);
  }
  EXPECT_EQ(bipedFootContactsOfMode(kBipedModeLf),
            (BipedFeetArray<bool>{true, false}));
  EXPECT_EQ(bipedFootContactsOfMode(kBipedModeRf),
            (BipedFeetArray<bool>{false, true}));
  EXPECT_EQ(bipedFootContactsOfMode(/*modeNumber=*/7),
            (BipedFeetArray<bool>{false, false}))
      << "a number that is not a mode";
  EXPECT_EQ(modeNumber2String(/*modeNumber=*/7), "");
  EXPECT_EQ(parseModeNumber("stance", GaitModeLayout{}).status().code(),
            absl::StatusCode::kInvalidArgument)
      << "the names are upper case";
  const absl::StatusOr<size_t> bothFeet =
      parseModeNumber("11", GaitModeLayout{});
  ASSERT_TRUE(bothFeet.ok()) << bothFeet.status();
  EXPECT_EQ(*bothFeet, kBipedModeStance) << "one digit per foot, left first";
}

// A set of four point contacts: STANCE is all four, a mode is one digit per
// contact, and the foot names LF and RF, which name no contact of it, are
// refused naming the contact count.
TEST(MotionPhaseDefinitionTest, AFourPointSetReadsStanceAndDigitMasks) {
  const GaitModeLayout fourPoints{.numContacts = 4, .stanceMode = 0b1111};
  const absl::StatusOr<size_t> stance = parseModeNumber("STANCE", fourPoints);
  ASSERT_TRUE(stance.ok()) << stance.status();
  EXPECT_EQ(*stance, 0b1111U);
  const absl::StatusOr<size_t> fly = parseModeNumber("FLY", fourPoints);
  ASSERT_TRUE(fly.ok()) << fly.status();
  EXPECT_EQ(*fly, 0U);
  const absl::StatusOr<size_t> diagonal = parseModeNumber("1001", fourPoints);
  ASSERT_TRUE(diagonal.ok()) << diagonal.status();
  EXPECT_EQ(*diagonal, 0b1001U) << "contact 0 is the most significant bit";
  for (const absl::string_view refused : {"LF", "RF", "101", "10010", "10a1"}) {
    const absl::StatusOr<size_t> mode = parseModeNumber(refused, fourPoints);
    EXPECT_EQ(mode.status().code(), absl::StatusCode::kInvalidArgument)
        << refused;
    EXPECT_THAT(mode.status().message(),
                ::testing::HasSubstr("a set of 4 contacts"))
        << refused;
  }
}

// The DRC Atlas multi-contact set (3-DoF contacts first, then the two feet)
// keeps the biped names: STANCE is the two feet, the last two contacts.
TEST(MotionPhaseDefinitionTest, ASetEndingInTheFeetKeepsTheBipedNames) {
  const GaitModeLayout feetLast{.numContacts = 6,
                                .stanceMode = kBipedModeStance};
  for (const size_t mode :
       {kBipedModeFly, kBipedModeRf, kBipedModeLf, kBipedModeStance}) {
    const absl::StatusOr<size_t> parsed =
        parseModeNumber(modeNumber2String(mode), feetLast);
    ASSERT_TRUE(parsed.ok()) << parsed.status();
    EXPECT_EQ(*parsed, mode);
  }
  const absl::StatusOr<size_t> handsAndFeet =
      parseModeNumber("110011", feetLast);
  ASSERT_TRUE(handsAndFeet.ok()) << handsAndFeet.status();
  EXPECT_EQ(*handsAndFeet, 0b110011U);
}

// On the biped the aliases keep their meaning, and the default legs and the
// foot names name the same modes.
TEST(MotionPhaseDefinitionTest, TheBipedReadsItsAliasesLegsAndContacts) {
  const GaitModeLayout biped = bipedLayoutWithLegs();
  const std::vector<std::pair<std::string, size_t>> names = {
      {"FLY", kBipedModeFly},
      {"STANCE", kBipedModeStance},
      {"LF", kBipedModeLf},
      {"RF", kBipedModeRf},
      {"left", kBipedModeLf},
      {"right", kBipedModeRf},
      {"left_right", kBipedModeStance},
      {"right_left", kBipedModeStance},
      {"left_foot_contact", kBipedModeLf},
      {"left_foot_contact+right_foot_contact", kBipedModeStance},
      {"10", kBipedModeLf},
      {"01", kBipedModeRf}};
  for (const std::pair<std::string, size_t>& name : names) {
    const absl::StatusOr<size_t> mode = parseModeNumber(name.first, biped);
    ASSERT_TRUE(mode.ok()) << name.first << ": " << mode.status();
    EXPECT_EQ(*mode, name.second) << name.first;
  }
}

TEST(MotionPhaseDefinitionTest, AQuadrupedReadsItsLegNames) {
  const GaitModeLayout quadruped = quadrupedLayout();
  const std::vector<std::pair<std::string, size_t>> names = {
      {"LF_RH", 0b1001},
      {"RF_LH", 0b0110},
      {"RH_LF", 0b1001},
      {"LF", 0b1000},
      {"RF", 0b0100},
      {"LF_LH", 0b1010},
      {"LF_RF_RH", 0b1101},
      {"LF_RF_LH_RH", 0b1111},
      {"STANCE", 0b1111},
      {"FLY", 0},
      {"lf_foot+rh_foot", 0b1001},
      {"0110", 0b0110}};
  for (const std::pair<std::string, size_t>& name : names) {
    const absl::StatusOr<size_t> mode = parseModeNumber(name.first, quadruped);
    ASSERT_TRUE(mode.ok()) << name.first << ": " << mode.status();
    EXPECT_EQ(*mode, name.second) << name.first;
  }
}

TEST(MotionPhaseDefinitionTest, TheContactFormNamesPointContactsByName) {
  const absl::StatusOr<size_t> mode = parseModeNumber(
      "sole_l_toe_contact+sole_r_heel_contact", soleCornerLayout());
  ASSERT_TRUE(mode.ok()) << mode.status();
  EXPECT_EQ(*mode, 0b1001U) << "contacts 0 and 3";
}

TEST(MotionPhaseDefinitionTest, AnUnknownOrRepeatedLegIsRefusedListingTheLegs) {
  const GaitModeLayout quadruped = quadrupedLayout();
  for (const absl::string_view refused :
       {"LF_XX", "LF__RH", "lf", "LF+RH", "lf_foot+xx", "11111"}) {
    const absl::StatusOr<size_t> mode = parseModeNumber(refused, quadruped);
    EXPECT_EQ(mode.status().code(), absl::StatusCode::kInvalidArgument)
        << refused;
    EXPECT_THAT(mode.status().message(),
                ::testing::HasSubstr("the legs LF, RF, LH, RH joined by '_'"))
        << refused;
    EXPECT_THAT(mode.status().message(),
                ::testing::HasSubstr("a set of 4 contacts"))
        << refused;
  }
  const absl::StatusOr<size_t> repeatedLeg =
      parseModeNumber("LF_RH_LF", quadruped);
  EXPECT_EQ(repeatedLeg.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(repeatedLeg.status().message(),
              ::testing::HasSubstr("names the leg 'LF' more than once; the "
                                   "legs are LF, RF, LH, RH"));
  const absl::StatusOr<size_t> repeatedContact =
      parseModeNumber("lf_foot+lf_foot", quadruped);
  EXPECT_EQ(repeatedContact.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      repeatedContact.status().message(),
      ::testing::HasSubstr("names the contact 'lf_foot' more than once"));
}

TEST(MotionPhaseDefinitionTest, ModeNamesRoundTripOnEveryLayout) {
  // The biped prints its four modes as the gait files name them, with or
  // without legs.
  for (const GaitModeLayout& biped :
       {GaitModeLayout{}, bipedLayoutWithLegs()}) {
    EXPECT_EQ(modeNumber2String(kBipedModeFly, biped), "FLY");
    EXPECT_EQ(modeNumber2String(kBipedModeRf, biped), "RF");
    EXPECT_EQ(modeNumber2String(kBipedModeLf, biped), "LF");
    EXPECT_EQ(modeNumber2String(kBipedModeStance, biped), "STANCE");
    EXPECT_EQ(modeNumber2String(/*modeNumber=*/4, biped), "")
        << "no mode of two contacts";
  }
  const GaitModeLayout quadruped = quadrupedLayout();
  EXPECT_EQ(modeNumber2String(/*modeNumber=*/0b1001, quadruped), "LF_RH");
  EXPECT_EQ(modeNumber2String(/*modeNumber=*/0b0110, quadruped), "RF_LH");
  EXPECT_EQ(modeNumber2String(/*modeNumber=*/0b1101, quadruped), "LF_RF_RH");
  EXPECT_EQ(modeNumber2String(/*modeNumber=*/0b1111, quadruped), "STANCE");
  EXPECT_EQ(modeNumber2String(/*modeNumber=*/0, quadruped), "FLY");
  EXPECT_EQ(modeNumber2String(/*modeNumber=*/16, quadruped), "");
  // Without legs, and where a mode is no union of whole legs, the digits.
  const GaitModeLayout fourPoints{.numContacts = 4, .stanceMode = 0b1111};
  EXPECT_EQ(modeNumber2String(/*modeNumber=*/0b1001, fourPoints), "1001");
  GaitModeLayout twoLegs = quadrupedLayout();
  twoLegs.legs = {{.name = "left", .modeBits = 0b1010},
                  {.name = "right", .modeBits = 0b0101}};
  EXPECT_EQ(modeNumber2String(/*modeNumber=*/0b1010, twoLegs), "left");
  EXPECT_EQ(modeNumber2String(/*modeNumber=*/0b1000, twoLegs), "1000");
  // A leg named like a foot alias on a biped-stance set prints its digits,
  // which read back as the same mode.
  const GaitModeLayout aliasLeg{.numContacts = 4,
                                .stanceMode = kBipedModeStance,
                                .legs = {{.name = "LF", .modeBits = 0b1000}},
                                .contactNames = {}};
  EXPECT_EQ(modeNumber2String(/*modeNumber=*/0b1000, aliasLeg), "1000");

  const std::vector<GaitModeLayout> layouts = {
      GaitModeLayout{}, bipedLayoutWithLegs(), quadruped, fourPoints,
      twoLegs,          soleCornerLayout(),    aliasLeg};
  for (const GaitModeLayout& layout : layouts) {
    for (size_t mode = 0; mode < numModeNumbersOf(layout.numContacts); ++mode) {
      const std::string name = modeNumber2String(mode, layout);
      const absl::StatusOr<size_t> parsed = parseModeNumber(name, layout);
      ASSERT_TRUE(parsed.ok()) << name << ": " << parsed.status();
      EXPECT_EQ(*parsed, mode) << name;
    }
  }
}

TEST(ModeSequenceTemplateTest,
     SwitchingTimesOutOfOrderAreRejectedNamingTheField) {
  const absl::StatusOr<ModeSequenceTemplate> skip =
      modeSequenceTemplateFromConfig(
          gait("skip", {"LF", "FLY", "LF", "RF", "FLY", "RF"},
               {0.0, 0.25, 0.3, 0.5, 0.75, 0.08, 1.0}),
          "gaits[skip]");
  EXPECT_EQ(skip.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(skip.status().message(),
              ::testing::HasSubstr("gaits[skip].switching_times"));
  EXPECT_THAT(skip.status().message(),
              ::testing::HasSubstr("not strictly increasing at 5"));

  // Positive control: the same gait with 0.8 in its place is valid.
  const absl::StatusOr<ModeSequenceTemplate> fixed =
      modeSequenceTemplateFromConfig(
          gait("skip", {"LF", "FLY", "LF", "RF", "FLY", "RF"},
               {0.0, 0.25, 0.3, 0.5, 0.75, 0.8, 1.0}),
          "gaits[skip]");
  EXPECT_TRUE(fixed.ok()) << fixed.status();
}

TEST(ModeSequenceTemplateTest, ARepeatedSwitchingTimeIsRejected) {
  // A zero-length mode is as meaningless as a negative one: the strict
  // inequality is the property.
  EXPECT_FALSE(
      modeSequenceTemplateFromConfig(
          gait("repeated", {"LF", "STANCE", "RF"}, {0.0, 0.5, 0.5, 1.0}),
          "gaits[repeated]")
          .ok());
}

TEST(ModeSequenceTemplateTest, AMismatchedLengthIsRejected) {
  const absl::StatusOr<ModeSequenceTemplate> tooFew =
      modeSequenceTemplateFromConfig(gait("short", {"LF", "RF"}, {0.0, 1.0}),
                                     "gaits[short]");
  EXPECT_EQ(tooFew.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(tooFew.status().message(),
              ::testing::HasSubstr("gaits[short].switching_times has 2 times"));
}

// The log line GaitSchedule writes for its default template: AbslStringify, the
// same text the stream operator wrote.
TEST(ModeSequenceTemplateTest, FormatsAsItsSwitchingTimesAndModes) {
  const ModeSequenceTemplate modeSequenceTemplate(
      /*switchingTimesInput=*/{0.0, 0.35, 0.7},
      /*modeSequenceInput=*/{kBipedModeStance, kBipedModeLf});
  EXPECT_EQ(absl::StrCat(modeSequenceTemplate),
            "Template switching times: {0, 0.35, 0.7}\nTemplate mode sequence: "
            "  {3, 2}\n");
}

}  // namespace
}  // namespace ocs2::humanoid
