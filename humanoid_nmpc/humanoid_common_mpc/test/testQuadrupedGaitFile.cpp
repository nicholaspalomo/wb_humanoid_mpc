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

// The shared quadruped gait file (config/command/quadruped_gait.textproto) read
// on a four-leg layout, the gait schedule tiling its trot, and its gait
// switching table.

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/reference/GaitFromConfig.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/reference_manager/GaitSwitchingTable.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_mpc_config/gait_file.nproto.h"

namespace ocs2::humanoid {
namespace {

constexpr char kQuadrupedGaitFile[] =
    "humanoid_nmpc/humanoid_common_mpc/config/command/quadruped_gait.textproto";

// The mode bits of the legs for contacts ordered LF, RF, LH, RH, as the OCS2
// legged_robot example numbers them.
constexpr size_t kLf = 8;
constexpr size_t kRf = 4;
constexpr size_t kLh = 2;
constexpr size_t kRh = 1;
constexpr size_t kStance = kLf | kRf | kLh | kRh;
constexpr size_t kFly = 0;

/** The quadruped gait file in this test's runfiles, or empty when it is not
 * there. */
std::string quadrupedGaitFilePath() {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
    roots.emplace_back(std::filesystem::path(srcDir) / "wb_humanoid_mpc");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / kQuadrupedGaitFile;
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

/** Contacts and legs LF, RF, LH, RH, one contact per leg, standing on all. */
GaitModeLayout quadrupedLayout() {
  return GaitModeLayout{
      .numContacts = 4,
      .stanceMode = kStance,
      .legs = {{.name = "LF", .modeBits = kLf},
               {.name = "RF", .modeBits = kRf},
               {.name = "LH", .modeBits = kLh},
               {.name = "RH", .modeBits = kRh}},
      .contactNames = {"LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"}};
}

/** Every gait of the quadruped file on quadrupedLayout(). */
absl::StatusOr<std::map<std::string, ModeSequenceTemplate>> quadrupedGaits() {
  const std::string path = quadrupedGaitFilePath();
  if (path.empty()) {
    return absl::NotFoundError(
        absl::StrCat(kQuadrupedGaitFile, " is not in the runfiles"));
  }
  absl::StatusOr<mpc_config::GaitFile> file = loadGaitFile(path);
  if (!file.ok()) return file.status();
  return gaitMapFromConfig(*file, quadrupedLayout());
}

TEST(QuadrupedGaitFileTest, EveryGaitHasTheModesAndTimesOfTheOcs2Example) {
  const std::map<std::string, ModeSequenceTemplate> expected = {
      {"stance", ModeSequenceTemplate({0.0, 0.5}, {kStance})},
      {"trot", ModeSequenceTemplate({0.0, 0.35, 0.70}, {kLf | kRh, kRf | kLh})},
      {"standing_trot",
       ModeSequenceTemplate({0.0, 0.30, 0.35, 0.65, 0.70},
                            {kLf | kRh, kStance, kRf | kLh, kStance})},
      {"flying_trot", ModeSequenceTemplate({0.0, 0.27, 0.30, 0.57, 0.60},
                                           {kLf | kRh, kFly, kRf | kLh, kFly})},
      {"pace", ModeSequenceTemplate({0.0, 0.28, 0.30, 0.58, 0.60},
                                    {kLf | kLh, kFly, kRf | kRh, kFly})},
      {"static_walk", ModeSequenceTemplate({0.0, 0.3, 0.6, 0.9, 1.2},
                                           {kLf | kRf | kRh, kRf | kLh | kRh,
                                            kLf | kRf | kLh, kLf | kLh | kRh})},
      {"dynamic_walk",
       ModeSequenceTemplate({0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0},
                            {kLf | kRf | kRh, kRf | kRh, kRf | kLh | kRh,
                             kLf | kRf | kLh, kLf | kLh, kLf | kLh | kRh})},
  };
  const absl::StatusOr<std::map<std::string, ModeSequenceTemplate>> gaits =
      quadrupedGaits();
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
  EXPECT_EQ(gaits->find("trot")->second.modeSequence,
            (std::vector<size_t>{9, 6}));
}

TEST(QuadrupedGaitFileTest, LfIsTheLeftFrontLegOnTheQuadruped) {
  // The foot aliases are the biped's only: on the quadruped LF is the leg, so
  // a biped trot LF, RF would stand on one leg at a time.
  const absl::StatusOr<size_t> lf = parseModeNumber("LF", quadrupedLayout());
  ASSERT_TRUE(lf.ok()) << lf.status();
  EXPECT_EQ(*lf, kLf);
}

TEST(QuadrupedGaitFileTest, TheScheduleStandsOnAllFourAndTilesTheTrot) {
  const absl::StatusOr<std::map<std::string, ModeSequenceTemplate>> gaits =
      quadrupedGaits();
  ASSERT_TRUE(gaits.ok()) << gaits.status();
  constexpr scalar_t kTransitionStanceTime = 0.2;
  GaitSchedule schedule(ModeSchedule(/*eventTimesInput=*/{0.5},
                                     /*modeSequenceInput=*/{kStance, kStance}),
                        gaits->find("stance")->second, kTransitionStanceTime,
                        quadrupedLayout());
  EXPECT_EQ(schedule.modeLayout().stanceMode, kStance);
  schedule.getModeSchedule(/*lowerBoundTime=*/0.0, /*upperBoundTime=*/2.0);
  // The trot follows the stance at once (no transition after a stance); the
  // static walk follows the trot after a transition stance on all four.
  constexpr scalar_t kTrotStart = 1.0;
  constexpr scalar_t kWalkStart = 2.0;
  constexpr scalar_t kHorizonEnd = 4.0;
  schedule.insertModeSequenceTemplate(gaits->find("trot")->second, kTrotStart,
                                      kHorizonEnd);
  const ModeSchedule trotting = schedule.getCurrentModeSchedule();
  for (const size_t mode : trotting.modeSequence) {
    EXPECT_TRUE(mode == kStance || mode == 9 || mode == 6) << mode;
  }
  schedule.insertModeSequenceTemplate(gaits->find("static_walk")->second,
                                      kWalkStart, kHorizonEnd);
  const ModeSchedule tiled = schedule.getCurrentModeSchedule();
  EXPECT_EQ(tiled.modeAtTime(0.95), kStance);
  // LF_RH and RF_LH alternate every 0.35 s from the trot's start.
  constexpr scalar_t kHalfCycle = 0.35;
  for (size_t i = 0; i < 3; ++i) {
    const scalar_t midPhase =
        kTrotStart + (static_cast<scalar_t>(i) + 0.5) * kHalfCycle;
    EXPECT_EQ(tiled.modeAtTime(midPhase), i % 2 == 0 ? 9U : 6U)
        << "phase " << i;
  }
  EXPECT_EQ(tiled.modeAtTime(kWalkStart + 0.5 * kTransitionStanceTime), kStance)
      << "the transition stance is all four";
  EXPECT_EQ(tiled.modeAtTime(kWalkStart + kTransitionStanceTime + 0.1),
            kLf | kRf | kRh);
  EXPECT_EQ(tiled.modeSequence.back(), kStance);
}

TEST(QuadrupedGaitFileTest, ItsSwitchingTableNamesItsOwnGaits) {
  const std::string path = quadrupedGaitFilePath();
  ASSERT_FALSE(path.empty()) << kQuadrupedGaitFile << " is not in the runfiles";
  const absl::StatusOr<mpc_config::GaitFile> file = loadGaitFile(path);
  ASSERT_TRUE(file.ok()) << file.status();
  ASSERT_TRUE(file->gait_switching.has_value())
      << "the biped default table names gaits the quadruped file lacks";
  const absl::StatusOr<std::vector<GaitModeStateConfig>> table =
      gaitModeStatesFromConfig(*file);
  ASSERT_TRUE(table.ok()) << table.status();
  std::vector<std::string> names;
  for (const GaitModeStateConfig& row : *table) {
    names.push_back(row.gaitCommand);
  }
  EXPECT_EQ(names, (std::vector<std::string>{"stance", "static_walk", "trot",
                                             "flying_trot"}));
  const absl::StatusOr<std::map<std::string, ModeSequenceTemplate>> gaits =
      gaitMapFromConfig(*file, quadrupedLayout());
  ASSERT_TRUE(gaits.ok()) << gaits.status();
  EXPECT_TRUE(ProceduralMpcMotionManager::checkEveryGaitIsLoaded(
                  *gaits, *table, kQuadrupedGaitFile)
                  .ok());
  // Each row switches up before the next one would switch back down.
  for (size_t i = 1; i < table->size(); ++i) {
    SCOPED_TRACE((*table)[i].gaitCommand);
    EXPECT_LT((*table)[i].cmdMinLinvelBaseRtWorldEwrtHeadingXy,
              (*table)[i - 1].cmdMaxLinvelBaseRtWorldEwrtHeadingXy);
    EXPECT_LT((*table)[i].cmdMinAngvelBaseRtWorldEwrtWorldZ,
              (*table)[i - 1].cmdMaxAngvelBaseRtWorldEwrtWorldZ);
  }
}

}  // namespace
}  // namespace ocs2::humanoid
