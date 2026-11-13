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

// The shared gait file of the wheeled robots
// (config/command/wheeled_gait.textproto) read on a two-wheel and a four-wheel
// layout: every gait stands on every contact, and its switching table drives
// on a forward or a turning command and stands at rest.

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/command/BasePlanarMotion.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/reference/GaitFromConfig.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/reference_manager/GaitSwitchingTable.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_mpc_config/gait_file.nproto.h"

namespace ocs2::humanoid {
namespace {

constexpr char kWheeledGaitFile[] =
    "humanoid_nmpc/humanoid_common_mpc/config/command/wheeled_gait.textproto";

// [m] A base height for the velocity command, which the gait switch does not
// read.
constexpr scalar_t kCommandHeight = 0.7;
// [m/s] and [rad/s] The commands of the closed-loop drive and turn.
constexpr scalar_t kDriveSpeed = 0.5;
constexpr scalar_t kTurnRate = 0.5;

/** The wheeled gait file in this test's runfiles, or empty when it is not
 * there. */
std::string wheeledGaitFilePath() {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
    roots.emplace_back(std::filesystem::path(srcDir) / "wb_humanoid_mpc");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / kWheeledGaitFile;
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

/** Two wheels on legs left and right, as on the LimX Tron1. */
GaitModeLayout twoWheelLayout() {
  return GaitModeLayout{.numContacts = 2,
                        .stanceMode = 3,
                        .legs = {{.name = "left", .modeBits = 2},
                                 {.name = "right", .modeBits = 1}},
                        .contactNames = {"wheel_L_contact", "wheel_R_contact"}};
}

/** Four wheels on legs LF, RF, LH and RH, as on the wheeled ANYmal C. */
GaitModeLayout fourWheelLayout() {
  return GaitModeLayout{
      .numContacts = 4,
      .stanceMode = 15,
      .legs = {{.name = "LF", .modeBits = 8},
               {.name = "RF", .modeBits = 4},
               {.name = "LH", .modeBits = 2},
               {.name = "RH", .modeBits = 1}},
      .contactNames = {"LF_WHEEL", "RF_WHEEL", "LH_WHEEL", "RH_WHEEL"}};
}

/** The wheeled gait file, read strictly. */
absl::StatusOr<mpc_config::GaitFile> wheeledGaitFile() {
  const std::string path = wheeledGaitFilePath();
  if (path.empty()) {
    return absl::NotFoundError(
        absl::StrCat(kWheeledGaitFile, " is not in the runfiles"));
  }
  return loadGaitFile(path);
}

/** The velocity command [v_x, v_y, height, yaw rate]. */
vector4_t command(scalar_t linvelX, scalar_t angvelZ) {
  return vector4_t(linvelX, 0.0, kCommandHeight, angvelZ);
}

TEST(WheeledGaitFileTest, EveryGaitStandsOnEveryContactOfEitherLayout) {
  const absl::StatusOr<mpc_config::GaitFile> file = wheeledGaitFile();
  ASSERT_TRUE(file.ok()) << file.status();
  for (const GaitModeLayout& layout : {twoWheelLayout(), fourWheelLayout()}) {
    SCOPED_TRACE(absl::StrCat(layout.numContacts, " contacts"));
    const absl::StatusOr<absl::btree_map<std::string, ModeSequenceTemplate>>
        gaits = gaitMapFromConfig(*file, layout);
    ASSERT_TRUE(gaits.ok()) << gaits.status();
    ASSERT_EQ(gaits->size(), 2U);
    for (const std::string name : {"stance", "drive"}) {
      ASSERT_TRUE(gaits->contains(name)) << name;
      const ModeSequenceTemplate& gait = gaits->find(name)->second;
      EXPECT_EQ(gait.modeSequence, std::vector<size_t>{layout.stanceMode})
          << name;
      EXPECT_EQ(gait.switchingTimes, (std::vector<scalar_t>{0.0, 0.5})) << name;
    }
  }
}

TEST(WheeledGaitFileTest, ItsSwitchingTableIsStanceThenDrive) {
  const absl::StatusOr<mpc_config::GaitFile> file = wheeledGaitFile();
  ASSERT_TRUE(file.ok()) << file.status();
  ASSERT_TRUE(file->gait_switching.has_value())
      << "the biped default table names gaits the wheeled file lacks";
  const absl::StatusOr<std::vector<GaitModeStateConfig>> table =
      gaitModeStatesFromConfig(*file);
  ASSERT_TRUE(table.ok()) << table.status();
  ASSERT_EQ(table->size(), 2U);
  EXPECT_EQ((*table)[0].gaitCommand, "stance");
  EXPECT_EQ((*table)[1].gaitCommand, "drive");
  const absl::StatusOr<absl::btree_map<std::string, ModeSequenceTemplate>>
      gaits = gaitMapFromConfig(*file, twoWheelLayout());
  ASSERT_TRUE(gaits.ok()) << gaits.status();
  EXPECT_TRUE(ProceduralMpcMotionManager::checkEveryGaitIsLoaded(
                  *gaits, *table, kWheeledGaitFile)
                  .ok());
  // drive switches down below the bounds stance switches up above.
  EXPECT_LT((*table)[1].cmdMinLinvelBaseRtWorldEwrtHeadingXy,
            (*table)[0].cmdMaxLinvelBaseRtWorldEwrtHeadingXy);
  EXPECT_LT((*table)[1].cmdMinAngvelBaseRtWorldEwrtWorldZ,
            (*table)[0].cmdMaxAngvelBaseRtWorldEwrtWorldZ);
}

TEST(WheeledGaitFileTest, AForwardOrATurningCommandDrivesAndRestStands) {
  const absl::StatusOr<mpc_config::GaitFile> file = wheeledGaitFile();
  ASSERT_TRUE(file.ok()) << file.status();
  const absl::StatusOr<std::vector<GaitModeStateConfig>> table =
      gaitModeStatesFromConfig(*file);
  ASSERT_TRUE(table.ok()) << table.status();
  ASSERT_EQ(table->size(), 2U);
  const GaitModeStateConfig& stance = (*table)[0];
  const GaitModeStateConfig& drive = (*table)[1];
  const BasePlanarMotion atRest;

  // From stance on the command alone, before any motion is measured.
  EXPECT_TRUE(ProceduralMpcMotionManager::transitionToFasterGait(
      command(kDriveSpeed, /*angvelZ=*/0.0), /*strideMeanMotion=*/nullptr,
      stance));
  EXPECT_TRUE(ProceduralMpcMotionManager::transitionToFasterGait(
      command(/*linvelX=*/0.0, kTurnRate), /*strideMeanMotion=*/nullptr,
      stance));
  EXPECT_TRUE(ProceduralMpcMotionManager::transitionToFasterGait(
      command(kDriveSpeed, /*angvelZ=*/0.0), &atRest, stance));
  EXPECT_FALSE(ProceduralMpcMotionManager::transitionToFasterGait(
      command(/*linvelX=*/0.0, /*angvelZ=*/0.0), &atRest, stance));

  // drive is the last row and keeps the drive and the turn, and gives way to
  // stance at rest.
  BasePlanarMotion driving;
  driving.linvelBaseRtWorldEwrtHeadingXy = vector2_t(kDriveSpeed, 0.0);
  EXPECT_FALSE(ProceduralMpcMotionManager::transitionToSlowerGait(
      command(kDriveSpeed, /*angvelZ=*/0.0), &driving, drive));
  BasePlanarMotion turning;
  turning.rateYawBaseRtWorld = kTurnRate;
  EXPECT_FALSE(ProceduralMpcMotionManager::transitionToSlowerGait(
      command(/*linvelX=*/0.0, kTurnRate), &turning, drive));
  EXPECT_TRUE(ProceduralMpcMotionManager::transitionToSlowerGait(
      command(/*linvelX=*/0.0, /*angvelZ=*/0.0), &atRest, drive));
  // A stop is not taken while the base still drives.
  EXPECT_FALSE(ProceduralMpcMotionManager::transitionToSlowerGait(
      command(/*linvelX=*/0.0, /*angvelZ=*/0.0), &driving, drive));
}

}  // namespace
}  // namespace ocs2::humanoid
