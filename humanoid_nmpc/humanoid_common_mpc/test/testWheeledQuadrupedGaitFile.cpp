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

// The gait file of the wheeled quadrupeds
// (config/command/wheeled_quadruped_gait.textproto) read on the four-wheel
// layout of the wheeled ANYmal C: stance and drive stand on every wheel,
// turning_trot swings the diagonal pairs with a four-wheel phase after each,
// and its switching table drives on a forward command, trots on a turning one
// and stands at rest.

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <string>
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

constexpr char kWheeledQuadrupedGaitFile[] =
    "humanoid_nmpc/humanoid_common_mpc/config/command/"
    "wheeled_quadruped_gait.textproto";

// [m] A base height for the velocity command, which the gait switch does not
// read.
constexpr scalar_t kCommandHeight = 0.63588;
// [m/s] and [rad/s] The commands of the closed-loop drive and turn.
constexpr scalar_t kDriveSpeed = 0.5;
constexpr scalar_t kTurnRate = 0.5;
// The mode bits of the layout: LF = 8, RF = 4, LH = 2, RH = 1.
constexpr size_t kStanceMode = 15;
constexpr size_t kLfRhMode = 9;
constexpr size_t kRfLhMode = 6;

/** Returns the gait file in this test's runfiles, or empty when it is not
 * there. */
std::string gaitFilePath() {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
    roots.emplace_back(std::filesystem::path(srcDir) / "wb_humanoid_mpc");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / kWheeledQuadrupedGaitFile;
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

/** Returns the four wheels on legs LF, RF, LH and RH of the wheeled ANYmal C.
 */
GaitModeLayout fourWheelLayout() {
  return GaitModeLayout{
      .numContacts = 4,
      .stanceMode = kStanceMode,
      .legs = {{.name = "LF", .modeBits = 8},
               {.name = "RF", .modeBits = 4},
               {.name = "LH", .modeBits = 2},
               {.name = "RH", .modeBits = 1}},
      .contactNames = {"LF_wheel_contact", "RF_wheel_contact",
                       "LH_wheel_contact", "RH_wheel_contact"}};
}

/** Reads the gait file strictly. */
absl::StatusOr<mpc_config::GaitFile> gaitFile() {
  const std::string path = gaitFilePath();
  if (path.empty()) {
    return absl::NotFoundError(
        absl::StrCat(kWheeledQuadrupedGaitFile, " is not in the runfiles"));
  }
  return loadGaitFile(path);
}

/** Returns the velocity command [v_x, v_y, height, yaw rate]. */
vector4_t command(scalar_t linvelX, scalar_t angvelZ) {
  return vector4_t(linvelX, 0.0, kCommandHeight, angvelZ);
}

TEST(WheeledQuadrupedGaitFileTest, StanceAndDriveStandAndTheTurnTrots) {
  const absl::StatusOr<mpc_config::GaitFile> file = gaitFile();
  ASSERT_TRUE(file.ok()) << file.status();
  const absl::StatusOr<absl::btree_map<std::string, ModeSequenceTemplate>>
      gaits = gaitMapFromConfig(*file, fourWheelLayout());
  ASSERT_TRUE(gaits.ok()) << gaits.status();
  ASSERT_EQ(gaits->size(), 3U);
  for (const std::string name : {"stance", "drive"}) {
    ASSERT_TRUE(gaits->contains(name)) << name;
    const ModeSequenceTemplate& gait = gaits->find(name)->second;
    EXPECT_EQ(gait.modeSequence, std::vector<size_t>{kStanceMode}) << name;
    EXPECT_EQ(gait.switchingTimes, (std::vector<scalar_t>{0.0, 0.5})) << name;
  }
  ASSERT_TRUE(gaits->contains("turning_trot"));
  const ModeSequenceTemplate& trot = gaits->find("turning_trot")->second;
  EXPECT_EQ(trot.modeSequence, (std::vector<size_t>{kLfRhMode, kStanceMode,
                                                    kRfLhMode, kStanceMode}));
  EXPECT_EQ(trot.switchingTimes,
            (std::vector<scalar_t>{0.0, 0.30, 0.45, 0.75, 0.90}));
}

TEST(WheeledQuadrupedGaitFileTest, ItsSwitchingTableIsStanceDriveThenTrot) {
  const absl::StatusOr<mpc_config::GaitFile> file = gaitFile();
  ASSERT_TRUE(file.ok()) << file.status();
  ASSERT_TRUE(file->gait_switching.has_value())
      << "the biped default table names gaits this file lacks";
  const absl::StatusOr<std::vector<GaitModeStateConfig>> table =
      gaitModeStatesFromConfig(*file);
  ASSERT_TRUE(table.ok()) << table.status();
  ASSERT_EQ(table->size(), 3U);
  EXPECT_EQ((*table)[0].gaitCommand, "stance");
  EXPECT_EQ((*table)[1].gaitCommand, "drive");
  EXPECT_EQ((*table)[2].gaitCommand, "turning_trot");
  const absl::StatusOr<absl::btree_map<std::string, ModeSequenceTemplate>>
      gaits = gaitMapFromConfig(*file, fourWheelLayout());
  ASSERT_TRUE(gaits.ok()) << gaits.status();
  EXPECT_TRUE(ProceduralMpcMotionManager::checkEveryGaitIsLoaded(
                  *gaits, *table, kWheeledQuadrupedGaitFile)
                  .ok());
  // Each row switches down below the bounds the row before switches up above.
  for (size_t row = 1; row < table->size(); ++row) {
    EXPECT_LT((*table)[row].cmdMinAngvelBaseRtWorldEwrtWorldZ,
              (*table)[row - 1].cmdMaxAngvelBaseRtWorldEwrtWorldZ)
        << row;
  }
  EXPECT_LT((*table)[1].cmdMinLinvelBaseRtWorldEwrtHeadingXy,
            (*table)[0].cmdMaxLinvelBaseRtWorldEwrtHeadingXy);
}

TEST(WheeledQuadrupedGaitFileTest, ADriveRollsATurnTrotsAndRestStands) {
  const absl::StatusOr<mpc_config::GaitFile> file = gaitFile();
  ASSERT_TRUE(file.ok()) << file.status();
  const absl::StatusOr<std::vector<GaitModeStateConfig>> table =
      gaitModeStatesFromConfig(*file);
  ASSERT_TRUE(table.ok()) << table.status();
  ASSERT_EQ(table->size(), 3U);
  const GaitModeStateConfig& stance = (*table)[0];
  const GaitModeStateConfig& drive = (*table)[1];
  const GaitModeStateConfig& trot = (*table)[2];
  const BasePlanarMotion atRest;
  BasePlanarMotion driving;
  driving.linvelBaseRtWorldEwrtHeadingXy = vector2_t(kDriveSpeed, 0.0);
  BasePlanarMotion turning;
  turning.rateYawBaseRtWorld = kTurnRate;

  // stance leaves for drive on a drive or a turn, on the command alone.
  EXPECT_TRUE(ProceduralMpcMotionManager::transitionToFasterGait(
      command(kDriveSpeed, /*angvelZ=*/0.0), &atRest, stance));
  EXPECT_TRUE(ProceduralMpcMotionManager::transitionToFasterGait(
      command(/*linvelX=*/0.0, kTurnRate), &atRest, stance));
  EXPECT_FALSE(ProceduralMpcMotionManager::transitionToFasterGait(
      command(/*linvelX=*/0.0, /*angvelZ=*/0.0), &atRest, stance));

  // drive keeps a drive, and leaves for the trot on a turn although the base
  // does not yaw while it drives.
  EXPECT_FALSE(ProceduralMpcMotionManager::transitionToFasterGait(
      command(kDriveSpeed, /*angvelZ=*/0.0), &driving, drive));
  EXPECT_FALSE(ProceduralMpcMotionManager::transitionToSlowerGait(
      command(kDriveSpeed, /*angvelZ=*/0.0), &driving, drive));
  EXPECT_TRUE(ProceduralMpcMotionManager::transitionToFasterGait(
      command(/*linvelX=*/0.0, kTurnRate), &atRest, drive));
  EXPECT_TRUE(ProceduralMpcMotionManager::transitionToSlowerGait(
      command(/*linvelX=*/0.0, /*angvelZ=*/0.0), &atRest, drive));
  // A stop is not taken while the base still drives.
  EXPECT_FALSE(ProceduralMpcMotionManager::transitionToSlowerGait(
      command(/*linvelX=*/0.0, /*angvelZ=*/0.0), &driving, drive));

  // The trot keeps a turn, and gives way to drive once the turn has stopped.
  EXPECT_FALSE(ProceduralMpcMotionManager::transitionToSlowerGait(
      command(/*linvelX=*/0.0, kTurnRate), &turning, trot));
  EXPECT_FALSE(ProceduralMpcMotionManager::transitionToSlowerGait(
      command(/*linvelX=*/0.0, /*angvelZ=*/0.0), &turning, trot));
  EXPECT_TRUE(ProceduralMpcMotionManager::transitionToSlowerGait(
      command(kDriveSpeed, /*angvelZ=*/0.0), &atRest, trot));
}

}  // namespace
}  // namespace ocs2::humanoid
