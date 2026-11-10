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

#include <cstddef>
#include <string>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/reference/GaitFromConfig.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/reference_manager/GaitSwitchingTable.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_mpc_config/gait_file.nproto.h"
#include "humanoid_mpc_config/gait_file.nproto.pb.h"
#include "humanoid_mpc_config/gait_file.pb.h"
#include "nproto/Textproto.h"

/*
 * The gait switching table of ProceduralMpcMotionManager: the default table a
 * gait file without gait_switching gets is the one the manager had as a member
 * initializer, row for row and field for field; a gait file's own table is
 * read in order and refused naming the row and the field when it is malformed;
 * and checkEveryGaitIsLoaded() refuses a table that names a gait the file
 * lacks.
 */

namespace ocs2::humanoid {
namespace {

using ::testing::HasSubstr;

/** The gait file `text` as its struct; a failed expectation and the default
 * struct when it does not parse. */
mpc_config::GaitFile gaitFileOf(absl::string_view text) {
  const absl::StatusOr<humanoid_mpc_config::GaitFile> proto =
      nproto::ParseTextproto<humanoid_mpc_config::GaitFile>(text,
                                                            "gait.textproto");
  EXPECT_TRUE(proto.ok()) << proto.status();
  mpc_config::GaitFile file;
  if (!proto.ok()) return file;
  EXPECT_TRUE(mpc_config::FromProto(*proto, &file).ok());
  return file;
}

// Two gaits and a table of both, the second row's thresholds distinct numbers.
constexpr char kTwoGaitFile[] = R"pb(
  gait_list: "stance"
  gait_list: "amble"
  gaits {
    name: "stance"
    mode_sequence: "STANCE"
    switching_times: [ 0.0, 0.5 ]
  }
  gaits {
    name: "amble"
    mode_sequence: [ "LF", "RF" ]
    switching_times: [ 0.0, 0.4, 0.8 ]
  }
  gait_switching {
    rows {
      gait: "stance"
      cmd_min_linvel_base_rt_world_ewrt_heading_xy: -0.2
      cmd_max_linvel_base_rt_world_ewrt_heading_xy: 0.2
      cmd_min_angvel_base_rt_world_ewrt_world_z: -0.3
      cmd_max_angvel_base_rt_world_ewrt_world_z: 0.3
      max_err_linvel_base_rt_world_ewrt_heading_xy: 5.0
      max_err_angvel_base_rt_world_ewrt_world_z: 6.0
    }
    rows {
      gait: "amble"
      cmd_min_linvel_base_rt_world_ewrt_heading_xy: 0.1
      cmd_max_linvel_base_rt_world_ewrt_heading_xy: 0.7
      cmd_min_angvel_base_rt_world_ewrt_world_z: 0.15
      cmd_max_angvel_base_rt_world_ewrt_world_z: 0.9
      max_err_linvel_base_rt_world_ewrt_heading_xy: 0.25
      max_err_angvel_base_rt_world_ewrt_world_z: 0.35
    }
  }
)pb";

void expectRow(const GaitModeStateConfig& actual,
               const GaitModeStateConfig& expected) {
  SCOPED_TRACE(expected.gaitCommand);
  EXPECT_EQ(actual.gaitCommand, expected.gaitCommand);
  EXPECT_EQ(actual.cmdMinLinvelBaseRtWorldEwrtHeadingXy,
            expected.cmdMinLinvelBaseRtWorldEwrtHeadingXy);
  EXPECT_EQ(actual.cmdMaxLinvelBaseRtWorldEwrtHeadingXy,
            expected.cmdMaxLinvelBaseRtWorldEwrtHeadingXy);
  EXPECT_EQ(actual.cmdMinAngvelBaseRtWorldEwrtWorldZ,
            expected.cmdMinAngvelBaseRtWorldEwrtWorldZ);
  EXPECT_EQ(actual.cmdMaxAngvelBaseRtWorldEwrtWorldZ,
            expected.cmdMaxAngvelBaseRtWorldEwrtWorldZ);
  EXPECT_EQ(actual.maxErrLinvelBaseRtWorldEwrtHeadingXy,
            expected.maxErrLinvelBaseRtWorldEwrtHeadingXy);
  EXPECT_EQ(actual.maxErrAngvelBaseRtWorldEwrtWorldZ,
            expected.maxErrAngvelBaseRtWorldEwrtWorldZ);
}

TEST(GaitSwitchingTable, TheDefaultIsTheTableTheManagerHad) {
  // The rows of ProceduralMpcMotionManager's former member initializer.
  const std::vector<GaitModeStateConfig> before = {
      {"stance", -0.1, 0.1, -0.1, 0.1, 10.0, 10.0},
      {"slow_walk", 0.05, 0.3, 0.05, 0.2, 0.05, 0.05},
      {"walk", 0.25, 0.5, 0.15, 0.35, 0.05, 0.05},
      {"slower_trot", 0.45, 0.7, 0.3, 0.55, 0.1, 0.1},
      {"slow_trot", 0.65, 0.9, 0.5, 0.7, 0.2, 0.2},
      {"trot", 0.8, 1.3, 0.65, 10.0, 0.2, 0.2},
      {"run", 1.2, 10.0, 0.65, 10.0, 0.2, 0.2},
  };
  const std::vector<GaitModeStateConfig> defaults = defaultGaitModeStates();
  ASSERT_EQ(defaults.size(), before.size());
  for (size_t i = 0; i < before.size(); ++i) expectRow(defaults[i], before[i]);

  // A gait file without gait_switching gets it, as the shipped biped one does.
  const absl::StatusOr<std::vector<GaitModeStateConfig>> absent =
      gaitModeStatesFromConfig(gaitFileOf(R"pb(gait_list: "stance")pb"));
  ASSERT_TRUE(absent.ok()) << absent.status();
  ASSERT_EQ(absent->size(), before.size());
  for (size_t i = 0; i < before.size(); ++i) expectRow((*absent)[i], before[i]);
}

TEST(GaitSwitchingTable, AConfiguredTableIsReadInOrder) {
  const absl::StatusOr<std::vector<GaitModeStateConfig>> table =
      gaitModeStatesFromConfig(gaitFileOf(kTwoGaitFile));
  ASSERT_TRUE(table.ok()) << table.status();
  ASSERT_EQ(table->size(), 2U);
  expectRow((*table)[0],
            GaitModeStateConfig{.gaitCommand = "stance",
                                .cmdMinLinvelBaseRtWorldEwrtHeadingXy = -0.2,
                                .cmdMaxLinvelBaseRtWorldEwrtHeadingXy = 0.2,
                                .cmdMinAngvelBaseRtWorldEwrtWorldZ = -0.3,
                                .cmdMaxAngvelBaseRtWorldEwrtWorldZ = 0.3,
                                .maxErrLinvelBaseRtWorldEwrtHeadingXy = 5.0,
                                .maxErrAngvelBaseRtWorldEwrtWorldZ = 6.0});
  expectRow((*table)[1],
            GaitModeStateConfig{.gaitCommand = "amble",
                                .cmdMinLinvelBaseRtWorldEwrtHeadingXy = 0.1,
                                .cmdMaxLinvelBaseRtWorldEwrtHeadingXy = 0.7,
                                .cmdMinAngvelBaseRtWorldEwrtWorldZ = 0.15,
                                .cmdMaxAngvelBaseRtWorldEwrtWorldZ = 0.9,
                                .maxErrLinvelBaseRtWorldEwrtHeadingXy = 0.25,
                                .maxErrAngvelBaseRtWorldEwrtWorldZ = 0.35});
}

/** gaitModeStatesFromConfig() of kTwoGaitFile with `from` replaced by `to`. */
absl::Status tableStatusWith(absl::string_view from, absl::string_view to) {
  const std::string text = absl::StrReplaceAll(kTwoGaitFile, {{from, to}});
  EXPECT_NE(text, kTwoGaitFile) << "no '" << from << "' to replace";
  return gaitModeStatesFromConfig(gaitFileOf(text)).status();
}

void expectRefusedNaming(const absl::Status& status,
                         absl::Span<const std::string> parts) {
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  for (const std::string& part : parts) {
    EXPECT_THAT(status.message(), HasSubstr(part));
  }
}

TEST(GaitSwitchingTable, MalformedRowsAreRefusedNamingTheRowAndTheField) {
  expectRefusedNaming(
      gaitModeStatesFromConfig(gaitFileOf(R"pb(gait_switching {})pb")).status(),
      {"gait_switching.rows is empty"});
  expectRefusedNaming(tableStatusWith(R"(gait: "amble")", R"(gait: "")"),
                      {"gait_switching.rows[1].gait is empty"});
  expectRefusedNaming(tableStatusWith(R"(gait: "amble")", R"(gait: "stance")"),
                      {"gait_switching.rows[1].gait 'stance'", "earlier row"});
  expectRefusedNaming(
      tableStatusWith("max_err_angvel_base_rt_world_ewrt_world_z: 0.35",
                      /*to=*/""),
      {"gait_switching.rows[1].max_err_angvel_base_rt_world_ewrt_world_z",
       "absent"});
  expectRefusedNaming(
      tableStatusWith("cmd_max_linvel_base_rt_world_ewrt_heading_xy: 0.7",
                      "cmd_max_linvel_base_rt_world_ewrt_heading_xy: nan"),
      {"gait_switching.rows[1].cmd_max_linvel_base_rt_world_ewrt_heading_xy"});
  expectRefusedNaming(
      tableStatusWith("cmd_max_linvel_base_rt_world_ewrt_heading_xy: 0.7",
                      "cmd_max_linvel_base_rt_world_ewrt_heading_xy: 0.05"),
      {"gait_switching.rows[1].cmd_min_linvel_base_rt_world_ewrt_heading_xy",
       "gait_switching.rows[1].cmd_max_linvel_base_rt_world_ewrt_heading_xy"});
  expectRefusedNaming(
      tableStatusWith("cmd_min_angvel_base_rt_world_ewrt_world_z: -0.3",
                      "cmd_min_angvel_base_rt_world_ewrt_world_z: 0.4"),
      {"gait_switching.rows[0].cmd_min_angvel_base_rt_world_ewrt_world_z",
       "gait_switching.rows[0].cmd_max_angvel_base_rt_world_ewrt_world_z"});
}

TEST(GaitSwitchingTable, AFirstRowOtherThanStanceIsRefused) {
  // The manager takes row 0 as the gait running at start; the schedule starts
  // in STANCE, so a table opening with any other gait would leave the robot
  // standing while the manager believes it walks.
  const std::string swapped = absl::StrReplaceAll(
      kTwoGaitFile, {{R"(gait: "stance")", R"(gait: "amble")"},
                     {R"(gait: "amble")", R"(gait: "stance")"}});
  expectRefusedNaming(
      gaitModeStatesFromConfig(gaitFileOf(swapped)).status(),
      {"gait_switching.rows[0].gait is 'amble'", "starts in, 'stance'"});
}

TEST(GaitSwitchingTable, ATableNamingAGaitTheFileLacksIsRefusedNamingIt) {
  const mpc_config::GaitFile file = gaitFileOf(kTwoGaitFile);
  const absl::StatusOr<absl::btree_map<std::string, ModeSequenceTemplate>>
      gaits = gaitMapFromConfig(file);
  ASSERT_TRUE(gaits.ok()) << gaits.status();
  absl::StatusOr<std::vector<GaitModeStateConfig>> table =
      gaitModeStatesFromConfig(file);
  ASSERT_TRUE(table.ok()) << table.status();
  EXPECT_TRUE(ProceduralMpcMotionManager::checkEveryGaitIsLoaded(
                  *gaits, *table, "gait.textproto")
                  .ok());
  table->push_back(
      GaitModeStateConfig{.gaitCommand = "gallop",
                          .cmdMinLinvelBaseRtWorldEwrtHeadingXy = 0.6,
                          .cmdMaxLinvelBaseRtWorldEwrtHeadingXy = 2.0,
                          .cmdMinAngvelBaseRtWorldEwrtWorldZ = 0.6,
                          .cmdMaxAngvelBaseRtWorldEwrtWorldZ = 2.0,
                          .maxErrLinvelBaseRtWorldEwrtHeadingXy = 0.2,
                          .maxErrAngvelBaseRtWorldEwrtWorldZ = 0.2});
  expectRefusedNaming(ProceduralMpcMotionManager::checkEveryGaitIsLoaded(
                          *gaits, *table, "gait.textproto"),
                      {"gait.textproto has no gait 'gallop'"});
  // The default table names gaits this file does not have either.
  expectRefusedNaming(ProceduralMpcMotionManager::checkEveryGaitIsLoaded(
                          *gaits, defaultGaitModeStates(), "gait.textproto"),
                      {"has no gait 'slow_walk'"});
}

}  // namespace
}  // namespace ocs2::humanoid
