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

// The visualization publisher's settings from a typed task file: a file that says nothing gives the default rate,
// reported as the default, and the contact frames; a rate is read and not reported as the default; a rate that is not a
// positive finite number, a frame name the Rerun bridge cannot carry and a frame named twice are refused by their field.

#include <limits>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc_app/visualization/VisualizationConfig.h"
#include "humanoid_common_mpc_app/visualization/config/robot/VisualizationConfigFromConfig.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid::visualization {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

std::vector<std::string> contactFrames() {
  return {"foot_l_contact", "foot_r_contact"};
}

TEST(VisualizationConfigFromConfigTest, AFileThatSaysNothingGivesTheDefaultRateAndTheContactFrames) {
  const absl::StatusOr<VisualizationConfig> config = visualizationConfigFromConfig(mpc_config::TaskFile{}, contactFrames());
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->sceneFrequency, kDefaultRerunSceneFrequency);
  EXPECT_TRUE(config->sceneFrequencyIsDefault);
  EXPECT_EQ(config->telemetryFrames, contactFrames());
  EXPECT_EQ(config->planFrames, contactFrames());
}

TEST(VisualizationConfigFromConfigTest, TheRateAndTheFrameListsAreRead) {
  mpc_config::TaskFile task;
  task.rerun_scene_frequency = 12.5;
  task.telemetry_frames = {"pelvis", "utorso"};
  task.rerun_plan_frames = {"left_hand_palm_joint"};
  const absl::StatusOr<VisualizationConfig> config = visualizationConfigFromConfig(task, contactFrames());
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->sceneFrequency, 12.5);
  EXPECT_FALSE(config->sceneFrequencyIsDefault);
  EXPECT_THAT(config->telemetryFrames, ElementsAre("pelvis", "utorso"));
  EXPECT_THAT(config->planFrames, ElementsAre("left_hand_palm_joint"));
}

TEST(VisualizationConfigFromConfigTest, ARateThatIsNotAPositiveFiniteNumberIsRefused) {
  for (const double bad : {0.0, -30.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    mpc_config::TaskFile task;
    task.rerun_scene_frequency = bad;
    const absl::StatusOr<VisualizationConfig> config = visualizationConfigFromConfig(task, contactFrames());
    EXPECT_EQ(config.status().code(), absl::StatusCode::kInvalidArgument) << bad;
    EXPECT_THAT(config.status().message(), HasSubstr("rerun_scene_frequency"));
  }
}

TEST(VisualizationConfigFromConfigTest, AFrameTheBridgeCannotCarryOrAFrameNamedTwiceIsRefusedByItsList) {
  for (const std::string& bad : {std::string("foot l"), std::string(".."), std::string("a/b"), std::string()}) {
    mpc_config::TaskFile task;
    task.rerun_plan_frames = {"pelvis", bad};
    const absl::StatusOr<VisualizationConfig> config = visualizationConfigFromConfig(task, contactFrames());
    EXPECT_EQ(config.status().code(), absl::StatusCode::kInvalidArgument) << bad;
    EXPECT_THAT(config.status().message(), HasSubstr("rerun_plan_frames names the frame"));
  }
  mpc_config::TaskFile twice;
  twice.telemetry_frames = {"pelvis", "utorso", "pelvis"};
  const absl::StatusOr<VisualizationConfig> config = visualizationConfigFromConfig(twice, contactFrames());
  EXPECT_EQ(config.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(config.status().message(), HasSubstr("telemetry_frames names the frame 'pelvis' twice"));
}

}  // namespace
}  // namespace ocs2::humanoid::visualization
