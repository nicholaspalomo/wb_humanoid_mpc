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

// Pinocchio forward declarations must be included first.
#include "pinocchio/fwd.hpp"

#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "pinocchio/multibody/joint/joint-free-flyer.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include "humanoid_common_mpc_app/visualization/SceneBuilder.h"
#include "humanoid_common_mpc_app/visualization/TelemetryBuilder.h"
#include "humanoid_common_mpc_app/visualization/VisualizationConfig.h"
#include "humanoid_common_mpc_app/visualization/VisualizationPublisher.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/visualization/test/VisualizationTestRobot.h"

namespace ocs2::humanoid::visualization {
namespace {

class VisualizationConfigTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { robot_ = test::TestRobot::load(test::g1CentroidalFiles(), test::Formulation::kCentroidal).release(); }
  static void TearDownTestSuite() {
    delete robot_;
    robot_ = nullptr;
  }

  /** `text` written to a task file of its own in the test's temporary directory; its path. */
  static std::string writeTaskFile(const std::string& text) {
    const char* absl_nullable directory = std::getenv("TEST_TMPDIR");
    const std::string file = absl::StrCat(directory != nullptr ? directory : "/tmp", "/visualization_task.textproto");
    std::ofstream(file, std::ios::trunc) << text;
    return file;
  }

  /** The visualization settings of a task file whose whole text is `text`. */
  absl::StatusOr<VisualizationConfig> load(const std::string& text) const {
    return loadVisualizationConfig(writeTaskFile(text), robot_->modelSettings());
  }

  /** `text` refused by the conversion, naming the task file and `field`. */
  void expectRefused(const std::string& text, absl::string_view field) const {
    const absl::StatusOr<VisualizationConfig> config = load(text);
    ASSERT_FALSE(config.ok()) << text;
    EXPECT_EQ(config.status().code(), absl::StatusCode::kInvalidArgument) << text;
    EXPECT_TRUE(absl::StrContains(config.status().message(), field)) << config.status();
    EXPECT_TRUE(absl::StartsWith(config.status().message(), writeTaskFile(text))) << config.status();
  }

  /** `text` refused by the strict parser where it is written: on its first line. */
  void expectNotParsed(const std::string& text) const {
    const absl::StatusOr<VisualizationConfig> config = load(text);
    ASSERT_FALSE(config.ok()) << text;
    EXPECT_EQ(config.status().code(), absl::StatusCode::kInvalidArgument) << text;
    EXPECT_TRUE(absl::StrContains(config.status().message(), absl::StrCat(writeTaskFile(text), ":1:"))) << config.status();
  }

  static test::TestRobot* absl_nullable robot_;
};

test::TestRobot* absl_nullable VisualizationConfigTest::robot_ = nullptr;

TEST_F(VisualizationConfigTest, WithoutTheFieldsTheDefaultsApply) {
  const absl::StatusOr<VisualizationConfig> config = load("mpc { time_horizon: 1.0 }\n");
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->sceneFrequency, kDefaultRerunSceneFrequency);
  EXPECT_TRUE(config->sceneFrequencyIsDefault);
  EXPECT_EQ(config->telemetryFrames, robot_->modelSettings().contactNames);
  EXPECT_EQ(config->planFrames, robot_->modelSettings().contactNames);
  EXPECT_TRUE(absl::StrContains(describeVisualizationConfig(*config), "the default"));
}

TEST_F(VisualizationConfigTest, TheFieldsAreRead) {
  const absl::StatusOr<VisualizationConfig> config =
      load("rerun_scene_frequency: 12.5\ntelemetry_frames: [\"pelvis\", \"foot_l_contact\"]\nrerun_plan_frames: \"torso_link\"\n");
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->sceneFrequency, 12.5);
  EXPECT_FALSE(config->sceneFrequencyIsDefault);
  EXPECT_EQ(config->telemetryFrames, (std::vector<std::string>{"pelvis", "foot_l_contact"}));
  EXPECT_EQ(config->planFrames, (std::vector<std::string>{"torso_link"}));
  EXPECT_TRUE(absl::StrContains(describeVisualizationConfig(*config), "12.5 Hz"));
  EXPECT_TRUE(absl::StrContains(describeVisualizationConfig(*config), kRerunPlanFramesField));
}

TEST_F(VisualizationConfigTest, AnEmptyFrameListMeansTheContacts) {
  const absl::StatusOr<VisualizationConfig> config = load("telemetry_frames: []\n");
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->telemetryFrames, robot_->modelSettings().contactNames);
  EXPECT_EQ(config->planFrames, robot_->modelSettings().contactNames);
}

TEST_F(VisualizationConfigTest, ASceneFrequencyThatIsNotAPositiveFiniteNumberIsRefused) {
  for (const absl::string_view value : {"0", "-30", "inf", "nan"}) {
    expectRefused(absl::StrCat("rerun_scene_frequency: ", value, "\n"), kRerunSceneFrequencyField);
  }
}

TEST_F(VisualizationConfigTest, AValueOfTheWrongTypeIsRefusedWhereItIsWritten) {
  for (const absl::string_view text : {"rerun_scene_frequency: fast\n", "rerun_scene_frequency: [30]\n", "rerun_scene_frequency: [30\n",
                                       "telemetry_frames: pelvis\n", "rerun_plan_frames: [[\"pelvis\"]]\n"}) {
    expectNotParsed(std::string(text));
  }
}

TEST_F(VisualizationConfigTest, AFrameListThatIsNotAListOfDistinctNamesIsRefused) {
  for (const absl::string_view field : {kTelemetryFramesField, kRerunPlanFramesField}) {
    const std::string name(field);
    expectRefused(name + ": [\"pelvis\", \"pelvis\"]\n", field);
    expectRefused(name + ": [\"foot/left\"]\n", field);
    expectRefused(name + ": [\"..\"]\n", field);
  }
}

TEST_F(VisualizationConfigTest, AMissingFileIsNotFound) {
  const absl::StatusOr<VisualizationConfig> config = loadVisualizationConfig("/nonexistent/task.textproto", robot_->modelSettings());
  ASSERT_FALSE(config.ok());
  EXPECT_EQ(config.status().code(), absl::StatusCode::kNotFound);
}

TEST_F(VisualizationConfigTest, AFrameTheModelLacksIsRefusedAtStartUp) {
  VisualizationConfig config;
  config.telemetryFrames = {"no_such_frame"};
  config.planFrames = robot_->modelSettings().contactNames;
  const absl::StatusOr<std::unique_ptr<TelemetryBuilder>> telemetry = TelemetryBuilder::Create(robot_->model(), config);
  ASSERT_FALSE(telemetry.ok());
  EXPECT_TRUE(absl::StrContains(telemetry.status().message(), kTelemetryFramesField)) << telemetry.status();
  EXPECT_TRUE(absl::StrContains(telemetry.status().message(), "no_such_frame")) << telemetry.status();

  config.telemetryFrames = robot_->modelSettings().contactNames;
  config.planFrames = {"no_such_frame"};
  const absl::StatusOr<std::unique_ptr<SceneBuilder>> scene = SceneBuilder::Create(robot_->model(), config);
  ASSERT_FALSE(scene.ok());
  EXPECT_TRUE(absl::StrContains(scene.status().message(), kRerunPlanFramesField)) << scene.status();
}

TEST_F(VisualizationConfigTest, AModelThatIsNotTheMpcsIsRefused) {
  VisualizationModel model = robot_->model();
  model.mpcRobotModel = nullptr;
  EXPECT_EQ(checkVisualizationModel(model).code(), absl::StatusCode::kInvalidArgument);
  // A free-flyer root: a quaternion in the configuration, so nq != nv.
  PinocchioInterface::Model freeFlyerModel;
  pinocchio::urdf::buildModel(robot_->urdfFile(), pinocchio::JointModelFreeFlyerTpl<scalar_t>(), freeFlyerModel);
  const PinocchioInterface fullModel(freeFlyerModel);
  model = robot_->model();
  model.pinocchioInterface = &fullModel;
  EXPECT_EQ(checkVisualizationModel(model).code(), absl::StatusCode::kInvalidArgument);
}

// Every shipped task file sets the fields, and every frame it names is a frame of its robot's MPC model: the MPC node of
// every robot starts.
TEST(ShippedTaskFilesTest, EveryRobotsVisualizationStarts) {
  for (const std::pair<test::RobotFiles, test::Formulation>& configuration : test::shippedConfigurations()) {
    SCOPED_TRACE(configuration.first.taskFile);
    const std::unique_ptr<test::TestRobot> robot = test::TestRobot::load(configuration.first, configuration.second);
    const absl::StatusOr<VisualizationConfig> config = loadVisualizationConfig(robot->taskFile(), robot->modelSettings());
    ASSERT_TRUE(config.ok()) << config.status();
    EXPECT_FALSE(config->sceneFrequencyIsDefault) << "the task file does not set " << kRerunSceneFrequencyField;
    EXPECT_GT(config->sceneFrequency, 0.0);
    EXPECT_FALSE(config->telemetryFrames.empty());
    const absl::StatusOr<std::unique_ptr<VisualizationPublisher>> publisher = VisualizationPublisher::Create(
        robot->model(), [](absl::string_view /*topic*/, const google::protobuf::Message& /*message*/) { return absl::OkStatus(); });
    EXPECT_TRUE(publisher.ok()) << publisher.status();
  }
}

}  // namespace
}  // namespace ocs2::humanoid::visualization
