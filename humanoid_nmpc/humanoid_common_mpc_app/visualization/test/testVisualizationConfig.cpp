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
#include <pinocchio/fwd.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <pinocchio/multibody/joint/joint-free-flyer.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"

#include "VisualizationTestRobot.h"
#include "humanoid_common_mpc_app/visualization/SceneBuilder.h"
#include "humanoid_common_mpc_app/visualization/TelemetryBuilder.h"
#include "humanoid_common_mpc_app/visualization/VisualizationConfig.h"
#include "humanoid_common_mpc_app/visualization/VisualizationPublisher.h"

namespace ocs2::humanoid::visualization {
namespace {

class VisualizationConfigTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { robot_ = test::TestRobot::load(test::g1CentroidalFiles(), test::Formulation::kCentroidal).release(); }
  static void TearDownTestSuite() {
    delete robot_;
    robot_ = nullptr;
  }

  absl::StatusOr<VisualizationConfig> parse(const std::string& text) const {
    return parseVisualizationConfig(text, "task.yaml", robot_->modelSettings());
  }

  void expectRefused(const std::string& text, absl::string_view key) const {
    const absl::StatusOr<VisualizationConfig> config = parse(text);
    ASSERT_FALSE(config.ok()) << text;
    EXPECT_EQ(config.status().code(), absl::StatusCode::kInvalidArgument) << text;
    EXPECT_TRUE(absl::StrContains(config.status().message(), key)) << config.status();
    EXPECT_TRUE(absl::StrContains(config.status().message(), "task.yaml")) << config.status();
  }

  static test::TestRobot* robot_;
};

test::TestRobot* VisualizationConfigTest::robot_ = nullptr;

TEST_F(VisualizationConfigTest, WithoutTheKeysTheDefaultsApply) {
  const absl::StatusOr<VisualizationConfig> config = parse("someOtherKey: 1\n");
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->sceneFrequency, kDefaultRerunSceneFrequency);
  EXPECT_TRUE(config->sceneFrequencyIsDefault);
  EXPECT_EQ(config->telemetryFrames, robot_->modelSettings().contactNames);
  EXPECT_EQ(config->planFrames, robot_->modelSettings().contactNames);
  EXPECT_TRUE(absl::StrContains(describeVisualizationConfig(*config), "the default"));
}

TEST_F(VisualizationConfigTest, TheKeysAreRead) {
  const absl::StatusOr<VisualizationConfig> config =
      parse("rerunSceneFrequency: 12.5\ntelemetryFrames: [pelvis, foot_l_contact]\nrerunPlanFrames:\n  - torso_link\n");
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->sceneFrequency, 12.5);
  EXPECT_FALSE(config->sceneFrequencyIsDefault);
  EXPECT_EQ(config->telemetryFrames, (std::vector<std::string>{"pelvis", "foot_l_contact"}));
  EXPECT_EQ(config->planFrames, (std::vector<std::string>{"torso_link"}));
  EXPECT_TRUE(absl::StrContains(describeVisualizationConfig(*config), "12.5 Hz"));
}

TEST_F(VisualizationConfigTest, AnEmptyFrameListMeansTheContacts) {
  const absl::StatusOr<VisualizationConfig> config = parse("telemetryFrames: []\nrerunPlanFrames:\n");
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->telemetryFrames, robot_->modelSettings().contactNames);
  EXPECT_EQ(config->planFrames, robot_->modelSettings().contactNames);
}

TEST_F(VisualizationConfigTest, ASceneFrequencyThatIsNotAPositiveNumberIsRefused) {
  for (const std::string& value : {"0", "-30", "fast", ".inf", ".nan", "[30]"}) {
    expectRefused("rerunSceneFrequency: " + value + "\n", kRerunSceneFrequencyKey);
  }
}

TEST_F(VisualizationConfigTest, AFrameListThatIsNotAListOfDistinctNamesIsRefused) {
  for (const absl::string_view key : {kTelemetryFramesKey, kRerunPlanFramesKey}) {
    const std::string name(key);
    expectRefused(name + ": pelvis\n", key);
    expectRefused(name + ": [[pelvis]]\n", key);
    expectRefused(name + ": [pelvis, pelvis]\n", key);
    expectRefused(name + ": [\"foot/left\"]\n", key);
    expectRefused(name + ": [\"..\"]\n", key);
  }
}

TEST_F(VisualizationConfigTest, TextThatDoesNotParseIsRefused) {
  const absl::StatusOr<VisualizationConfig> config = parse("rerunSceneFrequency: [30\n");
  ASSERT_FALSE(config.ok());
  EXPECT_EQ(config.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(VisualizationConfigTest, AMissingFileIsNotFound) {
  const absl::StatusOr<VisualizationConfig> config = loadVisualizationConfig("/nonexistent/task.yaml", robot_->modelSettings());
  ASSERT_FALSE(config.ok());
  EXPECT_EQ(config.status().code(), absl::StatusCode::kNotFound);
}

TEST_F(VisualizationConfigTest, AFrameTheModelLacksIsRefusedAtStartUp) {
  VisualizationConfig config;
  config.telemetryFrames = {"no_such_frame"};
  config.planFrames = robot_->modelSettings().contactNames;
  const absl::StatusOr<std::unique_ptr<TelemetryBuilder>> telemetry = TelemetryBuilder::Create(robot_->model(), config);
  ASSERT_FALSE(telemetry.ok());
  EXPECT_TRUE(absl::StrContains(telemetry.status().message(), kTelemetryFramesKey)) << telemetry.status();
  EXPECT_TRUE(absl::StrContains(telemetry.status().message(), "no_such_frame")) << telemetry.status();

  config.telemetryFrames = robot_->modelSettings().contactNames;
  config.planFrames = {"no_such_frame"};
  const absl::StatusOr<std::unique_ptr<SceneBuilder>> scene = SceneBuilder::Create(robot_->model(), config);
  ASSERT_FALSE(scene.ok());
  EXPECT_TRUE(absl::StrContains(scene.status().message(), kRerunPlanFramesKey)) << scene.status();
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

// Every shipped task file sets the keys, and every frame it names is a frame of its robot's MPC model: the MPC node of
// every robot starts.
TEST(ShippedTaskFilesTest, EveryRobotsVisualizationStarts) {
  for (const std::pair<test::RobotFiles, test::Formulation>& configuration : test::shippedConfigurations()) {
    SCOPED_TRACE(configuration.first.taskFile);
    const std::unique_ptr<test::TestRobot> robot = test::TestRobot::load(configuration.first, configuration.second);
    const absl::StatusOr<VisualizationConfig> config = loadVisualizationConfig(robot->taskFile(), robot->modelSettings());
    ASSERT_TRUE(config.ok()) << config.status();
    EXPECT_FALSE(config->sceneFrequencyIsDefault) << "the task file does not set " << kRerunSceneFrequencyKey;
    EXPECT_GT(config->sceneFrequency, 0.0);
    EXPECT_FALSE(config->telemetryFrames.empty());
    const absl::StatusOr<std::unique_ptr<VisualizationPublisher>> publisher = VisualizationPublisher::Create(
        robot->model(), [](absl::string_view /*topic*/, const google::protobuf::Message& /*message*/) { return absl::OkStatus(); });
    EXPECT_TRUE(publisher.ok()) << publisher.status();
  }
}

}  // namespace
}  // namespace ocs2::humanoid::visualization
