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
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/mrt/JointPdGains.h"
#include "humanoid_common_mpc_app/robot/ConfigFileStore.h"
#include "humanoid_common_mpc_app/robot/RobotConfigDirectory.h"
#include "humanoid_common_mpc_app/robot/RobotConfigurationCheck.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_msgs/config_file_kind.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/RobotTestSupport.h"
#include "robot_model/AlwaysInContactEstimator.h"
#include "robot_model/ContactEstimator.h"
#include "robot_model/ContactEstimatorRegistry.h"
#include "robot_model/RobotState.h"

/*
 * checkRobotConfiguration(), the checks start-up and a save share: a
 * configuration that passes, and each check refusing on its own, naming the
 * file and the field; checkConfigFileCandidate() checks a saved file in place
 * of its stored copy.
 */

namespace ocs2::humanoid {
namespace {

using ::testing::AllOf;
using ::testing::HasSubstr;

constexpr char kTask[] = R"(# proto-message: humanoid_mpc_config.TaskFile
model_settings { robot_name: "g1" }
contact_estimator: "robot_state"
telemetry_sinks: "bus"
)";
constexpr char kGains[] =
    "# proto-message: humanoid_mpc_config.JointPdGainsFile\ndefault_gains { "
    "kp: 100.0 kd: 2.0 max_joint_torque: 50.0 }\n";

/** The files of a configuration that passes, with `taskText` as its task file.
 */
/** `text` as a configuration file of `message`: its `# proto-message:` header
 * first, unless it opens with one. */
std::string withHeader(absl::string_view text, absl::string_view message) {
  if (absl::StartsWith(text, "# proto-message:")) return std::string(text);
  return absl::StrCat("# proto-message: ", message, "\n", text);
}

RobotConfigFiles filesWith(absl::string_view taskTextIn,
                           absl::string_view gainsTextIn = kGains) {
  const std::string taskText =
      withHeader(taskTextIn, "humanoid_mpc_config.TaskFile");
  const std::string gainsText =
      withHeader(gainsTextIn, "humanoid_mpc_config.JointPdGainsFile");
  RobotConfigFiles files;
  absl::StatusOr<mpc_config::TaskFile> task =
      parseTaskFile(taskText, /*source=*/"task.textproto");
  EXPECT_TRUE(task.ok()) << task.status();
  if (task.ok()) files.task = *std::move(task);
  absl::StatusOr<mpc_config::JointPdGainsFile> gains =
      parseJointPdGainsFile(gainsText, /*source=*/"joint_pd_gains.textproto");
  EXPECT_TRUE(gains.ok()) << gains.status();
  if (gains.ok()) files.pdGains = *std::move(gains);
  files.taskSource = "task.textproto";
  files.referenceSource = "reference.textproto";
  files.pdGainsSource = "joint_pd_gains.textproto";
  return files;
}

/** A context every check of which runs: robot, joints, estimators with their
 * probe, the mujoco backend. */
RobotConfigurationCheckContext fullContext(
    const robot::model::ContactEstimatorRegistry& registry) {
  RobotConfigurationCheckContext context;
  context.robotName = "g1";
  context.mpcJointNames = {"left_knee", "right_knee"};
  context.otherJointNames = {"neck"};
  context.contactEstimators = &registry;
  context.backendName = "mujoco";
  context.backendOptions.mjcfFile = robot_test::kAtlasScene;
  context.backendOptions.initialState.emplace(robot_test::atlasDescription());
  return context;
}

std::string taskWith(absl::string_view line) {
  return absl::StrCat(kTask, line, "\n");
}

TEST(RobotConfigurationCheck, AConfigurationTheRobotStartsWithPasses) {
  const robot::model::ContactEstimatorRegistry registry;
  const absl::Status status =
      checkRobotConfiguration(filesWith(kTask), fullContext(registry));
  EXPECT_TRUE(status.ok()) << status;
}

TEST(RobotConfigurationCheck, AnotherRobotsTaskFileIsRefused) {
  const robot::model::ContactEstimatorRegistry registry;
  const absl::Status status = checkRobotConfiguration(
      filesWith(
          R"(model_settings { robot_name: "atlas" } contact_estimator: "robot_state")"),
      fullContext(registry));
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.message(), AllOf(HasSubstr("task.textproto"),
                                      HasSubstr("model_settings.robot_name"),
                                      HasSubstr("atlas")));
}

TEST(RobotConfigurationCheck, ASettingOfTheRobotProcessItRefusesIsRefused) {
  const robot::model::ContactEstimatorRegistry registry;
  const absl::Status status = checkRobotConfiguration(
      filesWith(taskWith("telemetry_frequency: -1")), fullContext(registry));
  EXPECT_THAT(status.message(), AllOf(HasSubstr("task.textproto"),
                                      HasSubstr("telemetry_frequency")));
}

TEST(RobotConfigurationCheck, TheFormulationsCheckRunsOnTheFiles) {
  const robot::model::ContactEstimatorRegistry registry;
  RobotConfigurationCheckContext context = fullContext(registry);
  std::string checkedRobot;
  context.checkFormulation = [&checkedRobot](const RobotConfigFiles& files) {
    checkedRobot = files.task.model_settings.robot_name;
    return absl::InvalidArgumentError(
        "task.textproto: centroidal_model: the formulation refuses it");
  };
  const absl::Status status =
      checkRobotConfiguration(filesWith(kTask), context);
  EXPECT_THAT(status.message(), HasSubstr("centroidal_model"));
  EXPECT_EQ(checkedRobot, "g1");
}

TEST(RobotConfigurationCheck, PdGainsTheControllerRefusesAreRefused) {
  const robot::model::ContactEstimatorRegistry registry;
  const absl::Status otherRobot = checkRobotConfiguration(
      filesWith(kTask, absl::StrCat(
                           kGains, R"(joint_gains { joint: "tail" kp: 1.0 })")),
      fullContext(registry));
  EXPECT_THAT(otherRobot.message(),
              AllOf(HasSubstr("joint_pd_gains.textproto"),
                    HasSubstr("joint_gains"), HasSubstr("tail")));

  // Both controllers read the torque limits, and neither has gains of its own
  // to stand in for one left out.
  const RobotConfigFiles negativeLimit = filesWith(
      kTask, "default_gains { kp: 1.0 kd: 1.0 max_joint_torque: -1.0 }\n");
  EXPECT_THAT(
      checkRobotConfiguration(negativeLimit, fullContext(registry)).message(),
      HasSubstr("default_gains.max_joint_torque"));
  const RobotConfigFiles noLimit =
      filesWith(kTask, "default_gains { kp: 1.0 kd: 1.0 }\n");
  EXPECT_THAT(checkRobotConfiguration(noLimit, fullContext(registry)).message(),
              HasSubstr("default_gains.max_joint_torque is not set"));
}

TEST(RobotConfigurationCheck,
     AContactEstimatorTheBackendDidNotRegisterIsRefused) {
  robot::model::ContactEstimatorRegistry registry;
  const RobotConfigFiles files = filesWith(
      R"(model_settings { robot_name: "g1" } contact_estimator: "cheater_sim")");
  const absl::Status status =
      checkRobotConfiguration(files, fullContext(registry));
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              AllOf(HasSubstr("contact_estimator"), HasSubstr("cheater_sim"),
                    HasSubstr("always_in_contact")));
  // Once the backend registers it, it passes.
  registry.add("cheater_sim", "a simulator's contacts", []() {
    return std::make_shared<robot::model::AlwaysInContactEstimator>();
  });
  EXPECT_TRUE(checkRobotConfiguration(files, fullContext(registry)).ok());
}

TEST(RobotConfigurationCheck,
     AnEstimatorWithoutOneFlagPerContactPointIsRefusedByItsProbe) {
  const robot::model::ContactEstimatorRegistry registry;
  RobotConfigurationCheckContext context = fullContext(registry);
  // A robot state of three contact points: always_in_contact answers three
  // flags, and the controller has two.
  context.backendOptions.initialState.emplace(robot_test::atlasDescription(),
                                              /*contactSize=*/3);
  const absl::Status status = checkRobotConfiguration(
      filesWith(
          R"(model_settings { robot_name: "g1" } contact_estimator: "always_in_contact")"),
      context);
  EXPECT_THAT(status.message(), AllOf(HasSubstr("contact_estimator"),
                                      HasSubstr("3 contact flags")));
}

TEST(RobotConfigurationCheck, AnUnknownTelemetrySinkIsRefused) {
  const robot::model::ContactEstimatorRegistry registry;
  const absl::Status status = checkRobotConfiguration(
      filesWith(taskWith(R"(telemetry_sinks: "carrier_pigeon")")),
      fullContext(registry));
  EXPECT_THAT(status.message(),
              AllOf(HasSubstr("telemetry_sinks"), HasSubstr("carrier_pigeon"),
                    HasSubstr("bus")));
}

TEST(RobotConfigurationCheck, ASimulatorOptionTheBackendRefusesIsRefused) {
  const robot::model::ContactEstimatorRegistry registry;
  EXPECT_THAT(
      checkRobotConfiguration(filesWith(taskWith(R"(gantry_hold: "rope")")),
                              fullContext(registry))
          .message(),
      AllOf(HasSubstr("task.textproto"), HasSubstr("gantry_hold")));
  EXPECT_THAT(
      checkRobotConfiguration(filesWith(taskWith(R"(sim_projectile: "anvil")")),
                              fullContext(registry))
          .message(),
      HasSubstr("sim_projectile"));
  // The viewer skips a visualization it does not know, at start-up as later: no
  // refusal.
  EXPECT_TRUE(
      checkRobotConfiguration(
          filesWith(taskWith(R"(sim_visualizations: "no_such_marker")")),
          fullContext(registry))
          .ok());
  RobotConfigurationCheckContext otherBackend = fullContext(registry);
  otherBackend.backendName = "hardware_v2";
  EXPECT_THAT(checkRobotConfiguration(filesWith(kTask), otherBackend).message(),
              HasSubstr("hardware_v2"));
}

TEST(RobotConfigurationCheck,
     AContextWithoutTheRunningRobotChecksTheFilesAlone) {
  const RobotConfigurationCheckContext empty;
  EXPECT_TRUE(
      checkRobotConfiguration(
          filesWith(
              R"(model_settings { robot_name: "atlas" } contact_estimator: "magic")"),
          empty)
          .ok());
  EXPECT_FALSE(checkRobotConfiguration(
                   filesWith(taskWith("telemetry_frequency: 0")), empty)
                   .ok());
}

/** Writes `contents` at `path`, creating its directory. */
void writeFile(const std::string& path, absl::string_view contents) {
  std::filesystem::create_directories(
      std::filesystem::path(path).parent_path());
  std::ofstream(path, std::ios::trunc) << contents;
}

TEST(CheckConfigFileCandidate, ASavedFileIsCheckedInPlaceOfItsStoredCopy) {
  const std::string directory =
      (std::filesystem::path(std::getenv("TEST_TMPDIR")) / "candidate")
          .string();
  const RobotConfigDirectory::Files stored{
      .taskFile = directory + "/mpc/task.textproto",
      .referenceFile = directory + "/command/reference.textproto",
      .pdGainsFile = directory + "/controller/joint_pd_gains.textproto"};
  writeFile(stored.taskFile, kTask);
  writeFile(
      stored.referenceFile,
      "# proto-message: "
      "humanoid_mpc_config.ReferenceFile\nref_speed_base_rt_world: 0.5\n");
  writeFile(stored.pdGainsFile, kGains);
  const robot::model::ContactEstimatorRegistry registry;
  const RobotConfigurationCheckContext context = fullContext(registry);

  ConfigFileCandidate gains;
  gains.kind = msgs::ConfigFileKind::kJointPdGains;
  gains.source =
      "robot_models/unitree_g1/g1_wb_mpc/config/controller/"
      "joint_pd_gains.textproto (operator/config_save)";
  absl::StatusOr<mpc_config::JointPdGainsFile> otherRobots =
      parseJointPdGainsFile(
          absl::StrCat(kGains, R"(joint_gains { joint: "tail" })"),
          gains.source);
  ASSERT_TRUE(otherRobots.ok()) << otherRobots.status();
  gains.pdGains = *std::move(otherRobots);
  const absl::Status refused = checkConfigFileCandidate(stored, gains, context);
  EXPECT_THAT(refused.message(),
              AllOf(HasSubstr("(operator/config_save)"), HasSubstr("tail")));

  ConfigFileCandidate task;
  task.kind = msgs::ConfigFileKind::kTask;
  task.source = "saved task file";
  absl::StatusOr<mpc_config::TaskFile> saved =
      parseTaskFile(taskWith("telemetry_frequency: 20"), task.source);
  ASSERT_TRUE(saved.ok()) << saved.status();
  task.task = *std::move(saved);
  EXPECT_TRUE(checkConfigFileCandidate(stored, task, context).ok());
  // The stored files are read, not written.
  const absl::StatusOr<RobotConfigFiles> reread = loadRobotConfigFiles(stored);
  ASSERT_TRUE(reread.ok()) << reread.status();
  EXPECT_FALSE(reread->task.telemetry_frequency.has_value());
  EXPECT_FALSE(reread->contactPlanning.has_value())
      << "this configuration has no contact planner's file";
}

}  // namespace
}  // namespace ocs2::humanoid
