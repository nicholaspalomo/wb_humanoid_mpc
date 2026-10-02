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

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include <mujoco_sim_interface/MujocoSimInterface.h>
#include <robot_model/ContactEstimatorRegistry.h>

#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "humanoid_common_mpc_app/robot/TelemetrySinkRegistry.h"

/*
 * The robot process's keys of the task file: the defaults of the ROS sims for a file that says nothing, values of the
 * wrong type refused by key, the retired booleans refused with the name that replaced them, and every shipped task file
 * read with names the registries know.
 */

namespace ocs2::humanoid {
namespace {

// LINT.IfChange(shipped_task_files)
const std::vector<std::string> kShippedTaskFiles = {
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml",
    "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml",
    "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml",
    "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml",
    "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.yaml",
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/BUILD.bazel:shipped_task_files)

TEST(RobotProcessSettings, AFileThatSaysNothingGetsTheDefaultsOfTheSims) {
  const absl::StatusOr<RobotProcessSettings> settings = parseRobotProcessSettings("mpc:\n  timeHorizon: 1.0\n", "test");
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_EQ(settings->contactEstimator, "cheater_sim");
  EXPECT_DOUBLE_EQ(settings->simulator.contactForceThreshold, 5.0);
  EXPECT_DOUBLE_EQ(settings->simulator.contactTimelineWindow, 5.0);
  EXPECT_FALSE(settings->simulator.visualizations.has_value());
  EXPECT_EQ(settings->simulator.gantryHold, "weld_constraint");
  EXPECT_TRUE(settings->simulator.projectile.empty());
  EXPECT_DOUBLE_EQ(settings->fallRecovery.maxBaseTiltAngle, 0.0) << "the catch is off unless the file turns it on";
  EXPECT_DOUBLE_EQ(settings->fallRecovery.catchLift, 0.0);
  EXPECT_FALSE(settings->mpcEntryBlendTime.has_value());
  EXPECT_FALSE(settings->safetyDecayTimeConstant.has_value());
  EXPECT_FALSE(settings->contactWrenchGate.has_value());
  EXPECT_EQ(settings->wbMpcFeedforward, WbMpcFeedforward::kInverseDynamics);
  EXPECT_EQ(settings->telemetrySinks, std::vector<std::string>{"bus"});
  EXPECT_FALSE(settings->telemetryFrequency.has_value());
  // One lost TCP segment on the bus is retransmitted after Linux's minimum retransmission timeout, 200 ms: the default
  // link-loss timeout outlasts it with margin, so that a single lost packet does not hold the robot.
  constexpr scalar_t kLinuxMinimumTcpRetransmissionTimeout = 0.2;
  EXPECT_GE(settings->mpcLinkPolicyTimeout, 2.0 * kLinuxMinimumTcpRetransmissionTimeout);
}

TEST(RobotProcessSettings, EveryKeyIsRead) {
  const absl::StatusOr<RobotProcessSettings> settings = parseRobotProcessSettings(
      "contactEstimator: robot_state\n"
      "simContactForceThreshold: 8\n"
      "simContactTimelineWindow: 3\n"
      "simVisualizations: [zmp, dcm]\n"
      "gantryHold: kinematic_teleport\n"
      "simProjectile: dodgeball\n"
      "simMaxBaseTiltAngle: 1\n"
      "simGantryCatchLift: 0.15\n"
      "mpcEntryBlendTime: 2\n"
      "safetyDecayTimeConstant: 0.4\n"
      "contact_wrench_gate:\n  debounceTime: 0.01\n  rampTime: 0.04\n"
      "wbMpcFeedforward: gravity_compensation\n"
      "telemetrySinks: []\n"
      "telemetry_frequency: 50\n"
      "mpcLink:\n  policyTimeout: 0.35\n",
      "test");
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_EQ(settings->contactEstimator, "robot_state");
  EXPECT_DOUBLE_EQ(settings->simulator.contactForceThreshold, 8.0);
  EXPECT_DOUBLE_EQ(settings->simulator.contactTimelineWindow, 3.0);
  EXPECT_EQ(*settings->simulator.visualizations, (std::vector<std::string>{"zmp", "dcm"}));
  EXPECT_EQ(settings->simulator.gantryHold, "kinematic_teleport");
  EXPECT_EQ(settings->simulator.projectile, "dodgeball");
  EXPECT_DOUBLE_EQ(settings->fallRecovery.maxBaseTiltAngle, 1.0);
  EXPECT_DOUBLE_EQ(settings->fallRecovery.catchLift, 0.15);
  EXPECT_DOUBLE_EQ(*settings->mpcEntryBlendTime, 2.0);
  EXPECT_DOUBLE_EQ(*settings->safetyDecayTimeConstant, 0.4);
  EXPECT_DOUBLE_EQ(settings->contactWrenchGate->debounceTime, 0.01);
  EXPECT_DOUBLE_EQ(settings->contactWrenchGate->rampTime, 0.04);
  EXPECT_EQ(settings->wbMpcFeedforward, WbMpcFeedforward::kGravityCompensation);
  EXPECT_TRUE(settings->telemetrySinks.empty()) << "telemetry turned off by an empty list of sinks";
  EXPECT_DOUBLE_EQ(*settings->telemetryFrequency, 50.0);
  EXPECT_DOUBLE_EQ(settings->mpcLinkPolicyTimeout, 0.35);
}

TEST(RobotProcessSettings, AValueOfTheWrongTypeIsRefusedByItsKey) {
  for (const std::string& document :
       {std::string("simGantryCatchLift: high\n"), std::string("simVisualizations: zmp\n"),
        std::string("contact_wrench_gate:\n  rampTime: soon\n"), std::string("mpcLink:\n  policyTimeout: x\n")}) {
    const absl::StatusOr<RobotProcessSettings> settings = parseRobotProcessSettings(document, "task.yaml");
    EXPECT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument) << document;
    EXPECT_NE(settings.status().message().find("task.yaml"), std::string::npos) << settings.status().message();
  }
  EXPECT_FALSE(parseRobotProcessSettings("wbMpcFeedforward: magic\n", "test").ok());
  EXPECT_FALSE(parseRobotProcessSettings("telemetryFrequency: 0\n", "test").ok());
  EXPECT_FALSE(parseRobotProcessSettings("mpcLink:\n  policyTimeout: 0\n", "test").ok());
  EXPECT_FALSE(parseRobotProcessSettings("contact_wrench_gate:\n  rampTime: -0.1\n", "test").ok());
  EXPECT_FALSE(parseRobotProcessSettings("[1, 2]", "test").ok()) << "not a map of keys";
}

TEST(RobotProcessSettings, TheRetiredBooleansAreRefusedWithTheirReplacement) {
  const absl::StatusOr<RobotProcessSettings> telemetry = parseRobotProcessSettings("enableTelemetry: true\n", "task.yaml");
  EXPECT_EQ(telemetry.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_NE(telemetry.status().message().find("telemetrySinks"), std::string::npos) << telemetry.status().message();
  const absl::StatusOr<RobotProcessSettings> snakeCase = parseRobotProcessSettings("enable_telemetry: false\n", "task.yaml");
  EXPECT_NE(snakeCase.status().message().find("telemetrySinks"), std::string::npos) << snakeCase.status().message();
  const absl::StatusOr<RobotProcessSettings> feedforward = parseRobotProcessSettings("useGravityCompFeedforward: false\n", "task.yaml");
  EXPECT_EQ(feedforward.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_NE(feedforward.status().message().find("wbMpcFeedforward"), std::string::npos) << feedforward.status().message();
}

TEST(RobotProcessSettings, EveryShippedTaskFileReadsWithRegisteredNamesAndTheShippedDefaults) {
  robot::model::ContactEstimatorRegistry estimators;
  const TelemetrySinkRegistry sinks;
  for (const std::string& taskFile : kShippedTaskFiles) {
    const absl::StatusOr<RobotProcessSettings> settings = loadRobotProcessSettings(taskFile);
    ASSERT_TRUE(settings.ok()) << taskFile << ": " << settings.status();
    const std::string estimator = robot::model::ContactEstimatorRegistry::canonicalName(settings->contactEstimator);
    EXPECT_TRUE(estimators.has(estimator) || estimator == "cheater_sim") << taskFile << ": contactEstimator " << estimator;
    for (const std::string& sink : settings->telemetrySinks) EXPECT_TRUE(sinks.has(sink)) << taskFile << ": telemetry sink " << sink;
    EXPECT_FALSE(settings->telemetrySinks.empty()) << taskFile << ": every robot publishes its telemetry";
    EXPECT_TRUE(robot::mujoco_sim_interface::gantryHoldFromName(settings->simulator.gantryHold).ok()) << taskFile;
    // The debugging feedforward stays off in what ships.
    EXPECT_EQ(settings->wbMpcFeedforward, WbMpcFeedforward::kInverseDynamics) << taskFile;
  }
}

}  // namespace
}  // namespace ocs2::humanoid
