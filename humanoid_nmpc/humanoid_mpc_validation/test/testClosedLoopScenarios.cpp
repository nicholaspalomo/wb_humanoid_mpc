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

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/command/WalkingVelocityCommand.h"
#include "humanoid_common_mpc_app/node/WalkingVelocityCommandConversions.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"
#include "humanoid_mpc_validation/closed_loop/ClosedLoopScenario.h"
#include "humanoid_mpc_validation/closed_loop/MetricBands.h"
#include "humanoid_mpc_validation/closed_loop/RobotConfiguration.h"
#include "humanoid_mpc_validation/io/JsonValue.h"

/*
 * The scenario set of the quaternion design's section 4.5, the GUI's command
 * conversion, the configuration registry and the bands against a baseline: what
 * the closed-loop runs command and how their results are judged, without
 * running one.
 */

namespace ocs2::humanoid::validation {
namespace {

/** The integral of a command component over a scenario, by the rectangle rule
 * on a fine grid. */
double integrate(const ClosedLoopScenario& scenario,
                 int component,
                 double step = 1.0e-3) {
  double sum = 0.0;
  for (double t = 0.5 * step; t < getCommandDuration(scenario); t += step)
    sum += commandAt(scenario, t)(component) * step;
  return sum;
}

TEST(ClosedLoopScenarios, TheSetIsTheDesignsWithTheSlowerWalkAndTheSmokeRun) {
  std::set<std::string> names;
  for (const ClosedLoopScenario& scenario : closedLoopScenarios()) {
    EXPECT_TRUE(names.insert(scenario.name).second)
        << "the name " << scenario.name << " is used twice";
    EXPECT_FALSE(scenario.description.empty()) << scenario.name;
    EXPECT_GT(getCommandDuration(scenario), 0.0) << scenario.name;
  }
  const std::set<std::string> expected = {
      "standing",          "walk_0p5",    "walk_0p3", "lateral_0p2",
      "turn_in_place_720", "turn_1radps", "arc",      "smoke"};
  EXPECT_EQ(names, expected);
  EXPECT_EQ(findClosedLoopScenario("missing").status().code(),
            absl::StatusCode::kNotFound);
}

// The descriptions are formatted from the commands' constants: the text they
// read as.
TEST(ClosedLoopScenarios, TheDescriptionsSpellTheCommands) {
  const std::vector<std::pair<std::string, std::string>> expected = {
      {"standing", "at rest for 10 s"},
      {"walk_0p5", "0.5 m/s forward for 15 s, then at rest"},
      {"walk_0p3",
       "0.3 m/s forward for 15 s, then at rest (the slower walking variant)"},
      {"lateral_0p2", "0.2 m/s to the left for 10 s, then at rest"},
      {"turn_in_place_720",
       "+0.5 rad/s in place through 720 degrees, -0.5 rad/s back, then at "
       "rest"},
      {"turn_1radps", "1 rad/s in place for 12.6 s, then at rest"},
      {"arc",
       "0.3 m/s forward while turning at 0.3 rad/s for 15 s, then at rest"},
      {"smoke",
       "0.3 m/s forward while turning at 0.2 rad/s for 1.5 s after 0.5 s of "
       "standing (the determinism test)"},
  };
  for (const std::pair<std::string, std::string>& entry : expected) {
    const absl::StatusOr<ClosedLoopScenario> scenario =
        findClosedLoopScenario(entry.first);
    ASSERT_TRUE(scenario.ok()) << entry.first;
    EXPECT_EQ(scenario->description, entry.second) << entry.first;
  }
}

TEST(ClosedLoopScenarios, TheCommandsAreWhatSection4Point5Asks) {
  const ClosedLoopScenario standing = *findClosedLoopScenario("standing");
  EXPECT_DOUBLE_EQ(getCommandDuration(standing), 10.0);
  EXPECT_EQ(commandAt(standing, /*timeSinceStart=*/5.0),
            Eigen::Vector3d::Zero());

  const ClosedLoopScenario walk = *findClosedLoopScenario("walk_0p5");
  EXPECT_EQ(commandAt(walk, /*timeSinceStart=*/7.0),
            Eigen::Vector3d(0.5, 0.0, 0.0));
  EXPECT_NEAR(integrate(walk, /*component=*/0), 0.5 * 15.0, 1.0e-6)
      << "15 s at 0.5 m/s";

  const ClosedLoopScenario lateral = *findClosedLoopScenario("lateral_0p2");
  EXPECT_EQ(commandAt(lateral, /*timeSinceStart=*/1.0),
            Eigen::Vector3d(0.0, 0.2, 0.0));

  // Through 720 degrees and back: the heading the command integrates to peaks
  // at 4 pi and returns to 0.
  const ClosedLoopScenario turn = *findClosedLoopScenario("turn_in_place_720");
  double heading = 0.0;
  double peakHeading = 0.0;
  for (const CommandSegment& piece : turn.segments) {
    heading += piece.duration * piece.yawRate;
    peakHeading = std::max(peakHeading, heading);
  }
  EXPECT_NEAR(peakHeading, 4.0 * M_PI, 1.0e-12);
  EXPECT_NEAR(heading, 0.0, 1.0e-12);
  EXPECT_EQ(turn.segments[0].forwardVelocity, 0.0) << "in place";

  const ClosedLoopScenario fastTurn = *findClosedLoopScenario("turn_1radps");
  EXPECT_NEAR(integrate(fastTurn, /*component=*/2), 12.6, 1.0e-6);

  const ClosedLoopScenario arc = *findClosedLoopScenario("arc");
  EXPECT_EQ(commandAt(arc, /*timeSinceStart=*/3.0),
            Eigen::Vector3d(0.3, 0.0, 0.3));

  // Every motion ends at rest, so the metrics see the stop; nothing is
  // commanded outside the segments.
  for (const ClosedLoopScenario& scenario : closedLoopScenarios()) {
    EXPECT_EQ(commandAt(scenario, /*timeSinceStart=*/-0.1),
              Eigen::Vector3d::Zero())
        << scenario.name;
    EXPECT_EQ(commandAt(scenario, getCommandDuration(scenario) + 0.1),
              Eigen::Vector3d::Zero())
        << scenario.name;
  }
}

TEST(ClosedLoopScenarios, ASegmentHoldsItsStartButNotItsEnd) {
  ClosedLoopScenario scenario;
  scenario.segments = {CommandSegment{.duration = 1.0, .forwardVelocity = 0.1},
                       CommandSegment{.duration = 1.0, .forwardVelocity = 0.2}};
  EXPECT_EQ(commandAt(scenario, /*timeSinceStart=*/0.0)(0), 0.1);
  EXPECT_EQ(commandAt(scenario, /*timeSinceStart=*/1.0)(0), 0.2);
  EXPECT_EQ(commandAt(scenario, /*timeSinceStart=*/2.0)(0), 0.0);
}

TEST(GuiVelocityCommand,
     TheSticksScaleBackToTheCommandAndSaturateAtTheirStops) {
  const Eigen::Vector3d limits(1.2, 0.25, 1.0);
  const GuiVelocityCommand within = toGuiVelocityCommand(
      Eigen::Vector3d(0.5, 0.2, -0.5), limits, /*pelvisHeight=*/0.8952);
  EXPECT_FALSE(within.saturated);
  // What ProceduralMpcMotionManager::scaleWalkingVelocityCommand gives back.
  EXPECT_NEAR(within.message(0) * limits(0), 0.5, 1.0e-15);
  EXPECT_NEAR(within.message(1) * limits(1), 0.2, 1.0e-15);
  EXPECT_NEAR(within.message(3) * limits(2), -0.5, 1.0e-15);
  EXPECT_EQ(within.message(2), 0.8952);

  const GuiVelocityCommand beyond = toGuiVelocityCommand(
      Eigen::Vector3d(0.0, 0.3, 0.0), limits, /*pelvisHeight=*/0.8952);
  EXPECT_TRUE(beyond.saturated)
      << "0.3 m/s sideways is beyond a 0.25 m/s limit";
  EXPECT_EQ(beyond.message(1), 1.0);
  const GuiVelocityCommand atLimit = toGuiVelocityCommand(
      Eigen::Vector3d(0.0, 0.2, 0.0), Eigen::Vector3d(0.8, 0.2, 1.0),
      /*pelvisHeight=*/0.8135);
  EXPECT_FALSE(atLimit.saturated) << "exactly at the stick's stop is reachable";

  EXPECT_EQ(clampGuiVelocityMessage(Eigen::Vector4d(2.0, -3.0, 0.05, 1.5)),
            Eigen::Vector4d(1.0, -1.0, 0.2, 1.0));
}

TEST(GuiVelocityCommand, TheClampIsTheOneTheMpcNodeAppliesToAReceivedMessage) {
  // The runner's clamp (for the saturation flag and the commanded height)
  // against the MPC node's conversion of operator/walking_velocity_command,
  // inside, at and beyond every range: one clamp,
  // node::clampWalkingVelocityCommand().
  const std::vector<Eigen::Vector4d> messages = {
      Eigen::Vector4d(0.3, -0.2, 0.8, 0.1),
      Eigen::Vector4d(1.0, -1.0, 0.2, 1.0),
      Eigen::Vector4d(-1.0, 1.0, 1.0, -1.0),
      Eigen::Vector4d(2.0, -3.0, 0.05, 1.5),
      Eigen::Vector4d(-7.0, 4.0, 1.7, -2.5),
      Eigen::Vector4d(1.0e-12 + 1.0, 0.0, 0.2 - 1.0e-12, 0.0),
  };
  for (const Eigen::Vector4d& message : messages) {
    humanoid_mpc_msgs::WalkingVelocityCommand proto;
    proto.set_linear_velocity_x(message(0));
    proto.set_linear_velocity_y(message(1));
    proto.set_desired_pelvis_height(message(2));
    proto.set_angular_velocity_z(message(3));
    const absl::StatusOr<WalkingVelocityCommand> command =
        node::walkingVelocityCommandFromProto(proto);
    ASSERT_TRUE(command.ok()) << command.status();
    const Eigen::Vector4d clamped = clampGuiVelocityMessage(message);
    EXPECT_EQ(clamped(0), command->linear_velocity_x) << message.transpose();
    EXPECT_EQ(clamped(1), command->linear_velocity_y) << message.transpose();
    EXPECT_EQ(clamped(2), command->desired_pelvis_height)
        << message.transpose();
    EXPECT_EQ(clamped(3), command->angular_velocity_z) << message.transpose();
  }
}

TEST(RobotConfigurations, TheFiveConfigurationsNameTheirFiles) {
  const std::vector<RobotConfiguration>& configurations = robotConfigurations();
  ASSERT_EQ(configurations.size(), 5u);
  size_t wholeBody = 0;
  for (const RobotConfiguration& configuration : configurations) {
    EXPECT_TRUE(findRobotConfiguration(configuration.name).ok())
        << configuration.name;
    if (configuration.formulation == MpcFormulation::kWholeBody) ++wholeBody;
    for (const std::string& file :
         {configuration.taskFile, configuration.referenceFile,
          configuration.urdfFile, configuration.sceneFile,
          configuration.pdGainsFile, configuration.gaitFile}) {
      EXPECT_FALSE(file.empty()) << configuration.name;
    }
  }
  EXPECT_EQ(wholeBody, 1u);
  EXPECT_EQ(findRobotConfiguration("unknown").status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(formulationName(MpcFormulation::kWholeBody), "whole_body");
}

/**
 * The variables of the launch file at `path`
 * (tools/launch/proto/launch_file.proto), each written on a line of its own as
 * `variables { name: "<name>" value: "<value>" }`, with the references to other
 * variables ("{config_dir}/mpc/task.textproto") substituted. Empty when the
 * file cannot be read.
 */
absl::flat_hash_map<std::string, std::string> launchFileVariables(
    const std::string& path) {
  absl::flat_hash_map<std::string, std::string> variables;
  std::ifstream file(path);
  const std::regex declaration(
      R"re(^variables \{ name: "(\w+)" value: "([^"]*)" \})re");
  std::string line;
  while (std::getline(file, line)) {
    std::smatch match;
    if (std::regex_search(line, match, declaration))
      variables[match[1].str()] = match[2].str();
  }
  // Every "{name}" by the value of `name`, pass by pass: a value may refer to
  // variables that refer to others.
  std::vector<std::pair<std::string, std::string>> references;
  for (const std::pair<const std::string, std::string>& variable : variables) {
    references.emplace_back(absl::StrCat("{", variable.first, "}"),
                            variable.second);
  }
  for (std::pair<const std::string, std::string>& variable : variables) {
    for (int pass = 0; pass < 8; ++pass) {
      const std::string substituted =
          absl::StrReplaceAll(variable.second, references);
      if (substituted == variable.second) break;
      variable.second = substituted;
    }
  }
  return variables;
}

TEST(RobotConfigurations, TheFilesAreTheOnesTheRobotsLaunchFilesName) {
  // The robot process reads launch/robot.textproto's files and the MPC node
  // launch/mpc.textproto's; the runner, which stands in for both, must run on
  // the same ones.
  for (const RobotConfiguration& configuration : robotConfigurations()) {
    // <package>/config/mpc/task.textproto: the launch files are
    // <package>/launch/.
    const std::filesystem::path package =
        std::filesystem::path(configuration.taskFile)
            .parent_path()
            .parent_path()
            .parent_path();
    const absl::flat_hash_map<std::string, std::string> robot =
        launchFileVariables((package / "launch/robot.textproto").string());
    const absl::flat_hash_map<std::string, std::string> mpc =
        launchFileVariables((package / "launch/mpc.textproto").string());
    ASSERT_FALSE(robot.empty()) << configuration.name << ": no variables in "
                                << package / "launch/robot.textproto";
    ASSERT_FALSE(mpc.empty()) << configuration.name << ": no variables in "
                              << package / "launch/mpc.textproto";
    const std::vector<std::pair<std::string, std::string>> robotFiles = {
        {"task_file", configuration.taskFile},
        {"reference_file", configuration.referenceFile},
        {"urdf_file", configuration.urdfFile},
        {"mjcf_file", configuration.sceneFile}};
    for (const std::pair<std::string, std::string>& file : robotFiles) {
      ASSERT_TRUE(robot.contains(file.first))
          << configuration.name << ": robot.textproto has no " << file.first;
      EXPECT_EQ(robot.at(file.first), file.second)
          << configuration.name << ": robot.textproto's " << file.first;
    }
    const std::vector<std::pair<std::string, std::string>> mpcFiles = {
        {"task_file", configuration.taskFile},
        {"reference_file", configuration.referenceFile},
        {"urdf_file", configuration.urdfFile},
        {"gait_file", configuration.gaitFile}};
    for (const std::pair<std::string, std::string>& file : mpcFiles) {
      ASSERT_TRUE(mpc.contains(file.first))
          << configuration.name << ": mpc.textproto has no " << file.first;
      EXPECT_EQ(mpc.at(file.first), file.second)
          << configuration.name << ": mpc.textproto's " << file.first;
    }
  }
}

TEST(RobotConfigurations, TheLaunchFileReaderSubstitutesTheVariables) {
  const std::filesystem::path file =
      std::filesystem::path(::testing::TempDir()) / "variables.textproto";
  {
    std::ofstream out(file);
    out << "# a comment\n"
        << "variables { name: \"root\" value: \"robot_models/x\" }\n"
        << "variables { name: \"config\" value: \"{root}/config\" }\n"
        << "variables { name: \"task\" value: "
           "\"{config}/mpc/{root}.textproto\" }\n";
  }
  const absl::flat_hash_map<std::string, std::string> variables =
      launchFileVariables(file.string());
  ASSERT_EQ(variables.size(), 3u);
  EXPECT_EQ(variables.at("config"), "robot_models/x/config");
  EXPECT_EQ(variables.at("task"),
            "robot_models/x/config/mpc/robot_models/x.textproto");
  EXPECT_TRUE(launchFileVariables((std::filesystem::path(::testing::TempDir()) /
                                   "missing.textproto")
                                      .string())
                  .empty());
}

// -----------------------------------------------------------------------------------------------------
// the bands

/** The metrics a band comparison reads. */
struct DocumentValues {
  bool survived = true;
  double tiltRms = 0.03;
  double velocityError = 0.1;
  double p99 = 10.0;
};

JsonValue metricsDocument(const DocumentValues& values) {
  JsonValue document = JsonValue::object();
  JsonValue& survival = document.set("survival", JsonValue::object());
  survival.set("survived", JsonValue::boolean(values.survived));
  survival.set("fall_reason",
               JsonValue::string(values.survived ? "" : "the base tilted"));
  document.set("tilt", JsonValue::object())
      .set("rms_rad", JsonValue::number(values.tiltRms));
  document.set("velocity", JsonValue::object())
      .set("rms_error_mps", JsonValue::number(values.velocityError));
  document.set("base_height", JsonValue::object())
      .set("mean_m", JsonValue::number(0.8));
  document.set("solve_time_ms", JsonValue::object())
      .set("total", JsonValue::object())
      .set("p99", JsonValue::number(values.p99));
  document.set("quaternion_norm", JsonValue::object())
      .set("max_deviation", JsonValue());
  return document;
}

TEST(MetricBands, ARunIsWithinTheBandsOfItself) {
  const JsonValue baseline = metricsDocument({});
  const ClosedLoopComparison itself =
      compareClosedLoopMetrics(baseline, baseline, BandOptions());
  EXPECT_TRUE(itself.violations.empty());
  EXPECT_TRUE(itself.improvements.empty());
}

TEST(MetricBands, EachBandIsTheLargerOfItsRelativeAndAbsoluteParts) {
  const JsonValue baseline =
      metricsDocument({.tiltRms = 0.01, .velocityError = 0.5, .p99 = 10.0});
  // tilt: max(20 % of 0.01, 0.005) = 0.005; velocity: max(15 % of 0.5, 0.02) =
  // 0.075; p99: +10 %.
  EXPECT_TRUE(compareClosedLoopMetrics(
                  metricsDocument(
                      {.tiltRms = 0.0149, .velocityError = 0.574, .p99 = 10.9}),
                  baseline, BandOptions())
                  .violations.empty());
  const std::vector<std::string> outside =
      compareClosedLoopMetrics(
          metricsDocument(
              {.tiltRms = 0.0151, .velocityError = 0.576, .p99 = 11.1}),
          baseline, BandOptions())
          .violations;
  ASSERT_EQ(outside.size(), 3u);
  EXPECT_EQ(outside[0].rfind("tilt.rms_rad", 0), 0u) << outside[0];
  EXPECT_EQ(outside[1].rfind("velocity.rms_error_mps", 0), 0u) << outside[1];
  EXPECT_EQ(outside[2].rfind("solve_time_ms.total.p99", 0), 0u) << outside[2];

  BandOptions withoutTiming;
  withoutTiming.compareSolveTime = false;
  EXPECT_EQ(compareClosedLoopMetrics(
                metricsDocument(
                    {.tiltRms = 0.0151, .velocityError = 0.576, .p99 = 100.0}),
                baseline, withoutTiming)
                .violations.size(),
            2u);
}

TEST(MetricBands, SurvivalMustBeEqualOrBetter) {
  const JsonValue survived = metricsDocument({});
  const JsonValue fell = metricsDocument(
      {.survived = false, .tiltRms = 0.9, .velocityError = 2.0});
  const std::vector<std::string> worse =
      compareClosedLoopMetrics(fell, survived, BandOptions()).violations;
  ASSERT_EQ(worse.size(), 1u);
  EXPECT_EQ(worse[0].rfind("survival", 0), 0u) << worse[0];
  const ClosedLoopComparison better =
      compareClosedLoopMetrics(survived, fell, BandOptions());
  EXPECT_TRUE(better.violations.empty()) << "better than a baseline that fell";
  ASSERT_EQ(better.improvements.size(), 1u) << "and reported";
  EXPECT_EQ(better.improvements[0].rfind("survival", 0), 0u)
      << better.improvements[0];
  const ClosedLoopComparison bothFell =
      compareClosedLoopMetrics(fell, fell, BandOptions());
  EXPECT_TRUE(bothFell.violations.empty());
  EXPECT_TRUE(bothFell.improvements.empty());
}

TEST(MetricBands, TheQuaternionNormMustStayBelowItsBound) {
  const JsonValue baseline = metricsDocument({});
  JsonValue candidate = baseline;
  candidate.set("quaternion_norm", JsonValue::object())
      .set("max_deviation", JsonValue::number(1.0e-8));
  EXPECT_EQ(compareClosedLoopMetrics(candidate, baseline, BandOptions())
                .violations.size(),
            1u);
  candidate.set("quaternion_norm", JsonValue::object())
      .set("max_deviation", JsonValue::number(1.0e-12));
  EXPECT_TRUE(compareClosedLoopMetrics(candidate, baseline, BandOptions())
                  .violations.empty());
}

TEST(MetricBands, ANonFiniteQuaternionNormFailsItsBound) {
  const JsonValue baseline = metricsDocument({});
  // As held in memory: a NaN is not below the bound.
  JsonValue inMemory = baseline;
  inMemory.set("quaternion_norm", JsonValue::object())
      .set("max_deviation",
           JsonValue::number(std::numeric_limits<double>::quiet_NaN()));
  const std::vector<std::string> nan =
      compareClosedLoopMetrics(inMemory, baseline, BandOptions()).violations;
  ASSERT_EQ(nan.size(), 1u);
  EXPECT_EQ(nan[0].rfind("quaternion_norm.max_deviation", 0), 0u) << nan[0];

  // As read from its text, which writes the NaN as null: the count of the
  // solves that had one fails the bound.
  JsonValue fromText = baseline;
  JsonValue& norm = fromText.set("quaternion_norm", JsonValue::object());
  norm.set("max_deviation", JsonValue());
  norm.set("non_finite_solves", JsonValue::number(2.0));
  const std::vector<std::string> counted =
      compareClosedLoopMetrics(fromText, baseline, BandOptions()).violations;
  ASSERT_EQ(counted.size(), 1u);
  EXPECT_EQ(counted[0].rfind("quaternion_norm.non_finite_solves", 0), 0u)
      << counted[0];
  norm.set("non_finite_solves", JsonValue::number(0.0));
  EXPECT_TRUE(compareClosedLoopMetrics(fromText, baseline, BandOptions())
                  .violations.empty());
}

}  // namespace
}  // namespace ocs2::humanoid::validation
