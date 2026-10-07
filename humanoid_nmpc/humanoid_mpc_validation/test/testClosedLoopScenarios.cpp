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
#include <functional>
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
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/command/WalkingVelocityCommand.h"
#include "humanoid_common_mpc_app/node/WalkingVelocityCommandConversions.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"
#include "humanoid_mpc_validation/closed_loop/ClosedLoopScenario.h"
#include "humanoid_mpc_validation/closed_loop/MetricBands.h"
#include "humanoid_mpc_validation/closed_loop/RobotConfiguration.h"
#include "humanoid_mpc_validation/closed_loop/TimeSeriesLabels.h"
#include "humanoid_mpc_validation/io/GoldenIo.h"
#include "humanoid_mpc_validation/io/JsonValue.h"

/*
 * The scenario set of the quaternion design's section 4.5, the GUI's command conversion, the configuration registry and
 * the bands against a baseline: what the closed-loop runs command and how their results are judged, without running
 * one.
 */

namespace ocs2::humanoid::validation {
namespace {

/** The integral of a command component over a scenario, by the rectangle rule on a fine grid. */
double integrate(const ClosedLoopScenario& scenario, int component, double step = 1.0e-3) {
  double sum = 0.0;
  for (double t = 0.5 * step; t < getCommandDuration(scenario); t += step) sum += commandAt(scenario, t)(component) * step;
  return sum;
}

TEST(ClosedLoopScenarios, TheSetIsTheDesignsWithTheSlowerWalkAndTheSmokeRun) {
  std::set<std::string> names;
  for (const ClosedLoopScenario& scenario : closedLoopScenarios()) {
    EXPECT_TRUE(names.insert(scenario.name).second) << "the name " << scenario.name << " is used twice";
    EXPECT_FALSE(scenario.description.empty()) << scenario.name;
    EXPECT_GT(getCommandDuration(scenario), 0.0) << scenario.name;
  }
  const std::set<std::string> expected = {"standing",          "walk_0p5",    "walk_0p3", "lateral_0p2",
                                          "turn_in_place_720", "turn_1radps", "arc",      "smoke"};
  EXPECT_EQ(names, expected);
  EXPECT_EQ(findClosedLoopScenario("missing").status().code(), absl::StatusCode::kNotFound);
}

TEST(ClosedLoopScenarios, TheCommandsAreWhatSection4Point5Asks) {
  const ClosedLoopScenario standing = *findClosedLoopScenario("standing");
  EXPECT_DOUBLE_EQ(getCommandDuration(standing), 10.0);
  EXPECT_EQ(commandAt(standing, /*timeSinceStart=*/5.0), Eigen::Vector3d::Zero());

  const ClosedLoopScenario walk = *findClosedLoopScenario("walk_0p5");
  EXPECT_EQ(commandAt(walk, /*timeSinceStart=*/7.0), Eigen::Vector3d(0.5, 0.0, 0.0));
  EXPECT_NEAR(integrate(walk, /*component=*/0), 0.5 * 15.0, 1.0e-6) << "15 s at 0.5 m/s";

  const ClosedLoopScenario lateral = *findClosedLoopScenario("lateral_0p2");
  EXPECT_EQ(commandAt(lateral, /*timeSinceStart=*/1.0), Eigen::Vector3d(0.0, 0.2, 0.0));

  // Through 720 degrees and back: the heading the command integrates to peaks at 4 pi and returns to 0.
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
  EXPECT_EQ(commandAt(arc, /*timeSinceStart=*/3.0), Eigen::Vector3d(0.3, 0.0, 0.3));

  // Every motion ends at rest, so the metrics see the stop; nothing is commanded outside the segments.
  for (const ClosedLoopScenario& scenario : closedLoopScenarios()) {
    EXPECT_EQ(commandAt(scenario, /*timeSinceStart=*/-0.1), Eigen::Vector3d::Zero()) << scenario.name;
    EXPECT_EQ(commandAt(scenario, getCommandDuration(scenario) + 0.1), Eigen::Vector3d::Zero()) << scenario.name;
  }
}

TEST(ClosedLoopScenarios, ASegmentHoldsItsStartButNotItsEnd) {
  ClosedLoopScenario scenario;
  scenario.segments = {CommandSegment{.duration = 1.0, .forwardVelocity = 0.1}, CommandSegment{.duration = 1.0, .forwardVelocity = 0.2}};
  EXPECT_EQ(commandAt(scenario, /*timeSinceStart=*/0.0)(0), 0.1);
  EXPECT_EQ(commandAt(scenario, /*timeSinceStart=*/1.0)(0), 0.2);
  EXPECT_EQ(commandAt(scenario, /*timeSinceStart=*/2.0)(0), 0.0);
}

TEST(GuiVelocityCommand, TheSticksScaleBackToTheCommandAndSaturateAtTheirStops) {
  const Eigen::Vector3d limits(1.2, 0.25, 1.0);
  const GuiVelocityCommand within = toGuiVelocityCommand(Eigen::Vector3d(0.5, 0.2, -0.5), limits, /*pelvisHeight=*/0.8952);
  EXPECT_FALSE(within.saturated);
  // What ProceduralMpcMotionManager::scaleWalkingVelocityCommand gives back.
  EXPECT_NEAR(within.message(0) * limits(0), 0.5, 1.0e-15);
  EXPECT_NEAR(within.message(1) * limits(1), 0.2, 1.0e-15);
  EXPECT_NEAR(within.message(3) * limits(2), -0.5, 1.0e-15);
  EXPECT_EQ(within.message(2), 0.8952);

  const GuiVelocityCommand beyond = toGuiVelocityCommand(Eigen::Vector3d(0.0, 0.3, 0.0), limits, /*pelvisHeight=*/0.8952);
  EXPECT_TRUE(beyond.saturated) << "0.3 m/s sideways is beyond a 0.25 m/s limit";
  EXPECT_EQ(beyond.message(1), 1.0);
  const GuiVelocityCommand atLimit =
      toGuiVelocityCommand(Eigen::Vector3d(0.0, 0.2, 0.0), Eigen::Vector3d(0.8, 0.2, 1.0), /*pelvisHeight=*/0.8135);
  EXPECT_FALSE(atLimit.saturated) << "exactly at the stick's stop is reachable";

  EXPECT_EQ(clampGuiVelocityMessage(Eigen::Vector4d(2.0, -3.0, 0.05, 1.5)), Eigen::Vector4d(1.0, -1.0, 0.2, 1.0));
}

TEST(GuiVelocityCommand, TheClampIsTheOneTheMpcNodeAppliesToAReceivedMessage) {
  // The runner's clamp (for the saturation flag and the commanded height) against the MPC node's conversion of
  // operator/walking_velocity_command, inside, at and beyond every range: one clamp, node::clampWalkingVelocityCommand().
  const std::vector<Eigen::Vector4d> messages = {
      Eigen::Vector4d(0.3, -0.2, 0.8, 0.1),  Eigen::Vector4d(1.0, -1.0, 0.2, 1.0),  Eigen::Vector4d(-1.0, 1.0, 1.0, -1.0),
      Eigen::Vector4d(2.0, -3.0, 0.05, 1.5), Eigen::Vector4d(-7.0, 4.0, 1.7, -2.5), Eigen::Vector4d(1.0e-12 + 1.0, 0.0, 0.2 - 1.0e-12, 0.0),
  };
  for (const Eigen::Vector4d& message : messages) {
    humanoid_mpc_msgs::WalkingVelocityCommand proto;
    proto.set_linear_velocity_x(message(0));
    proto.set_linear_velocity_y(message(1));
    proto.set_desired_pelvis_height(message(2));
    proto.set_angular_velocity_z(message(3));
    const absl::StatusOr<WalkingVelocityCommand> command = node::walkingVelocityCommandFromProto(proto);
    ASSERT_TRUE(command.ok()) << command.status();
    const Eigen::Vector4d clamped = clampGuiVelocityMessage(message);
    EXPECT_EQ(clamped(0), command->linear_velocity_x) << message.transpose();
    EXPECT_EQ(clamped(1), command->linear_velocity_y) << message.transpose();
    EXPECT_EQ(clamped(2), command->desired_pelvis_height) << message.transpose();
    EXPECT_EQ(clamped(3), command->angular_velocity_z) << message.transpose();
  }
}

TEST(RobotConfigurations, TheFiveConfigurationsNameTheirFiles) {
  const std::vector<RobotConfiguration>& configurations = robotConfigurations();
  ASSERT_EQ(configurations.size(), 5u);
  size_t wholeBody = 0;
  for (const RobotConfiguration& configuration : configurations) {
    EXPECT_TRUE(findRobotConfiguration(configuration.name).ok()) << configuration.name;
    if (configuration.formulation == MpcFormulation::kWholeBody) ++wholeBody;
    for (const std::string& file : {configuration.taskFile, configuration.referenceFile, configuration.urdfFile, configuration.sceneFile,
                                    configuration.pdGainsFile, configuration.gaitFile}) {
      EXPECT_FALSE(file.empty()) << configuration.name;
    }
  }
  EXPECT_EQ(wholeBody, 1u);
  EXPECT_EQ(findRobotConfiguration("unknown").status().code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(formulationName(MpcFormulation::kWholeBody), "whole_body");
}

/**
 * The variables of the launch file at `path` (tools/launch/proto/launch_file.proto), each written on a line of its own as
 * `variables { name: "<name>" value: "<value>" }`, with the references to other variables ("{config_dir}/mpc/task.textproto")
 * substituted. Empty when the file cannot be read.
 */
absl::flat_hash_map<std::string, std::string> launchFileVariables(const std::string& path) {
  absl::flat_hash_map<std::string, std::string> variables;
  std::ifstream file(path);
  const std::regex declaration(R"re(^variables \{ name: "(\w+)" value: "([^"]*)" \})re");
  std::string line;
  while (std::getline(file, line)) {
    std::smatch match;
    if (std::regex_search(line, match, declaration)) variables[match[1].str()] = match[2].str();
  }
  // Every "{name}" by the value of `name`, pass by pass: a value may refer to variables that refer to others.
  std::vector<std::pair<std::string, std::string>> references;
  for (const std::pair<const std::string, std::string>& variable : variables) {
    references.emplace_back(absl::StrCat("{", variable.first, "}"), variable.second);
  }
  for (std::pair<const std::string, std::string>& variable : variables) {
    for (int pass = 0; pass < 8; ++pass) {
      const std::string substituted = absl::StrReplaceAll(variable.second, references);
      if (substituted == variable.second) break;
      variable.second = substituted;
    }
  }
  return variables;
}

TEST(RobotConfigurations, TheFilesAreTheOnesTheRobotsLaunchFilesName) {
  // The robot process reads launch/robot.textproto's files and the MPC node launch/mpc.textproto's; the runner, which
  // stands in for both, must run on the same ones.
  for (const RobotConfiguration& configuration : robotConfigurations()) {
    // <package>/config/mpc/task.textproto: the launch files are <package>/launch/.
    const std::filesystem::path package = std::filesystem::path(configuration.taskFile).parent_path().parent_path().parent_path();
    const absl::flat_hash_map<std::string, std::string> robot = launchFileVariables((package / "launch/robot.textproto").string());
    const absl::flat_hash_map<std::string, std::string> mpc = launchFileVariables((package / "launch/mpc.textproto").string());
    ASSERT_FALSE(robot.empty()) << configuration.name << ": no variables in " << package / "launch/robot.textproto";
    ASSERT_FALSE(mpc.empty()) << configuration.name << ": no variables in " << package / "launch/mpc.textproto";
    const std::vector<std::pair<std::string, std::string>> robotFiles = {{"task_file", configuration.taskFile},
                                                                         {"reference_file", configuration.referenceFile},
                                                                         {"urdf_file", configuration.urdfFile},
                                                                         {"mjcf_file", configuration.sceneFile}};
    for (const std::pair<std::string, std::string>& file : robotFiles) {
      ASSERT_TRUE(robot.contains(file.first)) << configuration.name << ": robot.textproto has no " << file.first;
      EXPECT_EQ(robot.at(file.first), file.second) << configuration.name << ": robot.textproto's " << file.first;
    }
    const std::vector<std::pair<std::string, std::string>> mpcFiles = {{"task_file", configuration.taskFile},
                                                                       {"reference_file", configuration.referenceFile},
                                                                       {"urdf_file", configuration.urdfFile},
                                                                       {"gait_file", configuration.gaitFile}};
    for (const std::pair<std::string, std::string>& file : mpcFiles) {
      ASSERT_TRUE(mpc.contains(file.first)) << configuration.name << ": mpc.textproto has no " << file.first;
      EXPECT_EQ(mpc.at(file.first), file.second) << configuration.name << ": mpc.textproto's " << file.first;
    }
  }
}

TEST(RobotConfigurations, TheLaunchFileReaderSubstitutesTheVariables) {
  const std::filesystem::path file = std::filesystem::path(::testing::TempDir()) / "variables.textproto";
  {
    std::ofstream out(file);
    out << "# a comment\n"
        << "variables { name: \"root\" value: \"robot_models/x\" }\n"
        << "variables { name: \"config\" value: \"{root}/config\" }\n"
        << "variables { name: \"task\" value: \"{config}/mpc/{root}.textproto\" }\n";
  }
  const absl::flat_hash_map<std::string, std::string> variables = launchFileVariables(file.string());
  ASSERT_EQ(variables.size(), 3u);
  EXPECT_EQ(variables.at("config"), "robot_models/x/config");
  EXPECT_EQ(variables.at("task"), "robot_models/x/config/mpc/robot_models/x.textproto");
  EXPECT_TRUE(launchFileVariables((std::filesystem::path(::testing::TempDir()) / "missing.textproto").string()).empty());
}

// ----------------------------------------------------------------------------------------------------- the bands

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
  survival.set("fall_reason", JsonValue::string(values.survived ? "" : "the base tilted"));
  document.set("tilt", JsonValue::object()).set("rms_rad", JsonValue::number(values.tiltRms));
  document.set("velocity", JsonValue::object()).set("rms_error_mps", JsonValue::number(values.velocityError));
  document.set("base_height", JsonValue::object()).set("mean_m", JsonValue::number(0.8));
  document.set("solve_time_ms", JsonValue::object()).set("total", JsonValue::object()).set("p99", JsonValue::number(values.p99));
  document.set("quaternion_norm", JsonValue::object()).set("max_deviation", JsonValue());
  return document;
}

TEST(MetricBands, ARunIsWithinTheBandsOfItself) {
  const JsonValue baseline = metricsDocument({});
  EXPECT_TRUE(compareClosedLoopMetrics(baseline, baseline, BandOptions()).empty());
}

TEST(MetricBands, EachBandIsTheLargerOfItsRelativeAndAbsoluteParts) {
  const JsonValue baseline = metricsDocument({.tiltRms = 0.01, .velocityError = 0.5, .p99 = 10.0});
  // tilt: max(20 % of 0.01, 0.005) = 0.005; velocity: max(15 % of 0.5, 0.02) = 0.075; p99: +10 %.
  EXPECT_TRUE(
      compareClosedLoopMetrics(metricsDocument({.tiltRms = 0.0149, .velocityError = 0.574, .p99 = 10.9}), baseline, BandOptions()).empty());
  const std::vector<std::string> outside =
      compareClosedLoopMetrics(metricsDocument({.tiltRms = 0.0151, .velocityError = 0.576, .p99 = 11.1}), baseline, BandOptions());
  ASSERT_EQ(outside.size(), 3u);
  EXPECT_EQ(outside[0].rfind("tilt.rms_rad", 0), 0u) << outside[0];
  EXPECT_EQ(outside[1].rfind("velocity.rms_error_mps", 0), 0u) << outside[1];
  EXPECT_EQ(outside[2].rfind("solve_time_ms.total.p99", 0), 0u) << outside[2];

  BandOptions withoutTiming;
  withoutTiming.compareSolveTime = false;
  EXPECT_EQ(
      compareClosedLoopMetrics(metricsDocument({.tiltRms = 0.0151, .velocityError = 0.576, .p99 = 100.0}), baseline, withoutTiming).size(),
      2u);
}

TEST(MetricBands, SurvivalMustBeEqualOrBetter) {
  const JsonValue survived = metricsDocument({});
  const JsonValue fell = metricsDocument({.survived = false, .tiltRms = 0.9, .velocityError = 2.0});
  const std::vector<std::string> worse = compareClosedLoopMetrics(fell, survived, BandOptions());
  ASSERT_EQ(worse.size(), 1u);
  EXPECT_EQ(worse[0].rfind("survival", 0), 0u) << worse[0];
  EXPECT_TRUE(compareClosedLoopMetrics(survived, fell, BandOptions()).empty()) << "better than a baseline that fell";
  EXPECT_TRUE(compareClosedLoopMetrics(fell, fell, BandOptions()).empty());
}

TEST(MetricBands, TheQuaternionNormMustStayBelowItsBound) {
  const JsonValue baseline = metricsDocument({});
  JsonValue candidate = baseline;
  candidate.set("quaternion_norm", JsonValue::object()).set("max_deviation", JsonValue::number(1.0e-8));
  EXPECT_EQ(compareClosedLoopMetrics(candidate, baseline, BandOptions()).size(), 1u);
  candidate.set("quaternion_norm", JsonValue::object()).set("max_deviation", JsonValue::number(1.0e-12));
  EXPECT_TRUE(compareClosedLoopMetrics(candidate, baseline, BandOptions()).empty());
}

TEST(ClosedLoopScenarios, TheCommandedHeadingPeakIsTheLargestTurnTheCommandsIntegrateTo) {
  for (const ClosedLoopScenario& scenario : closedLoopScenarios()) {
    // The running integral of the yaw rate, on a fine grid, peaks where the segments' sums do.
    double heading = 0.0;
    double peak = 0.0;
    const double step = 1.0e-3;
    for (double t = 0.5 * step; t < getCommandDuration(scenario); t += step) {
      heading += commandAt(scenario, t)(2) * step;
      peak = std::max(peak, heading);
    }
    EXPECT_NEAR(getCommandedHeadingPeak(scenario), peak, 1.0e-2) << scenario.name;
  }
  EXPECT_NEAR(getCommandedHeadingPeak(*findClosedLoopScenario("turn_in_place_720")), 4.0 * M_PI, 1.0e-9);
  EXPECT_EQ(getCommandedHeadingPeak(*findClosedLoopScenario("walk_0p5")), 0.0);
}

// ------------------------------------------------------------------------- the 720-degree turn exception

/** A lockstep time series at 50 Hz: heading psi(t), its rate as the yaw rate, the reference yaw rate `referenceRate`. */
GoldenFile turningSeries(double duration, double rate, double referenceRate, double initialHeading = 0.0) {
  const Eigen::Index rows = static_cast<Eigen::Index>(duration * 50.0);
  GoldenFile series;
  golden_matrix_t time(rows, 1);
  golden_matrix_t quaternions(rows, 4);
  golden_matrix_t velocities = golden_matrix_t::Zero(rows, 3);
  golden_matrix_t references = golden_matrix_t::Zero(rows, 3);
  for (Eigen::Index row = 0; row < rows; ++row) {
    const double t = static_cast<double>(row) / 50.0;
    const double heading = initialHeading + rate * t;
    time(row, /*col=*/0) = t;
    quaternions.row(row) << 0.0, 0.0, std::sin(0.5 * heading), std::cos(0.5 * heading);
    velocities(row, /*col=*/2) = rate;
    references(row, /*col=*/2) = referenceRate;
  }
  series.entries = {{time_series::kTime, time},
                    {time_series::kBaseQuaternion, quaternions},
                    {time_series::kBaseVelocity, velocities},
                    {time_series::kReferenceVelocity, references}};
  return series;
}

/**
 * `series` with the per-solve rotation gaps of a run solving at 100 Hz over `duration`: `baseGap` at every solve but
 * the (time, gap) of `spikes`, each at the solve nearest its time.
 */
GoldenFile withSolveRotationGaps(GoldenFile series,
                                 double duration,
                                 const std::vector<std::pair<double, double>>& spikes,
                                 double baseGap = 0.01) {
  const Eigen::Index rows = static_cast<Eigen::Index>(duration * 100.0);
  golden_matrix_t gaps(rows, 2);
  for (Eigen::Index row = 0; row < rows; ++row) {
    gaps(row, /*col=*/0) = static_cast<double>(row) / 100.0;
    gaps(row, /*col=*/1) = baseGap;
  }
  for (const std::pair<double, double>& spike : spikes) {
    gaps(static_cast<Eigen::Index>(std::lround(spike.first * 100.0)), /*col=*/1) = spike.second;
  }
  series.entries.push_back({time_series::kSolveRotationGap, gaps});
  return series;
}

/** A metrics document of a turning run, on Euler coordinates (no quaternion norm). */
JsonValue turningDocument(double yawRateError, double headingPeak, double rotationGap) {
  JsonValue document = metricsDocument({});
  document.set("yaw_rate", JsonValue::object()).set("rms_error_radps", JsonValue::number(yawRateError));
  JsonValue& heading = document.set("heading", JsonValue::object());
  heading.set("cumulative_max_rad", JsonValue::number(headingPeak));
  heading.set("cumulative_min_rad", JsonValue::number(0.0));
  document.set("initial_state_gap", JsonValue::object()).set("max_rotation_rad", JsonValue::number(rotationGap));
  return document;
}

/** The same, of a run on a quaternion base orientation (its quaternion norm is recorded). */
JsonValue quaternionTurningDocument(double yawRateError, double headingPeak, double rotationGap) {
  JsonValue document = turningDocument(yawRateError, headingPeak, rotationGap);
  document.set("quaternion_norm", JsonValue::object()).set("max_deviation", JsonValue::number(1.0e-12));
  return document;
}

TEST(HeadingCrossings, AreWhereTheUnwrappedHeadingCrossesAnOddMultipleOfPi) {
  // 0.5 rad/s from 0 for 30 s: through pi at 2 pi s, 3 pi at 6 pi s (and not 5 pi, reached at 31.4 s).
  const absl::StatusOr<std::vector<double>> crossings = eulerYawCrossingTimes(turningSeries(/*duration=*/30.0, /*rate=*/0.5,
                                                                                            /*referenceRate=*/0.5));
  ASSERT_TRUE(crossings.ok()) << crossings.status();
  ASSERT_EQ(crossings->size(), 2u);
  EXPECT_NEAR((*crossings)[0], 2.0 * M_PI, 0.03);
  EXPECT_NEAR((*crossings)[1], 6.0 * M_PI, 0.03);
  // A turn that stays within (-pi, pi) crosses nothing; a malformed series is refused.
  EXPECT_TRUE(eulerYawCrossingTimes(turningSeries(/*duration=*/5.0, /*rate=*/0.5, /*referenceRate=*/0.5))->empty());
  GoldenFile malformed = turningSeries(/*duration=*/1.0, /*rate=*/0.5, /*referenceRate=*/0.5);
  malformed.entries.pop_back();
  EXPECT_EQ(trackingErrorsOutsideWindows(malformed, /*windowCenters=*/{}, /*halfWidth=*/0.5).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(HeadingCrossings, TheWindowedErrorsLeaveOutTheSamplesNearACrossing) {
  // A yaw-rate error of 1 rad/s within 0.5 s of t = 3 s, none elsewhere.
  GoldenFile series = turningSeries(/*duration=*/10.0, /*rate=*/0.5, /*referenceRate=*/0.5);
  golden_matrix_t& velocities = series.entries[2].value;
  for (Eigen::Index row = 0; row < velocities.rows(); ++row) {
    if (std::abs(static_cast<double>(row) / 50.0 - 3.0) <= 0.5) velocities(row, /*col=*/2) += 1.0;
  }
  const absl::StatusOr<WindowedTrackingErrors> all = trackingErrorsOutsideWindows(series, /*windowCenters=*/{}, /*halfWidth=*/0.5);
  const absl::StatusOr<WindowedTrackingErrors> outside = trackingErrorsOutsideWindows(series, /*windowCenters=*/{3.0}, /*halfWidth=*/0.5);
  ASSERT_TRUE(all.ok() && outside.ok());
  EXPECT_GT(all->yawRateRms, 0.3);
  EXPECT_EQ(outside->yawRateRms, 0.0);
  EXPECT_EQ(outside->samples, all->samples - 51);
}

TEST(HeadingCrossings, ARunThatTracksTheTurnBetterAcrossTheCutPasses) {
  // The baseline crosses pi (an Euler run, disturbed there: a 2 pi gap and a large yaw-rate error); the candidate turns
  // as far, tracks better, and has no gap spike. The documents' yaw-rate errors differ far beyond the band, but outside
  // the windows the runs agree.
  GoldenFile baselineSeries = turningSeries(/*duration=*/15.0, /*rate=*/0.5, /*referenceRate=*/0.5);
  golden_matrix_t& disturbed = baselineSeries.entries[2].value;
  for (Eigen::Index row = 0; row < disturbed.rows(); ++row) {
    if (std::abs(static_cast<double>(row) / 50.0 - 2.0 * M_PI) <= 0.5) disturbed(row, /*col=*/2) += 2.0;
  }
  const GoldenFile candidateSeries = turningSeries(/*duration=*/15.0, /*rate=*/0.5, /*referenceRate=*/0.5);
  const JsonValue baseline = turningDocument(/*yawRateError=*/0.5, /*headingPeak=*/7.5, /*rotationGap=*/2.0 * M_PI);
  const JsonValue candidate = turningDocument(/*yawRateError=*/0.01, /*headingPeak=*/7.5, /*rotationGap=*/0.02);
  ASSERT_FALSE(compareClosedLoopMetrics(candidate, baseline, BandOptions()).empty()) << "the plain bands would refuse it";
  const std::vector<std::string> violations =
      compareClosedLoopRuns(candidate, candidateSeries, baseline, &baselineSeries, BandOptions(), HeadingCrossingOptions());
  EXPECT_TRUE(violations.empty()) << (violations.empty() ? std::string() : violations.front());
}

TEST(HeadingCrossings, ARunWithAGapSpikeOrAShorterTurnFails) {
  const GoldenFile series = turningSeries(/*duration=*/15.0, /*rate=*/0.5, /*referenceRate=*/0.5);
  const JsonValue baseline = turningDocument(/*yawRateError=*/0.1, /*headingPeak=*/7.5, /*rotationGap=*/2.0 * M_PI);
  const std::vector<std::string> spike = compareClosedLoopRuns(quaternionTurningDocument(/*yawRateError=*/0.1, /*headingPeak=*/7.5,
                                                                                         /*rotationGap=*/2.0),
                                                               series, baseline, &series, BandOptions(), HeadingCrossingOptions());
  ASSERT_EQ(spike.size(), 1u);
  EXPECT_EQ(spike[0].rfind("initial_state_gap.max_rotation_rad", 0), 0u) << spike[0];
  const std::vector<std::string> shorter =
      compareClosedLoopRuns(turningDocument(/*yawRateError=*/0.1, /*headingPeak=*/6.0, /*rotationGap=*/0.02), series, baseline, &series,
                            BandOptions(), HeadingCrossingOptions());
  ASSERT_EQ(shorter.size(), 1u);
  EXPECT_EQ(shorter[0].rfind("heading.cumulative_max_rad", 0), 0u) << shorter[0];

  // Where the baseline reached the commanded 4 pi, the candidate must too.
  HeadingCrossingOptions commanded;
  commanded.commandedHeadingPeak = 4.0 * M_PI;
  const JsonValue fullTurn = turningDocument(/*yawRateError=*/0.1, /*headingPeak=*/4.0 * M_PI, /*rotationGap=*/2.0 * M_PI);
  EXPECT_EQ(compareClosedLoopRuns(turningDocument(/*yawRateError=*/0.1, /*headingPeak=*/4.0 * M_PI + 0.2, /*rotationGap=*/0.02), series,
                                  fullTurn, &series, BandOptions(), commanded)
                .size(),
            1u);
}

/**
 * The per-solve gaps of an Euler wrap at `time`: 2 pi + 1e-3 at the solve that sees the wrap, then decaying over ten
 * solves to 2 rad as one SQP iteration per solve follows the wrapped yaw, as the whole-body G1's turn of M0_main does.
 */
std::vector<std::pair<double, double>> eulerWrap(double time) {
  std::vector<std::pair<double, double>> gaps;
  for (int solve = 0; solve <= 10; ++solve) gaps.emplace_back(time + 0.01 * solve, 2.0 * M_PI + 1.0e-3 - 0.428 * solve);
  return gaps;
}

TEST(HeadingCrossings, AnEulerRunMayWrapOnlyWhereItsEulerBaselineWrapped) {
  // Both on Euler coordinates: the measured yaw wraps where the heading crosses pi (at 2 pi s), a 2 pi gap in each (a
  // main-line rerun of M0, or M1 and M2 against M1).
  const GoldenFile baselineSeries = turningSeries(/*duration=*/15.0, /*rate=*/0.5, /*referenceRate=*/0.5);
  const JsonValue wrappedBaseline = turningDocument(/*yawRateError=*/0.1, /*headingPeak=*/7.5, /*rotationGap=*/2.0 * M_PI);
  const double wrap = 2.0 * M_PI + 1.0e-3;
  const GoldenFile wrappingSeries = withSolveRotationGaps(baselineSeries, /*duration=*/15.0, eulerWrap(/*time=*/2.0 * M_PI));
  const JsonValue wrappedEulerRun = turningDocument(/*yawRateError=*/0.1, /*headingPeak=*/7.5, /*rotationGap=*/wrap);
  const std::vector<std::string> euler =
      compareClosedLoopRuns(wrappedEulerRun, wrappingSeries, wrappedBaseline, &baselineSeries, BandOptions(), HeadingCrossingOptions());
  EXPECT_TRUE(euler.empty()) << (euler.empty() ? std::string() : euler.front());

  // The quaternion run of the same gap spikes; so does an Euler run whose baseline did not.
  const std::vector<std::string> quaternion =
      compareClosedLoopRuns(quaternionTurningDocument(/*yawRateError=*/0.1, /*headingPeak=*/7.5, /*rotationGap=*/wrap), wrappingSeries,
                            wrappedBaseline, &baselineSeries, BandOptions(), HeadingCrossingOptions());
  ASSERT_EQ(quaternion.size(), 1u);
  EXPECT_EQ(quaternion[0].rfind("initial_state_gap.max_rotation_rad", 0), 0u) << quaternion[0];
  const JsonValue smoothBaseline = turningDocument(/*yawRateError=*/0.1, /*headingPeak=*/7.5, /*rotationGap=*/0.02);
  const std::vector<std::string> newSpike =
      compareClosedLoopRuns(wrappedEulerRun, wrappingSeries, smoothBaseline, &baselineSeries, BandOptions(), HeadingCrossingOptions());
  ASSERT_EQ(newSpike.size(), 1u);
  EXPECT_EQ(newSpike[0].rfind("initial_state_gap.max_rotation_rad", 0), 0u) << newSpike[0];
}

TEST(HeadingCrossings, AnEulerRunMaySpikeOnlyAtItsOwnWrapsAndByNoMoreThanAWrap) {
  const GoldenFile baselineSeries = turningSeries(/*duration=*/15.0, /*rate=*/0.5, /*referenceRate=*/0.5);
  const JsonValue wrappedBaseline = turningDocument(/*yawRateError=*/0.1, /*headingPeak=*/7.5, /*rotationGap=*/2.0 * M_PI);
  const double crossing = 2.0 * M_PI;
  const double wrap = 2.0 * M_PI + 1.0e-3;
  // The violations of an Euler candidate turning as `candidateSeries` with the per-solve gaps `spikes`, its document's
  // maximum the largest of them.
  const std::function<std::vector<std::string>(const GoldenFile&, const std::vector<std::pair<double, double>>&)> violationsOf =
      [&](const GoldenFile& candidateSeries, const std::vector<std::pair<double, double>>& spikes) {
        double largest = 0.01;
        for (const std::pair<double, double>& spike : spikes) largest = std::max(largest, spike.second);
        return compareClosedLoopRuns(turningDocument(/*yawRateError=*/0.1, /*headingPeak=*/7.5, /*rotationGap=*/largest),
                                     withSolveRotationGaps(candidateSeries, /*duration=*/15.0, spikes), wrappedBaseline, &baselineSeries,
                                     BandOptions(), HeadingCrossingOptions());
      };
  const std::function<void(const std::vector<std::string>&, absl::string_view)> expectOneGapViolation =
      [](const std::vector<std::string>& violations, absl::string_view words) {
        ASSERT_EQ(violations.size(), 1u) << (violations.empty() ? std::string() : violations.front());
        EXPECT_EQ(violations[0].rfind("initial_state_gap.max_rotation_rad", 0), 0u) << violations[0];
        EXPECT_NE(violations[0].find(std::string(words)), std::string::npos) << violations[0];
      };

  // A wrap 2 s away from the crossing, or a second spike away from it after the wrap at the crossing, is a spike no
  // wrap explains, however the document's maximum compares.
  expectOneGapViolation(violationsOf(baselineSeries, eulerWrap(crossing + 2.0)), "outside the");
  std::vector<std::pair<double, double>> wrapAndMore = eulerWrap(crossing);
  wrapAndMore.emplace_back(/*time=*/11.0, /*gap=*/2.0);
  expectOneGapViolation(violationsOf(baselineSeries, wrapAndMore), "outside the");
  // At the crossing, a gap of 2 pi + pi / 2 or more is more than a wrap of a gap below the threshold.
  expectOneGapViolation(violationsOf(baselineSeries, {{crossing, 2.0 * M_PI + 2.0}}), "more than a wrap");

  // The wraps are the candidate's own: a run whose heading starts 0.5 rad further crosses pi 1 s earlier and may wrap
  // there, not where the baseline crossed.
  const GoldenFile earlierTurn = turningSeries(/*duration=*/15.0, /*rate=*/0.5, /*referenceRate=*/0.5, /*initialHeading=*/0.5);
  EXPECT_TRUE(violationsOf(earlierTurn, eulerWrap(crossing - 1.0)).empty());
  expectOneGapViolation(violationsOf(earlierTurn, eulerWrap(crossing)), "outside the");

  // Without the per-solve gaps the spike cannot be located; with gaps of another run it cannot be trusted.
  const JsonValue wrapped = turningDocument(/*yawRateError=*/0.1, /*headingPeak=*/7.5, /*rotationGap=*/wrap);
  expectOneGapViolation(
      compareClosedLoopRuns(wrapped, baselineSeries, wrappedBaseline, &baselineSeries, BandOptions(), HeadingCrossingOptions()),
      "cannot locate");
  expectOneGapViolation(
      compareClosedLoopRuns(wrapped, withSolveRotationGaps(baselineSeries, /*duration=*/15.0, /*spikes=*/{{crossing, 6.0}}),
                            wrappedBaseline, &baselineSeries, BandOptions(), HeadingCrossingOptions()),
      "not of the same run");
}

TEST(HeadingCrossings, ABaselineThatReachesPiWithoutItsTimeSeriesIsReported) {
  const GoldenFile series = turningSeries(/*duration=*/15.0, /*rate=*/0.5, /*referenceRate=*/0.5);
  const JsonValue baseline = turningDocument(/*yawRateError=*/0.1, /*headingPeak=*/7.5, /*rotationGap=*/2.0 * M_PI);
  const std::vector<std::string> violations =
      compareClosedLoopRuns(baseline, series, baseline, /*baselineSeries=*/nullptr, BandOptions(), HeadingCrossingOptions());
  ASSERT_EQ(violations.size(), 1u);
  EXPECT_NE(violations[0].find("time series is not archived"), std::string::npos) << violations[0];
  // A baseline that stays short of pi needs none, and a baseline series without a crossing is the plain bands.
  const JsonValue small = turningDocument(/*yawRateError=*/0.1, /*headingPeak=*/0.3, /*rotationGap=*/0.01);
  EXPECT_TRUE(compareClosedLoopRuns(small, series, small, /*baselineSeries=*/nullptr, BandOptions(), HeadingCrossingOptions()).empty());
  const GoldenFile shortTurn = turningSeries(/*duration=*/2.0, /*rate=*/0.5, /*referenceRate=*/0.5);
  EXPECT_TRUE(compareClosedLoopRuns(small, shortTurn, small, &shortTurn, BandOptions(), HeadingCrossingOptions()).empty());
}

}  // namespace
}  // namespace ocs2::humanoid::validation
