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
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "humanoid_mpc_validation/closed_loop/ClosedLoopMetrics.h"
#include "humanoid_mpc_validation/closed_loop/ClosedLoopMetricsSchema.h"
#include "humanoid_mpc_validation/closed_loop/MetricBands.h"
#include "humanoid_mpc_validation/io/JsonValue.h"

/*
 * The closed-loop metrics: each definition on synthetic samples whose answer is known (perfect tracking is no error, a
 * constant height has no spread, a turn at a constant rate unwraps through +/-pi to rate * time, a planted foot does not
 * slip), and the schema test - every document the metrics write validates, the validator catches what is missing,
 * mistyped or unknown. Then the comparison rules (MetricBands.h): the band directions, improvements reported and
 * passed, and the offline comparison of two labels archived from synthetic runs, which reaches the verdict of the runs
 * as held in memory, run by run, names the runs without a baseline and refuses what it cannot compare.
 */

namespace ocs2::humanoid::validation {
namespace {

constexpr double kTimeStep = 0.01;

Eigen::Vector4d yawQuaternion(double yaw) {
  return Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())).coeffs();
}

/** A robot at height 0.8 m tracking (vx, 0, yaw rate) perfectly with its feet planted; time step kTimeStep. */
ControlCycleSample trackingSample(size_t k, double forwardVelocity, double yawRate) {
  ControlCycleSample sample;
  sample.time = static_cast<double>(k) * kTimeStep;
  const double yaw = yawRate * sample.time;
  sample.basePosition = Eigen::Vector3d(0.0, 0.0, 0.8);
  sample.baseQuaternion = yawQuaternion(yaw);
  sample.baseLinearVelocityWorld = Eigen::Vector3d(forwardVelocity * std::cos(yaw), forwardVelocity * std::sin(yaw), 0.0);
  sample.baseAngularVelocityLocal = Eigen::Vector3d(0.0, 0.0, yawRate);
  sample.referenceVelocity = Eigen::Vector3d(forwardVelocity, 0.0, yawRate);
  sample.referenceBaseHeight = 0.8;
  sample.contactPositions = {Eigen::Vector3d(0.0, 0.1, 0.0), Eigen::Vector3d(0.0, -0.1, 0.0)};
  sample.contactFlags = {true, true};
  sample.jointTorques = Eigen::VectorXd::Constant(/*size=*/3, 2.0);
  return sample;
}

double numberAt(const JsonValue& document, const std::string& path) {
  const JsonValue* absl_nullable value = document.findPath(path);
  EXPECT_NE(value, nullptr) << path;
  if (value == nullptr || !value->isNumber()) return std::nan("");
  return value->asNumber();
}

JsonValue reportOf(const ClosedLoopMetrics& metrics) {
  ClosedLoopRunInfo info;
  info.label = "test";
  info.robot = "robot";
  info.formulation = "centroidal";
  info.scenario = "scenario";
  info.settings.set("anything", JsonValue::number(1.0));
  return metrics.report(info);
}

TEST(ClosedLoopMetrics, PerfectTrackingHasNoErrorAndAConstantHeightNoSpread) {
  ClosedLoopMetrics metrics;
  for (size_t k = 0; k < 200; ++k) metrics.addControlCycle(trackingSample(k, /*forwardVelocity=*/0.5, /*yawRate=*/0.3));
  const JsonValue report = reportOf(metrics);
  EXPECT_NEAR(numberAt(report, "velocity.rms_error_mps"), 0.0, 1.0e-12);
  EXPECT_NEAR(numberAt(report, "yaw_rate.rms_error_radps"), 0.0, 1.0e-12);
  EXPECT_NEAR(numberAt(report, "base_height.mean_m"), 0.8, 1.0e-12);
  EXPECT_NEAR(numberAt(report, "base_height.std_m"), 0.0, 1.0e-6);
  EXPECT_NEAR(numberAt(report, "base_height.rms_error_m"), 0.0, 1.0e-12);
  EXPECT_NEAR(numberAt(report, "tilt.max_rad"), 0.0, 1.0e-12) << "a level base has no tilt, whatever its heading";
  EXPECT_NEAR(numberAt(report, "joint_torque.rms_nm"), 2.0, 1.0e-12);
  EXPECT_TRUE(report.findPath("survival.survived")->asBool());
  EXPECT_TRUE(report.findPath("quaternion_norm.max_deviation")->isNull()) << "no solves, nothing measured";
}

TEST(ClosedLoopMetrics, TheErrorsAreTheRmsOfTheDifferences) {
  ClosedLoopMetrics metrics;
  for (size_t k = 0; k < 100; ++k) {
    ControlCycleSample sample = trackingSample(k, /*forwardVelocity=*/0.0, /*yawRate=*/0.0);
    sample.referenceVelocity = Eigen::Vector3d(0.3, 0.4, (k % 2 == 0) ? 0.2 : -0.2);  // the robot stands still
    sample.basePosition.z() = (k % 2 == 0) ? 0.81 : 0.79;
    metrics.addControlCycle(sample);
  }
  const JsonValue report = reportOf(metrics);
  EXPECT_NEAR(numberAt(report, "velocity.rms_error_mps"), 0.5, 1.0e-12);
  EXPECT_NEAR(numberAt(report, "yaw_rate.rms_error_radps"), 0.2, 1.0e-12);
  EXPECT_NEAR(numberAt(report, "base_height.std_m"), 0.01, 1.0e-9);
  EXPECT_NEAR(numberAt(report, "base_height.rms_error_m"), 0.01, 1.0e-12);
}

TEST(ClosedLoopMetrics, TheVelocityErrorIsTakenInTheHeadingFrame) {
  // Walking forward at 0.5 m/s while facing 2 rad: in the world the velocity points at 2 rad, in the heading frame it is
  // straight ahead, as the command is.
  ClosedLoopMetrics metrics;
  ControlCycleSample sample = trackingSample(/*k=*/0, /*forwardVelocity=*/0.5, /*yawRate=*/0.0);
  sample.baseQuaternion = yawQuaternion(2.0);
  sample.baseLinearVelocityWorld = Eigen::Vector3d(0.5 * std::cos(2.0), 0.5 * std::sin(2.0), 0.0);
  metrics.addControlCycle(sample);
  EXPECT_NEAR(numberAt(reportOf(metrics), "velocity.rms_error_mps"), 0.0, 1.0e-12);
}

TEST(ClosedLoopMetrics, TheHeadingUnwrapsThroughPlusMinusPi) {
  // 1 rad/s for 12.6 s: two full turns, through the +/-pi seam four times.
  ClosedLoopMetrics metrics;
  const size_t steps = 1260;
  for (size_t k = 0; k <= steps; ++k) metrics.addControlCycle(trackingSample(k, /*forwardVelocity=*/0.0, /*yawRate=*/1.0));
  const JsonValue report = reportOf(metrics);
  EXPECT_NEAR(numberAt(report, "heading.cumulative_final_rad"), 12.6, 1.0e-9);
  EXPECT_NEAR(numberAt(report, "heading.cumulative_max_rad"), 12.6, 1.0e-9);
  EXPECT_NEAR(numberAt(report, "heading.cumulative_min_rad"), 0.0, 1.0e-12);
}

TEST(ClosedLoopMetrics, TheTiltIsTheAngleOfTheBaseFromUpright) {
  ClosedLoopMetrics metrics;
  ControlCycleSample sample = trackingSample(/*k=*/0, /*forwardVelocity=*/0.0, /*yawRate=*/0.0);
  // Rolled 0.1 rad after a yaw of 2.5 rad: the tilt does not depend on the heading.
  sample.baseQuaternion = (Eigen::Quaterniond(Eigen::AngleAxisd(2.5, Eigen::Vector3d::UnitZ())) *
                           Eigen::Quaterniond(Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX())))
                              .coeffs();
  metrics.addControlCycle(sample);
  EXPECT_NEAR(numberAt(reportOf(metrics), "tilt.max_rad"), 0.1, 1.0e-12);
}

TEST(ClosedLoopMetrics, APlantedFootDoesNotSlipAndASlidingOneDoes) {
  ClosedLoopMetrics metrics;
  for (size_t k = 0; k < 50; ++k) {
    ControlCycleSample sample = trackingSample(k, /*forwardVelocity=*/0.0, /*yawRate=*/0.0);
    sample.contactPositions[1].x() = 0.001 * static_cast<double>(k);  // the right foot slides 1 mm per cycle
    metrics.addControlCycle(sample);
  }
  // The right foot lifts off and lands again 0.3 m further: a new stance, not slip.
  ControlCycleSample swing = trackingSample(/*k=*/50, /*forwardVelocity=*/0.0, /*yawRate=*/0.0);
  swing.contactFlags = {true, false};
  metrics.addControlCycle(swing);
  for (size_t k = 51; k < 60; ++k) {
    ControlCycleSample sample = trackingSample(k, /*forwardVelocity=*/0.0, /*yawRate=*/0.0);
    sample.contactPositions[1].x() = 0.3;
    metrics.addControlCycle(sample);
  }
  const JsonValue report = reportOf(metrics);
  EXPECT_NEAR(numberAt(report, "stance_foot_slip.max_m"), 0.049, 1.0e-12);
  EXPECT_EQ(numberAt(report, "stance_foot_slip.stance_phases"), 3.0) << "the left foot's one, and two of the right foot";
  EXPECT_NEAR(numberAt(report, "stance_foot_slip.rms_m"), std::sqrt(0.049 * 0.049 / 3.0), 1.0e-12);
}

TEST(ClosedLoopMetrics, SolveStatisticsAreNearestRankPercentiles) {
  ClosedLoopMetrics metrics;
  for (int k = 1; k <= 100; ++k) {
    SolveSample solve;
    solve.time = 0.01 * k;
    solve.wallTimeMs = static_cast<double>(k);
    solve.succeeded = k != 7;
    solve.initialStateRotationGap = k == 40 ? 6.28 : 0.01;
    solve.initialStateGapNorm = 0.1;
    metrics.addSolve(solve);
  }
  const JsonValue report = reportOf(metrics);
  EXPECT_EQ(numberAt(report, "solve_time_ms.total.p50"), 50.0);
  EXPECT_EQ(numberAt(report, "solve_time_ms.total.p99"), 99.0);
  EXPECT_EQ(numberAt(report, "solve_time_ms.total.max"), 100.0);
  EXPECT_NEAR(numberAt(report, "solve_time_ms.total.mean"), 50.5, 1.0e-12);
  EXPECT_EQ(numberAt(report, "failures.failed_solves"), 1.0);
  EXPECT_EQ(numberAt(report, "initial_state_gap.max_rotation_rad"), 6.28);
  EXPECT_NEAR(numberAt(report, "initial_state_gap.max_rotation_time_s"), 0.40, 1.0e-12) << "when the gap peaked";
  EXPECT_TRUE(summarizeTimes({}).find("p99")->isNull());
}

TEST(ClosedLoopMetrics, AFallIsReportedWithItsTimeAndReason) {
  ClosedLoopMetrics metrics;
  metrics.setFall(/*time=*/3.5, "the base tilted 1.2 rad");
  const JsonValue report = reportOf(metrics);
  EXPECT_FALSE(report.findPath("survival.survived")->asBool());
  EXPECT_EQ(numberAt(report, "survival.fall_time_s"), 3.5);
  EXPECT_EQ(report.findPath("survival.fall_reason")->asString(), "the base tilted 1.2 rad");
  EXPECT_TRUE(report.findPath("tilt.rms_rad")->isNull()) << "nothing was sampled";
  EXPECT_TRUE(validateClosedLoopMetrics(report).ok()) << validateClosedLoopMetrics(report);
}

TEST(ClosedLoopMetrics, EveryWrapAngleIsInMinusPiToPi) {
  for (double angle = -20.0; angle <= 20.0; angle += 0.37) {
    const double wrapped = wrapAngle(angle);
    EXPECT_GT(wrapped, -M_PI);
    EXPECT_LE(wrapped, M_PI);
    EXPECT_NEAR(std::remainder(wrapped - angle, 2.0 * M_PI), 0.0, 1.0e-12);
  }
}

// ----------------------------------------------------------------------------------------------------- the schema

TEST(ClosedLoopMetricsSchema, EveryDocumentTheMetricsWriteValidates) {
  ClosedLoopMetrics metrics;
  for (size_t k = 0; k < 20; ++k) metrics.addControlCycle(trackingSample(k, /*forwardVelocity=*/0.3, /*yawRate=*/0.1));
  SolveSample solve;
  solve.quaternionNormDeviation = 1.0e-12;
  metrics.addSolve(solve);
  const JsonValue report = reportOf(metrics);
  EXPECT_TRUE(validateClosedLoopMetrics(report).ok()) << validateClosedLoopMetrics(report);
  EXPECT_TRUE(validateClosedLoopMetrics(reportOf(ClosedLoopMetrics())).ok()) << "the document of a run without samples";

  // And the document reads back from its text to the same, still valid.
  const absl::StatusOr<JsonValue> parsed = JsonValue::parse(report.serialize());
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  EXPECT_EQ(*parsed, report);
  EXPECT_TRUE(validateClosedLoopMetrics(*parsed).ok());
}

TEST(ClosedLoopMetricsSchema, MissingMistypedAndUnknownFieldsAreNamed) {
  const JsonValue report = reportOf(ClosedLoopMetrics());

  // A copy without base_height.mean_m.
  JsonValue missing = JsonValue::object();
  for (size_t i = 0; i < report.size(); ++i) {
    if (report.keyAt(i) != "base_height") {
      missing.set(report.keyAt(i), report.valueAt(i));
      continue;
    }
    JsonValue height = JsonValue::object();
    height.set("std_m", JsonValue());
    height.set("rms_error_m", JsonValue());
    missing.set("base_height", height);
  }
  const absl::Status missingStatus = validateClosedLoopMetrics(missing);
  EXPECT_TRUE(absl::StrContains(missingStatus.message(), "base_height.mean_m is missing")) << missingStatus;

  JsonValue mistyped = report;
  mistyped.set("non_finite_values", JsonValue::string("zero"));
  const absl::Status mistypedStatus = validateClosedLoopMetrics(mistyped);
  EXPECT_TRUE(absl::StrContains(mistypedStatus.message(), "non_finite_values has the wrong type")) << mistypedStatus;

  JsonValue unknown = report;
  unknown.set("undocumented", JsonValue::number(1.0));
  const absl::Status unknownStatus = validateClosedLoopMetrics(unknown);
  EXPECT_TRUE(absl::StrContains(unknownStatus.message(), "undocumented is not in the schema")) << unknownStatus;

  JsonValue wrongVersion = report;
  wrongVersion.set("schema", JsonValue::string("humanoid_mpc_validation.closed_loop_metrics.v0"));
  EXPECT_FALSE(validateClosedLoopMetrics(wrongVersion).ok());
}

// ----------------------------------------------------------------------------------------------- the band rules

/** A surviving run's metrics document with the numbers at their dotted paths (each "<group>.<field>"). */
JsonValue documentWith(const std::vector<std::pair<std::string, double>>& numbers) {
  JsonValue document = JsonValue::object();
  document.set("survival", JsonValue::object()).set("survived", JsonValue::boolean(/*value=*/true));
  for (const std::pair<std::string, double>& number : numbers) {
    const size_t dot = number.first.find('.');
    const std::string group = number.first.substr(0, dot);
    const JsonValue* absl_nullable existing = document.find(group);
    JsonValue members = existing != nullptr ? *existing : JsonValue::object();
    members.set(number.first.substr(dot + 1), JsonValue::number(number.second));
    document.set(group, members);
  }
  return document;
}

/** Every banded metric at `value`, but `path` at `pathValue`. */
JsonValue bandedDocument(double value, const std::string& path, double pathValue) {
  std::vector<std::pair<std::string, double>> numbers;
  for (const MetricBand& band : closedLoopMetricBands()) numbers.emplace_back(band.path, band.path == path ? pathValue : value);
  return documentWith(numbers);
}

double allowedDifference(const MetricBand& band, double reference) {
  return std::max(band.relative * std::abs(reference), band.absolute);
}

TEST(MetricBands, TheErrorsMayOnlyImproveAndTheHeightMeanMovesNeitherWay) {
  const std::vector<std::pair<std::string, BandDirection>> expected = {
      {"base_height.mean_m", BandDirection::kTwoSided},    {"base_height.rms_error_m", BandDirection::kNoWorse},
      {"tilt.rms_rad", BandDirection::kNoWorse},           {"tilt.max_rad", BandDirection::kNoWorse},
      {"velocity.rms_error_mps", BandDirection::kNoWorse}, {"yaw_rate.rms_error_radps", BandDirection::kNoWorse},
      {"stance_foot_slip.max_m", BandDirection::kNoWorse},
  };
  ASSERT_EQ(closedLoopMetricBands().size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(closedLoopMetricBands()[i].path, expected[i].first);
    EXPECT_EQ(closedLoopMetricBands()[i].direction, expected[i].second) << expected[i].first;
  }
}

TEST(MetricBands, AnImprovementPassesAndIsReportedAndARegressionFails) {
  const double reference = 0.1;
  for (const MetricBand& band : closedLoopMetricBands()) {
    if (band.direction != BandDirection::kNoWorse) continue;
    const double allowed = allowedDifference(band, reference);
    const JsonValue baseline = bandedDocument(reference, band.path, reference);

    const ClosedLoopComparison better =
        compareClosedLoopMetrics(bandedDocument(reference, band.path, reference - 2.0 * allowed), baseline, BandOptions());
    EXPECT_TRUE(better.violations.empty()) << band.path << ": " << describeClosedLoopComparison(better);
    ASSERT_EQ(better.improvements.size(), 1u) << band.path;
    EXPECT_EQ(better.improvements[0].rfind(band.path, 0), 0u) << better.improvements[0];

    const ClosedLoopComparison worse =
        compareClosedLoopMetrics(bandedDocument(reference, band.path, reference + 2.0 * allowed), baseline, BandOptions());
    EXPECT_TRUE(worse.improvements.empty()) << band.path;
    ASSERT_EQ(worse.violations.size(), 1u) << band.path;
    EXPECT_EQ(worse.violations[0].rfind(band.path, 0), 0u) << worse.violations[0];

    // Within the band either way: nothing to say.
    for (const double value : {reference - 0.5 * allowed, reference + 0.5 * allowed}) {
      const ClosedLoopComparison within = compareClosedLoopMetrics(bandedDocument(reference, band.path, value), baseline, BandOptions());
      EXPECT_TRUE(within.violations.empty() && within.improvements.empty()) << band.path << ": " << describeClosedLoopComparison(within);
    }
  }
}

TEST(MetricBands, TheHeightMeanIsHeldWithinItsBandBothWays) {
  const JsonValue baseline = bandedDocument(/*value=*/0.1, "base_height.mean_m", /*pathValue=*/0.8);
  for (const double mean : {0.79, 0.81}) {
    const ClosedLoopComparison moved =
        compareClosedLoopMetrics(bandedDocument(/*value=*/0.1, "base_height.mean_m", mean), baseline, BandOptions());
    ASSERT_EQ(moved.violations.size(), 1u) << mean;
    EXPECT_EQ(moved.violations[0].rfind("base_height.mean_m", 0), 0u) << moved.violations[0];
    EXPECT_TRUE(moved.improvements.empty()) << "a height is no error: neither way is better";
  }
  EXPECT_TRUE(compareClosedLoopMetrics(bandedDocument(/*value=*/0.1, "base_height.mean_m", /*pathValue=*/0.796), baseline, BandOptions())
                  .violations.empty());
}

TEST(MetricBands, TheDescriptionListsBothKinds) {
  const ClosedLoopComparison comparison{.violations = {"tilt.max_rad: worse"}, .improvements = {"tilt.rms_rad: better", "slip: better"}};
  EXPECT_EQ(describeClosedLoopComparison(comparison),
            "violations (1):\n  tilt.max_rad: worse\nimprovements (2):\n  tilt.rms_rad: better\n  slip: better");
  EXPECT_EQ(describeClosedLoopComparison(ClosedLoopComparison()), "violations (0):\nimprovements (0):");
}

// ------------------------------------------------------------------------------- the offline comparison of two labels

/** A surviving run's metrics document, as ClosedLoopMetrics writes it: a walk at `forwardVelocity`, one solve. */
JsonValue walkDocument(double forwardVelocity) {
  ClosedLoopMetrics metrics;
  for (size_t k = 0; k < 100; ++k) metrics.addControlCycle(trackingSample(k, forwardVelocity, /*yawRate=*/0.1));
  SolveSample solve;
  solve.wallTimeMs = 5.0;
  solve.quaternionNormDeviation = 1.0e-12;
  metrics.addSolve(solve);
  return reportOf(metrics);
}

/** `document` with the number `value` at the two-level path `path` ("<group>.<field>"); members keep their order. */
JsonValue withNumberAt(const JsonValue& document, const std::string& path, double value) {
  const size_t dot = path.find('.');
  const std::string group = path.substr(0, dot);
  const JsonValue* absl_nullable existing = document.find(group);
  JsonValue members = existing != nullptr ? *existing : JsonValue::object();
  members.set(path.substr(dot + 1), JsonValue::number(value));
  JsonValue copy = document;
  copy.set(group, members);
  return copy;
}

/** One run of a label: what a run holds in memory and `make closed-loop-metrics` archives. */
struct LabelRun {
  std::string robot;
  std::string scenario;
  JsonValue document;
};

/** A fresh directory of the test's temporary directory, `name`, with `runs` archived as `make closed-loop-metrics` does. */
std::string archiveLabel(const std::string& name, const std::vector<LabelRun>& runs) {
  const std::filesystem::path directory = std::filesystem::path(::testing::TempDir()) / name;
  std::error_code error;
  std::filesystem::remove_all(directory, error);
  std::filesystem::create_directories(directory, error);
  for (const LabelRun& run : runs) {
    const absl::Status written = writeJsonFile((directory / absl::StrCat(run.robot, "_", run.scenario, ".json")).string(), run.document);
    EXPECT_TRUE(written.ok()) << written;
  }
  return directory.string();
}

/** The runs of a baseline label and of a candidate that changed two of them, ran one more and left one out. */
struct TwoLabels {
  std::vector<LabelRun> baseline;
  std::vector<LabelRun> candidate;
};

TwoLabels twoLabels() {
  const JsonValue walk = walkDocument(/*forwardVelocity=*/0.5);
  const JsonValue standing = walkDocument(/*forwardVelocity=*/0.0);
  TwoLabels labels;
  labels.baseline = {
      {.robot = "drc_atlas", .scenario = "standing", .document = withNumberAt(standing, "stance_foot_slip.max_m", /*value=*/0.05)},
      {.robot = "unitree_g1", .scenario = "standing", .document = standing},
      {.robot = "unitree_g1", .scenario = "walk_0p5", .document = walk},
      {.robot = "unitree_r1", .scenario = "standing", .document = standing}};
  // Out of order on purpose: the comparison lists the runs as robotConfigurations(), then closedLoopScenarios() do.
  labels.candidate = {{.robot = "unitree_g1", .scenario = "walk_0p5", .document = withNumberAt(walk, "tilt.max_rad", /*value=*/0.2)},
                      {.robot = "unitree_g1", .scenario = "standing", .document = standing},
                      {.robot = "engineai_sa01", .scenario = "standing", .document = standing},
                      {.robot = "drc_atlas", .scenario = "standing", .document = standing}};
  return labels;
}

/** The candidate's run `robot` `scenario` of `labels`, which it has. */
const LabelRun& candidateRun(const TwoLabels& labels, const std::string& robot, const std::string& scenario) {
  const std::vector<LabelRun>::const_iterator run =
      std::find_if(labels.candidate.begin(), labels.candidate.end(),
                   [&robot, &scenario](const LabelRun& candidate) { return candidate.robot == robot && candidate.scenario == scenario; });
  CHECK(run != labels.candidate.end()) << robot << " " << scenario;
  return *run;
}

TEST(ArchivedComparison, HoldsEveryRunToTheSameRunOfTheBaselineAsTheRunItselfDoes) {
  const TwoLabels labels = twoLabels();
  const absl::StatusOr<ArchivedComparison> comparison =
      compareArchivedClosedLoopRuns({.candidateDirectory = archiveLabel("candidate", labels.candidate),
                                     .baselineDirectory = archiveLabel("baseline", labels.baseline),
                                     .robot = "",
                                     .compareSolveTime = false});
  ASSERT_TRUE(comparison.ok()) << comparison.status();
  ASSERT_EQ(comparison->runs.size(), 3u);
  EXPECT_EQ(comparison->withoutBaseline, std::vector<std::string>({"engineai_sa01 standing"}));
  const std::vector<std::pair<std::string, std::string>> order = {
      {"drc_atlas", "standing"}, {"unitree_g1", "standing"}, {"unitree_g1", "walk_0p5"}};
  for (size_t i = 0; i < order.size(); ++i) {
    const ArchivedRunComparison& run = comparison->runs[i];
    EXPECT_EQ(run.robot, order[i].first);
    EXPECT_EQ(run.scenario, order[i].second);
    // The archived documents read back bit for bit: the verdict is the run's own, sentence for sentence.
    const std::vector<LabelRun>::const_iterator baseline =
        std::find_if(labels.baseline.begin(), labels.baseline.end(),
                     [&run](const LabelRun& entry) { return entry.robot == run.robot && entry.scenario == run.scenario; });
    ASSERT_NE(baseline, labels.baseline.end());
    const ClosedLoopComparison inRun = compareClosedLoopMetrics(candidateRun(labels, run.robot, run.scenario).document, baseline->document,
                                                                BandOptions{.compareSolveTime = false});
    EXPECT_EQ(run.comparison.violations, inRun.violations) << run.robot << " " << run.scenario;
    EXPECT_EQ(run.comparison.improvements, inRun.improvements) << run.robot << " " << run.scenario;
  }
  // The unchanged run says nothing; the tilt spike is a violation, the slip that went from 5 cm to none an improvement.
  EXPECT_TRUE(comparison->runs[1].comparison.violations.empty() && comparison->runs[1].comparison.improvements.empty())
      << describeClosedLoopComparison(comparison->runs[1].comparison);
  ASSERT_EQ(comparison->runs[2].comparison.violations.size(), 1u) << describeClosedLoopComparison(comparison->runs[2].comparison);
  EXPECT_EQ(comparison->runs[2].comparison.violations[0].rfind("tilt.max_rad: ", 0), 0u) << comparison->runs[2].comparison.violations[0];
  EXPECT_TRUE(comparison->runs[0].comparison.violations.empty()) << describeClosedLoopComparison(comparison->runs[0].comparison);
  ASSERT_EQ(comparison->runs[0].comparison.improvements.size(), 1u) << describeClosedLoopComparison(comparison->runs[0].comparison);
  EXPECT_EQ(comparison->runs[0].comparison.improvements[0].rfind("stance_foot_slip.max_m: ", 0), 0u)
      << comparison->runs[0].comparison.improvements[0];
}

TEST(ArchivedComparison, ComparesOneRobotOrTheSolveTimesOnRequest) {
  const TwoLabels labels = twoLabels();
  const std::string candidate = archiveLabel("candidate_one_robot", labels.candidate);
  const std::string baseline = archiveLabel("baseline_one_robot", labels.baseline);
  const absl::StatusOr<ArchivedComparison> oneRobot = compareArchivedClosedLoopRuns(
      {.candidateDirectory = candidate, .baselineDirectory = baseline, .robot = "unitree_g1", .compareSolveTime = false});
  ASSERT_TRUE(oneRobot.ok()) << oneRobot.status();
  ASSERT_EQ(oneRobot->runs.size(), 2u);
  EXPECT_EQ(oneRobot->runs[0].robot, "unitree_g1");
  EXPECT_EQ(oneRobot->runs[1].robot, "unitree_g1");
  EXPECT_TRUE(oneRobot->withoutBaseline.empty()) << "the run without a baseline is another robot's";

  // A candidate twice as slow passes unless the solve times are compared (labels of one machine).
  const JsonValue standing = walkDocument(/*forwardVelocity=*/0.0);
  JsonValue slowDocument = standing;
  JsonValue times = *standing.find("solve_time_ms");
  JsonValue total = *times.find("total");
  total.set("p99", JsonValue::number(2.0 * total.find("p99")->asNumber()));
  times.set("total", total);
  slowDocument.set("solve_time_ms", times);
  const std::string slow = archiveLabel("slow", {{.robot = "unitree_g1", .scenario = "standing", .document = slowDocument}});
  for (const bool compareSolveTime : {false, true}) {
    const absl::StatusOr<ArchivedComparison> timed = compareArchivedClosedLoopRuns(
        {.candidateDirectory = slow, .baselineDirectory = baseline, .robot = "", .compareSolveTime = compareSolveTime});
    ASSERT_TRUE(timed.ok()) << timed.status();
    ASSERT_EQ(timed->runs.size(), 1u);
    EXPECT_EQ(timed->runs[0].comparison.violations.size(), compareSolveTime ? 1u : 0u)
        << describeClosedLoopComparison(timed->runs[0].comparison);
  }
}

TEST(ArchivedComparison, RefusesWhatItCannotCompare) {
  const TwoLabels labels = twoLabels();
  const std::string candidate = archiveLabel("candidate_refused", labels.candidate);
  const std::string baseline = archiveLabel("baseline_refused", labels.baseline);
  const std::string missing = (std::filesystem::path(::testing::TempDir()) / "no_such_label").string();
  EXPECT_EQ(
      compareArchivedClosedLoopRuns({.candidateDirectory = missing, .baselineDirectory = baseline, .robot = "", .compareSolveTime = false})
          .status()
          .code(),
      absl::StatusCode::kNotFound);
  EXPECT_EQ(
      compareArchivedClosedLoopRuns({.candidateDirectory = candidate, .baselineDirectory = missing, .robot = "", .compareSolveTime = false})
          .status()
          .code(),
      absl::StatusCode::kNotFound);
  EXPECT_EQ(compareArchivedClosedLoopRuns(
                {.candidateDirectory = candidate, .baselineDirectory = baseline, .robot = "atlas", .compareSolveTime = false})
                .status()
                .code(),
            absl::StatusCode::kNotFound);

  // A document that does not parse is an error of its own, not a run without a baseline.
  const std::filesystem::path broken = std::filesystem::path(archiveLabel("broken", /*runs=*/{})) / "unitree_g1_standing.json";
  {
    std::ofstream out(broken);
    out << "{ not json";
  }
  const absl::StatusOr<ArchivedComparison> unreadable = compareArchivedClosedLoopRuns(
      {.candidateDirectory = broken.parent_path().string(), .baselineDirectory = baseline, .robot = "", .compareSolveTime = false});
  EXPECT_FALSE(unreadable.ok());
}

}  // namespace
}  // namespace ocs2::humanoid::validation
