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
#include <string>
#include <vector>

#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"

#include "humanoid_mpc_validation/closed_loop/ClosedLoopMetrics.h"
#include "humanoid_mpc_validation/closed_loop/ClosedLoopMetricsSchema.h"
#include "humanoid_mpc_validation/closed_loop/MetricBands.h"
#include "humanoid_mpc_validation/closed_loop/TimeSeriesLabels.h"
#include "humanoid_mpc_validation/io/GoldenIo.h"
#include "humanoid_mpc_validation/io/JsonValue.h"

/*
 * The closed-loop metrics: each definition on synthetic samples whose answer is known (perfect tracking is no error, a
 * constant height has no spread, a turn at a constant rate unwraps through +/-pi to rate * time, a planted foot does not
 * slip), and the schema test - every document the metrics write validates, the validator catches what is missing,
 * mistyped or unknown, and every document recorded under data/closed_loop/ still validates.
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

TEST(ClosedLoopMetricsSchema, EveryRecordedDocumentValidates) {
  const std::filesystem::path directory("humanoid_nmpc/humanoid_mpc_validation/data/closed_loop");
  ASSERT_TRUE(std::filesystem::is_directory(directory)) << "the recorded documents are a data dependency of this test";
  size_t documents = 0;
  for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(directory)) {
    if (entry.path().extension() != ".json") continue;
    const absl::StatusOr<JsonValue> document = readJsonFile(entry.path().string());
    ASSERT_TRUE(document.ok()) << document.status();
    EXPECT_TRUE(validateClosedLoopMetrics(*document).ok()) << entry.path() << ": " << validateClosedLoopMetrics(*document);
    ++documents;
  }
  RecordProperty("documents", static_cast<int>(documents));
}

TEST(ClosedLoopMetricsSchema, EveryRecordedRunWhoseHeadingReachesPiKeepsItsTimeSeries) {
  // The 720-degree turn exception of section 4.5 locates the crossings of the Euler yaw's cut in the baseline's time
  // series, so a baseline run that reaches +-pi must have it archived beside its document, and it must cross there.
  const std::filesystem::path directory("humanoid_nmpc/humanoid_mpc_validation/data/closed_loop");
  ASSERT_TRUE(std::filesystem::is_directory(directory));
  for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(directory)) {
    if (entry.path().extension() != ".json") continue;
    const absl::StatusOr<JsonValue> document = readJsonFile(entry.path().string());
    ASSERT_TRUE(document.ok()) << document.status();
    const JsonValue* absl_nullable peak = document->findPath("heading.cumulative_max_rad");
    const JsonValue* absl_nullable trough = document->findPath("heading.cumulative_min_rad");
    const bool reachesPi = (peak != nullptr && peak->isNumber() && peak->asNumber() >= M_PI) ||
                           (trough != nullptr && trough->isNumber() && trough->asNumber() <= -M_PI);
    if (!reachesPi) continue;
    std::filesystem::path seriesFile = entry.path();
    seriesFile.replace_filename(entry.path().stem().string() + "_timeseries.txt");
    ASSERT_TRUE(std::filesystem::exists(seriesFile)) << entry.path() << " reaches pi; archive " << seriesFile;
    const absl::StatusOr<GoldenFile> series = readGoldenFile(seriesFile.string());
    ASSERT_TRUE(series.ok()) << series.status();
    const absl::StatusOr<std::vector<double>> crossings = eulerYawCrossingTimes(*series);
    ASSERT_TRUE(crossings.ok()) << crossings.status();
    EXPECT_FALSE(crossings->empty()) << seriesFile;
  }
}

TEST(ClosedLoopMetricsSchema, EveryRecordedPerSolveGapSeriesHoldsItsDocumentsMaximum) {
  // The 720-degree turn exception trusts a series' per-solve gaps only as the document's own: its largest must be the
  // document's initial_state_gap.max_rotation_rad.
  const std::filesystem::path directory("humanoid_nmpc/humanoid_mpc_validation/data/closed_loop");
  for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(directory)) {
    const std::string name = entry.path().filename().string();
    if (!absl::EndsWith(name, "_timeseries.txt")) continue;
    const absl::StatusOr<GoldenFile> series = readGoldenFile(entry.path().string());
    ASSERT_TRUE(series.ok()) << series.status();
    const golden_matrix_t* absl_nullable gaps = series->find(time_series::kSolveRotationGap);
    if (gaps == nullptr) continue;
    std::filesystem::path documentFile = entry.path();
    documentFile.replace_filename(name.substr(0, name.size() - std::string("_timeseries.txt").size()) + ".json");
    const absl::StatusOr<JsonValue> document = readJsonFile(documentFile.string());
    ASSERT_TRUE(document.ok()) << document.status();
    const JsonValue* absl_nullable maximum = document->findPath("initial_state_gap.max_rotation_rad");
    ASSERT_TRUE(maximum != nullptr && maximum->isNumber()) << documentFile;
    ASSERT_EQ(gaps->cols(), 2) << entry.path();
    ASSERT_GT(gaps->rows(), 0) << entry.path();
    EXPECT_EQ(gaps->col(1).maxCoeff(), maximum->asNumber()) << entry.path();
  }
}

TEST(ClosedLoopMetricsSchema, TheMainLinesEulerTurnSpikesOnlyAtItsOwnWraps) {
  // The whole-body G1's 720-degree turn of M0_main against M0's, both on Euler coordinates: each wrap of the measured
  // yaw makes a 2 pi gap that decays over a few solves, a little later than in M0, and nothing else spikes. The
  // exception lets exactly that through.
  const std::filesystem::path directory("humanoid_nmpc/humanoid_mpc_validation/data/closed_loop");
  const absl::StatusOr<JsonValue> candidate = readJsonFile((directory / "M0_main/unitree_g1_wb_turn_in_place_720.json").string());
  const absl::StatusOr<GoldenFile> candidateSeries =
      readGoldenFile((directory / "M0_main/unitree_g1_wb_turn_in_place_720_timeseries.txt").string());
  const absl::StatusOr<JsonValue> baseline = readJsonFile((directory / "M0/unitree_g1_wb_turn_in_place_720.json").string());
  const absl::StatusOr<GoldenFile> baselineSeries =
      readGoldenFile((directory / "M0/unitree_g1_wb_turn_in_place_720_timeseries.txt").string());
  ASSERT_TRUE(candidate.ok() && candidateSeries.ok() && baseline.ok() && baselineSeries.ok());
  ASSERT_NE(candidateSeries->find(time_series::kSolveRotationGap), nullptr) << "recorded with its per-solve gaps";
  const std::vector<std::string> violations =
      compareClosedLoopRuns(*candidate, *candidateSeries, *baseline, &*baselineSeries, BandOptions(), HeadingCrossingOptions());
  for (const std::string& violation : violations) {
    EXPECT_NE(violation.rfind("initial_state_gap", 0), 0u) << violation;
    EXPECT_NE(violation.rfind("candidate time series", 0), 0u) << violation;
  }
  // Without the localization the same comparison would have to pass any spike; a gap moved away from the wraps fails.
  GoldenFile shifted = *candidateSeries;
  for (GoldenEntry& entry : shifted.entries) {
    if (entry.label == time_series::kSolveRotationGap) entry.value.col(0).array() += 5.0;
  }
  const std::vector<std::string> moved =
      compareClosedLoopRuns(*candidate, shifted, *baseline, &*baselineSeries, BandOptions(), HeadingCrossingOptions());
  EXPECT_TRUE(std::any_of(moved.begin(), moved.end(),
                          [](const std::string& violation) { return violation.rfind("initial_state_gap.max_rotation_rad", 0) == 0; }));
}

}  // namespace
}  // namespace ocs2::humanoid::validation
