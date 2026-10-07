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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "gtest/gtest.h"

#include "humanoid_mpc_validation/closed_loop/ClosedLoopMetricsSchema.h"
#include "humanoid_mpc_validation/closed_loop/LockstepClosedLoop.h"
#include "humanoid_mpc_validation/closed_loop/LockstepOutputs.h"
#include "humanoid_mpc_validation/closed_loop/MetricBands.h"
#include "humanoid_mpc_validation/io/GoldenIo.h"
#include "humanoid_mpc_validation/io/RunProvenance.h"

/*
 * The closed-loop scenarios of the quaternion design's section 4.5 on one robot configuration (--robot), each from a
 * fresh MPC and simulator, in lockstep (LockstepClosedLoop). Every run writes its metrics document, its time series and
 * (walking) the recorded robot states into --output_dir, by default the test's undeclared outputs, where
 * `make closed-loop-metrics` collects them. With a baseline (--baseline, a label under data/closed_loop/), a run is
 * also compared with it within the bands of section 4.5.
 *
 * Tagged manual and exclusive: a run builds the MPC (compiling its CppAD libraries on the first one) and simulates up to
 * a minute of walking. Run one robot at a time.
 */

// The defaults of the flags that are not strings.
namespace {
constexpr bool kDefaultCompareSolveTime = false;
constexpr int kDefaultThreads = 0;  // not positive: the task file's multiple_shooting.n_threads
}  // namespace

// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(std::string, robot, "", "The configuration to run (RobotConfiguration::name).");
// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(std::string, label, "M0", "The label written into the metrics documents.");
// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(std::string, baseline, "", "A label under data/closed_loop/ to compare with within the bands of section 4.5; empty: none.");
// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(bool,
          compare_solve_time,
          kDefaultCompareSolveTime,
          "Also compare the p99 solve time with the baseline (only on the machine it was recorded on).");
// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(std::string, output_dir, "", "Where the outputs go; empty: $TEST_UNDECLARED_OUTPUTS_DIR.");
// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(int, threads, kDefaultThreads, "Overrides the task file's multiple_shooting.n_threads when positive.");

namespace ocs2::humanoid::validation {
namespace {

constexpr char kBaselineDirectory[] = "humanoid_nmpc/humanoid_mpc_validation/data/closed_loop";

/** The scenarios this suite runs: every one but the determinism test's smoke run. */
std::vector<std::string> suiteScenarioNames() {
  std::vector<std::string> names;
  for (const ClosedLoopScenario& scenario : closedLoopScenarios()) {
    if (scenario.name != "smoke") names.push_back(scenario.name);
  }
  return names;
}

bool isWalkingScenario(const std::string& name) {
  return name == "walk_0p5" || name == "walk_0p3";
}

std::string outputDirectory() {
  if (!absl::GetFlag(FLAGS_output_dir).empty()) return absl::GetFlag(FLAGS_output_dir);
  const char* absl_nullable undeclared = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  return undeclared != nullptr ? std::string(undeclared) : std::string("closed_loop_outputs");
}

class ClosedLoopScenarioTest : public ::testing::TestWithParam<std::string> {};

TEST_P(ClosedLoopScenarioTest, RunsAndStaysWithinTheBaselineBands) {
  const absl::StatusOr<RobotConfiguration> configuration = findRobotConfiguration(absl::GetFlag(FLAGS_robot));
  ASSERT_TRUE(configuration.ok()) << configuration.status();
  const absl::StatusOr<ClosedLoopScenario> scenario = findClosedLoopScenario(GetParam());
  ASSERT_TRUE(scenario.ok()) << scenario.status();

  LockstepOptions options;
  if (absl::GetFlag(FLAGS_threads) > 0) options.driver.solverThreads = static_cast<size_t>(absl::GetFlag(FLAGS_threads));
  options.recordRobotStates = isWalkingScenario(scenario->name);

  const RunEnvironment environment = runEnvironmentFromEnvironment();
  const absl::StatusOr<JsonValue> provenance = makeRunProvenanceJson(environment, configuration->configurationFiles());
  ASSERT_TRUE(provenance.ok()) << provenance.status();
  const absl::StatusOr<GoldenProvenance> goldenProvenance = makeRunGoldenProvenance(environment, configuration->configurationFiles());
  ASSERT_TRUE(goldenProvenance.ok()) << goldenProvenance.status();
  ClosedLoopRunInfo info;
  info.label = absl::GetFlag(FLAGS_label);
  info.provenance = *provenance;

  const LockstepClosedLoop closedLoop(*configuration, options);
  const absl::StatusOr<LockstepResult> result = closedLoop.run(*scenario, info);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_TRUE(validateClosedLoopMetrics(result->metrics).ok()) << validateClosedLoopMetrics(result->metrics);
  const absl::Status written = writeLockstepOutputs(*result, outputDirectory(), configuration->name, scenario->name, *goldenProvenance);
  ASSERT_TRUE(written.ok()) << written;
  LOG(WARNING) << configuration->name << " " << scenario->name << ":\n" << result->metrics.serialize();

  const std::string baselineLabel = absl::GetFlag(FLAGS_baseline);
  if (baselineLabel.empty()) return;
  const std::filesystem::path baselineFile =
      std::filesystem::path(kBaselineDirectory) / baselineLabel / metricsFileName(configuration->name, scenario->name);
  if (!std::filesystem::exists(baselineFile)) {
    LOG(WARNING) << "No baseline " << baselineFile << ": nothing to compare with.";
    return;
  }
  const absl::StatusOr<JsonValue> baseline = readJsonFile(baselineFile.string());
  ASSERT_TRUE(baseline.ok()) << baseline.status();
  BandOptions bands;
  bands.compareSolveTime = absl::GetFlag(FLAGS_compare_solve_time);
  // The baseline's time series, where archived (the turning scenarios), locates the crossings of the Euler yaw's cut.
  const std::filesystem::path baselineSeriesFile =
      std::filesystem::path(kBaselineDirectory) / baselineLabel / timeSeriesFileName(configuration->name, scenario->name);
  std::optional<GoldenFile> baselineSeries;
  if (std::filesystem::exists(baselineSeriesFile)) {
    absl::StatusOr<GoldenFile> series = readGoldenFile(baselineSeriesFile.string());
    ASSERT_TRUE(series.ok()) << series.status();
    baselineSeries = *std::move(series);
  }
  HeadingCrossingOptions crossings;
  crossings.commandedHeadingPeak = getCommandedHeadingPeak(*scenario);
  const std::vector<std::string> violations = compareClosedLoopRuns(
      result->metrics, result->timeSeries, *baseline, baselineSeries.has_value() ? &*baselineSeries : nullptr, bands, crossings);
  EXPECT_TRUE(violations.empty()) << "outside the bands of " << baselineLabel << ":\n  " << absl::StrJoin(violations, "\n  ");
}

INSTANTIATE_TEST_SUITE_P(Scenarios,
                         ClosedLoopScenarioTest,
                         ::testing::ValuesIn(suiteScenarioNames()),
                         [](const ::testing::TestParamInfo<std::string>& param) { return param.param; });

}  // namespace
}  // namespace ocs2::humanoid::validation

int main(int argc, char* absl_nonnull* absl_nonnull argv) {
  ::testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  // The controllers log every transition at INFO; the runs report through the metrics documents instead.
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kWarning);
  return RUN_ALL_TESTS();
}
