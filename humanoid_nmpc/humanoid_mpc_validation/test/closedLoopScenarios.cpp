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
#include <fstream>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "humanoid_mpc_validation/closed_loop/ClosedLoopMetricsSchema.h"
#include "humanoid_mpc_validation/closed_loop/LockstepClosedLoop.h"
#include "humanoid_mpc_validation/closed_loop/LockstepOutputs.h"
#include "humanoid_mpc_validation/closed_loop/MetricBands.h"
#include "humanoid_mpc_validation/io/GoldenIo.h"
#include "humanoid_mpc_validation/io/JsonValue.h"
#include "humanoid_mpc_validation/io/RunProvenance.h"

/*
 * The closed-loop scenarios of the quaternion design's section 4.5 on one robot configuration (--robot), each from a
 * fresh MPC and simulator, in lockstep (LockstepClosedLoop). Every run writes its metrics document, its time series and
 * (walking) the recorded robot states into --output_dir, by default the test's undeclared outputs, where
 * `make closed-loop-metrics` collects them. With a baseline (--baseline, a label under data/closed_loop/), a run is
 * also compared with it within the bands of section 4.5 (compareClosedLoopMetrics()), and its violations and
 * improvements are logged and written beside the outputs, where the Make target prints them. --label is required.
 *
 * The closed_loop_<robot> tests are tagged manual and exclusive: a run builds the MPC (compiling its CppAD libraries on
 * the first one) and simulates up to a minute of walking. Run one robot at a time. closed_loop_scenarios, the same source
 * as a binary of every configuration, is not manual, so that `bazel build //...` compiles it.
 */

// The defaults of the flags that are not strings.
namespace {
constexpr bool kDefaultCompareSolveTime = false;
constexpr int kDefaultThreads = 0;  // not positive: the task file's multiple_shooting.n_threads
}  // namespace

// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(std::string, robot, "", "The configuration to run (RobotConfiguration::name).");
// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(std::string, label, "", "The label written into the metrics documents (required).");
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

// LINT.IfChange(comparison_file)
/** Writes `description`, the run's comparison with `baselineLabel`, into <robot>_<scenario>_comparison.txt of `directory`. */
absl::Status writeComparison(const std::string& directory,
                             const std::string& robot,
                             const std::string& scenario,
                             const std::string& baselineLabel,
                             const std::string& description) {
  const std::filesystem::path file = std::filesystem::path(directory) / absl::StrCat(robot, "_", scenario, "_comparison.txt");
  std::ofstream stream(file);
  stream << absl::StrCat(robot, " ", scenario, " against ", baselineLabel, ":\n", description, "\n");
  if (!stream) return absl::InternalError(absl::StrCat("cannot write ", file.string()));
  return absl::OkStatus();
}
// LINT.ThenChange(//Makefile:closed_loop_targets)

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
  const ClosedLoopComparison comparison =
      compareClosedLoopMetrics(result->metrics, *baseline, BandOptions{.compareSolveTime = absl::GetFlag(FLAGS_compare_solve_time)});
  const std::string description = describeClosedLoopComparison(comparison);
  LOG(WARNING) << configuration->name << " " << scenario->name << " against " << baselineLabel << ":\n" << description;
  const absl::Status comparisonWritten =
      writeComparison(outputDirectory(), configuration->name, scenario->name, baselineLabel, description);
  EXPECT_TRUE(comparisonWritten.ok()) << comparisonWritten;
  EXPECT_TRUE(comparison.violations.empty()) << "outside the bands of " << baselineLabel << ":\n" << description;
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
  if (absl::GetFlag(FLAGS_label).empty()) {
    LOG(ERROR) << "--label is required: the label the metrics documents are recorded under.";
    return 2;
  }
  // The controllers log every transition at INFO; the runs report through the metrics documents instead.
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kWarning);
  return RUN_ALL_TESTS();
}
