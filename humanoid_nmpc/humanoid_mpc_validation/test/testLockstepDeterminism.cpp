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

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/globals.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "humanoid_mpc_validation/closed_loop/ClosedLoopMetricsSchema.h"
#include "humanoid_mpc_validation/closed_loop/LockstepClosedLoop.h"
#include "humanoid_mpc_validation/io/GoldenIo.h"
#include "humanoid_mpc_validation/io/JsonValue.h"

/*
 * The lockstep closed loop is reproducible: runs of the short smoke scenario on the Unitree G1 centroidal MPC, with one
 * solver thread, agree to 1e-9 in every metric that does not time the machine, in their time series and in their last
 * observation - two runs in this process, and a third in a process of its own (this binary, re-executed). That is what
 * makes a closed-loop comparison across a code change (M0 against M1, Step 6's bitwise rerun) mean something, and those
 * comparisons always come from separate processes: a per-process source of nondeterminism, such as Abseil's hash seed,
 * which orders RobotDescription's joint list differently from one process to the next, shows only across processes.
 *
 * Only the first run compiles: the others load the CppAD libraries it generated, so this test cannot see a difference
 * between two generations of them. That was what made the production runs differ in their last bits from one run to
 * the next, not their solver threads: CppAD's recorder hashed a CppADCodeGen constant by its bytes, a heap address
 * among them, so the generated sources of six whole-body models followed the heap's layout, and every sandboxed run
 * generates its libraries afresh. The vendored CppADCodeGen now hashes a constant by its value (lib/ocs2/README.md,
 * test_cppad_codegen_determinism), and the production runs, with the configured threads, repeat bit for bit; their
 * comparisons across a code change use the bands of section 4.5. Tagged exclusive.
 */

namespace ocs2::humanoid::validation {
namespace {

constexpr double kTolerance = 1.0e-9;

/** Every difference between two documents beyond kTolerance, skipping what measures the machine. */
void collectDifferences(const JsonValue& a, const JsonValue& b, const std::string& path, std::vector<std::string>& differences) {
  if (absl::StartsWith(path, "solve_time_ms") || absl::StartsWith(path, "provenance")) return;
  if (a.type() != b.type()) {
    differences.push_back(absl::StrCat(path, ": different types"));
    return;
  }
  if (a.isNumber()) {
    const double scale = std::max(1.0, std::abs(b.asNumber()));
    if (std::abs(a.asNumber() - b.asNumber()) > kTolerance * scale) {
      differences.push_back(absl::StrCat(path, ": ", a.asNumber(), " against ", b.asNumber()));
    }
  } else if (a.isObject()) {
    if (a.size() != b.size()) differences.push_back(absl::StrCat(path, ": different members"));
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
      collectDifferences(a.valueAt(i), b.valueAt(i), path.empty() ? a.keyAt(i) : absl::StrCat(path, ".", a.keyAt(i)), differences);
    }
  } else if (a.isArray()) {
    if (a.size() != b.size()) differences.push_back(absl::StrCat(path, ": different lengths"));
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
      collectDifferences(a.at(i), b.at(i), absl::StrCat(path, "[", i, "]"), differences);
    }
  } else if (a != b) {
    differences.push_back(absl::StrCat(path, ": ", a.serialize(), " against ", b.serialize()));
  }
}

/** The environment variable that makes this binary the child of ARunInAProcessOfItsOwnAgrees: where it writes its run. */
constexpr char kChildOutputVariable[] = "LOCKSTEP_DETERMINISM_CHILD_OUTPUT";
constexpr char kChildTestName[] = "LockstepDeterminism.ChildProcessRun";
constexpr char kFinalStateLabel[] = "final_observation_state";

/** The smoke run of every test here: the Unitree G1 centroidal MPC, one solver thread. */
absl::StatusOr<LockstepResult> runSmoke() {
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kWarning);
  absl::StatusOr<RobotConfiguration> configuration = findRobotConfiguration("unitree_g1");
  if (!configuration.ok()) return configuration.status();
  absl::StatusOr<ClosedLoopScenario> scenario = findClosedLoopScenario("smoke");
  if (!scenario.ok()) return scenario.status();
  LockstepOptions options;
  options.driver.solverThreads = 1;
  options.recordRobotStates = true;
  options.recordingDelay = 0.0;
  const LockstepClosedLoop closedLoop(*configuration, options);
  ClosedLoopRunInfo info;
  info.label = "determinism";
  return closedLoop.run(*scenario, info);
}

/** `a` against `b`: the metrics, the last observation and the time series, to kTolerance. */
void expectRunsAgree(const JsonValue& firstMetrics,
                     const vector_t& firstFinalState,
                     const GoldenFile& firstSeries,
                     const JsonValue& secondMetrics,
                     const vector_t& secondFinalState,
                     const GoldenFile& secondSeries) {
  std::vector<std::string> differences;
  collectDifferences(firstMetrics, secondMetrics, /*path=*/"", differences);
  EXPECT_TRUE(differences.empty()) << differences.size() << " differences, the first: " << (differences.empty() ? "" : differences.front());

  ASSERT_EQ(firstFinalState.size(), secondFinalState.size());
  EXPECT_LE((firstFinalState - secondFinalState).cwiseAbs().maxCoeff(), kTolerance);
  ASSERT_EQ(firstSeries.entries.size(), secondSeries.entries.size());
  for (size_t i = 0; i < firstSeries.entries.size(); ++i) {
    const GoldenEntry& a = firstSeries.entries[i];
    const GoldenEntry& b = secondSeries.entries[i];
    ASSERT_EQ(a.label, b.label);
    ASSERT_EQ(a.value.rows(), b.value.rows()) << a.label;
    ASSERT_EQ(a.value.cols(), b.value.cols()) << a.label;
    if (a.value.size() > 0) {
      EXPECT_LE((a.value - b.value).cwiseAbs().maxCoeff(), kTolerance) << a.label;
    }
  }
}

/**
 * Runs this test binary again, on the child test alone, with `outputPrefix` for its run, and waits for it. The test
 * runner's own files (the XML report, the premature-exit and shard files) are left out of the child's environment, so
 * that only this process reports.
 */
int runChildProcess(const std::string& outputPrefix) {
  std::vector<std::string> environment;
  for (char* absl_nullable* absl_nonnull variable = environ; *variable != nullptr; ++variable) {
    const std::string entry(*variable);
    if (absl::StartsWith(entry, "XML_OUTPUT_FILE=") || absl::StartsWith(entry, "GTEST_") ||
        absl::StartsWith(entry, "TEST_PREMATURE_EXIT_FILE=") || absl::StartsWith(entry, "TEST_SHARD_STATUS_FILE=") ||
        absl::StartsWith(entry, "TEST_TOTAL_SHARDS=") || absl::StartsWith(entry, "TEST_SHARD_INDEX=")) {
      continue;
    }
    environment.push_back(entry);
  }
  environment.push_back(absl::StrCat(kChildOutputVariable, "=", outputPrefix));
  std::vector<char* absl_nullable> envp;
  for (std::string& entry : environment) envp.push_back(entry.data());
  envp.push_back(nullptr);

  std::string program = std::filesystem::read_symlink("/proc/self/exe").string();
  std::string filter = absl::StrCat("--gtest_filter=", kChildTestName);
  std::vector<char* absl_nullable> argv = {program.data(), filter.data(), nullptr};
  pid_t child = 0;
  if (posix_spawn(&child, program.c_str(), /*file_actions=*/nullptr, /*attrp=*/nullptr, argv.data(), envp.data()) != 0) return -1;
  int status = 0;
  if (waitpid(child, &status, /*options=*/0) != child) return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

TEST(LockstepDeterminism, ChildProcessRun) {
  // Only as the child of ARunInAProcessOfItsOwnAgrees, which names where the run goes.
  const char* absl_nullable outputPrefix = std::getenv(kChildOutputVariable);
  if (outputPrefix == nullptr) GTEST_SKIP() << "the child process of ARunInAProcessOfItsOwnAgrees";
  const absl::StatusOr<LockstepResult> run = runSmoke();
  ASSERT_TRUE(run.ok()) << run.status();
  ASSERT_TRUE(writeJsonFile(absl::StrCat(outputPrefix, ".json"), run->metrics).ok());
  GoldenFile series = run->timeSeries;
  series.entries.push_back(GoldenEntry{.label = kFinalStateLabel, .value = golden_matrix_t(run->finalObservationState.transpose())});
  ASSERT_TRUE(writeGoldenFile(absl::StrCat(outputPrefix, "_series.txt"), series).ok());
}

TEST(LockstepDeterminism, TwoRunsWithOneSolverThreadAgree) {
  const absl::StatusOr<LockstepResult> first = runSmoke();
  ASSERT_TRUE(first.ok()) << first.status();
  const absl::StatusOr<LockstepResult> second = runSmoke();
  ASSERT_TRUE(second.ok()) << second.status();
  const absl::StatusOr<ClosedLoopScenario> scenario = findClosedLoopScenario("smoke");
  ASSERT_TRUE(scenario.ok()) << scenario.status();

  // The runs did what the scenario asks: valid documents, the robot standing, solves at the task file's rate.
  ASSERT_TRUE(validateClosedLoopMetrics(first->metrics).ok()) << validateClosedLoopMetrics(first->metrics);
  EXPECT_TRUE(first->metrics.findPath("survival.survived")->asBool()) << first->metrics.serialize();
  EXPECT_EQ(first->metrics.findPath("non_finite_values")->asNumber(), 0.0);
  const double solves = first->metrics.findPath("evaluation.solves")->asNumber();
  const double expectedSolves = getCommandDuration(*scenario) * first->metrics.findPath("settings.mpc_frequency_hz")->asNumber();
  EXPECT_NEAR(solves, expectedSolves, 2.0);
  EXPECT_GT(first->recordedStates.records.size(), 0u);

  expectRunsAgree(first->metrics, first->finalObservationState, first->timeSeries, second->metrics, second->finalObservationState,
                  second->timeSeries);
}

TEST(LockstepDeterminism, ARunInAProcessOfItsOwnAgrees) {
  const absl::StatusOr<LockstepResult> here = runSmoke();
  ASSERT_TRUE(here.ok()) << here.status();
  const char* absl_nullable tmp = std::getenv("TEST_TMPDIR");
  const std::string outputPrefix = (std::filesystem::path(tmp != nullptr ? tmp : "/tmp") / "lockstep_determinism_child").string();
  ASSERT_EQ(runChildProcess(outputPrefix), 0) << "the child process failed";

  const absl::StatusOr<JsonValue> childMetrics = readJsonFile(absl::StrCat(outputPrefix, ".json"));
  ASSERT_TRUE(childMetrics.ok()) << childMetrics.status();
  absl::StatusOr<GoldenFile> childSeries = readGoldenFile(absl::StrCat(outputPrefix, "_series.txt"));
  ASSERT_TRUE(childSeries.ok()) << childSeries.status();
  const golden_matrix_t* absl_nullable childFinalState = childSeries->find(kFinalStateLabel);
  ASSERT_NE(childFinalState, nullptr);
  const vector_t childFinal = childFinalState->row(0).transpose();
  childSeries->entries.pop_back();
  expectRunsAgree(here->metrics, here->finalObservationState, here->timeSeries, *childMetrics, childFinal, *childSeries);
}

}  // namespace
}  // namespace ocs2::humanoid::validation
