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

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

#include "humanoid_mpc_validation/benchmark/MpcSolveBenchmark.h"
#include "humanoid_mpc_validation/benchmark/SolveBenchmarkGate.h"
#include "humanoid_mpc_validation/io/GoldenIo.h"
#include "humanoid_mpc_validation/io/RunProvenance.h"

/*
 * benchmark_mpc_solve: the solve benchmark of the quaternion design's section 4.6 (runSolveBenchmark) for one robot
 * configuration, on robot states a lockstep walking run recorded. Writes <robot>_solve_benchmark.json into --output_dir
 * (by default the test's undeclared outputs, where `make benchmark-mpc-solve` collects it) and prints it.
 */

// The defaults of the numeric flags.
namespace {
constexpr int kDefaultThreads = 0;  // not positive: the task file's multiple_shooting.n_threads
constexpr int kDefaultRepeats = 3;
constexpr int kDefaultWarmupSolves = 25;
}  // namespace

// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(std::string, robot, "", "The configuration to benchmark (RobotConfiguration::name).");
// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(std::string,
          states,
          "",
          "The recorded robot states; empty: data/benchmark/states/<robot>_<walking scenario>_states.txt, the walking scenario of the "
          "robot's configuration (walk_0p5, or walk_0p3 for EngineAI SA01 and Unitree R1).");
// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(std::string,
          baseline,
          "",
          "A label under data/benchmark/ (e.g. B0) to hold the run to with the real-time gate of section 4.6; empty: none.");
// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(std::string, label, "B0", "The label written into the document.");
// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(std::string, output_dir, "", "Where the document goes; empty: $TEST_UNDECLARED_OUTPUTS_DIR, else the working directory.");
// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(int,
          threads,
          kDefaultThreads,
          "Overrides the task file's multiple_shooting.n_threads when positive; the real-time gate uses the configured threads.");
// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(int, repeats, kDefaultRepeats, "Passes over the recording.");
// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(int, warmup, kDefaultWarmupSolves, "Untimed solves at the start of each pass.");

namespace ocs2::humanoid::validation {
namespace {

// LINT.IfChange(benchmark_paths)
constexpr char kBenchmarkDirectory[] = "humanoid_nmpc/humanoid_mpc_validation/data/benchmark";
constexpr char kStatesDirectory[] = "humanoid_nmpc/humanoid_mpc_validation/data/benchmark/states";
// LINT.ThenChange(//Makefile:closed_loop_targets)

int runBenchmark() {
  const absl::StatusOr<RobotConfiguration> configuration = findRobotConfiguration(absl::GetFlag(FLAGS_robot));
  if (!configuration.ok()) {
    std::cerr << configuration.status() << "\n";
    return 2;
  }
  const std::string statesFile = absl::GetFlag(FLAGS_states).empty()
                                     ? (std::filesystem::path(kStatesDirectory) /
                                        absl::StrCat(configuration->name, "_", configuration->walkingScenario, "_states.txt"))
                                           .string()
                                     : absl::GetFlag(FLAGS_states);
  const absl::StatusOr<GoldenFile> golden = readGoldenFile(statesFile);
  if (!golden.ok()) {
    std::cerr << golden.status() << "\n";
    return 2;
  }
  const absl::StatusOr<RecordedRobotStates> states = fromGoldenFile(*golden);
  if (!states.ok()) {
    std::cerr << states.status() << "\n";
    return 2;
  }

  SolveBenchmarkOptions options;
  if (absl::GetFlag(FLAGS_threads) > 0) options.driver.solverThreads = static_cast<size_t>(absl::GetFlag(FLAGS_threads));
  options.repeats = static_cast<size_t>(std::max(1, absl::GetFlag(FLAGS_repeats)));
  options.warmupSolves = static_cast<size_t>(std::max(0, absl::GetFlag(FLAGS_warmup)));

  std::vector<std::string> files = configuration->configurationFiles();
  files.push_back(statesFile);
  const absl::StatusOr<JsonValue> provenance = makeRunProvenanceJson(runEnvironmentFromEnvironment(), files);
  if (!provenance.ok()) {
    std::cerr << provenance.status() << "\n";
    return 2;
  }
  const absl::StatusOr<JsonValue> document = runSolveBenchmark(*configuration, *states, options, absl::GetFlag(FLAGS_label), *provenance);
  if (!document.ok()) {
    std::cerr << document.status() << "\n";
    return 1;
  }

  std::string outputDir = absl::GetFlag(FLAGS_output_dir);
  if (outputDir.empty()) {
    const char* absl_nullable undeclared = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
    outputDir = undeclared != nullptr ? undeclared : ".";
  }
  const std::string outputFile = (std::filesystem::path(outputDir) / absl::StrCat(configuration->name, "_solve_benchmark.json")).string();
  const absl::Status written = writeJsonFile(outputFile, *document);
  if (!written.ok()) {
    std::cerr << written << "\n";
    return 1;
  }
  std::cout << document->serialize() << "\nWritten to " << outputFile << "\n";

  // The real-time gate against a baseline of the same robot, recorded on the same machine and states.
  if (!absl::GetFlag(FLAGS_baseline).empty()) {
    const std::string baselineFile = (std::filesystem::path(kBenchmarkDirectory) / absl::GetFlag(FLAGS_baseline) /
                                      absl::StrCat(configuration->name, "_solve_benchmark.json"))
                                         .string();
    const absl::StatusOr<JsonValue> baseline = readJsonFile(baselineFile);
    if (!baseline.ok()) {
      std::cerr << baseline.status() << "\n";
      return 2;
    }
    const std::vector<std::string> violations = compareSolveBenchmarks(*document, *baseline, SolveBenchmarkGate());
    if (!violations.empty()) {
      std::cerr << "Outside the real-time gate of " << absl::GetFlag(FLAGS_baseline) << ":\n  " << absl::StrJoin(violations, "\n  ")
                << "\n";
      return 1;
    }
    std::cout << "Within the real-time gate of " << absl::GetFlag(FLAGS_baseline) << ".\n";
  }
  return 0;
}

}  // namespace
}  // namespace ocs2::humanoid::validation

int main(int argc, char* absl_nonnull* absl_nonnull argv) {
  absl::ParseCommandLine(argc, argv);
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kWarning);
  return ocs2::humanoid::validation::runBenchmark();
}
