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

#include <cstdio>
#include <filesystem>
#include <string>

#include "absl/base/nullability.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"

#include "humanoid_common_mpc_app/node/ExitCodes.h"
#include "humanoid_mpc_validation/closed_loop/MetricBands.h"

/*
 * compare_closed_loop: two recorded labels of data/closed_loop/ compared
 * offline (compareArchivedClosedLoopRuns()): every archived run of --candidate
 * held to the same run of --baseline under the rules the run itself applies
 * (compareClosedLoopMetrics()), from the archived documents, without a
 * simulation. Prints each run's violations and improvements and a summary;
 * exits 1 when a run is outside the rules and 2 when the comparison cannot be
 * made. `make compare-closed-loop CANDIDATE=<label> BASELINE=<label>
 * [ROBOT=<robot>]` runs it under `bazel test`.
 */

// The default of the flag that is not a string: as the run's
// (closedLoopScenarios.cpp), and `make closed-loop-metrics` passes neither.
namespace {
constexpr bool kDefaultCompareSolveTime = false;
}  // namespace

// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(std::string,
          candidate,
          "",
          "The label under data/closed_loop/ whose runs are compared.");
// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(std::string,
          baseline,
          "",
          "The label under data/closed_loop/ they are compared with.");
// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(std::string,
          robot,
          "",
          "The configuration to compare alone (RobotConfiguration::name); "
          "empty: every one.");
// NOLINTNEXTLINE(misc-use-internal-linkage): ABSL_FLAG must be at global scope.
ABSL_FLAG(bool,
          compare_solve_time,
          kDefaultCompareSolveTime,
          "Also compare the p99 solve time (only between labels recorded on "
          "one machine), as the run's flag of that name.");

namespace ocs2::humanoid::validation {
namespace {

// LINT.IfChange(comparison_paths)
constexpr char kClosedLoopDirectory[] =
    "humanoid_nmpc/humanoid_mpc_validation/data/closed_loop";
// LINT.ThenChange(//Makefile:closed_loop_targets)

int compareLabels() {
  const std::string candidate = absl::GetFlag(FLAGS_candidate);
  const std::string baseline = absl::GetFlag(FLAGS_baseline);
  if (candidate.empty() || baseline.empty()) {
    absl::FPrintF(stderr,
                  "Usage: compare_closed_loop --candidate=<label> "
                  "--baseline=<label> [--robot=<robot>]\n");
    return node::kUsageErrorExitCode;
  }
  const ArchivedComparisonOptions options{
      .candidateDirectory =
          (std::filesystem::path(kClosedLoopDirectory) / candidate).string(),
      .baselineDirectory =
          (std::filesystem::path(kClosedLoopDirectory) / baseline).string(),
      .robot = absl::GetFlag(FLAGS_robot),
      .compareSolveTime = absl::GetFlag(FLAGS_compare_solve_time),
  };
  const absl::StatusOr<ArchivedComparison> comparison =
      compareArchivedClosedLoopRuns(options);
  if (!comparison.ok()) {
    absl::FPrintF(stderr, "%s\n", comparison.status().ToString());
    return node::kUsageErrorExitCode;
  }
  size_t outside = 0;
  size_t improved = 0;
  for (const ArchivedRunComparison& run : comparison->runs) {
    absl::PrintF("%s %s against %s:\n%s\n", run.robot, run.scenario, baseline,
                 describeClosedLoopComparison(run.comparison));
    if (!run.comparison.violations.empty()) ++outside;
    if (!run.comparison.improvements.empty()) ++improved;
  }
  for (const std::string& run : comparison->withoutBaseline) {
    absl::PrintF("No baseline of %s in %s: nothing to compare with.\n", run,
                 baseline);
  }
  absl::PrintF(
      "%s against %s: %d runs compared, %d outside the rules, %d with "
      "improvements, %d without a baseline.\n",
      candidate, baseline, comparison->runs.size(), outside, improved,
      comparison->withoutBaseline.size());
  if (comparison->runs.empty()) {
    absl::FPrintF(stderr, "No run of %s has a baseline in %s.\n", candidate,
                  baseline);
    return node::kUsageErrorExitCode;
  }
  return outside == 0 ? 0 : 1;
}

}  // namespace
}  // namespace ocs2::humanoid::validation

int main(int argc, char* absl_nonnull* absl_nonnull argv) {
  absl::ParseCommandLine(argc, argv);
  return ocs2::humanoid::validation::compareLabels();
}
