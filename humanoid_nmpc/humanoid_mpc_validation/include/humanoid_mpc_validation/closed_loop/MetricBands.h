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

#pragma once

#include <string>
#include <vector>

#include "absl/status/statusor.h"

#include "humanoid_mpc_validation/io/JsonValue.h"

namespace ocs2::humanoid::validation {

/** Which way a metric may move away from its baseline. */
enum class BandDirection {
  kTwoSided,  ///< by at most the band either way: a level, not an error (the base height's mean)
  kNoWorse,   ///< up by at most the band; down by more is an improvement, reported and not failed (an error, a tilt, a slip)
};

/**
 * One band of the quaternion design's section 4.5: a metric may differ from its baseline by max(relative |b|, absolute),
 * in the directions `direction` holds it to.
 */
struct MetricBand {
  std::string path;  ///< dotted path in the metrics document
  double relative = 0.0;
  double absolute = 0.0;
  BandDirection direction = BandDirection::kTwoSided;
};

// LINT.IfChange(metric_bands)
/**
 * The bands of section 4.5 against a baseline label, each one-sided (kNoWorse) but for the height's mean, so that the
 * improvements a change is meant to bring pass:
 *  - height: the mean within 5 mm either way, the RMS error no more than 5 mm worse;
 *  - tilt (RMS and maximum) no more than max(20 %, 0.005 rad) worse;
 *  - velocity no more than max(15 %, 0.02 m/s) worse, yaw rate no more than max(15 %, 0.02 rad/s) worse;
 *  - slip no more than max(20 %, 5 mm) worse.
 * Survival (equal or better), the p99 solve time (at most +10 %) and the quaternion norm (below 1e-9) are one-sided and
 * checked by compareClosedLoopMetrics() itself.
 */
const std::vector<MetricBand>& closedLoopMetricBands();
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/docs/quaternion_base_orientation/README.md, //humanoid_nmpc/humanoid_mpc_validation/README.md:metric_bands)
// clang-format on

/** What compareClosedLoopMetrics() compares. */
struct BandOptions {
  /// The p99 solve time is compared only between runs on the same machine; off, it is skipped.
  bool compareSolveTime = true;
};

/**
 * What a comparison of a candidate run with its baseline found: one sentence each, starting with the metric's path. The
 * candidate passes when there is no violation, whatever its improvements.
 */
struct ClosedLoopComparison {
  std::vector<std::string> violations;  ///< where the candidate breaks a rule
  /// Where it is better than its baseline by more than a kNoWorse band, or survived where the baseline fell: what a
  /// change meant to improve the formulation lists in its label's README.
  std::vector<std::string> improvements;
};

/** `comparison` as text: "violations (n):" and "improvements (m):", each followed by its sentences, indented. */
std::string describeClosedLoopComparison(const ClosedLoopComparison& comparison);

/**
 * `candidate` held to the bands around `baseline`: every way in which it falls outside them, and every way in which it
 * is better than a kNoWorse band. A robot that fell in the baseline is compared on survival alone (an improvement where
 * the candidate survived); one that fell in the candidate only, fails on survival and is compared no further. A metric
 * that is null in either document is skipped. A run (`make closed-loop-metrics ... BASELINE=`) and the offline
 * comparison of its archived document (compareArchivedClosedLoopRuns()) both compare through it, so they reach the same
 * verdict.
 */
ClosedLoopComparison compareClosedLoopMetrics(const JsonValue& candidate, const JsonValue& baseline, const BandOptions& options);

/** Which archived runs compareArchivedClosedLoopRuns() compares. */
struct ArchivedComparisonOptions {
  std::string candidateDirectory;  ///< the candidate label's directory, data/closed_loop/<label>
  std::string baselineDirectory;   ///< the baseline label's
  std::string robot;               ///< a RobotConfiguration::name to compare alone; empty: every configuration
  bool compareSolveTime = false;   ///< BandOptions::compareSolveTime, as the run's `--compare_solve_time`
};

/** One archived run of the candidate label, compared with the same run of the baseline label. */
struct ArchivedRunComparison {
  std::string robot;     ///< RobotConfiguration::name
  std::string scenario;  ///< ClosedLoopScenario::name
  ClosedLoopComparison comparison;
};

/** What compareArchivedClosedLoopRuns() found. */
struct ArchivedComparison {
  std::vector<ArchivedRunComparison> runs;  ///< in the order of robotConfigurations(), then of closedLoopScenarios()
  /// "<robot> <scenario>" of each candidate document the baseline has no document of: nothing to compare with.
  std::vector<std::string> withoutBaseline;
};

/**
 * The offline comparison of two recorded labels: every metrics document of the candidate's directory
 * (`<robot>_<scenario>.json`, as LockstepOutputs.h names it) with a document of the same run in the baseline's, held to
 * it by compareClosedLoopMetrics() as the run itself compared it. The archived documents read back to the run's values
 * bit for bit (JsonValue), so the verdict is the run's. NotFound for a directory that does not exist or an unknown
 * robot; the error of a file that cannot be read.
 */
absl::StatusOr<ArchivedComparison> compareArchivedClosedLoopRuns(const ArchivedComparisonOptions& options);

}  // namespace ocs2::humanoid::validation
