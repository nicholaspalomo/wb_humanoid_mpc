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

#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"

#include "humanoid_mpc_validation/io/GoldenIo.h"
#include "humanoid_mpc_validation/io/JsonValue.h"

namespace ocs2::humanoid::validation {

/** One band of the quaternion design's section 4.5: a metric may differ from its baseline by max(relative |b|, absolute). */
struct MetricBand {
  std::string path;  ///< dotted path in the metrics document
  double relative = 0.0;
  double absolute = 0.0;
};

// LINT.IfChange(metric_bands)
/**
 * The bands of section 4.5 against a baseline (M1, or M2 for the final validation):
 *  - height within 5 mm (mean and RMS error);
 *  - tilt within max(20 %, 0.005 rad) (RMS and maximum);
 *  - velocity within max(15 %, 0.02 m/s), yaw rate within max(15 %, 0.02 rad/s);
 *  - slip within max(20 %, 5 mm).
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
  /// Band paths not compared (compareClosedLoopRuns() replaces the tracking errors by their windowed values).
  std::vector<std::string> skippedPaths;
};

/**
 * Every way in which `candidate` falls outside the bands around `baseline`, one sentence each; empty when it is within
 * them. A robot that fell in the baseline is compared on survival alone; one that fell in the candidate only, fails on
 * survival and is compared no further. A metric that is null in either document is skipped.
 */
std::vector<std::string> compareClosedLoopMetrics(const JsonValue& candidate, const JsonValue& baseline, const BandOptions& options);

// LINT.IfChange(heading_crossing_exception)
/**
 * The 720-degree turn exception of section 4.5. An Euler run measures its yaw atan2-wrapped against an unwrapped warm
 * start, so where its heading crosses an odd multiple of pi (the cut of the wrapped yaw) its initial-state gap jumps by
 * 2 pi and its tracking is disturbed. A run whose baseline crosses the cut is compared:
 *  - on the velocity and yaw-rate errors recomputed from the two time series OUTSIDE windows of +-windowHalfWidth around
 *    the baseline's crossings, instead of the documents' values, within the same bands;
 *  - on reaching the baseline's heading peak: heading.cumulative_max_rad at least the baseline's minus
 *    headingTolerance (and the min, mirrored), and within commandedHeadingPeak +- headingTolerance where the baseline
 *    reached the commanded peak (the design asks for 4 pi; no robot of M0 reaches it);
 *  - on no spike of the initial-state gap: initial_state_gap.max_rotation_rad below maxRotationGap (the SQP's
 *    warm-start guard of a quarter turn), which a 2 pi wrap exceeds. A candidate on Euler coordinates (its
 *    quaternion_norm.max_deviation is null: M1, M2, a rerun of M0) wraps where its own heading crosses the cut, as its
 *    Euler baseline does, and its gap then decays over a few solves. Against a baseline whose own
 *    initial_state_gap.max_rotation_rad reached maxRotationGap, such a candidate is checked solve by solve instead,
 *    from its time series' solve_initial_state_rotation_gap (TimeSeriesLabels.h): a solve at or above maxRotationGap
 *    must lie within +-windowHalfWidth of one of the candidate's own crossings (eulerYawCrossingTimes() of its series)
 *    and stay below 2 pi + maxRotationGap (no more than a wrap of a gap below the threshold), and the series' largest gap
 *    must be the document's. Anywhere else, larger, or without the per-solve gaps, the spike is a violation. A
 *    quaternion candidate, or one against a baseline that did not spike, is held to the threshold everywhere;
 *  - on every other band as compareClosedLoopMetrics() compares it.
 */
struct HeadingCrossingOptions {
  double windowHalfWidth = 0.5;                ///< [s]
  double headingTolerance = 0.1;               ///< [rad]
  double maxRotationGap = 0.5 * M_PI;          ///< [rad]
  std::optional<double> commandedHeadingPeak;  ///< [rad] the largest heading change the scenario commands, if known
};

/**
 * The times [s] of a lockstep time series (LockstepResult::timeSeries: `time`, `base_quaternion_xyzw`) at which its twist
 * heading, unwrapped from the first sample's absolute heading, crosses an odd multiple of pi. InvalidArgument when the
 * series lacks a matrix or its sizes disagree.
 */
absl::StatusOr<std::vector<double>> eulerYawCrossingTimes(const GoldenFile& series);

/** The RMS tracking errors of a time series over the samples outside the windows. */
struct WindowedTrackingErrors {
  double velocityRms = 0.0;  ///< [m/s] the base velocity in the heading frame against the reference, as the metrics
  double yawRateRms = 0.0;   ///< [rad/s] the world yaw rate against the reference
  size_t samples = 0;        ///< the samples outside the windows
};

/**
 * The errors of `series` (`time`, `base_quaternion_xyzw`, `base_velocity_world_xy_yaw_rate`,
 * `reference_velocity_heading_xy_yaw_rate`) without the samples within `halfWidth` of a time of `windowCenters`.
 * InvalidArgument when the series lacks a matrix, its sizes disagree, or no sample is left.
 */
absl::StatusOr<WindowedTrackingErrors> trackingErrorsOutsideWindows(const GoldenFile& series,
                                                                    const std::vector<double>& windowCenters,
                                                                    double halfWidth);

/**
 * compareClosedLoopMetrics(), with the exception above when the baseline crosses the Euler yaw cut. `candidateSeries` is
 * the candidate's lockstep time series (LockstepResult::timeSeries). `baselineSeries` may be null: then a baseline whose
 * heading reaches +-pi (its documents' cumulative heading) is a violation of its own, "no time series to locate the
 * crossings", rather than a comparison that silently keeps the disturbed values.
 */
std::vector<std::string> compareClosedLoopRuns(const JsonValue& candidate,
                                               const GoldenFile& candidateSeries,
                                               const JsonValue& baseline,
                                               const GoldenFile* absl_nullable baselineSeries,
                                               const BandOptions& bandOptions,
                                               const HeadingCrossingOptions& crossingOptions);
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/docs/quaternion_base_orientation/README.md, //humanoid_nmpc/humanoid_mpc_validation/README.md:metric_bands)
// clang-format on

}  // namespace ocs2::humanoid::validation
