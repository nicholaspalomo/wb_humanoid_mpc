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

#include "humanoid_mpc_validation/closed_loop/MetricBands.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_mpc_validation/closed_loop/TimeSeriesLabels.h"

namespace ocs2::humanoid::validation {
namespace {

std::optional<double> numberAt(const JsonValue& document, const std::string& path) {
  const JsonValue* absl_nullable value = document.findPath(path);
  if (value == nullptr || !value->isNumber()) return std::nullopt;
  return value->asNumber();
}

bool survivedIn(const JsonValue& document) {
  const JsonValue* absl_nullable survived = document.findPath("survival.survived");
  return survived != nullptr && survived->isBool() && survived->asBool();
}

/** An angle wrapped to (-pi, pi]. */
double wrapAngle(double angle) {
  return angle - 2.0 * M_PI * std::ceil((angle - M_PI) / (2.0 * M_PI));
}

/** The matrix `label` of `series`, with `cols` columns and `rows` rows (any number when rows < 0). */
absl::StatusOr<const golden_matrix_t* absl_nonnull> seriesMatrix(const GoldenFile& series,
                                                                 const std::string& label,
                                                                 Eigen::Index cols,
                                                                 Eigen::Index rows) {
  const golden_matrix_t* absl_nullable matrix = series.find(label);
  if (matrix == nullptr) return absl::InvalidArgumentError(absl::StrCat("the time series has no matrix '", label, "'"));
  if (matrix->cols() != cols || (rows >= 0 && matrix->rows() != rows)) {
    return absl::InvalidArgumentError(absl::StrCat("the time series matrix '", label, "' is ", matrix->rows(), " x ", matrix->cols()));
  }
  return matrix;
}

/** The twist heading of row `row` of a base_quaternion_xyzw matrix. */
double headingAt(const golden_matrix_t& quaternions, Eigen::Index row) {
  return twistHeading(vector4_t(quaternions.row(row).transpose()));
}

bool withinAWindow(double time, const std::vector<double>& windowCenters, double halfWidth) {
  return std::any_of(windowCenters.begin(), windowCenters.end(),
                     [time, halfWidth](double center) { return std::abs(time - center) <= halfWidth; });
}

/** The solves a check counts, and the first of them, for its sentence. */
struct CountedSolves {
  size_t count = 0;
  double firstTime = 0.0;
  double firstGap = 0.0;

  void add(double time, double gap) {
    if (count++ == 0) {
      firstTime = time;
      firstGap = gap;
    }
  }
};

/**
 * The 720-degree turn exception's gap check of a candidate on Euler coordinates against a baseline whose own gap spiked
 * (see HeadingCrossingOptions): one sentence for each way in which the candidate's per-solve gaps (`candidateSeries`'s
 * solve_initial_state_rotation_gap) spike where no Euler wrap of its own explains it, or cannot show where they spiked.
 * `documentGap` is the candidate document's initial_state_gap.max_rotation_rad.
 *
 * Its measured yaw wraps where its own heading crosses the cut, which the solve at the crossing sees as a gap of 2 pi
 * plus the gap it would have had without the wrap, so less than 2 pi + maxRotationGap where that one was below the
 * threshold (the triangle inequality). The next solves start from the solutions that one SQP iteration each moved
 * towards the wrapped yaw, so the gap then decays over a few solves (on the whole-body G1 of M0_main, from 2 pi to below
 * pi / 2 within 0.17 s). A spike is therefore explained within +-windowHalfWidth of one of the candidate's own
 * crossings and below 2 pi + maxRotationGap.
 */
std::vector<std::string> unexplainedRotationGapSpikes(const GoldenFile& candidateSeries,
                                                      double documentGap,
                                                      const HeadingCrossingOptions& options) {
  const golden_matrix_t* absl_nullable gaps = candidateSeries.find(time_series::kSolveRotationGap);
  if (gaps == nullptr || gaps->cols() != 2) {
    return {absl::StrCat("initial_state_gap.max_rotation_rad: ", documentGap,
                         " rad, a spike the candidate's time series cannot locate (no ", time_series::kSolveRotationGap,
                         " matrix of two columns)")};
  }
  const absl::StatusOr<std::vector<double>> crossings = eulerYawCrossingTimes(candidateSeries);
  if (!crossings.ok()) return {absl::StrCat("candidate time series: ", crossings.status().message())};

  std::vector<std::string> violations;
  std::optional<double> largest;
  CountedSolves outside;
  CountedSolves beyondAWrap;
  const double largestWrap = 2.0 * M_PI + options.maxRotationGap;
  for (Eigen::Index row = 0; row < gaps->rows(); ++row) {
    const double time = (*gaps)(row, 0);
    const double gap = (*gaps)(row, 1);
    if (!largest.has_value() || gap > *largest) largest = gap;
    if (!(gap >= options.maxRotationGap)) continue;
    if (!withinAWindow(time, *crossings, options.windowHalfWidth)) {
      outside.add(time, gap);
    } else if (!(gap < largestWrap)) {
      beyondAWrap.add(time, gap);
    }
  }
  if (!largest.has_value() || *largest != documentGap) {
    violations.push_back(absl::StrCat("initial_state_gap.max_rotation_rad: ", documentGap, " rad in the document, but the largest of the ",
                                      gaps->rows(), " solves of the time series is ",
                                      largest.has_value() ? absl::StrCat(*largest) : std::string("none"),
                                      ": the two are not of the same run"));
  }
  if (outside.count > 0) {
    violations.push_back(absl::StrCat("initial_state_gap.max_rotation_rad: ", outside.count, " solves at or above ", options.maxRotationGap,
                                      " rad outside the +-", options.windowHalfWidth, " s windows around the candidate's ",
                                      crossings->size(), " crossings of the Euler yaw's cut, the first at ", outside.firstTime, " s (",
                                      outside.firstGap, " rad): a spike no Euler wrap explains"));
  }
  if (beyondAWrap.count > 0) {
    violations.push_back(absl::StrCat("initial_state_gap.max_rotation_rad: ", beyondAWrap.count,
                                      " solves at the candidate's crossings at or above 2 pi + ", options.maxRotationGap,
                                      " rad, the first at ", beyondAWrap.firstTime, " s (", beyondAWrap.firstGap,
                                      " rad): more than a wrap of a gap below the threshold"));
  }
  return violations;
}

}  // namespace

const std::vector<MetricBand>& closedLoopMetricBands() {
  static const absl::NoDestructor<std::vector<MetricBand>> kBands(std::vector<MetricBand>{
      {"base_height.mean_m", /*relative=*/0.0, /*absolute=*/0.005},
      {"base_height.rms_error_m", /*relative=*/0.0, /*absolute=*/0.005},
      {"tilt.rms_rad", /*relative=*/0.20, /*absolute=*/0.005},
      {"tilt.max_rad", /*relative=*/0.20, /*absolute=*/0.005},
      {"velocity.rms_error_mps", /*relative=*/0.15, /*absolute=*/0.02},
      {"yaw_rate.rms_error_radps", /*relative=*/0.15, /*absolute=*/0.02},
      {"stance_foot_slip.max_m", /*relative=*/0.20, /*absolute=*/0.005},
  });
  return *kBands;
}

std::vector<std::string> compareClosedLoopMetrics(const JsonValue& candidate, const JsonValue& baseline, const BandOptions& options) {
  std::vector<std::string> violations;
  const bool baselineSurvived = survivedIn(baseline);
  const bool candidateSurvived = survivedIn(candidate);
  if (!baselineSurvived) return violations;  // nothing to be worse than but the survival, which a fall cannot be
  if (!candidateSurvived) {
    const JsonValue* absl_nullable reason = candidate.findPath("survival.fall_reason");
    violations.push_back(absl::StrCat("survival: the baseline survived and the candidate fell (",
                                      reason != nullptr && reason->isString() ? reason->asString() : std::string("no reason"), ")"));
    return violations;
  }

  for (const MetricBand& band : closedLoopMetricBands()) {
    if (std::find(options.skippedPaths.begin(), options.skippedPaths.end(), band.path) != options.skippedPaths.end()) continue;
    const std::optional<double> value = numberAt(candidate, band.path);
    const std::optional<double> reference = numberAt(baseline, band.path);
    if (!value.has_value() || !reference.has_value()) continue;
    const double allowed = std::max(band.relative * std::abs(*reference), band.absolute);
    if (std::abs(*value - *reference) > allowed) {
      violations.push_back(absl::StrCat(band.path, ": ", *value, " against ", *reference, ", outside +/-", allowed));
    }
  }

  if (options.compareSolveTime) {
    const std::optional<double> p99 = numberAt(candidate, "solve_time_ms.total.p99");
    const std::optional<double> baselineP99 = numberAt(baseline, "solve_time_ms.total.p99");
    if (p99.has_value() && baselineP99.has_value() && *p99 > 1.10 * *baselineP99) {
      violations.push_back(absl::StrCat("solve_time_ms.total.p99: ", *p99, " ms against ", *baselineP99, " ms, more than +10 %"));
    }
  }

  const std::optional<double> normDeviation = numberAt(candidate, "quaternion_norm.max_deviation");
  if (normDeviation.has_value() && *normDeviation >= 1.0e-9) {
    violations.push_back(absl::StrCat("quaternion_norm.max_deviation: ", *normDeviation, ", not below 1e-9"));
  }
  return violations;
}

absl::StatusOr<std::vector<double>> eulerYawCrossingTimes(const GoldenFile& series) {
  absl::StatusOr<const golden_matrix_t* absl_nonnull> time = seriesMatrix(series, time_series::kTime, /*cols=*/1, /*rows=*/-1);
  if (!time.ok()) return time.status();
  absl::StatusOr<const golden_matrix_t* absl_nonnull> quaternions =
      seriesMatrix(series, time_series::kBaseQuaternion, /*cols=*/4, (*time)->rows());
  if (!quaternions.ok()) return quaternions.status();
  std::vector<double> crossings;
  if ((*time)->rows() == 0) return crossings;
  // The absolute heading, unwrapped; a crossing is a change of the 2 pi branch the wrapped yaw (-pi, pi] would be on.
  double previousHeading = headingAt(**quaternions, /*row=*/0);
  double unwrapped = previousHeading;
  double previousBranch = std::floor((unwrapped + M_PI) / (2.0 * M_PI));
  for (Eigen::Index row = 1; row < (*time)->rows(); ++row) {
    const double heading = headingAt(**quaternions, row);
    unwrapped += wrapAngle(heading - previousHeading);
    previousHeading = heading;
    const double branch = std::floor((unwrapped + M_PI) / (2.0 * M_PI));
    if (branch != previousBranch) crossings.push_back((**time)(row, 0));
    previousBranch = branch;
  }
  return crossings;
}

absl::StatusOr<WindowedTrackingErrors> trackingErrorsOutsideWindows(const GoldenFile& series,
                                                                    const std::vector<double>& windowCenters,
                                                                    double halfWidth) {
  absl::StatusOr<const golden_matrix_t* absl_nonnull> time = seriesMatrix(series, time_series::kTime, /*cols=*/1, /*rows=*/-1);
  if (!time.ok()) return time.status();
  const Eigen::Index rows = (*time)->rows();
  absl::StatusOr<const golden_matrix_t* absl_nonnull> quaternions = seriesMatrix(series, time_series::kBaseQuaternion, /*cols=*/4, rows);
  if (!quaternions.ok()) return quaternions.status();
  absl::StatusOr<const golden_matrix_t* absl_nonnull> velocities = seriesMatrix(series, time_series::kBaseVelocity, /*cols=*/3, rows);
  if (!velocities.ok()) return velocities.status();
  absl::StatusOr<const golden_matrix_t* absl_nonnull> references = seriesMatrix(series, time_series::kReferenceVelocity, /*cols=*/3, rows);
  if (!references.ok()) return references.status();

  WindowedTrackingErrors errors;
  double velocitySquaredSum = 0.0;
  double yawRateSquaredSum = 0.0;
  for (Eigen::Index row = 0; row < rows; ++row) {
    if (withinAWindow((**time)(row, 0), windowCenters, halfWidth)) continue;
    // As ClosedLoopMetrics: the world velocity turned into the heading frame, and the world yaw rate.
    const double heading = headingAt(**quaternions, row);
    const double vx = (**velocities)(row, 0);
    const double vy = (**velocities)(row, 1);
    const Eigen::Vector2d headingVelocity(std::cos(heading) * vx + std::sin(heading) * vy,
                                          -std::sin(heading) * vx + std::cos(heading) * vy);
    velocitySquaredSum += (headingVelocity - Eigen::Vector2d((**references)(row, 0), (**references)(row, 1))).squaredNorm();
    const double yawRateError = (**velocities)(row, 2) - (**references)(row, 2);
    yawRateSquaredSum += yawRateError * yawRateError;
    ++errors.samples;
  }
  if (errors.samples == 0) return absl::InvalidArgumentError("no sample of the time series lies outside the windows");
  errors.velocityRms = std::sqrt(velocitySquaredSum / static_cast<double>(errors.samples));
  errors.yawRateRms = std::sqrt(yawRateSquaredSum / static_cast<double>(errors.samples));
  return errors;
}

std::vector<std::string> compareClosedLoopRuns(const JsonValue& candidate,
                                               const GoldenFile& candidateSeries,
                                               const JsonValue& baseline,
                                               const GoldenFile* absl_nullable baselineSeries,
                                               const BandOptions& bandOptions,
                                               const HeadingCrossingOptions& crossingOptions) {
  if (baselineSeries == nullptr) {
    std::vector<std::string> violations = compareClosedLoopMetrics(candidate, baseline, bandOptions);
    const std::optional<double> peak = numberAt(baseline, "heading.cumulative_max_rad");
    const std::optional<double> trough = numberAt(baseline, "heading.cumulative_min_rad");
    if ((peak.has_value() && *peak >= M_PI) || (trough.has_value() && *trough <= -M_PI)) {
      violations.push_back(
          "heading: the baseline's heading reaches +-pi, but its time series is not archived to locate the crossings of the "
          "Euler yaw's cut (section 4.5, 720-degree turn exception)");
    }
    return violations;
  }

  const absl::StatusOr<std::vector<double>> crossings = eulerYawCrossingTimes(*baselineSeries);
  if (!crossings.ok()) return {absl::StrCat("baseline time series: ", crossings.status().message())};
  if (crossings->empty()) return compareClosedLoopMetrics(candidate, baseline, bandOptions);

  BandOptions options = bandOptions;
  options.skippedPaths.push_back("velocity.rms_error_mps");
  options.skippedPaths.push_back("yaw_rate.rms_error_radps");
  std::vector<std::string> violations = compareClosedLoopMetrics(candidate, baseline, options);
  if (!survivedIn(baseline) || !survivedIn(candidate)) return violations;

  // The tracking errors outside the windows around the baseline's crossings, in the same bands.
  const absl::StatusOr<WindowedTrackingErrors> candidateErrors =
      trackingErrorsOutsideWindows(candidateSeries, *crossings, crossingOptions.windowHalfWidth);
  const absl::StatusOr<WindowedTrackingErrors> baselineErrors =
      trackingErrorsOutsideWindows(*baselineSeries, *crossings, crossingOptions.windowHalfWidth);
  if (!candidateErrors.ok() || !baselineErrors.ok()) {
    violations.push_back(absl::StrCat("windowed tracking errors: ",
                                      candidateErrors.ok() ? baselineErrors.status().message() : candidateErrors.status().message()));
  } else {
    for (const MetricBand& band : closedLoopMetricBands()) {
      const bool velocity = band.path == "velocity.rms_error_mps";
      if (!velocity && band.path != "yaw_rate.rms_error_radps") continue;
      const double value = velocity ? candidateErrors->velocityRms : candidateErrors->yawRateRms;
      const double reference = velocity ? baselineErrors->velocityRms : baselineErrors->yawRateRms;
      const double allowed = std::max(band.relative * std::abs(reference), band.absolute);
      if (std::abs(value - reference) > allowed) {
        violations.push_back(absl::StrCat(band.path, " outside the +-", crossingOptions.windowHalfWidth, " s windows around the ",
                                          crossings->size(), " crossings of the Euler yaw's cut: ", value, " against ", reference,
                                          ", outside +/-", allowed));
      }
    }
  }

  // The turn itself: at least as far as the baseline, and to the commanded peak where the baseline got there.
  const std::optional<double> peak = numberAt(candidate, "heading.cumulative_max_rad");
  const std::optional<double> baselinePeak = numberAt(baseline, "heading.cumulative_max_rad");
  const std::optional<double> trough = numberAt(candidate, "heading.cumulative_min_rad");
  const std::optional<double> baselineTrough = numberAt(baseline, "heading.cumulative_min_rad");
  const double tolerance = crossingOptions.headingTolerance;
  if (peak.has_value() && baselinePeak.has_value() && *peak < *baselinePeak - tolerance) {
    violations.push_back(absl::StrCat("heading.cumulative_max_rad: ", *peak, " short of the baseline's ", *baselinePeak));
  }
  if (trough.has_value() && baselineTrough.has_value() && *trough > *baselineTrough + tolerance) {
    violations.push_back(absl::StrCat("heading.cumulative_min_rad: ", *trough, " short of the baseline's ", *baselineTrough));
  }
  if (crossingOptions.commandedHeadingPeak.has_value() && peak.has_value() && baselinePeak.has_value() &&
      std::abs(*baselinePeak - *crossingOptions.commandedHeadingPeak) <= tolerance &&
      std::abs(*peak - *crossingOptions.commandedHeadingPeak) > tolerance) {
    violations.push_back(absl::StrCat("heading.cumulative_max_rad: ", *peak, ", not the commanded ", *crossingOptions.commandedHeadingPeak,
                                      " +- ", tolerance, " the baseline reached"));
  }

  // No spike of the initial-state gap. A run on Euler coordinates (no quaternion norm: M1, M2, a main-line rerun of M0)
  // wraps its measured yaw by 2 pi where its heading crosses the cut, as its Euler baseline does; against a baseline whose
  // own gap spiked, it may spike too, but only at its own crossings and only by such a wrap.
  const std::optional<double> gap = numberAt(candidate, "initial_state_gap.max_rotation_rad");
  if (!gap.has_value() || *gap < crossingOptions.maxRotationGap) return violations;
  const std::optional<double> baselineGap = numberAt(baseline, "initial_state_gap.max_rotation_rad");
  const bool eulerCandidate = !numberAt(candidate, "quaternion_norm.max_deviation").has_value();
  const bool baselineSpiked = baselineGap.has_value() && *baselineGap >= crossingOptions.maxRotationGap;
  if (!eulerCandidate || !baselineSpiked) {
    violations.push_back(
        absl::StrCat("initial_state_gap.max_rotation_rad: ", *gap, " rad, a spike (not below ", crossingOptions.maxRotationGap, " rad)"));
    return violations;
  }
  for (std::string& violation : unexplainedRotationGapSpikes(candidateSeries, *gap, crossingOptions)) {
    violations.push_back(std::move(violation));
  }
  return violations;
}

}  // namespace ocs2::humanoid::validation
