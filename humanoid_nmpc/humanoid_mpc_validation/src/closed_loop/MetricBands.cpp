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
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_mpc_validation/closed_loop/ClosedLoopScenario.h"
#include "humanoid_mpc_validation/closed_loop/RobotConfiguration.h"

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

std::string fallReasonIn(const JsonValue& document) {
  const JsonValue* absl_nullable reason = document.findPath("survival.fall_reason");
  return reason != nullptr && reason->isString() ? reason->asString() : std::string("no reason");
}

/**
 * `value` held to `reference` within `band`, the sentence (prefixed with the band's path) appended to `comparison`: a
 * violation outside the band in a direction it holds, an improvement beyond a kNoWorse band in the other.
 */
void holdToBand(const MetricBand& band, double value, double reference, ClosedLoopComparison& comparison) {
  const double allowed = std::max(band.relative * std::abs(reference), band.absolute);
  const double excess = value - reference;
  switch (band.direction) {
    case BandDirection::kTwoSided:
      if (std::abs(excess) > allowed) {
        comparison.violations.push_back(absl::StrCat(band.path, ": ", value, " against ", reference, ", outside +/-", allowed));
      }
      return;
    case BandDirection::kNoWorse:
      if (excess > allowed) {
        comparison.violations.push_back(absl::StrCat(band.path, ": ", value, " against ", reference, ", worse by more than ", allowed));
      } else if (-excess > allowed) {
        comparison.improvements.push_back(absl::StrCat(band.path, ": ", value, " against ", reference, ", better by more than ", allowed));
      }
      return;
  }
  comparison.violations.push_back(absl::StrCat(band.path, ": a band of no known direction"));
}

// LINT.IfChange(archived_file_names)
/** The name LockstepOutputs.h gives a run's metrics document, which `make closed-loop-metrics` archives. */
std::string archivedMetricsFileName(const std::string& robot, const std::string& scenario) {
  return absl::StrCat(robot, "_", scenario, ".json");
}
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/closed_loop/LockstepOutputs.h:output_names)
// clang-format on

/** Whether `path` is a regular file; false also when that cannot be told. */
bool isRegularFile(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::is_regular_file(path, error);
}

/** The archived run `robot` `scenario` of the candidate directory against the baseline directory's; both documents exist. */
absl::StatusOr<ClosedLoopComparison> compareArchivedRun(const ArchivedComparisonOptions& options,
                                                        const std::string& robot,
                                                        const std::string& scenario) {
  const std::string fileName = archivedMetricsFileName(robot, scenario);
  ASSIGN_OR_RETURN(const JsonValue candidate, readJsonFile((std::filesystem::path(options.candidateDirectory) / fileName).string()));
  ASSIGN_OR_RETURN(const JsonValue baseline, readJsonFile((std::filesystem::path(options.baselineDirectory) / fileName).string()));
  return compareClosedLoopMetrics(candidate, baseline, BandOptions{.compareSolveTime = options.compareSolveTime});
}

}  // namespace

const std::vector<MetricBand>& closedLoopMetricBands() {
  static const absl::NoDestructor<std::vector<MetricBand>> kBands(std::vector<MetricBand>{
      {.path = "base_height.mean_m", .relative = 0.0, .absolute = 0.005, .direction = BandDirection::kTwoSided},
      {.path = "base_height.rms_error_m", .relative = 0.0, .absolute = 0.005, .direction = BandDirection::kNoWorse},
      {.path = "tilt.rms_rad", .relative = 0.20, .absolute = 0.005, .direction = BandDirection::kNoWorse},
      {.path = "tilt.max_rad", .relative = 0.20, .absolute = 0.005, .direction = BandDirection::kNoWorse},
      {.path = "velocity.rms_error_mps", .relative = 0.15, .absolute = 0.02, .direction = BandDirection::kNoWorse},
      {.path = "yaw_rate.rms_error_radps", .relative = 0.15, .absolute = 0.02, .direction = BandDirection::kNoWorse},
      {.path = "stance_foot_slip.max_m", .relative = 0.20, .absolute = 0.005, .direction = BandDirection::kNoWorse},
  });
  return *kBands;
}

std::string describeClosedLoopComparison(const ClosedLoopComparison& comparison) {
  std::string text = absl::StrCat("violations (", comparison.violations.size(), "):");
  for (const std::string& violation : comparison.violations) absl::StrAppend(&text, "\n  ", violation);
  absl::StrAppend(&text, "\nimprovements (", comparison.improvements.size(), "):");
  for (const std::string& improvement : comparison.improvements) absl::StrAppend(&text, "\n  ", improvement);
  return text;
}

ClosedLoopComparison compareClosedLoopMetrics(const JsonValue& candidate, const JsonValue& baseline, const BandOptions& options) {
  ClosedLoopComparison comparison;
  const bool baselineSurvived = survivedIn(baseline);
  const bool candidateSurvived = survivedIn(candidate);
  if (!baselineSurvived) {
    // Nothing to be worse than but the survival, which a fall cannot be.
    if (candidateSurvived) {
      comparison.improvements.push_back(
          absl::StrCat("survival: the baseline fell (", fallReasonIn(baseline), ") and the candidate survived"));
    }
    return comparison;
  }
  if (!candidateSurvived) {
    comparison.violations.push_back(absl::StrCat("survival: the baseline survived and the candidate fell (", fallReasonIn(candidate), ")"));
    return comparison;
  }

  for (const MetricBand& band : closedLoopMetricBands()) {
    const std::optional<double> value = numberAt(candidate, band.path);
    const std::optional<double> reference = numberAt(baseline, band.path);
    if (!value.has_value() || !reference.has_value()) continue;
    holdToBand(band, *value, *reference, comparison);
  }

  if (options.compareSolveTime) {
    const std::optional<double> p99 = numberAt(candidate, "solve_time_ms.total.p99");
    const std::optional<double> baselineP99 = numberAt(baseline, "solve_time_ms.total.p99");
    if (p99.has_value() && baselineP99.has_value() && *p99 > 1.10 * *baselineP99) {
      comparison.violations.push_back(
          absl::StrCat("solve_time_ms.total.p99: ", *p99, " ms against ", *baselineP99, " ms, more than +10 %"));
    }
  }

  const std::optional<double> normDeviation = numberAt(candidate, "quaternion_norm.max_deviation");
  if (normDeviation.has_value() && *normDeviation >= 1.0e-9) {
    comparison.violations.push_back(absl::StrCat("quaternion_norm.max_deviation: ", *normDeviation, ", not below 1e-9"));
  }
  return comparison;
}

absl::StatusOr<ArchivedComparison> compareArchivedClosedLoopRuns(const ArchivedComparisonOptions& options) {
  for (const std::string& directory : {options.candidateDirectory, options.baselineDirectory}) {
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) {
      return absl::NotFoundError(absl::StrCat("[compareArchivedClosedLoopRuns] no directory '", directory, "'"));
    }
  }
  if (!options.robot.empty()) RETURN_IF_ERROR(findRobotConfiguration(options.robot).status());

  ArchivedComparison result;
  const std::filesystem::path candidateDirectory(options.candidateDirectory);
  const std::filesystem::path baselineDirectory(options.baselineDirectory);
  for (const RobotConfiguration& configuration : robotConfigurations()) {
    if (!options.robot.empty() && configuration.name != options.robot) continue;
    for (const ClosedLoopScenario& scenario : closedLoopScenarios()) {
      if (!isRegularFile(candidateDirectory / archivedMetricsFileName(configuration.name, scenario.name))) continue;
      if (!isRegularFile(baselineDirectory / archivedMetricsFileName(configuration.name, scenario.name))) {
        result.withoutBaseline.push_back(absl::StrCat(configuration.name, " ", scenario.name));
        continue;
      }
      ASSIGN_OR_RETURN(ClosedLoopComparison comparison, compareArchivedRun(options, configuration.name, scenario.name));
      result.runs.push_back({.robot = configuration.name, .scenario = scenario.name, .comparison = std::move(comparison)});
    }
  }
  return result;
}

}  // namespace ocs2::humanoid::validation
