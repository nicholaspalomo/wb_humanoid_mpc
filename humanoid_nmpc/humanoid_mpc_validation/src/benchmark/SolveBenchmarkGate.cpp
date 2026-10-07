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

#include "humanoid_mpc_validation/benchmark/SolveBenchmarkGate.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"

#include "humanoid_common_mpc/common/StateLayout.h"

namespace ocs2::humanoid::validation {
namespace {

std::optional<double> numberAt(const JsonValue& document, const std::string& path) {
  const JsonValue* absl_nullable value = document.findPath(path);
  if (value == nullptr || !value->isNumber()) return std::nullopt;
  return value->asNumber();
}

/** The normalized library keys of a document's tape_operation_counts, with their counts. */
absl::flat_hash_map<std::string, double> tapeOperationCounts(const JsonValue& document) {
  absl::flat_hash_map<std::string, double> counts;
  const JsonValue* absl_nullable libraries = document.findPath("tape_operation_counts");
  if (libraries == nullptr || !libraries->isObject()) return counts;
  for (size_t index = 0; index < libraries->size(); ++index) {
    const JsonValue& count = libraries->valueAt(index);
    if (count.isNumber()) counts[normalizedLibraryKey(libraries->keyAt(index))] = count.asNumber();
  }
  return counts;
}

/** A CppAD library whose tape operations both documents count. */
struct SharedLibrary {
  std::string key;  ///< normalizedLibraryKey()
  double candidateCount = 0.0;
  double baselineCount = 0.0;
};

/** A breach when `candidate` exceeds `baseline` by more than `increase`, relative. */
void checkIncrease(const std::string& what,
                   std::optional<double> candidate,
                   std::optional<double> baseline,
                   double increase,
                   std::vector<std::string>& violations) {
  if (!candidate.has_value() || !baseline.has_value()) {
    violations.push_back(absl::StrCat(what, ": missing in the ", candidate.has_value() ? "baseline" : "candidate"));
    return;
  }
  if (*candidate > (1.0 + increase) * *baseline) {
    violations.push_back(absl::StrCat(what, ": ", *candidate, " against ", *baseline, ", more than +", 100.0 * increase, " %"));
  }
}

}  // namespace

std::string normalizedLibraryKey(absl::string_view key) {
  std::vector<absl::string_view> components;
  for (absl::string_view component : absl::StrSplit(key, '/')) {
    if (component != kStateLayoutTag) components.push_back(component);
  }
  return absl::StrJoin(components, "/");
}

std::vector<std::string> compareSolveBenchmarks(const JsonValue& candidate, const JsonValue& baseline, const SolveBenchmarkGate& gate) {
  std::vector<std::string> violations;
  checkIncrease("solve_time_ms.total.mean", numberAt(candidate, "solve_time_ms.total.mean"), numberAt(baseline, "solve_time_ms.total.mean"),
                gate.meanIncrease, violations);
  checkIncrease("solve_time_ms.total.p99", numberAt(candidate, "solve_time_ms.total.p99"), numberAt(baseline, "solve_time_ms.total.p99"),
                gate.p99Increase, violations);
  checkIncrease("solve_time_ms.lq_approximation.mean", numberAt(candidate, "solve_time_ms.lq_approximation.mean"),
                numberAt(baseline, "solve_time_ms.lq_approximation.mean"), gate.lqApproximationIncrease, violations);
  const std::optional<double> fraction = numberAt(candidate, "real_time.p99_fraction_of_period");
  if (!fraction.has_value()) {
    violations.push_back("real_time.p99_fraction_of_period: missing in the candidate");
  } else if (*fraction >= gate.maxP99FractionOfPeriod) {
    violations.push_back(
        absl::StrCat("real_time.p99_fraction_of_period: ", *fraction, ", not below ", gate.maxP99FractionOfPeriod, " of the MPC period"));
  }

  const absl::flat_hash_map<std::string, double> candidateCounts = tapeOperationCounts(candidate);
  const absl::flat_hash_map<std::string, double> baselineCounts = tapeOperationCounts(baseline);
  std::vector<SharedLibrary> libraries;
  for (const std::pair<const std::string, double>& library : candidateCounts) {
    const absl::flat_hash_map<std::string, double>::const_iterator baselineCount = baselineCounts.find(library.first);
    if (baselineCount != baselineCounts.end()) {
      libraries.push_back({.key = library.first, .candidateCount = library.second, .baselineCount = baselineCount->second});
    }
  }
  // The hash maps' order is not reproducible.
  std::sort(libraries.begin(), libraries.end(), [](const SharedLibrary& a, const SharedLibrary& b) { return a.key < b.key; });
  for (const SharedLibrary& library : libraries) {
    checkIncrease(absl::StrCat("tape_operation_counts.", library.key), library.candidateCount, library.baselineCount,
                  gate.tapeOperationIncrease, violations);
  }
  checkIncrease("total_tape_operations", numberAt(candidate, "total_tape_operations"), numberAt(baseline, "total_tape_operations"),
                gate.tapeOperationIncrease, violations);
  return violations;
}

}  // namespace ocs2::humanoid::validation
