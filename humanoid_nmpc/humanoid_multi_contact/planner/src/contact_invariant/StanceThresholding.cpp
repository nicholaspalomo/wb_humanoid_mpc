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

#include "humanoid_multi_contact/planner/contact_invariant/StanceThresholding.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// [s] Node times closer than this count as equal: the grid's rounding.
constexpr double kTimeTolerance = 1.0e-9;
// The rounds of support mending and change separation, a bound that is never
// reached: each round removes a gap or a close change for good.
constexpr int kMaximumRounds = 100;

/** A run of equal values of one candidate: nodes [begin, end). Passive
 * data. */
struct Run {
  size_t begin = 0;
  size_t end = 0;
};

std::vector<Run> runsOf(const std::vector<bool>& column) {
  std::vector<Run> runs;
  size_t begin = 0;
  for (size_t n = 1; n <= column.size(); ++n) {
    if (n == column.size() || column[n] != column[begin]) {
      runs.push_back(Run{.begin = begin, .end = n});
      begin = n;
    }
  }
  return runs;
}

/** [s] How long `run` lasts: to the first node of the next run, or to the
 * last node. */
double durationOf(const Run& run, const std::vector<double>& times) {
  const size_t last = std::min(run.end, times.size() - 1);
  return times[last] - times[run.begin];
}

/** Merges the interior runs shorter than `minimumDwell` into their
 * neighbors, shortest first. */
void mergeShortRuns(const std::vector<double>& times,
                    double minimumDwell,
                    std::vector<bool>& column) {
  while (true) {
    const std::vector<Run> runs = runsOf(column);
    std::optional<size_t> shortest;
    for (size_t r = 1; r + 1 < runs.size(); ++r) {
      const double duration = durationOf(runs[r], times);
      if (duration >= minimumDwell - kTimeTolerance) continue;
      if (!shortest.has_value() ||
          duration < durationOf(runs[*shortest], times)) {
        shortest = r;
      }
    }
    if (!shortest.has_value()) return;
    const Run& run = runs[*shortest];
    for (size_t n = run.begin; n < run.end; ++n) column[n] = !column[n];
  }
}

/** A contact change: candidate `candidate` changes at node `node`. */
struct Change {
  size_t node = 0;
  size_t candidate = 0;
};

std::vector<Change> changesOf(const std::vector<std::vector<bool>>& columns) {
  std::vector<Change> changes;
  for (size_t k = 0; k < columns.size(); ++k) {
    for (size_t n = 1; n < columns[k].size(); ++n) {
      if (columns[k][n] != columns[k][n - 1]) {
        changes.push_back(Change{.node = n, .candidate = k});
      }
    }
  }
  std::sort(
      changes.begin(), changes.end(), [](const Change& a, const Change& b) {
        return a.node != b.node ? a.node < b.node : a.candidate < b.candidate;
      });
  return changes;
}

/** Delays the first change closer than `separation` to the change before it;
 * false when no change is. */
bool delayOneChange(const std::vector<double>& times,
                    double separation,
                    std::vector<std::vector<bool>>& columns) {
  double last = -std::numeric_limits<double>::infinity();
  for (const Change& change : changesOf(columns)) {
    if (times[change.node] - last >= separation - kTimeTolerance) {
      last = times[change.node];
      continue;
    }
    size_t delayed = change.node;
    while (delayed + 1 < times.size() &&
           times[delayed] - last < separation - kTimeTolerance) {
      ++delayed;
    }
    if (delayed == change.node) {
      // At the last node: nothing later to delay to.
      last = times[change.node];
      continue;
    }
    std::vector<bool>& column = columns[change.candidate];
    const bool before = column[change.node - 1];
    for (size_t n = change.node; n < delayed; ++n) column[n] = before;
    return true;
  }
  return false;
}

/** The candidates in contact at node `node`. */
int contactsAt(const std::vector<std::vector<bool>>& columns, size_t node) {
  int count = 0;
  for (const std::vector<bool>& column : columns) count += column[node] ? 1 : 0;
  return count;
}

/** Keeps the support of nodes with fewer than `minimum` contacts: the contact
 * that lifted off last stays until another one has touched down and
 * `separation` has passed. False when no node lacks support (or none can be
 * mended). */
bool mendOneSupportGap(const std::vector<double>& times,
                       int minimum,
                       double separation,
                       std::vector<std::vector<bool>>& columns) {
  for (size_t n = 1; n < times.size(); ++n) {
    if (contactsAt(columns, n) >= minimum) continue;
    // The contact that lifted off last, at or before n.
    std::optional<size_t> candidate;
    size_t liftOff = 0;
    for (size_t k = 0; k < columns.size(); ++k) {
      for (size_t j = n; j >= 1; --j) {
        if (columns[k][j - 1] && !columns[k][j]) {
          if (!candidate.has_value() || j > liftOff) {
            candidate = k;
            liftOff = j;
          }
          break;
        }
      }
    }
    if (!candidate.has_value()) return false;
    // Until another contact has touched down, then the separation.
    size_t until = n;
    while (until + 1 < times.size() && contactsAt(columns, until) < minimum) {
      ++until;
    }
    const double release = times[until] + separation;
    while (until + 1 < times.size() &&
           times[until] < release - kTimeTolerance) {
      ++until;
    }
    for (size_t j = liftOff; j < until; ++j) columns[*candidate][j] = true;
    return true;
  }
  return false;
}

/** Whether `time` is a multiple of `quantum`. */
bool onQuantum(double time, double quantum) {
  const double ratio = time / quantum;
  return std::abs(ratio - std::round(ratio)) * quantum < kTimeTolerance;
}

/** Delays one change that is not at a multiple of `quantum` to the next
 * node that is; false when every change is (or none can move). */
bool quantizeOneChange(const std::vector<double>& times,
                       double quantum,
                       std::vector<std::vector<bool>>& columns) {
  for (const Change& change : changesOf(columns)) {
    if (onQuantum(times[change.node], quantum)) continue;
    size_t delayed = change.node;
    while (delayed + 1 < times.size() && !onQuantum(times[delayed], quantum)) {
      ++delayed;
    }
    if (delayed == change.node) continue;
    std::vector<bool>& column = columns[change.candidate];
    const bool before = column[change.node - 1];
    for (size_t n = change.node; n < delayed; ++n) column[n] = before;
    return true;
  }
  return false;
}

}  // namespace

absl::StatusOr<StanceSchedule> thresholdActivations(
    const std::vector<double>& times,
    const std::vector<std::vector<double>>& activations,
    const std::vector<bool>& initial,
    const std::vector<bool>& final,
    const ThresholdingOptions& options) {
  if (times.empty() || activations.size() != times.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("thresholding: ", activations.size(),
                     " activation nodes for ", times.size(), " times"));
  }
  const size_t numCandidates = initial.size();
  if (final.size() != numCandidates) {
    return absl::InvalidArgumentError(
        "thresholding: the initial and final stances differ in size");
  }
  for (size_t n = 0; n < times.size(); ++n) {
    if (activations[n].size() != numCandidates) {
      return absl::InvalidArgumentError(
          absl::StrCat("thresholding: node ", n, " has ", activations[n].size(),
                       " activations, not ", numCandidates));
    }
    if (n > 0 && !(times[n] > times[n - 1])) {
      return absl::InvalidArgumentError(
          absl::StrCat("thresholding: the times do not increase at node ", n));
    }
  }
  std::vector<std::vector<bool>> columns(numCandidates,
                                         std::vector<bool>(times.size()));
  for (size_t k = 0; k < numCandidates; ++k) {
    for (size_t n = 0; n < times.size(); ++n) {
      columns[k][n] = activations[n][k] >= options.threshold;
    }
    columns[k].front() = initial[k];
    columns[k].back() = final[k];
    mergeShortRuns(times, options.minimumDwell, columns[k]);
  }
  // Mend the support gaps, then separate the changes; a separation can open
  // a gap again, so both run until neither changes anything (each mend only
  // adds contact, each delay moves a change later: both end).
  for (int round = 0; round < kMaximumRounds; ++round) {
    bool changed = false;
    while (options.minimumContacts > 0 &&
           mendOneSupportGap(times, options.minimumContacts,
                             options.minimumChangeSeparation, columns)) {
      changed = true;
    }
    while (delayOneChange(times, options.minimumChangeSeparation, columns)) {
      changed = true;
    }
    while (options.changeQuantum > 0.0 &&
           quantizeOneChange(times, options.changeQuantum, columns)) {
      changed = true;
    }
    if (!changed) break;
  }
  StanceSchedule schedule;
  schedule.active.assign(times.size(), std::vector<bool>(numCandidates));
  for (size_t n = 0; n < times.size(); ++n) {
    for (size_t k = 0; k < numCandidates; ++k) {
      schedule.active[n][k] = columns[k][n];
    }
  }
  return schedule;
}

}  // namespace ocs2::humanoid::multi_contact
