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

// The thresholding of the contact variables into a stance sequence (design
// section 4.2): the threshold, the boundary stances, the minimum dwell and the
// separation of the contact changes.

#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "humanoid_multi_contact/planner/contact_invariant/StanceThresholding.h"

namespace ocs2::humanoid::multi_contact {
namespace {

constexpr double kDt = 0.05;

std::vector<double> timesOf(int nodes) {
  std::vector<double> times;
  for (int n = 0; n < nodes; ++n) times.push_back(kDt * n);
  return times;
}

/** One candidate's activations as nodes of one entry. */
std::vector<std::vector<double>> column(const std::vector<double>& values) {
  std::vector<std::vector<double>> nodes;
  for (const double value : values) nodes.push_back({value});
  return nodes;
}

std::vector<bool> activeOf(const StanceSchedule& schedule, int candidate) {
  std::vector<bool> active;
  for (const std::vector<bool>& node : schedule.active) {
    active.push_back(node[candidate]);
  }
  return active;
}

TEST(StanceThresholding, ThresholdsAtOneHalfAndKeepsTheBoundaryStances) {
  // 0.6 s on, 0.6 s off, 0.6 s on: phases longer than the dwell.
  std::vector<double> values(37, 0.9);
  for (int n = 12; n < 24; ++n) values[n] = 0.2;
  values.front() = 0.1;  // the initial stance has the contact anyway
  const absl::StatusOr<StanceSchedule> schedule = thresholdActivations(
      timesOf(37), column(values), {true}, {true}, ThresholdingOptions());
  ASSERT_TRUE(schedule.ok()) << schedule.status();
  const std::vector<bool> active = activeOf(*schedule, /*candidate=*/0);
  EXPECT_TRUE(active.front());
  EXPECT_TRUE(active[11]);
  EXPECT_FALSE(active[12]);
  EXPECT_FALSE(active[23]);
  EXPECT_TRUE(active[24]);
  EXPECT_TRUE(active.back());
}

TEST(StanceThresholding, MergesPhasesShorterThanTheDwell) {
  // An off blip of 0.1 s and an on blip of 0.15 s inside long phases.
  std::vector<double> values(40, 0.9);
  values[10] = 0.3;
  values[11] = 0.3;
  for (int n = 20; n < 40; ++n) values[n] = 0.1;
  values[30] = 0.8;
  values[31] = 0.8;
  values[32] = 0.8;
  const absl::StatusOr<StanceSchedule> schedule = thresholdActivations(
      timesOf(40), column(values), {true}, {false}, ThresholdingOptions());
  ASSERT_TRUE(schedule.ok()) << schedule.status();
  const std::vector<bool> active = activeOf(*schedule, /*candidate=*/0);
  for (int n = 0; n < 20; ++n) EXPECT_TRUE(active[n]) << n;
  for (int n = 20; n < 40; ++n) EXPECT_FALSE(active[n]) << n;
}

TEST(StanceThresholding, SeparatesSimultaneousChanges) {
  // Two candidates that lift off at the same node.
  std::vector<std::vector<double>> nodes;
  for (int n = 0; n < 40; ++n) {
    const double value = n < 20 ? 0.9 : 0.1;
    nodes.push_back({value, value});
  }
  const absl::StatusOr<StanceSchedule> schedule = thresholdActivations(
      timesOf(40), nodes, {true, true}, {false, false}, ThresholdingOptions());
  ASSERT_TRUE(schedule.ok()) << schedule.status();
  const std::vector<bool> first = activeOf(*schedule, /*candidate=*/0);
  const std::vector<bool> second = activeOf(*schedule, /*candidate=*/1);
  EXPECT_TRUE(first[19]);
  EXPECT_FALSE(first[20]);
  // The second waits the minimum separation, 0.1 s.
  EXPECT_TRUE(second[21]);
  EXPECT_FALSE(second[22]);
}

TEST(StanceThresholding, KeepsSupportWithoutFlight) {
  // The first candidate lifts off at node 10, the second at 12 and touches
  // down again at 20: a flight phase from 12 to 20.
  std::vector<std::vector<double>> nodes;
  for (int n = 0; n < 40; ++n) {
    nodes.push_back({n < 10 ? 0.9 : 0.1, n < 12 || n >= 20 ? 0.9 : 0.1});
  }
  ThresholdingOptions options;
  options.minimumContacts = 1;
  const absl::StatusOr<StanceSchedule> schedule = thresholdActivations(
      timesOf(40), nodes, {true, true}, {false, true}, options);
  ASSERT_TRUE(schedule.ok()) << schedule.status();
  for (int n = 0; n < 40; ++n) {
    EXPECT_TRUE(schedule->active[n][0] || schedule->active[n][1]) << n;
  }
  // The second keeps its contact until it touches down again: no change.
  const std::vector<bool> second = activeOf(*schedule, /*candidate=*/1);
  for (int n = 0; n < 40; ++n) EXPECT_TRUE(second[n]) << n;
  // Without the option the flight remains.
  const absl::StatusOr<StanceSchedule> flight = thresholdActivations(
      timesOf(40), nodes, {true, true}, {false, true}, ThresholdingOptions());
  ASSERT_TRUE(flight.ok());
  EXPECT_FALSE(flight->active[15][0] || flight->active[15][1]);
}

TEST(StanceThresholding, QuantizesTheChanges) {
  // A lift-off at node 11 (0.55 s) moves to node 12 (0.6 s), a multiple of
  // 0.1 s.
  std::vector<double> values(40, 0.9);
  for (int n = 11; n < 40; ++n) values[n] = 0.1;
  ThresholdingOptions options;
  options.changeQuantum = 0.1;
  const absl::StatusOr<StanceSchedule> schedule = thresholdActivations(
      timesOf(40), column(values), {true}, {false}, options);
  ASSERT_TRUE(schedule.ok()) << schedule.status();
  const std::vector<bool> active = activeOf(*schedule, /*candidate=*/0);
  EXPECT_TRUE(active[11]);
  EXPECT_FALSE(active[12]);
}

TEST(StanceThresholding, RefusesInconsistentSizes) {
  EXPECT_FALSE(thresholdActivations(timesOf(3), column({0.1, 0.2}), {true},
                                    {true}, ThresholdingOptions())
                   .ok());
  EXPECT_FALSE(thresholdActivations({0.0, 0.0}, column({0.1, 0.2}), {true},
                                    {true}, ThresholdingOptions())
                   .ok());
  EXPECT_FALSE(thresholdActivations(timesOf(2), column({0.1, 0.2}), {true},
                                    {true, false}, ThresholdingOptions())
                   .ok());
}

}  // namespace
}  // namespace ocs2::humanoid::multi_contact
